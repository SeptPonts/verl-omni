# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Opt-in H20/MLU H3 comparison state, separate from resumable checkpoints."""

import hashlib
import json
import shutil
from pathlib import Path

import torch
from peft import get_peft_model_state_dict
from safetensors import safe_open
from safetensors.torch import load_file, save_file

SNAPSHOT_STEPS = (0, 10, 20, 30, 40, 50)
PACKAGE_FILES = ("adapter_model.safetensors", "adapter_config.json", "transformer_config.json", "tensor_manifest.json")


def tensor_manifest(state):
    """Record CPU adapter tensor identities independently of the container format."""
    return {
        name: {
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "sha256": hashlib.sha256(value.detach().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest(),
        }
        for name, value in sorted(state.items())
    }


def validate_h3_comparison(config):
    """Reject unsupported experiment settings before allocating Ray workers."""
    model = config.actor_rollout_ref.model
    initial = model.get("h3_comparison_initial_lora_path")
    snapshots = config.trainer.get("h3_comparison_lora_snapshot_dir")
    backup = config.trainer.get("h3_comparison_checkpoint_backup_dir")
    if not initial and not snapshots and not backup:
        return
    if not initial:
        raise ValueError("H3 comparison requires the common initial LoRA package")
    if config.trainer.resume_mode not in ("disable", "resume_path"):
        raise ValueError("H3 comparison requires explicit trainer.resume_mode=disable or resume_path")
    if model.lora_adapter_path is not None or model.lora_rank <= 0:
        raise ValueError("H3 comparison requires newly created LoRA, with lora_adapter_path=null")
    if tuple(model.policy_state_adapters) != ("default",):
        raise ValueError("H3 comparison supports only the default policy adapter")
    if config.actor_rollout_ref.actor.strategy != "fsdp2" or config.trainer.use_v1:
        raise ValueError("H3 comparison hooks require the V0 trainer and FSDP2")
    for name in PACKAGE_FILES:
        if not (Path(initial) / name).is_file():
            raise FileNotFoundError(Path(initial) / name)
    checkpoint = config.actor_rollout_ref.actor.checkpoint
    if snapshots or backup or config.trainer.resume_mode == "resume_path":
        if checkpoint.async_save or checkpoint.save_lora_only or "model" not in checkpoint.save_contents:
            raise ValueError("H3 snapshots and resume require synchronous full model checkpoints")
    if backup:
        if not {"model", "optimizer", "extra"}.issubset(checkpoint.save_contents):
            raise ValueError("H3 persistent backup requires model, optimizer, and extra checkpoint state")
        if Path(backup).resolve().is_relative_to(Path(config.trainer.default_local_dir).resolve()):
            raise ValueError("H3 persistent backup must live outside the rotating local checkpoint directory")
    if config.trainer.resume_mode == "resume_path":
        required = {"model", "optimizer", "extra"}
        if not required.issubset(checkpoint.save_contents) or not required.issubset(checkpoint.load_contents):
            raise ValueError("H3 resume requires saving and loading model, optimizer, and extra checkpoint state")
        resume_path = config.trainer.resume_from_path
        if not resume_path:
            raise ValueError("H3 resume requires trainer.resume_from_path pointing to global_step_<step>")
        resume = Path(resume_path)
        if not resume.name.startswith("global_step_") or not resume.name.removeprefix("global_step_").isdigit():
            raise ValueError("H3 resume requires trainer.resume_from_path pointing to global_step_<step>")
        backup_receipt = resume.parent / "backup_manifest.json"
        if backup_receipt.is_file():
            receipt = json.loads(backup_receipt.read_text())
            if receipt["status"] != "COMPLETE" or receipt["step"] != int(resume.name.removeprefix("global_step_")):
                raise ValueError("H3 resume cannot use an incomplete or superseded persistent backup")
        world_size = config.trainer.nnodes * config.trainer.n_gpus_per_node
        files = [resume / "data.pt"]
        files.extend(
            resume / "actor" / f"{prefix}_world_size_{world_size}_rank_{rank}.pt"
            for rank in range(world_size)
            for prefix in ("model", "optim", "extra_state")
        )
        for path in files:
            if not path.is_file():
                raise FileNotFoundError(path)
    if snapshots:
        if config.trainer.save_freq != 10:
            raise ValueError("H3 snapshots at steps 10/20/30/40/50 require trainer.save_freq=10")
        if Path(snapshots).resolve().is_relative_to(Path(config.trainer.default_local_dir).resolve()):
            raise ValueError("H3 LoRA snapshots must live outside the rotating checkpoint directory")


def load_h3_comparison_lora(module, directory):
    """Copy common A/B tensors into existing adapters before FSDP broadcasts rank zero."""
    directory = Path(directory)
    expected_config = json.loads((directory / "adapter_config.json").read_text())
    actual_config = module.peft_config["default"].to_dict()
    actual_config["target_modules"] = sorted(actual_config["target_modules"])
    if actual_config != expected_config:
        raise ValueError("Common LoRA configuration differs from the newly created adapter")
    expected_model = json.loads((directory / "transformer_config.json").read_text())
    actual_model = json.loads(module.to_json_string())
    if {k: v for k, v in expected_model.items() if not k.startswith("_")} != {
        k: v for k, v in actual_model.items() if not k.startswith("_")
    }:
        raise ValueError("Common LoRA transformer configuration differs from the actor")
    expected = json.loads((directory / "tensor_manifest.json").read_text())
    current = get_peft_model_state_dict(module, adapter_name="default")
    if current.keys() != expected.keys():
        raise ValueError("Common LoRA tensor keys differ from the actor")
    # Nonzero FSDP2 ranks stay on meta; the existing rank-zero broadcast owns materialization.
    with safe_open(directory / "adapter_model.safetensors", framework="pt", device="cpu") as source:
        if set(source.keys()) != current.keys():
            raise ValueError("Common LoRA tensor keys differ from its manifest")
        for name, destination in current.items():
            record = expected[name]
            shape = source.get_slice(name).get_shape()
            dtype = source.get_slice(name).get_dtype()
            if shape != record["shape"] or list(destination.shape) != shape:
                raise ValueError(f"Common LoRA shape mismatch: {name}")
            if dtype != "BF16" or record["dtype"] != "torch.bfloat16" or destination.dtype != torch.bfloat16:
                raise ValueError(f"Common LoRA dtype mismatch: {name}")
            if destination.device.type == "meta":
                continue
            if destination.device.type != "cpu":
                raise ValueError("Load common LoRA on CPU before FSDP device placement")
            value = source.get_tensor(name)
            if tensor_manifest({name: value})[name] != record:
                raise ValueError(f"Common LoRA checksum mismatch: {name}")
            with torch.no_grad():
                destination.copy_(value)
            if tensor_manifest({name: destination})[name] != record:
                raise ValueError(f"Common LoRA copy mismatch: {name}")


def save_h3_comparison_snapshot(config, step, actor_checkpoint=None):
    """Publish independent LoRA snapshots after synchronous checkpoint completion."""
    root = config.trainer.get("h3_comparison_lora_snapshot_dir")
    if not root or step not in SNAPSHOT_STEPS:
        return
    initial = Path(config.actor_rollout_ref.model.h3_comparison_initial_lora_path)
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    destination = root / f"step_{step:04d}"
    if destination.exists():
        raise FileExistsError(destination)
    staging = root / f"step_{step:04d}.incomplete"
    staging.mkdir()
    if step == 0:
        for name in PACKAGE_FILES:
            shutil.copyfile(initial / name, staging / name)
        state = load_file(staging / "adapter_model.safetensors")
        if tensor_manifest(state) != json.loads((staging / "tensor_manifest.json").read_text()):
            raise ValueError("Initial LoRA snapshot checksum mismatch")
    else:
        from verl_omni.utils.fsdp_utils import export_fsdp_lora_adapter

        export_fsdp_lora_adapter(actor_checkpoint, staging)
        original_config = json.loads((initial / "adapter_config.json").read_text())
        exported_config = json.loads((staging / "adapter_config.json").read_text())
        if any(exported_config[key] != original_config[key] for key in ("r", "lora_alpha")):
            raise ValueError("Checkpoint LoRA rank/alpha differs from common initialization")
        exported = load_file(staging / "adapter_model.safetensors")
        state = {name.removeprefix("base_model.model."): tensor for name, tensor in exported.items()}
        expected = json.loads((initial / "tensor_manifest.json").read_text())
        records = tensor_manifest(state)
        if records.keys() != expected.keys() or any(
            records[name][key] != expected[name][key] for name in records for key in ("shape", "dtype")
        ):
            raise ValueError("Checkpoint LoRA layout differs from common initialization")
        # Keep H3's original target names/config; the generic exporter infers suffixes.
        normalized = staging / "normalized.safetensors"
        save_file(state, normalized)
        normalized.replace(staging / "adapter_model.safetensors")
        shutil.copyfile(initial / "adapter_config.json", staging / "adapter_config.json")
        shutil.copyfile(initial / "transformer_config.json", staging / "transformer_config.json")
        (staging / "tensor_manifest.json").write_text(json.dumps(records, indent=2) + "\n")
    files = {}
    for name in PACKAGE_FILES:
        with (staging / name).open("rb") as stream:
            files[name] = hashlib.file_digest(stream, "sha256").hexdigest()
    (staging / "snapshot_manifest.json").write_text(
        json.dumps(
            {
                "step": step,
                "initial_lora": str(initial),
                "actor_checkpoint": actor_checkpoint,
                "files_sha256": files,
                "exact_training_resume": False,
            },
            indent=2,
        )
        + "\n"
    )
    staging.rename(destination)
