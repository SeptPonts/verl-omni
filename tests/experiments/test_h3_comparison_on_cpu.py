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
"""Exercise the temporary comparison hooks with real tiny H3 adapters and checkpoint I/O."""

import ast
import importlib
import json
import os
import shutil
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from diffusers import MiniMaxH3Transformer3DModel
from omegaconf import OmegaConf
from peft import get_peft_model_state_dict
from safetensors.torch import load_file, save_file

ROOT = Path(__file__).parents[2]
TINY = dict(
    num_attention_heads=4,
    attention_head_dim=8,
    hidden_size=24,
    num_layers=2,
    num_refiner_layers=1,
    ffn_dim=32,
    in_channels=4,
    audio_in_channels=4,
    text_dim=16,
    freq_dim=16,
    time_embed_hidden_dim=24,
    time_embed_dim=16,
    rope_freq_dim=1,
)
TARGETS = ["to_q", "to_k", "to_v", "to_out.0", "ff.net.0.proj", "ff.net.2"]


@pytest.fixture
def api(monkeypatch):
    """Skip unrelated device registration while retaining actual PEFT and checkpoint code."""
    before = set(sys.modules)
    for name in ("verl_omni", "verl_omni.utils", "verl_omni.workers", "verl_omni.workers.engine"):
        package = ModuleType(name)
        package.__path__ = [str(ROOT.joinpath(*name.split(".")))]
        monkeypatch.setitem(sys.modules, name, package)
    upstream = ModuleType("verl.utils.fsdp_utils")
    for name in ("fsdp_version", "collect_lora_params", "layered_summon_lora_params"):
        setattr(upstream, name, Mock(side_effect=AssertionError("CPU export must not gather live FSDP weights")))
    monkeypatch.setitem(sys.modules, upstream.__name__, upstream)
    yield importlib.import_module("verl_omni.experiments.h3_comparison")
    for name in set(sys.modules) - before:
        if name.startswith("verl_omni."):
            sys.modules.pop(name, None)


def tiny_actor(seed, device="cpu"):
    """Build LoRA through the application's existing mixin with the agreed H3 recipe."""
    from verl_omni.workers.engine.lora_adapter_mixin import LoRAAdapterMixin

    torch.manual_seed(seed)
    with torch.device(device):
        model = MiniMaxH3Transformer3DModel(**TINY).to(dtype=torch.bfloat16)
    # The actual engine leaves its meta initialization context before add_adapter.
    engine = LoRAAdapterMixin()
    engine.model_config = SimpleNamespace(
        lora_adapter_path=None,
        policy_state_adapters=("default",),
        lora_rank=64,
        lora_alpha=128,
        lora_init_weights="gaussian",
        target_modules=TARGETS,
        target_parameters=None,
        exclude_modules=None,
        lora_dtype=None,
    )
    return engine._build_lora_module(model)


@pytest.fixture
def case(api, tmp_path):
    """Create a complete tiny canonical package and an opt-in trainer configuration."""
    torch.set_num_threads(2)
    model = tiny_actor(42)
    initial = tmp_path / "initial"
    initial.mkdir()
    state = get_peft_model_state_dict(model, adapter_name="default")
    save_file(state, initial / "adapter_model.safetensors")
    metadata = model.peft_config["default"].to_dict()
    metadata["target_modules"] = sorted(metadata["target_modules"])
    (initial / "adapter_config.json").write_text(json.dumps(metadata))
    (initial / "transformer_config.json").write_text(model.to_json_string())
    (initial / "tensor_manifest.json").write_text(json.dumps(api.tensor_manifest(state)))
    config = OmegaConf.create(
        {
            "actor_rollout_ref": {
                "model": {
                    "h3_comparison_initial_lora_path": str(initial),
                    "lora_adapter_path": None,
                    "lora_rank": 64,
                    "policy_state_adapters": ["default"],
                },
                "actor": {
                    "strategy": "fsdp2",
                    "checkpoint": {
                        "async_save": False,
                        "save_lora_only": False,
                        "save_contents": ["model", "optimizer", "extra"],
                        "load_contents": ["model", "optimizer", "extra"],
                    },
                },
            },
            "trainer": {
                "h3_comparison_lora_snapshot_dir": str(tmp_path / "snapshots"),
                "default_local_dir": str(tmp_path / "checkpoints"),
                "default_hdfs_dir": None,
                "resume_mode": "disable",
                "resume_from_path": None,
                "del_local_ckpt_after_load": False,
                "nnodes": 1,
                "n_gpus_per_node": 2,
                "use_v1": False,
                "save_freq": 10,
                "max_actor_ckpt_to_keep": 1,
            },
        }
    )
    return SimpleNamespace(model=model, initial=initial, config=config)


def test_common_initialization_preserves_base_and_scale(api, case):
    """Different RNG states converge only on adapter bytes, preserving base weights and trainability."""
    target = tiny_actor(937)
    base = {name: value.clone() for name, value in target.named_parameters() if "lora_" not in name}
    trainable = {name for name, value in target.named_parameters() if value.requires_grad}
    expected = json.loads((case.initial / "tensor_manifest.json").read_text())
    assert api.tensor_manifest(get_peft_model_state_dict(target)) != expected
    api.load_h3_comparison_lora(target, case.initial)
    assert api.tensor_manifest(get_peft_model_state_dict(target)) == expected
    assert target.peft_config["default"].r == 64
    assert target.peft_config["default"].lora_alpha == 128
    assert all(layer.scaling["default"] == 2 for layer in target.modules() if hasattr(layer, "scaling"))
    assert trainable == {name for name, value in target.named_parameters() if value.requires_grad}
    assert all(torch.equal(value, base[name]) for name, value in target.named_parameters() if name in base)
    meta = tiny_actor(17, "meta")
    api.load_h3_comparison_lora(meta, case.initial)
    assert all(value.is_meta for value in meta.parameters())


@pytest.mark.parametrize("corruption", ["alpha", "key", "shape", "dtype", "checksum"])
def test_reject_incompatible_package(api, case, corruption):
    """Fail on artifact incompatibility instead of silently changing scale or casting weights."""
    if corruption == "alpha":
        path = case.initial / "adapter_config.json"
        metadata = json.loads(path.read_text())
        metadata["lora_alpha"] = 64
        path.write_text(json.dumps(metadata))
    else:
        path = case.initial / "adapter_model.safetensors"
        state = load_file(path)
        key = next(iter(state))
        if corruption == "key":
            del state[key]
        elif corruption == "shape":
            state[key] = state[key][:-1].contiguous()
        elif corruption == "dtype":
            state[key] = state[key].float()
        else:
            state[key] = state[key] + 1
        save_file(state, case.initial / "replacement.safetensors")
        (case.initial / "replacement.safetensors").replace(path)
    with pytest.raises(ValueError):
        api.load_h3_comparison_lora(tiny_actor(937), case.initial)


@pytest.fixture
def trainer_methods():
    """Execute actual checkpoint methods and fit startup without importing the Ray/device stack."""
    source = ROOT / "verl_omni/trainer/diffusion/ray_diffusion_trainer.py"
    tree = ast.parse(source.read_text())
    classes = {node.name: node for node in tree.body if isinstance(node, ast.ClassDef)}
    methods = [
        node
        for node in classes["BaseRayDiffusionTrainer"].body
        if isinstance(node, ast.FunctionDef) and node.name in ("_save_checkpoint", "_load_checkpoint")
    ]
    fit = next(node for node in classes["PolicyGradientRayTrainer"].body if getattr(node, "name", None) == "fit")
    start = next(i for i, node in enumerate(fit.body) if ast.unparse(node) == "self.global_steps = 0")
    end = next(i for i, node in enumerate(fit.body) if ast.unparse(node).startswith("current_epoch ="))
    fit.body = fit.body[start:end]
    namespace = {"os": os, "torch": torch, "find_latest_ckpt_path": Mock(return_value=None)}
    exec(compile(ast.Module(body=[*methods, fit], type_ignores=[]), str(source), "exec"), namespace)
    return SimpleNamespace(
        save=namespace["_save_checkpoint"], load=namespace["_load_checkpoint"], start=namespace["fit"]
    )


@pytest.mark.parametrize("persistent_backup", [False, True])
def test_snapshots_follow_real_checkpoint_hook_and_survive_rotation(api, case, trainer_methods, persistent_backup):
    """Resume after step 30 through actual trainer hooks; later snapshots survive full-checkpoint rotation."""
    config = case.config
    if persistent_backup:
        config.trainer.h3_comparison_checkpoint_backup_dir = str(case.initial.parent / "persistent")
    api.validate_h3_comparison(config)
    snapshots = Path(config.trainer.h3_comparison_lora_snapshot_dir)
    saved_paths = []

    def save_checkpoint(path, remote, step, max_ckpt_to_keep):
        """Write real two-rank replicated state and apply the configured full-checkpoint rotation."""
        assert remote is None and max_ckpt_to_keep == 1
        path = Path(path)
        path.mkdir(parents=True)
        with torch.no_grad():
            for name, value in case.model.named_parameters():
                if "lora_B" in name:
                    value.fill_(step / 100)
        state = {key: value.clone() for key, value in case.model.state_dict().items()}
        for rank in range(2):
            torch.save(state, path / f"model_world_size_2_rank_{rank}.pt")
            torch.save({"step": step}, path / f"optim_world_size_2_rank_{rank}.pt")
            torch.save({"lr_scheduler": {"last_epoch": step}}, path / f"extra_state_world_size_2_rank_{rank}.pt")
        (path / "fsdp_config.json").write_text(json.dumps({"world_size": 2, "FSDP_version": 2}))
        (path / "lora_train_meta.json").write_text(json.dumps({"r": 64, "lora_alpha": 128}))
        # VeRL tracks only checkpoints saved by this manager, after each completed save.
        saved_paths.append(path)
        if len(saved_paths) > max_ckpt_to_keep:
            shutil.rmtree(saved_paths.pop(0))

    def load_checkpoint(path, del_local_after_load):
        """Restore real model bytes while leaving distributed optimizer/RNG recovery to device acceptance."""
        assert not del_local_after_load
        case.model.load_state_dict(torch.load(Path(path) / "model_world_size_2_rank_0.pt", weights_only=False))

    actor = SimpleNamespace(
        config=config,
        actor_rollout_wg=SimpleNamespace(
            save_checkpoint=save_checkpoint, load_checkpoint=Mock(side_effect=load_checkpoint)
        ),
        train_dataloader=SimpleNamespace(state_dict=lambda: {"position": 12}, load_state_dict=Mock()),
        checkpoint_manager=SimpleNamespace(update_weights=Mock()),
    )
    actor._load_checkpoint = lambda: trainer_methods.load(actor)
    trainer_methods.start(actor)
    assert actor.global_steps == 0
    actor.actor_rollout_wg.load_checkpoint.assert_not_called()
    assert (snapshots / "step_0000/adapter_model.safetensors").is_file()
    for step in (10, 20, 30, 40, 50):
        actor.global_steps = step
        trainer_methods.save(actor)
        if persistent_backup:
            backup = Path(config.trainer.h3_comparison_checkpoint_backup_dir)
            receipt = json.loads((backup / "backup_manifest.json").read_text())
            assert receipt["status"] == "COMPLETE" and receipt["step"] == step
            assert (backup / f"global_step_{step}/data.pt").is_file()
            assert (snapshots / f"step_{step:04d}/snapshot_manifest.json").is_file()
        restored = tiny_actor(801)
        api.load_h3_comparison_lora(restored, snapshots / f"step_{step:04d}")
        assert api.tensor_manifest(get_peft_model_state_dict(restored)) == api.tensor_manifest(
            get_peft_model_state_dict(case.model)
        )
        if step == 30:
            config.trainer.resume_mode = "resume_path"
            resume = Path(config.trainer.default_local_dir) / "global_step_30"
            config.trainer.resume_from_path = str(resume)
            api.validate_h3_comparison(config)
            before = {path: path.stat().st_mtime_ns for path in snapshots.rglob("*")}
            case.model = tiny_actor(937)
            saved_paths.clear()  # A new FSDP manager does not register the loaded path for rotation.
            trainer_methods.start(actor)
            assert actor.global_steps == 30
            actor.actor_rollout_wg.load_checkpoint.assert_called_once_with(
                str(resume / "actor"), del_local_after_load=False
            )
            actor.train_dataloader.load_state_dict.assert_called_once_with({"position": 12})
            actor.checkpoint_manager.update_weights.assert_called_with(30)
            assert api.tensor_manifest(get_peft_model_state_dict(case.model)) == api.tensor_manifest(
                get_peft_model_state_dict(restored)
            )
            assert before == {path: path.stat().st_mtime_ns for path in snapshots.rglob("*")}
    assert sorted(path.name for path in snapshots.iterdir()) == [f"step_{step:04d}" for step in api.SNAPSHOT_STEPS]
    assert len(list(Path(config.trainer.default_local_dir).glob("global_step_*/actor"))) == 2
    assert (Path(config.trainer.resume_from_path) / "actor").is_dir()
    assert (snapshots / "step_0010/adapter_model.safetensors").is_file()
    assert json.loads((snapshots / "step_0050/snapshot_manifest.json").read_text())["exact_training_resume"] is False
    with pytest.raises(FileExistsError):
        api.save_h3_comparison_snapshot(config, 0)


@pytest.mark.parametrize(
    "key,value",
    [
        ("trainer.resume_mode", "auto"),
        ("actor_rollout_ref.actor.checkpoint.async_save", True),
        ("actor_rollout_ref.actor.checkpoint.save_lora_only", True),
        ("trainer.save_freq", 20),
    ],
)
def test_reject_unsupported_execution(api, case, key, value):
    """Reject settings that would destroy the explicit start or completed-file contract."""
    OmegaConf.update(case.config, key, value)
    with pytest.raises(ValueError):
        api.validate_h3_comparison(case.config)


@pytest.mark.parametrize(
    "missing",
    [
        "data.pt",
        "actor/model_world_size_2_rank_1.pt",
        "actor/optim_world_size_2_rank_1.pt",
        "actor/extra_state_world_size_2_rank_1.pt",
    ],
)
def test_reject_incomplete_resume(api, case, tmp_path, missing):
    """Refuse partial checkpoint recovery, including the trainer's data-from-scratch fallback."""
    resume = tmp_path / "global_step_30"
    (resume / "actor").mkdir(parents=True)
    (resume / "data.pt").touch()
    for rank in range(2):
        for prefix in ("model", "optim", "extra_state"):
            (resume / "actor" / f"{prefix}_world_size_2_rank_{rank}.pt").touch()
    case.config.trainer.resume_mode = "resume_path"
    case.config.trainer.resume_from_path = str(resume)
    (resume / missing).unlink()
    with pytest.raises(FileNotFoundError, match=Path(missing).name):
        api.validate_h3_comparison(case.config)


@pytest.mark.parametrize("contents", ["save_contents", "load_contents"])
def test_reject_partial_resume_configuration(api, case, contents):
    """Require optimizer and scheduler/RNG state rather than accepting a model-only restart."""
    case.config.trainer.resume_mode = "resume_path"
    case.config.actor_rollout_ref.actor.checkpoint[contents] = ["model"]
    with pytest.raises(ValueError, match="model, optimizer, and extra"):
        api.validate_h3_comparison(case.config)


def test_disabled_experiment_and_initial_only(api, case):
    """Leave ordinary runs untouched and allow initialization without snapshot I/O for timing."""
    api.validate_h3_comparison(OmegaConf.create({"actor_rollout_ref": {"model": {}}, "trainer": {}}))
    case.config.trainer.h3_comparison_lora_snapshot_dir = None
    case.config.trainer.save_freq = -1
    api.validate_h3_comparison(case.config)
    api.save_h3_comparison_snapshot(case.config, 0)
    assert not Path(case.config.trainer.default_local_dir).exists()
