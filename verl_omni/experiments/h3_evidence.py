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
"""H3 experiment evidence callbacks, re-exported directly by the selected provider.

Run manifests authenticate source inputs. Attempt ledgers record observed rows,
request seeds and rewards before balancing or queue cleanup. Adapter snapshots
live on the run evidence volume, independently of checkpoint retention; their
receipts describe files, not live model loading or completed training.
"""

import json
import math
import os
import shutil
from numbers import Integral, Real
from pathlib import Path

import numpy as np

from verl_omni.experiments.h3_comparison import tensor_manifest
from verl_omni.experiments.h3_evaluation_manifest import (
    EVALUATION_SIZE,
    create_h3_evaluation_manifest,
    file_sha256,
    h3_evaluation_samples,
    validate_h3_evaluation_dataset,
)

TRAINING_MANIFEST = None
EVALUATION_MANIFEST = None
INPUT_FIELDS = ("raw_prompt", "data_source", "extra_info", "reward_model")
REWARD_FIELDS = ("reward/clap", "reward/imagebind", "reward/combined")


class H3EvidenceContractError(ValueError):
    """Observed evidence is missing, ambiguous, or differs from the frozen contract."""


def json_value(value):
    """Normalize actual NumPy columns recursively without rounding their numeric values."""
    if isinstance(value, np.ndarray | np.generic):
        return json_value(value.tolist())
    if isinstance(value, dict):
        return {key: json_value(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [json_value(item) for item in value]
    return value


def require_fields(mapping, names, context):
    """Reject absent external fields before projecting an evidence record."""
    missing = set(names) - mapping.keys()
    if missing:
        raise H3EvidenceContractError(f"{context}: missing fields {sorted(missing)}")


def finite_score(value, context):
    """Require a real, finite observed scalar rather than fabricating a missing score."""
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
        raise H3EvidenceContractError(f"{context}: expected a finite numeric score, got {value!r}")
    return float(value)


def write_json(path, value):
    """Serialize completely before exclusively creating an immutable JSON receipt."""
    serialized = json.dumps(json_value(value), ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        stream.write(serialized + "\n")


def write_ledger(path, records):
    """Validate serialization of the full batch before creating its one-attempt JSONL ledger."""
    serialized = "".join(
        json.dumps(json_value(record), ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n" for record in records
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        stream.write(serialized)


def input_columns(fields, count):
    """Read the original dataset columns, never the randomized uid or overwritten index."""
    require_fields(fields, INPUT_FIELDS, "H3 input")
    values = {name: json_value(fields[name]) for name in INPUT_FIELDS}
    if any(len(column) != count for column in values.values()):
        raise H3EvidenceContractError(f"H3 input requires exactly {count} actual rows")
    return values


def observed_sampling(fields, count):
    """Read every actual request's seed and indices from the returned agent-loop metadata."""
    require_fields(fields, ("experiment_sampling",), "H3 sampling")
    rows = json_value(fields["experiment_sampling"])
    if len(rows) != count:
        raise H3EvidenceContractError(f"H3 sampling requires exactly {count} returned metadata rows")
    for row in rows:
        if not isinstance(row, dict):
            raise H3EvidenceContractError("H3 sampling metadata must be present for every request")
        require_fields(row, ("seed", "prompt_index", "session_id"), "H3 sampling")
        if any(type(row[key]) is not int or row[key] < 0 for key in ("seed", "prompt_index", "session_id")):
            raise H3EvidenceContractError("H3 sampling metadata must contain nonnegative integer seed and indices")
    return rows


def reward_rows(columns, count):
    """Project actual extract_reward columns, preserving every returned component key."""
    values = json_value(columns)
    if any(len(column) != count for column in values.values()):
        raise H3EvidenceContractError(f"H3 reward columns require exactly {count} values")
    return [{key: column[index] for key, column in values.items()} for index in range(count)]


def checked_rewards(rows, count):
    """Require the manager's three raw reward keys on every row, retaining all extras."""
    if len(rows) != count:
        raise H3EvidenceContractError(f"H3 rewards require exactly {count} actual rows")
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise H3EvidenceContractError(f"H3 reward row {index} has no raw component mapping")
        require_fields(row, REWARD_FIELDS, f"H3 reward row {index}")
        for name in REWARD_FIELDS:
            finite_score(row[name], f"H3 row {index} {name}")
    return rows


def prepare_data(config, train_dataset, val_dataset):
    """Freeze or verify real V1 inputs once per run without consuming the live sampler."""
    global TRAINING_MANIFEST, EVALUATION_MANIFEST
    from verl_omni.experiments.h3_training_manifest import (
        create_h3_training_manifest,
        validate_h3_training_dataset,
    )

    directory = Path(os.environ["H3_RUN_DIR"]) / "manifests"
    training, evaluation = directory / "training.json", directory / "evaluation.json"
    if config.trainer.resume_mode == "disable":
        write_json(
            training,
            create_h3_training_manifest(
                config, train_dataset, config.data.train_files, num_steps=config.trainer.total_training_steps
            ),
        )
        write_json(evaluation, create_h3_evaluation_manifest(config.data.val_files))
    elif config.trainer.resume_mode != "resume_path":
        raise H3EvidenceContractError("H3 evidence requires explicit fresh or resume_path mode")
    training_manifest = validate_h3_training_dataset(config, train_dataset, training)
    evaluation_manifest = validate_h3_evaluation_dataset(config, val_dataset, evaluation)
    TRAINING_MANIFEST, EVALUATION_MANIFEST = training_manifest, evaluation_manifest


def record_input_batch(config, batch_dict, global_step):
    """Authenticate the observed original 32 rows before repeat or transport index replacement."""
    from verl_omni.experiments.h3_training_manifest import PROMPTS_PER_STEP, record_h3_training_batch

    if TRAINING_MANIFEST is None:
        raise H3EvidenceContractError("Call prepare_data before recording H3 inputs")
    values = input_columns(batch_dict, PROMPTS_PER_STEP)
    path = Path(os.environ["H3_ATTEMPT_DIR"]) / "training-inputs.jsonl"
    if path.exists():
        with path.open(encoding="utf-8") as stream:
            if any(json.loads(line)["global_step"] == global_step for line in stream):
                raise H3EvidenceContractError(f"Duplicate H3 input batch at step {global_step}")
    return record_h3_training_batch(TRAINING_MANIFEST, values, global_step, path)


def record_sampling_request(result, sampling_params, session_id, prompt_fields):
    """Attach the seed actually supplied to the completed request, not a reconstructed expectation."""
    require_fields(sampling_params, ("seed",), "H3 request sampling_params")
    require_fields(prompt_fields, ("index",), "H3 request prompt fields")
    values = {"seed": sampling_params["seed"], "prompt_index": prompt_fields["index"], "session_id": session_id}
    if any(isinstance(value, bool) or not isinstance(value, Integral) or value < 0 for value in values.values()):
        raise H3EvidenceContractError("H3 request requires an actual nonnegative integer seed and indices")
    result.extra_fields["experiment_sampling"] = {key: int(value) for key, value in values.items()}


def record_training_candidates(config, global_step, data):
    """Record all 32 x 8 observed candidates before balancing, rejecting partial or duplicate coverage."""
    from verl.trainer.ppo.reward import extract_reward

    from verl_omni.experiments.h3_training_manifest import PROMPTS_PER_STEP, ROLLOUT_N, h3_training_prompt

    if TRAINING_MANIFEST is None or not 1 <= global_step <= TRAINING_MANIFEST["num_steps"]:
        raise H3EvidenceContractError("H3 candidates require a prepared manifest and frozen training step")
    expected = TRAINING_MANIFEST["steps"][global_step - 1]
    count = PROMPTS_PER_STEP * ROLLOUT_N
    fields = data.non_tensor_batch
    values = input_columns(fields, count)
    sampling = observed_sampling(fields, count)
    reward_tensor, columns = extract_reward(data)
    scores = reward_tensor.sum(-1).detach().cpu().tolist()
    # Colocated scoring owns flattened columns; TQ returns the original per-row dictionaries.
    if columns:
        components = reward_rows(columns, count)
    else:
        require_fields(fields, ("reward_extra_info",), "H3 TQ rewards")
        components = json_value(fields["reward_extra_info"])
    checked_rewards(components, count)
    if len(scores) != count:
        raise H3EvidenceContractError("H3 candidate rm_scores must cover all 256 rows")
    records, seen = [], set()
    for index, metadata in enumerate(sampling):
        prompt_index, session_id = metadata["prompt_index"], metadata["session_id"]
        if not 0 <= prompt_index < PROMPTS_PER_STEP or not 0 <= session_id < ROLLOUT_N:
            raise H3EvidenceContractError("H3 candidate prompt/session indices are outside the frozen 32 x 8 grid")
        key = (prompt_index, session_id)
        if key in seen:
            raise H3EvidenceContractError(f"Duplicate H3 candidate {key}")
        seen.add(key)
        prompt = expected["prompts"][prompt_index]
        actual = h3_training_prompt(
            prompt["source_row_index"],
            values["data_source"][index],
            values["raw_prompt"][index],
            values["extra_info"][index],
            values["reward_model"][index],
        )
        if actual != prompt:
            raise H3EvidenceContractError(f"H3 candidate source identity differs from manifest at row {index}")
        expanded_index = prompt_index * ROLLOUT_N + session_id
        if metadata["seed"] != expected["request_seeds"][expanded_index]:
            raise H3EvidenceContractError(f"H3 actual request seed differs from manifest at candidate {key}")
        records.append(
            {
                **prompt,
                "global_step": global_step,
                "input_index": index,
                "expanded_index": expanded_index,
                **metadata,
                "total": finite_score(scores[index], f"H3 candidate {key} total"),
                "components": components[index],
            }
        )
    write_ledger(Path(os.environ["H3_ATTEMPT_DIR"]) / f"training-candidates-{global_step}.jsonl", records)


def record_evaluation(config, global_step, data, scores, reward_extra_info):
    """Authenticate and record all fixed128 rows, their actual seeds and every raw component."""
    if EVALUATION_MANIFEST is None:
        raise H3EvidenceContractError("Call prepare_data before recording H3 evaluation")
    values = input_columns(data.non_tensor_batch, EVALUATION_SIZE)
    sampling = observed_sampling(data.non_tensor_batch, EVALUATION_SIZE)
    rows = [
        {"prompt": values["raw_prompt"][index], **{name: values[name][index] for name in INPUT_FIELDS[1:]}}
        for index in range(EVALUATION_SIZE)
    ]
    samples = h3_evaluation_samples(rows)
    if samples != EVALUATION_MANIFEST["samples"]:
        raise H3EvidenceContractError("H3 validation rows differ from frozen source identities or order")
    components = checked_rewards(reward_rows(reward_extra_info, EVALUATION_SIZE), EVALUATION_SIZE)
    if len(scores) != EVALUATION_SIZE:
        raise H3EvidenceContractError("H3 evaluation scores must cover all fixed128 rows")
    records = []
    for index, sample in enumerate(samples):
        metadata = sampling[index]
        if metadata != {"seed": sample["seed"], "prompt_index": index, "session_id": 0}:
            raise H3EvidenceContractError(f"H3 evaluation actual request seed/identity differs at row {index}")
        records.append(
            {
                **sample,
                **metadata,
                "global_step": global_step,
                "total": finite_score(scores[index], f"H3 evaluation row {index} total"),
                "components": components[index],
            }
        )
    write_ledger(Path(os.environ["H3_ATTEMPT_DIR"]) / f"evaluation-{global_step}.jsonl", records)


def complete_step(config, global_step):
    """Retain saved checkpoints only after validation media and the step's scalar logging complete."""
    if global_step > 0 and global_step % config.trainer.save_freq == 0:
        from verl_omni.experiments.h3_checkpoint_retention import audit_and_retain_checkpoint

        actor = Path(config.trainer.default_local_dir) / f"global_step_{global_step}" / "actor"
        audit_and_retain_checkpoint(config, global_step, actor)


def snapshot_destination(config, global_step):
    """Keep attempt-specific adapter weights on the run evidence volume, outside checkpoint storage."""
    if type(global_step) is not int or global_step < 0:
        raise H3EvidenceContractError("H3 snapshot step must be a nonnegative integer")
    attempt = Path(os.environ["H3_ATTEMPT_DIR"]).resolve().name
    output = Path(os.environ["H3_RUN_DIR"]).expanduser().resolve() / "lora-snapshots" / attempt / f"step{global_step}"
    if output.is_relative_to(Path(config.trainer.default_local_dir).expanduser().resolve()):
        raise H3EvidenceContractError("H3 LoRA snapshots must stay on run evidence storage, outside checkpoint storage")
    if output.exists():
        raise FileExistsError(output)
    return output


def adapter_evidence(directory, weights_name):
    """Read the actual saved PEFT config and all 624 CPU tensor identities without a model."""
    from safetensors.torch import load_file

    config = json.loads((directory / "adapter_config.json").read_text(encoding="utf-8"))
    require_fields(config, ("r", "lora_alpha"), "H3 adapter config")
    if config["r"] != 64 or config["lora_alpha"] != 128:
        raise H3EvidenceContractError("H3 snapshot requires actual rank64/alpha128/scaling2")
    if any(config.get(name) for name in ("use_rslora", "use_dora", "rank_pattern", "alpha_pattern")):
        raise H3EvidenceContractError("H3 snapshot requires uniform ordinary LoRA scaling")
    tensors = tensor_manifest(load_file(directory / weights_name, device="cpu"))
    if len(tensors) != 624:
        raise H3EvidenceContractError(f"H3 snapshot requires 624 tensors, found {len(tensors)}")
    evidence = {
        "adapter_config": config,
        "r": config["r"],
        "lora_alpha": config["lora_alpha"],
        "scaling": config["lora_alpha"] / config["r"],
        "tensor_count": len(tensors),
        "tensors": tensors,
    }
    if weights_name == "pytorch_lora_weights.safetensors":
        from safetensors import safe_open

        with safe_open(directory / weights_name, framework="pt", device="cpu") as stream:
            metadata = stream.metadata()
        if metadata is None:
            raise H3EvidenceContractError("H3 converted safetensors metadata is absent")
        require_fields(metadata, ("format", "lora_adapter_metadata"), "H3 converted safetensors metadata")
        actual_metadata = json.loads(metadata["lora_adapter_metadata"])
        if metadata["format"] != "pt" or actual_metadata != {
            "transformer." + key: value for key, value in config.items()
        }:
            raise H3EvidenceContractError("H3 converted safetensors metadata differs from the actual adapter config")
        conversion = json.loads((directory / "conversion_receipt.json").read_text(encoding="utf-8"))
        require_fields(
            conversion,
            ("status", "r", "lora_alpha", "scaling", "tensor_count", "tensors", "target_files_sha256"),
            "H3 conversion receipt",
        )
        if any(
            conversion[key] != value
            for key, value in {
                "status": "COMPLETE",
                "r": 64,
                "lora_alpha": 128,
                "scaling": 2,
                "tensor_count": 624,
            }.items()
        ):
            raise H3EvidenceContractError("H3 conversion receipt is incomplete or has different rank/alpha/scaling")
        mapped_tensors = {
            record["main_key"]: {key: record[key] for key in ("shape", "dtype", "sha256")}
            for record in conversion["tensors"].values()
        }
        if len(conversion["tensors"]) != 624 or mapped_tensors != tensors:
            raise H3EvidenceContractError("H3 converted snapshot tensors differ from the complete conversion receipt")
        for name in ("adapter_config.json", weights_name):
            if file_sha256(directory / name) != conversion["target_files_sha256"][name]:
                raise H3EvidenceContractError(f"H3 converted snapshot file differs from conversion receipt: {name}")
        evidence["safetensors_metadata"] = metadata
        evidence["conversion_receipt"] = conversion
    return evidence


def snapshot_initial(config, global_step):
    """Copy a fresh converted step0 adapter, or snapshot the actual explicitly resumed checkpoint."""
    if config.trainer.resume_mode == "resume_path":
        checkpoint = Path(config.trainer.resume_from_path).expanduser().resolve()
        if (
            global_step <= 0
            or checkpoint.name != f"global_step_{global_step}"
            or not (checkpoint / "data.pt").is_file()
        ):
            raise H3EvidenceContractError(
                "H3 resume snapshot requires the explicit matching full checkpoint and data.pt"
            )
        return snapshot_checkpoint(config, global_step, checkpoint / "actor")
    if config.trainer.resume_mode != "disable" or global_step != 0:
        raise H3EvidenceContractError("Only an explicitly fresh step0 may snapshot the converted canonical adapter")
    source = Path(config.actor_rollout_ref.model.lora_adapter_path).expanduser().resolve()
    before = adapter_evidence(source, "pytorch_lora_weights.safetensors")
    output = snapshot_destination(config, global_step)
    output.mkdir(parents=True, exist_ok=False)
    for name in ("adapter_config.json", "pytorch_lora_weights.safetensors", "conversion_receipt.json"):
        shutil.copyfile(source / name, output / name)
    after = adapter_evidence(output, "pytorch_lora_weights.safetensors")
    if before != after:
        raise H3EvidenceContractError("H3 step0 snapshot changed the converted canonical adapter")
    receipt = {
        "global_step": global_step,
        "source": str(source),
        "output": str(output),
        "source_format": "main-diffusers-converted",
        "model_load": "NOT_RUN",
        **after,
    }
    write_json(Path(os.environ["H3_ATTEMPT_DIR"]) / "lora-snapshots" / "step0.json", receipt)
    return receipt


def snapshot_checkpoint(config, global_step, actor_checkpoint_path):
    """Export only LoRA through the existing native shard reader after the full checkpoint save."""
    from verl_omni.utils.fsdp_utils import export_fsdp_lora_adapter

    if global_step <= 0:
        raise H3EvidenceContractError("H3 native checkpoint snapshots require a positive saved step")
    source = Path(actor_checkpoint_path).expanduser().resolve()
    if not source.is_dir():
        raise H3EvidenceContractError(f"H3 saved actor checkpoint does not exist: {source}")
    output = snapshot_destination(config, global_step)
    output.mkdir(parents=True, exist_ok=False)
    exported = export_fsdp_lora_adapter(source, output)
    evidence = adapter_evidence(output, "adapter_model.safetensors")
    receipt = {
        "global_step": global_step,
        "source": str(source),
        "output": str(output),
        "source_format": "native-peft",
        "main_compatible_reevaluation": "Requires prepare_h3_adapter conversion; model loading NOT_RUN",
        "export": exported,
        **evidence,
    }
    write_json(Path(os.environ["H3_ATTEMPT_DIR"]) / "lora-snapshots" / f"step{global_step}.json", receipt)
    return receipt
