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
"""Freeze V1 sync H3 input batches on CPU without advancing the live sampling stream.

Adapted from feature/h3-mlu-support-legacy. Generation batches are recorded before
repeat or TransferQueue submission; successful learner steps remain the caller's
responsibility. Frozen source indices are checked online, never resampled there.
"""

import hashlib
import json
from pathlib import Path
from typing import Any

from verl_omni.agent_loop.utils import maybe_per_rollout_seeds
from verl_omni.experiments.h3_evaluation_manifest import file_sha256, json_sha256

TRAINING_STEPS = 50
PROMPTS_PER_STEP = 32
ROLLOUT_N = 8


def h3_training_prompt(
    source_row_index: int, data_source: str, messages: list[dict], extra_info: dict, reward_model: dict
) -> dict:
    """Identify an actual text-only H3 input, including message structure and source metadata."""
    if any(not isinstance(message["content"], str) for message in messages):
        raise ValueError("H3 training manifest requires text-only prompts")
    prompt_text = "\n".join(message["content"] for message in messages if message["content"]).strip()
    return {
        "source_row_index": source_row_index,
        "source_index": extra_info["index"],
        "sample_id": f"{data_source}:{extra_info['split']}:{extra_info['index']}",
        "prompt_sha256": hashlib.sha256(prompt_text.encode("utf-8")).hexdigest(),
        "raw_prompt_sha256": json_sha256(messages),
        "extra_info_sha256": json_sha256(extra_info),
        "reward_model_sha256": json_sha256(reward_model),
    }


def h3_training_schedule(
    config: Any, train_dataset: Any, source_indices: list[int] | None = None, *, num_steps: int = TRAINING_STEPS
) -> dict:
    """Freeze real DataLoader order offline, or check supplied frozen indices without touching RNG online."""
    data = config.data
    rollout = config.actor_rollout_ref.rollout
    v1 = config.trainer.v1
    if v1.trainer_mode != "sync" or v1.sampler.drop_incomplete_groups:
        raise ValueError("H3 frozen input schedule requires V1 sync without incomplete-group refill")
    sampler_config = data.get("sampler")
    if sampler_config and sampler_config.get("class_path") is not None and sampler_config.get("class_name") is not None:
        raise ValueError("H3 training manifest requires the default verl sampler with its independent generator")
    dataset_config = data.get("custom_cls")
    if dataset_config and dataset_config.get("path") is not None:
        raise ValueError("H3 training manifest requires the default RLHFDataset")
    if (
        data.seed is None
        or not data.shuffle
        or data.train_batch_size != PROMPTS_PER_STEP
        or (data.get("gen_batch_size") or data.train_batch_size) != PROMPTS_PER_STEP
        or rollout.seed != 42
        or rollout.n != ROLLOUT_N
    ):
        raise ValueError(
            "H3 training manifest requires an explicit data seed, rollout seed42, shuffle=true, and 32 prompts x n8"
        )
    if type(num_steps) is not int or num_steps <= 0:
        raise ValueError("H3 training manifest requires a positive integer num_steps")
    if len(train_dataset) < PROMPTS_PER_STEP:
        raise ValueError("H3 training manifest requires at least one complete batch")
    rows = [train_dataset.dataframe[index] for index in range(len(train_dataset))]
    source_ids = [(row["data_source"], row["extra_info"]["split"], row["extra_info"]["index"]) for row in rows]
    if len(set(source_ids)) != len(rows):
        raise ValueError("H3 training source identities must be unique")
    if source_indices is None:
        import torch
        from torchdata.stateful_dataloader import StatefulDataLoader

        from verl_omni.utils.dataset.rl_dataset import create_rl_sampler

        # Multiworker StatefulDataLoader resets the sampler iterator during initialization.
        # Its real lifecycle, not direct sampler iteration, determines the first permutation.
        loader = StatefulDataLoader(
            range(len(train_dataset)),
            batch_size=PROMPTS_PER_STEP,
            sampler=create_rl_sampler(data, train_dataset),
            drop_last=True,
            num_workers=data.dataloader_num_workers,
            generator=torch.Generator(),
        )
        iterator = iter(loader)
        source_indices = []
        for step in range(num_steps):
            try:
                batch = next(iterator)
            except StopIteration:
                iterator = iter(loader)
                batch = next(iterator)
            source_indices.extend(int(index) for index in batch)
    if len(source_indices) != num_steps * PROMPTS_PER_STEP or any(
        type(index) is not int or not 0 <= index < len(rows) for index in source_indices
    ):
        raise ValueError("H3 frozen indices must name exactly num_steps complete source batches")
    steps = []
    for step in range(1, num_steps + 1):
        prompts = []
        for index in source_indices[(step - 1) * PROMPTS_PER_STEP : step * PROMPTS_PER_STEP]:
            row = rows[index]
            if train_dataset[index]["raw_prompt"] != row["prompt"]:
                raise ValueError(f"H3 training raw_prompt differs from source row {index}")
            prompts.append(
                h3_training_prompt(index, row["data_source"], row["prompt"], row["extra_info"], row["reward_model"])
            )
        record = {
            "global_step": step,
            "rollout_base_seed": 42 + step - 1,
            "prompts": prompts,
            "request_seeds": maybe_per_rollout_seeds({"rollout_seed": 42 + step - 1}, PROMPTS_PER_STEP * ROLLOUT_N),
        }
        record["input_identity_sha256"] = json_sha256(record)
        steps.append(record)
    return {
        "format": "h3-training-manifest-v2",
        "num_steps": num_steps,
        "dataset_row_count": len(rows),
        "all_row_fields_sha256": json_sha256(rows),
        "sampling": {
            "data_seed": data.seed,
            "shuffle": True,
            "prompts_per_step": 32,
            "rollout_n": 8,
            "rollout_seed": 42,
            "dataloader_num_workers": data.dataloader_num_workers,
        },
        "request_order": "prompt-major interleave: expanded_index = prompt_position * 8 + completion_index",
        "steps": steps,
    }


def create_h3_training_manifest(
    config: Any, train_dataset: Any, parquet_path: str | Path, *, num_steps: int = TRAINING_STEPS
) -> dict:
    """Freeze actual dataset semantics and separately record the source Parquet byte identity."""
    import pyarrow.parquet as parquet

    manifest = h3_training_schedule(config, train_dataset, num_steps=num_steps)
    path = Path(parquet_path)
    table = parquet.read_table(path)
    if json_sha256(table.to_pylist()) != manifest["all_row_fields_sha256"]:
        raise ValueError("H3 training dataset differs from the complete source Parquet")
    manifest["source_files"] = [
        {
            "path": str(path.resolve()),
            "bytes": path.stat().st_size,
            "parquet_sha256": file_sha256(path),
            "row_count": table.num_rows,
            "columns": table.column_names,
        }
    ]
    return manifest


def validate_h3_training_dataset(config: Any, train_dataset: Any, manifest_path: str | Path) -> dict:
    """Check frozen indices and data without resampling; actual batches must still be recorded.

    The trusted manifest is created with an independent loader before the live
    iterator starts. This check does not prove that arbitrary supplied indices
    came from that loader; record_h3_training_batch checks each observed batch.
    """
    manifest = json.loads(Path(manifest_path).read_text())
    if manifest["format"] != "h3-training-manifest-v2" or len(manifest["steps"]) != manifest["num_steps"]:
        raise ValueError("H3 training manifest format or step count differs from V1 input contract")
    source_indices = [prompt["source_row_index"] for step in manifest["steps"] for prompt in step["prompts"]]
    actual = h3_training_schedule(config, train_dataset, source_indices, num_steps=manifest["num_steps"])
    if any(manifest[field] != value for field, value in actual.items()):
        raise ValueError("H3 training dataset or input schedule differs from the frozen manifest")
    return manifest


def record_h3_training_batch(manifest: dict, fields: dict[str, Any], global_step: int, output_path: str | Path) -> dict:
    """Check pre-repeat V1 batch fields and append identities without modifying caller data.

    Pass raw_prompt/data_source/extra_info/reward_model sequences, extracted by
    the owning hook from its TensorDict. No DataProto or live sampler is needed.
    """
    if not 1 <= global_step <= manifest["num_steps"]:
        raise ValueError("H3 training input recording requires a frozen global step")
    expected = manifest["steps"][global_step - 1]
    if any(
        len(fields[name]) != PROMPTS_PER_STEP for name in ("raw_prompt", "data_source", "extra_info", "reward_model")
    ):
        raise ValueError("H3 training input recording requires the original 32 prompts before rollout repeat")
    prompts = [
        h3_training_prompt(
            expected["prompts"][index]["source_row_index"],
            fields["data_source"][index],
            fields["raw_prompt"][index],
            fields["extra_info"][index],
            fields["reward_model"][index],
        )
        for index in range(PROMPTS_PER_STEP)
    ]
    record = {
        "global_step": global_step,
        "rollout_base_seed": 42 + global_step - 1,
        "prompts": prompts,
        "request_seeds": maybe_per_rollout_seeds({"rollout_seed": 42 + global_step - 1}, PROMPTS_PER_STEP * ROLLOUT_N),
    }
    record["input_identity_sha256"] = json_sha256(record)
    if record != expected:
        raise ValueError(f"H3 training batch identities or order differ from manifest at step {global_step}")
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    return record
