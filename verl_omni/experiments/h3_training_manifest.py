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
"""Freeze and verify the first 50 H3 training batches without changing live RNG state."""

import hashlib
import json
from itertools import islice
from pathlib import Path

from verl_omni.agent_loop.utils import maybe_per_rollout_seeds
from verl_omni.experiments.h3_evaluation_manifest import json_sha256
from verl_omni.utils.dataset.rl_dataset import create_rl_sampler

TRAINING_STEPS = 50
PROMPTS_PER_STEP = 32
ROLLOUT_N = 8


def h3_training_prompt(source_row_index, data_source, messages, extra_info):
    """Identify an actual text-only H3 input, including message structure and source metadata."""
    prompt_text = "\n".join(message["content"] for message in messages if message["content"]).strip()
    return {
        "source_row_index": source_row_index,
        "source_index": extra_info["index"],
        "sample_id": f"{data_source}:{extra_info['split']}:{extra_info['index']}",
        "prompt_sha256": hashlib.sha256(prompt_text.encode("utf-8")).hexdigest(),
        "raw_prompt_sha256": json_sha256(messages),
        "extra_info_sha256": json_sha256(extra_info),
    }


def h3_training_schedule(config, train_dataset, source_indices=None):
    """Freeze real DataLoader order offline, or check supplied frozen indices without touching RNG online."""
    data = config.data
    rollout = config.actor_rollout_ref.rollout
    sampler_config = data.get("sampler")
    if sampler_config and sampler_config.get("class_path") is not None and sampler_config.get("class_name") is not None:
        raise ValueError("H3 training manifest requires the default VeRL sampler with its independent generator")
    dataset_config = data.get("custom_cls")
    if dataset_config and dataset_config.get("path") is not None:
        raise ValueError("H3 training manifest requires the default RLHFDataset")
    if (
        data.seed != 42
        or not data.shuffle
        or data.train_batch_size != PROMPTS_PER_STEP
        or data.get("gen_batch_size", data.train_batch_size) != PROMPTS_PER_STEP
        or rollout.seed != 42
        or rollout.n != ROLLOUT_N
    ):
        raise ValueError("H3 training manifest requires data/rollout seed42, shuffle=true, and 32 prompts x n8")
    if len(train_dataset) < TRAINING_STEPS * PROMPTS_PER_STEP:
        raise ValueError("H3 training manifest requires all 50 batches to fit in the first epoch")
    rows = [train_dataset.dataframe[index] for index in range(len(train_dataset))]
    if source_indices is None:
        import torch
        from torchdata.stateful_dataloader import StatefulDataLoader

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
        source_indices = [int(index) for batch in islice(loader, TRAINING_STEPS) for index in batch]
    steps = []
    for step in range(1, TRAINING_STEPS + 1):
        prompts = []
        for index in source_indices[(step - 1) * PROMPTS_PER_STEP : step * PROMPTS_PER_STEP]:
            row = rows[index]
            if train_dataset[index]["raw_prompt"] != row["prompt"]:
                raise ValueError(f"H3 training raw_prompt differs from source row {index}")
            prompts.append(h3_training_prompt(index, row["data_source"], row["prompt"], row["extra_info"]))
        record = {
            "global_step": step,
            "rollout_base_seed": 42 + step - 1,
            "prompts": prompts,
            "request_seeds": maybe_per_rollout_seeds({"rollout_seed": 42 + step - 1}, PROMPTS_PER_STEP * ROLLOUT_N),
        }
        record["input_identity_sha256"] = json_sha256(record)
        steps.append(record)
    return {
        "format": "h3-training-manifest-v1",
        "dataset_row_count": len(rows),
        "all_row_fields_sha256": json_sha256(rows),
        "sampling": {
            "data_seed": 42,
            "shuffle": True,
            "prompts_per_step": 32,
            "rollout_n": 8,
            "rollout_seed": 42,
            "dataloader_num_workers": data.dataloader_num_workers,
        },
        "request_order": "prompt-major interleave: expanded_index = prompt_position * 8 + completion_index",
        "steps": steps,
    }


def create_h3_training_manifest(config, train_dataset, parquet_path):
    """Freeze actual dataset semantics and separately record the source Parquet byte identity."""
    import pyarrow.parquet as parquet

    manifest = h3_training_schedule(config, train_dataset)
    path = Path(parquet_path)
    table = parquet.read_table(path)
    if json_sha256(table.to_pylist()) != manifest["all_row_fields_sha256"]:
        raise ValueError("H3 training dataset differs from the complete source Parquet")
    manifest["source_files"] = [
        {
            "path": str(path.resolve()),
            "bytes": path.stat().st_size,
            "parquet_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "row_count": table.num_rows,
            "columns": table.column_names,
        }
    ]
    return manifest


def validate_h3_training_dataset(config, train_dataset, manifest_path):
    """Reject changed data or input schedules while leaving the live sampler and global RNG untouched."""
    manifest = json.loads(Path(manifest_path).read_text())
    source_indices = [prompt["source_row_index"] for step in manifest["steps"] for prompt in step["prompts"]]
    actual = h3_training_schedule(config, train_dataset, source_indices)
    if any(manifest[field] != value for field, value in actual.items()):
        raise ValueError("H3 training dataset or first-50-step input schedule differs from the frozen manifest")
    return manifest


def record_h3_training_batch(manifest, batch, global_step, output_path):
    """Check a real pre-repeat 32-row DataProto and append its verified identities and derived request seeds."""
    if not 1 <= global_step <= TRAINING_STEPS:
        raise ValueError("H3 training input recording supports frozen steps 1 through 50")
    expected = manifest["steps"][global_step - 1]
    fields = batch.non_tensor_batch
    if len(fields["raw_prompt"]) != PROMPTS_PER_STEP:
        raise ValueError("H3 training input recording requires the original 32 prompts before rollout repeat")
    prompts = [
        h3_training_prompt(
            expected["prompts"][index]["source_row_index"],
            fields["data_source"][index],
            fields["raw_prompt"][index],
            fields["extra_info"][index],
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
