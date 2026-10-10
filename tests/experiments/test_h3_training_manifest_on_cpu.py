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
"""Validate frozen H3 inputs against real Parquet, VeRL samplers, and StatefulDataLoader."""

import importlib
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pyarrow as arrow
import pyarrow.parquet as parquet
import pytest
import torch
from datasets import Dataset
from omegaconf import OmegaConf
from torchdata.stateful_dataloader import StatefulDataLoader
from verl import DataProto

ROOT = Path(__file__).parents[2]


@pytest.fixture
def api(monkeypatch):
    """Import actual data and seed utilities without unrelated accelerator/model registration."""
    before = set(sys.modules)
    for name in ("verl_omni", "verl_omni.utils", "verl_omni.utils.dataset", "verl_omni.agent_loop"):
        package = ModuleType(name)
        package.__path__ = [str(ROOT.joinpath(*name.split(".")))]
        monkeypatch.setitem(sys.modules, name, package)
    module = importlib.import_module("verl_omni.experiments.h3_training_manifest")
    dataset = importlib.import_module("verl_omni.utils.dataset.rl_dataset")
    yield SimpleNamespace(manifest=module, dataset=dataset)
    for name in set(sys.modules) - before:
        if name.startswith("verl_omni."):
            sys.modules.pop(name, None)


@pytest.fixture
def case(api, tmp_path):
    """Build more than 50 full batches, with source IDs distinct from physical row positions."""
    rows = [
        {
            "data_source": "minimax_h3_t2va",
            "prompt": [{"role": "user", "content": f"  声音 {index}: a bell rings.\n  "}],
            "ability": "text_to_audio_video",
            "reward_model": {"style": "model", "ground_truth": f"sample {index}"},
            "extra_info": {"index": index + 5000, "split": "train"},
        }
        for index in range(1667)
    ]
    path = tmp_path / "train.parquet"
    parquet.write_table(arrow.Table.from_pylist(rows), path)
    config = OmegaConf.create(
        {
            "data": {
                "seed": 42,
                "shuffle": True,
                "train_batch_size": 32,
                "dataloader_num_workers": 0,
                "filter_overlong_prompts": False,
                "cache_dir": str(tmp_path / "cache"),
            },
            "actor_rollout_ref": {"rollout": {"seed": 42, "n": 8}},
        }
    )
    dataset = api.dataset.RLHFDataset(data_files=str(path), tokenizer=None, processor=None, config=config.data)
    manifest = api.manifest.create_h3_training_manifest(config, dataset, path)
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    return SimpleNamespace(
        rows=rows, path=path, config=config, dataset=dataset, manifest=manifest, manifest_path=manifest_path
    )


@pytest.mark.parametrize("workers", [0, 8])
def test_real_fifty_batches_match_and_validator_preserves_rng(api, case, tmp_path, workers):
    """Check all 50 actual pre-repeat batches without advancing the live sampler or default RNG."""
    case.config.data.dataloader_num_workers = workers
    global_state = torch.random.get_rng_state().clone()
    case.manifest = api.manifest.create_h3_training_manifest(case.config, case.dataset, case.path)
    assert torch.equal(torch.random.get_rng_state(), global_state)
    case.manifest_path.write_text(json.dumps(case.manifest))
    sampler = api.dataset.create_rl_sampler(case.config.data, case.dataset)
    loader = StatefulDataLoader(
        case.dataset,
        batch_size=32,
        sampler=sampler,
        drop_last=True,
        num_workers=workers,
        collate_fn=api.dataset.get_collate_fn(case.config.data),
    )
    generator_state = sampler.generator.get_state().clone()
    global_state = torch.random.get_rng_state().clone()
    manifest = api.manifest.validate_h3_training_dataset(case.config, case.dataset, case.manifest_path)
    assert torch.equal(sampler.generator.get_state(), generator_state)
    assert torch.equal(torch.random.get_rng_state(), global_state)
    output = tmp_path / "run/input_batches.jsonl"
    for step, batch_dict in enumerate(loader, start=1):
        if step > 50:
            break
        batch = DataProto.from_single_dict(batch_dict)
        before = torch.random.get_rng_state().clone()
        record = api.manifest.record_h3_training_batch(manifest, batch, step, output)
        assert torch.equal(torch.random.get_rng_state(), before)
        assert record == manifest["steps"][step - 1]
        assert len(record["request_seeds"]) == 256
        for prompt in record["prompts"]:
            assert prompt["source_index"] == prompt["source_row_index"] + 5000
    actual = [json.loads(line) for line in output.read_text().splitlines()]
    assert actual == manifest["steps"]
    assert actual[2]["request_seeds"] == list(range(44_000_132, 44_000_388))
    assert actual[-1]["request_seeds"] == list(range(91_000_273, 91_000_529))


def test_fresh_runs_share_six_step_prefix_after_unrelated_rng_use(api, case, tmp_path):
    """Initial validation/default RNG activity cannot alter a fresh sampler's first six training batches."""
    case.config.data.dataloader_num_workers = 8
    case.manifest = api.manifest.create_h3_training_manifest(case.config, case.dataset, case.path)
    runs = []
    for run in range(3):
        torch.rand(17 * (run + 1))
        loader = StatefulDataLoader(
            case.dataset,
            batch_size=32,
            sampler=api.dataset.create_rl_sampler(case.config.data, case.dataset),
            drop_last=True,
            num_workers=8,
            collate_fn=api.dataset.get_collate_fn(case.config.data),
        )
        records = []
        for step, batch_dict in enumerate(loader, start=1):
            records.append(
                api.manifest.record_h3_training_batch(
                    case.manifest, DataProto.from_single_dict(batch_dict), step, tmp_path / f"run-{run}.jsonl"
                )
            )
            if step == 6:
                break
        runs.append(records)
    assert runs[0] == runs[1] == runs[2] == case.manifest["steps"][:6]


def test_parquet_bytes_are_separate_from_complete_row_semantics(api, case, tmp_path):
    """Different Parquet encoding is allowed only when all source rows and the full schedule match."""
    path = tmp_path / "other.parquet"
    rows = [dict(reversed(list(row.items()))) for row in case.rows]
    parquet.write_table(arrow.Table.from_pylist(rows), path, compression=None)
    other = api.manifest.create_h3_training_manifest(case.config, case.dataset, path)
    assert other["source_files"][0]["parquet_sha256"] != case.manifest["source_files"][0]["parquet_sha256"]
    assert other["all_row_fields_sha256"] == case.manifest["all_row_fields_sha256"]
    assert other["steps"] == case.manifest["steps"]
    case.rows[-1]["ability"] = "changed metadata"
    case.dataset.dataframe = Dataset.from_list(case.rows)
    with pytest.raises(ValueError, match="dataset or first-50-step"):
        api.manifest.validate_h3_training_dataset(case.config, case.dataset, case.manifest_path)


@pytest.mark.parametrize("change", ["order", "prompt", "source_index", "extra_info", "expanded"])
def test_changed_actual_batch_fails_before_writing(api, case, tmp_path, change):
    """Reject wrong prompt-to-seed pairings and changed source metadata before emitting evidence."""
    indices = [prompt["source_row_index"] for prompt in case.manifest["steps"][2]["prompts"]]
    batch = DataProto.from_single_dict(api.dataset.get_collate_fn(case.config.data)([case.dataset[i] for i in indices]))
    if change == "order":
        batch = batch.select_idxs(list(reversed(range(32))))
    elif change == "prompt":
        batch.non_tensor_batch["raw_prompt"][0][0]["content"] += "changed"
    elif change == "source_index":
        batch.non_tensor_batch["extra_info"][0]["index"] = 9999
    elif change == "extra_info":
        batch.non_tensor_batch["extra_info"][0]["new_condition"] = True
    else:
        batch = batch.repeat(repeat_times=8, interleave=True)
    output = tmp_path / "invalid.jsonl"
    with pytest.raises(ValueError, match="batch identities or order|original 32 prompts"):
        api.manifest.record_h3_training_batch(case.manifest, batch, 3, output)
    assert not output.exists()


@pytest.mark.parametrize(
    "key,value",
    [
        ("data.seed", 43),
        ("data.shuffle", False),
        ("data.train_batch_size", 16),
        ("data.gen_batch_size", 64),
        ("actor_rollout_ref.rollout.seed", 43),
        ("actor_rollout_ref.rollout.n", 4),
    ],
)
def test_reject_changed_sampling_contract(api, case, key, value):
    """Fail explicitly if a run changes the frozen data ordering or prompt expansion contract."""
    OmegaConf.update(case.config, key, value)
    with pytest.raises(ValueError, match="data/rollout seed42"):
        api.manifest.validate_h3_training_dataset(case.config, case.dataset, case.manifest_path)


def test_reject_edited_request_seed_manifest(api, case):
    """Keep the manifest's request seeds bound to the real worker seed implementation."""
    case.manifest["steps"][2]["request_seeds"][0] += 1
    case.manifest_path.write_text(json.dumps(case.manifest))
    with pytest.raises(ValueError, match="dataset or first-50-step"):
        api.manifest.validate_h3_training_dataset(case.config, case.dataset, case.manifest_path)


@pytest.mark.parametrize(
    "key,value",
    [
        ("data.sampler", {"class_path": "unused.py", "class_name": "CustomSampler"}),
        ("data.custom_cls", {"path": "unused.py", "name": "CustomDataset"}),
    ],
)
def test_reject_custom_rng_ownership_before_import(api, case, key, value):
    """Do not invoke custom hooks whose sampler or item loading may consume global RNG state."""
    OmegaConf.update(case.config, key, value)
    with pytest.raises(ValueError, match="requires the default"):
        api.manifest.validate_h3_training_dataset(case.config, case.dataset, case.manifest_path)


@pytest.mark.parametrize("restored_step", [3, 40])
def test_real_multiworker_data_pt_resumes_at_next_frozen_step(api, case, tmp_path, restored_step):
    """Restore actual prefetched loader state and verify the next batch still matches the frozen schedule."""
    case.config.data.dataloader_num_workers = 8
    manifest = api.manifest.create_h3_training_manifest(case.config, case.dataset, case.path)
    loader = StatefulDataLoader(
        case.dataset,
        batch_size=32,
        sampler=api.dataset.create_rl_sampler(case.config.data, case.dataset),
        drop_last=True,
        num_workers=8,
        collate_fn=api.dataset.get_collate_fn(case.config.data),
    )
    iterator = iter(loader)
    for step in range(restored_step):
        next(iterator)
    state_path = tmp_path / "data.pt"
    torch.save(loader.state_dict(), state_path)
    del iterator, loader
    torch.rand(173)
    restored = StatefulDataLoader(
        case.dataset,
        batch_size=32,
        sampler=api.dataset.create_rl_sampler(case.config.data, case.dataset),
        drop_last=True,
        num_workers=8,
        collate_fn=api.dataset.get_collate_fn(case.config.data),
    )
    restored.load_state_dict(torch.load(state_path, weights_only=False))
    record = api.manifest.record_h3_training_batch(
        manifest, DataProto.from_single_dict(next(iter(restored))), restored_step + 1, tmp_path / "resumed.jsonl"
    )
    assert record == manifest["steps"][restored_step]
