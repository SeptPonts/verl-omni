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
"""Check V1 input identity utilities without advancing a live algorithm stream."""

import copy
import importlib
import json
import random
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf

ROOT = Path(__file__).parents[2]


@pytest.fixture
def api(monkeypatch):
    """Load the real pure utilities without accelerator registration side effects."""
    before = set(sys.modules)
    for name in ("verl_omni", "verl_omni.agent_loop", "verl_omni.experiments"):
        package = ModuleType(name)
        package.__path__ = [str(ROOT.joinpath(*name.split(".")))]
        monkeypatch.setitem(sys.modules, name, package)
    yield importlib.import_module("verl_omni.experiments.h3_training_manifest")
    for name in set(sys.modules) - before:
        if name.startswith("verl_omni."):
            sys.modules.pop(name, None)


class SourceDataset:
    """Expose the dataframe/raw_prompt contract consumed by the utility."""

    def __init__(self, rows):
        """Keep source rows so tests can mutate observed input semantics."""
        self.dataframe = rows

    def __len__(self):
        """Return the number of source rows."""
        return len(self.dataframe)

    def __getitem__(self, index):
        """Expose raw messages exactly as the text-only RLHFDataset does."""
        return {**self.dataframe[index], "raw_prompt": self.dataframe[index]["prompt"]}


@pytest.fixture
def case(api, tmp_path):
    """Freeze three explicit input batches with source IDs distinct from row positions."""
    rows = [
        {
            "data_source": "minimax_h3_t2va",
            "prompt": [{"role": "user", "content": f"  声音 {index}: a bell.\n  "}],
            "reward_model": {"style": "model", "ground_truth": str(index)},
            "extra_info": {"index": index + 5000, "split": "train"},
        }
        for index in range(101)
    ]
    config = OmegaConf.create(
        {
            "data": {
                "seed": 42,
                "shuffle": True,
                "train_batch_size": 32,
                "gen_batch_size": None,
                "dataloader_num_workers": 0,
            },
            "actor_rollout_ref": {"rollout": {"seed": 42, "n": 8}},
            "trainer": {"use_v1": False},
        }
    )
    dataset = SourceDataset(rows)
    indices = list(reversed(range(96)))
    manifest = api.h3_training_schedule(config, dataset, indices, num_steps=3)
    path = tmp_path / "training.json"
    path.write_text(json.dumps(manifest))
    return SimpleNamespace(rows=rows, dataset=dataset, config=config, indices=indices, manifest=manifest, path=path)


def batch_fields(case, step):
    """Build the original four raw columns plus intentionally overwritten V1 transport IDs."""
    selected = [case.dataset[index] for index in case.indices[(step - 1) * 32 : step * 32]]
    fields = {
        name: [row[name] for row in selected] for name in ("raw_prompt", "data_source", "extra_info", "reward_model")
    }
    fields.update(index=list(range(32)), uid=["transport-only"] * 32)
    return fields


def test_v0_uses_the_same_frozen_source_indices_without_v1_settings(api, case):
    """A V0 input ledger authenticates source rows without acquiring V1 queue semantics."""
    case.config.trainer.use_v1 = False
    assert "v1" not in case.config.trainer
    assert api.validate_h3_training_dataset(case.config, case.dataset, case.path) == case.manifest


def test_raw_batch_uses_manifest_positions_and_preserves_rng(api, case, tmp_path):
    """Verification consumes no RNG and trusts original source metadata, never transport index/uid."""
    sampler = torch.Generator().manual_seed(42)
    sampler_state = sampler.get_state().clone()
    torch_state = torch.random.get_rng_state().clone()
    python_state = random.getstate()
    fields = batch_fields(case, 3)
    original = copy.deepcopy(fields)
    assert api.validate_h3_training_dataset(case.config, case.dataset, case.path) == case.manifest
    result = api.record_h3_training_batch(case.manifest, fields, 3, tmp_path / "inputs.jsonl")
    assert result == case.manifest["steps"][2]
    assert result["prompts"][0]["source_row_index"] == 31
    assert result["prompts"][0]["source_index"] == 5031
    assert result["request_seeds"] == list(range(44_000_132, 44_000_388))
    assert fields == original
    assert sampler.get_state().equal(sampler_state)
    assert torch.random.get_rng_state().equal(torch_state)
    assert random.getstate() == python_state
    assert json.loads((tmp_path / "inputs.jsonl").read_text()) == result
    assert "raw_prompt" not in result["prompts"][0]


@pytest.mark.parametrize("change", ["order", "prompt", "source_id", "metadata", "reward", "expanded"])
def test_wrong_batch_fails_before_append(api, case, tmp_path, change):
    """Reject input, reward and prompt-to-seed changes before emitting evidence."""
    fields = copy.deepcopy(batch_fields(case, 1))
    if change == "order":
        fields = {name: list(reversed(values)) for name, values in fields.items()}
    elif change == "prompt":
        fields["raw_prompt"][0][0]["content"] += "changed"
    elif change == "source_id":
        fields["extra_info"][0]["index"] += 1
    elif change == "metadata":
        fields["extra_info"][0]["condition"] = "changed"
    elif change == "reward":
        fields["reward_model"][0]["ground_truth"] += "changed"
    else:
        fields = {name: [value for value in values for completion in range(8)] for name, values in fields.items()}
    destination = tmp_path / "records.jsonl"
    with pytest.raises(ValueError, match="identities or order|original 32"):
        api.record_h3_training_batch(case.manifest, fields, 1, destination)
    assert not destination.exists()


@pytest.mark.parametrize("field", ["raw_prompt_sha256", "request_seeds", "input_identity_sha256", "all_rows"])
def test_manifest_corruption_or_unused_row_change_is_rejected(api, case, field):
    """Validate complete source identity, including rows outside the frozen prefix."""
    if field == "all_rows":
        case.rows[-1]["reward_model"]["ground_truth"] = "changed"
    elif field == "raw_prompt_sha256":
        case.manifest["steps"][0]["prompts"][0][field] = "bad"
    elif field == "request_seeds":
        case.manifest["steps"][0][field][0] += 1
    else:
        case.manifest["steps"][0][field] = "bad"
    case.path.write_text(json.dumps(case.manifest))
    with pytest.raises(ValueError, match="dataset or input schedule"):
        api.validate_h3_training_dataset(case.config, case.dataset, case.path)


@pytest.mark.parametrize(
    "key,value",
    [
        ("data.seed", None),
        ("data.shuffle", False),
        ("data.train_batch_size", 16),
        ("data.gen_batch_size", 1),
        ("actor_rollout_ref.rollout.seed", 43),
        ("actor_rollout_ref.rollout.n", 4),
        ("trainer.use_v1", True),
        ("actor_rollout_ref.separate", True),
        ("data.custom_cls", {"path": "not_imported.py"}),
        ("data.sampler", {"class_path": "not_imported.py", "class_name": "Custom"}),
    ],
)
def test_reject_different_sampling_contract(api, case, key, value):
    """Reject modes whose mapping of generation batches to fixed steps is different."""
    OmegaConf.update(case.config, key, value)
    with pytest.raises(ValueError, match="requires"):
        api.validate_h3_training_dataset(case.config, case.dataset, case.path)


@pytest.mark.parametrize("indices", [list(range(95)), [-1] + list(range(95)), [101] + list(range(95))])
def test_invalid_frozen_index_shape_or_range(api, case, indices):
    """Do not accept truncated schedules or Python negative indexing as source provenance."""
    with pytest.raises(ValueError, match="exactly num_steps"):
        api.h3_training_schedule(case.config, case.dataset, indices, num_steps=3)


def test_ambiguous_source_identity_is_rejected(api, case):
    """Physical row positions cannot authenticate duplicated original sample IDs."""
    case.rows[-1]["extra_info"] = case.rows[0]["extra_info"].copy()
    with pytest.raises(ValueError, match="identities must be unique"):
        api.h3_training_schedule(case.config, case.dataset, case.indices, num_steps=3)


def test_longer_frozen_window_and_step_boundaries(api, case, tmp_path):
    """Use the manifest's explicit length and retain step one as the first training record."""
    indices = list(range(32)) * 51
    manifest = api.h3_training_schedule(case.config, case.dataset, indices, num_steps=51)
    fields = {
        name: [case.dataset[index][name] for index in range(32)]
        for name in ("raw_prompt", "data_source", "extra_info", "reward_model")
    }
    assert api.record_h3_training_batch(manifest, fields, 51, tmp_path / "last.jsonl") == manifest["steps"][-1]
    for step in (0, 52):
        with pytest.raises(ValueError, match="frozen global step"):
            api.record_h3_training_batch(manifest, fields, step, tmp_path / "invalid.jsonl")
    assert not (tmp_path / "invalid.jsonl").exists()


@pytest.mark.parametrize("workers", [0, 2])
def test_actual_loader_schedule_and_resume_preserve_live_stream(api, case, tmp_path, workers, monkeypatch):
    """Check real StatefulDataLoader order across epochs and resumed state when dependencies exist."""
    loader_module = pytest.importorskip("torchdata.stateful_dataloader")
    pytest.importorskip("verl")
    for name in ("verl_omni.utils", "verl_omni.utils.dataset"):
        package = ModuleType(name)
        package.__path__ = [str(ROOT.joinpath(*name.split(".")))]
        monkeypatch.setitem(sys.modules, name, package)
    dataset_api = importlib.import_module("verl_omni.utils.dataset.rl_dataset")
    case.config.data.dataloader_num_workers = workers
    global_state = torch.random.get_rng_state().clone()
    manifest = api.h3_training_schedule(case.config, case.dataset, num_steps=7)
    assert torch.random.get_rng_state().equal(global_state)
    sampler = dataset_api.create_rl_sampler(case.config.data, case.dataset)
    loader = loader_module.StatefulDataLoader(
        range(len(case.dataset)), batch_size=32, sampler=sampler, drop_last=True, num_workers=workers
    )
    iterator = iter(loader)
    for step in range(1, 8):
        try:
            indices = next(iterator).tolist()
        except StopIteration:
            iterator = iter(loader)
            indices = next(iterator).tolist()
        assert indices == [prompt["source_row_index"] for prompt in manifest["steps"][step - 1]["prompts"]]
        if step == 4:
            saved = loader.state_dict()
    restored = loader_module.StatefulDataLoader(
        range(len(case.dataset)),
        batch_size=32,
        sampler=dataset_api.create_rl_sampler(case.config.data, case.dataset),
        drop_last=True,
        num_workers=workers,
    )
    restored.load_state_dict(saved)
    assert next(iter(restored)).tolist() == [prompt["source_row_index"] for prompt in manifest["steps"][4]["prompts"]]
