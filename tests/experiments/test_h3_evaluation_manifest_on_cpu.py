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
"""Verify fixed128 source semantics separately from Parquet byte identity."""

import copy
import importlib
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pyarrow as arrow
import pyarrow.parquet as parquet
import pytest
from omegaconf import OmegaConf

ROOT = Path(__file__).parents[2]


@pytest.fixture
def api(monkeypatch):
    """Load the real utility without importing device/model registrations."""
    before = set(sys.modules)
    for name in ("verl_omni", "verl_omni.experiments", "verl_omni.agent_loop"):
        package = ModuleType(name)
        package.__path__ = [str(ROOT.joinpath(*name.split(".")))]
        monkeypatch.setitem(sys.modules, name, package)
    yield importlib.import_module("verl_omni.experiments.h3_evaluation_manifest")
    for name in set(sys.modules) - before:
        if name.startswith("verl_omni."):
            sys.modules.pop(name, None)


class SourceDataset:
    """Expose the source dataframe and the runtime raw_prompt projection."""

    def __init__(self, rows):
        """Keep the ordered source inputs."""
        self.dataframe = rows

    def __len__(self):
        """Return source row count."""
        return len(self.dataframe)

    def __getitem__(self, index):
        """Return raw source messages."""
        return {"raw_prompt": self.dataframe[index]["prompt"]}


@pytest.fixture
def case(api, tmp_path):
    """Build the existing all-row contract with stable IDs distinct from row positions."""
    rows = [
        {
            "data_source": "minimax_h3_t2va",
            "prompt": [{"role": "system", "content": ""}, {"role": "user", "content": f"  样本 {i}\n  "}],
            "reward_model": {"style": "model", "ground_truth": str(i)},
            "extra_info": {"index": i + 5000, "split": "test"},
            "ability": "text_to_audio_video",
        }
        for i in range(128)
    ]
    path = tmp_path / "validation.parquet"
    parquet.write_table(arrow.Table.from_pylist(rows), path)
    manifest = api.create_h3_evaluation_manifest(path)
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    config = OmegaConf.create(
        {"data": {"validation_shuffle": False}, "actor_rollout_ref": {"rollout": {"val_kwargs": {"seed": 42, "n": 1}}}}
    )
    return SimpleNamespace(
        rows=rows, path=path, manifest=manifest, manifest_path=manifest_path, config=config, dataset=SourceDataset(rows)
    )


def test_fixed128_identity_and_real_text_consumer(api, case):
    """Match the existing message utility and fixed reevaluation subset without resampling."""
    text_api = importlib.import_module("verl_omni.agent_loop.utils")
    manifest = api.validate_h3_evaluation_dataset(case.config, case.dataset, case.manifest_path)
    assert manifest == case.manifest
    assert manifest["reevaluation"]["evaluation_indices"] == [0, 18, 36, 54, 72, 90, 108, 127]
    for index, sample in enumerate(manifest["samples"]):
        assert sample["source_row_index"] == index
        assert sample["sample_id"] == f"minimax_h3_t2va:test:{index + 5000}"
        assert sample["prompt_text"] == text_api.messages_to_text(case.rows[index]["prompt"])
        assert sample["seed"] == 42 and sample["n"] == 1


def test_different_parquet_bytes_same_semantics(api, case, tmp_path):
    """Accept alternate compression and key ordering without equating their byte hashes."""
    path = tmp_path / "other.parquet"
    rows = [dict(reversed(list(row.items()))) for row in case.rows]
    parquet.write_table(arrow.Table.from_pylist(rows), path, compression=None)
    actual = api.create_h3_evaluation_manifest(path)
    assert actual["source_files"][0]["parquet_sha256"] != case.manifest["source_files"][0]["parquet_sha256"]
    assert actual["consumed_fields_sha256"] == case.manifest["consumed_fields_sha256"]
    assert actual["samples"] == case.manifest["samples"]
    api.validate_h3_evaluation_dataset(case.config, SourceDataset(rows), case.manifest_path)


@pytest.mark.parametrize("change", ["order", "prompt", "ground_truth", "source_id", "negative_prompt", "metadata"])
def test_changed_source_inputs_rejected(api, case, change):
    """Check reward, metadata and added conditioning fields as well as prompt text."""
    rows = copy.deepcopy(case.rows)
    if change == "order":
        rows.reverse()
    elif change == "prompt":
        rows[0]["prompt"][1]["content"] += "changed"
    elif change == "ground_truth":
        rows[0]["reward_model"]["ground_truth"] = "changed"
    elif change == "source_id":
        rows[0]["extra_info"]["index"] += 1000
    elif change == "negative_prompt":
        rows[0]["negative_prompt"] = [{"role": "user", "content": "changed"}]
    else:
        rows[-1]["ability"] = "changed"
    with pytest.raises(ValueError, match="source rows differ"):
        api.validate_h3_evaluation_dataset(case.config, SourceDataset(rows), case.manifest_path)


@pytest.mark.parametrize("change", ["count", "duplicate_id", "reevaluation", "seed", "shuffle", "n"])
def test_reject_changed_selection_or_sampling(api, case, change):
    """Fail on subset changes and on evaluation settings that break the fixed128 contract."""
    if change == "count":
        case.rows.pop()
    elif change == "duplicate_id":
        case.rows[1]["extra_info"] = case.rows[0]["extra_info"].copy()
    elif change == "reevaluation":
        case.manifest["reevaluation"]["evaluation_indices"][1] = 19
        case.manifest_path.write_text(json.dumps(case.manifest))
    elif change == "shuffle":
        case.config.data.validation_shuffle = True
    else:
        case.config.actor_rollout_ref.rollout.val_kwargs[change] += 1
    with pytest.raises(ValueError):
        api.validate_h3_evaluation_dataset(case.config, case.dataset, case.manifest_path)


def test_canonical_hash_rejects_nonfinite_values_and_preserves_types(api):
    """Canonical identity ignores key order, not numeric type or nonfinite data."""
    assert api.json_sha256({"a": "声音", "b": 2}) == api.json_sha256({"b": 2, "a": "声音"})
    assert api.json_sha256(1) != api.json_sha256(1.0)
    with pytest.raises(ValueError):
        api.json_sha256(float("nan"))
