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
"""Check fixed H3 evaluation identities against real Parquet and the actual V0 dataset."""

import ast
import importlib
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pyarrow as arrow
import pyarrow.parquet as parquet
import pytest
from omegaconf import OmegaConf

ROOT = Path(__file__).parents[2]


@pytest.fixture
def api(monkeypatch):
    """Import the actual dataset without unrelated accelerator and model registration."""
    before = set(sys.modules)
    for name in ("verl_omni", "verl_omni.utils", "verl_omni.utils.dataset"):
        package = ModuleType(name)
        package.__path__ = [str(ROOT.joinpath(*name.split(".")))]
        monkeypatch.setitem(sys.modules, name, package)
    module = importlib.import_module("verl_omni.experiments.h3_evaluation_manifest")
    dataset = importlib.import_module("verl_omni.utils.dataset.rl_dataset").RLHFDataset
    yield SimpleNamespace(manifest=module, dataset=dataset)
    for name in set(sys.modules) - before:
        if name.startswith("verl_omni."):
            sys.modules.pop(name, None)


@pytest.fixture
def case(api, tmp_path):
    """Prepare 128 real Parquet rows with original IDs distinct from their file positions."""
    rows = [
        {
            "data_source": "minimax_h3_t2va",
            "prompt": [{"role": "user", "content": f"  样本 {index}: a bell rings.\n  "}],
            "ability": "text_to_audio_video",
            "reward_model": {"style": "model", "ground_truth": f"sample {index}"},
            "extra_info": {"index": index + 5000, "split": "test"},
        }
        for index in range(128)
    ]
    path = tmp_path / "test.parquet"
    parquet.write_table(arrow.Table.from_pylist(rows), path)
    config = OmegaConf.create(
        {
            "data": {
                "seed": 42,
                "shuffle": True,
                "validation_shuffle": False,
                "filter_overlong_prompts": False,
                "cache_dir": str(tmp_path / "cache"),
            },
            "actor_rollout_ref": {"rollout": {"val_kwargs": {"seed": 42, "n": 1}}},
        }
    )
    dataset = api.dataset(data_files=str(path), tokenizer=None, processor=None, config=config.data, max_samples=128)
    manifest = api.manifest.create_h3_evaluation_manifest(path)
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False))
    return SimpleNamespace(
        rows=rows, path=path, config=config, dataset=dataset, manifest=manifest, manifest_path=manifest_path
    )


def test_actual_dataset_and_h3_prompt_semantics(api, case):
    """Preserve all source rows and match the actual H3 raw-text consumer and equal-spacing rule."""
    manifest = api.manifest.validate_h3_evaluation_dataset(case.config, case.dataset, case.manifest_path)
    assert manifest == case.manifest
    assert manifest["reevaluation"]["evaluation_indices"] == [0, 18, 36, 54, 72, 90, 108, 127]
    assert manifest["reevaluation"]["sample_ids"] == [
        f"minimax_h3_t2va:test:{index + 5000}" for index in [0, 18, 36, 54, 72, 90, 108, 127]
    ]
    source = ROOT / "verl_omni/pipelines/minimax_h3_diffusion_nft/common.py"
    method = next(
        node
        for node in ast.parse(source.read_text()).body
        if isinstance(node, ast.FunctionDef) and node.name == "messages_to_text"
    )
    namespace = {"Any": Any}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(source), "exec"), namespace)
    for index, sample in enumerate(manifest["samples"]):
        assert sample["source_row_index"] == index
        assert sample["source_index"] == index + 5000
        assert sample["prompt_text"] == namespace["messages_to_text"](case.dataset[index]["raw_prompt"])
    assert "uid" not in case.dataset[0]


def test_parquet_encoding_is_separate_from_consumed_fields(api, case, tmp_path):
    """Accept equivalent rows despite different Parquet bytes and nested mapping-key order."""
    rows = [dict(reversed(list(row.items()))) for row in case.rows]
    for row in rows:
        row["prompt"] = [dict(reversed(list(message.items()))) for message in row["prompt"]]
    path = tmp_path / "other.parquet"
    parquet.write_table(arrow.Table.from_pylist(rows), path, compression=None)
    other = api.manifest.create_h3_evaluation_manifest(path)
    assert other["source_files"][0]["parquet_sha256"] != case.manifest["source_files"][0]["parquet_sha256"]
    assert other["consumed_fields_sha256"] == case.manifest["consumed_fields_sha256"]
    assert other["samples"] == case.manifest["samples"]
    dataset = api.dataset(
        data_files=str(path), tokenizer=None, processor=None, config=case.config.data, max_samples=128
    )
    api.manifest.validate_h3_evaluation_dataset(case.config, dataset, case.manifest_path)


@pytest.mark.parametrize("change", ["order", "prompt", "ground_truth", "source_index", "negative_prompt"])
def test_reject_changed_consumed_dataset(api, case, change):
    """Reject row reordering and altered generation, reward, or source-identity inputs."""
    if change == "order":
        case.dataset.dataframe = case.dataset.dataframe.select(list(reversed(range(128))))
    else:
        rows = case.rows
        if change == "prompt":
            rows[0]["prompt"][0]["content"] += "changed"
        elif change == "ground_truth":
            rows[0]["reward_model"]["ground_truth"] += "changed"
        elif change == "source_index":
            rows[0]["extra_info"]["index"] = 9999
        else:
            rows[0]["negative_prompt"] = [{"role": "user", "content": "new conditioning input"}]
        case.dataset.dataframe = rows
    with pytest.raises(ValueError, match="source rows differ"):
        api.manifest.validate_h3_evaluation_dataset(case.config, case.dataset, case.manifest_path)


@pytest.mark.parametrize(
    "key,value",
    [
        ("data.validation_shuffle", True),
        ("actor_rollout_ref.rollout.val_kwargs.seed", 43),
        ("actor_rollout_ref.rollout.val_kwargs.n", 2),
    ],
)
def test_reject_changed_evaluation_sampling(api, case, key, value):
    """Prevent the runtime from changing fixed evaluation order or request seeds."""
    OmegaConf.update(case.config, key, value)
    with pytest.raises(ValueError, match="validation_shuffle=false"):
        api.manifest.validate_h3_evaluation_dataset(case.config, case.dataset, case.manifest_path)


@pytest.mark.parametrize("change", ["count", "duplicate_id", "reevaluation"])
def test_reject_ambiguous_or_selected_inputs(api, case, change):
    """Refuse partial datasets, ambiguous original IDs, and hand-picked reevaluation subsets."""
    if change == "reevaluation":
        case.manifest["reevaluation"]["evaluation_indices"][1] = 19
        case.manifest_path.write_text(json.dumps(case.manifest))
        with pytest.raises(ValueError, match="equally spaced"):
            api.manifest.validate_h3_evaluation_dataset(case.config, case.dataset, case.manifest_path)
    else:
        if change == "count":
            case.rows.pop()
        else:
            case.rows[1]["extra_info"] = case.rows[0]["extra_info"].copy()
        with pytest.raises(ValueError, match="128-row|identities must be unique"):
            api.manifest.h3_evaluation_samples(case.rows)
