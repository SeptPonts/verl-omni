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
"""Check original per-step evidence, exact tensor hashes and explicit MAPE boundaries."""

import copy
import importlib
import json
import sys
from pathlib import Path
from types import ModuleType

import pytest
import torch

ROOT = Path(__file__).parents[2]


@pytest.fixture
def api(monkeypatch):
    """Load the CPU utility without model, PEFT, checkpoint or device registration."""
    before = set(sys.modules)
    for name in ("verl_omni", "verl_omni.experiments"):
        package = ModuleType(name)
        package.__path__ = [str(ROOT.joinpath(*name.split(".")))]
        monkeypatch.setitem(sys.modules, name, package)
    yield importlib.import_module("verl_omni.experiments.h3_comparison")
    for name in set(sys.modules) - before:
        if name.startswith("verl_omni."):
            sys.modules.pop(name, None)


def test_same_window_mape_is_mean_of_individual_ratios(api):
    """Use the exact requested window and reference magnitudes, including negative references."""
    reference = [
        {"global_step": 1, "reward": 10000},
        {"global_step": 2, "reward": 1},
        {"global_step": 3, "reward": -10},
    ]
    candidate = [{"global_step": 3, "reward": -11}, {"global_step": 2, "reward": 2}]
    originals = copy.deepcopy((reference, candidate))
    result = api.compare_h3_records(reference, candidate, steps=[2, 3], metric_names=["reward"])
    assert result["summary"]["reward"]["mape_percent"] == pytest.approx(55)
    assert result["summary"]["reward"]["mae"] == 1
    assert result["pairs"][1]["components"]["reward"] == {
        "reference": -10,
        "candidate": -11,
        "delta": -1,
        "absolute_error": 1,
        "absolute_percentage_error": 10,
    }
    assert (reference, candidate) == originals


@pytest.mark.parametrize("candidate", [0, 3])
def test_zero_reference_is_explicitly_undefined(api, candidate):
    """Never hide zero/zero or nonzero/zero behind epsilon or a silently reduced denominator."""
    result = api.compare_h3_records(
        [{"global_step": 1, "reward": 0}, {"global_step": 2, "reward": 4}],
        [{"global_step": 1, "reward": candidate}, {"global_step": 2, "reward": 5}],
        steps=[1, 2],
        metric_names=["reward"],
    )
    summary = result["summary"]["reward"]
    assert summary["mape_percent"] is None
    assert summary["nonzero_reference_mape_percent"] == 25
    assert summary["zero_reference_count"] == summary["nonzero_reference_count"] == 1
    assert summary["zero_reference_mismatch_count"] == int(candidate != 0)
    assert result["pairs"][0]["components"]["reward"]["absolute_percentage_error"] is None
    json.dumps(result, allow_nan=False)


def test_all_zero_reference_reports_counts_and_absolute_errors(api):
    """No denominator remains available when all baseline component values are zero."""
    result = api.compare_h3_records(
        [{"global_step": 0, "loss": 0}], [{"global_step": 0, "loss": 2}], steps=[0], metric_names=["loss"]
    )
    assert result["summary"]["loss"]["mape_percent"] is None
    assert result["summary"]["loss"]["nonzero_reference_mape_percent"] is None
    assert result["summary"]["loss"]["mae"] == 2


def test_per_sample_components_keep_original_values(api):
    """Pair stable sample IDs independently of completion order and retain distinct reward components."""
    reference = [
        {"global_step": 10, "sample_id": sample, "prompt_sha256": sample, "audio": 2, "video": 4}
        for sample in ("first", "second")
    ]
    candidate = copy.deepcopy(list(reversed(reference)))
    candidate[0]["audio"] = 3
    candidate[0]["video"] = 2
    result = api.compare_h3_records(
        reference,
        candidate,
        steps=[10],
        metric_names=["audio", "video"],
        sample_key="sample_id",
        identity_fields=("prompt_sha256",),
    )
    assert result["summary"]["audio"]["mape_percent"] == 25
    assert result["summary"]["video"]["mape_percent"] == 25
    assert result["pairs"][1]["sample_id"] == "second"
    assert result["pairs"][1]["components"]["video"]["delta"] == -2


@pytest.mark.parametrize(
    "problem", ["missing_step", "duplicate", "sample_id", "identity", "missing_metric", "nan", "inf", "string", "bool"]
)
def test_reject_unmatched_or_invalid_raw_evidence(api, problem):
    """Do not turn incomplete, ambiguous or numerically invalid raw data into a successful comparison."""
    reference = [{"global_step": step, "sample_id": "a", "hash": "same", "reward": 2} for step in (1, 2)]
    candidate = copy.deepcopy(reference)
    if problem == "missing_step":
        candidate.pop()
    elif problem == "duplicate":
        candidate.append(copy.deepcopy(candidate[0]))
    elif problem == "sample_id":
        candidate[0]["sample_id"] = "b"
    elif problem == "identity":
        candidate[0]["hash"] = "different"
    elif problem == "missing_metric":
        del candidate[0]["reward"]
    else:
        candidate[0]["reward"] = {"nan": float("nan"), "inf": float("inf"), "string": "2", "bool": True}[problem]
    with pytest.raises((ValueError, KeyError)):
        api.compare_h3_records(
            reference,
            candidate,
            steps=[1, 2],
            metric_names=["reward"],
            sample_key="sample_id",
            identity_fields=("hash",),
        )


@pytest.mark.parametrize("steps", [[], [1, 1], [-1], [1.0]])
def test_invalid_window_is_rejected(api, steps):
    """Require an explicit nonempty window instead of inferring an intersection."""
    with pytest.raises(ValueError, match="steps"):
        api.compare_h3_records([], [], steps=steps, metric_names=["reward"])


def test_tensor_hash_is_cpu_exact_for_scalar_bf16_and_noncontiguous(api):
    """Container layout is irrelevant, while dtype, shape and even signed zero bits remain significant."""
    source = torch.arange(12, dtype=torch.bfloat16).reshape(3, 4).t()
    before = torch.random.get_rng_state().clone()
    assert api.tensor_manifest({"weight": source}) == api.tensor_manifest({"weight": source.contiguous()})
    assert api.tensor_manifest({"weight": source}) != api.tensor_manifest({"weight": source.float()})
    assert api.tensor_manifest({"scalar": torch.tensor(1.0)})["scalar"]["shape"] == []
    assert api.tensor_manifest({"a": torch.tensor(0.0)}) != api.tensor_manifest({"a": torch.tensor(-0.0)})
    assert torch.random.get_rng_state().equal(before)
    with pytest.raises(ValueError, match="materialized CPU"):
        api.tensor_manifest({"unmaterialized": torch.empty(1, device="meta")})


def test_manifest_comparison_ignores_paths_and_bytes_but_reports_semantics(api):
    """Separate encoding identity from input equivalence and retain raw field changes."""
    reference = {
        "format": "h3-evaluation-manifest-v1",
        "selection_rule": "all 128 rows",
        "sampling": {"seed": 42, "n": 1},
        "consumed_fields_sha256": "same",
        "reevaluation": {},
        "source_files": [{"path": "/gpu/test.parquet", "parquet_sha256": "encoding1", "all_row_fields_sha256": "same"}],
        "samples": [{"sample_id": "source:test:1", "prompt_text": "original"}],
    }
    candidate = copy.deepcopy(reference)
    candidate["source_files"][0].update(path="/mlu/test.parquet", parquet_sha256="encoding2")
    assert api.compare_h3_manifests(reference, candidate)["equal"]
    candidate["samples"][0]["prompt_text"] = "changed"
    result = api.compare_h3_manifests(reference, candidate)
    assert not result["equal"]
    assert result["differences"] == [
        {"path": "/samples/0/prompt_text", "reference": "original", "candidate": "changed"}
    ]
    candidate = copy.deepcopy(reference)
    candidate["sampling"]["seed"] = 42.0
    assert not api.compare_h3_manifests(reference, candidate)["equal"]


def test_training_manifest_hash_comparison(api):
    """Report per-step hash changes and exclude source-file encoding metadata."""
    reference = {
        "format": "h3-training-manifest-v2",
        "num_steps": 1,
        "dataset_row_count": 32,
        "all_row_fields_sha256": "same",
        "sampling": {"data_seed": 42},
        "request_order": "prompt-major",
        "steps": [{"global_step": 1, "input_identity_sha256": "frozen"}],
    }
    candidate = copy.deepcopy(reference)
    assert api.compare_h3_manifests(reference, candidate)["equal"]
    candidate["steps"][0]["input_identity_sha256"] = "different"
    assert api.compare_h3_manifests(reference, candidate)["differences"][0]["path"] == "/steps/0/input_identity_sha256"


def test_raw_diff_distinguishes_missing_and_null(api):
    """Report field presence so a missing value cannot masquerade as a recorded null."""
    difference = api.json_differences({}, {"optional": None})[0]
    assert not difference["reference_present"] and difference["candidate_present"]


@pytest.mark.parametrize("change", ["complete", "missing_both", "wrong_hash_both", "duplicate"])
def test_fixed128_raw_records_are_bound_to_manifest(api, change):
    """Two runs with identical omissions or wrong identities must not pass fixed128 acceptance."""
    samples = [{"sample_id": f"sample:{index}", "prompt_sha256": f"hash:{index}"} for index in range(128)]
    manifest = {"format": "h3-evaluation-manifest-v1", "samples": samples, "consumed_fields_sha256": "frozen"}
    reference = [{**sample, "global_step": step, "reward": 1} for step in (0, 10) for sample in samples]
    if change == "missing_both":
        reference.pop()
    elif change == "wrong_hash_both":
        reference[0]["prompt_sha256"] = "same but unfrozen"
    elif change == "duplicate":
        reference.append(copy.deepcopy(reference[0]))
    candidate = copy.deepcopy(list(reversed(reference)))
    if change == "complete":
        report = api.compare_h3_evaluation_records(
            reference, candidate, manifest, steps=[0, 10], metric_names=["reward"]
        )
        assert report["summary"]["reward"]["pair_count"] == 256
        assert report["evaluation_inputs_sha256"] == "frozen"
    else:
        with pytest.raises(ValueError):
            api.compare_h3_evaluation_records(reference, candidate, manifest, steps=[0, 10], metric_names=["reward"])


def test_existing_file_logger_projection_and_cli(api, tmp_path, monkeypatch):
    """Read main's actual step/data shape and write an immutable component comparison report."""
    reference, candidate, output = [tmp_path / name for name in ("left.jsonl", "right.jsonl", "report.json")]
    reference.write_text(json.dumps({"step": 1, "data": {"reward/audio": 2}}) + "\n")
    candidate.write_text(json.dumps({"step": 1, "data": {"reward/audio": 3}}) + "\n")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "compare",
            "--reference",
            str(reference),
            "--candidate",
            str(candidate),
            "--output",
            str(output),
            "--steps",
            "1",
            "--metrics",
            "reward/audio",
            "--file-logger",
        ],
    )
    api.main()
    report = json.loads(output.read_text())
    assert report["summary"]["reward/audio"]["mape_percent"] == 50
    assert report["source_files"]["reference"]["sha256"] == api.file_sha256(reference)
    with pytest.raises(FileExistsError):
        api.main()
