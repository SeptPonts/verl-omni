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
"""Verify source-authenticated H3 ledgers and native snapshot delegation on CPU."""

import importlib
import importlib.util
import json
import shutil
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pyarrow as arrow
import pyarrow.parquet as parquet
import pytest
import torch
from omegaconf import OmegaConf
from safetensors.torch import load_file, save_file

ROOT = Path(__file__).parents[2]


@pytest.fixture
def api(monkeypatch):
    """Load real evidence/manifest code while isolating package registration and external inference."""
    before = set(sys.modules)
    for name in ("verl_omni", "verl_omni.experiments", "verl_omni.agent_loop"):
        package = ModuleType(name)
        package.__path__ = [str(ROOT.joinpath(*name.split(".")))]
        monkeypatch.setitem(sys.modules, name, package)
    reward = ModuleType("verl.trainer.ppo.reward")
    reward.extract_reward = Mock()
    monkeypatch.setitem(sys.modules, reward.__name__, reward)
    evidence = importlib.import_module("verl_omni.experiments.h3_evidence")
    monkeypatch.setattr(evidence, "TRAINING_MANIFEST", None)
    monkeypatch.setattr(evidence, "EVALUATION_MANIFEST", None)
    yield SimpleNamespace(evidence=evidence, extract_reward=reward.extract_reward)
    for name in set(sys.modules) - before:
        if name.startswith("verl_omni."):
            sys.modules.pop(name, None)


class SourceDataset:
    """Expose only the real dataframe/raw_prompt input contract."""

    def __init__(self, rows):
        """Keep source rows for the real frozen-manifest validators."""
        self.dataframe = rows

    def __len__(self):
        """Return actual source row count."""
        return len(self.dataframe)

    def __getitem__(self, index):
        """Return unchanged raw source messages."""
        return {"raw_prompt": self.dataframe[index]["prompt"]}


def source_rows(count, split):
    """Use original IDs distinct from physical row and transport indices."""
    return [
        {
            "data_source": "minimax_h3_t2va",
            "prompt": [{"role": "user", "content": f"  声音 {index}: a bell.\n"}],
            "extra_info": {"split": split, "index": 9000 + index},
            "reward_model": {"ground_truth": f"reference {index}", "style": "model"},
        }
        for index in range(count)
    ]


def raw_columns(rows):
    """Create the object-array columns actually restored by the V1 TQ envelope."""
    return {
        name: np.array([row["prompt" if name == "raw_prompt" else name] for row in rows], dtype=object)
        for name in ("raw_prompt", "data_source", "extra_info", "reward_model")
    }


def read_ledger(path):
    """Read every immutable row rather than accepting a mean as evidence of coverage."""
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_evidence_import_does_not_initialize_agent_loop_or_training_manifest(api, monkeypatch):
    """Provider re-exports must not pull the accelerator registration tree into its initial import."""
    monkeypatch.setitem(sys.modules, "verl_omni.agent_loop", None)
    monkeypatch.setitem(sys.modules, "verl_omni.experiments.h3_training_manifest", None)
    spec = importlib.util.spec_from_file_location("evidence_import_probe", api.evidence.__file__)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for name in (
        "prepare_data",
        "record_input_batch",
        "record_evaluation",
        "record_training_candidates",
        "record_sampling_request",
        "snapshot_initial",
        "snapshot_checkpoint",
    ):
        assert callable(getattr(module, name))


@pytest.fixture
def case(api, tmp_path, monkeypatch):
    """Build real manifests with explicit frozen indices, avoiding a live training sampler."""
    run, attempt = tmp_path / "run", tmp_path / "run/attempts/fresh-one"
    monkeypatch.setenv("H3_RUN_DIR", str(run))
    monkeypatch.setenv("H3_ATTEMPT_DIR", str(attempt))
    train_rows, val_rows = source_rows(64, "train"), source_rows(128, "test")
    train_file, val_file = tmp_path / "train.parquet", tmp_path / "val.parquet"
    for path, rows in ((train_file, train_rows), (val_file, val_rows)):
        parquet.write_table(arrow.Table.from_pylist(rows), path)
    config = OmegaConf.create(
        {
            "data": {
                "seed": 30,
                "shuffle": True,
                "train_batch_size": 32,
                "gen_batch_size": None,
                "dataloader_num_workers": 0,
                "train_files": str(train_file),
                "val_files": str(val_file),
                "validation_shuffle": False,
            },
            "actor_rollout_ref": {
                "rollout": {"seed": 42, "n": 8, "val_kwargs": {"seed": 42, "n": 1}},
                "model": {"lora_adapter_path": str(tmp_path / "converted")},
            },
            "trainer": {
                "use_v1": False,
                "total_training_steps": 1,
                "resume_mode": "disable",
                "resume_from_path": None,
                "default_local_dir": str(tmp_path / "hongshan"),
            },
        }
    )
    train_dataset, val_dataset = SourceDataset(train_rows), SourceDataset(val_rows)
    indices = list(reversed(range(32)))
    training_api = importlib.import_module("verl_omni.experiments.h3_training_manifest")
    evaluation_api = importlib.import_module("verl_omni.experiments.h3_evaluation_manifest")
    training = training_api.h3_training_schedule(config, train_dataset, indices, num_steps=1)
    evaluation = evaluation_api.create_h3_evaluation_manifest(val_file)
    monkeypatch.setattr(api.evidence, "TRAINING_MANIFEST", training)
    monkeypatch.setattr(api.evidence, "EVALUATION_MANIFEST", evaluation)
    return SimpleNamespace(
        config=config,
        run=run,
        attempt=attempt,
        train_rows=train_rows,
        val_rows=val_rows,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        indices=indices,
        training=training,
        evaluation=evaluation,
    )


@pytest.fixture
def candidates(api, case):
    """Supply 256 real source/seed rows with raw float64 components and independent extracted totals."""
    rows = [case.train_rows[index] for index in case.indices for session in range(8)]
    fields = raw_columns(rows)
    fields["experiment_sampling"] = np.array(
        [
            {"prompt_index": index // 8, "session_id": index % 8, "seed": seed}
            for index, seed in enumerate(case.training["steps"][0]["request_seeds"])
        ],
        dtype=object,
    )
    components = [
        {
            "reward/clap": np.float64(index / 1234),
            "reward/imagebind": np.float64(-index / 4567),
            "reward/combined": np.float64(index + 0.123456789012345),
            "reward/clap/source_sample_rate": np.int64(48000),
        }
        for index in range(256)
    ]
    fields["reward_extra_info"] = np.array(components, dtype=object)
    fields["uid"] = np.array([f"random-transport-{255 - index}" for index in range(256)], dtype=object)
    fields["index"] = np.full(256, -500)
    data = SimpleNamespace(non_tensor_batch=fields, batch={"rm_scores": torch.zeros(256, 1)})
    rewards = torch.tensor([[row["reward/combined"]] for row in components], dtype=torch.float32)
    api.extract_reward.return_value = rewards, {}
    return SimpleNamespace(data=data, components=components, rewards=rewards)


def test_prepare_data_uses_shared_manifests_and_resume_never_recreates_them(api, case, monkeypatch):
    """Delegate static construction, validate real datasets and preserve one immutable run manifest pair."""
    training = Mock(return_value=case.training)
    evaluation = Mock(return_value=case.evaluation)
    training_api = importlib.import_module("verl_omni.experiments.h3_training_manifest")
    monkeypatch.setattr(training_api, "create_h3_training_manifest", training)
    monkeypatch.setattr(api.evidence, "create_h3_evaluation_manifest", evaluation)
    api.evidence.prepare_data(case.config, case.train_dataset, case.val_dataset)
    training.assert_called_once_with(case.config, case.train_dataset, case.config.data.train_files, num_steps=1)
    assert json.loads((case.run / "manifests/training.json").read_text()) == case.training
    assert json.loads((case.run / "manifests/evaluation.json").read_text()) == case.evaluation
    case.config.trainer.resume_mode = "resume_path"
    api.evidence.prepare_data(case.config, case.train_dataset, case.val_dataset)
    assert training.call_count == evaluation.call_count == 1
    assert api.evidence.TRAINING_MANIFEST == case.training
    assert api.evidence.EVALUATION_MANIFEST == case.evaluation
    assert not case.attempt.exists()
    case.config.trainer.resume_mode = "disable"
    with pytest.raises(FileExistsError):
        api.evidence.prepare_data(case.config, case.train_dataset, case.val_dataset)


def test_input_batch_keeps_original_identity_and_rejects_duplicate_callback(api, case):
    """Record pre-repeat source rows in the attempt, independently of transport uid and index."""
    fields = raw_columns([case.train_rows[index] for index in case.indices])
    fields.update(uid=["random"] * 32, index=list(range(32)))
    result = api.evidence.record_input_batch(case.config, fields, 1)
    assert result == case.training["steps"][0]
    assert read_ledger(case.attempt / "training-inputs.jsonl") == [result]
    with pytest.raises(api.evidence.H3EvidenceContractError, match="Duplicate"):
        api.evidence.record_input_batch(case.config, fields, 1)


def test_sampling_callback_uses_actual_request_and_whole_prompt_kwargs(api):
    """Keep the supplied seed verbatim and preserve unrelated extra fields without deriving a replacement."""
    result = SimpleNamespace(extra_fields={"other": "retained"})
    api.evidence.record_sampling_request(result, {"seed": np.int64(777)}, 3, {"index": np.int64(12), "uid": "ignored"})
    assert result.extra_fields == {
        "other": "retained",
        "experiment_sampling": {"seed": 777, "prompt_index": 12, "session_id": 3},
    }
    for sampling, prompt in (({}, {"index": 0}), ({"seed": None}, {"index": 0}), ({"seed": 42}, {})):
        with pytest.raises(api.evidence.H3EvidenceContractError):
            api.evidence.record_sampling_request(result, sampling, 0, prompt)


def test_candidates_preserve_all_256_identities_actual_seeds_and_float_values(api, case, candidates):
    """Record the actual row order even when transport order is unrelated to the manifest grid."""
    fields = candidates.data.non_tensor_batch
    candidates.data.non_tensor_batch = {key: value[::-1] for key, value in fields.items()}
    api.extract_reward.return_value = candidates.rewards.flip(0), {}
    api.evidence.record_training_candidates(case.config, 1, candidates.data)
    api.extract_reward.assert_called_once_with(candidates.data)
    records = read_ledger(case.attempt / "training-candidates-1.jsonl")
    assert len(records) == 256
    assert [row["expanded_index"] for row in records] == list(reversed(range(256)))
    assert {(row["prompt_index"], row["session_id"]) for row in records} == {
        (p, s) for p in range(32) for s in range(8)
    }
    for index, record in enumerate(records):
        original = 255 - index
        prompt = case.training["steps"][0]["prompts"][original // 8]
        assert all(record[key] == value for key, value in prompt.items())
        assert record["sample_id"] == f"minimax_h3_t2va:train:{9000 + case.indices[original // 8]}"
        assert record["seed"] == case.training["steps"][0]["request_seeds"][original]
        assert record["components"] == candidates.components[original]
        assert record["total"] == candidates.rewards[original].item()
        assert "uid" not in record and "index" not in record
    assert records[0]["total"] != records[0]["components"]["reward/combined"]
    with pytest.raises(FileExistsError):
        api.evidence.record_training_candidates(case.config, 1, candidates.data)


def test_candidates_use_authoritative_colocated_columns_without_filling_missing_raw_values(api, case, candidates):
    """Use actual extract_reward component columns when colocated scoring replaced TQ rewards."""
    columns = {key: np.array([row[key] for row in candidates.components]) for key in candidates.components[0]}
    candidates.data.non_tensor_batch["reward_extra_info"] = np.array([None] * 256, dtype=object)
    api.extract_reward.return_value = candidates.rewards, columns
    api.evidence.record_training_candidates(case.config, 1, candidates.data)
    assert read_ledger(case.attempt / "training-candidates-1.jsonl")[7]["components"] == candidates.components[7]
    candidates.data.non_tensor_batch["reward_extra_info"] = np.array(candidates.components, dtype=object)
    del columns["reward/clap"]
    with pytest.raises(api.evidence.H3EvidenceContractError, match="reward/clap"):
        api.evidence.record_training_candidates(case.config, 1, candidates.data)


@pytest.mark.parametrize(
    "problem",
    [
        "missing_row",
        "duplicate",
        "wrong_seed",
        "missing_sampling",
        "none_sampling",
        "changed_source",
        "changed_prompt",
        "missing_component",
        "none_score",
        "nan",
        "inf",
        "nonfinite_total",
        "wrong_session",
        "short_scores",
    ],
)
def test_candidates_reject_incomplete_or_fabricated_evidence_before_writing(api, case, candidates, problem):
    """Invalid evidence cannot create a partial ledger that looks like a complete successful step."""
    fields = candidates.data.non_tensor_batch
    if problem == "missing_row":
        fields["raw_prompt"] = fields["raw_prompt"][:-1]
    elif problem == "duplicate":
        fields["experiment_sampling"][1] = fields["experiment_sampling"][0].copy()
    elif problem == "wrong_seed":
        fields["experiment_sampling"][0]["seed"] += 1
    elif problem == "missing_sampling":
        del fields["experiment_sampling"]
    elif problem == "none_sampling":
        fields["experiment_sampling"][0] = None
    elif problem == "changed_source":
        fields["extra_info"][0] = {"split": "train", "index": -1}
    elif problem == "changed_prompt":
        fields["raw_prompt"][0] = [{"role": "user", "content": "wrong"}]
    elif problem == "missing_component":
        del candidates.components[0]["reward/clap"]
    elif problem == "none_score":
        candidates.components[0]["reward/clap"] = None
    elif problem in ("nan", "inf"):
        candidates.components[0]["reward/imagebind"] = float(problem)
    elif problem == "nonfinite_total":
        candidates.rewards[0] = float("nan")
    elif problem == "wrong_session":
        fields["experiment_sampling"][0]["session_id"] = 8
    else:
        api.extract_reward.return_value = candidates.rewards[:-1], {}
    with pytest.raises(api.evidence.H3EvidenceContractError):
        api.evidence.record_training_candidates(case.config, 1, candidates.data)
    assert not (case.attempt / "training-candidates-1.jsonl").exists()


@pytest.fixture
def evaluation(case):
    """Supply complete fixed128 actual rows and request metadata with extra raw reward components."""
    fields = raw_columns(case.val_rows)
    fields["experiment_sampling"] = np.array(
        [{"seed": np.int64(42), "prompt_index": index, "session_id": 0} for index in range(128)], dtype=object
    )
    columns = {
        key: np.array([np.float64(index / 113) for index in range(128)])
        for key in ("reward/clap", "reward/imagebind", "reward/combined", "reward/other")
    }
    return SimpleNamespace(
        data=SimpleNamespace(non_tensor_batch=fields), scores=np.arange(128, dtype=np.float64), columns=columns
    )


def test_evaluation_keeps_all_128_original_ids_actual_seeds_and_components(api, case, evaluation):
    """Authenticate the full fixed set and serialize NumPy values without reducing or dropping components."""
    api.evidence.record_evaluation(case.config, 10, evaluation.data, evaluation.scores, evaluation.columns)
    records = read_ledger(case.attempt / "evaluation-10.jsonl")
    assert len(records) == 128
    for index, record in enumerate(records):
        assert record["sample_id"] == case.evaluation["samples"][index]["sample_id"]
        assert record["source_index"] == 9000 + index and record["source_row_index"] == index
        assert record["seed"] == 42 and record["session_id"] == 0
        assert record["prompt_sha256"] == case.evaluation["samples"][index]["prompt_sha256"]
        assert record["components"] == {key: values[index] for key, values in evaluation.columns.items()}
        assert record["total"] == evaluation.scores[index]
    with pytest.raises(FileExistsError):
        api.evidence.record_evaluation(case.config, 10, evaluation.data, evaluation.scores, evaluation.columns)


@pytest.mark.parametrize(
    "problem",
    [
        "short_rows",
        "short_scores",
        "short_component",
        "missing_component",
        "duplicate_id",
        "wrong_order",
        "wrong_seed",
        "missing_sampling",
        "wrong_session",
        "nan",
        "extra_nan",
    ],
)
def test_evaluation_rejects_partial_or_unmatched_fixed128(api, case, evaluation, problem):
    """Reject missing, repeated, reordered and nonfinite observed validation evidence before file creation."""
    fields = evaluation.data.non_tensor_batch
    if problem == "short_rows":
        fields["raw_prompt"] = fields["raw_prompt"][:-1]
    elif problem == "short_scores":
        evaluation.scores = evaluation.scores[:-1]
    elif problem == "short_component":
        evaluation.columns["reward/clap"] = evaluation.columns["reward/clap"][:-1]
    elif problem == "missing_component":
        del evaluation.columns["reward/clap"]
    elif problem == "duplicate_id":
        fields["extra_info"][1] = fields["extra_info"][0].copy()
    elif problem == "wrong_order":
        evaluation.data.non_tensor_batch = {key: value[::-1] for key, value in fields.items()}
    elif problem == "wrong_seed":
        fields["experiment_sampling"][0]["seed"] = 43
    elif problem == "missing_sampling":
        del fields["experiment_sampling"]
    elif problem == "wrong_session":
        fields["experiment_sampling"][0]["session_id"] = 1
    elif problem == "nan":
        evaluation.scores[0] = float("nan")
    else:
        evaluation.columns["reward/other"][0] = float("nan")
    with pytest.raises(ValueError):
        api.evidence.record_evaluation(case.config, 10, evaluation.data, evaluation.scores, evaluation.columns)
    assert not (case.attempt / "evaluation-10.jsonl").exists()


@pytest.fixture
def adapters(api, case, tmp_path, monkeypatch):
    """Use the real converter for step0; mock only the already-tested native exporter boundary."""
    source = tmp_path / "canonical"
    source.mkdir()
    state = {
        f"blocks.{index}.projection.lora_{part}.weight": torch.full(shape, float(index))
        for index in range(312)
        for part, shape in (("A", (64, 1)), ("B", (1, 64)))
    }
    config = {"r": 64, "lora_alpha": 128, "target_modules": ["projection"], "bias": "none"}
    (source / "adapter_config.json").write_text(json.dumps(config))
    save_file(state, source / "adapter_model.safetensors")
    converter = importlib.import_module("verl_omni.experiments.prepare_h3_adapter")
    converted = Path(case.config.actor_rollout_ref.model.lora_adapter_path)
    converter.prepare_h3_adapter(source, converted)
    (converted / "backbone-do-not-copy.bin").write_text("unrelated")

    def export_checkpoint(actor, output):
        """Supply known PEFT output bytes without recreating any FSDP loading or merging logic."""
        for name in ("adapter_config.json", "adapter_model.safetensors"):
            shutil.copyfile(source / name, Path(output) / name)
        return {"adapter_tensors": 624, "world_size": 1, "output_dir": str(output)}

    native = ModuleType("verl_omni.utils.fsdp_utils")
    native.export_fsdp_lora_adapter = Mock(side_effect=export_checkpoint)
    monkeypatch.setitem(sys.modules, native.__name__, native)
    return SimpleNamespace(source=source, converted=converted, native=native.export_fsdp_lora_adapter, state=state)


def test_fresh_snapshot_copies_converted_weights_config_receipt_and_actual_metadata(api, case, adapters):
    """Keep step0 on checkpoint storage, authenticate all tensors and skip unrelated backbone files."""
    receipt = api.evidence.snapshot_initial(case.config, 0)
    output = Path(case.config.trainer.default_local_dir) / "lora-snapshots/fresh-one/step0"
    assert set(path.name for path in output.iterdir()) == {
        "pytorch_lora_weights.safetensors",
        "adapter_config.json",
        "conversion_receipt.json",
    }
    assert receipt["source_format"] == "main-diffusers-converted"
    assert receipt["tensor_count"] == len(receipt["tensors"]) == 624
    assert (receipt["r"], receipt["lora_alpha"], receipt["scaling"]) == (64, 128, 2)
    metadata = json.loads(receipt["safetensors_metadata"]["lora_adapter_metadata"])
    assert metadata["transformer.r"] == 64 and metadata["transformer.lora_alpha"] == 128
    assert json.loads((case.attempt / "lora-snapshots/step0.json").read_text()) == receipt
    adapters.native.assert_not_called()


def test_saved_and_resumed_snapshots_delegate_to_native_export_and_keep_actual_step(api, case, adapters):
    """An eval20 resume snapshot comes from that actor checkpoint and remains explicitly native PEFT."""
    root = Path(case.config.trainer.default_local_dir)
    actor10, actor20 = root / "global_step_10/actor", root / "global_step_20/actor"
    actor10.mkdir(parents=True)
    actor20.mkdir(parents=True)
    (actor20.parent / "data.pt").write_bytes(b"saved sampler state")
    first = api.evidence.snapshot_checkpoint(case.config, 10, actor10)
    case.config.trainer.resume_mode = "resume_path"
    case.config.trainer.resume_from_path = str(actor20.parent)
    second = api.evidence.snapshot_initial(case.config, 20)
    assert [call.args[0] for call in adapters.native.call_args_list] == [actor10, actor20]
    assert [call.args[1] for call in adapters.native.call_args_list] == [
        root / "lora-snapshots/fresh-one/step10",
        root / "lora-snapshots/fresh-one/step20",
    ]
    for step, receipt in ((10, first), (20, second)):
        assert receipt["global_step"] == step and receipt["source_format"] == "native-peft"
        assert "prepare_h3_adapter" in receipt["main_compatible_reevaluation"]
        output = Path(receipt["output"])
        assert set(path.name for path in output.iterdir()) == {"adapter_config.json", "adapter_model.safetensors"}
        assert len(load_file(output / "adapter_model.safetensors")) == 624
    assert not (root / "lora-snapshots/fresh-one/step0").exists()


@pytest.mark.parametrize(
    "problem",
    ["step_mismatch", "missing_sampler", "missing_actor", "fresh_nonzero", "resume_zero", "auto", "run_storage"],
)
def test_snapshot_rejects_invalid_resume_or_storage_without_canonical_fallback(api, case, adapters, problem):
    """A missing actual checkpoint cannot become a seemingly valid canonical eval20 snapshot."""
    root = Path(case.config.trainer.default_local_dir)
    checkpoint = root / "global_step_20"
    (checkpoint / "actor").mkdir(parents=True)
    (checkpoint / "data.pt").write_bytes(b"sampler")
    case.config.trainer.resume_mode = "resume_path"
    case.config.trainer.resume_from_path = str(checkpoint)
    step = 20
    if problem == "step_mismatch":
        step = 30
    elif problem == "missing_sampler":
        (checkpoint / "data.pt").unlink()
    elif problem == "missing_actor":
        (checkpoint / "actor").rmdir()
    elif problem == "fresh_nonzero":
        case.config.trainer.resume_mode = "disable"
    elif problem == "resume_zero":
        step = 0
    elif problem == "auto":
        case.config.trainer.resume_mode = "auto"
    else:
        case.config.trainer.default_local_dir = str(case.run / "forbidden")
    with pytest.raises(api.evidence.H3EvidenceContractError):
        api.evidence.snapshot_initial(case.config, step)
    adapters.native.assert_not_called()


@pytest.mark.parametrize("problem", ["metadata_scale", "config_scale", "tensor_count", "receipt_hash"])
def test_fresh_snapshot_checks_actual_tensor_metadata_and_complete_conversion(api, case, adapters, problem):
    """Reject mislabeled converted adapters before copying weights or writing a successful receipt."""
    weights = adapters.converted / "pytorch_lora_weights.safetensors"
    if problem in ("metadata_scale", "tensor_count"):
        from safetensors import safe_open

        with safe_open(weights, framework="pt", device="cpu") as stream:
            metadata = stream.metadata()
        state = load_file(weights)
        if problem == "metadata_scale":
            config = json.loads(metadata["lora_adapter_metadata"])
            config["transformer.lora_alpha"] = 64
            metadata["lora_adapter_metadata"] = json.dumps(config)
        else:
            state.pop(next(iter(state)))
        save_file(state, weights, metadata=metadata)
    elif problem == "config_scale":
        config_path = adapters.converted / "adapter_config.json"
        config = json.loads(config_path.read_text())
        config["lora_alpha"] = 64
        config_path.write_text(json.dumps(config))
    else:
        path = adapters.converted / "conversion_receipt.json"
        receipt = json.loads(path.read_text())
        receipt["target_files_sha256"][weights.name] = "bad"
        path.write_text(json.dumps(receipt))
    with pytest.raises(api.evidence.H3EvidenceContractError):
        api.evidence.snapshot_initial(case.config, 0)
    assert not (case.attempt / "lora-snapshots/step0.json").exists()
