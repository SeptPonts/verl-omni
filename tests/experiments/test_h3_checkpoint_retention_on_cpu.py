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
"""Exercise native CPU state audits and retention only in isolated temporary run roots."""

import hashlib
import importlib
import json
import pickle
import random
import sys
import zipfile
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf
from safetensors.torch import save_file

ROOT = Path(__file__).parents[2]


@pytest.fixture
def api(monkeypatch):
    """Import actual experiment sources without triggering accelerator package registration."""
    before = set(sys.modules)
    for name in ("verl_omni", "verl_omni.experiments"):
        package = ModuleType(name)
        package.__path__ = [str(ROOT.joinpath(*name.split(".")))]
        monkeypatch.setitem(sys.modules, name, package)
    retention = importlib.import_module("verl_omni.experiments.h3_checkpoint_retention")
    evidence = importlib.import_module("verl_omni.experiments.h3_evidence")
    manifest = importlib.import_module("verl_omni.experiments.h3_evaluation_manifest")
    yield SimpleNamespace(retention=retention, evidence=evidence, manifest=manifest)
    for name in set(sys.modules) - before:
        if name.startswith("verl_omni."):
            sys.modules.pop(name, None)


@pytest.fixture
def case(tmp_path, monkeypatch, api):
    """Separate this new run's checkpoint storage from its PV2-equivalent evidence storage."""
    base = tmp_path.resolve()
    run = base / "pv2" / "new-run"
    attempt = run / "attempts" / "fresh"
    root = base / "hongshan" / "new-run"
    attempt.mkdir(parents=True)
    root.mkdir(parents=True)
    monkeypatch.setenv("H3_RUN_DIR", str(run))
    monkeypatch.setenv("H3_ATTEMPT_DIR", str(attempt))
    monkeypatch.setenv("H3_CHECKPOINT_DIR", str(root))
    config = OmegaConf.create(
        {
            "trainer": {"default_local_dir": str(root), "max_actor_ckpt_to_keep": None, "device": "cuda"},
            "data": {"train_batch_size": 32},
        }
    )
    samples = api.manifest.h3_evaluation_samples(
        [
            {
                "data_source": "h3",
                "prompt": [{"role": "user", "content": f"sound {index}"}],
                "extra_info": {"index": index + 1000, "split": "val"},
                "reward_model": {"ground_truth": "bell"},
            }
            for index in range(128)
        ]
    )
    (run / "manifests").mkdir()
    (run / "manifests" / "evaluation.json").write_text(json.dumps({"samples": samples}))
    return SimpleNamespace(root=root, run=run, attempt=attempt, config=config, samples=samples, api=api)


def save_checkpoint(case, step):
    """Save real CPU model/Adam/scheduler state in eight native shards and the historical loader format."""
    actor = case.root / f"global_step_{step}" / "actor"
    actor.mkdir(parents=True)
    parameter = torch.nn.Parameter(torch.ones(2, 2))
    optimizer = torch.optim.AdamW([parameter], lr=0.0003)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: 1.0)
    for update in range(step * 2):
        parameter.grad = torch.full_like(parameter, 0.25)
        optimizer.step()
        if update % 2:
            scheduler.step()
    rng = {
        "cpu": torch.get_rng_state(),
        "cuda": torch.get_rng_state(),
        "numpy": np.random.get_state(),
        "random": random.getstate(),
    }
    for rank in range(8):
        torch.save({"weight": parameter.detach()}, actor / f"model_world_size_8_rank_{rank}.pt")
        torch.save(optimizer.state_dict(), actor / f"optim_world_size_8_rank_{rank}.pt")
        torch.save(
            {"lr_scheduler": scheduler.state_dict(), "rng": rng}, actor / f"extra_state_world_size_8_rank_{rank}.pt"
        )
    torch.save(
        {
            "_snapshot": {
                "_snapshot_step": step,
                "_main_snapshot": {
                    "_sampler_iter_yielded": step,
                    "_sampler_iter_state": {"samples_yielded": step * 32},
                },
            },
            "_steps_since_snapshot": 0,
        },
        actor.parent / "data.pt",
    )
    (case.root / "latest_checkpointed_iteration.txt").write_text(str(step))
    output = case.run / "lora-snapshots" / "fresh" / f"step{step}"
    output.mkdir(parents=True)
    (output / "adapter_config.json").write_text(json.dumps({"r": 64, "lora_alpha": 128}))
    save_file(
        {f"layer{index}.lora_A.weight": torch.full((1, 1), float(index)) for index in range(624)},
        output / "adapter_model.safetensors",
    )
    snapshot = {
        "global_step": step,
        "source": str(actor),
        "output": str(output),
        **case.api.evidence.adapter_evidence(output, "adapter_model.safetensors"),
    }
    (case.attempt / "lora-snapshots").mkdir(exist_ok=True)
    (case.attempt / "lora-snapshots" / f"step{step}.json").write_text(json.dumps(snapshot))
    rows = [
        {
            **sample,
            "global_step": step,
            "prompt_index": index,
            "session_id": 0,
            "total": 0.75,
            "components": {"reward/clap": 0.25, "reward/imagebind": 0.5, "reward/combined": 0.75},
        }
        for index, sample in enumerate(case.samples)
    ]
    (case.attempt / f"evaluation-{step}.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    return actor


def audit(case, step):
    """Call the public synchronous post-evaluation boundary with the exact native actor path."""
    return case.api.retention.audit_and_retain_checkpoint(
        case.config, step, case.root / f"global_step_{step}" / "actor"
    )


@pytest.fixture
def previous(case):
    """Establish an actually audited owned recovery point before testing replacement failures."""
    actor = save_checkpoint(case, 10)
    audit(case, 10)
    return actor.parent


def test_complete_cpu_audit_is_persisted_before_only_owned_previous_is_deleted(case, previous, monkeypatch):
    """Observe the deletion boundary and independently verify all persisted hashes and eight rank results."""
    actor = save_checkpoint(case, 20)
    foreign = case.root.parent / "old-run" / "global_step_10"
    foreign.mkdir(parents=True)
    protected = foreign / "history"
    protected.write_text("preserve")
    remove_tree = case.api.retention.shutil.rmtree
    deleted = []

    def inspect_then_remove(path):
        """Assert complete durable evidence is readable before allowing temporary-fixture deletion."""
        assert path == previous and previous.is_dir() and actor.is_dir()
        receipt_path = case.run / "checkpoint-audits" / "step20.json"
        receipt = json.loads(receipt_path.read_text())
        marker = json.loads((receipt_path.parent / "verified.json").read_text())
        assert marker["audit_sha256"] == hashlib.sha256(receipt_path.read_bytes()).hexdigest()
        assert receipt["rank_states"] == [
            {"rank": rank, "optimizer_step": 40, "scheduler_step": 20, "rng_present": True} for rank in range(8)
        ]
        assert receipt["loader_step"] == 20 and receipt["validation_rows"] == 128
        assert receipt["distributed_restore"] == "NOT_RUN" and len(receipt["files"]) == 25
        for relative, record in receipt["files"].items():
            assert hashlib.sha256((actor.parent / relative).read_bytes()).hexdigest() == record["sha256"]
        for name, record in receipt["evidence_files"].items():
            assert hashlib.sha256(Path(name).read_bytes()).hexdigest() == record["sha256"]
        deleted.append(path)
        remove_tree(path)

    monkeypatch.setattr(case.api.retention.shutil, "rmtree", inspect_then_remove)
    result = audit(case, 20)
    assert deleted == [previous] and result["removed"] == [str(previous)]
    assert list(case.root.glob("global_step_*")) == [actor.parent]
    assert protected.read_text() == "preserve"


@pytest.mark.parametrize(
    "problem",
    [
        "data_bytes",
        "loader_step",
        "loader_cursor",
        "loader_delta",
        "model_bytes",
        "missing_rank",
        "optim_step",
        "optim_fractional_step",
        "optim_missing_moment",
        "optim_nan",
        "optim_shape",
        "optim_negative",
        "optim_empty",
        "scheduler",
        "rng_missing",
        "rng_bad_cpu",
    ],
)
def test_invalid_native_state_preserves_previous_checkpoint(case, previous, problem):
    """Reject load corruption and semantic state corruption, rather than merely checking file existence."""
    actor = save_checkpoint(case, 20)
    if problem in ("data_bytes", "model_bytes"):
        path = actor.parent / "data.pt" if problem == "data_bytes" else actor / "model_world_size_8_rank_7.pt"
        path.write_bytes(b"truncated native checkpoint")
    elif problem == "missing_rank":
        (actor / "extra_state_world_size_8_rank_7.pt").unlink()
    elif problem.startswith("loader_"):
        path = actor.parent / "data.pt"
        state = torch.load(path, weights_only=False)
        if problem == "loader_step":
            state["_snapshot"]["_snapshot_step"] = 19
        elif problem == "loader_cursor":
            state["_snapshot"]["_main_snapshot"]["_sampler_iter_state"]["samples_yielded"] = 639
        else:
            state["_steps_since_snapshot"] = 1
        torch.save(state, path)
    elif problem.startswith("optim_"):
        path = actor / "optim_world_size_8_rank_7.pt"
        state = torch.load(path, weights_only=False)
        value = next(iter(state["state"].values()))
        if problem in ("optim_step", "optim_fractional_step"):
            value["step"] = torch.tensor(39.0 if problem == "optim_step" else 40.5)
        elif problem == "optim_missing_moment":
            del value["exp_avg"]
        elif problem == "optim_nan":
            value["exp_avg"][0, 0] = float("nan")
        elif problem == "optim_shape":
            value["exp_avg_sq"] = torch.ones(3)
        elif problem == "optim_negative":
            value["exp_avg_sq"][0, 0] = -1
        else:
            state["state"] = {}
        torch.save(state, path)
    else:
        path = actor / "extra_state_world_size_8_rank_7.pt"
        state = torch.load(path, weights_only=False)
        if problem == "scheduler":
            state["lr_scheduler"]["last_epoch"] = 19
        elif problem == "rng_missing":
            del state["rng"]["cuda"]
        else:
            state["rng"]["cpu"] = torch.ones(1, dtype=torch.uint8)
        torch.save(state, path)
    with pytest.raises((ValueError, RuntimeError, pickle.UnpicklingError, zipfile.BadZipFile)):
        audit(case, 20)
    assert previous.is_dir()
    assert not (case.run / "checkpoint-audits" / "step20.json").exists()
    assert json.loads((case.run / "checkpoint-audits" / "verified.json").read_text())["global_step"] == 10


@pytest.mark.parametrize(
    "problem", ["short_eval", "eval_nan", "eval_identity", "eval_step", "lora_tensor", "lora_source"]
)
def test_eval_and_lora_must_both_pass_before_retention(case, previous, problem):
    """A valid native save alone never authorizes losing the last recovery point."""
    save_checkpoint(case, 20)
    if problem.startswith("eval_") or problem == "short_eval":
        path = case.attempt / "evaluation-20.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        if problem == "short_eval":
            rows.pop()
        elif problem == "eval_nan":
            rows[7]["components"]["reward/clap"] = float("nan")
        elif problem == "eval_identity":
            rows[7]["prompt_sha256"] = "foreign"
        else:
            rows[7]["global_step"] = 10
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    elif problem == "lora_tensor":
        path = case.run / "lora-snapshots" / "fresh" / "step20" / "adapter_model.safetensors"
        save_file({f"layer{index}.lora_A.weight": torch.zeros(1, 1) for index in range(624)}, path)
    else:
        path = case.attempt / "lora-snapshots" / "step20.json"
        snapshot = json.loads(path.read_text())
        snapshot["source"] = str(previous / "actor")
        path.write_text(json.dumps(snapshot))
    with pytest.raises(ValueError):
        audit(case, 20)
    assert previous.is_dir()


@pytest.mark.parametrize(
    "problem",
    [
        "foreign_actor",
        "mismatched_root",
        "misnamed",
        "future",
        "symlink_previous",
        "symlink_current",
        "symlink_nested",
        "symlink_ancestor",
        "foreign_marker",
        "changed_receipt",
        "replaced_previous",
    ],
)
def test_foreign_paths_and_symlinks_are_rejected_before_deletion(case, previous, problem, monkeypatch):
    """Do not traverse aliases, adopt extra checkpoints, or consume another run's recorded ownership."""
    actor = save_checkpoint(case, 20)
    foreign = case.root.parent / "protected-history"
    foreign.mkdir()
    (foreign / "keep").write_text("old evidence")
    if problem == "foreign_actor":
        actor = foreign
    elif problem == "mismatched_root":
        case.config.trainer.default_local_dir = str(foreign)
    elif problem in ("misnamed", "future"):
        (case.root / ("global_step_010" if problem == "misnamed" else "global_step_30")).mkdir()
    elif problem == "symlink_previous":
        previous.rename(case.root / "preserved-original")
        previous.symlink_to(foreign, target_is_directory=True)
    elif problem == "symlink_current":
        actor.rename(actor.with_name("preserved-actor"))
        actor.symlink_to(foreign, target_is_directory=True)
    elif problem == "symlink_nested":
        (previous / "foreign").symlink_to(foreign, target_is_directory=True)
    elif problem == "symlink_ancestor":
        alias = case.root.parent / "alias"
        alias.symlink_to(case.root, target_is_directory=True)
        monkeypatch.setenv("H3_CHECKPOINT_DIR", str(alias))
        case.config.trainer.default_local_dir = str(alias)
        actor = alias / "global_step_20" / "actor"
    elif problem == "foreign_marker":
        path = case.run / "checkpoint-audits" / "verified.json"
        marker = json.loads(path.read_text())
        marker["owner"]["checkpoint_root"] = str(foreign)
        path.write_text(json.dumps(marker))
    elif problem == "changed_receipt":
        (case.run / "checkpoint-audits" / "step10.json").write_text("{}")
    else:
        previous.rename(case.root / "preserved-original")
        previous.mkdir()
        (previous / "foreign").write_text("unowned checkpoint data")
    with pytest.raises(ValueError):
        case.api.retention.audit_and_retain_checkpoint(case.config, 20, actor)
    assert previous.exists() and (foreign / "keep").read_text() == "old evidence"


def test_first_call_cannot_adopt_an_existing_old_checkpoint(case):
    """A fresh ownership marker cannot retroactively authorize deletion of old run data."""
    old = case.root / "global_step_1"
    old.mkdir()
    save_checkpoint(case, 10)
    with pytest.raises(ValueError, match="unowned"):
        audit(case, 10)
    assert old.is_dir()


@pytest.mark.parametrize("failure", ["receipt", "marker", "changed_data"])
def test_audit_or_publication_failure_never_prunes_previous(case, previous, monkeypatch, failure):
    """Inject errors after successful native checks and observe that the recovery point survives."""
    actor = save_checkpoint(case, 20)
    if failure == "changed_data":
        native_audit = case.api.retention.audit_native_checkpoint

        def audit_then_change(*args):
            """Simulate a writer replacing data after the CPU reader has checked it."""
            result = native_audit(*args)
            (actor.parent / "data.pt").write_bytes(b"changed after load")
            return result

        monkeypatch.setattr(case.api.retention, "audit_native_checkpoint", audit_then_change)
    else:
        persist = case.api.retention.persist_json

        def fail_publication(path, value):
            """Fail one durable write without replacing any filesystem deletion operation."""
            if path.name == ("step20.json" if failure == "receipt" else "verified.next.json"):
                raise OSError("injected evidence storage failure")
            persist(path, value)

        monkeypatch.setattr(case.api.retention, "persist_json", fail_publication)
    with pytest.raises((ValueError, OSError)):
        audit(case, 20)
    assert previous.is_dir()


def test_native_keep_one_is_refused(case, previous):
    """Do not certify a configuration that can erase the old save before data.pt is written."""
    save_checkpoint(case, 20)
    case.config.trainer.max_actor_ckpt_to_keep = 1
    with pytest.raises(ValueError, match="premature"):
        audit(case, 20)
    assert previous.is_dir()


def test_real_dtensor_adam_shards_load_locally_without_a_process_group(case):
    """Round-trip real DTensor optimizer payloads while keeping all tensor inspection on CPU."""
    from torch.distributed.device_mesh import DeviceMesh
    from torch.distributed.tensor import DTensor, Shard
    from torch.distributed.tensor._dtensor_spec import DTensorSpec, TensorMeta

    actor = save_checkpoint(case, 10)
    # DeviceMesh needs the writer's rank to construct metadata, but no process groups
    # or collectives participate in the CPU checkpoint read under test.
    with patch("torch.distributed.device_mesh.get_rank", return_value=0):
        mesh = DeviceMesh("cpu", list(range(8)), _init_backend=False)
    spec = DTensorSpec(mesh, (Shard(0),), TensorMeta(torch.Size((16, 2)), (2, 1), torch.float32))
    torch.save({"weight": DTensor(torch.ones(2, 2), spec, requires_grad=False)}, actor / "model_world_size_8_rank_0.pt")
    optim_path = actor / "optim_world_size_8_rank_0.pt"
    optim = torch.load(optim_path, weights_only=False)
    for value in optim["state"].values():
        for name in ("exp_avg", "exp_avg_sq"):
            value[name] = DTensor(value[name], spec, requires_grad=False)
    torch.save(optim, optim_path)
    assert audit(case, 10)["status"] == "VERIFIED_LATEST_ONE_RETAINED"
    assert not torch.distributed.is_initialized()


def test_model_archives_are_hashed_without_deserializing_full_model_shards(case, monkeypatch):
    """Preserve the historical archive-only model audit instead of adding another full-model read."""
    save_checkpoint(case, 10)
    load = torch.load

    def load_training_state(path, **kwargs):
        """Reject model deserialization at the actual torch.load boundary."""
        assert not Path(path).name.startswith("model_world_size_")
        return load(path, **kwargs)

    monkeypatch.setattr(torch, "load", load_training_state)
    result = audit(case, 10)
    receipt = json.loads(Path(result["audit_receipt"]).read_text())
    assert receipt["model_validation"] == "ARCHIVE_AND_SHA256_ONLY"
