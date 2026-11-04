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
"""Run production worker methods with CPU tensors and stand-in RPC/data plumbing.

AST extraction removes only decorators and package imports, keeping the actual
train_mini_batch, train_batch and _postprocess_output bodies. This avoids optional
accelerator/Ray dependencies; collectives and TensorDict plumbing are test doubles.
"""

import ast
import importlib.util
import json
import logging
import os
from contextlib import nullcontext
from copy import deepcopy
from itertools import chain
from pathlib import Path
from types import SimpleNamespace
from typing import Optional
from unittest.mock import Mock

import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
TRACE_PATH = ROOT / "verl_omni/utils/optimizer_update_trace.py"
WORKER_PATH = ROOT / "verl_omni/workers/engine_workers.py"


def load_trace_module():
    """Import the standalone writer without triggering package registrations."""
    spec = importlib.util.spec_from_file_location("optimizer_update_trace", TRACE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_methods(path, class_name, names, namespace):
    """Compile unmodified method bodies, omitting RPC/profiler decorators only."""
    tree = ast.parse(path.read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == class_name)
    methods = [node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name in names]
    assert {node.name for node in methods} == set(names)
    for method in methods:
        method.decorator_list = []
    exec(compile(ast.Module(body=methods, type_ignores=[]), str(path), "exec"), namespace)


class CpuBatch(dict):
    """Minimal already-unwrapped TensorDict stand-in for worker control flow."""

    shape = (4,)

    def cpu(self):
        """Keep the CPU-only payload in place."""
        return self


def append_metrics(destination, source):
    """Match append_to_dict's metric-list accumulation."""
    for name, value in source.items():
        destination.setdefault(name, []).append(value)


@pytest.fixture
def harness(monkeypatch):
    """Wire real worker methods to deterministic CPU metrics and mock collectives."""
    monkeypatch.delenv("VERL_OPTIMIZER_UPDATE_TRACE_DIR", raising=False)
    trace = load_trace_module()
    rank = Mock(return_value=4)
    members = Mock(return_value=[7, 4])
    monkeypatch.setattr(torch.distributed, "get_rank", rank)
    monkeypatch.setattr(torch.distributed, "get_process_group_ranks", members)
    monkeypatch.setattr(torch.distributed, "get_world_size", Mock(return_value=2))

    def average_loss(loss, op, group):
        """Model an existing DP peer whose micro/timestep loss sum is 3x local."""
        assert op == torch.distributed.ReduceOp.AVG
        loss.mul_(2)

    reduce = Mock(side_effect=average_loss)
    monkeypatch.setattr(torch.distributed, "all_reduce", reduce)
    monkeypatch.setattr(torch.distributed, "all_gather_object", Mock(side_effect=AssertionError("new collective")))

    def make_iterator(data, mini_batch_size, epochs, seed, dataloader_kwargs):
        """Simulate slicing metadata so identity must be read before iteration."""
        for batch_idx in range(data.shape[0] // mini_batch_size * epochs):
            yield CpuBatch(values=torch.ones(mini_batch_size), global_steps=[999, 999], batch_idx=batch_idx)

    tu = SimpleNamespace(
        get=Mock(side_effect=lambda data, key, default=None: data.get(key, default)),
        pop=lambda data, key, default=None: data.pop(key, default),
        assign_non_tensor=lambda data, **kwargs: data.update(kwargs),
        make_iterator=Mock(side_effect=make_iterator),
        get_tensordict=lambda tensor_dict, non_tensor_dict: CpuBatch({**tensor_dict, **non_tensor_dict}),
    )
    namespace = {
        "os": os,
        "torch": torch,
        "TensorDict": CpuBatch,
        "NonTensorData": lambda value: value,
        "Optional": Optional,
        "tu": tu,
        "Timer": lambda **kwargs: nullcontext(SimpleNamespace(last=1.0)),
        "maybe_fix_3d_position_ids": lambda data: None,
        "collect_diffusion_flops_meta": lambda *args, **kwargs: None,
        "DiffusionFlopsCounter": type("UnusedFlopsCounter", (), {}),
        "allgather_dict_into_dict": lambda data, group: {key: [value, value] for key, value in data.items()},
        "Metric": type("UnusedMetric", (), {}),
        "append_to_dict": append_metrics,
        "chain": chain,
        "append_optimizer_update_trace": trace.append_optimizer_update_trace,
    }
    names = ("train_mini_batch", "train_batch", "_postprocess_output")
    load_methods(WORKER_PATH, "TrainingWorker", names, namespace)
    worker_type = type("CpuTrainingWorker", (), {name: namespace[name] for name in names})
    worker = worker_type()
    raw_outputs = []

    def engine_train_batch(data, loss_function):
        """Return two micro-batches times two timesteps for exactly one attempt."""
        index = data["batch_idx"]
        output = {
            "model_output": {},
            "loss": [float(index + 1)] * 4,
            "metrics": {"grad_norm": 0.25 + index, "extra": [10.0, 20.0]},
        }
        raw_outputs.append(deepcopy(output))
        return output

    worker.engine = SimpleNamespace(
        get_data_parallel_rank=Mock(return_value=1),
        get_data_parallel_size=Mock(return_value=2),
        get_data_parallel_group=Mock(return_value=object()),
        is_mp_src_rank_with_outputs=Mock(return_value=True),
        train_mode=lambda **kwargs: nullcontext(),
        train_batch=Mock(side_effect=engine_train_batch),
        lr_scheduler_step=Mock(return_value=0.001),
    )
    worker.profiler = SimpleNamespace(step=Mock())
    worker.loss_fn = object()
    worker.device_name = "cpu"
    worker.flops_counter = None
    worker.model_config = SimpleNamespace(get=lambda key, default: default)
    worker.engine_config = SimpleNamespace(
        forward_only=False,
        use_dynamic_bsz=False,
        max_token_len_per_gpu=None,
        micro_batch_size_per_gpu=1,
        use_fused_kernels=False,
    )
    return SimpleNamespace(
        worker=worker,
        trace=trace,
        rank=rank,
        members=members,
        reduce=reduce,
        tu=tu,
        namespace=namespace,
        raw_outputs=raw_outputs,
    )


def training_data(global_steps=23):
    """Build two mini-batches per epoch for two epochs."""
    return CpuBatch(mini_batch_size=4, epochs=2, global_steps=global_steps)


def read_records(directory):
    """Read complete JSONL records in rank-file order."""
    return [json.loads(line) for path in sorted(directory.glob("*.jsonl")) for line in path.read_text().splitlines()]


def test_attempt_identity_survives_epochs_and_final_aggregation(harness, monkeypatch, tmp_path):
    """Four updates retain their existing DP loss instead of becoming 32 samples."""
    monkeypatch.setenv("VERL_OPTIMIZER_UPDATE_TRACE_DIR", str(tmp_path))
    rng = torch.random.get_rng_state().clone()
    result = harness.worker.train_mini_batch(training_data())
    records = read_records(tmp_path)
    assert len(records) == harness.worker.engine.train_batch.call_count == 4
    assert [record["optimizer_update_id"] for record in records] == [f"23:{i}" for i in range(4)]
    assert [record["batch_idx"] for record in records] == list(range(4))
    assert all(record["global_steps"] == 23 for record in records)
    assert [record["metrics"]["loss"] for record in records] == [8, 16, 24, 32]
    assert result["metrics"]["loss"] == [8, 16, 24, 32]
    assert result["metrics"]["extra"] == [[10.0, 20.0, 10.0, 20.0]] * 4
    assert all("lr" not in record["metrics"] for record in records[:-1])
    assert records[-1]["metrics"]["lr"] == 0.001
    assert all(record["granularity"] == "optimizer_step_attempt" for record in records)
    assert all(record["optimizer_step_observation"] == "finite_grad_norm" for record in records)
    assert all(
        (record["rank"], record["dp_rank"], record["dp_size"], record["dp_source_rank"]) == (4, 1, 2, 4)
        for record in records
    )
    assert harness.reduce.call_count == 4
    assert harness.worker.engine.lr_scheduler_step.call_count == 1
    assert torch.equal(torch.random.get_rng_state(), rng)
    assert harness.tu.make_iterator.call_args.kwargs["seed"] == 43
    assert [record["metrics"]["grad_norm"] for record in records] == [0.25, 1.25, 2.25, 3.25]


@pytest.mark.parametrize("env_value", [None, ""])
def test_default_off_has_no_trace_io_or_identity_requirement(harness, monkeypatch, tmp_path, env_value):
    """Disabled tracing does not read identity/topology or invoke the writer."""
    if env_value is not None:
        monkeypatch.setenv("VERL_OPTIMIZER_UPDATE_TRACE_DIR", env_value)
    writer = Mock(side_effect=AssertionError("trace I/O while disabled"))
    harness.namespace["append_optimizer_update_trace"] = writer
    data = training_data()
    del data["global_steps"]
    result = harness.worker.train_mini_batch(data)
    assert result["metrics"]["loss"] == [8, 16, 24, 32]
    writer.assert_not_called()
    harness.rank.assert_not_called()
    harness.members.assert_not_called()
    assert not any(
        call.kwargs.get("key") == "global_steps" or call.args[1:] == ("global_steps",)
        for call in harness.tu.get.call_args_list
    )
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("identity", [None, [23, 23], True, "23"])
def test_invalid_identity_fails_before_training(harness, monkeypatch, tmp_path, identity):
    """An absent or sliced identity never silently becomes a successful update."""
    monkeypatch.setenv("VERL_OPTIMIZER_UPDATE_TRACE_DIR", str(tmp_path))
    with pytest.raises(ValueError, match="global_steps"):
        harness.worker.train_mini_batch(training_data(identity))
    harness.tu.make_iterator.assert_not_called()
    harness.worker.engine.train_batch.assert_not_called()
    assert read_records(tmp_path) == []


def test_rank_files_and_canonical_source_do_not_assume_global_zero(harness, monkeypatch, tmp_path):
    """Two DP replica files yield one baseline series by their recorded source."""
    monkeypatch.setenv("VERL_OPTIMIZER_UPDATE_TRACE_DIR", str(tmp_path))
    harness.worker.train_mini_batch(training_data())
    harness.rank.return_value = 7
    harness.worker.engine.get_data_parallel_rank.return_value = 0
    harness.worker.train_mini_batch(training_data())
    records = read_records(tmp_path)
    assert {path.name for path in tmp_path.iterdir()} == {
        "optimizer_updates.rank-4.jsonl",
        "optimizer_updates.rank-7.jsonl",
    }
    canonical = [record for record in records if record["rank"] == record["dp_source_rank"]]
    assert len(records) == 8
    assert len(canonical) == 4
    assert all(record["rank"] == 4 for record in canonical)
    assert len({record["optimizer_update_id"] for record in records}) == 4


def test_no_dp_group_uses_local_source(harness, monkeypatch, tmp_path):
    """A None DP group matches the existing local-only postprocess behavior."""
    monkeypatch.setenv("VERL_OPTIMIZER_UPDATE_TRACE_DIR", str(tmp_path))
    harness.worker.engine.get_data_parallel_group.return_value = None
    harness.worker.engine.get_data_parallel_size.return_value = 1
    base_train = harness.worker.engine.train_batch.side_effect

    def scalar_metrics_only(data, loss_function):
        """Exercise the local path without the DP-shaped auxiliary metric fixture."""
        output = base_train(data, loss_function)
        output["metrics"].pop("extra")
        return output

    harness.worker.engine.train_batch.side_effect = scalar_metrics_only
    harness.worker.train_mini_batch(training_data())
    records = read_records(tmp_path)
    assert [record["metrics"]["loss"] for record in records] == [4, 8]
    assert all(record["dp_source_rank"] == 4 for record in records)
    harness.reduce.assert_not_called()
    harness.members.assert_not_called()


def test_no_model_parallel_output_emits_no_record(harness, monkeypatch, tmp_path):
    """Non-output MP ranks do not fabricate metrics or attempt records."""
    monkeypatch.setenv("VERL_OPTIMIZER_UPDATE_TRACE_DIR", str(tmp_path))
    harness.worker.engine.is_mp_src_rank_with_outputs.return_value = False
    assert harness.worker.train_mini_batch(training_data()) is None
    assert read_records(tmp_path) == []


@pytest.mark.parametrize("gradient", [float("nan"), float("inf"), -float("inf"), 0.0])
def test_fsdp_nonfinite_skip_and_zero_change_are_not_success_flags(harness, monkeypatch, tmp_path, gradient):
    """Execute real FSDP skip logic; even a finite zero gradient proves no change."""
    monkeypatch.setenv("VERL_OPTIMIZER_UPDATE_TRACE_DIR", str(tmp_path))
    namespace = {
        "torch": torch,
        "FSDP": type("UnusedFSDP", (), {}),
        "FSDPModule": type("UnusedFSDPModule", (), {}),
        "DTensor": type("UnusedDTensor", (), {}),
        "logger": logging.getLogger(__name__),
    }
    load_methods(
        ROOT / "verl_omni/workers/engine/fsdp/diffusers_impl.py", "DiffusersFSDPEngine", ["optimizer_step"], namespace
    )
    model = torch.nn.Linear(1, 1, bias=False)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    optimizer.step = Mock(wraps=optimizer.step)
    optimizer.zero_grad = Mock(wraps=optimizer.zero_grad)
    before = model.weight.detach().clone()
    engine = SimpleNamespace(module=model, optimizer=optimizer, optimizer_config=SimpleNamespace(clip_grad=1.0))
    base_train = harness.worker.engine.train_batch.side_effect

    def train_with_optimizer(data, loss_function):
        """Insert the actual optimizer-step implementation before postprocessing."""
        model.weight.grad = torch.full_like(model.weight, gradient)
        norm = namespace["optimizer_step"](engine)
        output = base_train(data, loss_function)
        output["metrics"]["grad_norm"] = norm
        return output

    harness.worker.engine.train_batch.side_effect = train_with_optimizer
    harness.worker.train_mini_batch(training_data())
    records = read_records(tmp_path)
    assert torch.equal(before, model.weight)
    assert len(records) == 4
    if gradient == 0:
        assert optimizer.step.call_count == 4
        optimizer.zero_grad.assert_not_called()
        assert all(record["optimizer_step_observation"] == "finite_grad_norm" for record in records)
    else:
        optimizer.step.assert_not_called()
        assert optimizer.zero_grad.call_count == 4
        assert all(record["optimizer_step_observation"] == "nonfinite_grad_norm" for record in records)
        assert all(isinstance(record["metrics"]["grad_norm"], str) for record in records)
    assert all("actual_update" not in record and "optimizer_step_succeeded" not in record for record in records)


def test_training_exception_preserves_only_completed_attempts(harness, monkeypatch, tmp_path):
    """An exception before train_batch returns leaves no record for that attempt."""
    monkeypatch.setenv("VERL_OPTIMIZER_UPDATE_TRACE_DIR", str(tmp_path))
    base_train = harness.worker.engine.train_batch.side_effect

    def fail_second_update(data, loss_function):
        """Fail during the second attempt without generating a synthetic result."""
        if data["batch_idx"] == 1:
            raise RuntimeError("optimizer failed")
        return base_train(data, loss_function)

    harness.worker.engine.train_batch.side_effect = fail_second_update
    with pytest.raises(RuntimeError, match="optimizer failed"):
        harness.worker.train_mini_batch(training_data())
    assert [record["optimizer_update_id"] for record in read_records(tmp_path)] == ["23:0"]


def test_io_failure_is_visible(harness, monkeypatch, tmp_path):
    """A bad trace destination does not silently claim complete evidence."""
    destination = tmp_path / "file"
    destination.write_text("occupied")
    monkeypatch.setenv("VERL_OPTIMIZER_UPDATE_TRACE_DIR", str(destination))
    with pytest.raises(FileExistsError):
        harness.worker.train_mini_batch(training_data())
    assert destination.read_text() == "occupied"
