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
"""Test H3 capture orchestration, not device traces, distributed progress, or TPA native support."""

import ast
import builtins
import gc
import hashlib
import importlib.util
import json
import os
import sys
import weakref
from contextlib import contextmanager, nullcontext
from pathlib import Path
from types import MethodType, ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest

ROOT = Path(__file__).parents[2]


def load_method(relative_path, class_name, method_name, namespace):
    """Execute a production method with native imports and unrelated worker methods excluded."""
    source = ROOT / relative_path
    tree = ast.parse(source.read_text())
    owner = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == class_name)
    method = next(node for node in owner.body if isinstance(node, ast.FunctionDef) and node.name == method_name)
    method.decorator_list = []
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(source), "exec"), namespace)
    return namespace[method_name]


class Config(SimpleNamespace):
    """Expose the attribute/get interface used by the actual configuration consumers."""

    def get(self, name, default=None):
        """Read one optional configuration field."""
        return vars(self).get(name, default)


class EqualGroup:
    """Make distinct group identities compare equal, exposing equality-based deduplication."""

    def __eq__(self, other):
        """Compare group values without treating their identities as interchangeable."""
        return isinstance(other, EqualGroup)

    def __hash__(self):
        """Give equal group values the same hash."""
        return 1


class FakeProfiler:
    """Represent the TPA lifecycle while producing only a small fake trace on CPU."""

    def __init__(self, config, state):
        """Keep per-instance state separate from package class methods."""
        self.config = config
        self.config.dir_name = config.profile_path
        self.state = state
        self.rank = 1 if state.distributed else 0
        self.has_exported = False
        self.file_name = "trace.json"

    def _export_extra_info(self):
        """Implement the existing TPA hook, whose unrestricted default must never run."""
        raise AssertionError("The unrestricted TPA exporter ran")

    def start_trace(self):
        """Record native trace startup before the orchestration barrier."""
        self.state.events.append("kineto-start")
        self._export_extra_info()

    def start(self):
        """Exercise the actual instance-bound start_trace override."""
        self.state.events.append("profiler-start")
        self.record_shapes_at_start = self.record_shapes
        self.start_trace()

    def step(self):
        """Record the sole profiler step after the complete captured phase."""
        self.state.events.append("profiler-step")

    def stop(self):
        """Produce a trace file so the real receipt writer computes its digest."""
        self.state.events.append("profiler-stop")
        (Path(self.config.dir_name) / self.file_name).write_bytes(b'{"traceEvents":[]}\n')


@pytest.fixture
def api(monkeypatch):
    """Load the real module with device imports isolated and default TPA imports forbidden."""
    permissions = {"tpa": False}
    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        """Fail an eager or disabled-path TPA import before native dependencies can load."""
        if name.startswith("torch_profiler_analysis") and not permissions["tpa"]:
            raise AssertionError("Disabled H3 profiling imported TPA")
        return original_import(name, *args, **kwargs)

    @contextmanager
    def record_function(name):
        """Provide the record_function boundary without starting a device collector."""
        yield

    def tensor_get(data, key, default=None):
        """Read the marked batch fields used by the actual actor context."""
        return data.get(key, default)

    def tensor_assign(data, **values):
        """Preserve actual actor batch assignments for the extracted worker method."""
        data.update(values)

    for name in ("verl_omni", "verl_omni.experiments", "verl", "verl.utils"):
        module = ModuleType(name)
        module.__path__ = [str(ROOT.joinpath(*name.split(".")))]
        monkeypatch.setitem(sys.modules, name, module)
    torch = ModuleType("torch")
    torch.__version__ = "cpu-contract"
    torch.profiler = SimpleNamespace(record_function=record_function)
    dist = ModuleType("torch.distributed")
    dist.is_initialized = Mock(return_value=False)
    torch.distributed = dist
    tu = ModuleType("verl.utils.tensordict_utils")
    tu.get = tensor_get
    tu.assign_non_tensor = tensor_assign
    ray = ModuleType("ray")
    ray.get = Mock()
    for module in (torch, dist, tu, ray):
        monkeypatch.setitem(sys.modules, module.__name__, module)
    monkeypatch.setattr(builtins, "__import__", guarded_import)
    path = ROOT / "verl_omni/experiments/h3_profiling.py"
    spec = importlib.util.spec_from_file_location("verl_omni.experiments.h3_profiling", path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    module.import_permissions = permissions
    return module


@pytest.fixture
def capture(api, monkeypatch):
    """Supply explicit TPA and distributed boundaries for opt-in capture tests."""
    api.import_permissions["tpa"] = True
    state = SimpleNamespace(events=[], profiles=[], registered=[], exports=[], distributed=False)
    tpa = ModuleType("torch_profiler_analysis")

    def initialize_profiler(config):
        """Create a distinct profiler instance for each actual window."""
        profiler = FakeProfiler(config, state)
        state.profiles.append(profiler)
        return profiler

    def register_group(group, group_type, is_custom):
        """Retain exact object identities rather than comparing fake group values."""
        state.registered.append((group, group_type, is_custom))

    def group_info(strict_check):
        """Return communication metadata without host environment or argv."""
        assert strict_check is True
        state.exports.append("communication")
        return {"communication_groups": ["global", "data_parallel"]}

    def export_configs(config, mode, rank):
        """Write the allowed config output of the real restricted exporter."""
        assert mode == "online"
        state.exports.append("config")
        path = Path(config.dir_name) / "extra_info" / f"config_{rank}.json"
        path.write_text(json.dumps({"step_start": config.step_start, "step_end": config.step_end}))

    tpa.ProfileConfig = Config
    tpa.initialize_profiler = initialize_profiler
    tpa.register_comm_group_type = register_group
    analysis = ModuleType("torch_profiler_analysis.holistic_analysis.analysis_utils")
    analysis.get_group_info_by_parallel_type = group_info
    configs = ModuleType("torch_profiler_analysis.holistic_analysis.config_utils")
    configs.export_configs = export_configs
    for module in (tpa, analysis, configs):
        monkeypatch.setitem(sys.modules, module.__name__, module)
    monkeypatch.setattr(api, "version", Mock(return_value="0.14.0"))
    api.dist.group = SimpleNamespace(WORLD=EqualGroup())
    api.dist.get_rank = Mock(return_value=1)
    api.dist.get_world_size = Mock(return_value=8)
    api.dist.new_group = Mock(return_value=object())
    api.dist.destroy_process_group = Mock()

    def barrier(group):
        """Expose alignment order relative to Kineto and the measured phase."""
        assert group is api.dist.new_group.return_value
        state.events.append("barrier")

    api.dist.barrier = Mock(side_effect=barrier)
    return state


def profiling_config(directory):
    """Build the supported online H3 V0 configuration for contract validation."""
    return Config(
        trainer=Config(
            h3_comparison_profile_dir=directory,
            h3_comparison_profile_step=3,
            h3_comparison_profile_windows=[
                Config(role="actor", step=3, record_shapes=True),
                Config(role="rollout", step=3, record_shapes=True, replica=2, request_start=3, request_end=4),
                Config(role="reward", step=3, record_shapes=True),
            ],
            total_training_steps=6,
            use_v1=False,
        ),
        actor_rollout_ref=Config(actor=Config(strategy="fsdp2"), rollout=Config(step_execution=False)),
        global_profiler=Config(tool=None, steps=None),
        reward=Config(custom_reward_function=Config(path="clap-imagebind.py"), reward_model=Config(enable=False)),
        algorithm=Config(adv_estimator="flow_grpo", sample_source="online"),
    )


@pytest.fixture
def rollout_worker(api, monkeypatch):
    """Bind the actual extension RPCs to an isolated backend full-request entrypoint."""
    parallel = ModuleType("vllm.distributed.parallel_state")
    parallel.get_tp_group = Mock(return_value=SimpleNamespace(device_group=object()))
    monkeypatch.setitem(sys.modules, parallel.__name__, parallel)
    worker = SimpleNamespace(execute_model=Mock())
    source = "verl_omni/workers/rollout/vllm_rollout/utils.py"
    for name in ("start_h3_comparison_profile", "stop_h3_comparison_profile"):
        method = load_method(source, "vLLMOmniColocateWorkerExtension", name, {})
        setattr(worker, name, MethodType(method, worker))
    return worker


def test_disabled_profile_has_no_import_output_or_worker_access(api, tmp_path):
    """Default paths neither import TPA nor create output or touch unavailable workers."""
    directory = tmp_path / "disabled"
    config = profiling_config(None)
    api.validate_h3_profiling(config)
    with api.h3_actor_profile({}, object()):
        pass
    trainer = SimpleNamespace(config=config, global_steps=3)
    with api.h3_generation_profile(trainer):
        pass
    assert not directory.exists()
    assert list(tmp_path.iterdir()) == []
    sys.modules["ray"].get.assert_not_called()


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("failed", [False, True])
def test_actor_worker_capture_is_between_onload_and_offload(api, capture, tmp_path, enabled, failed):
    """Run both actual worker entrypoints and require profiler destruction before train-mode exit."""
    tu = sys.modules["verl.utils.tensordict_utils"]
    output = SimpleNamespace(cpu=Mock(return_value="cpu-result"))
    references = []

    class TrainingBatch(dict):
        """Supply only the tensor-dict batch dimension consumed by the actual mini-batch loop."""

        shape = (32,)

    def tensor_pop(data, key, default=None):
        """Remove production mini-batch metadata using the TensorDict utility signature."""
        return data.pop(key, default)

    tu.pop = tensor_pop
    tu.make_iterator = Mock(return_value=[{}, {}])
    tu.get_tensordict = Mock(return_value=SimpleNamespace(cpu=Mock(return_value=output)))

    @contextmanager
    def train_mode(disable_auto_offload):
        """Model transfers as strict boundaries around production train_mini_batch."""
        assert disable_auto_offload is False
        capture.events.append("onload")
        assert capture.profiles == []
        try:
            yield
        finally:
            assert all(reference() is None for reference in references)
            capture.events.append("offload")

    def train_batch(data):
        """Expose each optimizer update and retain only weak ownership of the active profiler."""
        assert data["disable_auto_offload"] is True
        if enabled and not references:
            references.append(weakref.ref(capture.profiles.pop()))
        if enabled:
            assert references[0]().record_shapes_at_start is True
        capture.events.append("mini-update")
        if failed:
            raise RuntimeError("mini-update failed")
        return {"metrics": {}}

    group = object()
    engine = SimpleNamespace(
        train_mode=train_mode,
        get_data_parallel_size=Mock(return_value=1),
        get_data_parallel_rank=Mock(return_value=0),
        get_data_parallel_group=Mock(return_value=group),
        device_mesh=SimpleNamespace(get_all_groups=Mock(return_value=[group])),
        is_mp_src_rank_with_outputs=Mock(return_value=True),
    )
    namespace = {
        "TensorDict": dict,
        "tu": tu,
        "maybe_fix_3d_position_ids": Mock(),
        "Timer": Mock(return_value=nullcontext()),
        "NonTensorData": Mock(),
        "append_to_dict": Mock(),
    }
    source = "verl_omni/workers/engine_workers.py"
    train_mini_batch = load_method(source, "TrainingWorker", "train_mini_batch", namespace)
    update_actor = load_method(source, "ActorRolloutRefWorker", "update_actor", namespace)
    actor = SimpleNamespace(engine=engine, train_batch=train_batch)
    actor.train_mini_batch = MethodType(train_mini_batch, actor)
    worker = SimpleNamespace(config=Config(actor=Config()), actor=actor)
    batch = TrainingBatch(mini_batch_size=16)
    if enabled:
        batch.update(h3_profile_directory=str(tmp_path / "actor"), h3_profile_rl_step=3, h3_profile_record_shapes=True)
    if failed:
        with pytest.raises(RuntimeError, match="mini-update failed"):
            update_actor(worker, batch)
    else:
        assert update_actor(worker, batch) == "cpu-result"
    expected = ["onload"]
    if enabled:
        expected += ["profiler-start", "kineto-start"]
    expected += ["mini-update"] * (1 if failed else 2)
    if enabled:
        expected += ["profiler-step", "profiler-stop"]
    assert capture.events == expected + ["offload"]
    if not enabled:
        assert list(tmp_path.iterdir()) == []
        return
    receipt = json.loads((tmp_path / "actor/h3_metadata/h3_profile_rank0.json").read_text())
    assert receipt["completed"] is not failed
    assert receipt["role"] == "actor-update-compute"
    assert (receipt["rl_step"], receipt["local_profiler_step"]) == (3, 0)
    assert receipt["trace_sha256"] == hashlib.sha256(b'{"traceEvents":[]}\n').hexdigest()


def test_distributed_identity_registration_and_alignment(api, capture, tmp_path):
    """Register repeated objects once while retaining distinct groups, then align before measured work."""
    capture.distributed = True
    api.dist.is_initialized.return_value = True
    world, dp = api.dist.group.WORLD, EqualGroup()
    profiler = api.start_h3_profile(
        tmp_path / "distributed", "actor-update", 3, [(world, "fsdp"), (dp, "dp"), (dp, "fsdp")]
    )
    assert [id(group) for group, kind, custom in capture.registered] == [id(world), id(dp)]
    assert [kind for group, kind, custom in capture.registered] == ["global", "dp"]
    assert all(custom for group, kind, custom in capture.registered)
    assert capture.events == ["profiler-start", "kineto-start", "barrier"]
    capture.events.append("measured-work")
    api.stop_h3_profile(profiler)
    assert capture.events == [
        "profiler-start",
        "kineto-start",
        "barrier",
        "measured-work",
        "profiler-step",
        "profiler-stop",
    ]
    api.dist.barrier.assert_called_once()
    api.dist.destroy_process_group.assert_called_once_with(profiler.h3_control_group)
    assert (profiler.config.step_start, profiler.config.step_end) == (0, 1)
    assert profiler.config.distributed_profile is True


def test_metadata_is_restricted_and_override_is_instance_local(api, capture, monkeypatch, tmp_path):
    """Allow communication/config exports without emitting sentinel credentials or changing package methods."""
    capture.distributed = True
    api.dist.is_initialized.return_value = True
    sentinel = "FAKE_WANDB_SECRET_FOR_CPU_CONTRACT"
    monkeypatch.setenv("WANDB_API_KEY", sentinel)
    monkeypatch.setattr(sys, "argv", ["cpu-test", f"--token={sentinel}"])
    original_exporter, original_start = FakeProfiler._export_extra_info, FakeProfiler.start_trace
    profiler = api.start_h3_profile(tmp_path / "metadata", "rollout-generation", 3)
    api.export_h3_profile_metadata(profiler)
    api.stop_h3_profile(profiler)
    assert capture.exports == ["communication", "config"]
    assert FakeProfiler._export_extra_info is original_exporter
    assert FakeProfiler.start_trace is original_start
    unrelated = FakeProfiler(Config(profile_path=str(tmp_path / "unrelated")), capture)
    assert unrelated._export_extra_info.__func__ is original_exporter
    assert unrelated.start_trace.__func__ is original_start
    extras = tmp_path / "metadata/extra_info"
    assert {path.name for path in extras.iterdir()} == {"config_1.json", "meta_data_1.json"}
    assert json.loads((extras / "meta_data_1.json").read_text())["communication_groups"] == ["global", "data_parallel"]
    for path in (tmp_path / "metadata").rglob("*"):
        if path.is_file():
            text = path.read_text()
            assert sentinel not in text
            assert "WANDB_API_KEY" not in text
            assert "sys.argv" not in text


@pytest.mark.parametrize("failed", [False, True])
def test_actor_phase_failure_marks_receipt(api, capture, tmp_path, failed):
    """Stop a selected actor window on failure and never call an incomplete capture completed."""
    engine = SimpleNamespace(
        get_data_parallel_group=Mock(), device_mesh=SimpleNamespace(get_all_groups=Mock(return_value=[]))
    )
    batch = {"h3_profile_directory": str(tmp_path / "actor"), "h3_profile_rl_step": 3, "h3_profile_record_shapes": True}
    if failed:
        with pytest.raises(RuntimeError, match="mini-update failed"), api.h3_actor_profile(batch, engine):
            raise RuntimeError("mini-update failed")
    else:
        with api.h3_actor_profile(batch, engine):
            pass
    receipt = json.loads((tmp_path / "actor/h3_metadata/h3_profile_rank0.json").read_text())
    assert receipt["completed"] is not failed
    assert capture.events.count("profiler-step") == capture.events.count("profiler-stop") == 1


@pytest.mark.parametrize("failed", [False, True])
def test_generation_fanout_selects_one_replica_and_optional_rewards(api, tmp_path, failed):
    """Contact only the selected replica, keeping its real index and optional reward lifecycle."""
    servers = [
        SimpleNamespace(collective_rpc=SimpleNamespace(remote=Mock(return_value=f"server-{i}"))) for i in range(4)
    ]
    rewards = [
        SimpleNamespace(
            start_h3_comparison_profile=SimpleNamespace(remote=Mock(return_value=f"reward-start-{i}")),
            stop_h3_comparison_profile=SimpleNamespace(remote=Mock(return_value=f"reward-stop-{i}")),
        )
        for i in range(2)
    ]
    trainer = SimpleNamespace(
        config=profiling_config(str(tmp_path / "generation")),
        global_steps=3,
        llm_server_manager=SimpleNamespace(server_handles=servers),
        reward_loop_manager=SimpleNamespace(reward_loop_workers=rewards),
    )

    def phase():
        """Confirm all starts completed before entering generation, then optionally fail it."""
        assert sys.modules["ray"].get.call_count == 1
        assert len(sys.modules["ray"].get.call_args.args[0]) == 3
        assert [server.collective_rpc.remote.call_count for server in servers] == [0, 0, 1, 0]
        assert all(worker.start_h3_comparison_profile.remote.call_count == 1 for worker in rewards)
        if failed:
            raise RuntimeError("generation failed")

    if failed:
        with pytest.raises(RuntimeError, match="generation failed"), api.h3_generation_profile(trainer):
            phase()
    else:
        with api.h3_generation_profile(trainer):
            phase()
    root = tmp_path / "generation/rl_step_0003"
    for index, server in enumerate(servers):
        if index != 2:
            server.collective_rpc.remote.assert_not_called()
            continue
        assert server.collective_rpc.remote.call_args_list[0].args == ("start_h3_comparison_profile",)
        assert server.collective_rpc.remote.call_args_list[0].kwargs == {
            "kwargs": {
                "directory": str(root / f"rollout_replica_{index:03d}"),
                "rl_step": 3,
                "record_shapes": True,
                "request_start": 3,
                "request_end": 4,
            }
        }
        assert server.collective_rpc.remote.call_args_list[1].kwargs == {"kwargs": {"completed": not failed}}
        assert server.collective_rpc.remote.call_args_list[1].args == ("stop_h3_comparison_profile",)
    for index, worker in enumerate(rewards):
        worker.start_h3_comparison_profile.remote.assert_called_once_with(
            str(root / f"reward_worker_{index:03d}"), 3, True
        )
        worker.stop_h3_comparison_profile.remote.assert_called_once_with(not failed)
    assert sys.modules["ray"].get.call_count == 2
    assert len(sys.modules["ray"].get.call_args.args[0]) == 3


def test_non_target_generation_step_does_not_contact_workers(api, tmp_path):
    """An enabled capture directory does not collect any unselected RL step."""
    trainer = SimpleNamespace(config=profiling_config(str(tmp_path / "profiles")), global_steps=2)
    with api.h3_generation_profile(trainer):
        pass
    sys.modules["ray"].get.assert_not_called()
    assert list(tmp_path.iterdir()) == []


def test_rollout_request_window_releases_before_next_request_and_resets(api, capture, rollout_worker, tmp_path):
    """Capture only requests3/4 in successive RL steps and restore the exact backend callable."""
    backend = rollout_worker.execute_model
    references = []
    request_index = 0
    output = object()
    config = object()
    prefetch = object()

    def execute_model(req, od_config, *, kv_prefetch_job):
        """Observe profiler ownership from inside the unmodified backend request call."""
        nonlocal request_index
        request_index += 1
        assert od_config is config and kv_prefetch_job is prefetch
        assert req.request_id == f"request-{request_index}"
        if request_index == 3:
            references.append(weakref.ref(capture.profiles.pop()))
        if request_index in (3, 4):
            assert references[-1]() is not None
            assert references[-1]().record_shapes_at_start is True
        else:
            assert all(reference() is None for reference in references)
        capture.events.append(f"request-{request_index}")
        return output

    backend.side_effect = execute_model
    for step in (2, 3):
        request_index = 0
        capture.events.clear()
        directory = tmp_path / f"rl_step_{step:04d}/rollout_replica_000"
        rollout_worker.start_h3_comparison_profile(str(directory), step, True, 3, 4)
        assert capture.events == []
        for index in range(1, 7):
            request = SimpleNamespace(request_id=f"request-{index}")
            assert rollout_worker.execute_model(request, config, kv_prefetch_job=prefetch) is output
        assert all(reference() is None for reference in references)
        rollout_worker.stop_h3_comparison_profile(completed=True)
        assert rollout_worker.execute_model is backend
        assert rollout_worker.h3_comparison_profile is None
        assert capture.events == [
            "request-1",
            "request-2",
            "profiler-start",
            "kineto-start",
            "request-3",
            "request-4",
            "profiler-step",
            "profiler-stop",
            "request-5",
            "request-6",
        ]
        receipt = json.loads((directory / "h3_metadata/h3_profile_rank0.json").read_text())
        assert receipt["completed"] is True
        assert receipt["role"] == "rollout-requests"
        assert receipt["rl_step"] == step
        assert (receipt["request_start"], receipt["request_end"]) == (3, 4)
        assert receipt["requests"] == [
            {"ordinal": index, "request_id": f"request-{index}", "completed": True} for index in (3, 4)
        ]


@pytest.mark.parametrize("request_count", [0, 2, 3])
def test_rollout_short_generation_rejects_incomplete_window(api, capture, rollout_worker, tmp_path, request_count):
    """A successful generation cannot silently pass without both selected requests and a complete trace."""
    backend = rollout_worker.execute_model
    rollout_worker.start_h3_comparison_profile(str(tmp_path), 2, True, 3, 4)
    reference = None
    for index in range(1, request_count + 1):
        rollout_worker.execute_model(SimpleNamespace(request_id=f"request-{index}"), object())
        if index == 3:
            reference = weakref.ref(capture.profiles.pop())
    with pytest.raises(RuntimeError, match="expected requests 3..4"):
        rollout_worker.stop_h3_comparison_profile(completed=True)
    assert rollout_worker.execute_model is backend
    assert rollout_worker.h3_comparison_profile is None
    if request_count == 3:
        assert reference() is None
        receipt = json.loads((tmp_path / "h3_metadata/h3_profile_rank0.json").read_text())
        assert receipt["completed"] is False
        assert len(receipt["requests"]) == 1
    else:
        assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("failed_request", [2, 3, 4])
def test_rollout_request_failure_releases_once_and_restores_dispatch(
    api, capture, rollout_worker, tmp_path, failed_request
):
    """Propagate backend errors, export only started windows as incomplete, and avoid a second export."""
    backend = rollout_worker.execute_model
    references = []

    def execute_model(req, od_config):
        """Fail a real backend call before or within the selected capture window."""
        if req.request_id == "request-3":
            references.append(weakref.ref(capture.profiles.pop()))
        if req.request_id == f"request-{failed_request}":
            raise RuntimeError("backend request failed")

    backend.side_effect = execute_model
    rollout_worker.start_h3_comparison_profile(str(tmp_path), 2, True, 3, 4)
    with pytest.raises(RuntimeError, match="backend request failed"):
        for index in range(1, 5):
            rollout_worker.execute_model(SimpleNamespace(request_id=f"request-{index}"), object())
    rollout_worker.stop_h3_comparison_profile(completed=False)
    assert rollout_worker.execute_model is backend
    assert all(reference() is None for reference in references)
    if failed_request >= 3:
        receipt = json.loads((tmp_path / "h3_metadata/h3_profile_rank0.json").read_text())
        assert receipt["completed"] is False
        assert receipt["requests"][-1]["completed"] is False
        assert capture.events.count("profiler-stop") == 1
    else:
        assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("platform", ["mlu", "cuda"])
@pytest.mark.parametrize("step", [1, 2, 3])
def test_actor_only_schedule_uses_actual_dispatch(api, tmp_path, platform, step):
    """Select two Actor windows with per-step shapes and never access generation profiler workers."""
    config = profiling_config(str(tmp_path / "profiles"))
    config.trainer.total_training_steps = 3
    config.trainer.h3_comparison_profile_windows = [
        Config(role="actor", step=2, record_shapes=False),
        Config(role="actor", step=3, record_shapes=False),
    ]
    config.actor_rollout_ref.rollout = Config(multi_turn=Config(enable=False), n=8)
    config.actor_rollout_ref.actor = Config(
        strategy="fsdp2", ppo_mini_batch_size=16, ppo_epochs=1, data_loader_seed=42, shuffle=False
    )
    config.actor_rollout_ref.model = Config(pipeline=Config(height=256, width=384), vae_scale_factor=8)
    api.validate_h3_profiling(config)
    trainer = SimpleNamespace(
        config=config,
        global_steps=step,
        actor_rollout_wg=SimpleNamespace(update_actor=Mock(return_value={"metrics": {}})),
    )
    # Neither generation manager is exposed: Actor-only windows cannot contact their profiler RPCs.
    with api.h3_generation_profile(trainer):
        pass
    sys.modules["ray"].get.assert_not_called()

    source = ROOT / "verl_omni/trainer/diffusion/ray_diffusion_trainer.py"
    method = next(
        node
        for node in ast.walk(ast.parse(source.read_text()))
        if isinstance(node, ast.FunctionDef) and node.name == "_update_actor"
    )
    batch_td = {}
    namespace = {
        "DataProto": SimpleNamespace(from_single_dict=Mock(return_value="actor-result")),
        "_to_diffusion_worker_tensordict": Mock(return_value=batch_td),
        "embeds_padding_2_no_padding": Mock(return_value=batch_td),
        "tu": sys.modules["verl.utils.tensordict_utils"],
        "rename_dict": Mock(return_value={}),
        "os": os,
    }
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(source), "exec"), namespace)
    assert namespace["_update_actor"](trainer, SimpleNamespace(meta_info={})) == "actor-result"
    trainer.actor_rollout_wg.update_actor.assert_called_once_with(batch_td)
    if step in (2, 3):
        assert batch_td["h3_profile_directory"] == str(tmp_path / f"profiles/rl_step_{step:04d}/actor_update_compute")
        assert batch_td["h3_profile_rl_step"] == step
        assert batch_td["h3_profile_record_shapes"] is False
    else:
        assert "h3_profile_directory" not in batch_td
        assert "h3_profile_rl_step" not in batch_td
        assert "h3_profile_record_shapes" not in batch_td


@pytest.mark.parametrize("shape_flags", [(False, False), (True, True)])
def test_actor_shapes_and_release_across_successive_windows(api, capture, tmp_path, shape_flags):
    """Apply shapes before native startup and release each exported trace before the next RL step."""
    engine = SimpleNamespace(
        get_data_parallel_group=Mock(), device_mesh=SimpleNamespace(get_all_groups=Mock(return_value=[]))
    )
    gc_was_enabled = gc.isenabled()
    gc.disable()
    try:
        for step, record_shapes in enumerate(shape_flags, start=2):
            directory = tmp_path / f"rl_step_{step:04d}" / "actor_update"
            batch = {
                "h3_profile_directory": str(directory),
                "h3_profile_rl_step": step,
                "h3_profile_record_shapes": record_shapes,
            }
            with api.h3_actor_profile(batch, engine):
                reference = weakref.ref(capture.profiles.pop())
                assert reference().record_shapes_at_start is record_shapes
            assert reference() is None
            receipt = json.loads((directory / "h3_metadata/h3_profile_rank0.json").read_text())
            assert receipt["completed"] is True
            assert receipt["record_shapes"] is record_shapes
            assert receipt["rl_step"] == step
    finally:
        gc.collect()
        if gc_was_enabled:
            gc.enable()


@pytest.mark.parametrize(
    "invalid",
    ["v1", "strategy", "collector", "step", "reward", "offline", "duplicate", "role", "replica", "range", "stepwise"],
)
def test_enabled_profile_rejects_incompatible_contract(api, tmp_path, invalid):
    """Reject competing collectors and unsupported trainer/reward/workload configurations before workers start."""
    config = profiling_config(str(tmp_path / "profiles"))
    api.validate_h3_profiling(config)
    if invalid == "v1":
        config.trainer.use_v1 = True
    elif invalid == "strategy":
        config.actor_rollout_ref.actor.strategy = "veomni"
    elif invalid == "collector":
        config.global_profiler.steps = [3]
    elif invalid == "step":
        config.trainer.h3_comparison_profile_windows[0].step = 7
    elif invalid == "reward":
        config.reward.reward_model.enable = True
    elif invalid == "duplicate":
        config.trainer.h3_comparison_profile_windows.append(config.trainer.h3_comparison_profile_windows[0])
    elif invalid == "role":
        config.trainer.h3_comparison_profile_windows[0].role = "acotr"
    elif invalid == "replica":
        config.trainer.h3_comparison_profile_windows[1].replica = -1
    elif invalid == "range":
        config.trainer.h3_comparison_profile_windows[1].request_start = 5
    elif invalid == "stepwise":
        config.actor_rollout_ref.rollout.step_execution = True
    else:
        config.algorithm.sample_source = "offline"
    with pytest.raises(ValueError):
        api.validate_h3_profiling(config)
    assert list(tmp_path.iterdir()) == []
