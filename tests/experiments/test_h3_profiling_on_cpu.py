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
import hashlib
import importlib.util
import json
import sys
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest

ROOT = Path(__file__).parents[2]


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
            total_training_steps=6,
            use_v1=False,
        ),
        actor_rollout_ref=Config(actor=Config(strategy="fsdp2")),
        global_profiler=Config(tool=None, steps=None),
        reward=Config(custom_reward_function=Config(path="clap-imagebind.py"), reward_model=Config(enable=False)),
        algorithm=Config(adv_estimator="flow_grpo", sample_source="online"),
    )


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


def test_actor_two_mini_updates_share_actual_worker_window(api, capture, tmp_path):
    """Execute the production worker method and keep both optimizer mini-updates inside one window."""
    source = ROOT / "verl_omni/workers/engine_workers.py"
    tree = ast.parse(source.read_text())
    worker = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "ActorRolloutRefWorker")
    method = next(node for node in worker.body if isinstance(node, ast.FunctionDef) and node.name == "update_actor")
    method.decorator_list = []
    namespace = {"TensorDict": dict, "tu": sys.modules["verl.utils.tensordict_utils"]}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(source), "exec"), namespace)
    output = SimpleNamespace(cpu=Mock(return_value="cpu-result"))

    def train_mini_batch(data):
        """Represent two optimizer updates dispatched together by the real worker boundary."""
        assert data["enable_timestep_staging"] is False
        for index in range(2):
            assert capture.events == ["profiler-start", "kineto-start"] + [f"mini-update-{i}" for i in range(index)]
            capture.events.append(f"mini-update-{index}")
        return output

    group = object()
    engine = SimpleNamespace(
        get_data_parallel_group=Mock(return_value=group),
        device_mesh=SimpleNamespace(get_all_groups=Mock(return_value=[group])),
    )
    actor = SimpleNamespace(engine=engine, train_mini_batch=train_mini_batch)
    worker = SimpleNamespace(config=Config(actor=Config()), actor=actor)
    batch = {"h3_profile_directory": str(tmp_path / "actor"), "h3_profile_rl_step": 3}
    assert namespace["update_actor"](worker, batch) == "cpu-result"
    assert len(capture.profiles) == 1
    assert capture.events == [
        "profiler-start",
        "kineto-start",
        "mini-update-0",
        "mini-update-1",
        "profiler-step",
        "profiler-stop",
    ]
    receipt = json.loads((tmp_path / "actor/h3_metadata/h3_profile_rank0.json").read_text())
    assert receipt["completed"] is True
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
    batch = {"h3_profile_directory": str(tmp_path / "actor"), "h3_profile_rl_step": 3}
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
def test_generation_fanout_covers_all_servers_rewards_and_failure(api, tmp_path, failed):
    """Fan out start/stop to every generation participant and propagate phase completion accurately."""
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
        assert len(sys.modules["ray"].get.call_args.args[0]) == 6
        assert all(server.collective_rpc.remote.call_count == 1 for server in servers)
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
        assert server.collective_rpc.remote.call_args_list[0].args == ("start_h3_comparison_profile",)
        assert server.collective_rpc.remote.call_args_list[0].kwargs == {
            "kwargs": {"directory": str(root / f"rollout_replica_{index:03d}"), "rl_step": 3}
        }
        assert server.collective_rpc.remote.call_args_list[1].kwargs == {"kwargs": {"completed": not failed}}
        assert server.collective_rpc.remote.call_args_list[1].args == ("stop_h3_comparison_profile",)
    for index, worker in enumerate(rewards):
        worker.start_h3_comparison_profile.remote.assert_called_once_with(str(root / f"reward_worker_{index:03d}"), 3)
        worker.stop_h3_comparison_profile.remote.assert_called_once_with(not failed)
    assert sys.modules["ray"].get.call_count == 2
    assert len(sys.modules["ray"].get.call_args.args[0]) == 6


def test_non_target_generation_step_does_not_contact_workers(api, tmp_path):
    """An enabled capture directory does not collect any unselected RL step."""
    trainer = SimpleNamespace(config=profiling_config(str(tmp_path / "profiles")), global_steps=2)
    with api.h3_generation_profile(trainer):
        pass
    sys.modules["ray"].get.assert_not_called()
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("invalid", ["v1", "strategy", "collector", "step", "reward", "offline"])
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
        config.trainer.h3_comparison_profile_step = 7
    elif invalid == "reward":
        config.reward.reward_model.enable = True
    else:
        config.algorithm.sample_source = "offline"
    with pytest.raises(ValueError):
        api.validate_h3_profiling(config)
    assert list(tmp_path.iterdir()) == []
