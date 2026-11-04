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
"""Check opt-in dispatch, process ownership and reversible worker patch boundaries.

Load real modules with standalone package shells; Ray and device APIs are test
doubles. No accelerator, live Ray job, model registration or installed worker is
used. These tests establish lifecycle contracts, not numerical determinism.
"""

import builtins
import hashlib
import importlib
import json
import os
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, call

import pytest
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[2]
PROVIDER = "verl_omni.experiments.h3_determinism"
WORKER_SOURCE = '''import os

class Worker:
    """Expose the reviewed device-binding boundary without a real worker."""

    device = 3

    def initialize(self):
        """Bind before downstream distributed/model initialization."""
        current_omni_platform.set_device(self.device)
        downstream(self.device)
'''


@pytest.fixture
def api(monkeypatch):
    """Use the neighboring experiments tests' standalone import isolation."""
    before = set(sys.modules)
    for name in ("verl_omni", "verl_omni.pipelines", "verl_omni.experiments"):
        package = ModuleType(name)
        package.__path__ = [str(ROOT.joinpath(*name.split(".")))]
        monkeypatch.setitem(sys.modules, name, package)
    names = ("verl_omni.pipelines.experiment_hooks", PROVIDER, "verl_omni.experiments.h3_runtime_patch")
    for name in names:
        monkeypatch.delitem(sys.modules, name, raising=False)
    hooks, policy, patcher = (importlib.import_module(name) for name in names)
    yield SimpleNamespace(hooks=hooks, policy=policy, patcher=patcher)
    for name in set(sys.modules) - before:
        if name.startswith("verl_omni."):
            sys.modules.pop(name, None)


def test_default_dispatch_never_imports_provider(api, monkeypatch):
    """An ordinary run does not import an optional provider or its dependencies."""
    monkeypatch.delenv("VERL_EXPERIMENT_MODULE", raising=False)
    importer = Mock(side_effect=AssertionError("provider imported while disabled"))
    monkeypatch.setattr(api.hooks.importlib, "import_module", importer)
    assert api.hooks.invoke_experiment("runtime_env", object()) is None
    assert api.hooks.invoke_experiment("configure_compute_process", "actor", 42, seeded=True) is None
    importer.assert_not_called()


def test_selected_provider_dispatch_and_missing_callback_fail_explicitly(api, monkeypatch):
    """Only the selected module receives the exact event arguments and return value."""
    for name in ("first_policy", "second_policy"):
        provider = ModuleType(name)
        provider.configure_compute_process = Mock(return_value=object())
        monkeypatch.setitem(sys.modules, name, provider)
    monkeypatch.setenv("VERL_EXPERIMENT_MODULE", "second_policy")
    selected = sys.modules["second_policy"].configure_compute_process
    result = api.hooks.invoke_experiment("configure_compute_process", "actor", 42, seeded=True)
    assert result is selected.return_value
    selected.assert_called_once_with("actor", 42, seeded=True)
    sys.modules["first_policy"].configure_compute_process.assert_not_called()
    with pytest.raises(AttributeError, match="missing_callback"):
        api.hooks.invoke_experiment("missing_callback")


def test_selected_provider_import_failure_is_not_silenced(api, monkeypatch):
    """Selecting an unavailable module must fail instead of silently disabling policy."""
    monkeypatch.setenv("VERL_EXPERIMENT_MODULE", "missing_experiment_provider_for_cpu_test")
    with pytest.raises(ModuleNotFoundError, match="missing_experiment_provider_for_cpu_test"):
        api.hooks.invoke_experiment("runtime_env", object())


@pytest.fixture
def runtime_case(api, monkeypatch, tmp_path):
    """Declare a valid strict startup contract with real OmegaConf and a fake Ray state."""
    for name in api.policy.ENVIRONMENT_NAMES:
        monkeypatch.delenv(name, raising=False)
    environment = {
        "VERL_EXPERIMENT_MODULE": PROVIDER,
        "H3_DETERMINISTIC": "1",
        "H3_DETERMINISTIC_DEVICE": "mlu",
        "H3_RUN_DIR": str(tmp_path / "run"),
        "H3_ATTEMPT_DIR": str(tmp_path / "run/attempt-001"),
        "H3_RUNTIME_PATCH_RECEIPT": str(tmp_path / "worker-receipt.json"),
        "PYTHONHASHSEED": "42",
        "VERL_SEED": "42",
        "TRITON_F32_DEFAULT": "ieee",
        "TORCH_ALLOW_TF32_CUBLAS_OVERRIDE": "0",
        "TORCH_ALLOW_TF32_CNMATMUL_OVERRIDE": "0",
        "VERL_CLAP_BATCH_SIZE": "1",
        "VERL_OPTIMIZER_UPDATE_TRACE_DIR": str(tmp_path / "updates"),
        "VERL_FILE_LOGGER_PATH": str(tmp_path / "run/attempt-001/metrics.jsonl"),
    }
    for name, value in environment.items():
        monkeypatch.setenv(name, value)
    ray = ModuleType("ray")
    ray.is_initialized = Mock(return_value=False)
    monkeypatch.setitem(sys.modules, "ray", ray)
    monkeypatch.setattr(api.policy, "validate_runtime_callback", Mock())
    config = OmegaConf.create(
        {
            "algorithm": {"trainer_type": "policy_gradient"},
            "trainer": {
                "device": "mlu",
                "logger": ["file"],
                "use_v1": False,
                "resume_mode": "disable",
                "total_training_steps": 50,
            },
            "actor_rollout_ref": {
                "actor": {"fsdp_config": {"full_determinism": True, "seed": 42}, "ppo_epochs": 1},
                "rollout": {"full_determinism": True, "seed": 42},
                "model": {"lora_adapter_path": "/canonical/adapter"},
            },
            "data": {"seed": 30, "val_batch_size": 128, "val_max_samples": 128},
            "ray_kwargs": {"ray_init": {"runtime_env": {"env_vars": {}}}},
        }
    )
    return SimpleNamespace(config=config, environment=environment, ray=ray, root=tmp_path / "run/attempt-001")


def test_runtime_env_allowlist_and_driver_never_initializes_devices(api, runtime_case, monkeypatch):
    """Driver startup exports only allowed values without importing compute APIs."""
    monkeypatch.setenv("H3_UNLISTED_POLICY_SETTING", "unlisted")
    monkeypatch.setenv("UNRELATED_PRIVATE_TOKEN", "do-not-forward")
    compute = Mock(side_effect=AssertionError("driver configured a compute device"))
    monkeypatch.setattr(api.policy, "configure_compute_process", compute)
    original_import = builtins.__import__

    def reject_compute_imports(name, *args, **kwargs):
        """Fail before a driver can acquire Torch or a device-owning module."""
        if name.split(".")[0] in ("torch", "torch_mlu") or name.startswith("verl.utils.device"):
            raise AssertionError(f"driver imported compute API: {name}")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", reject_compute_imports)
    result = api.hooks.invoke_experiment("runtime_env", runtime_case.config)
    assert result == {"env_vars": runtime_case.environment}
    api.policy.validate_runtime_callback.assert_called_once_with()
    assert "worker_process_setup_hook" not in result
    assert OmegaConf.to_container(OmegaConf.load(runtime_case.root / "resolved-config.yaml")) == OmegaConf.to_container(
        runtime_case.config, resolve=True
    )
    compute.assert_not_called()
    assert api.policy.PROCESS_SEEDS == {}


@pytest.mark.parametrize("name", ["VERL_SEED", "VERL_OPTIMIZER_UPDATE_TRACE_DIR", "VERL_FILE_LOGGER_PATH"])
def test_runtime_env_rejects_conflicting_ray_values(api, runtime_case, name):
    """Ray must not silently override an experiment seed or trace destination."""
    runtime_case.config.ray_kwargs.ray_init.runtime_env.env_vars[name] = "conflicting-value"
    with pytest.raises(ValueError, match=f"conflicts.*{name}"):
        api.policy.runtime_env(runtime_case.config)
    assert not runtime_case.root.exists()


def test_runtime_env_accepts_matching_ray_value(api, runtime_case):
    """An explicit identical seed is compatible with the startup environment."""
    runtime_case.config.ray_kwargs.ray_init.runtime_env.env_vars.VERL_SEED = "42"
    assert api.policy.runtime_env(runtime_case.config)["env_vars"] == runtime_case.environment


def test_runtime_env_accepts_v0_without_v1_configuration(api, runtime_case):
    """The historical V0 route does not depend on TransferQueue or V1 sync settings."""
    runtime_case.config.trainer.use_v1 = False
    assert "v1" not in runtime_case.config.trainer
    assert api.policy.runtime_env(runtime_case.config)["env_vars"] == runtime_case.environment


def test_runtime_env_rejects_v0_separate_topology(api, runtime_case):
    """Changing Actor/rollout placement would change the historical comparison contract."""
    runtime_case.config.trainer.use_v1 = False
    runtime_case.config.actor_rollout_ref.separate = True
    with pytest.raises(ValueError, match="colocated"):
        api.policy.runtime_env(runtime_case.config)


def test_runtime_env_accepts_bounded_two_step_acceptance(api, runtime_case):
    """A separate two-step run can end before the fresh longrun is authorized."""
    runtime_case.config.trainer.use_v1 = False
    runtime_case.config.trainer.total_training_steps = 2
    assert api.policy.runtime_env(runtime_case.config)["env_vars"] == runtime_case.environment


def test_runtime_env_rejects_unplanned_training_length(api, runtime_case):
    """Do not silently change the comparison window through a launcher override."""
    runtime_case.config.trainer.total_training_steps = 3
    with pytest.raises(ValueError, match="2-step.*50-step"):
        api.policy.runtime_env(runtime_case.config)


def test_runtime_env_rejects_each_mismatched_seed_before_artifacts(api, runtime_case):
    """Actor and rollout independently require 42 before validating or writing evidence."""
    for field in ("actor_rollout_ref.actor.fsdp_config.seed", "actor_rollout_ref.rollout.seed"):
        OmegaConf.update(runtime_case.config, field, 43)
        with pytest.raises(ValueError, match="seeds.*42"):
            api.policy.runtime_env(runtime_case.config)
        OmegaConf.update(runtime_case.config, field, 42)
    api.policy.validate_runtime_callback.assert_not_called()
    assert not runtime_case.root.exists()


def test_runtime_env_requires_native_file_logger(api, runtime_case):
    """The native scalar evidence sink must be enabled for the selected experiment."""
    runtime_case.config.trainer.logger = ["console"]
    with pytest.raises(ValueError, match="file|logger"):
        api.policy.runtime_env(runtime_case.config)
    assert not runtime_case.root.exists()


def test_runtime_env_keeps_attempt_config_and_log_paths_separate(api, runtime_case, monkeypatch):
    """Two attempts under one run keep distinct config snapshots and propagated log paths."""
    first = api.policy.runtime_env(runtime_case.config)
    original_config = (runtime_case.root / "resolved-config.yaml").read_bytes()
    second_attempt = runtime_case.root.parent / "attempt-002"
    second_log = str(second_attempt / "metrics.jsonl")
    monkeypatch.setenv("H3_ATTEMPT_DIR", str(second_attempt))
    monkeypatch.setenv("VERL_FILE_LOGGER_PATH", second_log)
    second = api.policy.runtime_env(runtime_case.config)
    assert second["env_vars"]["VERL_FILE_LOGGER_PATH"] == second_log
    assert first["env_vars"]["VERL_FILE_LOGGER_PATH"] != second_log
    assert first["env_vars"]["H3_RUN_DIR"] == second["env_vars"]["H3_RUN_DIR"]
    assert (runtime_case.root / "resolved-config.yaml").read_bytes() == original_config
    assert (second_attempt / "resolved-config.yaml").exists()
    assert not (runtime_case.root.parent / "resolved-config.yaml").exists()


def test_runtime_env_rejects_ray_setup_hook(api, runtime_case):
    """Compute policy cannot be moved before device binding via a Ray setup hook."""
    runtime_case.config.ray_kwargs.ray_init.runtime_env.worker_process_setup_hook = "module.bootstrap"
    with pytest.raises(ValueError, match="setup hook"):
        api.policy.runtime_env(runtime_case.config)
    assert not runtime_case.root.exists()


def test_runtime_env_rejects_already_initialized_ray(api, runtime_case):
    """An existing Ray job cannot retroactively acquire the process startup policy."""
    runtime_case.ray.is_initialized.return_value = True
    with pytest.raises(ValueError, match="before ray.init"):
        api.policy.runtime_env(runtime_case.config)
    assert not runtime_case.root.exists()


@pytest.mark.parametrize("selected", [False, True])
def test_worker_callback_runs_after_binding_and_before_downstream(api, monkeypatch, selected):
    """Execute the patched synthetic worker and check actual event order."""
    events = []
    provider = ModuleType("worker_policy")
    provider.configure_compute_process = Mock(side_effect=lambda *args: events.append(("policy", args)))
    monkeypatch.setitem(sys.modules, provider.__name__, provider)
    monkeypatch.delenv("VERL_EXPERIMENT_MODULE", raising=False)
    if selected:
        monkeypatch.setenv("VERL_EXPERIMENT_MODULE", provider.__name__)
    monkeypatch.setenv("VERL_SEED", "42")
    namespace = {
        "current_omni_platform": SimpleNamespace(set_device=lambda device: events.append(("bind", device))),
        "downstream": lambda device: events.append(("downstream", device)),
    }
    patched = api.patcher.patched_worker_source(WORKER_SOURCE)
    exec(compile(patched, "synthetic_diffusion_worker.py", "exec"), namespace)
    namespace["Worker"]().initialize()
    expected = [("bind", 3), ("downstream", 3)]
    if selected:
        expected.insert(1, ("policy", ("diffusion-worker", 42)))
        provider.configure_compute_process.assert_called_once_with("diffusion-worker", 42)
    else:
        provider.configure_compute_process.assert_not_called()
    assert events == expected
    assert patched.count(api.patcher.DEVICE_ANCHOR) == 1
    assert api.patcher.patched_worker_source(patched) == patched


@pytest.mark.parametrize(
    "layout", ["missing", "duplicate", "binding_signature", "orphan_callback", "patched_duplicate"]
)
def test_worker_patch_rejects_unreviewed_layout_without_writing(api, tmp_path, layout):
    """Ambiguous or changed binding layouts must fail even if a callback already exists."""
    anchor = api.patcher.DEVICE_ANCHOR
    sources = {
        "missing": WORKER_SOURCE.replace(anchor, ""),
        "duplicate": WORKER_SOURCE.replace(anchor, anchor * 2),
        "binding_signature": WORKER_SOURCE.replace(anchor, anchor.replace("self.device", "device=self.device")),
        "orphan_callback": WORKER_SOURCE.replace(anchor, api.patcher.WORKER_CALLBACK),
        "patched_duplicate": api.patcher.patched_worker_source(WORKER_SOURCE).replace(anchor, anchor * 2),
    }
    source = sources[layout]
    worker = tmp_path / "diffusion_worker.py"
    worker.write_text(source)
    receipts = tmp_path / "receipts"
    with pytest.raises(ValueError, match="device-binding boundary"):
        api.patcher.install_worker_callback(worker, receipts)
    assert worker.read_text() == source
    assert not receipts.exists()


def test_worker_patch_hashes_and_backup_are_independently_verifiable(api, tmp_path):
    """The original bytes, installed bytes and on-disk receipt form a reversible chain."""
    worker = tmp_path / "diffusion_worker.py"
    worker.write_text(WORKER_SOURCE)
    receipts = tmp_path / "receipts"
    result = api.patcher.install_worker_callback(worker, receipts)
    before_hash = hashlib.sha256(WORKER_SOURCE.encode()).hexdigest()
    after_hash = hashlib.sha256(worker.read_bytes()).hexdigest()
    assert result["changed"] is True
    assert result["before_sha256"] == before_hash
    assert result["after_sha256"] == after_hash
    assert result["worker_path"] == str(worker)
    backup = receipts / f"diffusion_worker.{before_hash}.py"
    assert backup.read_bytes() == WORKER_SOURCE.encode()
    assert hashlib.sha256(backup.read_bytes()).hexdigest() == result["before_sha256"]
    receipt_path = receipts / f"worker-callback.{after_hash}.installed.json"
    assert json.loads(receipt_path.read_text()) == result
    assert api.patcher.verify_worker_callback(receipt_path, worker) == result


@pytest.mark.parametrize("problem", ["changed_path", "changed_bytes", "unpatched"])
def test_worker_receipt_rejects_path_hash_or_missing_callback(api, tmp_path, problem):
    """Verify real temporary bytes; a matching forged hash cannot authorize an unpatched worker."""
    worker = tmp_path / "diffusion_worker.py"
    worker.write_text(WORKER_SOURCE)
    receipt = api.patcher.install_worker_callback(worker, tmp_path / "receipts")
    receipt_path = tmp_path / "receipts" / f"worker-callback.{receipt['after_sha256']}.installed.json"
    if problem == "changed_path":
        other_worker = tmp_path / "other_worker.py"
        other_worker.write_bytes(worker.read_bytes())
        worker = other_worker
        message = "different installed worker"
    elif problem == "changed_bytes":
        worker.write_bytes(worker.read_bytes() + b"\n# modification outside the callback\n")
        message = "changed after"
    else:
        worker.write_text(WORKER_SOURCE)
        receipt["after_sha256"] = hashlib.sha256(worker.read_bytes()).hexdigest()
        receipt_path.write_text(json.dumps(receipt))
        message = "device-bound worker callback"
    original_bytes = worker.read_bytes()
    with pytest.raises(ValueError, match=message):
        api.patcher.verify_worker_callback(receipt_path, worker)
    assert worker.read_bytes() == original_bytes


def test_worker_patch_reinstall_preserves_first_original_and_receipt(api, tmp_path):
    """Idempotent installation must preserve the first restore source and its provenance."""
    worker = tmp_path / "diffusion_worker.py"
    worker.write_text(WORKER_SOURCE)
    receipts = tmp_path / "receipts"
    first = api.patcher.install_worker_callback(worker, receipts)
    assert first["changed"] is True
    snapshot = {path.name: path.read_bytes() for path in receipts.iterdir()}
    installed = worker.read_bytes()
    second = api.patcher.install_worker_callback(worker, receipts)
    assert second["changed"] is False
    observed = receipts / f"worker-callback.{second['after_sha256']}.observed.json"
    assert json.loads(observed.read_text()) == second
    assert worker.read_bytes() == installed
    for name, original in snapshot.items():
        assert (receipts / name).read_bytes() == original, (
            "Reinstall overwrote first-install provenance; the original before_sha256 must remain recoverable"
        )


def test_worker_patch_compile_error_leaves_source_and_receipts_untouched(api, tmp_path):
    """Syntax validation precedes every persistent mutation."""
    source = WORKER_SOURCE + "invalid syntax here\n"
    worker = tmp_path / "diffusion_worker.py"
    worker.write_text(source)
    with pytest.raises(SyntaxError):
        api.patcher.install_worker_callback(worker, tmp_path / "receipts")
    assert worker.read_text() == source
    assert not (tmp_path / "receipts").exists()


@pytest.fixture
def process_case(api, runtime_case, monkeypatch):
    """Supply observable process/device boundaries without importing a real backend."""
    torch = ModuleType("torch")
    torch.backends = SimpleNamespace(
        mlu=SimpleNamespace(matmul=SimpleNamespace(allow_tf32=True), custom=SimpleNamespace(allow_tf32=True)),
        cnnl=SimpleNamespace(allow_tf32=True, deterministic=False),
    )
    torch.set_float32_matmul_precision = Mock()
    torch.use_deterministic_algorithms = Mock()
    torch.are_deterministic_algorithms_enabled = Mock(return_value=True)
    torch.is_deterministic_algorithms_warn_only_enabled = Mock(return_value=False)
    torch.get_float32_matmul_precision = Mock(return_value="highest")
    monkeypatch.setitem(sys.modules, "torch", torch)
    for name in ("verl", "verl.utils", "verl.workers", "verl.workers.engine"):
        package = ModuleType(name)
        package.__path__ = []
        monkeypatch.setitem(sys.modules, name, package)
    device = ModuleType("verl.utils.device")
    current_device = Mock(return_value=2)
    device.get_device_name = Mock(return_value="mlu")
    device.get_torch_device = Mock(return_value=SimpleNamespace(current_device=current_device))
    device.is_cuda_available = False
    engine_utils = ModuleType("verl.workers.engine.utils")
    seed_owner = Mock()
    engine_utils.enable_full_determinism = seed_owner
    monkeypatch.setitem(sys.modules, device.__name__, device)
    monkeypatch.setitem(sys.modules, engine_utils.__name__, engine_utils)
    pid = Mock(return_value=101)
    monkeypatch.setattr(api.policy, "os", SimpleNamespace(environ=os.environ, getpid=pid, getppid=lambda: 99))
    return SimpleNamespace(seed_owner=seed_owner, pid=pid, torch=torch, root=runtime_case.root)


def test_existing_seed_owner_and_same_pid_do_not_seed_again(api, process_case):
    """An already-seeded Actor owns RNG even when a second role shares its PID."""
    state = api.policy.configure_compute_process("actor", 42, seeded=True)
    api.policy.configure_compute_process("reward", 42)
    process_case.seed_owner.assert_not_called()
    assert api.policy.PROCESS_SEEDS == {101: 42}
    assert state["pid"] == 101 and state["seed"] == 42 and state["device_index"] == 2
    assert state["matmul_allow_tf32"] is False and state["custom_allow_tf32"] is False
    process_case.torch.set_float32_matmul_precision.assert_called_with("highest")
    process_case.torch.use_deterministic_algorithms.assert_called_with(True, warn_only=False)
    receipt = process_case.root / "determinism/101-actor.json"
    assert json.loads(receipt.read_text()) == state
    assert state["policy_sha256"] == hashlib.sha256(Path(api.policy.__file__).read_bytes()).hexdigest()


def test_same_pid_seeds_once_but_fork_child_initializes(api, process_case):
    """A simulated fork inherits PROCESS_SEEDS yet its new PID still initializes RNG."""
    api.policy.configure_compute_process("reward", 42)
    api.policy.configure_compute_process("diffusion-worker", 42)
    process_case.seed_owner.assert_called_once_with(seed=42)
    process_case.pid.return_value = 102
    api.policy.configure_compute_process("reward", 42)
    assert process_case.seed_owner.call_args_list == [call(seed=42), call(seed=42)]
    assert api.policy.PROCESS_SEEDS == {101: 42, 102: 42}
    assert json.loads((process_case.root / "determinism/102-reward.json").read_text())["pid"] == 102


def test_same_pid_cannot_change_seed(api, process_case):
    """A conflicting seed fails without reseeding or replacing the previous receipt."""
    api.policy.configure_compute_process("reward", 42)
    receipt = process_case.root / "determinism/101-reward.json"
    original = receipt.read_bytes()
    with pytest.raises(ValueError, match="different experiment seed"):
        api.policy.configure_compute_process("reward", 43)
    process_case.seed_owner.assert_called_once_with(seed=42)
    assert api.policy.PROCESS_SEEDS == {101: 42}
    assert receipt.read_bytes() == original


def test_seed_owner_failure_does_not_cache_initialization(api, process_case):
    """Failed RNG initialization cannot masquerade as an initialized process."""
    process_case.seed_owner.side_effect = RuntimeError("seed initialization failed")
    with pytest.raises(RuntimeError, match="seed initialization failed"):
        api.policy.configure_compute_process("reward", 42)
    assert api.policy.PROCESS_SEEDS == {}
    assert not (process_case.root / "determinism").exists()
    process_case.seed_owner.side_effect = None
    api.policy.configure_compute_process("reward", 42)
    assert process_case.seed_owner.call_count == 2
