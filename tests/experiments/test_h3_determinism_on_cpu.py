"""Check the H3 strict policy contract and the ordinary-mode boundary on CPU."""

import importlib.util
import json
import os
import random
import subprocess
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

ROOT = Path(__file__).parents[2]
POLICY = ROOT / "verl_omni/experiments/h3_determinism.py"


@pytest.fixture
def policy(monkeypatch, tmp_path):
    """Load the real policy and restore environment, RNG and backend flags after each check."""
    spec = importlib.util.spec_from_file_location("h3_determinism", POLICY)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    enabled = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    cudnn = (torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark, torch.backends.cudnn.enabled)
    environment = dict(os.environ)
    python_rng, numpy_rng, torch_rng = random.getstate(), np.random.get_state(), torch.get_rng_state()
    monkeypatch.setenv("H3_DETERMINISTIC", "1")
    monkeypatch.setenv("H3_DETERMINISTIC_DEVICE", "cuda")
    monkeypatch.setenv("H3_DETERMINISM_DIR", str(tmp_path / "receipts"))
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":16:8")
    monkeypatch.setenv("VERL_FULL_DETERMINISM", "1")
    monkeypatch.setenv("VLLM_BATCH_INVARIANT", "1")
    yield module
    torch.use_deterministic_algorithms(enabled, warn_only=warn_only)
    torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark, torch.backends.cudnn.enabled = cudnn
    random.setstate(python_rng)
    np.random.set_state(numpy_rng)
    torch.set_rng_state(torch_rng)
    os.environ.clear()
    os.environ.update(environment)


def test_strict_policy_uses_official_seed_and_flags(policy):
    """Exercise the installed official helper, not a copied or mocked implementation."""
    state = policy.configure_h3_determinism("actor-engine", 42)
    assert state["seed"] == 42
    assert state["deterministic_algorithms"] is True and state["warn_only"] is True
    assert state["backend_deterministic"] is True and state["backend_benchmark"] is False
    assert state["cudnn_enabled"] is False
    assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == ":16:8"
    assert os.environ["NCCL_ALGO"] == "Ring" and os.environ["NCCL_PROTO"] == "Simple"
    assert random.random() == random.Random(42).random()
    assert np.random.rand() == np.random.RandomState(42).rand()
    assert torch.equal(torch.rand(4), torch.rand(4, generator=torch.Generator().manual_seed(42)))
    receipt = Path(os.environ["H3_DETERMINISM_DIR"]) / f"{os.getpid()}-actor-engine.json"
    assert json.loads(receipt.read_text()) == state


def test_disabled_policy_preserves_rng_environment_and_flags(policy, monkeypatch):
    """Ordinary training must not reseed, write receipts or overwrite external settings."""
    monkeypatch.setenv("H3_DETERMINISTIC", "0")
    torch.use_deterministic_algorithms(False)
    environment = dict(os.environ)
    python_rng, torch_rng = random.getstate(), torch.get_rng_state().clone()
    assert policy.configure_h3_determinism("actor-engine", 42) is None
    assert policy.h3_determinism_runtime_env(None) == {}
    assert torch.are_deterministic_algorithms_enabled() is False
    assert dict(os.environ) == environment
    assert random.getstate() == python_rng and torch.equal(torch.get_rng_state(), torch_rng)
    assert not Path(os.environ["H3_DETERMINISM_DIR"]).exists()


def test_mlu_policy_does_not_write_unsupported_benchmark(policy, monkeypatch):
    """Check CNNL policy dispatch only; this mocked backend cannot validate MLU kernels."""
    monkeypatch.setenv("H3_DETERMINISTIC_DEVICE", "mlu")
    monkeypatch.setitem(sys.modules, "torch_mlu", ModuleType("torch_mlu"))
    cnnl = SimpleNamespace(deterministic=False, benchmark=True)
    monkeypatch.setattr(torch.backends, "cnnl", cnnl, raising=False)
    state = policy.configure_h3_determinism("reward-worker", 42)
    assert cnnl.deterministic is True and cnnl.benchmark is True
    assert state["device_type"] == "mlu" and state["warn_only"] is True


def test_runtime_env_keeps_data_seed_independent(policy):
    """Propagate complete strict settings while preserving the seed30 data schedule."""
    config = OmegaConf.create({
        "trainer": {"device": "cuda"}, "data": {"seed": 30},
        "actor_rollout_ref": {"actor": {"fsdp_config": {"seed": 42}}, "rollout": {"seed": 42}},
    })
    before = OmegaConf.to_container(config)
    runtime = policy.h3_determinism_runtime_env(config)
    assert "worker_process_setup_hook" not in runtime
    assert runtime["env_vars"]["NCCL_PROTO"] == "Simple"
    assert runtime["env_vars"]["VERL_FULL_DETERMINISM"] == "1"
    assert runtime["env_vars"]["VERL_SEED"] == "42"
    assert OmegaConf.to_container(config) == before


def test_wrong_workspace_fails_before_compute(policy, monkeypatch):
    """Reject a stale kernel-only launcher before any model work."""
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    with pytest.raises(ValueError, match="before Python initialization"):
        policy.configure_h3_determinism("actor-engine", 42)


def test_mlu_runtime_env_carries_tuner_paths(policy, monkeypatch, tmp_path):
    """Keep the tested CNCL plugin and rules available in actual Ray workers."""
    monkeypatch.setenv("H3_DETERMINISTIC_DEVICE", "mlu")
    monkeypatch.setenv("CNCL_TUNER_PLUGIN", str(tmp_path / "libcncl-tuner-csv.so"))
    monkeypatch.setenv("CNCL_TUNER_CONFIG_FILE", str(tmp_path / "ring.conf"))
    monkeypatch.setitem(sys.modules, "torch_mlu", ModuleType("torch_mlu"))
    monkeypatch.setattr(torch.backends, "cnnl", SimpleNamespace(deterministic=False, benchmark=False), raising=False)
    config = OmegaConf.create({
        "trainer": {"device": "mlu"},
        "actor_rollout_ref": {"actor": {"fsdp_config": {"seed": 42}}, "rollout": {"seed": 42}},
    })
    environment = policy.h3_determinism_runtime_env(config)["env_vars"]
    assert environment["CNCL_TUNER_PLUGIN"] == str(tmp_path / "libcncl-tuner-csv.so")
    assert environment["CNCL_TUNER_CONFIG_FILE"] == str(tmp_path / "ring.conf")


@pytest.mark.parametrize("enabled", ["0", "1"])
def test_shell_policy_is_selected_before_python(tmp_path, enabled):
    """Strict mode owns its environment; off mode leaves an external environment unchanged."""
    env = {
        **os.environ, "H3_DETERMINISTIC": enabled, "H3_RUN_DIR": str(tmp_path),
        "PYTHONHASHSEED": "19", "NCCL_ALGO": "Tree", "NCCL_PROTO": "LL128",
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
    }
    script = "platform=cuda\nsource verl_omni/experiments/h3_determinism.sh\n"
    script += "printf '%s\\n' \"$PYTHONHASHSEED\" \"$NCCL_ALGO\" \"$NCCL_PROTO\" \"$CUBLAS_WORKSPACE_CONFIG\""
    result = subprocess.run(["bash", "-c", script], cwd=ROOT, env=env, check=True, capture_output=True, text=True)
    assert result.stdout.splitlines() == (["42", "Ring", "Simple", ":16:8"] if enabled == "1" else ["19", "Tree", "LL128", ":4096:8"])


@pytest.mark.parametrize("enabled", ["0", "1"])
def test_mlu_shell_tuner_boundary(tmp_path, enabled):
    """Select the frozen tuner only in strict mode and preserve external settings otherwise."""
    env = {**os.environ, "H3_DETERMINISTIC": enabled, "H3_RUN_DIR": str(tmp_path),
           "H3_CNCL_TUNER_DIR": str(tmp_path / "tuner"),
           "CNCL_TUNER_PLUGIN": "external-plugin", "CNCL_TUNER_CONFIG_FILE": "external-rules"}
    script = "platform=mlu\nsource verl_omni/experiments/h3_determinism.sh\n"
    script += "printf '%s\\n' \"$CNCL_TUNER_PLUGIN\" \"$CNCL_TUNER_CONFIG_FILE\""
    result = subprocess.run(["bash", "-c", script], cwd=ROOT, env=env, check=True, capture_output=True, text=True)
    assert result.stdout.splitlines() == ([str(tmp_path / "tuner/libcncl-tuner-csv.so"), str(tmp_path / "tuner/ring.conf")]
                                         if enabled == "1" else ["external-plugin", "external-rules"])
