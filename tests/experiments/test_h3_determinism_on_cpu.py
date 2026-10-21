"""Verify kernel-policy propagation and preservation of RNG and communication state."""

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
    """Load the experiment without application registration and restore process flags afterward."""
    spec = importlib.util.spec_from_file_location("h3_determinism", POLICY)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    enabled = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    cudnn = (torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark, torch.backends.cudnn.enabled)
    monkeypatch.setenv("H3_DETERMINISTIC", "1")
    monkeypatch.setenv("H3_DETERMINISTIC_DEVICE", "cuda")
    monkeypatch.setenv("H3_DETERMINISM_DIR", str(tmp_path / "receipts"))
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    monkeypatch.setenv("TORCHINDUCTOR_CACHE_DIR", str(tmp_path / "inductor"))
    yield module
    torch.use_deterministic_algorithms(enabled, warn_only=warn_only)
    torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark, torch.backends.cudnn.enabled = cudnn


def test_cuda_policy_preserves_rng_collectives_and_precision(policy, monkeypatch):
    """Changing kernels must not reseed, select NCCL algorithms, disable cuDNN, or change TF32."""
    monkeypatch.setenv("NCCL_ALGO", "Tree")
    monkeypatch.setenv("NCCL_PROTO", "LL128")
    monkeypatch.setenv("PYTHONHASHSEED", "123")
    python_rng, numpy_rng, torch_rng = random.getstate(), np.random.get_state(), torch.get_rng_state().clone()
    environment = dict(os.environ)
    enabled, tf32 = torch.backends.cudnn.enabled, torch.backends.cuda.matmul.allow_tf32
    state = policy.configure_h3_determinism("actor")
    assert state["deterministic_algorithms"] is True and state["warn_only"] is False
    assert state["backend_deterministic"] is True and state["backend_benchmark"] is False
    assert torch.backends.cudnn.enabled == enabled
    assert torch.backends.cuda.matmul.allow_tf32 == tf32
    assert dict(os.environ) == environment
    assert random.getstate() == python_rng
    actual_numpy_rng = np.random.get_state()
    assert actual_numpy_rng[0] == numpy_rng[0]
    assert np.array_equal(actual_numpy_rng[1], numpy_rng[1])
    assert actual_numpy_rng[2:] == numpy_rng[2:]
    assert torch.equal(torch.get_rng_state(), torch_rng)
    receipt = Path(os.environ["H3_DETERMINISM_DIR"]) / f"{os.getpid()}-actor.json"
    assert json.loads(receipt.read_text()) == state


def test_disabled_policy_is_noop(policy, monkeypatch):
    """The ordinary experiment must keep flags, environment, and filesystem untouched."""
    monkeypatch.setenv("H3_DETERMINISTIC", "0")
    torch.use_deterministic_algorithms(False)
    environment = dict(os.environ)
    assert policy.configure_h3_determinism("actor") is None
    assert policy.h3_determinism_runtime_env(None) == {}
    assert torch.are_deterministic_algorithms_enabled() is False
    assert dict(os.environ) == environment
    assert not Path(os.environ["H3_DETERMINISM_DIR"]).exists()


def test_mlu_policy_sets_cnnl_without_cuda_side_effects(policy, monkeypatch):
    """A mocked MLU backend checks dispatch only; this is not an MLU kernel validation."""
    monkeypatch.setenv("H3_DETERMINISTIC_DEVICE", "mlu")
    monkeypatch.setitem(sys.modules, "torch_mlu", ModuleType("torch_mlu"))
    cnnl = SimpleNamespace(deterministic=False, benchmark=True)
    monkeypatch.setattr(torch.backends, "cnnl", cnnl, raising=False)
    before = (torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark, torch.backends.cudnn.enabled)
    state = policy.configure_h3_determinism("reward")
    assert cnnl.deterministic is True and cnnl.benchmark is False
    assert state["device_type"] == "mlu" and state["warn_only"] is False
    assert (torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark, torch.backends.cudnn.enabled) == before


@pytest.mark.parametrize(
    "path",
    [
        "actor_rollout_ref.actor.fsdp_config.full_determinism",
        "actor_rollout_ref.ref.fsdp_config.full_determinism",
        "actor_rollout_ref.rollout.full_determinism",
        "reward.reward_model.rollout.full_determinism",
    ],
)
def test_reseeding_helper_is_rejected(policy, path):
    """Reject the existing helper because its seed and communication changes violate this experiment."""
    config = OmegaConf.create({"trainer": {"device": "cuda"}})
    OmegaConf.update(config, path, True, force_add=True)
    with pytest.raises(ValueError, match="preserve RNG and communication"):
        policy.h3_determinism_runtime_env(config)


def test_runtime_env_preserves_config_and_copies_only_kernel_policy(policy):
    """Propagate inputs without a setup hook that would initialize devices before Ray assigns them."""
    config = OmegaConf.create({"trainer": {"device": "cuda"}, "data": {"seed": 30}})
    before = OmegaConf.to_container(config)
    runtime = policy.h3_determinism_runtime_env(config)
    assert "worker_process_setup_hook" not in runtime
    assert set(runtime["env_vars"]) == {
        "H3_DETERMINISTIC",
        "H3_DETERMINISTIC_DEVICE",
        "H3_DETERMINISM_DIR",
        "CUBLAS_WORKSPACE_CONFIG",
    }
    assert OmegaConf.to_container(config) == before


def test_wrong_workspace_fails_before_kernel_work(policy, monkeypatch):
    """Catch the old helper's workspace overwrite rather than report a falsely configured launch."""
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":16:8")
    with pytest.raises(ValueError, match="before Python initialization"):
        policy.configure_h3_determinism("actor")


def test_fresh_python_requires_its_own_policy(policy):
    """Prove a new interpreter needs explicit initialization even when its parent is strict."""
    policy.configure_h3_determinism("parent")
    script = f"""
import importlib.util, json, torch
spec = importlib.util.spec_from_file_location('h3_determinism', {str(POLICY)!r})
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
before = torch.are_deterministic_algorithms_enabled()
state = module.configure_h3_determinism('child')
print(json.dumps({{'before': before, 'after': state['deterministic_algorithms'], 'warn_only': state['warn_only']}}))
"""
    result = subprocess.run([sys.executable, "-c", script], check=True, capture_output=True, text=True)
    assert json.loads(result.stdout) == {"before": False, "after": True, "warn_only": False}


@pytest.mark.parametrize("platform", ["cuda", "mlu"])
def test_shell_policy_does_not_change_seed_or_collectives(tmp_path, platform):
    """Check the real shell source before Python startup for both platforms."""
    env = {
        **os.environ,
        "H3_DETERMINISTIC": "1",
        "H3_RUN_DIR": str(tmp_path),
        "PYTHONHASHSEED": "19",
        "NCCL_ALGO": "Tree",
        "CUBLAS_WORKSPACE_CONFIG": ":16:8",
    }
    script = f"platform={platform}\nsource verl_omni/experiments/h3_determinism.sh\n"
    script += 'printf \'%s\\n\' "$PYTHONHASHSEED" "$NCCL_ALGO" "$CUBLAS_WORKSPACE_CONFIG" "$H3_DETERMINISTIC_DEVICE"'
    result = subprocess.run(["bash", "-c", script], cwd=ROOT, env=env, check=True, capture_output=True, text=True)
    assert result.stdout.splitlines() == ["19", "Tree", ":4096:8" if platform == "cuda" else ":16:8", platform]
