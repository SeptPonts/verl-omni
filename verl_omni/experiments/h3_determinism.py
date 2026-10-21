"""Opt-in H3 kernel determinism without changing RNG or collective configuration."""

import hashlib
import json
import multiprocessing
import os
from datetime import UTC, datetime
from pathlib import Path


def configure_h3_determinism(role):
    """Apply kernel policy after Ray binds devices and before this process loads models."""
    if os.environ.get("H3_DETERMINISTIC") != "1":
        return None

    import torch

    device = os.environ["H3_DETERMINISTIC_DEVICE"]
    if device == "cuda":
        if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
            raise ValueError("Start H3 with CUBLAS_WORKSPACE_CONFIG=:4096:8 before Python initialization")
        backend = torch.backends.cudnn
    elif device == "mlu":
        import torch_mlu  # noqa: F401 -- registers torch.backends.cnnl in this process

        backend = torch.backends.cnnl
    else:
        raise ValueError(f"H3 deterministic experiments support cuda or mlu, received {device!r}")

    torch.use_deterministic_algorithms(True, warn_only=False)
    backend.deterministic = True
    backend.benchmark = False
    state = {
        "observed_at": datetime.now(UTC).isoformat(),
        "pid": os.getpid(),
        "parent_pid": os.getppid(),
        "process_name": multiprocessing.current_process().name,
        "role": role,
        "rank": os.environ.get("RANK"),
        "local_rank": os.environ.get("LOCAL_RANK"),
        "device_type": device,
        "torch_version": torch.__version__,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "warn_only": torch.is_deterministic_algorithms_warn_only_enabled(),
        "backend_deterministic": backend.deterministic,
        "backend_benchmark": backend.benchmark,
        "policy_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "environment": {
            name: os.environ.get(name)
            for name in (
                "H3_DETERMINISTIC",
                "CUBLAS_WORKSPACE_CONFIG",
                "PYTHONHASHSEED",
                "NCCL_ALGO",
                "NCCL_PROTO",
                "NCCL_DETERMINISTIC",
                "CNCL_ALGO",
                "CNCL_PROTO",
                "CNNL_INTERNAL_OVERRIDE_RAND_RNG_TYPE",
                "CUDA_VISIBLE_DEVICES",
                "MLU_VISIBLE_DEVICES",
                "CN_VISIBLE_DEVICES",
            )
        },
    }
    if device == "cuda":
        state["cudnn_enabled"] = backend.enabled
        state["matmul_allow_tf32"] = torch.backends.cuda.matmul.allow_tf32
    directory = Path(os.environ["H3_DETERMINISM_DIR"])
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"{os.getpid()}-{role}.json").write_text(json.dumps(state, indent=2) + "\n")
    return state


def h3_determinism_runtime_env(config):
    """Validate the kernel-only contract and propagate policy inputs to model workers."""
    if os.environ.get("H3_DETERMINISTIC") != "1":
        return {}

    from omegaconf import OmegaConf

    # VeRL's full_determinism reseeds RNG, changes NCCL, and disables cuDNN.
    # The H3 experiment deliberately owns only the kernel policy.
    for path in (
        "actor_rollout_ref.actor.fsdp_config.full_determinism",
        "actor_rollout_ref.ref.fsdp_config.full_determinism",
        "actor_rollout_ref.rollout.full_determinism",
        "reward.reward_model.rollout.full_determinism",
    ):
        if OmegaConf.select(config, path, default=False):
            raise ValueError(f"H3 kernel determinism requires {path}=false to preserve RNG and communication")
    if config.trainer.device != os.environ["H3_DETERMINISTIC_DEVICE"]:
        raise ValueError("H3_DETERMINISTIC_DEVICE must match trainer.device")
    if OmegaConf.select(config, "ray_kwargs.ray_init.runtime_env.worker_process_setup_hook") is not None:
        raise ValueError("H3 determinism requires no Ray setup hook before per-actor device binding")

    configure_h3_determinism("driver")
    names = ["H3_DETERMINISTIC", "H3_DETERMINISTIC_DEVICE", "H3_DETERMINISM_DIR"]
    if config.trainer.device == "cuda":
        names.append("CUBLAS_WORKSPACE_CONFIG")
    # Ray runs its process setup hook before assigning an actor's visible devices.
    # Importing the accelerator there caches all cards; constructors own policy setup.
    return {"env_vars": {name: os.environ[name] for name in names}}
