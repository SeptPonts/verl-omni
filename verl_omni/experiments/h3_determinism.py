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
"""Strict H3 experiment policy, applied after each process binds its devices.

Adapted from feature/h3-mlu-support-legacy. This provider owns experiment state;
the trainer and workers only expose model-independent lifecycle callbacks.
"""

import functools
import hashlib
import importlib.util
import inspect
import json
import os
from datetime import UTC, datetime
from pathlib import Path

from verl_omni.experiments.h3_evidence import (  # noqa: F401 -- provider lifecycle exports
    prepare_data,
    record_evaluation,
    record_input_batch,
    record_sampling_request,
    record_training_candidates,
    snapshot_checkpoint,
    snapshot_initial,
)
from verl_omni.experiments.h3_v0_sampling import (  # noqa: F401 -- provider lifecycle exports
    prepare_v0_validation_batch,
    record_v0_sampling_request,
)
from verl_omni.experiments.prepare_h3_adapter import (  # noqa: F401 -- provider lifecycle exports
    adapter_loaded,
    lora_loader_kwargs,
)

FA4_REVISION = "86c34f1cec40ea37e019fe79b054d9a1e189ffb2"
PROCESS_SEEDS = {}
FA4_KERNEL_SOURCES = {}

ENVIRONMENT_NAMES = (
    "VERL_EXPERIMENT_MODULE",
    "H3_DETERMINISTIC",
    "H3_DETERMINISTIC_DEVICE",
    "H3_RUN_DIR",
    "H3_ATTEMPT_DIR",
    "H3_RUNTIME_PATCH_RECEIPT",
    "H3_FA4_HUB_REVISION",
    "H3_TRAINING_MANIFEST",
    "H3_EVALUATION_MANIFEST",
    "PYTHONHASHSEED",
    "VERL_SEED",
    "VERL_FULL_DETERMINISM",
    "VLLM_BATCH_INVARIANT",
    "CUBLAS_WORKSPACE_CONFIG",
    "FLASH_ATTENTION_DETERMINISTIC",
    "NCCL_DETERMINISTIC",
    "NCCL_ALGO",
    "NCCL_PROTO",
    "VERL_DISABLE_FLASH_ATTN_CE",
    "TRITON_F32_DEFAULT",
    "TORCH_ALLOW_TF32_CUBLAS_OVERRIDE",
    "TORCH_ALLOW_TF32_CNMATMUL_OVERRIDE",
    "VERL_CLAP_BATCH_SIZE",
    "VERL_OPTIMIZER_UPDATE_TRACE_DIR",
    "VERL_FILE_LOGGER_PATH",
    "CNCL_TUNER_PLUGIN",
    "CNCL_TUNER_CONFIG_FILE",
    "HF_HOME",
    "HF_HUB_OFFLINE",
    "TRITON_CACHE_DIR",
    "PYTHONPATH",
)


def runtime_env(config):
    """Validate the selected synchronous trainer before Ray and propagate its allowlist."""
    import ray
    from omegaconf import OmegaConf

    if ray.is_initialized():
        raise ValueError("Select the experiment before ray.init(); an existing job cannot acquire its startup policy")
    if os.environ["H3_DETERMINISTIC"] != "1" or config.trainer.device != os.environ["H3_DETERMINISTIC_DEVICE"]:
        raise ValueError("H3 strict policy and trainer.device must agree")
    if config.algorithm.trainer_type != "policy_gradient":
        raise ValueError("The H3 longrun comparison requires policy-gradient training")
    if config.trainer.use_v1:
        raise ValueError("Use the V0 launcher for the historical H3 longrun comparison")
    if config.actor_rollout_ref.get("separate", False):
        raise ValueError("The H3 V0 comparison requires the historical colocated Actor/rollout topology")
    if (
        not config.actor_rollout_ref.actor.fsdp_config.full_determinism
        or not config.actor_rollout_ref.rollout.full_determinism
    ):
        raise ValueError("Enable full_determinism for both Actor and rollout")
    if config.trainer.resume_mode not in ("disable", "resume_path"):
        raise ValueError("Use an independent fresh run, or explicitly resume this run's complete checkpoint")
    if "file" not in config.trainer.logger:
        raise ValueError("Enable the native file logger for the experiment's complete scalar record")
    if config.actor_rollout_ref.actor.fsdp_config.seed != 42 or config.actor_rollout_ref.rollout.seed != 42:
        raise ValueError("Actor, rollout, reward and worker seeds must all be 42")
    validate_runtime_callback()
    if config.trainer.total_training_steps not in (2, 50) or config.data.seed != 30:
        raise ValueError("Use an independent 2-step acceptance or a fresh 50-step longrun, with data.seed=30")
    if config.actor_rollout_ref.actor.ppo_epochs != 1:
        raise ValueError("The longrun update identities require one PPO epoch")
    if config.data.val_batch_size not in (None, 128) or config.data.val_max_samples != 128:
        raise ValueError("The comparison records one complete 128-sample validation batch")
    if not config.actor_rollout_ref.model.lora_adapter_path:
        raise ValueError("Use the shared canonical adapter through model.lora_adapter_path")
    if OmegaConf.select(config, "ray_kwargs.ray_init.runtime_env.worker_process_setup_hook") is not None:
        raise ValueError("Experiment compute policy must run after device binding, not in a Ray setup hook")
    required = {
        "PYTHONHASHSEED": "42",
        "VERL_SEED": "42",
        "TRITON_F32_DEFAULT": "ieee",
        "TORCH_ALLOW_TF32_CUBLAS_OVERRIDE": "0",
        "TORCH_ALLOW_TF32_CNMATMUL_OVERRIDE": "0",
        "VERL_CLAP_BATCH_SIZE": "1",
    }
    if any(os.environ.get(name) != value for name, value in required.items()):
        raise ValueError(f"Start the recipe with these strict values: {required}")
    values = {name: os.environ[name] for name in ENVIRONMENT_NAMES if name in os.environ}
    explicit = OmegaConf.select(config, "ray_kwargs.ray_init.runtime_env.env_vars", default={})
    for name, value in values.items():
        if name in explicit and str(explicit[name]) != value:
            raise ValueError(f"Ray override conflicts with experiment environment: {name}")
    root = Path(os.environ["H3_ATTEMPT_DIR"])
    root.mkdir(parents=True, exist_ok=True)
    (root / "resolved-config.yaml").write_text(OmegaConf.to_yaml(config, resolve=True))
    return {"env_vars": values}


def validate_runtime_callback():
    """Resolve the actual installed worker without importing it before device assignment."""
    from verl_omni.experiments.h3_runtime_patch import verify_worker_callback

    receipt = os.environ.get("H3_RUNTIME_PATCH_RECEIPT")
    if receipt is None:
        raise ValueError("Install h3_runtime_patch and set H3_RUNTIME_PATCH_RECEIPT before selecting this experiment")
    package = importlib.util.find_spec("vllm_omni")
    worker = Path(package.origin).parent / "diffusion/worker/diffusion_worker.py"
    return verify_worker_callback(receipt, worker)


def configure_reward_process(config):
    """Bind the declared primary scorer device before initializing the custom-scorer process."""
    import torch
    from verl.utils.device import get_torch_device

    scorers = config.reward.reward_functions
    devices = [torch.device(scorers[name].device) for name in ("clap", "imagebind")]
    backend = get_torch_device()
    if any(device.type != os.environ["H3_DETERMINISTIC_DEVICE"] for device in devices):
        raise ValueError("CLAP/ImageBind scorer devices differ from the experiment platform")
    if any(device.index is None or device.index >= backend.device_count() for device in devices):
        raise ValueError("CLAP/ImageBind scorer device mapping is not visible in this reward worker")
    backend.set_device(devices[0])
    state = configure_compute_process("reward", config.actor_rollout_ref.rollout.seed)
    state["scorer_devices"] = [str(device) for device in devices]
    path = Path(os.environ["H3_ATTEMPT_DIR"]) / "determinism" / f"{os.getpid()}-reward.json"
    path.write_text(json.dumps(state, indent=2) + "\n")
    return state


def configure_fa4(role):
    """Pin cached Actor FA4 and make both rollout entrypoints explicitly deterministic."""
    identity = (os.getpid(), role)
    if identity in FA4_KERNEL_SOURCES:
        return FA4_KERNEL_SOURCES[identity]
    if role == "actor":
        import diffusers.models.attention_dispatch as attention

        config = attention._HUB_KERNELS_REGISTRY[attention.AttentionBackendName.FLASH_4_HUB]
        config.revision = os.environ["H3_FA4_HUB_REVISION"]
        if config.revision != FA4_REVISION:
            raise ValueError("Use the frozen FA4 Hub revision for this comparison")
        config.version = None
        with attention.attention_backend("flash_4_hub"):
            kernel = config.kernel_fn
            config.kernel_fn = functools.partial(kernel, deterministic=True)
        paths = {"actor_fa4": inspect.getfile(kernel)}
    elif role == "diffusion-worker":
        from vllm_omni.diffusion.attention.backends.utils import fa

        # Native FA discovery must have selected the PV-backed CuTe package.
        if fa.flash_attn_func.__module__ != "flash_attn.cute.interface":
            raise ValueError("The experiment requires the frozen CuTe FA4 package for FLASH_ATTN rollout")
        paths = {"rollout_fa4": inspect.getfile(fa.flash_attn_func)}
        fa.flash_attn_func = functools.partial(fa.flash_attn_func, deterministic=True)
        fa.flash_attn_varlen_func = functools.partial(fa.flash_attn_varlen_func, deterministic=True)
    else:
        return {}
    sources = {
        name: {"path": path, "sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest()}
        for name, path in paths.items()
    }
    FA4_KERNEL_SOURCES[identity] = sources
    return sources


def configure_compute_process(role, seed, *, seeded=False):
    """Apply strict arithmetic after binding; the existing owner initializes RNG at most once."""
    import torch
    from verl.utils.device import get_device_name, get_torch_device, is_cuda_available
    from verl.workers.engine.utils import enable_full_determinism

    seed = int(seed)
    device = os.environ["H3_DETERMINISTIC_DEVICE"]
    if get_device_name() != device:
        raise ValueError("Compute process platform differs from the experiment contract")
    for name, expected in (
        ("TRITON_F32_DEFAULT", "ieee"),
        ("TORCH_ALLOW_TF32_CUBLAS_OVERRIDE", "0"),
        ("TORCH_ALLOW_TF32_CNMATMUL_OVERRIDE", "0"),
    ):
        if os.environ[name] != expected:
            raise ValueError(f"Compute-process precision override differs from the contract: {name}")
    process = os.getpid()
    if process not in PROCESS_SEEDS:
        if not seeded:
            enable_full_determinism(seed=seed)
        PROCESS_SEEDS[process] = seed
    elif PROCESS_SEEDS[process] != seed:
        raise ValueError("A compute process cannot acquire a different experiment seed")
    torch.set_float32_matmul_precision("highest")
    torch.use_deterministic_algorithms(True, warn_only=False)
    backend = getattr(torch.backends, device)
    backend.matmul.allow_tf32 = False
    if device == "mlu":
        torch.backends.mlu.custom.allow_tf32 = False
        convolution = torch.backends.cnnl
        convolution.allow_tf32 = False
        convolution.deterministic = True
        kernels = {}
    elif is_cuda_available:
        convolution = torch.backends.cudnn
        convolution.enabled = False
        convolution.allow_tf32 = False
        convolution.deterministic = True
        convolution.benchmark = False
        kernels = configure_fa4(role)
    else:
        raise ValueError("The H3 longrun recipe requires an accelerator")
    state = {
        "observed_at": datetime.now(UTC).isoformat(),
        "role": role,
        "pid": os.getpid(),
        "parent_pid": os.getppid(),
        "seed": seed,
        "device": device,
        "device_index": get_torch_device().current_device(),
        "visible_devices": {
            name: os.environ.get(name) for name in ("CUDA_VISIBLE_DEVICES", "MLU_VISIBLE_DEVICES", "CN_VISIBLE_DEVICES")
        },
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "warn_only": torch.is_deterministic_algorithms_warn_only_enabled(),
        "matmul_allow_tf32": backend.matmul.allow_tf32,
        "convolution_allow_tf32": convolution.allow_tf32,
        "convolution_deterministic": convolution.deterministic,
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "policy_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "kernels": kernels,
        "environment": {name: os.environ[name] for name in ENVIRONMENT_NAMES if name in os.environ},
    }
    if device == "mlu":
        state["custom_allow_tf32"] = torch.backends.mlu.custom.allow_tf32
    directory = Path(os.environ["H3_ATTEMPT_DIR"]) / "determinism"
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"{os.getpid()}-{role}.json").write_text(json.dumps(state, indent=2) + "\n")
    return state
