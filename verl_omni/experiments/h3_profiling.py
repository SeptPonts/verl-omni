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
"""Temporary H3 role-local TPA windows, with explicit RL-step identity and safe metadata."""

import gc
import hashlib
import json
import os
import socket
from contextlib import contextmanager
from importlib.metadata import version
from pathlib import Path
from types import MethodType

import torch
import torch.distributed as dist


def export_h3_profile_metadata(prof):
    """Export TPA communication/config data without its unrestricted environment/argv dump."""
    from torch_profiler_analysis.holistic_analysis.analysis_utils import get_group_info_by_parallel_type
    from torch_profiler_analysis.holistic_analysis.config_utils import export_configs

    if prof.has_exported:
        return
    output = Path(prof.config.dir_name) / "extra_info"
    output.mkdir(parents=True, exist_ok=True)
    if dist.is_initialized():
        prof.dist_info = get_group_info_by_parallel_type(strict_check=True)
        (output / f"meta_data_{prof.rank}.json").write_text(json.dumps(prof.dist_info, indent=2) + "\n")
    export_configs(prof.config, "online", prof.rank)
    prof.has_exported = True


def start_h3_profile(directory, role, rl_step, groups=(), record_shapes=True):
    """Start one role-local window on every rank, aligned after Kineto starts and before its step."""
    from torch_profiler_analysis import ProfileConfig, initialize_profiler, register_comm_group_type

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    distributed = dist.is_initialized()
    rank = dist.get_rank() if distributed else 0
    identity = {
        "role": role,
        "rl_step": rl_step,
        "local_profiler_step": 0,
        "host": socket.gethostname(),
        "pid": os.getpid(),
        "rank": rank,
        "world_size": dist.get_world_size() if distributed else 1,
        "torch_version": torch.__version__,
        "tpa_version": version("torch-profiler-analysis"),
        "record_shapes": record_shapes,
        "profile_memory": False,
        "completed": False,
    }
    # TPA treats every top-level JSON as a trace; keep our identities outside that scan.
    metadata_directory = directory / "h3_metadata"
    metadata_directory.mkdir(exist_ok=True)
    identity_path = metadata_directory / f"h3_profile_rank{rank}.json"
    # A new run/window must not silently mix with an earlier capture of the same rank.
    with identity_path.open("x") as stream:
        json.dump(identity, stream, indent=2)
        stream.write("\n")

    registered = set()
    if distributed:
        for group, group_type in ((dist.group.WORLD, "global"), *groups):
            if id(group) not in registered:
                register_comm_group_type(group, group_type, is_custom=True)
                registered.add(id(group))
    control_group = dist.new_group(backend="gloo") if distributed else None
    prof = initialize_profiler(
        config=ProfileConfig(
            step_start=0,
            step_end=1,
            profile_path=str(directory),
            rename_sub_dir=False,
            distributed_profile=distributed,
            profile_performance=True,
            profile_memory=False,
            with_stack=False,
            skip_online_analyse=True,
            enable_real_comm_time=distributed,
        )
    )
    # TPA 0.14 hardcodes True in its factory; Torch reads this public attribute in prepare_trace().
    # Set it before start(), so FSDP swap_tensors can run without profiler shape weakrefs on H20.
    prof.record_shapes = record_shapes
    # TPA 0.14's default exporter writes all os.environ and sys.argv, including possible credentials.
    # Bind only this capture instance; do not patch the installed package or unrelated profilers.
    prof._export_extra_info = MethodType(export_h3_profile_metadata, prof)
    prof.h3_control_group = control_group
    prof.h3_identity = identity
    prof.h3_identity_path = identity_path
    if distributed:
        start_trace = prof.start_trace

        def start_aligned_trace(self):
            """Exclude rank-local Kineto startup skew from the first measured device collective."""
            start_trace()
            dist.barrier(group=self.h3_control_group)

        prof.start_trace = MethodType(start_aligned_trace, prof)
    prof.start()
    return prof


def stop_h3_profile(prof, completed=True):
    """End the one complete phase, restore TPA hooks, and record its actual raw trace identity."""
    try:
        prof.step()
    finally:
        prof.stop()
        if prof.h3_control_group is not None:
            dist.destroy_process_group(prof.h3_control_group)
    trace = Path(prof.config.dir_name) / prof.file_name
    with trace.open("rb") as stream:
        trace_sha256 = hashlib.file_digest(stream, "sha256").hexdigest()
    prof.h3_identity.update(
        completed=completed,
        trace_file=trace.name,
        trace_bytes=trace.stat().st_size,
        trace_sha256=trace_sha256,
    )
    prof.h3_identity_path.write_text(json.dumps(prof.h3_identity, indent=2) + "\n")


@contextmanager
def h3_actor_profile(data, engine):
    """Capture both optimizer mini-updates as one actor-update, only for a marked comparison batch."""
    import verl.utils.tensordict_utils as tu

    directory = tu.get(data, "h3_profile_directory", default=None)
    if directory is None:
        yield
        return
    groups = [(engine.get_data_parallel_group(), "data_parallel")]
    groups.extend((group, "fsdp") for group in engine.device_mesh.get_all_groups())
    prof = start_h3_profile(
        directory,
        "actor-update",
        tu.get(data, "h3_profile_rl_step"),
        groups,
        record_shapes=tu.get(data, "h3_profile_record_shapes"),
    )
    completed = False
    try:
        with torch.profiler.record_function("h3/actor_update"):
            yield
        completed = True
    finally:
        stop_h3_profile(prof, completed=completed)
        # Profiler callbacks retain their owner cyclically; release its trace before the next RL step.
        del prof
        gc.collect()


@contextmanager
def h3_generation_profile(trainer):
    """Capture complete TP replicas and reward workers only on their selected RL steps."""
    import ray

    directory = trainer.config.trainer.get("h3_comparison_profile_dir")
    if directory is None:
        yield
        return
    windows = {
        window.role: window
        for window in trainer.config.trainer.h3_comparison_profile_windows
        if window.step == trainer.global_steps and window.role in ("rollout", "reward")
    }
    if not windows:
        yield
        return
    root = Path(directory) / f"rl_step_{trainer.global_steps:04d}"
    servers = trainer.llm_server_manager.server_handles if "rollout" in windows else []
    rewards = trainer.reward_loop_manager.reward_loop_workers if "reward" in windows else []
    starts = [
        server.collective_rpc.remote(
            "start_h3_comparison_profile",
            kwargs={
                "directory": str(root / f"rollout_replica_{replica:03d}"),
                "rl_step": trainer.global_steps,
                "record_shapes": windows["rollout"].record_shapes,
            },
        )
        for replica, server in enumerate(servers)
    ]
    starts.extend(
        worker.start_h3_comparison_profile.remote(
            str(root / f"reward_worker_{index:03d}"), trainer.global_steps, windows["reward"].record_shapes
        )
        for index, worker in enumerate(rewards)
    )
    ray.get(starts)
    completed = False
    try:
        yield
        completed = True
    finally:
        stops = [
            server.collective_rpc.remote("stop_h3_comparison_profile", kwargs={"completed": completed})
            for server in servers
        ]
        stops.extend(worker.stop_h3_comparison_profile.remote(completed) for worker in rewards)
        ray.get(stops)


def validate_h3_profiling(config):
    """Enforce the actual H3 V0 capture contract before workers or competing collectors start."""
    if config.trainer.get("h3_comparison_profile_dir") is None:
        return
    if config.trainer.use_v1 or config.actor_rollout_ref.actor.strategy != "fsdp2":
        raise ValueError("H3 TPA capture requires the V0 trainer and FSDP2")
    if config.global_profiler.tool is not None or config.global_profiler.steps is not None:
        raise ValueError("H3 TPA capture owns the collectors; disable global_profiler tool and steps")
    seen = set()
    for window in config.trainer.h3_comparison_profile_windows:
        if window.role not in ("actor", "rollout", "reward"):
            raise ValueError(f"Unsupported H3 TPA role: {window.role}")
        if not 0 < window.step <= config.trainer.total_training_steps:
            raise ValueError(f"H3 TPA {window.role} target must be a training RL step in this run")
        identity = (window.role, window.step)
        if identity in seen:
            raise ValueError(f"Duplicate H3 TPA window: {identity}")
        seen.add(identity)
    if not config.reward.custom_reward_function.path or config.reward.reward_model.enable:
        raise ValueError("H3 TPA capture expects the existing custom CLAP/ImageBind reward workers")
    if config.algorithm.adv_estimator != "flow_grpo" or config.algorithm.sample_source != "online":
        raise ValueError("H3 TPA capture requires the online FlowGRPO recipe")
