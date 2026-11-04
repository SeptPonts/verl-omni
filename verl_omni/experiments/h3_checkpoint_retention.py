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
"""Audit full native H3 saves before retaining this new run's latest checkpoint.

State checks are adapted from the context repository's
docs/h3-group-vs-bs1-quality50-20261025/retain_checkpoint.py. Call synchronously
after LoRA export and the fixed128 evaluation ledger. CPU unpickling and local
tensor inspection do not establish distributed restore acceptance.
"""

import hashlib
import json
import os
import random
import shutil
import stat
import zipfile
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path


def owned_path(value: str | Path) -> Path:
    """Reject aliases, including symlink ancestors, before accessing owned storage."""
    path = Path(value).expanduser()
    if not path.is_absolute() or ".." in path.parts or any(part.is_symlink() for part in (path, *path.parents)):
        raise ValueError(f"H3 retention requires an absolute path without symlinks: {path}")
    return path


def checkpoint_files(directory: Path) -> list[Path]:
    """Inventory a checkpoint tree without accepting symlinks or special files."""
    files = []
    for path in sorted(directory.rglob("*")):
        owned_path(path)
        mode = path.stat().st_mode
        if stat.S_ISREG(mode):
            files.append(path)
        elif not stat.S_ISDIR(mode):
            raise ValueError(f"H3 checkpoint contains a special file: {path}")
    return files


def file_identity(path: Path) -> dict:
    """Record identity as well as size so replacement during the audit is rejected."""
    info = path.stat()
    return {
        "dev": info.st_dev,
        "ino": info.st_ino,
        "size": info.st_size,
        "mtime_ns": info.st_mtime_ns,
        "ctime_ns": info.st_ctime_ns,
    }


def hash_file(path: Path) -> dict:
    """Hash every byte and reject a file modified while its hash was being read."""
    owned_path(path)
    before = file_identity(path)
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    if file_identity(path) != before:
        raise ValueError(f"H3 file changed during hashing: {path}")
    return {**before, "sha256": digest}


def audit_native_checkpoint(saved: Path, step: int, backend: str) -> dict:
    """Check Adam/scheduler/RNG and loader state on CPU; model archives are hashed separately."""
    import numpy as np
    import torch
    from torch.distributed.tensor import DTensor

    loader = torch.load(saved / "data.pt", map_location="cpu", weights_only=False)
    snapshot = loader["_snapshot"]
    main = snapshot["_main_snapshot"]
    if (
        snapshot["_snapshot_step"] != step
        or loader["_steps_since_snapshot"] != 0
        or main["_sampler_iter_yielded"] != step
        or main["_sampler_iter_state"]["samples_yielded"] != step * 32
    ):
        raise ValueError("H3 data state does not match the saved global step and batch32 cursor")
    ranks = []
    for rank in range(8):
        optim = torch.load(
            saved / "actor" / f"optim_world_size_8_rank_{rank}.pt", map_location="cpu", weights_only=False
        )
        active = [value for value in optim["state"].values() if value]
        if not active or not optim["param_groups"]:
            raise ValueError(f"H3 rank {rank} Adam state is empty")
        parameter_ids = [key for group in optim["param_groups"] for key in group["params"]]
        if len(set(parameter_ids)) != len(parameter_ids) or set(optim["state"]) - set(parameter_ids):
            raise ValueError(f"H3 rank {rank} Adam parameter ownership is inconsistent")
        for value in active:
            if not {"step", "exp_avg", "exp_avg_sq"} <= value.keys() or value["step"] != 2 * step:
                raise ValueError(f"H3 rank {rank} Adam state is incomplete or has the wrong step")
            moments = []
            for name in ("exp_avg", "exp_avg_sq"):
                tensor = value[name]
                local = tensor.to_local() if isinstance(tensor, DTensor) else tensor
                if local.device.type != "cpu" or not torch.isfinite(local).all():
                    raise ValueError(f"H3 rank {rank} Adam {name} is not finite on CPU")
                moments.append(local)
            if moments[0].shape != moments[1].shape or (moments[1] < 0).any():
                raise ValueError(f"H3 rank {rank} Adam moments have invalid shapes or second moments")
        extra = torch.load(
            saved / "actor" / f"extra_state_world_size_8_rank_{rank}.pt", map_location="cpu", weights_only=False
        )
        scheduler = extra["lr_scheduler"]
        if (
            scheduler["last_epoch"] != step
            or scheduler["_last_lr"] != [0.0003]
            or any(group["lr"] != 0.0003 for group in optim["param_groups"])
        ):
            raise ValueError(f"H3 rank {rank} scheduler state disagrees with the saved step/Adam learning rates")
        rng = extra["rng"]
        if set(rng) != {"cpu", backend, "numpy", "random"}:
            raise ValueError(f"H3 rank {rank} RNG state is incomplete")
        if any(rng[key].dtype != torch.uint8 or not rng[key].numel() for key in ("cpu", backend)):
            raise ValueError(f"H3 rank {rank} tensor RNG state is empty or malformed")
        # Validate CPU/Python/NumPy state without changing the trainer's live generators.
        torch.Generator(device="cpu").set_state(rng["cpu"])
        np.random.RandomState().set_state(rng["numpy"])
        random.Random().setstate(rng["random"])
        ranks.append({"rank": rank, "optimizer_step": 2 * step, "scheduler_step": step, "rng_present": True})
        del optim, extra, active, value, moments, tensor, local
    return {
        "rank_states": ranks,
        "loader_step": step,
        "model_validation": "ARCHIVE_AND_SHA256_ONLY",
        "distributed_restore": "NOT_RUN",
    }


def persist_json(path: Path, value: dict) -> None:
    """Exclusively persist a complete receipt before it can authorize deletion."""
    owned_path(path)
    with path.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def audit_and_retain_checkpoint(config, global_step: int, actor_checkpoint_path: str | Path) -> dict:
    """Verify this run's complete save/eval/LoRA, publish CPU evidence, then delete its prior save.

    Args:
        config: Resolved H3 config with native pruning disabled and train batch32.
        global_step: Positive completed save/evaluation step.
        actor_checkpoint_path: Exact H3_CHECKPOINT_DIR/global_step_N/actor path.

    Returns:
        The persisted CPU audit path and the previous checkpoint paths removed.

    Raises:
        ValueError: Ownership, state, evaluation, or adapter verification fails.
        OSError: Evidence cannot be read/persisted or retention cannot complete.
    """
    from verl_omni.experiments.h3_evidence import adapter_evidence, checked_rewards, finite_score

    if type(global_step) is not int or global_step <= 0:
        raise ValueError("H3 retention requires a positive saved global step")
    if config.trainer.max_actor_ckpt_to_keep is not None or config.trainer.get("remove_previous_ckpt_in_save", False):
        raise ValueError("Disable native premature checkpoint pruning before H3 retention")
    if config.data.train_batch_size != 32:
        raise ValueError("H3 retention requires the frozen train batch32 cursor")
    root = owned_path(os.environ["H3_CHECKPOINT_DIR"])
    run = owned_path(os.environ["H3_RUN_DIR"])
    attempt = owned_path(os.environ["H3_ATTEMPT_DIR"])
    saved = root / f"global_step_{global_step}"
    actor = owned_path(actor_checkpoint_path)
    if owned_path(config.trainer.default_local_dir) != root or actor != saved / "actor" or not actor.is_dir():
        raise ValueError("H3 checkpoint path differs from this run's exact configured checkpoint root")
    if run.is_relative_to(root) or root.is_relative_to(run) or not attempt.is_relative_to(run) or attempt == run:
        raise ValueError("H3 run evidence and checkpoints must be disjoint, with an attempt inside the run")
    owned_path(saved)
    latest = owned_path(root / "latest_checkpointed_iteration.txt")
    if latest.read_text().strip() != str(global_step):
        raise ValueError("H3 native full-save completion marker differs from the requested step")
    audit_dir = owned_path(run / "checkpoint-audits")
    marker = owned_path(audit_dir / "verified.json")
    identity = {"run": str(run), "checkpoint_root": str(root)}
    previous = []
    previous_files = {}
    if marker.exists():
        verified = json.loads(marker.read_text())
        if verified["owner"] != identity or verified["global_step"] >= global_step:
            raise ValueError("H3 verified marker belongs to a foreign run or a nonprevious checkpoint")
        previous = [root / f"global_step_{verified['global_step']}"]
        old_receipt = owned_path(audit_dir / f"step{verified['global_step']}.json")
        if hash_file(old_receipt)["sha256"] != verified["audit_sha256"]:
            raise ValueError("H3 previous checkpoint audit receipt no longer matches its verified marker")
        old_audit = json.loads(old_receipt.read_text())
        if (
            old_audit["owner"] != identity
            or old_audit["path"] != str(previous[0])
            or old_audit["global_step"] != verified["global_step"]
            or old_audit["status"] != "PASS_CPU_ARCHIVE_AND_STATE"
        ):
            raise ValueError("H3 previous checkpoint does not match its owned CPU audit")
        previous_files = {previous[0] / name: record for name, record in old_audit["files"].items()}
        if set(checkpoint_files(previous[0])) != previous_files.keys():
            raise ValueError("H3 previous checkpoint inventory changed since its owned audit")
    candidates = sorted(root.glob("global_step_*"))
    if set(candidates) != {saved, *previous}:
        raise ValueError("H3 retention refuses unowned, misnamed, or out-of-order checkpoint directories")
    for path in candidates:
        owned_path(path)
        if not path.is_dir():
            raise ValueError(f"H3 checkpoint is not a real directory: {path}")
        checkpoint_files(path)

    evaluation_path = owned_path(attempt / f"evaluation-{global_step}.jsonl")
    manifest_path = owned_path(run / "manifests" / "evaluation.json")
    snapshot_path = owned_path(attempt / "lora-snapshots" / f"step{global_step}.json")
    evidence_files = {path: hash_file(path) for path in (evaluation_path, manifest_path, snapshot_path, latest)}
    rows = [json.loads(line) for line in evaluation_path.read_text().splitlines()]
    samples = json.loads(manifest_path.read_text())["samples"]
    if len(rows) != 128 or len(samples) != 128 or len({row["sample_id"] for row in rows}) != 128:
        raise ValueError("H3 retention requires the complete fixed128 evaluation ledger")
    checked_rewards([row["components"] for row in rows], 128)
    for index, (row, sample) in enumerate(zip(rows, samples, strict=True)):
        if (
            any(row[key] != value for key, value in sample.items())
            or row["global_step"] != global_step
            or row["prompt_index"] != index
            or row["session_id"] != 0
        ):
            raise ValueError("H3 evaluation ledger differs from its frozen sample identities or step")
        finite_score(row["total"], "H3 checkpoint evaluation total")
    snapshot = json.loads(snapshot_path.read_text())
    output = owned_path(snapshot["output"])
    if snapshot["global_step"] != global_step or snapshot["source"] != str(actor) or not output.is_relative_to(run):
        raise ValueError("H3 LoRA receipt must identify this checkpoint and an export on run evidence storage")
    evidence_files.update({path: hash_file(path) for path in checkpoint_files(output)})
    adapter = adapter_evidence(output, "adapter_model.safetensors")
    if any(snapshot[key] != value for key, value in adapter.items()):
        raise ValueError("H3 LoRA export differs from its verified tensor receipt")

    files = checkpoint_files(saved)
    archives = [saved / "data.pt"] + [
        actor / f"{kind}_world_size_8_rank_{rank}.pt" for kind in ("model", "optim", "extra_state") for rank in range(8)
    ]
    if set(actor.glob("*_world_size_*_rank_*.pt")) != set(archives[1:]):
        raise ValueError("H3 checkpoint must contain exactly eight model/optim/extra_state rank shards")
    with ThreadPoolExecutor(max_workers=4) as pool:
        hashes = dict(zip(files, pool.map(hash_file, files), strict=True))
    if not set(archives) <= hashes.keys():
        raise ValueError("H3 full checkpoint is missing data or rank state")
    for path in archives:
        with zipfile.ZipFile(path) as archive:
            if not archive.namelist():
                raise ValueError(f"H3 native checkpoint archive is empty: {path}")
    state = audit_native_checkpoint(saved, global_step, config.trainer.device)
    if checkpoint_files(saved) != files:
        raise ValueError("H3 checkpoint inventory changed during CPU audit")
    for path, record in (hashes | evidence_files | previous_files).items():
        if file_identity(owned_path(path)) != {key: value for key, value in record.items() if key != "sha256"}:
            raise ValueError(f"H3 audited file changed before retention: {path}")
    receipt = {
        "status": "PASS_CPU_ARCHIVE_AND_STATE",
        "owner": identity,
        "global_step": global_step,
        "path": str(saved),
        "attempt": str(attempt),
        "verified_at": datetime.now(UTC).isoformat(),
        "files": {str(path.relative_to(saved)): value for path, value in hashes.items()},
        "evidence_files": {str(path): value for path, value in evidence_files.items()},
        "validation_rows": 128,
        "previous_checkpoints": [str(path) for path in previous],
        **state,
    }
    audit_dir.mkdir(exist_ok=True)
    receipt_path = audit_dir / f"step{global_step}.json"
    persist_json(receipt_path, receipt)
    pending = audit_dir / "verified.next.json"
    persist_json(
        pending, {"owner": identity, "global_step": global_step, "audit_sha256": hash_file(receipt_path)["sha256"]}
    )
    pending.replace(marker)
    directory_fd = os.open(audit_dir, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)
    for path in previous:
        owned_path(path)
        checkpoint_files(path)
        shutil.rmtree(path)
    return {
        "status": "VERIFIED_LATEST_ONE_RETAINED",
        "audit_receipt": str(receipt_path),
        "removed": receipt["previous_checkpoints"],
    }
