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
"""Keep one verified persistent H3 checkpoint while training saves to the local disk."""

import hashlib
import json
import subprocess
from pathlib import Path


def backup_h3_checkpoint(source, backup_root, step):
    """Replace the owned single-generation mirror and publish only after full SHA256 verification.

    The persistent volume cannot hold two complete generations. The previous
    generation becomes unavailable during the in-place transfer; a failed copy
    stays explicitly incomplete. The synchronous trainer retains its complete
    local checkpoint and does not advance past this call on failure.
    """
    source = Path(source).resolve()
    root = Path(backup_root).resolve()
    if root.is_relative_to(source) or source.is_relative_to(root):
        raise ValueError("H3 checkpoint source and persistent backup must be disjoint directories")
    if source.name != f"global_step_{step}" or not (source / "data.pt").is_file():
        raise ValueError("H3 backup requires a completed global_step directory including data.pt")
    root.mkdir(parents=True, exist_ok=True)
    receipt_path = root / "backup_manifest.json"
    staging = root / "checkpoint.incomplete"
    destination = root / f"global_step_{step}"
    previous = json.loads(receipt_path.read_text()) if receipt_path.exists() else None
    if previous is None and any(root.iterdir()):
        raise ValueError("A new H3 backup root must be empty; do not reuse unrelated directories")
    if previous is not None and previous["format"] != "h3-checkpoint-backup-v1":
        raise ValueError("H3 backup root belongs to a different format")
    if previous is not None and previous["status"] == "COMPLETE":
        if step <= previous["step"]:
            raise ValueError("H3 backup must advance beyond the completed persistent checkpoint")
        previous_checkpoint = f"global_step_{previous['step']}"
    elif previous is not None:
        if step != previous["step"]:
            raise ValueError("Finish the incomplete H3 backup before advancing to another checkpoint")
        previous_checkpoint = previous["previous_checkpoint"]
    else:
        previous_checkpoint = None

    receipt = {
        "format": "h3-checkpoint-backup-v1",
        "status": "INCOMPLETE",
        "step": step,
        "source": str(source),
        "destination": str(destination),
        "previous_checkpoint": previous_checkpoint,
        "policy": "one generation; previous backup unavailable during transfer",
    }
    pending_receipt = root / "backup_manifest.pending.json"
    pending_receipt.write_text(json.dumps(receipt, indent=2) + "\n")
    pending_receipt.replace(receipt_path)
    # A crash can precede the old-directory rename or follow the new-directory rename.
    # Reuse the single owned generation in every case; never allocate a second full copy.
    candidates = [path for path in (staging, destination) if path.exists()]
    if previous_checkpoint is not None and (root / previous_checkpoint).exists():
        candidates.append(root / previous_checkpoint)
    if len(candidates) > 1:
        raise ValueError("H3 backup has multiple generations; inspect them before retrying")
    if candidates and candidates[0] != staging:
        candidates[0].rename(staging)
    else:
        staging.mkdir(exist_ok=True)
    # Only this run's owned mirror is replaced. --inplace avoids a second 64-GiB generation.
    subprocess.run(["rsync", "-a", "--inplace", "--ignore-times", "--delete", f"{source}/", f"{staging}/"], check=True)
    source_files = sorted(path.relative_to(source) for path in source.rglob("*") if path.is_file())
    copied_files = sorted(path.relative_to(staging) for path in staging.rglob("*") if path.is_file())
    if copied_files != source_files:
        raise ValueError("H3 persistent checkpoint file inventory differs from its complete local source")
    files = {}
    for relative in source_files:
        source_file = source / relative
        copied_file = staging / relative
        with source_file.open("rb") as stream:
            expected = hashlib.file_digest(stream, "sha256").hexdigest()
        with copied_file.open("rb") as stream:
            actual = hashlib.file_digest(stream, "sha256").hexdigest()
        if source_file.stat().st_size != copied_file.stat().st_size or expected != actual:
            raise ValueError(f"H3 persistent checkpoint checksum mismatch: {relative}")
        files[str(relative)] = {"bytes": copied_file.stat().st_size, "sha256": actual}
    staging.rename(destination)
    receipt.update(status="COMPLETE", files=files)
    pending_receipt.write_text(json.dumps(receipt, indent=2) + "\n")
    pending_receipt.replace(receipt_path)
    print(f"H3 persistent checkpoint verified: {destination}", flush=True)
