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
"""Verify real small-file checkpoint transfers, publication failures, and single-generation retries."""

import hashlib
import importlib.util
import json
import subprocess
from pathlib import Path

import pytest

SOURCE = Path(__file__).parents[2] / "verl_omni/experiments/h3_checkpoint_backup.py"
SPEC = importlib.util.spec_from_file_location("h3_checkpoint_backup", SOURCE)
API = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(API)


def make_checkpoint(root, step):
    """Write distinct model/optimizer/data bytes with the real checkpoint directory shape."""
    path = root / f"global_step_{step}"
    (path / "actor").mkdir(parents=True)
    (path / "data.pt").write_bytes(f"data position {step}".encode())
    for prefix in ("model", "optim", "extra_state"):
        for rank in range(2):
            (path / "actor" / f"{prefix}_world_size_2_rank_{rank}.pt").write_bytes(
                f"{prefix} {rank} {step}".encode() * 100
            )
    return path


def test_verified_replacement_retains_exactly_one_generation(tmp_path):
    """Real rsync updates the mirror and removes a superseded generation without losing local state."""
    root = tmp_path / "persistent"
    for step in (10, 20):
        source = make_checkpoint(tmp_path / "local", step)
        API.backup_h3_checkpoint(source, root, step)
        receipt = json.loads((root / "backup_manifest.json").read_text())
        assert receipt["status"] == "COMPLETE" and receipt["step"] == step
        assert sorted(path.name for path in root.iterdir()) == ["backup_manifest.json", f"global_step_{step}"]
        for name, record in receipt["files"].items():
            assert (source / name).read_bytes() == (root / f"global_step_{step}" / name).read_bytes()
            assert record["sha256"] == hashlib.sha256((source / name).read_bytes()).hexdigest()
    assert (tmp_path / "local/global_step_10/data.pt").exists()


@pytest.mark.parametrize("failure", ["copy", "checksum", "before_incomplete", "after_rename"])
def test_failures_do_not_publish_and_retry_reuses_single_generation(tmp_path, monkeypatch, failure):
    """Interrupt each IO boundary, retain truthful status, and recover without allocating a second copy."""
    root = tmp_path / "persistent"
    API.backup_h3_checkpoint(make_checkpoint(tmp_path / "local", 10), root, 10)
    source = make_checkpoint(tmp_path / "local", 20)
    original_run = subprocess.run
    original_replace = Path.replace

    def transfer_with_fault(args, **kwargs):
        """Inject a transfer failure or corrupt copied bytes after actual rsync completes."""
        original_run(args, **kwargs)
        if failure == "copy":
            raise subprocess.CalledProcessError(23, args)
        (root / "checkpoint.incomplete/data.pt").write_bytes(b"corrupted")

    def publish_with_fault(path, target):
        """Interrupt atomic receipt publication before replacing its previous truthful status."""
        if path.name == "backup_manifest.pending.json":
            status = json.loads(path.read_text())["status"]
            if (failure == "before_incomplete" and status == "INCOMPLETE") or (
                failure == "after_rename" and status == "COMPLETE"
            ):
                raise OSError("injected publication failure")
        return original_replace(path, target)

    with monkeypatch.context() as patch:
        if failure in ("copy", "checksum"):
            patch.setattr(API.subprocess, "run", transfer_with_fault)
        else:
            patch.setattr(Path, "replace", publish_with_fault)
        with pytest.raises((OSError, ValueError, subprocess.CalledProcessError)):
            API.backup_h3_checkpoint(source, root, 20)
    receipt = json.loads((root / "backup_manifest.json").read_text())
    assert receipt["status"] == ("COMPLETE" if failure == "before_incomplete" else "INCOMPLETE")
    if failure == "before_incomplete":
        assert receipt["step"] == 10 and (root / "global_step_10/data.pt").exists()

    def transfer_existing_generation(args, **kwargs):
        """The retry must reuse existing bytes before invoking rsync, even after the final rename."""
        assert [path.name for path in root.iterdir() if path.is_dir()] == ["checkpoint.incomplete"]
        assert (root / "checkpoint.incomplete/actor/model_world_size_2_rank_0.pt").is_file()
        return original_run(args, **kwargs)

    monkeypatch.setattr(API.subprocess, "run", transfer_existing_generation)
    API.backup_h3_checkpoint(source, root, 20)
    assert json.loads((root / "backup_manifest.json").read_text())["status"] == "COMPLETE"
    assert (root / "global_step_20/data.pt").read_bytes() == (source / "data.pt").read_bytes()


def test_reject_unowned_root_and_overlapping_paths(tmp_path):
    """Do not let rsync delete unrelated files or copy a checkpoint into itself."""
    source = make_checkpoint(tmp_path / "local", 10)
    root = tmp_path / "unrelated"
    root.mkdir()
    sentinel = root / "keep.txt"
    sentinel.write_text("preserve")
    with pytest.raises(ValueError, match="empty"):
        API.backup_h3_checkpoint(source, root, 10)
    with pytest.raises(ValueError, match="disjoint"):
        API.backup_h3_checkpoint(source, source / "backup", 10)
    assert sentinel.read_text() == "preserve"
