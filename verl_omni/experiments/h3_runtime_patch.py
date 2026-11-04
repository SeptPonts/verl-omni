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
"""Install one explicit vLLM-Omni worker callback after device binding."""

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
from pathlib import Path

DEVICE_ANCHOR = "        current_omni_platform.set_device(self.device)\n"
WORKER_CALLBACK = """
        # Experiment initialization belongs after the actual worker binds its device.
        if os.environ.get("VERL_EXPERIMENT_MODULE") is not None:
            from verl_omni.pipelines.experiment_hooks import invoke_experiment

            invoke_experiment("configure_compute_process", "diffusion-worker", int(os.environ["VERL_SEED"]))
"""


def patched_worker_source(source):
    """Insert only the observed device-bound callback; reject a different source layout."""
    if source.count(DEVICE_ANCHOR) != 1:
        raise ValueError("Installed diffusion_worker.py does not match the reviewed device-binding boundary")
    if source.count(DEVICE_ANCHOR + WORKER_CALLBACK) == 1:
        return source
    if WORKER_CALLBACK in source:
        raise ValueError("Installed diffusion_worker.py does not match the reviewed device-binding boundary")
    return source.replace(DEVICE_ANCHOR, DEVICE_ANCHOR + WORKER_CALLBACK)


def install_worker_callback(worker_path, receipt_dir):
    """Keep source identity and an exact reverse patch outside the installed package."""
    worker_path = Path(worker_path)
    receipt_dir = Path(receipt_dir)
    before = worker_path.read_text()
    after = patched_worker_source(before)
    compile(after, str(worker_path), "exec")
    before_hash = hashlib.sha256(before.encode()).hexdigest()
    after_hash = hashlib.sha256(after.encode()).hexdigest()
    receipt = {
        "worker_path": str(worker_path),
        "before_sha256": before_hash,
        "after_sha256": after_hash,
        "callback": "after current_omni_platform.set_device; before distributed/model initialization",
        "changed": before != after,
    }
    receipt_dir.mkdir(parents=True, exist_ok=True)
    if before != after:
        # This is the exact installed source needed to undo this explicit runtime mutation.
        (receipt_dir / f"diffusion_worker.{before_hash}.py").write_text(before)
        worker_path.write_text(after)
    suffix = "installed" if before != after else "observed"
    receipt_path = receipt_dir / f"worker-callback.{after_hash}.{suffix}.json"
    if not receipt_path.exists():
        receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt


def verify_worker_callback(receipt_path, worker_path):
    """Authenticate the selected interpreter's worker against its installed patch receipt."""
    worker_path = Path(worker_path).resolve()
    receipt = json.loads(Path(receipt_path).read_text())
    source = worker_path.read_text()
    if Path(receipt["worker_path"]).resolve() != worker_path:
        raise ValueError("Runtime callback receipt belongs to a different installed worker")
    if hashlib.sha256(source.encode()).hexdigest() != receipt["after_sha256"]:
        raise ValueError("Installed worker changed after its runtime callback receipt")
    if patched_worker_source(source) != source:
        raise ValueError("Install the device-bound worker callback before starting the experiment")
    return receipt


def main():
    """Patch only the selected interpreter's installed vLLM-Omni after an explicit command."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt-dir", required=True, type=Path)
    args = parser.parse_args()
    package = importlib.util.find_spec("vllm_omni")
    path = Path(package.origin).parent / "diffusion/worker/diffusion_worker.py"
    receipt = install_worker_callback(path, args.receipt_dir)
    receipt["package_version"] = importlib.metadata.version("vllm-omni")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
