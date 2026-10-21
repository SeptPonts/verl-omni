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
"""Resolve the actual paired profile launcher arguments without importing a training runtime."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

ROOT = Path(__file__).parents[2]


@pytest.mark.parametrize("platform", ["mlu", "cuda"])
def test_profile_launcher_resolves_same_steps_and_bounded_requests(tmp_path, platform):
    """Capture shell argv, then compose real YAML to check the complete profile recipe on both backends."""
    binary_dir = tmp_path / "bin"
    binary_dir.mkdir()
    python = binary_dir / "python3"
    arguments_path = tmp_path / "arguments.json"
    python.write_text(
        f"#!{sys.executable}\n"
        "import json, os, pathlib, sys\n"
        "pathlib.Path(os.environ['H3_TEST_ARGUMENTS']).write_text(json.dumps(sys.argv[1:]))\n"
    )
    python.chmod(0o755)
    env = {
        **os.environ,
        "PATH": f"{binary_dir}{os.pathsep}{os.environ['PATH']}",
        "H3_TEST_ARGUMENTS": str(arguments_path),
        "H3_RUN_DIR": str(tmp_path / "run"),
        "H3_INITIAL_LORA": str(tmp_path / "canonical-v2"),
        "H3_EVALUATION_MANIFEST": str(tmp_path / "evaluation.json"),
        "H3_TRAINING_MANIFEST": str(tmp_path / "training.json"),
        "H3_CHECKPOINT_BACKUP_DIR": "",
        "H3_RESUME_FROM": "",
    }
    subprocess.run(
        ["bash", "verl_omni/experiments/run_h3_comparison.sh", platform, "profile", "resolve"],
        cwd=ROOT,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    arguments = json.loads(arguments_path.read_text())
    assert arguments[:3] == ["-", "-m", "verl_omni.trainer.main_diffusion"]
    assert arguments[-3:] == ["--cfg", "job", "--resolve"]
    with initialize_config_dir(config_dir=str(ROOT / "verl_omni/trainer/config"), version_base=None):
        config = compose(config_name="diffusion_trainer", overrides=arguments[3:-3])
    OmegaConf.resolve(config)
    assert config.trainer.total_training_steps == 4
    assert config.trainer.use_v1 is False
    assert config.trainer.test_freq == config.trainer.save_freq == -1
    assert config.global_profiler.tool is config.global_profiler.steps is None
    assert config.actor_rollout_ref.actor.fsdp_config.param_offload is True
    assert config.actor_rollout_ref.rollout.tensor_model_parallel_size == 2
    assert config.actor_rollout_ref.rollout.step_execution is False
    assert OmegaConf.to_container(config.trainer.h3_comparison_profile_windows) == [
        {"role": "actor", "step": 2, "record_shapes": True},
        {"role": "rollout", "step": 2, "record_shapes": True, "replica": 0, "request_start": 3, "request_end": 4},
        {"role": "actor", "step": 3, "record_shapes": True},
        {"role": "rollout", "step": 3, "record_shapes": True, "replica": 0, "request_start": 3, "request_end": 4},
    ]
