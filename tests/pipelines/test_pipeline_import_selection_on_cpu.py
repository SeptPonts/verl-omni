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
"""Tests for selecting the pipeline packages imported during registration."""

import os
import subprocess
import sys
import textwrap


def run_package_import(pipeline_selection: str) -> subprocess.CompletedProcess[str]:
    """Import verl-omni in a fresh interpreter with one pipeline selection."""
    env = os.environ.copy()
    env["VERL_OMNI_AUTO_IMPORT_PIPELINES"] = pipeline_selection
    program = textwrap.dedent(
        """
        import sys

        import verl_omni
        import verl_omni.pipelines as pipelines
        from verl_omni.pipelines.model_base import DiffusionModelBase, VllmOmniPipelineBase

        assert hasattr(pipelines, "minimax_h3_diffusion_nft")
        assert hasattr(pipelines, "minimax_h3_flow_grpo")
        assert "verl_omni.pipelines.boogu_image_flow_grpo" not in sys.modules
        assert DiffusionModelBase.get_class_by_name("MiniMaxH3Pipeline", "flow_grpo") is not None
        assert VllmOmniPipelineBase.get_class("MiniMaxH3Pipeline", "flow_grpo") is not None
        """
    )
    return subprocess.run(
        [sys.executable, "-c", program],
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )


def test_h3_only_selection_registers_h3_without_importing_boogu():
    """The H3 selection must not require unrelated vLLM-Omni model packages."""
    result = run_package_import("minimax_h3_diffusion_nft,minimax_h3_flow_grpo")

    assert result.returncode == 0, result.stderr


def test_unknown_selected_pipeline_fails_closed():
    """A misspelled selected package must remain an immediate import error."""
    result = run_package_import("pipeline_that_does_not_exist")

    assert result.returncode != 0
    assert "pipeline_that_does_not_exist" in result.stderr
