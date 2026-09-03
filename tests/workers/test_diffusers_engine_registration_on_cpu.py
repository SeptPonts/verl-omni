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
"""CPU tests for accelerator-specific diffusion engine registration."""

from verl.workers.engine.base import EngineRegistry

from verl_omni.workers.engine.fsdp.diffusers_impl import PPODiffusersFSDPEngine


def test_ppo_diffusers_fsdp2_engine_is_registered_for_mlu(monkeypatch):
    """FlowGRPO must resolve the generic Diffusers FSDP2 engine on MLU."""
    monkeypatch.setenv("VERL_ENGINE_DEVICE", "mlu")
    monkeypatch.setenv("VERL_ENGINE_VENDOR", "cambricon")

    assert EngineRegistry.get_engine_cls("diffusion_model", "fsdp2") is PPODiffusersFSDPEngine
