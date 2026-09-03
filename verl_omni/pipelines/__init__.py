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

import importlib
import os

PIPELINE_PACKAGES = (
    "bagel_flow_grpo",
    "boogu_image_flow_grpo",
    "ltx2_flow_grpo",
    "minimax_h3_diffusion_nft",
    "minimax_h3_flow_grpo",
    "qwen3_omni",
    "qwen_image_diffusion_nft",
    "qwen_image_dpo",
    "qwen_image_dual_grpo",
    "qwen_image_edit_flow_grpo",
    "qwen_image_flow_grpo",
    "qwen_image_mix_grpo",
    "sd3_dpo",
    "sd3_flow_grpo",
    "wan22_dance_grpo",
)

pipeline_selection = os.environ.get("VERL_OMNI_AUTO_IMPORT_PIPELINES")
selected_pipeline_packages = (
    PIPELINE_PACKAGES if pipeline_selection is None else tuple(name.strip() for name in pipeline_selection.split(","))
)

__all__ = []
for package_name in selected_pipeline_packages:
    package = importlib.import_module(f".{package_name}", __name__)
    globals()[package_name] = package
    for exported_name in package.__all__:
        globals()[exported_name] = getattr(package, exported_name)
    __all__.extend(package.__all__)
