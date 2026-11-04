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
"""Explicit, process-local callbacks for opt-in experiment recipes."""

import importlib
import os


def invoke_experiment(event: str, *args, **kwargs):
    """Call the selected provider's concrete lifecycle callback, or do nothing.

    The module is selected before Python/Ray startup. Importing it here keeps
    optional experiment dependencies out of ordinary training. Missing callbacks
    fail at their owning boundary instead of silently dropping experiment data.
    """
    module_name = os.environ.get("VERL_EXPERIMENT_MODULE")
    if module_name is None:
        return None
    provider = importlib.import_module(module_name)
    return getattr(provider, event)(*args, **kwargs)
