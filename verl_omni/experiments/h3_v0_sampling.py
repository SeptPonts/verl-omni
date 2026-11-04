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
"""Attach actual V0 request sampling to the existing H3 evidence envelope.

V0 expands 32 training prompts interleaved eight times before worker chunking.
Validation uses one request per prompt and requires its own trainer-assigned
batch positions before chunking; dataset/source indices are not those positions.
The seed is read from the completed request's sampling parameters, never derived
again here. The selected provider re-exports record_v0_sampling_request.
"""

from numbers import Integral

import numpy as np

from verl_omni.experiments.h3_evidence import H3EvidenceContractError, record_sampling_request, require_fields


def prepare_v0_validation_batch(batch):
    """Preserve validation batch positions before the trainer moves fields or chunks rows."""
    batch.non_tensor_batch["_rollout_seed_global_idx"] = np.arange(len(batch), dtype=np.int64)


def record_v0_sampling_request(result, sampling_params, prompt_fields, is_validate, rollout_n):
    """Translate the actual expanded batch index into the frozen prompt/session grid."""
    require_fields(prompt_fields, ("_rollout_seed_global_idx",), "H3 V0 sampling before worker chunking")
    expanded_index = prompt_fields["_rollout_seed_global_idx"]
    if isinstance(expanded_index, bool) or not isinstance(expanded_index, Integral) or expanded_index < 0:
        raise H3EvidenceContractError("H3 V0 sampling requires a nonnegative integer global rollout index")
    if rollout_n != 8:
        raise H3EvidenceContractError("H3 V0 sampling requires the frozen rollout.n=8 contract")
    prompt_index, session_id = (int(expanded_index), 0) if is_validate else divmod(int(expanded_index), rollout_n)
    record_sampling_request(result, sampling_params, session_id, {"index": prompt_index})
