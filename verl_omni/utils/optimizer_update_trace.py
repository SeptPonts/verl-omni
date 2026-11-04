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
"""Persist existing CPU scalar metrics before mini-batch identities are reduced.

One record is a returned train_batch / optimizer-step attempt, not proof of a
nonzero parameter update. In the FSDP and VeOmni diffusion engines, nonfinite
grad_norm skips optimizer.step; finite grad_norm only proves that step returned.
No generic engine skip/applied flag is exposed, so the trace reports the norm
observation instead of inventing an actual-update boolean.

loss is the existing _postprocess_output micro/timestep sum and DP average.
grad_norm is the existing pre-clip norm. lr is present only when the worker
advances the scheduler and is its returned *post-scheduler* first-group LR,
not the LR used by the just-completed optimizer step. Other metrics retain
their existing aggregation path and are not copied into this scalar trace.

For a baseline series, select rank == dp_source_rank and key by
(global_steps, batch_idx). Other ranks contain replicas, not extra updates.
Use a separate trace directory for each run/resume; files append and preserve
duplicate identities rather than silently dropping repeated calls.
"""

import json
import math
from pathlib import Path


def append_optimizer_update_trace(
    trace_dir: str,
    *,
    global_steps: int,
    batch_idx: int,
    rank: int,
    dp_rank: int,
    dp_size: int,
    dp_source_rank: int,
    metrics: dict,
) -> None:
    """Append one returned attempt using only the worker's existing CPU scalars.

    Nonfinite numbers use explicit strings to keep JSON standards-compliant.
    Serialization and I/O errors propagate; no record is synthesized on failure.
    """
    grad_norm = metrics.get("grad_norm")
    observation = "grad_norm_unavailable"
    if grad_norm is not None:
        observation = "finite_grad_norm" if math.isfinite(grad_norm) else "nonfinite_grad_norm"
    scalar_metrics = {}
    for name in ("loss", "grad_norm", "lr"):
        if name in metrics:
            value = metrics[name]
            scalar_metrics[name] = str(value) if not math.isfinite(value) else value
    record = {
        "granularity": "optimizer_step_attempt",
        "global_steps": global_steps,
        "batch_idx": batch_idx,
        "optimizer_update_id": f"{global_steps}:{batch_idx}",
        "rank": rank,
        "dp_rank": dp_rank,
        "dp_size": dp_size,
        "dp_source_rank": dp_source_rank,
        "metrics": scalar_metrics,
        "optimizer_step_observation": observation,
    }
    line = json.dumps(record, allow_nan=False) + "\n"
    directory = Path(trace_dir)
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / f"optimizer_updates.rank-{rank}.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(line)
