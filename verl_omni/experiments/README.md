# Temporary H3 H20/MLU comparison

This integration belongs to `feature/h3-mlu-support` for the paired experiment.
Remove this directory, its tests, and all `h3_comparison_*` config fields/hooks
when preparing the separate MLU delivery PR.

Use the existing V0 GPU or MLU launcher with these explicit overrides:

```bash
actor_rollout_ref.model.h3_comparison_initial_lora_path=/absolute/path/to/canonical-v2 \
trainer.h3_comparison_lora_snapshot_dir=/absolute/path/to/run/lora_snapshots \
trainer.default_local_dir=/absolute/path/to/run/checkpoints \
trainer.resume_mode=disable \
trainer.logger='[console,file,wandb]' \
trainer.save_freq=10 \
trainer.max_actor_ckpt_to_keep=1 \
trainer.total_training_steps=50
```

To continue the same run from a completed full checkpoint, keep the original
recipe, common package, checkpoint directory, and snapshot directory, and change
only the resume overrides:

```bash
trainer.resume_mode=resume_path \
trainer.resume_from_path=/absolute/path/to/run/checkpoints/global_step_30 \
trainer.total_training_steps=50
```

`total_training_steps=50` is the total target, so a step-30 checkpoint continues
at step 31. `auto` is rejected because a missing checkpoint must not silently
start a new comparison run. The checkpoint must contain `data.pt` and every
rank's `model_world_size_*_rank_*.pt`, `optim_world_size_*_rank_*.pt`, and
`extra_state_world_size_*_rank_*.pt`, with the same world size as the resumed run.
Both `checkpoint.save_contents` and `checkpoint.load_contents` must include
`model`, `optimizer`, and `extra`. The existing loader restores the actor model,
optimizer, scheduler/RNG, and dataloader state; a LoRA evaluation snapshot cannot
replace that checkpoint. Verify the checkpoint and snapshot identities before
launching. These CPU checks do not establish real eight-device checkpoint recovery
or bitwise continuation of the rollout process.

Resume keeps the existing `val_before_train` behavior: when enabled, validation
runs again at the restored step before the next training iteration. Do not change
the training or validation recipe merely to fit a resource lease. There is no
lease-expiry save or graceful-stop guarantee here; resume from the last completed
checkpoint, which can be older than the last logged training step.

The initialization and snapshot paths default to `null`. The initialization hook copies the common
BF16 A/B tensors after ordinary LoRA creation, preserving rank 64, alpha 128,
trainability, base weights, and the existing FSDP2 rank-zero broadcast. The
package includes `adapter_model.safetensors`, `adapter_config.json`,
`transformer_config.json`, and `tensor_manifest.json`. Keep the package immutable
throughout the run. The full-model FSDP/device path still needs device validation.

Snapshots at steps 0/10/20/30/40/50 live outside checkpoint rotation. Step 0 copies
the checked initial package before validation. Later snapshots use the existing
CPU FSDP checkpoint exporter after the synchronous save and tracker write finish.
The exporter normalizes keys and retains the original H3 LoRA configuration;
each snapshot records file hashes. It is an adapter for evaluation, not a full
training-resume checkpoint. Existing snapshot directories are never overwritten.
Failed exports retain their `.incomplete` directory and fail the run visibly.
Resume does not export step 0 or repeat the restored step's snapshot. If a save
completed but its snapshot export was interrupted, finish that missing snapshot
from the full checkpoint on CPU before resuming; preserve completed snapshots and
inspect the incomplete output instead of silently overwriting it. Subsequent
scheduled saves export their snapshots normally (for example, steps 40 and 50
after resuming step 30).
Full checkpoint retention of one still needs space for two during saving:
VeRL writes the new checkpoint before pruning the previous one.
The current FSDP manager tracks saves from its own process only. A loaded
checkpoint is not registered for rotation, so after a restart it can remain in
addition to those two checkpoint generations. Budget that extra space and inspect
completed checkpoints before any separate cleanup.

Snapshot export time is included in `timing_s/save_checkpoint` and run wall time.
For clean performance runs, use common initialization with the snapshot directory
left `null`. Do not subtract snapshot I/O selectively from quality-run timing.
These hooks do not alter offload, batch size, attention, or parallelism settings.

Keep W&B and local scalar logs enabled in every quality, clean-performance, and
profile run. Use separate run identities and record the common initialization,
source/config identities, platform, and measurement purpose. RL-Insight is a
separate monitoring integration; enabling its logger requires its client and
server to be prepared first.

CPU verification:

```bash
python -m pytest -q tests/experiments/test_h3_comparison_on_cpu.py --confcutdir=tests/experiments
```

## Complete launch entrypoint

From the application root, source the platform's pinned environment/assets and set
`H3_RUN_DIR`, `H3_INITIAL_LORA`, `H3_EVALUATION_MANIFEST`, and
`H3_TRAINING_MANIFEST`. Use a distinct persistent run directory for every fresh run:

```bash
bash verl_omni/experiments/run_h3_comparison.sh mlu clean resolve
bash verl_omni/experiments/run_h3_comparison.sh mlu clean run
```

The platform is `mlu` or `cuda`; the mode is `quality`, `clean`, or `profile`.
`resolve` writes pure resolved YAML plus the actual launcher arguments without
starting Ray, models, or logging clients. `run` retains console, W&B and a new
scalar JSONL per process attempt. Quality adds RL-Insight and requires its prepared
client/server. The current H20 preparation environment has no TPA; prepare and
validate a compatible build before its profile run. None of these launch/config
checks constitute full-model or eight-device acceptance.

Quality runs 50 steps with validation at 0/10/20/30/40/50, keep1 full checkpoints
and six LoRA snapshots. Clean runs six steps without validation/checkpoints/media.
Profile runs four steps in one continuous training job: step1 warms up, steps2/3
capture both Actor and Rollout, and step4 checks unprofiled continuation. It does
not capture Reward or run validation. Full validation remains in the quality run.
Use two independent clean runs. Inputs preserve the existing seed42/n8 recipe;
this integration does not add global seed initialization. Kernel determinism is
an explicit opt-in described below.
The manifest freezes the actual eight-worker StatefulDataLoader order, including
its sampler-reset behavior. Every live pre-repeat batch is checked and recorded
in `H3_RUN_DIR/input_batches.jsonl` before generation. Same input identities and
request seeds do not promise identical generated tensors or rewards.

## Opt-in H3 kernel determinism

Set `H3_DETERMINISTIC=1` before the existing experiment launcher. The shell entry
sets the device, receipt directory, and CUDA cuBLAS workspace before Python starts:

```bash
H3_DETERMINISTIC=1 bash verl_omni/experiments/run_h3_comparison.sh cuda quality resolve
# Run only after the experiment's resource and training approval checks.
H3_DETERMINISTIC=1 bash verl_omni/experiments/run_h3_comparison.sh cuda quality run
```

The default is off. When enabled, the driver and model-owning process constructors apply
`torch.use_deterministic_algorithms(True, warn_only=False)` and select deterministic
cuDNN/CNNL with benchmarking disabled. CUDA also uses
`CUBLAS_WORKSPACE_CONFIG=:4096:8`. Each initialized process writes its flags and
policy hash under `H3_RUN_DIR/determinism/`. Ray workers cover actor, rollout server,
and reward roles; a separate dependency patch initializes spawned diffusion
workers before device/model initialization. A Ray process setup hook must not
import this application: Ray assigns actor device visibility afterward, so such
an import can cache all cards before the single-card mask takes effect.

This mode preserves seed derivation, RNG calls, training noise, `PYTHONHASHSEED`,
NCCL/CNCL settings, cuDNN enablement, TF32, parallelism, and offload. Keep existing
VeRL `full_determinism` fields false: that helper reseeds, changes communication
settings, and disables cuDNN. Conflicting true values are rejected.

The frozen B300 environment additionally needs the reviewed dependency patch in
the parent context repository, `tools/h3_determinism/runtime.patch`. It forwards
`deterministic=True` through the actual Diffusers FA4 and vLLM-Omni dense attention
paths, and pins the actor Hub kernel to the baseline's recorded commit. Apply it
only with its source-hash manifest; it is specific to the audited B300 versions.
Do not apply that patch blindly to an MLU stack with different dependency versions.
The CNNL policy branch alone does not establish MLU attention determinism.

Strict mode fails on registered unsupported operations; it does not prove custom
kernels or distributed training reproducible. Component and full-model validation
remain separate gates. `resolve` verifies the configuration without exercising
the worker constructors or kernels.

```bash
python -m pytest -q tests/experiments/test_h3_determinism_on_cpu.py --confcutdir=tests/experiments
```

## One-generation persistent checkpoint backup

For quality training on a local disk, set `H3_CHECKPOINT_DIR` to the local
checkpoint root and `H3_CHECKPOINT_BACKUP_DIR` to a new, empty, run-owned persistent
backup root. Keep `H3_RUN_DIR` on persistent storage so metrics, media and LoRA
snapshots survive the allocation. Save and snapshot export finish before the
synchronous backup. Its complete file inventory, byte counts and SHA256 values
must match before `backup_manifest.json` becomes `COMPLETE`.

This policy assumes capacity for one persistent generation, not two. During an
in-place update the old backup is unavailable; failures leave `INCOMPLETE` and
stop training with the complete local source retained. A lost local allocation
during that interval can leave no valid persistent recovery point. Retry the
same step from its complete local source; publication failures reuse the existing
generation instead of allocating another complete copy. A backup directory with
an incomplete receipt is rejected for resume. Initial writes require an empty
root to prevent replacing unrelated data.

Set `H3_RESUME_FROM` to the verified `global_step_N` directory and retain the
original run directory/W&B ID. The target remains 50, and the existing
`val_before_train=true` behavior revalidates the restored step. There is no lease
expiry checkpoint or graceful-stop mechanism; budget time for validation, full
save, transfer and verification before the allocation ends.

## TPA boundary

TPA is imported lazily only by an enabled role-local capture. The old
`global_profiler` collector must remain disabled. Both backends use shapes, with
memory and stack collection disabled and online analysis skipped. Each selected
RL step produces eight `actor_update_compute/` traces and one complete TP2 pair
under `rollout_replica_000/`: 20 traces over steps2/3. Each trace has its own local
`ProfilerStep#0`; `h3_metadata/` records the actual RL step and capture scope.

The Actor window starts inside `engine.train_mode`, after parameter onload. It
includes both optimizer mini-updates and ends with stop/export plus profiler
object collection before train-mode offload. Its role/annotation is
`actor-update-compute` / `h3/actor_update_compute`, deliberately distinct from
historical full `actor_update/` traces that included parameter transfers.

Rollout selects one **replica**, never an incomplete subset of its TP ranks.
Its window configuration requires `replica`, `request_start`, and `request_end`;
the supplied recipe sets inclusive 1-based bounds to 3..4. Older custom Rollout
window lists must add these fields; there is no full-generation fallback.
The selected worker's existing
`execute_model` is intercepted only for that generation phase. It skips the
first two full requests, captures requests3/4 including all denoise iterations,
then exports and releases the profiler before request5 or replica sleep. Both
TP ranks retain request ordinals and backend request IDs in their receipts.
The generation-end RPC restores normal dispatch and rejects a successful phase
that never completed the requested window. Counters restart on the next selected
RL step. This route assumes H3 full-request execution with request batching and
stepwise execution disabled; it does not change sampling or reduce the training
batch. Check request identities against `input_batches.jsonl` and backend request
mapping before claiming paired inputs across platforms.

Generation completes before Actor update within the same RL step. The two role
collectors need not overlap; no second training launch is required. Old-logprob
and weight sync retain trainer timers without separate device traces. Profile-on
stage times include observer/export overhead and cannot replace clean timing.

Analyse each step's Actor group and TP2 group separately, outside training. The
optional `reward` window remains supported when explicitly selected; a reward
process may use two devices, requiring separate device-identity inspection.
TPA Device-Duration/Device-Gap are local profile-on metrics, not clean RL
E2E or proof of physical-device idleness. The exporter excludes unrestricted
environment/argv dumps; identity JSON lives in `h3_metadata/` because TPA scans
all top-level JSON as raw traces. Check every expected rank, completed flag,
trace SHA, actual device events, and absolute/percent summary consistency.

CPU contract tests cover dispatch, boundaries, counter reset, failure receipts,
and Python profiler ownership. They do not certify native weakref release,
distributed progress, export memory, full TPA FLOPs/communication coverage, or
MLU clock alignment. Before accepting the new recipe on the dedicated 692,
require all four training steps to complete, 20 complete/hash-verified traces
with shapes, matching TP request lists, and native offline summary inspection.
H20 needs its own hardware acceptance with the same scope.

Run the local contract and real-launcher configuration checks without loading
models or starting Ray (requires pytest, hydra-core, and omegaconf):

```bash
python -m pytest -q tests/experiments/test_h3_profiling_on_cpu.py tests/experiments/test_h3_profile_config_on_cpu.py --confcutdir=tests/experiments
```
