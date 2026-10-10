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
Profile runs six steps, captures only training step3, then validates once at step6.
Use two independent clean runs. Inputs preserve the existing seed42/n8 recipe;
this integration does not add global seed initialization or full determinism.
The manifest freezes the actual eight-worker StatefulDataLoader order, including
its sampler-reset behavior. Every live pre-repeat batch is checked and recorded
in `H3_RUN_DIR/input_batches.jsonl` before generation. Same input identities and
request seeds do not promise identical generated tensors or rewards.

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
`global_profiler` collector must remain disabled. Each process records local
`ProfilerStep#0` with an explicit RL step3 identity: eight actor-update traces,
four independent TP2 rollout pairs, and one custom reward process trace. The
actor window includes both optimizer mini-updates. Old-logprob and weight sync
currently retain trainer timers but do not get separate device traces.

Analyse the actor and each replica separately. Reward may execute on two devices
inside one process: inspect its device identities before interpreting a merged
summary. TPA Device-Duration/Device-Gap are local profile-on metrics, not clean RL
E2E or proof of physical-device idleness. The exporter excludes unrestricted
environment/argv dumps; identity JSON lives in `h3_metadata/` because TPA scans
all top-level JSON as raw traces. Check every expected rank, completed flag,
trace SHA, actual device events, and absolute/percent summary consistency.
