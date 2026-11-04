# H3 V1 comparison

This opt-in recipe prepares a 50 RL-step, eight-device V1 sync comparison on Blackwell GPU and MLU. It has **no full training acceptance yet**. Review the final configuration and runtime evidence before starting either side. Experiment flags remain off when `VERL_EXPERIMENT_MODULE` is not set.

Provide matching, verified model, data and reward assets on both sides. `H3_INITIAL_LORA` must point to a separately converted and verified main-compatible BF16 adapter, never the old canonical directory. Static audit found the default loader matches 0 of the old 624 unprefixed keys; without alpha metadata it infers scaling 1 instead of the historical 2. Keep the source unchanged. The offline converter takes a new target directory:

```bash
python -m verl_omni.experiments.prepare_h3_adapter --source '<canonical-source>' --target '<new-adapter-directory>'
```

Verify all 624 A/B tensors, keys, hashes, trainability, rank 64, alpha 128 and scaling 2, plus initial Actor-to-rollout sync. Matching eval0 alone does not prove this. Use separate, unique run and checkpoint directories; avoid nesting them. The main launcher also creates an unused `checkpoints` directory under the output directory before the override takes effect.

```bash
export H3_PLATFORM=cuda  # or mlu
export MODEL_PATH='<model-directory>' ACTOR_CONFIG_PATH='<transformer-config-directory>'
export DATA_DIR='<data-directory>' CLAP_MODEL_PATH='<clap-model>' IMAGEBIND_MODEL_PATH='<imagebind-model>'
export H3_INITIAL_LORA='<verified-canonical-adapter>'
export H3_RUN_DIR='<unique-output-directory>' H3_CHECKPOINT_DIR='<separate-unique-checkpoint-directory>'
# Install and verify the worker callback first, as described below.
export H3_RUNTIME_PATCH_RECEIPT='<receipt-directory>/worker-callback.<afterSHA>.installed.json'
# MLU only: contains vendor/libcncl-tuner-csv.so and ring.conf
export H3_CNCL_TUNER_DIR='<verified-tuner-directory>'
bash verl_omni/experiments/run_h3_comparison.sh
```

Run from the repository root. For Blackwell GPU, enable the separately provisioned FA4 b30 overlay in `PYTHONPATH` outside this script and provide the pinned FA4 Hub revision in the local cache. The Actor uses `flash_4_hub` and rollout uses `FLASH_ATTN`; the provider verifies the loaded rollout implementation and pins the Actor Hub revision. On MLU the backends are native and `TORCH_SDPA`. Confirm the tuner plugin loads and the ring rules cover the actual collectives and message sizes; this configuration does not establish that Simple protocol is forced.

Before a real run, the installed vLLM-Omni diffusion worker needs the explicit device-bound callback. In the selected runtime, review and execute:

```bash
python -m verl_omni.experiments.h3_runtime_patch --receipt-dir '<run-owned-path>'
```

This modifies the installed worker source and saves its original and hashes in the receipt directory. The installer prints the receipt contents, not its path. Select the unique `worker-callback.<afterSHA>.installed.json` in that directory and export its full path as `H3_RUNTIME_PATCH_RECEIPT`; preserve the original installed receipt if a later invocation reports an observed callback. Install it in each relevant runtime and inspect the receipt and actual worker import path. Before Ray starts, the provider verifies the current installed worker path, full after-SHA and callback position; a missing or mismatched installation fails. The receipt path is propagated through the Ray environment. The callback applies the provider after device binding and before distributed/model initialization; it is inactive without the experiment module.

Verify actual Actor, diffusion worker and reward-process policy receipts and backend settings. CLAP and ImageBind are custom scorers in the reward process, not named native reward executors: the provider binds the CLAP device before applying process-wide strict flags for both scorers. Verify both scorer devices and actual batch sizes. The rollout server is a control process: verify its environment and existing helper behavior, rather than expecting a separate compute-policy receipt. Driver environment alone does not establish worker policy, kernel precision, communication selection, or scorer batch size.

The wrapper selects seed 42, data seed 30, train batch 32 with eight candidates, BF16 LoRA, full determinism on Actor and rollout, parameter offload only, and no profiler. Validation uses the same fixed 128 samples in one batch at steps 0, 10, 20, 30, 40 and 50. Complete checkpoints are saved every 10 steps, keeping five; eval0 uses the converted canonical adapter. `WANDB_MODE` defaults to `offline` and an explicit value wins. The last arguments are Hydra overrides and take precedence: review them and the resolved config because they can change the experiment contract. A fresh run disables resume. Explicit recovery uses `trainer.resume_mode=resume_path` and `trainer.resume_from_path=<complete-checkpoint>` from the **same run** and retains its manifests. The native Actor checkpoint restores its RNG and sampler state along with training state; global RNG state in rollout, reward and other processes is not checkpointed. Request-derived seeds do not establish bit-identical recovery of every process. Resume accuracy acceptance is **NOT_RUN / INCONCLUSIVE**; use independent fresh repeats for this comparison. The first two steps are a review gate within the same 50-step run, not a separate training run. This wrapper does not pause at step 2; agree and verify a same-run gate mechanism before starting.

Manifests remain at the run root. Each fresh or resumed invocation creates a new `H3_RUN_DIR/attempts/<id>` for resolved config, role flags, native FileLogger `scalars.jsonl`, input records, evaluation scores and optimizer traces. Set `H3_ATTEMPT_ID` to choose a unique name; otherwise UTC time and PID are used. Existing attempt directories are rejected because FileLogger truncates existing files. Preserve stdout and the W&B run ID for each attempt. `ratio_std` can be NaN/null; retain the native record rather than inventing a value or treating that field alone as a failure.

Optimizer traces are per rank and per optimizer update: with the current batch and PPO settings expect **100 updates**, two per RL step, and verify the observed identities and coverage. A `finite_grad_norm` observation only says the optimizer step returned; it does not prove a nonzero parameter update. Verify the first nonzero update is consumed by rollout. The required per-step ledger includes all 32 source rows and 256 candidate identities, actual seeds and raw CLAP, ImageBind and combined scores; the 128-sample evaluation ledger must cover each evaluation. The experiment callbacks implement the candidate ledger and native LoRA export to `H3_CHECKPOINT_DIR/lora-snapshots` at eval0/10/20/30/40/50; complete H3 training and target-device acceptance are still pending. Verify the snapshots and their manifests; they do not replace complete checkpoints. Save actual configurations, input and model hashes, dependency versions, role receipts, scorer batches and communication/precision evidence; compare aligned raw samples and update identities. Check dataset size and epoch coverage permit all 50 steps.

Legacy migration is intentionally limited: the old PR and V0 wrapper implement a different path and are not carried forward. This experiment branch also fixes general V1 TransferQueue materialization and writeback to preserve caller key order, preventing a second random-UID sort from undoing prompt-major canonicalization; it affects all V1 calls on this branch, while the main worktree remains unchanged. Main already has the LoRA loader, so no old loader is copied; adapter format and scaling still need verification. Keeping five complete checkpoints removes the old backup-copy workflow for these five points. Old profiler and fixed group-batch machinery are excluded from this run: profiling is off and the V1 recipe already specifies the train batch and candidate count. Historical V0 results are a comparison anchor, not a single-variable TF32 ablation.
