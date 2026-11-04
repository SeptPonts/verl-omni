# H3 V0 comparison

This opt-in recipe targets an eight-device comparison using main's V0 trainer on Blackwell GPU and MLU. It has **no full training acceptance yet**. Obtain explicit plan approval before any real H3 execution, including eval0 and the two-step acceptance run. V0 hooks have CPU coverage and 2/50-step Hydra configuration previews passed; actual H3 execution still needs validation. This wrapper selects V0 only. Experiment flags remain off when `VERL_EXPERIMENT_MODULE` is not set.

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
# Independent acceptance run: use 2; omit or set 50 for a fresh full run.
export H3_TRAINING_STEPS=2
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

The wrapper selects seed 42, data seed 30, train batch 32 with eight candidates, BF16 LoRA, full determinism on Actor and rollout, parameter offload only, and no profiler. `H3_TRAINING_STEPS` defaults to 50; the strict provider accepts 2 or 50. For each platform, run an independent fresh 2-step acceptance and exit, then, if accepted, run two independent fresh 50-step repeats from the converted canonical adapter. Give every run unique run and checkpoint directories; never resume the acceptance checkpoint into a 50-step run. The 2-step run performs full fixed-128 validation at eval0 and eval2, so budget both evaluations; V0's last-step path also saves the final full checkpoint. Verify these outputs at runtime and confirm that a nonzero step-1 Actor update is consumed by step-2 rollout. The 50-step runs evaluate at 0/10/20/30/40/50 and save five complete checkpoints at 10-step intervals. `WANDB_MODE` defaults to `offline` and an explicit value wins; `WANDB_DIR` is set to the attempt's `wandb` directory. No login is performed. The last arguments are Hydra overrides and take precedence: review them and the resolved config because they can change the experiment contract. A fresh run disables resume. Explicit recovery uses `trainer.resume_mode=resume_path` and `trainer.resume_from_path=<complete-checkpoint>` from the **same run** and retains its manifests. Actor checkpoint and sampler restoration must be verified on V0; global RNG state in rollout, reward and other processes is not checkpointed. Request-derived seeds do not establish bit-identical recovery of every process. Resume accuracy acceptance is **NOT_RUN / INCONCLUSIVE**; use independent fresh repeats for this comparison.

Manifests remain at the run root. Each fresh or resumed invocation creates a new `H3_RUN_DIR/attempts/<id>` for resolved config, role flags, native FileLogger `scalars.jsonl`, input records, evaluation scores and optimizer traces. Set `H3_ATTEMPT_ID` to choose a unique name; otherwise UTC time and PID are used. Existing attempt directories are rejected because FileLogger truncates existing files. Preserve stdout and the W&B run ID for each attempt. `ratio_std` can be NaN/null; retain the native record rather than inventing a value or treating that field alone as a failure.

Optimizer traces are per rank and per optimizer update: with the current batch and PPO settings expect **100 updates** for each 50-step run, two per RL step, and verify the observed identities and coverage. A `finite_grad_norm` observation only says the optimizer step returned; it does not prove a nonzero parameter update. Verify the first nonzero update is consumed by rollout. The required per-step ledger includes all 32 source rows and 256 candidate identities, actual seeds and raw CLAP, ImageBind and combined scores; the 128-sample evaluation ledger must cover each evaluation. V0 callbacks and native LoRA export to `H3_CHECKPOINT_DIR/lora-snapshots` are implemented and CPU-tested; complete H3 training and target-device acceptance are pending. Verify the snapshots and their manifests; they do not replace complete checkpoints. Save actual configurations, input and model hashes, dependency versions, role receipts, scorer batches and communication/precision evidence; compare aligned raw samples and update identities. Check dataset size and epoch coverage permit all requested steps.

The wrapper reuses main's V0 launcher; it does not copy the old PR or legacy business wrapper. V0 avoids the V1 TransferQueue path, and V1-only experiment changes were withdrawn. Main already has the LoRA loader, so no old loader is copied; adapter format and scaling still need verification. Keeping five complete checkpoints removes the old backup-copy workflow for these five points. Old profiler and fixed group-batch machinery are excluded: profiling is off and the recipe specifies the train batch and candidate count. Historical V0 results are a comparison anchor; main and other changes still prevent a single-variable TF32 attribution.
