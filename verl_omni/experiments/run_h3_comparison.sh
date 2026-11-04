#!/usr/bin/env bash
set -euo pipefail

: "${H3_PLATFORM:?Set H3_PLATFORM to cuda or mlu}"
: "${MODEL_PATH:?}" "${ACTOR_CONFIG_PATH:?}" "${DATA_DIR:?}" "${CLAP_MODEL_PATH:?}" "${IMAGEBIND_MODEL_PATH:?}"
: "${H3_INITIAL_LORA:?}" "${H3_RUN_DIR:?}" "${H3_CHECKPOINT_DIR:?}" "${H3_RUNTIME_PATCH_RECEIPT:?}"
script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd "$script_dir/../.." && pwd)
export MODEL_PATH ACTOR_CONFIG_PATH DATA_DIR CLAP_MODEL_PATH IMAGEBIND_MODEL_PATH H3_RUNTIME_PATCH_RECEIPT
export OUTPUT_DIR="$H3_RUN_DIR" NUM_GPUS=8 TOTAL_TRAINING_STEPS=50
export ACTOR_SP=1 ROLLOUT_TP=2 TEXT_ENCODER_TP=2
export WANDB_MODE="${WANDB_MODE:-offline}"

case "$H3_PLATFORM" in
    cuda)
        launcher="$repo_root/examples/flowgrpo_trainer/minimax_h3/run_minimax_h3_t2va_lora_v1.sh"
        export H3_FA4_HUB_REVISION=86c34f1cec40ea37e019fe79b054d9a1e189ffb2 REWARD_DEVICE=cuda
        platform_args=(actor_rollout_ref.model.attn_backend=flash_4_hub actor_rollout_ref.rollout.rollout_attn_backend=FLASH_ATTN)
        ;;
    mlu)
        : "${H3_CNCL_TUNER_DIR:?Set H3_CNCL_TUNER_DIR to the tuner directory}"
        launcher="$repo_root/examples/flowgrpo_trainer/minimax_h3/run_minimax_h3_t2va_lora_v1_mlu.sh"
        export VERL_PLATFORM=cambricon REWARD_DEVICE=mlu
        export CNCL_TUNER_PLUGIN="$H3_CNCL_TUNER_DIR/vendor/libcncl-tuner-csv.so"
        export CNCL_TUNER_CONFIG_FILE="$H3_CNCL_TUNER_DIR/ring.conf"
        platform_args=(actor_rollout_ref.model.attn_backend=native actor_rollout_ref.rollout.rollout_attn_backend=TORCH_SDPA)
        ;;
    *) echo "H3_PLATFORM must be cuda or mlu" >&2; exit 2 ;;
esac

attempt_id=${H3_ATTEMPT_ID:-$(date -u +%Y%m%dT%H%M%SZ)-$$}
case "$attempt_id" in
    .|..|*/*) echo "H3_ATTEMPT_ID must be a single directory name" >&2; exit 2 ;;
esac
export H3_ATTEMPT_DIR="$H3_RUN_DIR/attempts/$attempt_id"
mkdir -p "$H3_RUN_DIR/attempts"
mkdir "$H3_ATTEMPT_DIR"
export VERL_FILE_LOGGER_PATH="$H3_ATTEMPT_DIR/scalars.jsonl"
if [[ -e "$VERL_FILE_LOGGER_PATH" || -L "$VERL_FILE_LOGGER_PATH" ]]; then
    echo "FileLogger path already exists: $VERL_FILE_LOGGER_PATH" >&2; exit 1
fi
source "$script_dir/h3_determinism.sh"

exec bash "$launcher" \
    data.seed=30 data.shuffle=true data.dataloader_num_workers=8 data.validation_shuffle=false \
    data.train_batch_size=32 data.val_batch_size=128 data.val_max_samples=128 \
    actor_rollout_ref.actor.shuffle=false actor_rollout_ref.actor.data_loader_seed=42 \
    actor_rollout_ref.actor.fsdp_config.seed=42 actor_rollout_ref.actor.fsdp_config.full_determinism=true \
    actor_rollout_ref.actor.fsdp_config.param_offload=true actor_rollout_ref.actor.fsdp_config.optimizer_offload=false \
    actor_rollout_ref.actor.fsdp_config.offload_policy=false \
    "actor_rollout_ref.model.lora_adapter_path=$H3_INITIAL_LORA" actor_rollout_ref.model.lora_dtype=bfloat16 \
    actor_rollout_ref.rollout.n=8 actor_rollout_ref.rollout.seed=42 actor_rollout_ref.rollout.full_determinism=true \
    actor_rollout_ref.rollout.val_kwargs.n=1 +actor_rollout_ref.rollout.val_kwargs.seed=42 \
    actor_rollout_ref.rollout.algo.sde_window_seed=42 \
    trainer.resume_mode=disable trainer.val_before_train=true trainer.test_freq=10 trainer.save_freq=10 \
    trainer.max_actor_ckpt_to_keep=5 "trainer.default_local_dir=$H3_CHECKPOINT_DIR" 'trainer.logger=[console,file,wandb]' \
    global_profiler.tool=null global_profiler.steps=null \
    "${platform_args[@]}" "$@"
