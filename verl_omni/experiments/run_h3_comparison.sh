#!/usr/bin/env bash
# Run from the application root after sourcing the platform's pinned environment and assets.
set -euo pipefail

platform=${1:?Usage: run_h3_comparison.sh mlu|cuda quality|clean|profile resolve|run}
mode=${2:?Specify quality, clean, or profile}
action=${3:?Specify resolve or run}
: "${H3_RUN_DIR:?Set an absolute, unique run directory on verified storage}"
: "${H3_INITIAL_LORA:?Set the immutable canonical-v2 directory}"
: "${H3_EVALUATION_MANIFEST:?Set the frozen evaluation-manifest.json}"
: "${H3_TRAINING_MANIFEST:?Set the frozen training-manifest.json}"
checkpoint_dir=${H3_CHECKPOINT_DIR:-$H3_RUN_DIR/checkpoints}
checkpoint_backup_dir=${H3_CHECKPOINT_BACKUP_DIR:-}
if [[ "$mode" != quality && -n "$checkpoint_backup_dir" ]]; then
    echo "Checkpoint backup is only supported by quality runs; unset H3_CHECKPOINT_BACKUP_DIR" >&2
    exit 2
fi

case "$platform" in
    mlu) launcher=examples/flowgrpo_trainer/minimax_h3/run_minimax_h3_t2va_lora_mlu.sh ;;
    cuda) launcher=examples/flowgrpo_trainer/minimax_h3/run_minimax_h3_t2va_lora.sh ;;
    *) echo "Unsupported platform: $platform" >&2; exit 2 ;;
esac
case "$action" in
    resolve|run) ;;
    *) echo "Unsupported action: $action" >&2; exit 2 ;;
esac

overrides=(
    trainer.use_v1=false
    trainer.project_name=h3_h20_mlu_comparison
    "trainer.experiment_name=$(basename "$H3_RUN_DIR")"
    "trainer.default_local_dir=$checkpoint_dir"
    "actor_rollout_ref.model.h3_comparison_initial_lora_path=$H3_INITIAL_LORA"
    "trainer.h3_comparison_evaluation_manifest_path=$H3_EVALUATION_MANIFEST"
    "trainer.h3_comparison_training_manifest_path=$H3_TRAINING_MANIFEST"
    "trainer.h3_comparison_input_records_path=$H3_RUN_DIR/input_batches.jsonl"
    trainer.h3_comparison_checkpoint_backup_dir=null
    trainer.resume_mode=disable
    data.shuffle=true
    data.dataloader_num_workers=8
    data.validation_shuffle=false
    actor_rollout_ref.actor.shuffle=false
    actor_rollout_ref.actor.data_loader_seed=42
    +actor_rollout_ref.rollout.val_kwargs.seed=42
    actor_rollout_ref.rollout.val_kwargs.n=1
    actor_rollout_ref.rollout.text_encoder_tp_size=2
    global_profiler.tool=null
    global_profiler.steps=null
    trainer.h3_comparison_profile_dir=null
    trainer.h3_comparison_lora_snapshot_dir=null
    trainer.max_actor_ckpt_to_keep=1
)
case "$mode" in
    quality)
        overrides+=(
            'trainer.logger=[console,file,wandb,rl_insight]'
            actor_rollout_ref.rollout.disable_log_stats=false
            trainer.total_training_steps=50 trainer.val_before_train=true
            trainer.test_freq=10 trainer.save_freq=10
            "trainer.h3_comparison_lora_snapshot_dir=$H3_RUN_DIR/lora_snapshots"
        )
        if [[ -n "$checkpoint_backup_dir" ]]; then
            overrides+=("trainer.h3_comparison_checkpoint_backup_dir=$checkpoint_backup_dir")
        fi
        if [[ "$action" == run ]]; then
            : "${RL_INSIGHT_SERVER_URL:?Prepare RL-Insight client and reachable server before quality training}"
            python3 -c 'import rl_insight'
        fi
        ;;
    clean|profile)
        overrides+=(
            'trainer.logger=[console,file,wandb]'
            trainer.total_training_steps=6 trainer.val_before_train=false
            trainer.test_freq=-1 trainer.save_freq=-1
            trainer.rollout_data_dir=null trainer.validation_data_dir=null
            trainer.log_val_generations=0
        )
        if [[ "$mode" == profile ]]; then
            overrides+=(
                trainer.test_freq=6
                "trainer.validation_data_dir=$H3_RUN_DIR/validation_data"
                trainer.log_val_generations=8
                "trainer.h3_comparison_profile_dir=$H3_RUN_DIR/profile"
                trainer.h3_comparison_profile_step=3
            )
            if [[ "$action" == run ]]; then
                python3 -c 'from torch_profiler_analysis import ProfileConfig, initialize_profiler, profile_analyse_offline'
            fi
        fi
        ;;
    *) echo "Unsupported mode: $mode" >&2; exit 2 ;;
esac

if [[ -n "${H3_RESUME_FROM:-}" ]]; then
    [[ "$mode" == quality ]] || { echo "Only quality runs resume" >&2; exit 2; }
    overrides+=(trainer.resume_mode=resume_path "trainer.resume_from_path=$H3_RESUME_FROM")
fi
export OUTPUT_DIR="$H3_RUN_DIR"
export NUM_GPUS=8 ROLLOUT_TP=2 TEXT_ENCODER_TP=2 REWARD_NUM_WORKERS=1
export REWARD_DEVICE="$platform"
export ACTOR_SP=1 ROLLOUT_USP=1 ROLLOUT_RING=1
export VAE_PATCH_PARALLEL_SIZE=1 VAE_PARALLEL_MODE=tile VAE_USE_TILING=False
export HEIGHT=256 WIDTH=384 NUM_FRAMES=121 INFER_STEPS=10 VAL_HEIGHT=512 VAL_WIDTH=768
export WANDB_MODE=online
export WANDB_DIR="$H3_RUN_DIR/wandb"
unset VERL_RL_INSIGHT_ENABLE
mkdir -p "$WANDB_DIR"

if [[ "$action" == resolve ]]; then
    export H3_CONFIG_SOURCE_ROOT="$PWD" H3_CONFIG_OUTPUT="$H3_RUN_DIR/resolved-config.yaml"
    # Capture the launcher's exact argv; compose YAML without importing the training/device entrypoint.
    # Read the installed VeRL config files directly, avoiding its package-level runtime initialization.
    python3() {
        command python3 - "$@" <<'PY'
import json
import os
import sys
from importlib.metadata import distribution
from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

arguments = sys.argv[1:]
if arguments[:2] != ["-m", "verl_omni.trainer.main_diffusion"] or arguments[-3:] != ["--cfg", "job", "--resolve"]:
    raise ValueError("Configuration resolution accepts only the H3 trainer --cfg job --resolve invocation")
config_root = str(Path(os.environ["H3_CONFIG_SOURCE_ROOT"]) / "verl_omni/trainer/config")
verl_configs = distribution("verl").locate_file("verl/trainer/config").resolve()
with initialize_config_dir(config_dir=config_root, version_base=None):
    config = compose(config_name="diffusion_trainer", overrides=arguments[2:-3] + [f"hydra.searchpath=[file://{verl_configs}]"])
OmegaConf.resolve(config)
output = Path(os.environ["H3_CONFIG_OUTPUT"])
output.write_text(OmegaConf.to_yaml(config))
output.with_suffix(".argv.json").write_text(json.dumps(arguments, indent=2) + "\n")
PY
    }
    export -f python3
    bash "$launcher" "${overrides[@]}" --cfg job --resolve > "$H3_RUN_DIR/resolve.log" 2>&1
    exit
fi

# FileLogger opens in write mode; each resumed process needs its own scalar file.
attempt="$(date -u +%Y%m%dT%H%M%SZ)-$$"
attempt_dir="$H3_RUN_DIR/attempts/$attempt"
mkdir -p "$attempt_dir"
export VERL_FILE_LOGGER_PATH="$attempt_dir/scalars.jsonl"
if [[ -n "${H3_RESUME_FROM:-}" ]]; then
    export WANDB_RUN_ID="$(cat "$H3_RUN_DIR/wandb-run-id")"
    export WANDB_RESUME=must
else
    mkdir "$H3_RUN_DIR/fresh-started"
    export WANDB_RUN_ID="$(python3 -c 'import uuid; print(uuid.uuid4().hex[:16])')"
    printf '%s\n' "$WANDB_RUN_ID" > "$H3_RUN_DIR/wandb-run-id"
    export WANDB_RESUME=never
fi
git rev-parse HEAD > "$attempt_dir/source-commit.txt"
git diff --binary HEAD > "$attempt_dir/source.patch"
printf '%q ' bash "$launcher" "${overrides[@]}" > "$attempt_dir/command.sh"
printf '\n' >> "$attempt_dir/command.sh"
bash "$launcher" "${overrides[@]}" 2>&1 | tee "$attempt_dir/console.log"
