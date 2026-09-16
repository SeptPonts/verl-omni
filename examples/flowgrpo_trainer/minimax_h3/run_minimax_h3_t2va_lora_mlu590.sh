#!/usr/bin/env bash
# MiniMax H3 T2VA on 2x8 MLU590, using the NPU recipe's TP/offload settings.
# Start one Ray cluster across both nodes, then run this launcher on the head only.
set -x

script_dir=$(dirname "$(readlink -f "$0")")
NNODES=${NNODES:-2}
export NUM_GPUS=${NUM_GPUS:-8}
export ROLLOUT_TP=${ROLLOUT_TP:-4}
export OUTPUT_DIR=${OUTPUT_DIR:-$script_dir/../../../outputs/run_minimax_h3_t2va_lora_mlu590}

# NUM_GPUS is per node; AgentLoop workers count replicas across the whole cluster.
exec bash "$script_dir/run_minimax_h3_t2va_lora_mlu.sh" \
    +ray_kwargs.ray_init.address="${RAY_ADDRESS:-auto}" \
    trainer.nnodes=$NNODES \
    trainer.n_gpus_per_node=$NUM_GPUS \
    actor_rollout_ref.rollout.agent.num_workers=$((NNODES * NUM_GPUS / ROLLOUT_TP)) \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    trainer.experiment_name=minimax_h3_t2va_lora_mlu590 \
    "$@"
