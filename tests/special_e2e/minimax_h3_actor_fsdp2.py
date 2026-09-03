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
"""Run a tiny MiniMax H3 FlowGRPO actor update through the real FSDP2 engine."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from functools import partial
from pathlib import Path

os.environ.setdefault(
    "VERL_OMNI_AUTO_IMPORT_PIPELINES",
    "minimax_h3_diffusion_nft,minimax_h3_flow_grpo",
)

import torch
import torch.distributed as dist
import torch_mlu  # noqa: F401
from diffusers import MiniMaxH3Transformer3DModel
from tensordict import TensorDict
from torch.distributed.tensor import DTensor
from verl.trainer.config import CheckpointConfig
from verl.utils import tensordict_utils as tu
from verl.utils.device import get_device_name, get_nccl_backend, get_torch_device
from verl.utils.fsdp_utils import fsdp_version
from verl.workers.config import FSDPEngineConfig, FSDPOptimizerConfig

from verl_omni.pipelines.minimax_h3_flow_grpo.common import flatten_joint_latents
from verl_omni.pipelines.minimax_h3_flow_grpo.weight_sync import H3_LORA_TARGETS
from verl_omni.workers.config import (
    DiffusionLossConfig,
    DiffusionModelConfig,
    DiffusionPipelineConfig,
    DiffusionRolloutAlgoConfig,
    FSDPDiffusionActorConfig,
)
from verl_omni.workers.engine.fsdp.diffusers_impl import PPODiffusersFSDPEngine
from verl_omni.workers.utils.losses import diffusion_loss

TINY_H3_CONFIG = {
    "num_attention_heads": 4,
    "attention_head_dim": 16,
    "hidden_size": 48,
    "num_layers": 2,
    "num_refiner_layers": 1,
    "ffn_dim": 64,
    "in_channels": 24,
    "audio_in_channels": 32,
    "patch_size": (1, 2, 2),
    "text_dim": 32,
    "freq_dim": 16,
    "time_embed_hidden_dim": 48,
    "time_embed_dim": 32,
    "rope_freq_dim": 2,
}


def parse_args() -> argparse.Namespace:
    """Parse the shared temporary checkpoint directory."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--work-dir", type=Path, required=True)
    return parser.parse_args()


def save_tiny_h3_checkpoint(work_dir: Path, rank: int) -> Path:
    """Create one deterministic tiny H3 Diffusers checkpoint on rank zero."""
    transformer_dir = work_dir / "transformer"
    if rank == 0:
        work_dir.mkdir(parents=True, exist_ok=False)
        torch.manual_seed(20260903)
        transformer = MiniMaxH3Transformer3DModel(**TINY_H3_CONFIG)
        transformer.save_pretrained(transformer_dir, safe_serialization=True)
        del transformer
    dist.barrier()
    return transformer_dir


def build_actor_config(work_dir: Path, transformer_dir: Path, world_size: int) -> FSDPDiffusionActorConfig:
    """Build the H20-recipe actor semantics around the tiny checkpoint."""
    model_config = DiffusionModelConfig(
        path=str(work_dir),
        config_path=str(transformer_dir),
        architecture="MiniMaxH3Pipeline",
        algorithm="flow_grpo",
        load_tokenizer=False,
        attn_backend="native",
        enable_gradient_checkpointing=True,
        lora_rank=64,
        lora_alpha=128,
        target_modules=sorted(H3_LORA_TARGETS),
        fsdp_layer_prefixes=["transformer_blocks.", "token_refiner.refiner_blocks."],
        pipeline=DiffusionPipelineConfig(num_inference_steps=3),
        algo=DiffusionRolloutAlgoConfig(noise_level=0.8, sde_type="cps"),
    )
    engine_config = FSDPEngineConfig(
        strategy="fsdp2",
        fsdp_size=world_size,
        model_dtype="bfloat16",
        dtype="bfloat16",
        wrap_policy={
            "transformer_layer_cls_to_wrap": [
                "MiniMaxH3TransformerBlock",
                "MiniMaxH3TokenRefinerBlock",
            ]
        },
        param_offload=False,
        optimizer_offload=False,
        offload_policy=False,
        reshard_after_forward=True,
        ulysses_sequence_parallel_size=1,
        forward_only=False,
    )
    optimizer_config = FSDPOptimizerConfig(
        lr=3e-4,
        weight_decay=1e-4,
        total_training_steps=1,
        lr_warmup_steps=0,
        lr_scheduler_type="constant",
        clip_grad=1.0,
    )
    return FSDPDiffusionActorConfig(
        strategy="fsdp2",
        ppo_mini_batch_size=2 * world_size,
        ppo_micro_batch_size_per_gpu=2,
        rollout_n=2,
        diffusion_loss=DiffusionLossConfig(loss_mode="flow_grpo", clip_ratio=0.0001, adv_clip_max=5.0),
        model_config=model_config,
        fsdp_config=engine_config,
        optim=optimizer_config,
        checkpoint=CheckpointConfig(save_lora_only=True),
    )


def build_replay_batch(engine: PPODiffusersFSDPEngine, device: torch.device) -> TensorDict:
    """Build one structurally valid H3 transition for the FlowGRPO replay path."""
    batch_size = 2
    video_rows = 4
    audio_rows = 3
    text_rows = 5
    sequence_length = video_rows + audio_rows + text_rows

    torch.manual_seed(20260903)
    video = torch.randn(batch_size, video_rows, 96, device=device, dtype=torch.bfloat16)
    audio = torch.randn(batch_size, audio_rows, 32, device=device, dtype=torch.bfloat16)
    next_video = (video.float() + 0.05 * torch.randn_like(video.float())).to(torch.bfloat16)
    next_audio = (audio.float() + 0.05 * torch.randn_like(audio.float())).to(torch.bfloat16)
    current_joint = flatten_joint_latents(video, audio)
    next_joint = flatten_joint_latents(next_video, next_audio)

    video_scheduler, audio_scheduler = engine.scheduler
    original_step = 1
    video_timestep = video_scheduler.timesteps[original_step]
    audio_timestep = audio_scheduler.timesteps[original_step]
    position_ids = torch.zeros(sequence_length, 3, device=device, dtype=torch.long)
    position_ids[:, 0] = torch.arange(sequence_length, device=device)
    token_tags = torch.cat(
        [
            torch.zeros(video_rows, device=device, dtype=torch.long),
            torch.full((audio_rows,), 2, device=device, dtype=torch.long),
            torch.ones(text_rows, device=device, dtype=torch.long),
        ]
    )
    video_indices = torch.arange(video_rows, device=device)
    audio_indices = torch.arange(video_rows, video_rows + audio_rows, device=device)
    text_indices = torch.arange(video_rows + audio_rows, sequence_length, device=device)

    batch = TensorDict(
        {
            "all_latents": torch.stack((current_joint, next_joint), dim=1),
            "all_next_latents": next_joint.unsqueeze(1),
            "all_timesteps": video_timestep.expand(batch_size, 1).clone(),
            "h3_audio_timesteps": audio_timestep.expand(batch_size, 1).clone(),
            "h3_step_indices": torch.full(
                (batch_size, 1), original_step, device=device, dtype=torch.long
            ),
            "prompt_embeds": torch.randn(
                batch_size,
                text_rows,
                TINY_H3_CONFIG["text_dim"],
                device=device,
                dtype=torch.bfloat16,
            ),
            "prompt_embeds_mask": torch.ones(batch_size, text_rows, device=device, dtype=torch.int32),
            "h3_video_rows": torch.full((batch_size,), video_rows, device=device, dtype=torch.long),
            "h3_audio_rows": torch.full((batch_size,), audio_rows, device=device, dtype=torch.long),
            "h3_seq_len": torch.full((batch_size,), sequence_length, device=device, dtype=torch.long),
            "h3_position_ids": position_ids.unsqueeze(0).expand(batch_size, -1, -1).clone(),
            "h3_token_tags": token_tags.unsqueeze(0).expand(batch_size, -1).clone(),
            "h3_video_indices": video_indices.unsqueeze(0).expand(batch_size, -1).clone(),
            "h3_audio_indices": audio_indices.unsqueeze(0).expand(batch_size, -1).clone(),
            "h3_text_indices": text_indices.unsqueeze(0).expand(batch_size, -1).clone(),
            "h3_video_update_mask": torch.ones(
                batch_size, video_rows, device=device, dtype=torch.bool
            ),
            "old_log_probs": torch.zeros(batch_size, 1, device=device),
            "advantages": torch.tensor([[1.0], [-0.5]], device=device),
        },
        batch_size=batch_size,
        device=device,
    )
    tu.assign_non_tensor(batch, gradient_accumulation_steps=1, sp_size=1)
    return batch


def local_parameter_copy(parameter: torch.nn.Parameter) -> torch.Tensor:
    """Copy one local FSDP2 parameter shard to CPU for exact update checks."""
    tensor = parameter.to_local() if isinstance(parameter, DTensor) else parameter
    return tensor.detach().float().cpu().clone()


def run_actor_gate(work_dir: Path) -> None:
    """Initialize FSDP2 and execute one real FlowGRPO optimizer update."""
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    if world_size != 2:
        raise ValueError(f"This component gate requires exactly two ranks, got {world_size}.")

    device_api = get_torch_device()
    device_api.set_device(local_rank)
    collective_backend = get_nccl_backend()
    device_name = get_device_name()
    device = torch.device(f"{device_name}:{local_rank}")
    dist.init_process_group(backend=collective_backend, device_id=device)

    transformer_dir = save_tiny_h3_checkpoint(work_dir, rank)
    actor_config = build_actor_config(work_dir, transformer_dir, world_size)
    engine = PPODiffusersFSDPEngine(
        model_config=actor_config.model_config,
        engine_config=actor_config.engine,
        optimizer_config=actor_config.optim,
        checkpoint_config=actor_config.checkpoint,
    )
    engine.initialize()
    if fsdp_version(engine.module) != 2:
        raise AssertionError(f"Expected FSDP2, got fsdp_version={fsdp_version(engine.module)}.")
    if not engine.module.gradient_checkpointing:
        raise AssertionError("MiniMax H3 gradient checkpointing was not enabled.")

    trainable = [(name, parameter) for name, parameter in engine.module.named_parameters() if parameter.requires_grad]
    if not trainable or any("lora_" not in name for name, _ in trainable):
        raise AssertionError(f"Expected LoRA-only trainable parameters, got {[name for name, _ in trainable]}.")
    before = {name: local_parameter_copy(parameter) for name, parameter in trainable}

    batch = build_replay_batch(engine, device)
    with torch.no_grad():
        _, replay = engine.forward_step(batch, loss_function=None, forward_only=True, step=0)
    old_log_probs = replay["model_output"]["log_probs"].detach()
    if not torch.isfinite(old_log_probs).all():
        raise AssertionError(f"Non-finite baseline log probabilities: {old_log_probs}.")
    batch["old_log_probs"] = old_log_probs.unsqueeze(1)

    engine.optimizer_zero_grad()
    loss, update = engine.forward_step(
        batch,
        loss_function=partial(diffusion_loss, config=actor_config),
        forward_only=False,
        step=0,
    )
    loss.backward()
    local_gradients = [local_parameter_copy(parameter.grad) for _, parameter in trainable if parameter.grad is not None]
    if not local_gradients or not all(torch.isfinite(gradient).all() for gradient in local_gradients):
        raise AssertionError("LoRA gradients are missing or non-finite.")
    if not any(torch.count_nonzero(gradient).item() for gradient in local_gradients):
        raise AssertionError("Every local LoRA gradient shard is zero.")

    grad_norm = engine.optimizer_step()
    learning_rate = engine.lr_scheduler_step()
    after = {name: local_parameter_copy(parameter) for name, parameter in trainable}
    updated_names = [name for name in before if not torch.equal(before[name], after[name])]
    if not updated_names:
        raise AssertionError("The optimizer did not update any local LoRA parameter shard.")
    if not torch.isfinite(torch.tensor([loss.item(), grad_norm, learning_rate])).all():
        raise AssertionError("Loss, gradient norm, or learning rate is non-finite.")

    dtype_counts: dict[str, int] = {}
    for _, parameter in engine.module.named_parameters():
        dtype_counts[str(parameter.dtype)] = dtype_counts.get(str(parameter.dtype), 0) + parameter.numel()
    result = {
        "backend": collective_backend,
        "device": str(device),
        "dtype_numel": dtype_counts,
        "fsdp_version": fsdp_version(engine.module),
        "grad_norm": grad_norm,
        "gradient_checkpointing": engine.module.gradient_checkpointing,
        "learning_rate": learning_rate,
        "loss": loss.item(),
        "old_log_probs": old_log_probs.float().cpu().tolist(),
        "rank": rank,
        "trainable_local_numel": sum(parameter.numel() for _, parameter in trainable),
        "trainable_parameter_count": len(trainable),
        "updated_local_parameter_count": len(updated_names),
        "updated_log_probs": update["model_output"]["log_probs"].detach().float().cpu().tolist(),
        "world_size": world_size,
    }
    print(json.dumps(result, sort_keys=True), flush=True)

    passed = torch.ones(1, device=device, dtype=torch.int32)
    dist.all_reduce(passed, op=dist.ReduceOp.SUM)
    if passed.item() != world_size:
        raise AssertionError(f"Only {passed.item()} of {world_size} ranks completed the actor update.")
    dist.barrier()
    if rank == 0:
        print("MiniMax H3 FlowGRPO LoRA64 FSDP2 actor update: PASS", flush=True)
        shutil.rmtree(work_dir)
    dist.barrier()
    dist.destroy_process_group()


def main() -> None:
    """Run the two-rank actor component gate."""
    args = parse_args()
    run_actor_gate(args.work_dir)


if __name__ == "__main__":
    main()
