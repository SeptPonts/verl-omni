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

"""Run the MiniMax H3 FlowGRPO rollout core with TP2 on an accelerator.

The authorized H3 checkpoint is deliberately not synthesized here.  This gate
bypasses checkpoint-bound tokenizer, Qwen3-VL, and VAE construction, then runs
the real H3 packed DiT, TP collectives, LoRA-64 layers, CPS transitions, and
trajectory materialization at the recipe's training geometry.
"""

from __future__ import annotations

import os
from collections import Counter

os.environ.setdefault(
    "VERL_OMNI_AUTO_IMPORT_PIPELINES",
    "minimax_h3_diffusion_nft,minimax_h3_flow_grpo",
)

import torch
import torch.nn as nn
from diffusers import MiniMaxH3Transformer3DModel
from peft import LoraConfig
from peft.utils.save_and_load import get_peft_model_state_dict
from verl.utils.device import get_device_name, get_nccl_backend, get_torch_device
from vllm.config import VllmConfig, set_current_vllm_config
from vllm.distributed import (
    destroy_distributed_environment,
    destroy_model_parallel,
    init_distributed_environment,
    initialize_model_parallel,
)
from vllm_omni.diffusion.config import set_current_diffusion_config
from vllm_omni.diffusion.data import DiffusionParallelConfig, OmniDiffusionConfig, TransformerConfig
from vllm_omni.diffusion.forward_context import set_forward_context
from vllm_omni.diffusion.lora.manager import DiffusionLoRAManager
from vllm_omni.diffusion.models.minimax_h3.minimax_h3_transformer import (
    MiniMaxH3Attention,
    MiniMaxH3DiTModel,
)
from vllm_omni.diffusion.models.minimax_h3.time_request import (
    MINIMAX_H3_SHAPE_PLANNER,
    minimax_h3_align_frame_count,
)

from verl_omni.pipelines.minimax_h3_flow_grpo.vllm_omni_rollout_adapter import (
    MiniMaxH3PipelineWithLogProb,
)
from verl_omni.utils.vllm_omni import OmniTensorLoRARequest, VLLMOmniHijack

RECIPE_HEIGHT = 256
RECIPE_WIDTH = 384
RECIPE_REQUESTED_FRAMES = 121
RECIPE_MAX_TEXT_LEN = 1024
TEST_TEXT_LEN = 16
TEST_DENOISE_POINTS = 3
TEST_TP_SIZE = 2

TINY_H3 = {
    "num_attention_heads": 4,
    "attention_head_dim": 128,
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
    "rope_freq_dim": 16,
}
LORA_TARGETS = ["to_q", "to_k", "to_v", "to_out.0", "ff.net.0.proj", "ff.net.2"]


def make_vllm_transformer_config() -> TransformerConfig:
    """Translate the tiny Diffusers H3 architecture to vLLM-Omni names."""
    config = {
        **TINY_H3,
        "token_refiner_num_layers": TINY_H3["num_refiner_layers"],
        "ffn_hidden_size": TINY_H3["ffn_dim"],
        "latents_dim": TINY_H3["in_channels"],
        "audio_latents_dim": TINY_H3["audio_in_channels"],
        "timestep_input_dim": TINY_H3["freq_dim"],
        "time_embed_hidden_size": TINY_H3["time_embed_hidden_dim"],
        "adaln_out_features": 18 * TINY_H3["hidden_size"],
        "final_adaln_out_features": 2 * TINY_H3["hidden_size"],
        "rope_inv_freq_len": TINY_H3["rope_freq_dim"],
    }
    return TransformerConfig.from_dict(config)


def initialize_tiny_weights(model: MiniMaxH3DiTModel, rank: int) -> None:
    """Materialize finite deterministic weights without imitating a checkpoint."""
    generator = torch.Generator(device="cpu").manual_seed(20260903 + rank)
    for name, parameter in model.named_parameters():
        if name.endswith("norm.weight") or name.endswith("q_norm.weight") or name.endswith("k_norm.weight"):
            parameter.data.fill_(1)
        elif parameter.is_floating_point():
            values = torch.randn(parameter.shape, generator=generator, dtype=torch.float32) * 0.01
            parameter.data.copy_(values.to(parameter.dtype))
    inv_freq = model.rope.inv_freq
    indices = torch.arange(0, 2 * inv_freq.numel(), 2, dtype=torch.float32)
    inv_freq.copy_(1.0 / (10000 ** (indices / (2 * inv_freq.numel()))))


def make_lora_payload() -> tuple[dict[str, torch.Tensor], dict]:
    """Create an in-memory rank-64 Actor update using the official targets."""
    torch.manual_seed(20260903)
    actor = MiniMaxH3Transformer3DModel(**TINY_H3)
    config = LoraConfig(
        r=64,
        lora_alpha=128,
        target_modules=LORA_TARGETS,
        bias="none",
    )
    actor.add_adapter(config, adapter_name="old")
    payload = {
        name: tensor.detach().clone() for name, tensor in get_peft_model_state_dict(actor, adapter_name="old").items()
    }
    generator = torch.Generator(device="cpu").manual_seed(20260904)
    for tensor in payload.values():
        tensor.copy_(torch.randn(tensor.shape, generator=generator, dtype=torch.float32).to(tensor.dtype) * 0.01)
    return payload, config.to_dict()


def make_rollout_pipeline(
    transformer: MiniMaxH3DiTModel,
    od_config: OmniDiffusionConfig,
    device: torch.device,
) -> MiniMaxH3PipelineWithLogProb:
    """Build only the checkpoint-independent part of the production adapter."""
    pipeline = object.__new__(MiniMaxH3PipelineWithLogProb)
    nn.Module.__init__(pipeline)
    pipeline.od_config = od_config
    pipeline.parallel_config = od_config.parallel_config
    pipeline.device = device
    pipeline.transformer = transformer
    pipeline.install_h3_lora_layout()
    pipeline._flow_grpo_noise_level = 0.8
    pipeline._flow_grpo_sde_type = "cps"
    pipeline._flow_grpo_window_size = None
    pipeline._flow_grpo_window_range = None
    pipeline._flow_grpo_sde_contiguous = True
    pipeline._flow_grpo_seed = 42
    pipeline._flow_grpo_trajectory = {}
    pipeline._h3_max_text_len = RECIPE_MAX_TEXT_LEN
    pipeline.set_progress_bar_config(disable=True)
    return pipeline


def assert_replicated_tensor(value: torch.Tensor, tp_size: int) -> None:
    """Require the TP-reduced result to agree on every rank."""
    gathered = [torch.empty_like(value) for _ in range(tp_size)]
    torch.distributed.all_gather(gathered, value)
    torch.testing.assert_close(
        torch.stack(gathered),
        gathered[0].unsqueeze(0).expand(tp_size, -1),
        rtol=0,
        atol=0,
    )


def run_rollout(rank: int, local_rank: int, tp_size: int, vllm_config: VllmConfig) -> None:
    """Construct the tiny TP2 model and execute the recipe-shaped rollout."""
    if tp_size != TEST_TP_SIZE:
        raise ValueError(f"This gate requires TP={TEST_TP_SIZE}, got world size {tp_size}.")
    device_name = get_device_name()
    device = torch.device(f"{device_name}:{local_rank}")
    parallel_config = DiffusionParallelConfig(tensor_parallel_size=tp_size)
    od_config = OmniDiffusionConfig(
        model="tiny-minimax-h3-rollout-core",
        tf_model_config=make_vllm_transformer_config(),
        dtype=torch.bfloat16,
        parallel_config=parallel_config,
        diffusion_attention_config={"default": "TORCH_SDPA"},
        enforce_eager=True,
    )
    with set_current_diffusion_config(od_config):
        transformer = MiniMaxH3DiTModel(od_config)
    initialize_tiny_weights(transformer, rank)
    transformer.to(device)

    pipeline = make_rollout_pipeline(transformer, od_config, device)
    payload, peft_config = make_lora_payload()
    VLLMOmniHijack.hijack()
    manager = DiffusionLoRAManager(pipeline, device=device, dtype=torch.bfloat16)
    request = OmniTensorLoRARequest(
        lora_name="h3-rollout-core",
        lora_int_id=1,
        lora_path="/tmp/h3-rollout-core-unused",
        peft_config=peft_config,
        lora_tensors=payload,
    )
    assert manager.add_adapter(request)
    manager.set_active_adapter(request)
    assert manager._active_adapter_id == 1
    assert len(manager._lora_modules) == 12

    attention_impls = Counter(
        type(module.attention.attention).__name__
        for module in transformer.modules()
        if isinstance(module, MiniMaxH3Attention)
    )
    assert attention_impls == {"SDPAImpl": 3}, attention_impls

    aligned_frames = minimax_h3_align_frame_count(RECIPE_REQUESTED_FRAMES)
    latent_t = MINIMAX_H3_SHAPE_PLANNER.video_latent_t(aligned_frames)
    audio_t = MINIMAX_H3_SHAPE_PLANNER.audio_latent_t(aligned_frames / 24)
    input_generator = torch.Generator(device="cpu").manual_seed(42)
    text_embeddings = torch.randn(
        TEST_TEXT_LEN,
        TINY_H3["text_dim"],
        generator=input_generator,
        dtype=torch.bfloat16,
    ).to(device)
    text_tags = torch.ones(TEST_TEXT_LEN, dtype=torch.long, device=device)
    with torch.no_grad(), set_forward_context(
        vllm_config=vllm_config,
        omni_diffusion_config=od_config,
    ):
        video, audio = pipeline.diffuse(
            task="t2va",
            text_embeddings=text_embeddings,
            text_tags=text_tags,
            seed=42,
            latent_t=latent_t,
            latent_h=RECIPE_HEIGHT // 16,
            latent_w=RECIPE_WIDTH // 16,
            audio_t=audio_t,
            num_frames=aligned_frames,
            num_steps=TEST_DENOISE_POINTS,
            video_shift=12.0,
            audio_shift=3.0,
            visual_condition=None,
            visual_condition_shape=None,
            audio_condition=None,
            ref_audio_t=None,
        )

    trajectory = pipeline._flow_grpo_trajectory
    assert tuple(video.shape) == (1, 24, 37, 16, 24)
    assert tuple(audio.shape) == (2, 32, 207)
    assert tuple(trajectory["all_latents"].shape[:3]) == (1, 2, 1)
    assert tuple(trajectory["all_log_probs"].shape) == (1, 2)
    assert tuple(trajectory["h3_position_ids"].shape) == (1, 4990, 3)
    assert tuple(trajectory["h3_token_tags"].shape) == (1, 4990)
    assert trajectory["h3_seq_len"].item() == 3982
    for name in ("all_latents", "all_next_latents", "all_log_probs", "prompt_embeds"):
        assert torch.isfinite(trajectory[name]).all(), f"non-finite trajectory field: {name}"
    assert torch.isfinite(video).all()
    assert torch.isfinite(audio).all()

    checksum = torch.stack(
        [
            video.float().mean(),
            audio.float().mean(),
            trajectory["all_log_probs"].float().mean(),
        ]
    )
    assert_replicated_tensor(checksum, tp_size)
    dtype_counts = Counter(str(parameter.dtype) for parameter in transformer.parameters())
    print(
        f"rank={rank}/{tp_size}: backend={dict(attention_impls)}, dtypes={dict(dtype_counts)}, "
        f"video={tuple(video.shape)}, audio={tuple(audio.shape)}, "
        f"trajectory={tuple(trajectory['all_latents'].shape)}, checksum={checksum.tolist()}",
        flush=True,
    )


def main() -> None:
    """Initialize CNCL/vLLM TP and run the rollout gate on both ranks."""
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    get_torch_device().set_device(local_rank)
    vllm_config = VllmConfig()
    with set_current_vllm_config(vllm_config):
        init_distributed_environment(local_rank=local_rank, backend=get_nccl_backend())
        initialize_model_parallel(tensor_model_parallel_size=world_size)
        try:
            run_rollout(rank, local_rank, world_size, vllm_config)
            if rank == 0:
                print("MiniMax H3 FlowGRPO rollout core TP2: PASS", flush=True)
        finally:
            destroy_model_parallel()
            destroy_distributed_environment()


if __name__ == "__main__":
    main()
