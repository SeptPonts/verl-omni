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

"""Exercise the MiniMax H3 reward cores without checkpoint weights.

This gate uses production audio/video preprocessing and small, structurally
equivalent CLAP and ImageBind models.  It proves the Python dependency and
accelerator operator paths, not the released checkpoints or reward quality.
"""

from __future__ import annotations

import argparse
import copy

import torch
from imagebind.models.imagebind_model import ImageBindModel, ModalityType, imagebind_huge
from torchaudio import functional as audio_functional
from transformers import ClapAudioConfig, ClapConfig, ClapFeatureExtractor, ClapModel, ClapTextConfig
from verl.utils.device import get_device_name, get_torch_device

from verl_omni.utils.reward_score.clap import _get_audio as get_clap_audio
from verl_omni.utils.reward_score.imagebind import (
    _compute_similarities as compute_imagebind_similarities,
    _preprocess_audio as preprocess_imagebind_audio,
    _preprocess_video as preprocess_imagebind_video,
)

RECIPE_HEIGHT = 256
RECIPE_WIDTH = 384
RECIPE_NUM_FRAMES = 121
IMAGEBIND_SAMPLE_RATE = 16_000
CLAP_SAMPLE_RATE = 48_000


def build_tiny_imagebind() -> ImageBindModel:
    """Build a cheap model that preserves ImageBind's production module types."""
    return ImageBindModel(
        out_embed_dim=32,
        vision_embed_dim=64,
        vision_num_blocks=2,
        vision_num_heads=4,
        audio_embed_dim=64,
        audio_num_blocks=2,
        audio_num_heads=4,
        text_embed_dim=32,
        text_num_blocks=1,
        text_num_heads=4,
        depth_embed_dim=32,
        depth_num_blocks=1,
        depth_num_heads=4,
        thermal_embed_dim=32,
        thermal_num_blocks=1,
        thermal_num_heads=4,
        imu_embed_dim=32,
        imu_num_blocks=1,
        imu_num_heads=4,
        audio_drop_path=0.0,
        imu_drop_path=0.0,
    ).eval()


def make_recipe_inputs() -> dict[str, torch.Tensor]:
    """Run production preprocessing on deterministic H3-shaped media."""
    generator = torch.Generator(device="cpu").manual_seed(20260903)
    audio = torch.randn(2 * IMAGEBIND_SAMPLE_RATE, generator=generator)
    video = torch.randint(
        0,
        256,
        (RECIPE_NUM_FRAMES, RECIPE_HEIGHT, RECIPE_WIDTH, 3),
        generator=generator,
        dtype=torch.uint8,
    )
    return {
        ModalityType.AUDIO: preprocess_imagebind_audio(audio, IMAGEBIND_SAMPLE_RATE, "cpu"),
        ModalityType.VISION: preprocess_imagebind_video(video, "cpu"),
    }


def run_imagebind_reward_core(device: torch.device) -> None:
    """Compare the same ImageBind reward graph on CPU and the accelerator."""
    torch.manual_seed(20260903)
    cpu_model = build_tiny_imagebind()
    accelerator_model = copy.deepcopy(cpu_model).to(device)
    cpu_inputs = make_recipe_inputs()
    accelerator_inputs = {name: value.to(device) for name, value in cpu_inputs.items()}

    with torch.no_grad():
        cpu_outputs = cpu_model(cpu_inputs)
        accelerator_outputs = accelerator_model(accelerator_inputs)
    accelerator_outputs_cpu = {name: value.cpu() for name, value in accelerator_outputs.items()}

    for modality in (ModalityType.AUDIO, ModalityType.VISION):
        assert torch.isfinite(accelerator_outputs_cpu[modality]).all()
        torch.testing.assert_close(
            accelerator_outputs_cpu[modality],
            cpu_outputs[modality],
            rtol=5e-3,
            atol=5e-3,
        )

    cpu_score = compute_imagebind_similarities(cpu_outputs, ModalityType)["audio_video"]
    accelerator_score = compute_imagebind_similarities(accelerator_outputs_cpu, ModalityType)["audio_video"]
    if abs(cpu_score - accelerator_score) > 5e-3:
        raise AssertionError(
            f"ImageBind audio/video score mismatch: CPU={cpu_score}, {device.type}={accelerator_score}."
        )

    parameter_count = sum(parameter.numel() for parameter in cpu_model.parameters())
    print(
        f"reward=imagebind, device={device.type}, parameters={parameter_count}, "
        f"audio_input={tuple(cpu_inputs[ModalityType.AUDIO].shape)}, "
        f"video_input={tuple(cpu_inputs[ModalityType.VISION].shape)}, "
        f"cpu_score={cpu_score:.8f}, accelerator_score={accelerator_score:.8f}",
        flush=True,
    )


def build_tiny_clap() -> ClapModel:
    """Build a cheap model that preserves CLAP's production module types."""
    audio_config = ClapAudioConfig(
        window_size=2,
        num_mel_bins=64,
        spec_size=64,
        patch_size=4,
        patch_stride=(4, 4),
        hidden_size=64,
        projection_dim=16,
        depths=(1, 1, 1, 1),
        num_attention_heads=(2, 2, 4, 8),
        patch_embeds_hidden_size=8,
        drop_path_rate=0.0,
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
    )
    text_config = ClapTextConfig(
        vocab_size=100,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=64,
        max_position_embeddings=32,
        projection_dim=16,
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
    )
    return ClapModel(ClapConfig(text_config=text_config, audio_config=audio_config, projection_dim=16)).eval()


def make_clap_inputs() -> dict[str, torch.Tensor]:
    """Run the production waveform normalization and resampling path."""
    generator = torch.Generator(device="cpu").manual_seed(20260903)
    raw_audio = torch.randn(IMAGEBIND_SAMPLE_RATE // 2, generator=generator)
    waveform, source_rate = get_clap_audio(
        {
            "audio": raw_audio,
            "audio_sample_rate": IMAGEBIND_SAMPLE_RATE,
        }
    )
    waveform = audio_functional.resample(
        waveform.unsqueeze(0),
        orig_freq=source_rate,
        new_freq=CLAP_SAMPLE_RATE,
    ).squeeze(0)
    feature_extractor = ClapFeatureExtractor(
        max_length_s=0.5,
        truncation="rand_trunc",
        padding="repeatpad",
    )
    features = feature_extractor(waveform.numpy(), sampling_rate=CLAP_SAMPLE_RATE, return_tensors="pt")
    input_ids = torch.tensor([[0, 5, 7, 2]], dtype=torch.long)
    return {
        "input_features": features["input_features"],
        "is_longer": features["is_longer"],
        "input_ids": input_ids,
        "attention_mask": torch.ones_like(input_ids),
    }


def run_clap_reward_core(device: torch.device) -> None:
    """Compare the same CLAP reward graph on CPU and the accelerator."""
    torch.manual_seed(20260903)
    cpu_model = build_tiny_clap()
    accelerator_model = copy.deepcopy(cpu_model).to(device)
    cpu_inputs = make_clap_inputs()
    accelerator_inputs = {name: value.to(device) for name, value in cpu_inputs.items()}

    with torch.no_grad():
        cpu_outputs = cpu_model(**cpu_inputs)
        accelerator_outputs = accelerator_model(**accelerator_inputs)
    accelerator_audio = accelerator_outputs.audio_embeds.cpu()
    accelerator_text = accelerator_outputs.text_embeds.cpu()
    torch.testing.assert_close(accelerator_audio, cpu_outputs.audio_embeds, rtol=5e-3, atol=5e-3)
    torch.testing.assert_close(accelerator_text, cpu_outputs.text_embeds, rtol=5e-3, atol=5e-3)

    cpu_score = (cpu_outputs.audio_embeds * cpu_outputs.text_embeds).sum(dim=-1).item()
    accelerator_score = (accelerator_audio * accelerator_text).sum(dim=-1).item()
    if abs(cpu_score - accelerator_score) > 5e-3:
        raise AssertionError(f"CLAP score mismatch: CPU={cpu_score}, {device.type}={accelerator_score}.")

    parameter_count = sum(parameter.numel() for parameter in cpu_model.parameters())
    print(
        f"reward=clap, device={device.type}, parameters={parameter_count}, "
        f"audio_input={tuple(cpu_inputs['input_features'].shape)}, "
        f"text_input={tuple(cpu_inputs['input_ids'].shape)}, "
        f"cpu_score={cpu_score:.8f}, accelerator_score={accelerator_score:.8f}",
        flush=True,
    )


def run_full_imagebind(device: torch.device) -> None:
    """Run the released ImageBind-Huge architecture with random FP32 weights."""
    model = imagebind_huge(pretrained=False).eval().to(device)
    cpu_inputs = make_recipe_inputs()
    accelerator_inputs = {name: value.to(device) for name, value in cpu_inputs.items()}
    with torch.no_grad():
        outputs = model(accelerator_inputs)
    outputs_cpu = {name: value.cpu() for name, value in outputs.items()}
    for value in outputs_cpu.values():
        assert torch.isfinite(value).all()
    score = compute_imagebind_similarities(outputs_cpu, ModalityType)["audio_video"]
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    parameter_bytes = sum(parameter.numel() * parameter.element_size() for parameter in model.parameters())
    print(
        f"reward=imagebind-huge-random, device={device.type}, parameters={parameter_count}, "
        f"parameter_bytes={parameter_bytes}, score={score:.8f}",
        flush=True,
    )


def main() -> None:
    """Run both reward gates on the active accelerator."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--full-imagebind",
        action="store_true",
        help="Also run the full ImageBind-Huge architecture with random weights.",
    )
    args = parser.parse_args()
    device_name = get_device_name()
    if device_name == "cpu":
        raise RuntimeError("This gate requires an accelerator device.")
    get_torch_device().set_device(0)
    device = torch.device(f"{device_name}:0")
    run_imagebind_reward_core(device)
    run_clap_reward_core(device)
    if args.full_imagebind:
        get_torch_device().empty_cache()
        run_full_imagebind(device)
    print("MiniMax H3 reward cores: PASS", flush=True)


if __name__ == "__main__":
    main()
