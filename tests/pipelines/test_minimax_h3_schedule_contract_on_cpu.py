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
"""Contract tests that pin MiniMax H3 FlowGRPO's step count to vLLM-Omni's.

Whether a sigma schedule holds ``num_steps`` or ``num_steps + 1`` boundaries is an upstream
convention that changed between vLLM-Omni releases. A rollout that derives its step count from
``num_steps`` then stops short of sigma 0 and returns partly noisy video. These tests compare our
rollout loop against upstream's own ``minimax_h3_denoise_loop`` so such a drift fails on CPU.
"""

from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
from vllm_omni.diffusion.models.minimax_h3.denoise_loop import minimax_h3_denoise_loop
from vllm_omni.diffusion.models.minimax_h3.time_request import minimax_h3_time_shift_sigmas

from verl_omni.pipelines.minimax_h3_flow_grpo.common import (
    H3_AUDIO_SHIFT,
    H3_AUDIO_WIDTH,
    H3_VIDEO_SHIFT,
    H3_VIDEO_WIDTH,
    h3_sigma_schedules,
    h3_transition_count,
    sample_h3_transition,
)
from verl_omni.pipelines.minimax_h3_flow_grpo.vllm_omni_rollout_adapter import MiniMaxH3PipelineWithLogProb
from verl_omni.pipelines.schedulers import flow_match_sde

STEP_COUNTS = [2, 5, 10, 40]
VIDEO_ROWS = 2
AUDIO_ROWS = 3


class _FakeBranch:
    """Minimal denoise branch shared by our rollout and upstream's loop."""

    def __init__(self, **kwargs):
        del kwargs
        self.img_pos = torch.arange(VIDEO_ROWS)
        self.audio_pos = torch.arange(VIDEO_ROWS, VIDEO_ROWS + AUDIO_ROWS)
        self.update_mask = torch.ones(VIDEO_ROWS, dtype=torch.bool)
        self.update_mask_dev = self.update_mask
        self.audio_update_mask = torch.ones(AUDIO_ROWS, dtype=torch.bool)
        self.audio_update_mask_dev = self.audio_update_mask
        self.locked_audio_rows = None

    def forward_kwargs(self, *, video_rows, audio_rows, **kwargs):
        del kwargs
        return {"hidden_states": video_rows, "audio_hidden_states": audio_rows}


class _CountingModel:
    """Zero-velocity DiT stand-in that counts evaluations."""

    def __init__(self):
        self.calls = 0

    def __call__(self, **model_inputs):
        self.calls += 1
        return torch.zeros_like(model_inputs["hidden_states"]), torch.zeros_like(model_inputs["audio_hidden_states"])


def _upstream_schedule(num_steps: int) -> tuple[list[float], list[float]]:
    return (
        minimax_h3_time_shift_sigmas(num_steps=num_steps, shift_scale=H3_VIDEO_SHIFT),
        minimax_h3_time_shift_sigmas(num_steps=num_steps, shift_scale=H3_AUDIO_SHIFT),
    )


@pytest.mark.parametrize("num_steps", STEP_COUNTS)
def test_schedule_runs_from_one_to_zero_and_matches_upstream(num_steps: int) -> None:
    video, audio = h3_sigma_schedules(num_steps)
    upstream_video, upstream_audio = _upstream_schedule(num_steps)

    assert video == upstream_video
    assert audio == upstream_audio
    assert video[0] == pytest.approx(1.0)
    assert video[-1] == pytest.approx(0.0)
    assert h3_transition_count(video, audio) == len(video) - 1


@pytest.mark.parametrize(
    ("video", "audio", "message"),
    [
        ([1.0, 0.5, 0.0], [1.0, 0.5], "differ in length"),
        ([1.0], [1.0], "at least two"),
        ([1.0, 0.5, 0.2], [1.0, 0.5, 0.0], "1.0 to 0.0"),
        ([0.9, 0.5, 0.0], [1.0, 0.5, 0.0], "1.0 to 0.0"),
        ([1.0, 0.5, 0.5, 0.0], [1.0, 0.5, 0.2, 0.0], "strictly decreasing"),
    ],
)
def test_transition_count_rejects_malformed_schedules(video, audio, message) -> None:
    with pytest.raises(RuntimeError, match=message):
        h3_transition_count(video, audio)


def _rollout_pipeline() -> MiniMaxH3PipelineWithLogProb:
    pipeline = object.__new__(MiniMaxH3PipelineWithLogProb)
    pipeline.device = torch.device("cpu")
    pipeline._flow_grpo_noise_level = 0.8
    pipeline._flow_grpo_sde_type = "cps"
    pipeline._flow_grpo_window_size = None
    pipeline._flow_grpo_window_range = None
    pipeline._flow_grpo_sde_contiguous = True
    pipeline._flow_grpo_seed = 123
    pipeline._h3_max_text_len = 2
    pipeline._initial_noise = MagicMock(
        return_value=(torch.zeros(VIDEO_ROWS, H3_VIDEO_WIDTH), torch.zeros(AUDIO_ROWS, H3_AUDIO_WIDTH))
    )
    pipeline.record_denoise_step = MagicMock()
    pipeline.progress_bar = lambda total: nullcontext(SimpleNamespace(update=MagicMock()))
    pipeline._resident_dit_layers_on_device = lambda enabled: nullcontext()
    pipeline._layout_outputs = MagicMock(return_value={})
    pipeline.transformer = _CountingModel()
    pipeline._transformer_for_task = MagicMock(return_value=pipeline.transformer)
    return pipeline


@pytest.mark.parametrize("num_steps", STEP_COUNTS)
def test_rollout_matches_upstream_loop_and_reaches_sigma_zero(monkeypatch, num_steps: int) -> None:
    module = "verl_omni.pipelines.minimax_h3_flow_grpo.vllm_omni_rollout_adapter"
    packed = {
        "token_tags": torch.zeros(VIDEO_ROWS + AUDIO_ROWS + 2, dtype=torch.long),
        "text_pos": torch.tensor([5, 6]),
    }
    boundaries: dict[str, list[tuple[float, float]]] = {"video": [], "audio": []}

    def recording_transition(scheduler, sample, _velocity, step, **kwargs):
        modality = "video" if sample.shape[-1] == H3_VIDEO_WIDTH else "audio"
        boundaries[modality].append((float(scheduler.sigmas[step]), float(scheduler.sigmas[step + 1])))
        log_prob = torch.zeros(1) if kwargs["return_log_prob"] else None
        return sample, log_prob, sample, torch.tensor(0.0), torch.tensor(0.0)

    monkeypatch.setattr(f"{module}.minimax_h3_packed_sequence", lambda **kwargs: packed)
    monkeypatch.setattr(f"{module}.MiniMaxH3DenoiseBranch", _FakeBranch)
    monkeypatch.setattr(f"{module}.sample_h3_transition", recording_transition)
    monkeypatch.setattr(f"{module}.minimax_h3_unpatchify_video_tokens", lambda *args, **kwargs: torch.zeros(1))
    monkeypatch.setattr(f"{module}.minimax_h3_unpack_audio_tokens", lambda *args, **kwargs: torch.zeros(1))

    pipeline = _rollout_pipeline()
    pipeline.diffuse(
        task="t2va",
        text_embeddings=torch.randn(2, 8),
        text_tags=torch.ones(2, dtype=torch.long),
        seed=7,
        latent_t=1,
        latent_h=4,
        latent_w=4,
        audio_t=3,
        num_frames=1,
        num_steps=num_steps,
        video_shift=H3_VIDEO_SHIFT,
        audio_shift=H3_AUDIO_SHIFT,
        visual_condition=None,
        visual_condition_shape=None,
        audio_condition=None,
        ref_audio_t=None,
    )

    upstream_model = _CountingModel()
    video_sigmas, audio_sigmas = _upstream_schedule(num_steps)
    minimax_h3_denoise_loop(
        model=upstream_model,
        positive=_FakeBranch(),
        initial_video_rows=torch.zeros(VIDEO_ROWS, H3_VIDEO_WIDTH),
        initial_audio_rows=torch.zeros(AUDIO_ROWS, H3_AUDIO_WIDTH),
        keyframe_cond_rows=None,
        sigmas_video=video_sigmas,
        sigmas_audio=audio_sigmas,
        device=torch.device("cpu"),
    )

    # Upstream's own loop defines how many denoiser evaluations a schedule means.
    assert upstream_model.calls == len(video_sigmas) - 1
    assert pipeline.transformer.calls == upstream_model.calls
    for modality in ("video", "audio"):
        assert len(boundaries[modality]) == upstream_model.calls
        # The walk ends exactly on sigma 0; stopping one boundary early is the failure this guards against.
        assert boundaries[modality][-1][1] == pytest.approx(0.0)
        assert boundaries[modality][0][0] == pytest.approx(1.0)


@pytest.mark.parametrize("deterministic_env", [None, "0", "true", "1"])
def test_cps_uses_one_request_seed_stream_including_zero_noise_draws(monkeypatch, deterministic_env):
    """Run the actual CPS scheduler and check device selection, draw order and reseeding."""
    if deterministic_env is None:
        monkeypatch.delenv("H3_DETERMINISTIC", raising=False)
    else:
        monkeypatch.setenv("H3_DETERMINISTIC", deterministic_env)
    module = "verl_omni.pipelines.minimax_h3_flow_grpo.vllm_omni_rollout_adapter"
    packed = {
        "token_tags": torch.zeros(VIDEO_ROWS + AUDIO_ROWS + 2, dtype=torch.long),
        "text_pos": torch.tensor([5, 6]),
    }
    monkeypatch.setattr(f"{module}.minimax_h3_packed_sequence", lambda **kwargs: packed)
    monkeypatch.setattr(f"{module}.MiniMaxH3DenoiseBranch", _FakeBranch)
    monkeypatch.setattr(f"{module}.minimax_h3_unpatchify_video_tokens", lambda rows, **kwargs: rows.clone())
    monkeypatch.setattr(f"{module}.minimax_h3_unpack_audio_tokens", lambda rows, **kwargs: rows.clone())
    pipeline = _rollout_pipeline()
    pipeline._flow_grpo_window_size = 1
    pipeline._flow_grpo_window_range = [1, 2]
    make_generator = torch.Generator
    generator_factory = MagicMock(wraps=make_generator)
    monkeypatch.setattr(torch, "Generator", generator_factory)
    randn_tensor = flow_match_sde.randn_tensor
    draws = []
    transitions = []

    def record_noise(shape, **kwargs):
        """Record real scheduler noise without changing generator consumption."""
        noise = randn_tensor(shape, **kwargs)
        draws.append((kwargs["generator"], noise.clone()))
        return noise

    def record_transition(scheduler, sample, velocity, step, **kwargs):
        """Observe modality and zero-noise calls while executing the real transition."""
        transitions.append((step, sample.shape[-1], kwargs["noise_level"]))
        return sample_h3_transition(scheduler, sample, velocity, step, **kwargs)

    monkeypatch.setattr(flow_match_sde, "randn_tensor", record_noise)
    monkeypatch.setattr(f"{module}.sample_h3_transition", record_transition)
    request_draws = []
    request_generators = []
    outputs = []
    for seed in (7, 7, 11):
        draws.clear()
        transitions.clear()
        generator_factory.reset_mock()
        outputs.append(
            pipeline.diffuse(
                task="t2va",
                text_embeddings=torch.zeros(2, 8),
                text_tags=torch.ones(2, dtype=torch.long),
                seed=seed,
                latent_t=1,
                latent_h=4,
                latent_w=4,
                audio_t=3,
                num_frames=1,
                num_steps=5,
                video_shift=H3_VIDEO_SHIFT,
                audio_shift=H3_AUDIO_SHIFT,
                visual_condition=None,
                visual_condition_shape=None,
                audio_condition=None,
                ref_audio_t=None,
            )
        )
        assert generator_factory.call_count == 2  # Window selection has its own unchanged CPU RNG.
        requested_device = generator_factory.call_args.kwargs["device"]
        if deterministic_env == "1":
            assert requested_device == "cpu"
        else:
            assert requested_device is pipeline.device
        sigmas = h3_sigma_schedules(5)
        num_transitions = h3_transition_count(*sigmas)
        assert transitions == [
            (step, width, 0.8 if step == 1 else 0.0)
            for step in range(num_transitions)
            for width in (H3_VIDEO_WIDTH, H3_AUDIO_WIDTH)
        ]
        assert len(draws) == 2 * num_transitions
        generator = draws[0][0]
        assert generator.device == torch.device("cpu")
        assert generator.initial_seed() == seed + 1
        reference = make_generator(device="cpu").manual_seed(seed + 1)
        for index, (draw_generator, noise) in enumerate(draws):
            assert draw_generator is generator
            expected_shape = (1, VIDEO_ROWS, H3_VIDEO_WIDTH) if index % 2 == 0 else (1, AUDIO_ROWS, H3_AUDIO_WIDTH)
            assert tuple(noise.shape) == expected_shape
            torch.testing.assert_close(noise, torch.randn(expected_shape, generator=reference), rtol=0, atol=0)
        assert torch.equal(generator.get_state(), reference.get_state())
        request_generators.append(generator)
        request_draws.append([noise for _, noise in draws])

    assert len({id(generator) for generator in request_generators}) == 3
    for first, repeated in zip(request_draws[0], request_draws[1], strict=True):
        torch.testing.assert_close(first, repeated, rtol=0, atol=0)
    assert not torch.equal(request_draws[0][0], request_draws[2][0])
    for first, repeated in zip(outputs[0], outputs[1], strict=True):
        torch.testing.assert_close(first, repeated, rtol=0, atol=0)
