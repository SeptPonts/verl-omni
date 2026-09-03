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

"""Run real MiniMax H3 CLAP and ImageBind checkpoint rewards.

Model paths are explicit and external so this gate can run fully offline and
never turns a public model identifier into an implicit runtime download.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
from pathlib import Path

os.environ.setdefault(
    "VERL_OMNI_AUTO_IMPORT_PIPELINES",
    "minimax_h3_diffusion_nft,minimax_h3_flow_grpo",
)

import torch
from omegaconf import OmegaConf
from verl import DataProto

from verl_omni.reward_loop.reward_manager.multi import MultiVisualRewardManager
from verl_omni.utils.reward_score import clap as clap_reward
from verl_omni.utils.reward_score import imagebind as imagebind_reward

H3_AUDIO_SAMPLE_RATE = 32_000
H3_ALIGNED_FRAMES = 124
H3_FRAME_RATE = 24
RECIPE_NUM_FRAMES = 121
RECIPE_HEIGHT = 256
RECIPE_WIDTH = 384


def make_audio_samples() -> list[tuple[str, torch.Tensor]]:
    """Create deterministic, distinct H3-duration audio/prompt pairs."""
    sample_count = int(H3_AUDIO_SAMPLE_RATE * H3_ALIGNED_FRAMES / H3_FRAME_RATE)
    time = torch.arange(sample_count, dtype=torch.float32) / H3_AUDIO_SAMPLE_RATE
    tone = 0.5 * torch.sin(2 * math.pi * 440 * time)
    generator = torch.Generator(device="cpu").manual_seed(20260903)
    noise = 0.15 * torch.randn(sample_count, generator=generator)
    return [
        ("a sustained pure musical tone", tone),
        ("broadband static noise", noise),
    ]


def make_video_samples() -> list[torch.Tensor]:
    """Create deterministic, distinct uint8 videos at the H3 recipe geometry."""
    ramp = torch.linspace(0, 255, RECIPE_NUM_FRAMES, dtype=torch.float32).to(torch.uint8)
    first = torch.zeros((RECIPE_NUM_FRAMES, RECIPE_HEIGHT, RECIPE_WIDTH, 3), dtype=torch.uint8)
    first[..., 0] = ramp.view(-1, 1, 1)
    first[..., 2] = torch.flip(ramp, dims=(0,)).view(-1, 1, 1)

    generator = torch.Generator(device="cpu").manual_seed(20260904)
    second = torch.randint(
        0,
        256,
        first.shape,
        dtype=torch.uint8,
        generator=generator,
    )
    return [first, second]


async def score_clap(model_path: str, device: str, audio_samples: list[tuple[str, torch.Tensor]]) -> list[dict]:
    """Score a burst through the production CLAP batching consumer."""
    requests = [
        clap_reward.compute_score(
            data_source="h3-real-reward-gate",
            solution_image=None,
            ground_truth=prompt,
            extra_info={"audio": audio, "audio_sample_rate": H3_AUDIO_SAMPLE_RATE},
            device=device,
            model_name_or_path=model_path,
        )
        for prompt, audio in audio_samples
    ]
    results = await asyncio.gather(*requests)
    state = clap_reward._get_batching_state()
    await state.queue.put((None, None, None, None, None))
    await state.consumer_task
    return results


def score_imagebind(
    model_path: str,
    device: str,
    audio_samples: list[tuple[str, torch.Tensor]],
    videos: list[torch.Tensor],
) -> list[dict]:
    """Score two samples through the production ImageBind reward."""
    return [
        imagebind_reward.compute_score(
            data_source="h3-real-reward-gate",
            solution_image=video,
            ground_truth=prompt,
            extra_info={"audio": audio, "audio_sample_rate": H3_AUDIO_SAMPLE_RATE},
            device=device,
            model_name_or_path=model_path,
            mode="audio_video",
        )
        for (prompt, audio), video in zip(audio_samples, videos, strict=True)
    ]


def require_finite_nonconstant(name: str, results: list[dict]) -> list[float]:
    """Require two finite scores with a measurable difference."""
    scores = [float(result["score"]) for result in results]
    if not all(math.isfinite(score) for score in scores):
        raise AssertionError(f"{name} produced non-finite scores: {scores}.")
    if abs(scores[0] - scores[1]) <= 1e-6:
        raise AssertionError(f"{name} produced constant scores: {scores}.")
    return scores


def capture_mlu_state() -> dict:
    """Capture logical visibility and instantaneous memory without treating it as a benchmark."""
    devices = []
    for index in range(torch.mlu.device_count()):
        free_bytes, total_bytes = torch.mlu.mem_get_info(index)
        devices.append(
            {
                "free_bytes": free_bytes,
                "logical_index": index,
                "name": torch.mlu.get_device_name(index),
                "total_bytes": total_bytes,
            }
        )
    return {
        "devices": devices,
        "mlu_device_count": torch.mlu.device_count(),
        "mlu_visible_devices": os.environ.get("MLU_VISIBLE_DEVICES"),
    }


async def run_manager_batch(manager: MultiVisualRewardManager, data: DataProto) -> list[dict]:
    """Run both H3 samples concurrently through the production reward manager."""
    return await asyncio.gather(*(manager.run_single(data[index : index + 1]) for index in range(len(data))))


async def stop_manager_clap_consumer(manager: MultiVisualRewardManager) -> None:
    """Stop the dynamically loaded CLAP batching task before the process exits."""
    clap_fn = next(sub_reward["fn"] for sub_reward in manager._sub_rewards if sub_reward["key"] == "clap")
    state = clap_fn.__globals__["_get_batching_state"]()
    await state.queue.put((None, None, None, None, None))
    await state.consumer_task


def score_multi_reward_manager(
    clap_model_path: str,
    imagebind_model_path: str,
    clap_device: str,
    imagebind_device: str,
    audio_samples: list[tuple[str, torch.Tensor]],
    videos: list[torch.Tensor],
) -> dict:
    """Run real CLAP, ImageBind, and weighted aggregation through the recipe manager."""
    config = OmegaConf.create(
        {
            "actor_rollout_ref": {
                "rollout": {
                    "pipeline": {"output_type": "video"},
                    "val_kwargs": {"pipeline": {"output_type": "video"}},
                }
            },
            "reward": {
                "reward_functions": {
                    "clap": {
                        "device": clap_device,
                        "model_name_or_path": clap_model_path,
                        "name": "compute_score",
                        "path": str(Path(clap_reward.__file__).resolve()),
                        "required": True,
                        "weight": 1.0,
                    },
                    "imagebind": {
                        "device": imagebind_device,
                        "mode": "audio_video",
                        "model_name_or_path": imagebind_model_path,
                        "name": "compute_score",
                        "path": str(Path(imagebind_reward.__file__).resolve()),
                        "required": True,
                        "weight": 1.0,
                    },
                }
            },
        }
    )
    manager = MultiVisualRewardManager(config, tokenizer=None, compute_score=None)
    data = DataProto.from_dict(
        tensors={"responses": torch.stack(videos)},
        non_tensors={
            "data_source": ["h3-real-reward-gate"] * len(videos),
            "extra_info": [
                {"audio": audio, "audio_sample_rate": H3_AUDIO_SAMPLE_RATE} for _, audio in audio_samples
            ],
            "reward_model": [{"ground_truth": prompt} for prompt, _ in audio_samples],
        },
    )
    results = manager.loop.run_until_complete(run_manager_batch(manager, data))
    manager.loop.run_until_complete(stop_manager_clap_consumer(manager))

    clap_scores = [float(result["reward_extra_info"]["reward/clap"]) for result in results]
    imagebind_scores = [float(result["reward_extra_info"]["reward/imagebind"]) for result in results]
    combined_scores = [float(result["reward_score"]) for result in results]
    require_finite_nonconstant("manager CLAP", [{"score": score} for score in clap_scores])
    require_finite_nonconstant("manager ImageBind", [{"score": score} for score in imagebind_scores])
    require_finite_nonconstant("manager combined", [{"score": score} for score in combined_scores])
    for clap_score, imagebind_score, combined_score in zip(
        clap_scores,
        imagebind_scores,
        combined_scores,
        strict=True,
    ):
        if not math.isclose(combined_score, clap_score + imagebind_score, rel_tol=0.0, abs_tol=1e-7):
            raise AssertionError(
                f"weighted sum mismatch: {combined_score} != {clap_score} + {imagebind_score}."
            )
    return {
        "clap_scores": clap_scores,
        "combined_scores": combined_scores,
        "imagebind_scores": imagebind_scores,
    }


def main() -> None:
    """Load the requested external checkpoints and run production rewards."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--clap-model")
    parser.add_argument("--imagebind-model")
    parser.add_argument("--clap-device", default="mlu:0")
    parser.add_argument("--imagebind-device", default="mlu:1")
    parser.add_argument("--manager", action="store_true")
    args = parser.parse_args()
    if not args.clap_model and not args.imagebind_model:
        parser.error("at least one checkpoint path is required")
    if args.manager and (not args.clap_model or not args.imagebind_model):
        parser.error("--manager requires both --clap-model and --imagebind-model")

    audio_samples = make_audio_samples()
    evidence = {"runtime_before": capture_mlu_state()}
    if args.manager:
        evidence["manager"] = score_multi_reward_manager(
            args.clap_model,
            args.imagebind_model,
            args.clap_device,
            args.imagebind_device,
            audio_samples,
            make_video_samples(),
        )
    else:
        if args.clap_model:
            clap_results = asyncio.run(score_clap(args.clap_model, args.clap_device, audio_samples))
            evidence["clap"] = {
                "device": args.clap_device,
                "model_path": args.clap_model,
                "scores": require_finite_nonconstant("CLAP", clap_results),
                "source_sample_rates": [result["source_sample_rate"] for result in clap_results],
            }
        if args.imagebind_model:
            imagebind_results = score_imagebind(
                args.imagebind_model,
                args.imagebind_device,
                audio_samples,
                make_video_samples(),
            )
            evidence["imagebind"] = {
                "device": args.imagebind_device,
                "model_path": args.imagebind_model,
                "scores": require_finite_nonconstant("ImageBind", imagebind_results),
            }
    evidence["runtime_after"] = capture_mlu_state()

    print(json.dumps(evidence, indent=2, sort_keys=True), flush=True)
    print("MiniMax H3 real checkpoint rewards: PASS", flush=True)


if __name__ == "__main__":
    main()
