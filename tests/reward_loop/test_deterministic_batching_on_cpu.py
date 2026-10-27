"""Fixed reward batches across worker counts, completion order and failures."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
import torch
from omegaconf import OmegaConf
from verl import DataProto

from verl_omni.reward_loop.reward_loop import OmniRewardLoopManager, OmniRewardLoopWorker, score_fixed_batches
from verl_omni.reward_loop.reward_manager.multi import MultiRewardManager
from verl_omni.workers.config.reward import (
    deterministic_reward_batch_size,
    reward_is_enabled,
    reward_role_required,
    streaming_reward_enabled,
)


def scalar_score(sample, offset=0):
    """Return a scalar score using filtered, configured arguments."""
    return sample + offset


def batch_score(requests):
    """Expose batch membership in both the score and diagnostic metadata."""
    members = ",".join(str(request["sample"]) for request in requests)
    return [{"score": scalar_score(**request) + len(requests), "members": members} for request in requests]


async def async_batch_score(requests):
    """Exercise the coroutine batch-function contract."""
    return batch_score(requests)


def failing_batch(requests):
    """Report a sample-local failure at its original position."""
    return [ValueError("bad row") if request["sample"] == 1 else 2.0 for request in requests]


def raising_batch(requests):
    """Report one inference failure affecting the whole batch."""
    raise RuntimeError("batch inference failed")


def incomplete_batch(requests):
    """Violate the row-count contract to exercise explicit rejection."""
    return [0.0]


class SampleRewardManager(MultiRewardManager):
    """Minimal input adapter exercising real multi-reward execution."""

    async def _build_reward_kwargs(self, data_item):
        """Pass sample identity and an unused key through signature filtering."""
        return {"sample": int(data_item.batch["sample"]), "unused": "must be filtered"}

    @classmethod
    def assemble_rm_scores(cls, data, scores):
        """Return one scalar per original input row."""
        return torch.tensor(scores, dtype=torch.float32).unsqueeze(-1)


def make_config(batch_size=16, batch_name="batch_score", required=True):
    """Create only the configuration owned by this reward path."""
    return OmegaConf.create(
        {
            "reward": {
                "deterministic_batch_size": batch_size,
                "reward_model": {"enable": False, "enable_resource_pool": False},
                "models": {},
                "reward_functions": {
                    "batched": {
                        "path": __file__,
                        "name": "scalar_score",
                        "batch_name": batch_name,
                        "offset": 10,
                        "weight": 0.5,
                        "required": required,
                    },
                    "scalar": {"path": __file__, "name": "scalar_score", "offset": 2, "weight": 1.0},
                    "observe": {"path": __file__, "name": "scalar_score", "weight": 0.0},
                },
            }
        }
    )


def sample_data(size):
    """Build immutable, position-identified inputs without padding rows."""
    return DataProto.from_dict(tensors={"sample": torch.arange(size)})


def make_worker(config, delay=0):
    """Run the real worker method behind an awaitable RPC test boundary."""
    worker = object.__new__(OmniRewardLoopWorker)
    worker.config = config
    worker.reward_manager = SampleRewardManager(config, None, None)

    async def compute(data):
        """Vary completion order without changing the assigned data."""
        await asyncio.sleep(delay)
        return await worker.compute_score_batch(data)

    return SimpleNamespace(compute_score_batch=SimpleNamespace(remote=compute))


@pytest.mark.parametrize("workers", [1, 2, 3, 8])
@pytest.mark.parametrize("size", [0, 1, 15, 16, 17, 33, 65])
@pytest.mark.asyncio
async def test_fixed_membership_and_output_order(workers, size):
    """Changing workers cannot change batches, including the unpadded tail."""
    config = make_config()
    handles = [make_worker(config, (workers - index) * 0.001) for index in range(workers)]
    results = await score_fixed_batches(sample_data(size), handles, 16)
    assert len(results) == size
    for index, result in enumerate(results):
        start = index // 16 * 16
        end = min(start + 16, size)
        info = result["reward_extra_info"]
        assert info["reward/batched/members"] == ",".join(map(str, range(start, end)))
        assert info["reward/observe"] == index
        assert result["reward_score"] == 0.5 * (index + 10 + end - start) + index + 2
        assert info["reward/combined"] == result["reward_score"]


@pytest.mark.parametrize("batch_name", ["batch_score", "async_batch_score", None])
@pytest.mark.asyncio
async def test_batch_hook_and_scalar_compatibility(batch_name):
    """Both hook kinds share aggregation; terms without hooks remain scalar."""
    manager = SampleRewardManager(make_config(batch_name=batch_name), None, None)
    results = await manager.run_batch(sample_data(3))
    increment = 3 if batch_name else 0
    assert [item["reward_score"] for item in results] == [0.5 * (i + 10 + increment) + i + 2 for i in range(3)]
    single = await manager.run_single(sample_data(1))
    assert single["reward_score"] == 7
    assert "reward/batched/members" not in single["reward_extra_info"]


@pytest.mark.parametrize("batch_name,failed_rows", [("failing_batch", [1]), ("raising_batch", [0, 1, 2])])
@pytest.mark.asyncio
async def test_optional_and_required_batch_failures(batch_name, failed_rows):
    """Optional failures stay sample-local; required failures abort scoring."""
    config = make_config(batch_name=batch_name, required=False)
    manager = SampleRewardManager(config, None, None)
    results = await manager.run_batch(sample_data(3))
    for index, result in enumerate(results):
        info = result["reward_extra_info"]
        assert info["reward/batched"] == (0 if index in failed_rows else 2)
        assert info.get("reward/batched/errors", 0) == int(index in failed_rows)
        assert info["reward/scalar"] == index + 2
    config.reward.reward_functions.batched.required = True
    with pytest.raises(RuntimeError, match="Required sub-reward 'batched' failed"):
        await SampleRewardManager(config, None, None).run_batch(sample_data(3))


@pytest.mark.asyncio
async def test_incomplete_batch_is_never_silently_truncated():
    """A broken batch interface is a contract error even for optional terms."""
    manager = SampleRewardManager(make_config(batch_name="incomplete_batch", required=False), None, None)
    with pytest.raises(ValueError, match="returned 1 results for 3 requests"):
        await manager.run_batch(sample_data(3))


@pytest.mark.parametrize("size", [0, -1, 1.5, True, "16"])
def test_invalid_fixed_batch_size_is_rejected(size):
    """Reject malformed public configuration before worker allocation."""
    with pytest.raises(ValueError, match="positive integer or null"):
        deterministic_reward_batch_size(make_config(batch_size=size))


def test_fixed_batch_scoring_preserves_custom_reward_streaming():
    """Fixed custom reward batches stream without allocating a model pool."""
    config = make_config()
    assert reward_is_enabled(config)
    assert streaming_reward_enabled(config)
    assert not reward_role_required(config)
    config.reward.deterministic_batch_size = None
    assert not reward_is_enabled(config)
    assert streaming_reward_enabled(config)


@pytest.mark.asyncio
async def test_complete_batch_manager_assembles_rows_and_lifecycle():
    """Exercise the public async entrypoint used after rollout completes."""
    config = make_config()
    manager = object.__new__(OmniRewardLoopManager)
    manager.config = config
    manager._score_lock = asyncio.Lock()
    manager.multi_reward_model_manager = SimpleNamespace(models={}, wake_up=AsyncMock(), sleep=AsyncMock())
    manager.reward_model_manager = None
    manager.reward_manager_cls = SampleRewardManager
    manager.reward_loop_workers = [make_worker(config), make_worker(config)]
    manager._reward_worker_groups = {"legacy": manager.reward_loop_workers}
    result = await manager.async_compute_rm_score(sample_data(33))
    assert result.batch["rm_scores"].shape == (33, 1)
    assert result.batch["rm_scores"][-1].item() == 0.5 * (32 + 10 + 1) + 32 + 2
    assert result.non_tensor_batch["reward/observe"].tolist() == list(range(33))
    manager.multi_reward_model_manager.wake_up.assert_awaited_once()
    manager.multi_reward_model_manager.sleep.assert_awaited_once()
