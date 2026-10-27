"""Exercise fixed reward groups through the real diffusion dispatch paths."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf
from tensordict import NonTensorData, NonTensorStack, TensorDict
from verl.experimental.agent_loop.agent_loop import AgentLoopMetrics
from verl.protocol import DataProto
from verl.utils.skip import SkipManager

from verl_omni.agent_loop import diffusion_agent_loop_tq
from verl_omni.agent_loop.diffusion_agent_loop import (
    DiffusionAgentLoopManager,
    DiffusionAgentLoopWorker,
    _InternalDiffusionAgentLoopOutput,
)
from verl_omni.agent_loop.diffusion_agent_loop_tq import DiffusionAgentLoopManagerTQ, DiffusionAgentLoopWorkerTQ


def make_config(batch_size=4, n=2):
    """Supply the rollout and reward settings used by these dispatch paths."""
    return OmegaConf.create(
        {
            "reward": {"deterministic_batch_size": batch_size},
            "actor_rollout_ref": {
                "rollout": {
                    "n": n,
                    "pipeline": {},
                    "algo": {},
                    "calculate_log_probs": False,
                    "agent": {"default_agent_loop": "test"},
                    "val_kwargs": {"n": 1, "pipeline": {}, "algo": {}, "seed": 42},
                }
            },
        }
    )


def sample_output(index, reward=None):
    """Retain generated audio and source metadata through the scoring handoff."""
    return _InternalDiffusionAgentLoopOutput(
        prompt_ids=torch.tensor([[index, 0]]),
        response_diffusion_output=torch.full((1, 3, 2, 2), index, dtype=torch.uint8),
        reward_score=reward,
        metrics=AgentLoopMetrics(),
        extra_fields={"audio": torch.full((1, 2, 8), float(index)), "audio_sample_rate": 48000},
    )


def make_reward_handle(received):
    """Record the RPC payload and expose batch members as score metadata."""

    async def score(data):
        """Check real media projection and return ordered row scores."""
        ids = data.batch["prompts"][:, 0].tolist()
        received.append((ids, data.meta_info["validate"]))
        for i, fields in enumerate(data.non_tensor_batch["tool_extra_fields"]):
            assert fields["audio"].shape == (2, 8)
            assert fields["audio"].mean().item() == ids[i]
            assert fields["audio_sample_rate"] == 48000
        return [{"reward_score": float(i), "reward_extra_info": {"members": ",".join(map(str, ids))}} for i in ids]

    return SimpleNamespace(compute_score_batch=SimpleNamespace(remote=score))


def make_generation_worker(config, rewards, generate=None):
    """Use production grouping/postprocessing with a deterministic generator."""
    worker = object.__new__(DiffusionAgentLoopWorker)
    worker.config = config
    worker.rollout_config = config.actor_rollout_ref.rollout
    worker.reward_loop_worker_handles = rewards

    async def default_generate(params, *, sample, score_reward=True, **kwargs):
        """Simulate out-of-order completion without touching global RNG state."""
        del params, kwargs
        assert not score_reward
        await asyncio.sleep((3 - sample % 4) * 0.001)
        return sample_output(sample)

    worker._run_agent_loop = generate or default_generate
    return worker


@pytest.mark.parametrize("batch_size,size", [(None, 6), (4, 7)])
@pytest.mark.asyncio
async def test_dispatch_calls_skip_hook_once_without_reward_handles(monkeypatch, batch_size, size):
    """Default and deferred scoring retain one skip lifecycle and all input rows."""
    config = make_config(batch_size)

    async def generate(params, *, sample, score_reward=True, **kwargs):
        """Return unscored media for the existing post-rollout reward lifecycle."""
        del params, kwargs
        assert score_reward
        return sample_output(sample)

    workers = [make_generation_worker(config, None, generate) for _ in range(2)]
    manager = object.__new__(DiffusionAgentLoopManager)
    manager.config = config
    manager.agent_loop_workers = [
        SimpleNamespace(generate_sequences=SimpleNamespace(remote=w.generate_sequences)) for w in workers
    ]
    manager._performance_metrics = lambda metrics, output: {"rows": len(output)}
    skip = SimpleNamespace(
        is_enabled=lambda: True,
        support_online_step=False,
        steps={1},
        meet_precondition=Mock(return_value=False),
        prepare_data=Mock(),
    )
    monkeypatch.setattr(SkipManager, "skip_instances", {"rollout": skip})
    monkeypatch.setattr(SkipManager, "step", 1)
    inputs = DataProto.from_dict(
        tensors={"unused": torch.zeros(size)},
        non_tensors={"sample": np.arange(size)},
        meta_info={"global_steps": 1},
    )
    result = await manager.generate_sequences(inputs)
    assert result.batch["prompts"][:, 0].tolist() == list(range(size))
    assert "rm_scores" not in result.batch
    skip.meet_precondition.assert_called_once()
    skip.prepare_data.assert_called_once()


@pytest.mark.parametrize("worker_count", [1, 2, 3])
@pytest.mark.parametrize("size", [1, 7, 17])
@pytest.mark.parametrize("validate", [False, True])
@pytest.mark.asyncio
async def test_global_dispatch_preserves_membership_and_tail(worker_count, size, validate):
    """Worker count and completion order cannot redefine the scoring groups."""
    config = make_config()
    received = []
    rewards = [make_reward_handle(received), make_reward_handle(received)]
    manager = object.__new__(DiffusionAgentLoopManager)
    manager.config = config
    workers = [make_generation_worker(config, rewards) for _ in range(worker_count)]
    manager.agent_loop_workers = [
        SimpleNamespace(generate_sequences=SimpleNamespace(remote=w.generate_sequences)) for w in workers
    ]
    manager._performance_metrics = lambda metrics, output: {"rows": len(output)}
    inputs = DataProto.from_dict(
        tensors={"unused": torch.zeros(size)},
        non_tensors={"sample": np.arange(size)},
        meta_info={"global_steps": 1, "validate": validate, "rollout_seed": 42},
    )
    result = await manager.generate_sequences(inputs)
    assert result.batch["rm_scores"].flatten().tolist() == list(range(size))
    assert result.batch["audio"].shape == (size, 2, 8)
    assert sorted(received) == [(list(range(start, min(start + 4, size))), validate) for start in range(0, size, 4)]
    assert "reward_batch_index" not in inputs.meta_info
    assert result.non_tensor_batch["_rollout_seed_global_idx"].tolist() == list(range(size))


@pytest.mark.asyncio
async def test_complete_group_scores_while_other_group_is_generating():
    """A slow group must not act as a barrier for an already complete group."""
    config = make_config(2)
    slow_release = asyncio.Event()
    fast_scored = asyncio.Event()
    received = []
    reward = make_reward_handle(received)
    original_score = reward.compute_score_batch.remote

    async def score(data):
        """Release the slow generator only after observing the fast reward."""
        result = await original_score(data)
        if data.batch["prompts"][0, 0] == 0:
            fast_scored.set()
        return result

    reward.compute_score_batch.remote = score

    async def generate(params, *, sample, **kwargs):
        """Keep the second group's media unavailable until the test releases it."""
        del params, kwargs
        if sample >= 2:
            await slow_release.wait()
        return sample_output(sample)

    worker = make_generation_worker(config, [reward], generate)
    fast = asyncio.create_task(worker.generate_and_score_batch([({}, {"sample": i}) for i in (0, 1)], 0, False))
    slow = asyncio.create_task(worker.generate_and_score_batch([({}, {"sample": i}) for i in (2, 3)], 1, False))
    try:
        await asyncio.wait_for(fast_scored.wait(), timeout=2)
        assert not slow.done()
        assert received == [([0, 1], False)]
    finally:
        slow_release.set()
        await asyncio.gather(fast, slow)


@pytest.mark.asyncio
async def test_generation_failure_cancels_group_siblings_before_returning():
    """Failed fixed groups must not score a smaller, opportunistic batch."""
    started = asyncio.Event()
    cancelled = asyncio.Event()
    received = []

    async def generate(params, *, sample, **kwargs):
        """Fail one sample after its sibling has started waiting."""
        del params, kwargs
        if sample == 0:
            await started.wait()
            raise RuntimeError("generation failed")
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()

    worker = make_generation_worker(make_config(), [make_reward_handle(received)], generate)
    with pytest.raises(ExceptionGroup, match="TaskGroup"):
        await worker.generate_and_score_batch([({}, {"sample": i}) for i in (0, 1)], 0, False)
    assert cancelled.is_set()
    assert received == []


@pytest.mark.asyncio
async def test_precomputed_rewards_and_validation_metadata_are_preserved():
    """Agents may supply rewards; only unscored outputs reach the scorer."""

    async def generate(params, *, sample, **kwargs):
        """Supply one agent-owned score and one generated but unscored sample."""
        del params, kwargs
        return sample_output(sample, reward=9.0 if sample == 0 else None)

    received = []
    worker = make_generation_worker(make_config(), [make_reward_handle(received)], generate)
    outputs = await worker.generate_and_score_batch([({}, {"sample": i}) for i in (0, 1)], 0, True)
    assert [output.reward_score for output in outputs] == [9.0, 1.0]
    assert received == [([1], True)]


@pytest.mark.parametrize("precomputed", [False, True])
@pytest.mark.asyncio
async def test_group_collection_preserves_heterogeneous_reward_metadata(precomputed):
    """Collect optional failures and agent-owned scores like a whole batch."""
    config = make_config(2)

    async def generate(params, *, sample, **kwargs):
        """Optionally supply all rewards for the first group inside the agent."""
        del params, kwargs
        return sample_output(sample, 9.0 if precomputed and sample < 2 else None)

    async def score(data):
        """Only one fixed group reports an optional scorer error."""
        ids = data.batch["prompts"][:, 0].tolist()
        return [
            {"reward_score": float(i), "reward_extra_info": {"score": float(i), **({"errors": 1} if i < 2 else {})}}
            for i in ids
        ]

    worker = make_generation_worker(
        config, [SimpleNamespace(compute_score_batch=SimpleNamespace(remote=score))], generate
    )
    manager = object.__new__(DiffusionAgentLoopManager)
    manager.config = config
    manager.agent_loop_workers = [SimpleNamespace(generate_sequences=SimpleNamespace(remote=worker.generate_sequences))]
    manager._performance_metrics = lambda metrics, output: {}
    inputs = DataProto.from_dict(
        tensors={"unused": torch.zeros(4)}, non_tensors={"sample": np.arange(4)}, meta_info={"global_steps": 1}
    )
    result = await manager.generate_sequences(inputs)
    assert result.batch["rm_scores"].flatten().tolist() == (
        [9.0, 9.0, 2.0, 3.0] if precomputed else [0.0, 1.0, 2.0, 3.0]
    )
    assert result.meta_info["reward_extra_keys"] == ([] if precomputed else ["score"])
    assert "errors" not in result.non_tensor_batch
    assert result.non_tensor_batch["reward_extra_info"][0] == (None if precomputed else {"score": 0.0, "errors": 1})


@pytest.mark.parametrize("batch_size,n", [(16, 8), (6, 8), (3, 2), (1, 4)])
@pytest.mark.parametrize("worker_count", [1, 3])
@pytest.mark.parametrize("validate", [False, True])
def test_tq_dispatch_aligns_prompt_and_reward_boundaries(monkeypatch, batch_size, n, worker_count, validate):
    """Arbitrary reward sizes preserve complete prompts and global batch order."""
    config = make_config(batch_size, n)
    manager = object.__new__(DiffusionAgentLoopManagerTQ)
    manager.config = config
    manager.rollout_config = config.actor_rollout_ref.rollout
    manager.reward_loop_worker_handles = [object()]
    chunks = []
    manager.agent_loop_workers = [
        SimpleNamespace(generate_sequences=SimpleNamespace(remote=chunks.append)) for _ in range(worker_count)
    ]
    monkeypatch.setattr(diffusion_agent_loop_tq.ray, "get", lambda refs: refs)
    count = 7
    prompts = TensorDict({"index": torch.arange(count), "validate": NonTensorData(validate)}, batch_size=[count])
    manager.generate_sequences(prompts)
    n = 1 if validate else n
    observed = []
    for chunk in chunks:
        first = int(chunk["index"][0]) * n
        assert first % batch_size == 0
        assert chunk["reward_batch_index"] == first // batch_size
        rows = [int(i) * n + session for i in chunk["index"] for session in range(n)]
        observed.extend(rows[start : start + batch_size] for start in range(0, len(rows), batch_size))
    assert observed == [
        list(range(start, min(start + batch_size, count * n))) for start in range(0, count * n, batch_size)
    ]
    assert "reward_batch_index" not in prompts


@pytest.mark.asyncio
@pytest.mark.parametrize("validate", [False, True])
@pytest.mark.parametrize("fail", [False, True])
async def test_tq_worker_scores_groups_and_preserves_prompt_publication(monkeypatch, validate, fail):
    """Run the TQ entrypoint through reward scoring and its existing write hook."""
    config = make_config(3, 2)
    worker_cls = DiffusionAgentLoopWorkerTQ.__ray_metadata__.modified_class
    worker = object.__new__(worker_cls)
    worker.config = config
    worker.rollout_config = config.actor_rollout_ref.rollout
    worker.background_tasks = set()
    received = []
    worker.reward_loop_worker_handles = [make_reward_handle(received)]
    lifecycle = []
    seeds = []

    async def kv_put(*, key, partition_id, tag):
        """Record prompt state transitions at the storage boundary."""
        lifecycle.append((key, partition_id, tag["status"]))

    async def generate(params, *, index, session_id, score_reward, **kwargs):
        """Identify expanded rows and capture globally assigned rollout seeds."""
        del kwargs
        assert not score_reward
        if fail and int(index) == 1:
            raise RuntimeError("generation failed")
        seeds.append(params["seed"])
        return sample_output(int(index) * (1 if validate else 2) + session_id)

    async def trajectories(step, indices, is_validate):
        """Return the same metadata shape as the tracing API."""
        return [{"step": step, "validate": is_validate} for _ in indices]

    monkeypatch.setattr(diffusion_agent_loop_tq.tq, "async_kv_put", kv_put)
    monkeypatch.setattr(diffusion_agent_loop_tq, "get_trajectory_info", trajectories)
    monkeypatch.setattr(diffusion_agent_loop_tq, "_config_to_sampling_dict", dict)
    worker._run_agent_loop = generate
    worker._write_trajectories_to_tq = AsyncMock()
    batch = TensorDict(
        {
            "index": torch.arange(3),
            "uid": NonTensorStack("a", "b", "c"),
            "rollout_seed": NonTensorData(42),
            "global_steps": NonTensorData(1),
            "validate": NonTensorData(validate),
            "reward_batch_index": NonTensorData(0),
        },
        batch_size=[3],
    )
    await worker.generate_sequences(batch)
    await asyncio.gather(*worker.background_tasks)
    if fail:
        worker._write_trajectories_to_tq.assert_not_awaited()
        assert all(state == "failure" for _, _, state in lifecycle[-3:])
        return
    n = 1 if validate else 2
    assert received == [(list(range(start, start + 3)), validate) for start in range(0, 3 * n, 3)]
    assert worker._write_trajectories_to_tq.await_count == 3
    for i, call in enumerate(worker._write_trajectories_to_tq.await_args_list):
        assert call.kwargs["uid"] == "abc"[i]
        assert [(session, row.reward_score) for session, row in call.args[0]] == [(j, i * n + j) for j in range(n)]
    assert all(state == "finished" for _, _, state in lifecycle[-3:])
    assert len(set(seeds)) == (1 if validate else 6)
