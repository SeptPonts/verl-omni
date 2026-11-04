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
"""Exercise real V0 generation/postprocess and evidence callbacks without Ray.

Standalone packages bypass model registration. AST loading keeps production
method bodies; only the inference endpoint and TensorDict/DataProto transport
are replaced. No training, live inference server or accelerator is used.
"""

import ast
import asyncio
import importlib
import sys
from collections import defaultdict
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any, Optional
from unittest.mock import AsyncMock, Mock

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]


def test_actual_v0_validation_dispatch_prepares_positions_before_generation(api):
    """Exercise the real trainer call and stop before any inference is requested."""
    native = pytest.importorskip("verl")
    namespace = {
        "DataProto": native.DataProto,
        "np": np,
        "defaultdict": defaultdict,
        "invoke_experiment": api.hooks.invoke_experiment,
    }
    trainer_type = production_class(
        ROOT / "verl_omni/trainer/diffusion/ray_diffusion_trainer.py",
        "BaseRayDiffusionTrainer",
        ("_validate",),
        namespace,
    )
    trainer = trainer_type()
    trainer.config = SimpleNamespace(
        actor_rollout_ref=SimpleNamespace(rollout=SimpleNamespace(val_kwargs=SimpleNamespace(n=1)))
    )
    trainer.val_dataloader = [
        {
            "prompts": torch.zeros(128, 1, dtype=torch.long),
            "uid": np.array([f"sample-{index}" for index in range(128)], dtype=object),
            "reward_model": np.array([{"ground_truth": ""}] * 128, dtype=object),
        }
    ]

    def stop_before_generation(batch):
        """Authenticate the exact call boundary without constructing rollout workers."""
        assert batch.non_tensor_batch["_rollout_seed_global_idx"].tolist() == list(range(128))
        raise RuntimeError("test stop before generation")

    trainer._get_gen_batch = stop_before_generation
    with pytest.raises(RuntimeError, match="test stop before generation"):
        trainer._validate()


class CpuTensorDict(dict):
    """Store actual CPU tensors while ignoring distributed container setup."""

    def __init__(self, values, batch_size):
        """Keep the worker's tensor mapping and declared batch size."""
        super().__init__(values)
        self.batch_size = batch_size


class CpuDataProto:
    """Expose only the transport fields read by the real production methods."""

    def __init__(self, batch=None, non_tensor_batch=None, meta_info=None):
        """Own the test payload without changing its row order or metadata."""
        self.batch = batch
        self.non_tensor_batch = non_tensor_batch
        self.meta_info = meta_info

    def __len__(self):
        """Use the preserved input index column as the test batch length."""
        return len(self.non_tensor_batch["index"])


def production_class(path, class_name, methods, namespace, bases=()):
    """Compile real class methods with package imports and Ray decorators omitted."""
    tree = ast.parse(path.read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == class_name)
    cls.body = [
        node for node in cls.body if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef) and node.name in methods
    ]
    assert {node.name for node in cls.body} == set(methods)
    cls.decorator_list = []
    cls.bases = [ast.Name(id=name, ctx=ast.Load()) for name in bases]
    for method in cls.body:
        method.decorator_list = []
    exec(compile(ast.fix_missing_locations(ast.Module(body=[cls], type_ignores=[])), str(path), "exec"), namespace)
    return namespace[class_name]


@pytest.fixture
def api(monkeypatch):
    """Load the selected provider and actual evidence consumer in standalone packages."""
    before = set(sys.modules)
    for name in ("verl_omni", "verl_omni.agent_loop", "verl_omni.experiments", "verl_omni.pipelines"):
        package = ModuleType(name)
        package.__path__ = [str(ROOT.joinpath(*name.split(".")))]
        monkeypatch.setitem(sys.modules, name, package)
    sampling = importlib.import_module("verl_omni.experiments.h3_v0_sampling")
    evidence = importlib.import_module("verl_omni.experiments.h3_evidence")
    hooks = importlib.import_module("verl_omni.pipelines.experiment_hooks")
    seeds = importlib.import_module("verl_omni.agent_loop.utils")
    provider = ModuleType("v0_sampling_test_provider")
    provider.prepare_v0_validation_batch = sampling.prepare_v0_validation_batch
    provider.record_v0_sampling_request = Mock(wraps=sampling.record_v0_sampling_request)
    provider.record_sampling_request = Mock(wraps=evidence.record_sampling_request)
    monkeypatch.setitem(sys.modules, provider.__name__, provider)
    monkeypatch.setenv("VERL_EXPERIMENT_MODULE", provider.__name__)
    yield SimpleNamespace(sampling=sampling, evidence=evidence, hooks=hooks, seeds=seeds, provider=provider)
    for name in set(sys.modules) - before:
        if name.startswith("verl_omni."):
            sys.modules.pop(name, None)


@pytest.fixture
def harness(api):
    """Run real base generation/postprocess around deterministic asynchronous CPU results."""
    namespace = {
        "asyncio": asyncio,
        "np": np,
        "torch": torch,
        "Any": Any,
        "Optional": Optional,
        "DataProto": CpuDataProto,
        "TensorDict": CpuTensorDict,
        "_InternalDiffusionAgentLoopOutput": SimpleNamespace,
        "_config_to_sampling_dict": lambda config: dict(config) if config is not None else {},
        "maybe_per_rollout_seeds": api.seeds.maybe_per_rollout_seeds,
        "invoke_experiment": api.hooks.invoke_experiment,
    }
    worker_type = production_class(
        ROOT / "verl_omni/agent_loop/diffusion_agent_loop.py",
        "DiffusionAgentLoopWorker",
        ("generate_sequences", "_postprocess", "_run_agent_loop"),
        namespace,
    )
    worker = worker_type()
    worker.rollout_config = SimpleNamespace(
        n=8,
        pipeline={"seed": 777},
        algo={},
        calculate_log_probs=True,
        val_kwargs=SimpleNamespace(seed=42, n=1, pipeline={}, algo={}),
        agent=SimpleNamespace(default_agent_loop="stub"),
    )
    requested, completed = [], []

    async def infer(sampling_params, *, validate, **kwargs):
        """Finish alternating rows out of order while preserving their actual request parameters."""
        requested.append((sampling_params.copy(), kwargs.copy(), validate))
        await asyncio.sleep(0)
        if int(kwargs["index"]) % 2 == 0:
            await asyncio.sleep(0)
        completed.append(int(kwargs["index"]))
        return SimpleNamespace(
            prompt_ids=torch.tensor([[int(kwargs["index"])]]),
            response_diffusion_output=torch.tensor([[int(kwargs["index"])]]),
            response_logprobs=None,
            reward_score=0.5,
            num_turns=1,
            metrics=SimpleNamespace(model_dump=lambda: {"finished": True}),
            extra_fields={"other": "retained"},
        )

    def record_after_gather(*args):
        """Require every scheduled request to complete before recording any returned metadata."""
        assert len(requested) == len(completed)
        return api.sampling.record_v0_sampling_request(*args)

    api.provider.record_v0_sampling_request.side_effect = record_after_gather
    worker._run_agent_loop = infer
    return SimpleNamespace(
        worker=worker, worker_type=worker_type, namespace=namespace, requested=requested, completed=completed
    )


def generation_batch(indices, *, validate=False, rollout_seed=42, include_global=True):
    """Give source/transport indices unrelated to positions, as in real V0 validation."""
    fields = {
        "index": np.array([9000 + int(index) for index in indices]),
        "raw_prompt": np.array([f"prompt-{int(index)}" for index in indices], dtype=object),
        "uid": np.array([f"uid-{int(index)}" for index in indices], dtype=object),
    }
    if include_global:
        fields["_rollout_seed_global_idx"] = np.array(indices, dtype=np.int64)
    meta = {"validate": validate, "global_steps": 1}
    if rollout_seed is not None:
        meta["rollout_seed"] = rollout_seed
    return CpuDataProto(non_tensor_batch=fields, meta_info=meta)


def test_training_grid_and_actual_seeds_survive_chunks_and_completion_order(api, harness):
    """All 32 x 8 identities survive splits through prompt groups without reordering native outputs."""

    async def generate_chunks():
        """Use uneven worker chunks so chunk-local positions cannot reproduce the global grid."""
        return [
            await harness.worker.generate_sequences(generation_batch(chunk))
            for chunk in np.array_split(np.arange(256), [37, 101, 193])
        ]

    outputs = asyncio.run(generate_chunks())
    metadata = np.concatenate([output.non_tensor_batch["experiment_sampling"] for output in outputs])
    rows = api.evidence.observed_sampling({"experiment_sampling": metadata}, 256)
    assert rows == [
        {"seed": 42_000_126 + index, "prompt_index": index // 8, "session_id": index % 8} for index in range(256)
    ]
    assert torch.cat([output.batch["responses"] for output in outputs]).flatten().tolist() == list(range(9000, 9256))
    assert harness.completed != list(range(9000, 9256))
    assert [params["seed"] for params, _, _ in harness.requested] == [row["seed"] for row in rows]
    assert all(params["global_steps"] == 1 and params["logprobs"] is True for params, _, _ in harness.requested)
    assert api.provider.record_v0_sampling_request.call_count == 256
    assert all(output.non_tensor_batch["other"].tolist() == ["retained"] * len(output) for output in outputs)


def test_validation_preparation_uses_batch_positions_without_changing_seed(api, harness):
    """Fixed128/n1 uses pre-chunk positions, never source indices or the training n8 layout."""
    batch = generation_batch(np.arange(128), validate=True, include_global=False)
    original_indices = batch.non_tensor_batch["index"].copy()
    original_meta = batch.meta_info.copy()
    api.hooks.invoke_experiment("prepare_v0_validation_batch", batch)
    assert batch.non_tensor_batch["_rollout_seed_global_idx"].dtype == np.int64
    assert batch.meta_info == original_meta
    np.testing.assert_array_equal(batch.non_tensor_batch["index"], original_indices)

    async def generate_chunks():
        """Preserve injected global positions when each worker receives only a local slice."""
        outputs = []
        for positions in np.array_split(np.arange(128), [13, 49, 89]):
            chunk = CpuDataProto(
                non_tensor_batch={key: values[positions] for key, values in batch.non_tensor_batch.items()},
                meta_info=batch.meta_info.copy(),
            )
            outputs.append(await harness.worker.generate_sequences(chunk))
        return outputs

    outputs = asyncio.run(generate_chunks())
    rows = np.concatenate([output.non_tensor_batch["experiment_sampling"] for output in outputs]).tolist()
    assert rows == [{"seed": 42, "prompt_index": index, "session_id": 0} for index in range(128)]
    assert all(params["seed"] == 42 and params["logprobs"] is False for params, _, _ in harness.requested)


def test_default_off_preserves_native_requests_and_adds_no_metadata(api, harness, monkeypatch):
    """Ordinary runs need neither global identity nor a selected provider or seed owner."""
    monkeypatch.delenv("VERL_EXPERIMENT_MODULE")
    importer = Mock(side_effect=AssertionError("imported optional provider while disabled"))
    monkeypatch.setattr(api.hooks, "importlib", SimpleNamespace(import_module=importer))
    batch = generation_batch([3, 1], rollout_seed=None, include_global=False)
    api.hooks.invoke_experiment("prepare_v0_validation_batch", batch)
    output = asyncio.run(harness.worker.generate_sequences(batch))
    assert "_rollout_seed_global_idx" not in batch.non_tensor_batch
    assert "experiment_sampling" not in output.non_tensor_batch
    assert [params["seed"] for params, _, _ in harness.requested] == [777, 777]
    assert output.batch["responses"].flatten().tolist() == [9003, 9001]
    importer.assert_not_called()


def test_callback_records_supplied_seed_verbatim_without_mutating_request(api):
    """A deliberately noncanonical actual seed is evidence to validate later, never replaced here."""
    result = SimpleNamespace(extra_fields={"other": "retained"})
    params = {"seed": np.int64(12345)}
    fields = {"_rollout_seed_global_idx": np.int64(19), "index": 98765}
    api.sampling.record_v0_sampling_request(result, params, fields, False, 8)
    assert result.extra_fields == {
        "other": "retained",
        "experiment_sampling": {"seed": 12345, "prompt_index": 2, "session_id": 3},
    }
    assert params == {"seed": 12345} and fields == {"_rollout_seed_global_idx": 19, "index": 98765}


@pytest.mark.parametrize(
    "params,fields,n",
    [
        ({}, {"_rollout_seed_global_idx": 1}, 8),
        ({"seed": None}, {"_rollout_seed_global_idx": 1}, 8),
        ({"seed": 42}, {"index": 1}, 8),
        ({"seed": 42}, {"_rollout_seed_global_idx": True}, 8),
        ({"seed": 42}, {"_rollout_seed_global_idx": -1}, 8),
        ({"seed": 42}, {"_rollout_seed_global_idx": 1.5}, 8),
        ({"seed": 42}, {"_rollout_seed_global_idx": 1}, 4),
    ],
)
def test_invalid_observation_fails_without_fabricating_metadata(api, params, fields, n):
    """Missing actual seeds/global positions cannot be repaired using source or local indices."""
    result = SimpleNamespace(extra_fields={"other": "retained"})
    with pytest.raises(api.evidence.H3EvidenceContractError):
        api.sampling.record_v0_sampling_request(result, params, fields, False, n)
    assert result.extra_fields == {"other": "retained"}


def test_request_failure_never_reaches_callback_or_batch_postprocess(api, harness):
    """A failed gather cannot produce a fabricated successful request envelope."""
    harness.worker._run_agent_loop = AsyncMock(side_effect=RuntimeError("inference failed"))
    harness.worker._postprocess = Mock()
    with pytest.raises(RuntimeError, match="inference failed"):
        asyncio.run(harness.worker.generate_sequences(generation_batch([0])))
    api.provider.record_v0_sampling_request.assert_not_called()
    harness.worker._postprocess.assert_not_called()


def test_tq_super_call_is_untouched_by_v0_sampling(api, harness):
    """The main V1 TQ path does not acquire V0's generation callback."""
    result = SimpleNamespace(extra_fields={})
    namespace = harness.namespace.copy()
    namespace.update(
        _agent_loop_registry={"stub": object()},
        DictConfigWrap=lambda config: config,
        hydra=SimpleNamespace(
            utils=SimpleNamespace(instantiate=Mock(return_value=SimpleNamespace(run=AsyncMock(return_value=result))))
        ),
    )
    worker_type = production_class(
        ROOT / "verl_omni/agent_loop/diffusion_agent_loop.py",
        "DiffusionAgentLoopWorker",
        ("_run_agent_loop",),
        namespace,
    )
    namespace["DiffusionAgentLoopWorker"] = worker_type
    tq_type = production_class(
        ROOT / "verl_omni/agent_loop/diffusion_agent_loop_tq.py",
        "DiffusionAgentLoopWorkerTQ",
        ("_run_agent_loop",),
        namespace,
        bases=("DiffusionAgentLoopWorker",),
    )
    worker = tq_type()
    worker.config = SimpleNamespace(data={})
    worker.model_config = SimpleNamespace(extra_tokenizer_map={})
    worker.server_manager = worker.tokenizer = worker.processor = worker.dataset_cls = worker.hf_model_type = None
    worker._agent_loop_postprocess = AsyncMock(return_value=result)
    returned = asyncio.run(
        worker._run_agent_loop(
            {"seed": 123},
            agent_name="stub",
            session_id=3,
            trajectory={"validate": False},
            index=2,
        )
    )
    assert returned.extra_fields == {}
    api.provider.record_sampling_request.assert_not_called()
    api.provider.record_v0_sampling_request.assert_not_called()
