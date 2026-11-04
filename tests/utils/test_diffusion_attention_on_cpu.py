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

import logging
import sys
from types import ModuleType
from unittest.mock import MagicMock

import pytest
import torch
from omegaconf import OmegaConf

from tests.utils.smoke_attention import resolve_smoke_attention_backends
from verl_omni.utils import diffusion_attention
from verl_omni.utils.diffusion_attention import validate_attention_consistency
from verl_omni.workers.config.diffusion import DiffusionModelConfig, DiffusionRolloutConfig


def test_rollout_config_builds_structured_vllm_omni_attention_config():
    config = DiffusionRolloutConfig(name="vllm_omni", rollout_attn_backend="FLASH_ATTN")

    assert config.to_vllm_omni_attention_config() == {"default": {"backend": "FLASH_ATTN"}}


def test_veomni_strategy_skips_with_warning(caplog):
    with caplog.at_level(logging.WARNING, logger="verl_omni.utils.diffusion_attention"):
        validate_attention_consistency(
            OmegaConf.create(
                {
                    "actor_rollout_ref": {
                        "model": {"attn_backend": "native"},
                        "actor": {"strategy": "veomni"},
                        "rollout": {"rollout_attn_backend": "FLASH_ATTN"},
                    }
                }
            )
        )
    assert "veomni" in caplog.text


def test_fa3_actor_allows_flash_attn_3_hub():
    validate_attention_consistency(
        OmegaConf.create(
            {
                "actor_rollout_ref": {
                    "model": {"attn_backend": "_flash_3_varlen_hub"},
                    "actor": {"strategy": "fsdp2"},
                    "rollout": {"rollout_attn_backend": "FLASH_ATTN_3_HUB"},
                }
            }
        )
    )


def test_fa3_actor_allows_flash_attn():
    validate_attention_consistency(
        OmegaConf.create(
            {
                "actor_rollout_ref": {
                    "model": {"attn_backend": "_flash_3_varlen_hub"},
                    "actor": {"strategy": "fsdp2"},
                    "rollout": {"rollout_attn_backend": "FLASH_ATTN"},
                }
            }
        )
    )


def test_native_actor_rejects_flash_attn_3_hub():
    try:
        validate_attention_consistency(
            OmegaConf.create(
                {
                    "actor_rollout_ref": {
                        "model": {"attn_backend": "native"},
                        "actor": {"strategy": "fsdp2"},
                        "rollout": {"rollout_attn_backend": "FLASH_ATTN_3_HUB"},
                    }
                }
            )
        )
    except ValueError as exc:
        assert "rollout_attn_backend='TORCH_SDPA'" in str(exc)
    else:
        raise AssertionError("expected ValueError")


@pytest.mark.parametrize("attn_backend", ["flash_varlen_hub", "_flash_3_varlen_hub", "flash_4_hub"])
def test_fa_actor_backend_without_kernels_raises(monkeypatch, tmp_path, attn_backend):
    """All hub FA actor backends require the kernels package."""
    monkeypatch.setitem(sys.modules, "kernels", None)
    with pytest.raises(ImportError, match="requires `kernels` package"):
        DiffusionModelConfig(
            path=str(tmp_path),
            architecture="StableDiffusion3Pipeline",
            algorithm="flow_grpo",
            attn_backend=attn_backend,
            load_tokenizer=False,
            transformer_config={},
        )


def test_fa2_actor_allows_flash_attn():
    validate_attention_consistency(
        OmegaConf.create(
            {
                "actor_rollout_ref": {
                    "model": {"attn_backend": "flash_varlen_hub"},
                    "actor": {"strategy": "fsdp2"},
                    "rollout": {"rollout_attn_backend": "FLASH_ATTN"},
                }
            }
        )
    )


def test_resolve_smoke_attention_backends_prefers_local_fa(monkeypatch):
    monkeypatch.setattr("tests.utils.smoke_attention.fa_available", lambda: True)
    assert resolve_smoke_attention_backends() == ("_flash_3_varlen_hub", "FLASH_ATTN")


def test_resolve_smoke_attention_backends_falls_back_without_fa3(monkeypatch):
    monkeypatch.setattr("tests.utils.smoke_attention.fa_available", lambda: False)
    assert resolve_smoke_attention_backends() == ("native", "TORCH_SDPA")


@pytest.mark.parametrize("attn_backend", [None, "flash_varlen_hub", "_flash_3_varlen_hub", "flash_4_hub"])
def test_model_config_accepts_explicit_fa_and_preserves_fa3_default(monkeypatch, tmp_path, attn_backend):
    """Construct real configs with only the optional kernels import replaced."""
    monkeypatch.setitem(sys.modules, "kernels", ModuleType("kernels"))
    overrides = {} if attn_backend is None else {"attn_backend": attn_backend}
    config = DiffusionModelConfig(
        path=str(tmp_path),
        architecture="StableDiffusion3Pipeline",
        algorithm="flow_grpo",
        load_tokenizer=False,
        transformer_config={},
        **overrides,
    )
    assert config.attn_backend == (attn_backend or "_flash_3_varlen_hub")
    assert DiffusionRolloutConfig(name="vllm_omni").rollout_attn_backend == "FLASH_ATTN_3_HUB"


@pytest.mark.parametrize("rollout_backend", ["FLASH_ATTN", "FLASH_ATTN_HUB", "FLASH_ATTN_3_HUB", "TORCH_SDPA"])
def test_explicit_fa4_actor_requires_local_flash_attn(rollout_backend):
    """Accept the FA4 local pair and reject every other known rollout backend."""
    config = OmegaConf.create(
        {
            "actor_rollout_ref": {
                "model": {"attn_backend": "flash_4_hub"},
                "actor": {"strategy": "fsdp2"},
                "rollout": {"rollout_attn_backend": rollout_backend},
            }
        }
    )
    if rollout_backend == "FLASH_ATTN":
        validate_attention_consistency(config)
    else:
        with pytest.raises(ValueError, match="requires rollout_attn_backend='FLASH_ATTN'"):
            validate_attention_consistency(config)


@pytest.mark.parametrize("capability", [(7, 5), (8, 0), (8, 9), (9, 0), (10, 0), (10, 3)])
@pytest.mark.parametrize("has_cute", [False, True])
def test_local_rollout_fa_requires_cute_on_blackwell(monkeypatch, capability, has_cute):
    """Do not mistake importable Hopper packages for a runnable Blackwell kernel."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda: capability)
    monkeypatch.setattr(diffusion_attention.importlib.util, "find_spec", lambda name: object())
    cute = ModuleType("flash_attn.cute")
    cute.flash_attn_func = MagicMock()
    cute.flash_attn_varlen_func = MagicMock()
    monkeypatch.setitem(sys.modules, "flash_attn", ModuleType("flash_attn"))
    monkeypatch.setitem(sys.modules, "flash_attn.cute", cute if has_cute else None)

    expected = has_cute if capability[0] >= 10 else capability[0] >= 8
    assert diffusion_attention.rollout_fa_available() is expected


def test_local_rollout_fa_unavailable_without_cuda(monkeypatch):
    """An installed FA package alone cannot enable rollout on a CPU host."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(diffusion_attention.importlib.util, "find_spec", lambda name: object())
    assert diffusion_attention.rollout_fa_available() is False
