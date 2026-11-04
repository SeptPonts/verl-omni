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
"""CPU tests for meta-rank config-only init in the diffusers FSDP engine."""

from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
from accelerate import init_empty_weights

import verl_omni.workers.engine.fsdp.diffusers_impl as diffusers_impl
from verl_omni.workers.config.diffusion import DiffusionModelConfig
from verl_omni.workers.engine.fsdp.diffusers_impl import PPODiffusersFSDPEngine


class _FakeDiffusersModule(torch.nn.Module):
    _keep_in_fp32_modules = None

    def __init__(self):
        super().__init__()
        self.proj = torch.nn.Linear(2, 2)
        self.config = SimpleNamespace()
        self.attn_backend = None

    def set_attention_backend(self, backend):
        self.attn_backend = backend


def _make_engine(**overrides):
    engine = object.__new__(PPODiffusersFSDPEngine)
    model_config = object.__new__(DiffusionModelConfig)
    object.__setattr__(model_config, "config_path", None)
    object.__setattr__(model_config, "local_path", "/fake/ckpt")
    object.__setattr__(model_config, "trust_remote_code", True)
    object.__setattr__(model_config, "transformer_subfolder", "transformer")
    object.__setattr__(model_config, "attn_backend", "flash_attn")
    object.__setattr__(model_config, "enable_gradient_checkpointing", False)
    for key, value in overrides.items():
        object.__setattr__(model_config, key, value)
    engine.model_config = model_config
    engine.engine_config = MagicMock(model_dtype=None, forward_only=False)
    engine.device_mesh = None
    return engine


def _build_default_path(engine, *, init_context, loaded_module):
    adapter_cls = MagicMock()
    adapter_cls.preserve_fp32_modules.return_value = False

    with (
        patch.object(engine, "_build_module_from_registry", return_value=None),
        patch.object(diffusers_impl, "get_init_weight_context_manager", return_value=init_context),
        patch.object(diffusers_impl.DiffusionModelBase, "get_class", return_value=adapter_cls),
        patch("diffusers.AutoModel.from_config", return_value=loaded_module) as mock_from_config,
        patch("diffusers.AutoModel.from_pretrained", return_value=loaded_module) as mock_from_pretrained,
        patch("verl.utils.torch_dtypes.PrecisionType") as precision_type,
    ):
        precision_type.to_dtype.side_effect = lambda value: value
        result = engine._build_module()
    return result, mock_from_config, mock_from_pretrained


def test_build_module_on_meta_rank_builds_from_config_without_reading_the_checkpoint():
    engine = _make_engine()
    loaded = _FakeDiffusersModule()

    result, mock_from_config, mock_from_pretrained = _build_default_path(
        engine, init_context=init_empty_weights, loaded_module=loaded
    )

    mock_from_pretrained.assert_not_called()
    mock_from_config.assert_called_once_with(
        engine.model_config.local_path,
        trust_remote_code=True,
        subfolder="transformer",
    )
    assert result is loaded
    assert loaded.attn_backend == "flash_attn"
    assert loaded.proj.weight.dtype == torch.float32
    assert callable(loaded.can_generate)
    assert callable(loaded.config.save_pretrained)


def test_build_module_on_rank0_still_loads_from_pretrained():
    engine = _make_engine()
    loaded = _FakeDiffusersModule()

    result, mock_from_config, mock_from_pretrained = _build_default_path(
        engine, init_context=nullcontext, loaded_module=loaded
    )

    mock_from_config.assert_not_called()
    mock_from_pretrained.assert_called_once_with(
        engine.model_config.local_path,
        torch_dtype=torch.float32,
        trust_remote_code=True,
        subfolder="transformer",
    )
    assert result is loaded
    assert loaded.attn_backend == "flash_attn"
    assert loaded.proj.weight.dtype == torch.float32


def test_build_module_on_meta_rank_uses_config_path_when_set():
    engine = _make_engine(config_path="/fake/config", transformer_subfolder="transformer")
    loaded = _FakeDiffusersModule()

    _, mock_from_config, mock_from_pretrained = _build_default_path(
        engine, init_context=init_empty_weights, loaded_module=loaded
    )

    mock_from_pretrained.assert_not_called()
    mock_from_config.assert_called_once_with(
        "/fake/config",
        trust_remote_code=True,
        subfolder="",
    )


@pytest.mark.parametrize("loader", ["add", "native"])
def test_meta_h3_matches_pretrained_dtypes_and_runs_bf16_lora(tmp_path, monkeypatch, loader):
    """Keep the real H3 mixed dtype layout through config-only init and LoRA forward."""
    from diffusers import MiniMaxH3Transformer3DModel
    from peft import LoraConfig, get_peft_model_state_dict
    from peft.tuners.tuners_utils import BaseTunerLayer

    model = MiniMaxH3Transformer3DModel(
        num_attention_heads=2,
        attention_head_dim=4,
        hidden_size=8,
        num_layers=1,
        num_refiner_layers=1,
        ffn_dim=16,
        in_channels=2,
        audio_in_channels=2,
        patch_size=(1, 1, 1),
        text_dim=8,
        freq_dim=4,
        time_embed_hidden_dim=8,
        time_embed_dim=4,
        rope_freq_dim=2,
    )
    model.save_pretrained(tmp_path)
    pretrained = MiniMaxH3Transformer3DModel.from_pretrained(tmp_path, torch_dtype=torch.bfloat16)
    expected_dtypes = {name: parameter.dtype for name, parameter in pretrained.named_parameters()}
    engine = _make_engine(config_path=str(tmp_path), attn_backend="native", trust_remote_code=False)
    engine.engine_config.model_dtype = torch.bfloat16
    adapter_class = MagicMock()
    adapter_class.preserve_fp32_modules.return_value = True
    monkeypatch.setattr(engine, "_build_module_from_registry", lambda dtype: None)
    monkeypatch.setattr(diffusers_impl, "get_init_weight_context_manager", lambda **kwargs: init_empty_weights)
    monkeypatch.setattr(diffusers_impl.DiffusionModelBase, "get_class", lambda config: adapter_class)
    # Real config construction must not read the checkpoint payload on meta ranks.
    with patch("diffusers.AutoModel.from_pretrained", side_effect=AssertionError("meta rank read weights")):
        meta_model = engine._build_module()
    assert all(parameter.is_meta for parameter in meta_model.parameters())
    assert {name: parameter.dtype for name, parameter in meta_model.named_parameters()} == expected_dtypes
    assert meta_model.proj_in.weight.dtype == torch.float32
    assert meta_model.context_embedder.weight.dtype == torch.bfloat16
    meta_model.to_empty(device="cpu")
    meta_model.load_state_dict(pretrained.state_dict())
    targets = ["to_q", "to_k", "to_v", "to_out.0", "ff.net.0.proj", "ff.net.2"]
    lora_config = LoraConfig(r=64, lora_alpha=128, target_modules=targets)
    if loader == "add":
        meta_model.add_adapter(lora_config, adapter_name="default")
    else:
        pretrained.add_adapter(lora_config, adapter_name="default")
        state = get_peft_model_state_dict(pretrained)
        meta_model.load_lora_adapter(
            state,
            adapter_name="default",
            prefix=None,
            metadata={"r": 64, "lora_alpha": 128, "target_modules": targets},
        )
    for parameter in meta_model.parameters():
        if parameter.requires_grad:
            parameter.data = parameter.data.to(torch.bfloat16)
    for layer in meta_model.modules():
        if isinstance(layer, BaseTunerLayer):
            layer.cast_input_dtype_enabled = False
    inputs = torch.ones(1, 3, 8, dtype=meta_model.context_embedder.weight.dtype)
    output = meta_model.token_refiner(meta_model.context_embedder(inputs))
    assert output.dtype == torch.bfloat16 and torch.isfinite(output).all()
