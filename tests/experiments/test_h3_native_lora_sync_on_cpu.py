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
"""Exercise native Diffusers/PEFT loading, admission and FlowGRPO sync on CPU."""

import importlib
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch
from diffusers import ConfigMixin, ModelMixin
from diffusers.loaders.peft import PeftAdapterMixin
from peft import get_peft_model_state_dict
from peft.tuners.tuners_utils import BaseTunerLayer
from safetensors.torch import save_file
from torch import nn


@pytest.fixture
def h3_modules(monkeypatch):
    """Isolate package registration while importing the real callbacks and mapper."""
    before = set(sys.modules)
    root = Path(__file__).parents[2]
    for name in (
        "verl_omni",
        "verl_omni.experiments",
        "verl_omni.pipelines",
        "verl_omni.pipelines.minimax_h3_diffusion_nft",
        "verl_omni.pipelines.minimax_h3_flow_grpo",
    ):
        package = ModuleType(name)
        package.__path__ = [str(root.joinpath(*name.split(".")))]
        monkeypatch.setitem(sys.modules, name, package)
    yield (
        importlib.import_module("verl_omni.experiments.prepare_h3_adapter"),
        importlib.import_module("verl_omni.pipelines.minimax_h3_flow_grpo.weight_sync"),
    )
    for name in set(sys.modules) - before:
        if name.startswith("verl_omni."):
            sys.modules.pop(name)


class Block(nn.Module):
    """Represent six H3 target projections with tiny GEGLU dimensions."""

    def __init__(self):
        """Allocate CPU projections matching the native H3 module paths."""
        super().__init__()
        self.attn = nn.ModuleDict({name: nn.Linear(4, 4) for name in ("to_q", "to_k", "to_v")})
        self.attn["to_out"] = nn.Sequential(nn.Linear(4, 4), nn.Dropout())
        geglu = nn.Module()
        geglu.proj = nn.Linear(4, 6)
        self.ff = nn.Module()
        self.ff.net = nn.Sequential(geglu, nn.Dropout(), nn.Linear(3, 4))


class TinyH3(ModelMixin, ConfigMixin, PeftAdapterMixin):
    """Keep native Diffusers loading and real PEFT injection with no H3 checkpoint."""

    def __init__(self):
        """Build 50 transformer and two refiner blocks plus unrelated projections."""
        super().__init__()
        self.transformer_blocks = nn.ModuleList([Block() for index in range(50)])
        self.token_refiner = nn.Module()
        self.token_refiner.refiner_blocks = nn.ModuleList([Block() for index in range(2)])
        self.other = nn.Sequential(nn.Linear(4, 4), nn.SiLU(), nn.Linear(4, 4))
        self.to(device="cpu", dtype=torch.bfloat16)


def create_package(model, directory):
    """Save all 624 BF16 A/B tensors at rank64/alpha128 with nonzero B."""
    directory.mkdir()
    tensors = {}
    blocks = [f"transformer_blocks.{index}" for index in range(50)]
    blocks += [f"token_refiner.refiner_blocks.{index}" for index in range(2)]
    for block in blocks:
        for suffix in ("attn.to_q", "attn.to_k", "attn.to_v", "attn.to_out.0", "ff.net.0.proj", "ff.net.2"):
            name = block + "." + suffix
            layer = model.get_submodule(name)
            tensors[name + ".lora_A.weight"] = (
                torch.arange(64 * layer.in_features)
                .reshape(64, layer.in_features)
                .remainder(13)
                .div(128)
                .to(torch.bfloat16)
            )
            tensors[name + ".lora_B.weight"] = (
                torch.arange(layer.out_features * 64)
                .reshape(layer.out_features, 64)
                .remainder(17)
                .add(1)
                .div(128)
                .to(torch.bfloat16)
            )
    save_file(tensors, directory / "adapter_model.safetensors")
    config = {
        "r": 64,
        "lora_alpha": 128,
        "bias": "none",
        "peft_type": "LORA",
        "target_modules": ["to_q", "to_k", "to_v", "to_out.0", "proj", "net.2"],
    }
    (directory / "adapter_config.json").write_text(json.dumps(config))
    return tensors


def verify_mapped_values(original, mapped):
    """Check all 728 engine names and values, including duplicated A and swapped B halves."""
    expected = {}
    for name, tensor in original.items():
        translated = (
            name.replace("token_refiner.refiner_blocks.", "token_refiner.blocks.")
            .replace("transformer_blocks.", "blocks.")
            .replace(".attn.to_out.0.", ".attn.out_proj.")
            .replace(".ff.net.2.", ".mlp.fc2.")
        )
        if ".ff.net.0.proj." in translated:
            parts = (tensor, tensor) if ".lora_A." in translated else (tensor[3:], tensor[:3])
            for index, part in enumerate(parts):
                expected["transformer." + translated.replace(".ff.net.0.proj.", f".mlp.fc1_{index}.")] = part
        else:
            expected["transformer." + translated] = tensor
    assert len(mapped) == len(expected) == 728
    assert mapped.keys() == expected.keys()
    for name, value in expected.items():
        torch.testing.assert_close(mapped[name], value, rtol=0, atol=0)


def verify_native_sync(api, mapping, model, directory, original, attempt, monkeypatch):
    """Load real PEFT, invoke admission, and verify metadata repair without changing weights or RNG."""
    config = SimpleNamespace(lora_adapter_path=str(directory))
    model.load_lora_adapter(
        str(directory), adapter_name="default", local_files_only=True, **api.lora_loader_kwargs(config)
    )
    model.set_adapter("default")
    # Match the engine's post-load trainable dtype and PEFT input-cast policies.
    for parameter in model.parameters():
        if parameter.requires_grad:
            parameter.data = parameter.data.to(torch.bfloat16)
    layers = {name: layer for name, layer in model.named_modules() if isinstance(layer, BaseTunerLayer)}
    expected_layers = {name.rsplit(".lora_", 1)[0] for name in original}
    assert len(layers) == len(expected_layers) == 312 and layers.keys() == expected_layers
    for layer in layers.values():
        layer.cast_input_dtype_enabled = False
        assert (layer.r["default"], layer.lora_alpha["default"], layer.scaling["default"]) == (64, 128, 2)
    loaded = get_peft_model_state_dict(model, adapter_name="default")
    assert len(loaded) == len(original) == 624 and loaded.keys() == original.keys()
    for name, value in original.items():
        assert loaded[name].device.type == "cpu" and loaded[name].dtype == torch.bfloat16
        torch.testing.assert_close(loaded[name], value, rtol=0, atol=0)
        if ".lora_B." in name:
            assert torch.count_nonzero(loaded[name]) == loaded[name].numel()
    mapper = object.__new__(mapping.MiniMaxH3WeightSyncMixin)
    mapper.transformer = SimpleNamespace(arch=SimpleNamespace(ffn_hidden_size=3))
    with pytest.raises(ValueError, match="unsupported targets") as rejection:
        mapper.map_lora_update_to_engine(loaded, model.peft_config["default"].to_dict())
    assert "'proj'" in str(rejection.value) and "'net.2'" in str(rejection.value)
    layer = model.transformer_blocks[0].attn["to_q"]
    inputs = torch.ones(2, 4, dtype=torch.bfloat16)
    output = layer(inputs).detach().clone()
    assert not torch.equal(output, layer.base_layer(inputs))
    rng = torch.get_rng_state().clone()
    monkeypatch.setenv("H3_ATTEMPT_DIR", str(attempt))
    report = api.adapter_loaded(model, config)
    assert torch.equal(rng, torch.get_rng_state())
    torch.testing.assert_close(layer(inputs), output, rtol=0, atol=0)
    assert report["status"] == "VERIFIED" and report["hash_verified_tensor_count"] == 624
    assert set(report["layers"]) == model.peft_config["default"].target_modules == expected_layers
    state = get_peft_model_state_dict(model, adapter_name="default")
    assert state.keys() == original.keys()
    for name, value in original.items():
        torch.testing.assert_close(state[name], value, rtol=0, atol=0)
    mapped, engine_config = mapper.map_lora_update_to_engine(state, model.peft_config["default"].to_dict())
    verify_mapped_values(original, mapped)
    assert set(engine_config["target_modules"]) == {"to_q", "to_k", "to_v", "out_proj", "fc1_0", "fc1_1", "fc2"}
    assert (engine_config["r"], engine_config["lora_alpha"]) == (64, 128)
    return state


def test_native_lora_admission_restores_sync_targets_and_export_reload(h3_modules, tmp_path, monkeypatch):
    """Regress native target condensation through the actual callback, mapping and PEFT export/reload."""
    api, mapping = h3_modules
    model = TinyH3()
    source, converted = tmp_path / "canonical", tmp_path / "converted"
    original = create_package(model, source)
    api.prepare_h3_adapter(source, converted)
    state = verify_native_sync(api, mapping, model, converted, original, tmp_path / "admission", monkeypatch)
    exported = tmp_path / "exported"
    exported.mkdir()
    save_file(state, exported / "adapter_model.safetensors")
    (exported / "adapter_config.json").write_text(json.dumps(model.peft_config["default"].to_dict(), default=list))
    reconverted = tmp_path / "reconverted"
    api.prepare_h3_adapter(exported, reconverted)
    verify_native_sync(api, mapping, TinyH3(), reconverted, original, tmp_path / "reload-admission", monkeypatch)
