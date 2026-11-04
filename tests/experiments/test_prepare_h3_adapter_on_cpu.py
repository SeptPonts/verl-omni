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
"""Convert 624 tensors and exercise actual Diffusers loading functions on tiny CPU layers.

The source-loader test executes unmodified functions from H3_DIFFUSERS_SOURCE.
PEFT injection is a small nn.Module test double, not a claim of PEFT/device
acceptance. Fetching, metadata parsing, prefix filtering, config construction,
and the main engine call are the real owning source functions.
"""

import ast
import builtins
import importlib
import importlib.util
import json
import logging
import os
import runpy
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import safetensors
import safetensors.torch
import torch
from huggingface_hub.utils import validate_hf_hub_args
from safetensors.torch import load_file, save_file

ROOT = Path(__file__).parents[2]


@pytest.fixture
def api(monkeypatch):
    """Import CPU conversion utilities without accelerator registration."""
    before = set(sys.modules)
    for name in ("verl_omni", "verl_omni.experiments"):
        package = ModuleType(name)
        package.__path__ = [str(ROOT.joinpath(*name.split(".")))]
        monkeypatch.setitem(sys.modules, name, package)
    yield importlib.import_module("verl_omni.experiments.prepare_h3_adapter")
    for name in set(sys.modules) - before:
        if name.startswith("verl_omni."):
            sys.modules.pop(name, None)


@pytest.fixture(params=["canonical", "native_export"])
def source(api, tmp_path, request):
    """Create the H3 52-block/six-projection layout with tiny widths and the real rank64."""
    root = tmp_path / "source"
    root.mkdir()
    paths = [f"transformer_blocks.{index}" for index in range(50)]
    paths += [f"token_refiner.refiner_blocks.{index}" for index in range(2)]
    state = {}
    for block in paths:
        for suffix in ("attn.to_q", "attn.to_k", "attn.to_v", "attn.to_out.0", "ff.net.0.proj", "ff.net.2"):
            prefix = "base_model.model." if request.param == "native_export" else ""
            module = prefix + block + "." + suffix
            state[module + ".lora_A.weight"] = torch.arange(128, dtype=torch.bfloat16).reshape(64, 2) / 128
            state[module + ".lora_B.weight"] = torch.full(
                (3, 64), 0 if request.param == "canonical" else 0.25, dtype=torch.bfloat16
            )
    config = {
        "r": 64,
        "lora_alpha": 128,
        "bias": "none",
        "peft_type": "LORA",
        "target_modules": ["to_q", "to_k", "to_v", "to_out.0", "proj", "net.2"],
    }
    (root / "adapter_config.json").write_text(json.dumps(config))
    save_file(state, root / "adapter_model.safetensors")
    return root


def test_conversion_preserves_all_values_and_source_bytes(api, source, tmp_path):
    """Support both file pairs without requiring old transformer_config or tensor_manifest files."""
    before = {path.name: api.file_sha256(path) for path in source.iterdir()}
    target = tmp_path / "converted"
    receipt = api.prepare_h3_adapter(source, target)
    assert before == {path.name: api.file_sha256(path) for path in source.iterdir()}
    assert receipt["tensor_count"] == 624 and receipt["module_count"] == 312
    assert (receipt["r"], receipt["lora_alpha"], receipt["scaling"]) == (64, 128, 2)
    original = api.tensor_manifest(load_file(source / "adapter_model.safetensors"))
    actual = api.tensor_manifest(load_file(target / "pytorch_lora_weights.safetensors"))
    assert len(actual) == 624
    for name, record in receipt["tensors"].items():
        assert actual[record["main_key"]] == original[name]
        assert {key: record[key] for key in ("shape", "dtype", "sha256")} == original[name]
    with safetensors.safe_open(target / "pytorch_lora_weights.safetensors", framework="pt") as stream:
        metadata = json.loads(stream.metadata()["lora_adapter_metadata"])
    assert metadata["transformer.r"] == 64 and metadata["transformer.lora_alpha"] == 128
    assert len(metadata["transformer.target_modules"]) == 312
    assert json.loads((target / "conversion_receipt.json").read_text()) == receipt


@pytest.mark.parametrize("placement", ["inplace", "inside", "existing", "symlink"])
def test_forbid_source_mutation_and_overwrite(api, source, tmp_path, placement):
    """Reject aliases of source and every existing target, including empty directories."""
    if placement == "inplace":
        target = source
    elif placement == "inside":
        target = source / "new"
    elif placement == "symlink":
        target = tmp_path / "alias"
        target.symlink_to(source, target_is_directory=True)
    else:
        target = tmp_path / "existing"
        target.mkdir()
    before = {path.name: api.file_sha256(path) for path in source.iterdir()}
    with pytest.raises((ValueError, FileExistsError)):
        api.prepare_h3_adapter(source, target)
    assert before == {path.name: api.file_sha256(path) for path in source.iterdir()}


@pytest.mark.parametrize("problem", ["count", "hash", "rank", "alpha", "rslora", "target", "shape", "key", "pair"])
def test_invalid_contract_fails_without_target(api, source, tmp_path, problem):
    """Refuse missing/unknown tensors and config changes instead of dropping state or changing scaling."""
    state = load_file(source / "adapter_model.safetensors")
    config = json.loads((source / "adapter_config.json").read_text())
    original_manifest = api.tensor_manifest(state)
    name = next(iter(state))
    if problem == "count":
        state.pop(name)
    elif problem == "hash":
        state[name] = state[name] + 1
    elif problem == "rank":
        config["r"] = 32
    elif problem == "alpha":
        config["lora_alpha"] = 64
    elif problem == "rslora":
        config["use_rslora"] = True
    elif problem == "target":
        config["target_modules"] = ["missing"]
    elif problem == "shape":
        state[name] = state[name][:1, :1]
    elif problem == "key":
        state["unrelated.weight"] = state.pop(name)
    else:
        state[name.replace("to_k", "missing")] = state.pop(name)
        config["target_modules"].append("missing")
    save_file(state, source / "adapter_model.safetensors")
    (source / "adapter_config.json").write_text(json.dumps(config))
    manifest = tmp_path / "expected.json"
    manifest.write_text(json.dumps(original_manifest))
    target = tmp_path / "rejected"
    with pytest.raises(ValueError):
        api.prepare_h3_adapter(source, target, manifest=manifest if problem == "hash" else None)
    assert not target.exists()


def load_source_function(path, name, namespace, owner=None):
    """Compile a complete owning-source function unchanged, including any decorators."""
    body = ast.parse(path.read_text()).body
    if owner:
        body = next(node for node in body if isinstance(node, ast.ClassDef) and node.name == owner).body
    node = next(node for node in body if isinstance(node, ast.FunctionDef) and node.name == name)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace)
    return namespace[name]


class CpuLoraConfig:
    """PEFT config boundary double; real Diffusers chooses all keyword values."""

    def __init__(self, **values):
        """Retain the real loader's configuration for injection assertions."""
        self.__dict__.update(values)


class CpuLoraLayer(torch.nn.Module):
    """Small trainable A/B layer for checking the actual state dict delivered by Diffusers."""

    def __init__(self, base, config, adapter_name):
        """Create real CPU parameters using the requested rank and base dtype."""
        super().__init__()
        self.base_layer = base
        self.lora_A = torch.nn.ModuleDict(
            {adapter_name: torch.nn.Linear(2, config.r, bias=False, dtype=base.weight.dtype)}
        )
        self.lora_B = torch.nn.ModuleDict(
            {adapter_name: torch.nn.Linear(config.r, 3, bias=False, dtype=base.weight.dtype)}
        )
        self.r = {adapter_name: config.r}
        self.lora_alpha = {adapter_name: config.lora_alpha}
        self.scaling = {adapter_name: config.lora_alpha / config.r}

    def forward(self, inputs):
        """Apply the loaded update so nonzero B also tests actual scale2 arithmetic."""
        return (
            self.base_layer(inputs) + self.lora_B["default"](self.lora_A["default"](inputs)) * self.scaling["default"]
        )


def inject_cpu_adapter(config, model, adapter_name, state_dict, **kwargs):
    """Replace exact target layers; do not implement namespace filtering or config inference here."""
    model.requires_grad_(False)
    for name in config.target_modules:
        parent, leaf = name.rsplit(".", 1)
        module = model.get_submodule(name)
        setattr(model.get_submodule(parent), leaf, CpuLoraLayer(module, config, adapter_name))
    model.peft_config = {adapter_name: config}


def set_cpu_adapter_state(model, state_dict, adapter_name, **kwargs):
    """Use real torch loading to prove every key received from the real loader is consumed."""
    mapped = {
        name.replace(".lora_A.weight", f".lora_A.{adapter_name}.weight").replace(
            ".lora_B.weight", f".lora_B.{adapter_name}.weight"
        ): tensor
        for name, tensor in state_dict.items()
    }
    result = model.load_state_dict(mapped, strict=False)
    assert not result.unexpected_keys
    assert not any("lora_" in name for name in result.missing_keys)
    model.consumed_keys = set(state_dict)
    return result


def cpu_adapter_state(model, *, adapter_name):
    """Expose the tiny module's actual A/B tensors at the PEFT state-query boundary."""
    return {
        name.replace(f".{adapter_name}.weight", ".weight"): tensor
        for name, tensor in model.state_dict().items()
        if ".lora_" in name and f".{adapter_name}.weight" in name
    }


def no_offloading(pipeline):
    """Represent a tiny CPU model with no offload hooks."""
    assert pipeline is None
    return False, False, False


def no_tensor_parallel(model, *args):
    """Represent the test's unsharded CPU module boundary."""
    assert all(parameter.device.type == "cpu" for parameter in model.parameters())


def local_source_path(path, *, use_shm):
    """Keep the engine test at its already-local, non-shared-memory filesystem boundary."""
    assert not use_shm and Path(path).is_dir()
    return path


@pytest.fixture
def real_loader(monkeypatch):
    """Execute real fetch/config/load methods with explicit doubles only at PEFT and device boundaries."""
    source_path = os.environ.get("H3_DIFFUSERS_SOURCE")
    if source_path:
        source = Path(source_path)
    else:
        spec = importlib.util.find_spec("diffusers")
        if spec is None:
            pytest.skip("Set H3_DIFFUSERS_SOURCE to audit the target Diffusers source without installation")
        source = Path(spec.origin).parent
    for name in (
        "h3_loader_source",
        "h3_loader_source.loaders",
        "h3_loader_source.utils",
        "h3_loader_source.hooks",
        "peft",
        "peft.tuners",
        "peft.tuners.tuners_utils",
        "verl",
        "verl.utils",
        "verl.utils.fs",
    ):
        package = ModuleType(name)
        package.__path__ = []
        monkeypatch.setitem(sys.modules, name, package)
    peft = sys.modules["peft"]
    peft.LoraConfig, peft.inject_adapter_in_model = CpuLoraConfig, inject_cpu_adapter
    peft.set_peft_model_state_dict = set_cpu_adapter_state
    peft.get_peft_model_state_dict = cpu_adapter_state
    sys.modules["peft.tuners.tuners_utils"].BaseTunerLayer = CpuLoraLayer
    offload = ModuleType("h3_loader_source.hooks.group_offloading")
    offload._maybe_remove_and_reapply_group_offloading = no_tensor_parallel
    monkeypatch.setitem(sys.modules, offload.__name__, offload)
    parallel = ModuleType("h3_loader_source.hooks.tensor_parallel")
    parallel._raise_if_tensor_parallel = no_tensor_parallel
    monkeypatch.setitem(sys.modules, parallel.__name__, parallel)
    base = ModuleType("h3_loader_source.loaders.lora_base")
    base.LORA_ADAPTER_METADATA_KEY = "lora_adapter_metadata"
    monkeypatch.setitem(sys.modules, base.__name__, base)
    namespace = {
        "__package__": "h3_loader_source.utils",
        "os": os,
        "Path": Path,
        "json": json,
        "torch": torch,
        "safetensors": safetensors,
        "HF_HUB_OFFLINE": True,
        "LORA_WEIGHT_NAME": "pytorch_lora_weights.bin",
        "LORA_WEIGHT_NAME_SAFE": "pytorch_lora_weights.safetensors",
        "logger": logging.getLogger(__name__),
        "validate_hf_hub_args": validate_hf_hub_args,
        "get_peft_kwargs": None,
    }
    for filename, names in (
        ("utils/hub_utils.py", ("_get_model_file",)),
        ("utils/state_dict_utils.py", ("_load_sft_file_metadata", "_load_sft_state_dict_metadata")),
        ("loaders/lora_base.py", ("_best_guess_weight_name", "_fetch_state_dict")),
        ("utils/peft_utils.py", ("_create_lora_config", "_maybe_warn_for_unhandled_keys")),
    ):
        for name in names:
            load_source_function(source / filename, name, namespace)
    namespace["__package__"] = "h3_loader_source.loaders"
    load = load_source_function(source / "loaders/peft.py", "load_lora_adapter", namespace, "PeftAdapterMixin")
    config_spec = importlib.util.spec_from_file_location("h3_native_config", ROOT / "verl_omni/utils/config.py")
    config_module = importlib.util.module_from_spec(config_spec)
    monkeypatch.setitem(sys.modules, config_spec.name, config_module)
    config_spec.loader.exec_module(config_module)
    sys.modules["verl.utils.fs"].copy_to_local = local_source_path
    hooks = {"os": os, "importlib": importlib}
    invoke = load_source_function(ROOT / "verl_omni/pipelines/experiment_hooks.py", "invoke_experiment", hooks)
    build = load_source_function(
        ROOT / "verl_omni/workers/engine/lora_adapter_mixin.py",
        "_build_lora_module",
        {"resolve_lora_config": config_module.resolve_lora_config, "invoke_experiment": invoke},
        "LoRAAdapterMixin",
    )
    return SimpleNamespace(load=load, build=build)


def tiny_model(receipt, load):
    """Build only tiny base linears using the real 312-module H3 path layout."""
    model = torch.nn.Module()
    modules = {
        record["main_key"].removeprefix("transformer.").rsplit(".lora_", 1)[0] for record in receipt["tensors"].values()
    }
    for name in sorted(modules):
        parent = model
        parts = name.split(".")
        for part in parts[:-1]:
            if part not in parent._modules:
                parent.add_module(part, torch.nn.Module())
            parent = parent.get_submodule(part)
        parent.add_module(parts[-1], torch.nn.Linear(2, 3, bias=False, dtype=torch.bfloat16))
    model.load_lora_adapter = load.__get__(model)
    model._optionally_disable_offloading = no_offloading
    model._prepare_lora_hotswap_kwargs = None
    model._hf_peft_config_loaded = False
    return model


def test_real_loader_consumes_624_and_preserves_scale(api, source, tmp_path, real_loader):
    """Run real metadata/fetch/filter/config/load flow and read back every A/B parameter."""
    target = tmp_path / "converted"
    receipt = api.prepare_h3_adapter(source, target)
    model = tiny_model(receipt, real_loader.load)
    model.load_lora_adapter(str(target), adapter_name="default", use_safetensors=True)
    assert len(model.consumed_keys) == 624
    loaded = {
        name.replace(".default.weight", ".weight"): tensor
        for name, tensor in model.state_dict().items()
        if ".lora_" in name
    }
    expected = {
        record["main_key"].removeprefix("transformer."): {key: record[key] for key in ("shape", "dtype", "sha256")}
        for record in receipt["tensors"].values()
    }
    assert api.tensor_manifest(loaded) == expected
    assert all(parameter.requires_grad for name, parameter in model.named_parameters() if ".lora_" in name)
    for layer in model.modules():
        if isinstance(layer, CpuLoraLayer):
            assert (layer.r["default"], layer.lora_alpha["default"], layer.scaling["default"]) == (64, 128, 2)
    layer = model.get_submodule("transformer_blocks.0.attn.to_q")
    with torch.no_grad():
        layer.lora_B["default"].weight.fill_(0.25)
        inputs = torch.ones(1, 2, dtype=torch.bfloat16)
        expected = layer.base_layer(inputs) + 2 * layer.lora_B["default"](layer.lora_A["default"](inputs))
        assert torch.equal(layer(inputs), expected)


def test_real_default_requires_explicit_safetensors(api, source, tmp_path, real_loader):
    """Prove Diffusers' unchanged default call cannot fetch safetensors metadata."""
    target = tmp_path / "converted"
    receipt = api.prepare_h3_adapter(source, target)
    model = tiny_model(receipt, real_loader.load)
    with pytest.raises(OSError, match="pytorch_lora_weights.bin"):
        model.load_lora_adapter(str(target), adapter_name="default")
    assert not any(isinstance(layer, CpuLoraLayer) for layer in model.modules())


def test_real_main_flow_with_provider_callbacks(api, source, tmp_path, real_loader, monkeypatch):
    """Exercise main's real engine, generic provider dispatch, loader and post-load verification."""
    target = tmp_path / "converted"
    receipt = api.prepare_h3_adapter(source, target)
    model = tiny_model(receipt, real_loader.load)
    monkeypatch.setenv("VERL_EXPERIMENT_MODULE", api.__name__)
    monkeypatch.setenv("H3_ATTEMPT_DIR", str(tmp_path / "attempt"))
    engine = SimpleNamespace(
        model_config=SimpleNamespace(
            lora_adapter_path=str(target),
            use_shm=False,
            policy_state_adapters=("default",),
            lora_dtype=None,
        )
    )
    assert real_loader.build(engine, model) is model
    assert len(model.consumed_keys) == 624
    assert (model.peft_config["default"].r, model.peft_config["default"].lora_alpha) == (64, 128)
    report = json.loads((tmp_path / "attempt/determinism" / f"{os.getpid()}-adapter-loaded.json").read_text())
    assert report["hash_verified_tensor_count"] == 624


def test_original_source_is_filtered_by_real_default_prefix(api, source, tmp_path, real_loader, caplog):
    """Reproduce the historical 0/624 admission without inventing the loader's filtering logic."""
    receipt = api.prepare_h3_adapter(source, tmp_path / "converted")
    model = tiny_model(receipt, real_loader.load)
    model.load_lora_adapter(str(source), adapter_name="default", use_safetensors=True)
    assert not any(isinstance(layer, CpuLoraLayer) for layer in model.modules())
    assert "No LoRA keys" in caplog.text


def test_file_entrypoint_needs_no_application_imports(api, source, tmp_path, monkeypatch, capsys):
    """Execute the file entrypoint while actively rejecting every application-package import."""
    target = tmp_path / "standalone"
    original_import = builtins.__import__

    def reject_application_import(name, *args, **kwargs):
        """Guard the standalone CLI's dependency boundary, without stubbing its tensor or file IO."""
        assert name != "verl_omni" and not name.startswith("verl_omni.")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", reject_application_import)
    monkeypatch.setattr(sys, "argv", ["prepare", "--source", str(source), "--target", str(target)])
    runpy.run_path(str(ROOT / "verl_omni/experiments/prepare_h3_adapter.py"), run_name="__main__")
    assert json.loads(capsys.readouterr().out)["tensor_count"] == 624


@pytest.mark.parametrize("meta", [False, True])
def test_provider_callbacks_verify_loaded_or_defer_meta(api, source, tmp_path, real_loader, monkeypatch, meta):
    """Validate real loaded CPU parameters or all meta metadata without ever copying meta to CPU."""
    target = tmp_path / "converted"
    api.prepare_h3_adapter(source, target)
    config = SimpleNamespace(lora_adapter_path=str(target))
    kwargs = api.lora_loader_kwargs(config)
    assert kwargs == {"use_safetensors": True}
    receipt = json.loads((target / "conversion_receipt.json").read_text())
    model = tiny_model(receipt, real_loader.load)
    model.load_lora_adapter(str(target), adapter_name="default", **kwargs)
    if meta:
        model.to("meta")
    original_cpu = torch.Tensor.cpu

    def checked_cpu(tensor, *args, **kwargs):
        """Prove meta rank verification performs no CPU materialization."""
        assert not tensor.is_meta
        return original_cpu(tensor, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "cpu", checked_cpu)
    monkeypatch.setenv("H3_ATTEMPT_DIR", str(tmp_path / "attempt"))
    report = api.adapter_loaded(model, config)
    assert report["tensor_count"] == 624 and report["module_count"] == 312
    assert report["status"] == ("DEFERRED_META" if meta else "VERIFIED")
    assert report["hash_verified_tensor_count"] == (0 if meta else 624)
    assert report["deferred_meta_tensor_count"] == (624 if meta else 0)
    output = tmp_path / "attempt/determinism" / f"{os.getpid()}-adapter-loaded.json"
    assert json.loads(output.read_text()) == report
    assert all(("sha256" not in record) if meta else ("sha256" in record) for record in report["tensors"].values())


@pytest.mark.parametrize("problem", ["weights", "config", "receipt_count", "receipt_alpha"])
def test_provider_rejects_corrupt_conversion_before_load(api, source, tmp_path, problem):
    """Do not trust a receipt when output files or the required H3 contract changed."""
    target = tmp_path / "converted"
    api.prepare_h3_adapter(source, target)
    if problem == "weights":
        with (target / "pytorch_lora_weights.safetensors").open("ab") as stream:
            stream.write(b"changed")
    elif problem == "config":
        (target / "adapter_config.json").write_text("{}")
    else:
        path = target / "conversion_receipt.json"
        receipt = json.loads(path.read_text())
        receipt["tensor_count" if problem == "receipt_count" else "lora_alpha"] = 1
        path.write_text(json.dumps(receipt))
    with pytest.raises(ValueError):
        api.lora_loader_kwargs(SimpleNamespace(lora_adapter_path=str(target)))


@pytest.mark.parametrize("problem", ["hash", "dtype", "trainability", "scaling", "rank", "missing"])
def test_provider_rejects_wrong_actual_adapter(api, source, tmp_path, real_loader, monkeypatch, problem):
    """A successful file load cannot mask missing tensors, dtype casts, frozen weights or wrong scale."""
    target = tmp_path / "converted"
    receipt = api.prepare_h3_adapter(source, target)
    model = tiny_model(receipt, real_loader.load)
    model.load_lora_adapter(str(target), adapter_name="default", use_safetensors=True)
    layer = model.get_submodule("transformer_blocks.0.attn.to_q")
    if problem == "hash":
        with torch.no_grad():
            layer.lora_A["default"].weight.add_(1)
    elif problem == "dtype":
        layer.lora_A["default"].weight.data = layer.lora_A["default"].weight.float()
    elif problem == "trainability":
        layer.lora_A["default"].weight.requires_grad_(False)
    elif problem == "scaling":
        layer.scaling["default"] = 1
    elif problem == "rank":
        layer.r["default"] = 32
    else:
        del layer.lora_A["default"]
    monkeypatch.setenv("H3_ATTEMPT_DIR", str(tmp_path / "attempt"))
    with pytest.raises(ValueError):
        api.adapter_loaded(model, SimpleNamespace(lora_adapter_path=str(target)))
    assert not (tmp_path / "attempt").exists()


def test_cli_manifest_and_receipt(api, source, tmp_path, monkeypatch, capsys):
    """Verify the optional complete legacy manifest and CLI's source/target separation."""
    manifest = tmp_path / "tensor_manifest.json"
    manifest.write_text(json.dumps(api.tensor_manifest(load_file(source / "adapter_model.safetensors"))))
    target = tmp_path / "converted"
    monkeypatch.setattr(
        sys, "argv", ["prepare", "--source", str(source), "--target", str(target), "--manifest", str(manifest)]
    )
    api.main()
    assert json.loads(capsys.readouterr().out)["tensor_count"] == 624
