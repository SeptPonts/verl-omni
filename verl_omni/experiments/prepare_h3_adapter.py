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
"""Convert a read-only canonical/native PEFT H3 adapter for main's Diffusers loader.

Only key namespaces and safetensors metadata change. The current main caller
must additionally pass use_safetensors=True to load_lora_adapter; Diffusers'
default None otherwise bypasses the safetensors metadata path. CPU conversion
receipts prove file/tensor identity, not full-model or device load acceptance.
"""

import argparse
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file


def file_sha256(path: str | Path) -> str:
    """Hash files without importing the application's model-registration package."""
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def tensor_manifest(state: dict[str, torch.Tensor]) -> dict:
    """Record canonical shape/dtype/byte identities for already loaded CPU tensors."""
    return {
        name: {
            "shape": list(tensor.shape),
            "dtype": str(tensor.dtype),
            "sha256": hashlib.sha256(tensor.contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest(),
        }
        for name, tensor in sorted(state.items())
    }


def read_conversion_receipt(model_config: Any) -> tuple[Path, dict]:
    """Validate the H3 conversion contract shared by the two opt-in provider callbacks."""
    directory = Path(model_config.lora_adapter_path).expanduser().resolve()
    receipt = json.loads((directory / "conversion_receipt.json").read_text(encoding="utf-8"))
    if (
        receipt["format"] != "h3-main-adapter-conversion-v1"
        or receipt["status"] != "COMPLETE"
        or (receipt["tensor_count"], receipt["module_count"], receipt["r"], receipt["lora_alpha"], receipt["scaling"])
        != (624, 312, 64, 128, 2)
        or len(receipt["tensors"]) != 624
        or len({record["main_key"] for record in receipt["tensors"].values()}) != 624
    ):
        raise ValueError("H3 requires a complete 624-tensor rank64/alpha128 conversion receipt")
    return directory, receipt


def lora_loader_kwargs(model_config: Any) -> dict:
    """Verify converted files and metadata before main loads any adapter tensors."""
    directory, receipt = read_conversion_receipt(model_config)
    for name in ("pytorch_lora_weights.safetensors", "adapter_config.json"):
        if file_sha256(directory / name) != receipt["target_files_sha256"][name]:
            raise ValueError(f"H3 converted file checksum mismatch: {name}")
    config = json.loads((directory / "adapter_config.json").read_text(encoding="utf-8"))
    with safe_open(directory / "pytorch_lora_weights.safetensors", framework="pt", device="cpu") as stream:
        metadata = json.loads(stream.metadata()["lora_adapter_metadata"])
        if metadata != {"transformer." + name: value for name, value in config.items()}:
            raise ValueError("H3 safetensors metadata differs from adapter_config")
        if (
            config["r"] != 64
            or config["lora_alpha"] != 128
            or set(stream.keys()) != {record["main_key"] for record in receipt["tensors"].values()}
        ):
            raise ValueError("H3 converted adapter keys or rank/alpha differ from receipt")
    return {"use_safetensors": True}


def adapter_loaded(module: torch.nn.Module, model_config: Any) -> dict:
    """Read back default A/B parameters; defer hashes only for actual meta tensors.

    Called after main's adapter load and dtype cast, before FSDP materialization.
    A DEFERRED_META result requires a later materialization/sync acceptance gate;
    neither its expected source hashes nor successful eval0 prove a loaded state.
    """
    from peft import get_peft_model_state_dict
    from peft.tuners.tuners_utils import BaseTunerLayer

    directory, receipt = read_conversion_receipt(model_config)
    expected = {record["main_key"].removeprefix("transformer."): record for record in receipt["tensors"].values()}
    raw_state = get_peft_model_state_dict(module, adapter_name="default")
    if len(raw_state) != 624:
        raise ValueError("H3 loaded adapter must contain exactly 624 A/B tensors before normalization")
    state = {name.removeprefix("base_model.model."): tensor for name, tensor in raw_state.items()}
    parameters = {
        name.removeprefix("base_model.model.")
        .replace(".lora_A.default.weight", ".lora_A.weight")
        .replace(".lora_B.default.weight", ".lora_B.weight"): parameter
        for name, parameter in module.named_parameters()
        if name.endswith((".lora_A.default.weight", ".lora_B.default.weight"))
    }
    if len(state) != 624 or state.keys() != expected.keys() or parameters.keys() != expected.keys():
        raise ValueError("H3 loaded adapter must consume all 624 expected default A/B tensors")
    records = {}
    for name, tensor in state.items():
        record = expected[name]
        if list(tensor.shape) != record["shape"] or str(tensor.dtype) != record["dtype"]:
            raise ValueError(f"H3 loaded adapter shape/dtype mismatch: {name}")
        if not parameters[name].requires_grad:
            raise ValueError(f"H3 loaded default adapter is not trainable: {name}")
        if tensor.is_meta:
            records[name] = {
                "status": "DEFERRED_META",
                "shape": list(tensor.shape),
                "dtype": str(tensor.dtype),
                "expected_sha256": record["sha256"],
                "trainable": True,
            }
        else:
            actual = tensor_manifest({name: tensor.detach().cpu()})[name]
            if actual != {field: record[field] for field in ("shape", "dtype", "sha256")}:
                raise ValueError(f"H3 loaded adapter checksum mismatch: {name}")
            records[name] = {"status": "VERIFIED", **actual, "trainable": True}
    layers = {}
    for name, layer in module.named_modules():
        if isinstance(layer, BaseTunerLayer):
            layout = {
                "r": layer.r["default"],
                "lora_alpha": layer.lora_alpha["default"],
                "scaling": layer.scaling["default"],
            }
            if layout != {"r": 64, "lora_alpha": 128, "scaling": 2}:
                raise ValueError(f"H3 loaded layer rank/alpha/scaling mismatch: {name}")
            layers[name.removeprefix("base_model.model.")] = layout
    if layers.keys() != {name.rsplit(".lora_", 1)[0] for name in expected}:
        raise ValueError("H3 loaded adapter layer coverage differs from its 312 source modules")
    # PEFT condenses targets during load; sync uses the verified full module paths.
    module.peft_config["default"].target_modules = set(layers)
    deferred = sum(record["status"] == "DEFERRED_META" for record in records.values())
    report = {
        "status": "DEFERRED_META" if deferred else "VERIFIED",
        "pid": os.getpid(),
        "adapter_path": str(directory),
        "tensor_count": 624,
        "module_count": len(layers),
        "hash_verified_tensor_count": 624 - deferred,
        "deferred_meta_tensor_count": deferred,
        "conversion_receipt_sha256": file_sha256(directory / "conversion_receipt.json"),
        "tensors": records,
        "layers": layers,
        "post_materialization_and_sync": "NOT_RUN",
    }
    output = Path(os.environ["H3_ATTEMPT_DIR"]) / "determinism" / f"{os.getpid()}-adapter-loaded.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    return report


def prepare_h3_adapter(source: str | Path, target: str | Path, *, manifest: str | Path | None = None) -> dict:
    """Write a new Diffusers adapter, preserving all 624 original A/B tensor bytes.

    Source needs only adapter_config.json and adapter_model.safetensors; native
    base_model.model prefixes are removed before adding transformer. An optional
    legacy tensor manifest must exactly match the original input key namespace.
    Neither an existing destination nor any destination inside source is allowed.
    Failed writes remain visible without a COMPLETE receipt and are never reused.
    """
    source, target = Path(source).expanduser().resolve(), Path(target).expanduser().resolve()
    if source == target or target.is_relative_to(source):
        raise ValueError("target must be separate from and outside the immutable source")
    if target.exists():
        raise FileExistsError(target)
    config_path, weights_path = source / "adapter_config.json", source / "adapter_model.safetensors"
    source_hashes = {path.name: file_sha256(path) for path in (config_path, weights_path)}
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config["r"] != 64 or config["lora_alpha"] != 128:
        raise ValueError("H3 adapter requires rank64/alpha128/scaling2")
    if (
        config.get("use_rslora", False)
        or config.get("use_dora", False)
        or config.get("rank_pattern")
        or config.get("alpha_pattern")
        or config.get("bias", "none") != "none"
        or config.get("lora_bias", False)
        or config.get("modules_to_save")
        or config.get("target_parameters")
    ):
        raise ValueError("H3 conversion requires uniform ordinary A/B LoRA without extra trainable state")
    targets = config["target_modules"]
    if not isinstance(targets, list) or not targets or any(not isinstance(name, str) or not name for name in targets):
        raise ValueError("H3 adapter target_modules must be a nonempty PEFT suffix list")
    state = load_file(weights_path, device="cpu")
    if len(state) != 624:
        raise ValueError(f"H3 adapter requires exactly 624 A/B tensors, found {len(state)}")
    original = tensor_manifest(state)
    if manifest is not None and json.loads(Path(manifest).read_text(encoding="utf-8")) != original:
        raise ValueError("Source tensors differ from the supplied complete tensor manifest")
    converted, mapping, modules = {}, {}, set()
    for name, tensor in state.items():
        normalized = name.removeprefix("base_model.model.")
        match = re.fullmatch(r"(.+)\.lora_([AB])\.weight", normalized)
        if match is None or normalized.startswith("transformer."):
            raise ValueError(f"Unsupported source LoRA key: {name}")
        module, component = match.groups()
        if tensor.ndim != 2 or tensor.shape[0 if component == "A" else 1] != 64 or not tensor.is_floating_point():
            raise ValueError(f"Invalid rank64 LoRA tensor shape/dtype: {name}")
        if not any(module == suffix or module.endswith("." + suffix) for suffix in targets):
            raise ValueError(f"Source tensor is outside adapter_config target_modules: {name}")
        output_name = "transformer." + normalized
        if output_name in converted:
            raise ValueError(f"Source key collision after namespace normalization: {name}")
        converted[output_name] = tensor
        mapping[name] = output_name
        modules.add(module)
    for module in modules:
        if any("transformer." + module + f".lora_{component}.weight" not in converted for component in ("A", "B")):
            raise ValueError(f"Missing A/B partner for {module}")
    if any(not any(module == suffix or module.endswith("." + suffix) for module in modules) for suffix in targets):
        raise ValueError("adapter_config contains target_modules without source A/B tensors")
    # Exact module paths prevent a broad exporter suffix from injecting unsaved adapters.
    output_config = {**config, "target_modules": sorted(modules)}
    metadata_config = {"transformer." + name: value for name, value in output_config.items()}
    metadata = {"format": "pt", "lora_adapter_metadata": json.dumps(metadata_config, sort_keys=True, allow_nan=False)}
    target.mkdir(parents=True, exist_ok=False)
    output_weights = target / "pytorch_lora_weights.safetensors"
    save_file(converted, output_weights, metadata=metadata)
    (target / "adapter_config.json").write_text(
        json.dumps(output_config, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    roundtrip = tensor_manifest(load_file(output_weights, device="cpu"))
    if roundtrip != {mapping[name]: record for name, record in original.items()}:
        raise ValueError("Converted adapter tensor roundtrip differs from original bytes")
    with safe_open(output_weights, framework="pt", device="cpu") as stream:
        if json.loads(stream.metadata()["lora_adapter_metadata"]) != metadata_config:
            raise ValueError("Converted adapter metadata roundtrip mismatch")
    if source_hashes != {path.name: file_sha256(path) for path in (config_path, weights_path)}:
        raise ValueError("Source changed during conversion")
    receipt = {
        "status": "COMPLETE",
        "format": "h3-main-adapter-conversion-v1",
        "source": str(source),
        "target": str(target),
        "source_files_sha256": source_hashes,
        "target_files_sha256": {
            name: file_sha256(target / name) for name in ("pytorch_lora_weights.safetensors", "adapter_config.json")
        },
        "r": 64,
        "lora_alpha": 128,
        "scaling": 2,
        "tensor_count": len(roundtrip),
        "module_count": len(modules),
        "required_loader_kwargs": {"use_safetensors": True, "adapter_name": "default"},
        "tensor_roundtrip": "PASS",
        "full_model_load": "NOT_RUN",
        "tensors": {name: {"main_key": mapping[name], **record} for name, record in original.items()},
    }
    with (target / "conversion_receipt.json").open("x", encoding="utf-8") as stream:
        json.dump(receipt, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    return receipt


def main() -> None:
    """Convert into a new directory; never overwrite or mutate the original PEFT package."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--target", required=True, type=Path)
    parser.add_argument("--manifest", type=Path, help="Optional complete original-key tensor manifest")
    args = parser.parse_args()
    receipt = prepare_h3_adapter(args.source, args.target, manifest=args.manifest)
    print(
        json.dumps({name: receipt[name] for name in ("status", "target", "tensor_count", "r", "lora_alpha", "scaling")})
    )


if __name__ == "__main__":
    main()
