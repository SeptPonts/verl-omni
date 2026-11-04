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
"""Compare frozen H3 identities and raw CPU evidence without model/checkpoint hooks.

Only tensor_manifest is retained from feature/h3-mlu-support-legacy. LoRA package
loading and checkpoint export belong to main's existing facilities. Relative
errors use the reference denominator; zero denominators remain explicitly
undefined rather than receiving an epsilon or being silently removed from MAPE.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from verl_omni.experiments.h3_evaluation_manifest import EVALUATION_SIZE, file_sha256, json_sha256


def tensor_manifest(state: dict) -> dict:
    """Hash detached CPU tensors, preserving shape, dtype and exact bytes."""
    import torch

    records = {}
    for name, value in sorted(state.items()):
        if value.device.type != "cpu":
            raise ValueError("tensor_manifest requires already materialized CPU tensors")
        records[name] = {
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "sha256": hashlib.sha256(
                value.detach().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
            ).hexdigest(),
        }
    return records


def json_differences(reference: Any, candidate: Any, path: str = "") -> list[dict]:
    """Return leaf differences with both raw values; distinguish missing keys from null."""
    if isinstance(reference, dict) and isinstance(candidate, dict):
        differences = []
        for key in sorted(reference.keys() | candidate.keys()):
            location = f"{path}/{key}"
            if key not in reference or key not in candidate:
                differences.append(
                    {
                        "path": location,
                        "reference_present": key in reference,
                        "candidate_present": key in candidate,
                        "reference": reference.get(key),
                        "candidate": candidate.get(key),
                    }
                )
            else:
                differences.extend(json_differences(reference[key], candidate[key], location))
        return differences
    if isinstance(reference, list) and isinstance(candidate, list) and len(reference) == len(candidate):
        return [
            difference
            for index, (left, right) in enumerate(zip(reference, candidate, strict=True))
            for difference in json_differences(left, right, f"{path}/{index}")
        ]
    if type(reference) is type(candidate) and reference == candidate:
        return []
    return [{"path": path, "reference": reference, "candidate": candidate}]


def compare_h3_manifests(reference: dict, candidate: dict) -> dict:
    """Compare consumed/source row semantics while excluding platform paths and Parquet bytes."""
    manifest_format = reference["format"]
    if candidate["format"] != manifest_format:
        raise ValueError("Cannot compare different H3 manifest formats")
    if manifest_format == "h3-training-manifest-v2":
        fields = (
            "format",
            "num_steps",
            "dataset_row_count",
            "all_row_fields_sha256",
            "sampling",
            "request_order",
            "steps",
        )
        left = {name: reference[name] for name in fields}
        right = {name: candidate[name] for name in fields}
    elif manifest_format == "h3-evaluation-manifest-v1":
        fields = ("format", "selection_rule", "sampling", "consumed_fields_sha256", "reevaluation", "samples")
        left = {name: reference[name] for name in fields}
        right = {name: candidate[name] for name in fields}
        left["all_row_fields_sha256"] = reference["source_files"][0]["all_row_fields_sha256"]
        right["all_row_fields_sha256"] = candidate["source_files"][0]["all_row_fields_sha256"]
    else:
        raise ValueError(f"Unsupported H3 manifest format: {manifest_format}")
    reference_sha256, candidate_sha256 = json_sha256(left), json_sha256(right)
    return {
        "equal": reference_sha256 == candidate_sha256,
        "reference_sha256": reference_sha256,
        "candidate_sha256": candidate_sha256,
        "differences": json_differences(left, right),
    }


def compare_h3_records(
    reference: list[dict],
    candidate: list[dict],
    *,
    steps: list[int],
    metric_names: list[str],
    sample_key: str | None = None,
    identity_fields: tuple[str, ...] = (),
) -> dict:
    """Compare the same explicit steps and sample keys, retaining each metric component.

    Records are flat mappings with global_step, requested metric names and any
    caller-selected identity fields (for example input_identity_sha256 or
    prompt_sha256). Duplicate/missing records, nonfinite values, missing metrics
    and identity mismatches fail instead of silently comparing an intersection.
    MAPE is a percentage and is null if any reference denominator is zero;
    nonzero_reference_mape_percent reports the separately labelled subset.
    """
    if not steps or len(set(steps)) != len(steps) or any(type(step) is not int or step < 0 for step in steps):
        raise ValueError("steps must be a nonempty list of unique nonnegative integers")
    if not metric_names or len(set(metric_names)) != len(metric_names):
        raise ValueError("metric_names must be nonempty and unique")
    window = set(steps)
    indexed = []
    for records in (reference, candidate):
        by_key = {}
        for record in records:
            step = record["global_step"]
            if type(step) is not int:
                raise ValueError("record global_step must be an integer")
            if step not in window:
                continue
            key = (step, record[sample_key]) if sample_key else (step,)
            if key in by_key:
                raise ValueError(f"Duplicate comparison record: {key}")
            by_key[key] = record
        if {key[0] for key in by_key} != window:
            raise ValueError("Comparison records do not cover the requested step window")
        indexed.append(by_key)
    left, right = indexed
    if left.keys() != right.keys():
        raise ValueError("Comparison sample identities differ within the requested step window")
    pairs = []
    errors = {name: [] for name in metric_names}
    for key, original in left.items():
        other = right[key]
        if any(original[name] != other[name] for name in identity_fields):
            raise ValueError(f"Comparison input identity mismatch at {key}")
        components = {}
        for name in metric_names:
            baseline, observed = original[name], other[name]
            if (
                type(baseline) not in (int, float)
                or type(observed) not in (int, float)
                or not math.isfinite(baseline)
                or not math.isfinite(observed)
            ):
                raise ValueError(f"Comparison requires finite numeric metric {name} at {key}")
            delta = observed - baseline
            absolute = abs(delta)
            relative = absolute / abs(baseline) if baseline != 0 else None
            component = {
                "reference": baseline,
                "candidate": observed,
                "delta": delta,
                "absolute_error": absolute,
                "absolute_percentage_error": 100 * relative if relative is not None else None,
            }
            components[name] = component
            errors[name].append(component)
        pair = {
            "global_step": key[0],
            "components": components,
            "identity": {name: original[name] for name in identity_fields},
        }
        if sample_key:
            pair[sample_key] = key[1]
        pairs.append(pair)
    summary = {}
    for name, components in errors.items():
        percentages = [
            part["absolute_percentage_error"] for part in components if part["absolute_percentage_error"] is not None
        ]
        nonzero_mape = math.fsum(percentages) / len(percentages) if percentages else None
        summary[name] = {
            "pair_count": len(components),
            "mae": math.fsum(part["absolute_error"] for part in components) / len(components),
            "max_absolute_error": max(part["absolute_error"] for part in components),
            "mape_percent": nonzero_mape if len(percentages) == len(components) else None,
            "nonzero_reference_mape_percent": nonzero_mape,
            "nonzero_reference_count": len(percentages),
            "zero_reference_count": len(components) - len(percentages),
            "zero_reference_mismatch_count": sum(
                part["reference"] == 0 and part["candidate"] != 0 for part in components
            ),
        }
    return {"steps": steps, "sample_key": sample_key, "summary": summary, "pairs": pairs}


def compare_h3_evaluation_records(
    reference: list[dict], candidate: list[dict], manifest: dict, *, steps: list[int], metric_names: list[str]
) -> dict:
    """Require every fixed128 sample and its frozen prompt hash at every requested evaluation step.

    Both raw streams must carry sample_id and prompt_sha256. Equal partial runs
    are still incomplete evaluations, even when their observed samples match.
    """
    expected = {sample["sample_id"]: sample["prompt_sha256"] for sample in manifest["samples"]}
    if (
        manifest["format"] != "h3-evaluation-manifest-v1"
        or len(manifest["samples"]) != EVALUATION_SIZE
        or len(expected) != EVALUATION_SIZE
    ):
        raise ValueError("Evaluation comparison requires a fixed128 manifest with unique sample IDs")
    for records in (reference, candidate):
        selected = [record for record in records if record["global_step"] in steps]
        for step in steps:
            if {record["sample_id"] for record in selected if record["global_step"] == step} != expected.keys():
                raise ValueError(f"Evaluation records do not cover fixed128 sample identities at step {step}")
        if any(record["prompt_sha256"] != expected[record["sample_id"]] for record in selected):
            raise ValueError("Evaluation record prompt hash differs from the frozen manifest")
    report = compare_h3_records(
        reference,
        candidate,
        steps=steps,
        metric_names=metric_names,
        sample_key="sample_id",
        identity_fields=("prompt_sha256",),
    )
    report["evaluation_inputs_sha256"] = manifest["consumed_fields_sha256"]
    return report


def read_h3_jsonl(path: str | Path, *, file_logger: bool = False) -> list[dict]:
    """Read raw records, or explicitly project main FileLogger's step/data schema."""
    with Path(path).open(encoding="utf-8") as stream:
        records = [json.loads(line) for line in stream]
    if file_logger:
        return [{**record["data"], "global_step": record["step"]} for record in records]
    return records


def main() -> None:
    """Write a reviewable comparison of original JSONL components on an explicit step window."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--steps", required=True, nargs="+", type=int)
    parser.add_argument("--metrics", required=True, nargs="+")
    parser.add_argument("--sample-key")
    parser.add_argument("--identity-fields", nargs="*", default=[])
    parser.add_argument("--file-logger", action="store_true")
    args = parser.parse_args()
    report = compare_h3_records(
        read_h3_jsonl(args.reference, file_logger=args.file_logger),
        read_h3_jsonl(args.candidate, file_logger=args.file_logger),
        steps=args.steps,
        metric_names=args.metrics,
        sample_key=args.sample_key,
        identity_fields=tuple(args.identity_fields),
    )
    report["source_files"] = {
        role: {"path": str(path.resolve()), "sha256": file_sha256(path)}
        for role, path in (("reference", args.reference), ("candidate", args.candidate))
    }
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


if __name__ == "__main__":
    main()
