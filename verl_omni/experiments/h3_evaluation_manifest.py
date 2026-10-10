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
"""Freeze the existing 128-row T2VA evaluation inputs without changing their sampling."""

import argparse
import hashlib
import json
from pathlib import Path

EVALUATION_SIZE = 128
EVALUATION_SEED = 42
INPUT_FIELDS = ("data_source", "prompt", "reward_model", "extra_info")
REEVALUATION_INDICES = tuple(index * (EVALUATION_SIZE - 1) // 7 for index in range(8))


def json_sha256(value):
    """Hash JSON values independently of mapping-key order and Parquet encoding."""
    serialized = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def h3_evaluation_samples(rows):
    """Project actual text-only H3 dataset inputs into ordered, source-indexed evaluation records."""
    if len(rows) != EVALUATION_SIZE:
        raise ValueError("H3 comparison requires the existing 128-row validation dataset in source order")
    samples = []
    for position, row in enumerate(rows):
        if any(not isinstance(message["content"], str) for message in row["prompt"]):
            raise ValueError("H3 evaluation manifest supports the existing text-only T2VA prompts")
        # H3's token-ID-native agent joins nonempty content and strips only the joined text.
        prompt_text = "\n".join(message["content"] for message in row["prompt"] if message["content"]).strip()
        source_index = row["extra_info"]["index"]
        source_split = row["extra_info"]["split"]
        samples.append(
            {
                "evaluation_index": position,
                "source_row_index": position,
                "source_index": source_index,
                "source_split": source_split,
                "sample_id": f"{row['data_source']}:{source_split}:{source_index}",
                "prompt_text": prompt_text,
                "prompt_sha256": hashlib.sha256(prompt_text.encode("utf-8")).hexdigest(),
                "seed": EVALUATION_SEED,
                "n": 1,
                "input_fields": {name: row[name] for name in INPUT_FIELDS},
            }
        )
    if len({sample["sample_id"] for sample in samples}) != len(samples):
        raise ValueError("H3 validation source identities must be unique")
    return samples


def create_h3_evaluation_manifest(parquet_path):
    """Read the actual Parquet and record byte identity separately from consumed input identity."""
    import pyarrow.parquet as parquet

    path = Path(parquet_path)
    table = parquet.read_table(path)
    rows = table.to_pylist()
    samples = h3_evaluation_samples(rows)
    with path.open("rb") as stream:
        parquet_sha256 = hashlib.file_digest(stream, "sha256").hexdigest()
    return {
        "format": "h3-evaluation-manifest-v1",
        "selection_rule": "all 128 rows in source order; no max_samples subsampling",
        "sampling": {"seed": EVALUATION_SEED, "n": 1},
        "source_files": [
            {
                "path": str(path.resolve()),
                "bytes": path.stat().st_size,
                "parquet_sha256": parquet_sha256,
                "row_count": len(rows),
                "columns": table.column_names,
                "all_row_fields_sha256": json_sha256(rows),
            }
        ],
        "consumed_fields_sha256": json_sha256([sample["input_fields"] for sample in samples]),
        "reevaluation": {
            "rule": "evaluation_index = floor(i * (128 - 1) / (8 - 1)), i = 0..7",
            "evaluation_indices": list(REEVALUATION_INDICES),
            "sample_ids": [samples[index]["sample_id"] for index in REEVALUATION_INDICES],
        },
        "samples": samples,
    }


def validate_h3_evaluation_dataset(config, val_dataset, manifest_path):
    """Check the real dataset and return stable IDs for validation metrics and media JSONL rows."""
    manifest = json.loads(Path(manifest_path).read_text())
    sampling = config.actor_rollout_ref.rollout.val_kwargs
    # DiffusionSamplingConfig supplies seed=42 when the YAML omits that dataclass field.
    if config.data.validation_shuffle or sampling.get("seed", EVALUATION_SEED) != EVALUATION_SEED or sampling.n != 1:
        raise ValueError("H3 evaluation requires validation_shuffle=false, val_kwargs.seed=42, and val_kwargs.n=1")
    if manifest["format"] != "h3-evaluation-manifest-v1" or manifest["sampling"] != {"seed": EVALUATION_SEED, "n": 1}:
        raise ValueError("H3 evaluation manifest sampling contract differs from seed42/n1")
    rows = [val_dataset.dataframe[index] for index in range(len(val_dataset))]
    samples = h3_evaluation_samples(rows)
    if json_sha256(rows) != manifest["source_files"][0]["all_row_fields_sha256"]:
        raise ValueError("H3 validation source rows differ from the frozen manifest")
    if (
        samples != manifest["samples"]
        or json_sha256([sample["input_fields"] for sample in samples]) != manifest["consumed_fields_sha256"]
    ):
        raise ValueError("H3 validation dataset order, source identities, or consumed fields differ from the manifest")
    for index, sample in enumerate(samples):
        if val_dataset[index]["raw_prompt"] != sample["input_fields"]["prompt"]:
            raise ValueError(f"H3 dataset raw_prompt differs from its source messages at evaluation index {index}")
    if manifest["reevaluation"]["evaluation_indices"] != list(REEVALUATION_INDICES) or manifest["reevaluation"][
        "sample_ids"
    ] != [samples[index]["sample_id"] for index in REEVALUATION_INDICES]:
        raise ValueError("H3 reevaluation must use the eight fixed, equally spaced evaluation indices")
    return manifest


def main():
    """Create an immutable CPU manifest directly from one platform's existing validation Parquet."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parquet", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    manifest = create_h3_evaluation_manifest(args.parquet)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(manifest, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(json.dumps({"manifest": str(args.output), "consumed_fields_sha256": manifest["consumed_fields_sha256"]}))


if __name__ == "__main__":
    main()
