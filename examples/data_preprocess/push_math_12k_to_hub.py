"""
Push preprocessed math_12k parquet splits to the Hugging Face Hub.

Expected on-disk layout (e.g. produced by `examples/data_preprocess/math_12k_srt.py`):

  examples/data_preprocess/data/math_12k/
    stage_0/
      train.parquet
      test.parquet
      (optional) validation.parquet
    stage_1/
      train.parquet
      test.parquet
      ...

This script pushes each stage as a separate dataset *config* (config_name="stage_0", ...),
containing splits train/test/(optional validation).

python examples/data_preprocess/push_math_12k_to_hub.py --data_dir ./data/math_12k/ --repo_id math_12k_srt_splits --hf_token $HF_TOKEN --private False
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Dict, List, Tuple

import datasets
from huggingface_hub import HfApi


def _discover_stage_dirs(data_dir: Path) -> List[Path]:
    stage_dirs = sorted([p for p in data_dir.iterdir() if p.is_dir() and p.name.startswith("stage_")])
    return stage_dirs


def _discover_splits(stage_dir: Path) -> Dict[str, Path]:
    # Common split naming conventions we support.
    split_candidates: List[Tuple[str, str]] = [
        ("train", "train.parquet"),
        ("validation", "validation.parquet"),
        ("valid", "valid.parquet"),  # will be normalized to "validation"
        ("val", "val.parquet"),      # will be normalized to "validation"
        ("test", "test.parquet"),
    ]

    found: Dict[str, Path] = {}
    for split, fname in split_candidates:
        p = stage_dir / fname
        if p.exists():
            normalized = "validation" if split in {"valid", "val"} else split
            # Prefer explicit validation.parquet if both exist; otherwise last one wins.
            found[normalized] = p
    return found


def _load_parquet_splits(split_files: Dict[str, Path]) -> datasets.DatasetDict:
    ds_dict: Dict[str, datasets.Dataset] = {}
    for split, path in split_files.items():
        ds_dict[split] = datasets.Dataset.from_parquet(str(path))
    return datasets.DatasetDict(ds_dict)


def main() -> int:
    parser = argparse.ArgumentParser(description="Push math_12k stage parquet splits to HF Hub.")
    parser.add_argument(
        "--data_dir",
        type=str,
        default="./examples/data_preprocess/data/math_12k",
        help="Directory containing stage_* subfolders with parquet split files.",
    )
    parser.add_argument(
        "--repo_id",
        type=str,
        required=True,
        help="Target HF dataset repo id, e.g. 'org_or_user/math_12k_srt'.",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Create repo as private (if it does not exist).",
    )
    parser.add_argument(
        "--hf_token",
        type=str,
        default=os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN"),
        help="HF token (or set HF_TOKEN / HUGGINGFACE_HUB_TOKEN env var).",
    )
    parser.add_argument(
        "--only_stage",
        type=str,
        default=None,
        help="Only push a specific stage folder name, e.g. 'stage_0'.",
    )
    parser.add_argument(
        "--max_shard_size",
        type=str,
        default="500MB",
        help="Max shard size for push_to_hub, e.g. '500MB' or '1GB'.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Discover and validate splits but do not push.",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir).expanduser().resolve()
    if not data_dir.exists():
        raise FileNotFoundError(f"data_dir does not exist: {data_dir}")

    stage_dirs = _discover_stage_dirs(data_dir)
    if args.only_stage is not None:
        stage_dirs = [p for p in stage_dirs if p.name == args.only_stage]

    if not stage_dirs:
        raise RuntimeError(
            f"No stage_* directories found under {data_dir}. "
            f"Expected e.g. {data_dir}/stage_0/train.parquet"
        )

    if not args.dry_run:
        if not args.hf_token:
            raise RuntimeError(
                "No HF token provided. Pass --hf_token or set HF_TOKEN / HUGGINGFACE_HUB_TOKEN."
            )
        # Ensure repo exists (idempotent).
        HfApi(token=args.hf_token).create_repo(
            repo_id=args.repo_id,
            repo_type="dataset",
            private=bool(args.private),
            exist_ok=True,
        )

    for stage_dir in stage_dirs:
        split_files = _discover_splits(stage_dir)
        if "train" not in split_files or "test" not in split_files:
            raise RuntimeError(
                f"{stage_dir} must contain at least train.parquet and test.parquet. "
                f"Found: {sorted(split_files.keys())}"
            )

        ds = _load_parquet_splits(split_files)

        print(f"[stage={stage_dir.name}] splits={list(ds.keys())} rows=" + ", ".join(
            f"{k}:{ds[k].num_rows}" for k in ds.keys()
        ))

        if args.dry_run:
            continue

        ds.push_to_hub(
            repo_id=args.repo_id,
            config_name=stage_dir.name,
            token=args.hf_token,
            max_shard_size=args.max_shard_size,
        )

    print("Done.")
    return 0


if __name__ == "__main__":
    
    main()

