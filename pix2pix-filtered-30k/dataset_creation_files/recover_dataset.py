#!/usr/bin/env python3
"""
recover_dataset.py

Purpose:
Rebuild a usable filtered InstructPix2Pix dataset from the public source dataset
`timbrooks/instructpix2pix-clip-filtered` using one or both of these files:

1) metadata.jsonl
2) final_indices.json

Why this exists:
- In some exported copies of the filtered dataset, the image folders are missing
  or incomplete.
- final_indices.json stores ORIGINAL source-dataset row indices.
- metadata.jsonl stores the exported file layout plus original_dataset_index.

This script supports two recovery modes:

Mode A: metadata.jsonl present
- Preserves your original relative paths from metadata.jsonl
- Rebuilds original_images/ and edited_images/
- Best when you want the recovered dataset to match your exported structure

Mode B: only final_indices.json present
- Rebuilds both image folders and generates a new metadata.jsonl
- Useful when metadata is missing but final_indices.json is available
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set

from PIL import Image
from datasets import load_dataset

SOURCE_DATASET = "timbrooks/instructpix2pix-clip-filtered"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rebuild original/edited images from metadata.jsonl and/or final_indices.json"
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=None,
        help="Path to metadata.jsonl (recommended if available)",
    )
    parser.add_argument(
        "--final-indices",
        type=Path,
        default=None,
        help="Path to final_indices.json (recommended if available)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Folder where the recovered dataset will be written",
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default=SOURCE_DATASET,
        help=f"Source Hugging Face dataset name (default: {SOURCE_DATASET})",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip saving images that already exist in the output folder",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional debug limit (recover only the first N matched samples)",
    )
    return parser.parse_args()


def load_final_indices(path: Path) -> List[int]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("final_indices.json must contain a JSON list")

    cleaned: List[int] = []
    for item in data:
        if not isinstance(item, int):
            raise ValueError("final_indices.json must contain only integers")
        cleaned.append(item)
    return cleaned


def load_metadata_rows(path: Path) -> List[dict]:
    rows: List[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if "original_dataset_index" not in row:
                raise ValueError(
                    f"metadata row {line_no} is missing 'original_dataset_index'"
                )
            rows.append(row)
    return rows


def ensure_rgb(img) -> Image.Image:
    if isinstance(img, Image.Image):
        return img.convert("RGB")
    return Image.open(img).convert("RGB")


def save_jsonl(rows: Iterable[dict], path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_targets_from_metadata(
    rows: List[dict], final_indices: Optional[Set[int]]
) -> Dict[int, dict]:
    target_map: Dict[int, dict] = {}
    for row in rows:
        idx = row["original_dataset_index"]
        if final_indices is not None and idx not in final_indices:
            continue
        target_map[idx] = row
    return target_map


def build_targets_from_indices(final_indices: List[int]) -> Dict[int, dict]:
    target_map: Dict[int, dict] = {}
    for new_id, idx in enumerate(final_indices):
        target_map[idx] = {
            "id": new_id,
            "original_dataset_index": idx,
            "edit_prompt": None,
            "original_image": f"original_images/{new_id:06d}.png",
            "edited_image": f"edited_images/{new_id:06d}.png",
        }
    return target_map


def main() -> int:
    args = parse_args()

    if args.metadata is None and args.final_indices is None:
        print("ERROR: provide at least one of --metadata or --final-indices")
        return 1

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "original_images").mkdir(exist_ok=True)
    (args.output_dir / "edited_images").mkdir(exist_ok=True)

    final_indices_list: Optional[List[int]] = None
    final_indices_set: Optional[Set[int]] = None
    if args.final_indices is not None:
        print(f"Loading final indices from: {args.final_indices}")
        final_indices_list = load_final_indices(args.final_indices)
        final_indices_set = set(final_indices_list)
        print(f"  Loaded {len(final_indices_list):,} original dataset indices")

    metadata_rows: Optional[List[dict]] = None
    if args.metadata is not None:
        print(f"Loading metadata rows from: {args.metadata}")
        metadata_rows = load_metadata_rows(args.metadata)
        print(f"  Loaded {len(metadata_rows):,} metadata rows")

    if metadata_rows is not None:
        # Best mode: preserve exported file layout from metadata.jsonl
        target_map = build_targets_from_metadata(metadata_rows, final_indices_set)
        print(
            "Using metadata-driven recovery mode "
            f"({len(target_map):,} rows selected after optional index filtering)"
        )
    else:
        assert final_indices_list is not None
        # Fallback mode: recreate a new exported dataset using final_indices order
        target_map = build_targets_from_indices(final_indices_list)
        print(f"Using index-only recovery mode ({len(target_map):,} rows selected)")

    target_indices = set(target_map.keys())
    if not target_indices:
        print("ERROR: no target rows were selected")
        return 1

    if args.limit is not None:
        # Keep a deterministic subset by original dataset index ordering.
        limited = sorted(target_indices)[: args.limit]
        target_indices = set(limited)
        target_map = {k: target_map[k] for k in limited}
        print(f"Debug limit active: only recovering {len(target_indices):,} rows")

    print(f"Streaming source dataset: {args.dataset_name}")
    dataset = load_dataset(args.dataset_name, split="train", streaming=True)

    rebuilt_rows: List[dict] = []
    processed = 0
    matched = 0

    print("Starting recovery...")
    for idx, sample in enumerate(dataset):
        if idx not in target_indices:
            continue

        row = dict(target_map[idx])
        processed += 1

        try:
            src = ensure_rgb(sample["original_image"])
            tgt = ensure_rgb(sample["edited_image"])
            prompt = sample["edit_prompt"]

            if row.get("edit_prompt") is None:
                row["edit_prompt"] = prompt

            src_rel = row["original_image"]
            tgt_rel = row["edited_image"]
            src_abs = args.output_dir / src_rel
            tgt_abs = args.output_dir / tgt_rel
            src_abs.parent.mkdir(parents=True, exist_ok=True)
            tgt_abs.parent.mkdir(parents=True, exist_ok=True)

            if not (args.skip_existing and src_abs.exists()):
                src.save(src_abs)
            if not (args.skip_existing and tgt_abs.exists()):
                tgt.save(tgt_abs)

            rebuilt_rows.append(row)
            matched += 1

            if matched % 500 == 0:
                print(f"  Recovered {matched:,} samples")

        except Exception as e:
            print(f"  Skipping dataset index {idx} because of error: {e}")

    metadata_out = args.output_dir / "metadata.jsonl"
    save_jsonl(rebuilt_rows, metadata_out)

    summary = {
        "source_dataset": args.dataset_name,
        "recovered_samples": matched,
        "mode": "metadata+indices" if metadata_rows is not None and final_indices_set is not None
        else "metadata_only" if metadata_rows is not None
        else "indices_only",
        "used_metadata": args.metadata is not None,
        "used_final_indices": args.final_indices is not None,
        "skip_existing": bool(args.skip_existing),
    }
    with (args.output_dir / "recovery_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\nDONE")
    print(f"Recovered samples: {matched:,}")
    print(f"Metadata saved to: {metadata_out}")
    print(f"Output folder: {args.output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
