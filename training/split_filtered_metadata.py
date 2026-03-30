from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Filter metadata.jsonl by a JSON index list and split it into train/val subsets.",
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        required=True,
        help="Source metadata.jsonl path.",
    )
    parser.add_argument(
        "--index-json",
        type=Path,
        required=True,
        help="JSON list of allowed indices, for example final_indices.json.",
    )
    parser.add_argument(
        "--train-output",
        type=Path,
        required=True,
        help="Output train subset jsonl path.",
    )
    parser.add_argument(
        "--val-output",
        type=Path,
        required=True,
        help="Output validation subset jsonl path.",
    )
    parser.add_argument(
        "--index-field",
        type=str,
        default="original_dataset_index",
        help="Metadata field used to match entries from the index JSON.",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.05,
        help="Validation ratio applied after filtering. Default: 0.05.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used for deterministic splitting.",
    )
    parser.add_argument(
        "--max-records",
        type=int,
        default=None,
        help="Optional cap on the number of filtered records before splitting.",
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=None,
        help="Optional JSON file summarising the split.",
    )
    return parser.parse_args()


def load_index_set(index_json: Path) -> set[int | str]:
    payload = json.loads(index_json.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"Index JSON must be a list: {index_json}")
    return {item for item in payload}


def load_filtered_rows(
    metadata_path: Path,
    allowed_indices: set[int | str],
    index_field: str,
    max_records: int | None,
) -> list[dict[str, Any]]:
    filtered_rows: list[dict[str, Any]] = []
    with metadata_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            raw = line.strip()
            if not raw:
                continue
            row = json.loads(raw)
            if index_field not in row:
                raise KeyError(
                    f"Missing '{index_field}' in {metadata_path} line {line_number}."
                )
            if row[index_field] not in allowed_indices:
                continue
            filtered_rows.append(row)
            if max_records is not None and len(filtered_rows) >= max_records:
                break
    if not filtered_rows:
        raise ValueError("No rows matched the provided index filter.")
    return filtered_rows


def split_rows(
    rows: list[dict[str, Any]],
    val_ratio: float,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not 0.0 < val_ratio < 1.0:
        raise ValueError("--val-ratio must be in the range (0, 1).")

    shuffled_rows = list(rows)
    random.Random(seed).shuffle(shuffled_rows)

    val_count = max(1, int(round(len(shuffled_rows) * val_ratio)))
    if val_count >= len(shuffled_rows):
        val_count = max(1, len(shuffled_rows) - 1)

    val_rows = shuffled_rows[:val_count]
    train_rows = shuffled_rows[val_count:]
    if not train_rows:
        raise ValueError("Validation split consumed all rows. Reduce --val-ratio.")
    return train_rows, val_rows


def write_jsonl(rows: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()
    metadata_path = args.metadata.expanduser().resolve()
    index_json = args.index_json.expanduser().resolve()
    train_output = args.train_output.expanduser().resolve()
    val_output = args.val_output.expanduser().resolve()
    summary_output = (
        args.summary_output.expanduser().resolve()
        if args.summary_output is not None
        else None
    )

    allowed_indices = load_index_set(index_json)
    filtered_rows = load_filtered_rows(
        metadata_path=metadata_path,
        allowed_indices=allowed_indices,
        index_field=args.index_field,
        max_records=args.max_records,
    )
    train_rows, val_rows = split_rows(
        rows=filtered_rows,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )

    write_jsonl(train_rows, train_output)
    write_jsonl(val_rows, val_output)

    project_root = Path.cwd().resolve()

    def _rel(path: Path) -> str:
        try:
            return path.relative_to(project_root).as_posix()
        except ValueError:
            return str(path)

    summary = {
        "metadata": _rel(metadata_path),
        "index_json": _rel(index_json),
        "index_field": args.index_field,
        "input_filtered_records": len(filtered_rows),
        "train_records": len(train_rows),
        "val_records": len(val_rows),
        "val_ratio": args.val_ratio,
        "seed": args.seed,
        "max_records": args.max_records,
        "train_output": _rel(train_output),
        "val_output": _rel(val_output),
    }

    if summary_output is not None:
        summary_output.parent.mkdir(parents=True, exist_ok=True)
        summary_output.write_text(
            json.dumps(summary, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
