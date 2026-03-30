from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path
from typing import Any

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build metadata.jsonl for InstructPix2Pix LoRA training.",
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=None,
        help="Directory to scan for paired images.",
    )
    parser.add_argument(
        "--output-metadata",
        type=Path,
        required=True,
        help="Where to write metadata.jsonl.",
    )
    parser.add_argument(
        "--images-dir",
        type=Path,
        default=None,
        help="Directory used to store copied training images. Defaults to <metadata_dir>/images.",
    )
    parser.add_argument(
        "--manifest-csv",
        type=Path,
        default=None,
        help="Optional CSV with columns: original_image, edited_image, edit_prompt.",
    )
    parser.add_argument(
        "--manifest-json",
        type=Path,
        default=None,
        help="Optional JSON file containing a list of metadata rows.",
    )
    parser.add_argument(
        "--prompt-map",
        type=Path,
        default=None,
        help="Optional JSON or CSV mapping pair key to edit prompt for scan mode.",
    )
    parser.add_argument(
        "--default-prompt",
        type=str,
        default="",
        help="Fallback prompt used when a row does not define edit_prompt.",
    )
    parser.add_argument(
        "--input-suffix",
        type=str,
        default="_input",
        help="Filename suffix used to identify source images in scan mode.",
    )
    parser.add_argument(
        "--target-suffix",
        type=str,
        default="_target",
        help="Filename suffix used to identify edited images in scan mode.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the generated rows without copying files or writing metadata.",
    )
    return parser.parse_args()


def _load_csv_rows(csv_path: Path) -> list[dict[str, str]]:
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        return [dict(row) for row in reader]


def _load_json_rows(json_path: Path) -> list[dict[str, Any]]:
    with json_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    if isinstance(payload, dict):
        if "items" in payload and isinstance(payload["items"], list):
            payload = payload["items"]
        else:
            payload = [payload]

    if not isinstance(payload, list):
        raise ValueError("manifest-json must contain a list of objects.")

    return [dict(item) for item in payload]


def _load_prompt_map(path: Path | None) -> dict[str, str]:
    if path is None:
        return {}

    path = path.expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Prompt map not found: {path}")

    if path.suffix.lower() == ".json":
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if isinstance(payload, dict):
            return {str(key): str(value) for key, value in payload.items()}
        if isinstance(payload, list):
            mapping: dict[str, str] = {}
            for item in payload:
                key = str(item["key"])
                mapping[key] = str(item["edit_prompt"])
            return mapping
        raise ValueError("prompt-map JSON must be a dict or a list of objects.")

    rows = _load_csv_rows(path)
    mapping = {}
    for row in rows:
        key = str(row.get("key", "")).strip()
        prompt = str(row.get("edit_prompt", "")).strip()
        if key and prompt:
            mapping[key] = prompt
    return mapping


def _iter_image_files(root: Path) -> list[Path]:
    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def _match_pair_key(path: Path, suffix: str) -> str | None:
    if not path.stem.endswith(suffix):
        return None
    return path.stem[: -len(suffix)]


def _read_manifest_rows(args: argparse.Namespace) -> list[dict[str, str]]:
    if args.manifest_csv is not None and args.manifest_json is not None:
        raise ValueError("Use either --manifest-csv or --manifest-json, not both.")

    if args.manifest_csv is not None:
        manifest_path = args.manifest_csv.expanduser().resolve()
        rows = _load_csv_rows(manifest_path)
    elif args.manifest_json is not None:
        manifest_path = args.manifest_json.expanduser().resolve()
        rows = _load_json_rows(manifest_path)
    else:
        manifest_path = None
        rows = []

    base_dir = manifest_path.parent if manifest_path is not None else None
    normalized_rows: list[dict[str, str]] = []
    for row in rows:
        normalized_rows.append(
            {
                "original_image": str(row.get("original_image", "")).strip(),
                "edited_image": str(row.get("edited_image", "")).strip(),
                "edit_prompt": str(
                    row.get("edit_prompt", "") or args.default_prompt
                ).strip(),
                "_base_dir": str(base_dir) if base_dir is not None else "",
            }
        )
    return normalized_rows


def _scan_rows(args: argparse.Namespace) -> list[dict[str, str]]:
    if args.source_dir is None:
        raise ValueError(
            "source-dir is required when manifest-csv / manifest-json is not provided."
        )

    source_dir = args.source_dir.expanduser().resolve()
    prompt_map = _load_prompt_map(args.prompt_map)
    input_index: dict[str, Path] = {}
    target_index: dict[str, Path] = {}

    for image_path in _iter_image_files(source_dir):
        input_key = _match_pair_key(image_path, args.input_suffix)
        if input_key:
            input_index[input_key] = image_path
            continue

        target_key = _match_pair_key(image_path, args.target_suffix)
        if target_key:
            target_index[target_key] = image_path

    rows: list[dict[str, str]] = []
    for key in sorted(input_index):
        original_image = input_index[key]
        edited_image = target_index.get(key)
        if edited_image is None:
            continue

        edit_prompt = prompt_map.get(key, args.default_prompt).strip()
        if not edit_prompt:
            raise ValueError(
                f"No edit_prompt found for pair '{key}'. "
                "Use --default-prompt or --prompt-map."
            )

        rows.append(
            {
                "original_image": str(original_image),
                "edited_image": str(edited_image),
                "edit_prompt": edit_prompt,
            }
        )

    if not rows:
        raise ValueError("No paired images were found.")

    return rows


def _resolve_existing_path(raw_path: str, base_dir: Path | None = None) -> Path:
    candidate = Path(raw_path).expanduser()
    if candidate.is_absolute():
        return candidate.resolve()
    if base_dir is not None:
        return (base_dir / candidate).resolve()
    return candidate.resolve()


def _build_output_rows(
    rows: list[dict[str, str]],
    output_metadata: Path,
    images_dir: Path,
) -> list[dict[str, str]]:
    metadata_dir = output_metadata.parent.resolve()
    output_rows: list[dict[str, str]] = []
    images_dir.mkdir(parents=True, exist_ok=True)

    for index, row in enumerate(rows, start=1):
        source_base_dir = Path(row.get("_base_dir", "")).resolve() if row.get("_base_dir") else metadata_dir
        original_path = _resolve_existing_path(row["original_image"], base_dir=source_base_dir)
        edited_path = _resolve_existing_path(row["edited_image"], base_dir=source_base_dir)

        if not original_path.exists():
            raise FileNotFoundError(f"Original image not found: {original_path}")
        if not edited_path.exists():
            raise FileNotFoundError(f"Edited image not found: {edited_path}")

        input_name = f"{index:04d}_input{original_path.suffix.lower() or '.png'}"
        target_name = f"{index:04d}_target{edited_path.suffix.lower() or '.png'}"
        copied_original = images_dir / input_name
        copied_edited = images_dir / target_name

        shutil.copy2(original_path, copied_original)
        shutil.copy2(edited_path, copied_edited)

        output_rows.append(
            {
                "original_image": str(copied_original.relative_to(metadata_dir)).replace(
                    "\\",
                    "/",
                ),
                "edited_image": str(copied_edited.relative_to(metadata_dir)).replace(
                    "\\",
                    "/",
                ),
                "edit_prompt": row["edit_prompt"].strip(),
            }
        )

    return output_rows


def _write_metadata_jsonl(metadata_path: Path, rows: list[dict[str, str]]) -> None:
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    with metadata_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()
    output_metadata = args.output_metadata.expanduser().resolve()
    images_dir = (
        args.images_dir.expanduser().resolve()
        if args.images_dir is not None
        else (output_metadata.parent / "images").resolve()
    )

    manifest_rows = _read_manifest_rows(args)
    rows = manifest_rows if manifest_rows else _scan_rows(args)

    if args.dry_run:
        for row in rows:
            print(json.dumps(row, ensure_ascii=False))
        print(f"\nDry run complete. {len(rows)} rows prepared.")
        return

    output_rows = _build_output_rows(rows, output_metadata, images_dir)
    _write_metadata_jsonl(output_metadata, output_rows)

    print(f"Wrote {len(output_rows)} rows to: {output_metadata}")
    print(f"Copied paired images to: {images_dir}")


if __name__ == "__main__":
    main()


