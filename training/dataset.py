from __future__ import annotations

import io
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageOps
from torch.utils.data import Dataset, IterableDataset
from torchvision import transforms
from transformers import PreTrainedTokenizerBase

DEFAULT_PROMPT_SUFFIX = (
    "Keep a single centered object, clean background, clear silhouette, "
    "product-style view, suitable for 3D asset generation."
)


@dataclass(frozen=True)
class Pix2PixRecord:
    original_image: Path
    edited_image: Path
    edit_prompt: str
    metadata_id: int | None = None
    original_dataset_index: int | None = None
    raw_index_value: int | str | None = None


@dataclass(frozen=True)
class MetadataSelection:
    selected_indices: set[int | str] | None = None
    index_field: str = "original_dataset_index"
    max_records: int | None = None
    skip_missing_images: bool = False


def build_training_prompt(
    edit_prompt: str,
    prompt_suffix: str = DEFAULT_PROMPT_SUFFIX,
) -> str:
    prompt = (edit_prompt or "").strip()
    if not prompt:
        raise ValueError("edit_prompt cannot be empty.")

    suffix = (prompt_suffix or "").strip()
    if not suffix:
        return prompt

    if prompt.endswith((".", "!", "?")):
        return f"{prompt} {suffix}"
    return f"{prompt}. {suffix}"


def resolve_image_path(metadata_path: str | Path, image_path: str) -> Path:
    metadata_file = Path(metadata_path).expanduser().resolve()
    candidate = Path(str(image_path)).expanduser()
    if candidate.is_absolute():
        return candidate.resolve()
    return (metadata_file.parent / candidate).resolve()


def load_index_filter_set(index_json_path: str | Path) -> set[int | str]:
    index_file = Path(index_json_path).expanduser().resolve()
    if not index_file.exists():
        raise FileNotFoundError(f"Index filter file not found: {index_file}")

    with index_file.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    if not isinstance(payload, list):
        raise ValueError(f"Index filter file must contain a JSON list: {index_file}")

    return {item for item in payload}


def _get_payload_index_value(
    payload: dict[str, Any],
    index_field: str,
    metadata_file: Path,
    line_number: int,
) -> int | str | None:
    if index_field not in payload:
        return None

    value = payload[index_field]
    if isinstance(value, (int, str)):
        return value

    raise TypeError(
        f"Unsupported index field type for '{index_field}' in {metadata_file} line {line_number}: "
        f"{type(value).__name__}."
    )


def _iter_metadata_records(
    metadata_path: str | Path,
    selection: MetadataSelection | None = None,
) -> list[Pix2PixRecord]:
    metadata_file = Path(metadata_path).expanduser().resolve()
    if not metadata_file.exists():
        raise FileNotFoundError(f"metadata.jsonl not found: {metadata_file}")

    selection = selection or MetadataSelection()
    records: list[Pix2PixRecord] = []

    with metadata_file.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            raw_line = line.strip()
            if not raw_line:
                continue

            payload = json.loads(raw_line)
            try:
                original_value = payload["original_image"]
                edited_value = payload["edited_image"]
                prompt_value = payload["edit_prompt"]
            except KeyError as exc:
                raise KeyError(
                    f"Missing field {exc!s} in {metadata_file} line {line_number}."
                ) from exc

            index_value = _get_payload_index_value(
                payload,
                index_field=selection.index_field,
                metadata_file=metadata_file,
                line_number=line_number,
            )
            if selection.selected_indices is not None and index_value not in selection.selected_indices:
                continue

            original_image = resolve_image_path(metadata_file, str(original_value))
            edited_image = resolve_image_path(metadata_file, str(edited_value))
            if selection.skip_missing_images and (
                not original_image.exists() or not edited_image.exists()
            ):
                continue

            records.append(
                Pix2PixRecord(
                    original_image=original_image,
                    edited_image=edited_image,
                    edit_prompt=str(prompt_value).strip(),
                    metadata_id=int(payload["id"]) if isinstance(payload.get("id"), int) else None,
                    original_dataset_index=(
                        int(payload["original_dataset_index"])
                        if isinstance(payload.get("original_dataset_index"), int)
                        else None
                    ),
                    raw_index_value=index_value,
                )
            )

            if selection.max_records is not None and len(records) >= selection.max_records:
                break

    return records


def load_metadata_records(
    metadata_path: str | Path,
    index_filter_json: str | Path | None = None,
    index_field: str = "original_dataset_index",
    max_records: int | None = None,
    skip_missing_images: bool = False,
    selected_indices: set[int | str] | None = None,
) -> list[Pix2PixRecord]:
    if selected_indices is None and index_filter_json is not None:
        selected_indices = load_index_filter_set(index_filter_json)

    records = _iter_metadata_records(
        metadata_path,
        selection=MetadataSelection(
            selected_indices=selected_indices,
            index_field=index_field,
            max_records=max_records,
            skip_missing_images=skip_missing_images,
        ),
    )
    if not records:
        raise ValueError(f"No records found in {Path(metadata_path).expanduser().resolve()}.")
    return records


def count_metadata_records(
    metadata_path: str | Path,
    index_filter_json: str | Path | None = None,
    index_field: str = "original_dataset_index",
    max_records: int | None = None,
    skip_missing_images: bool = False,
    selected_indices: set[int | str] | None = None,
) -> int:
    if selected_indices is None and index_filter_json is not None:
        selected_indices = load_index_filter_set(index_filter_json)

    return len(
        _iter_metadata_records(
            metadata_path,
            selection=MetadataSelection(
                selected_indices=selected_indices,
                index_field=index_field,
                max_records=max_records,
                skip_missing_images=skip_missing_images,
            ),
        )
    )


def _resample_mode() -> int:
    if hasattr(Image, "Resampling"):
        return Image.Resampling.BICUBIC
    return Image.BICUBIC


def prepare_square_image(
    image: Image.Image,
    resolution: int,
    resize_mode: str = "pad",
    background_color: tuple[int, int, int] = (255, 255, 255),
) -> Image.Image:
    if resolution <= 0:
        raise ValueError("resolution must be a positive integer.")

    image = ImageOps.exif_transpose(image).convert("RGB")
    if resize_mode == "crop":
        return ImageOps.fit(
            image,
            (resolution, resolution),
            method=_resample_mode(),
            centering=(0.5, 0.5),
        )

    if resize_mode != "pad":
        raise ValueError("resize_mode must be either 'pad' or 'crop'.")

    width, height = image.size
    if width <= 0 or height <= 0:
        raise ValueError("image has an invalid size.")

    scale = resolution / max(width, height)
    resized_size = (
        max(1, int(round(width * scale))),
        max(1, int(round(height * scale))),
    )
    resized = image.resize(resized_size, _resample_mode())

    canvas = Image.new("RGB", (resolution, resolution), color=background_color)
    offset = (
        (resolution - resized_size[0]) // 2,
        (resolution - resized_size[1]) // 2,
    )
    canvas.paste(resized, offset)
    return canvas


def coerce_image_to_pil(image_value: Any, field_name: str) -> Image.Image:
    if isinstance(image_value, Image.Image):
        return image_value.convert("RGB")

    if isinstance(image_value, np.ndarray):
        return Image.fromarray(image_value).convert("RGB")

    if isinstance(image_value, (str, Path)):
        image_path = Path(image_value).expanduser().resolve()
        if not image_path.exists():
            raise FileNotFoundError(f"{field_name} not found: {image_path}")
        with Image.open(image_path) as image:
            return image.convert("RGB")

    if isinstance(image_value, dict):
        image_bytes = image_value.get("bytes")
        image_path = image_value.get("path")

        if image_bytes is not None:
            with Image.open(io.BytesIO(image_bytes)) as image:
                return image.convert("RGB")

        if image_path:
            return coerce_image_to_pil(image_path, field_name)

    raise TypeError(
        f"Unsupported {field_name} type: {type(image_value).__name__}. "
        "Expected a PIL image, numpy array, path-like value, or datasets Image payload."
    )


class BasePix2PixDataset(Dataset):
    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase | None = None,
        resolution: int = 512,
        prompt_suffix: str = DEFAULT_PROMPT_SUFFIX,
        resize_mode: str = "pad",
        background_color: tuple[int, int, int] = (255, 255, 255),
    ) -> None:
        self.tokenizer = tokenizer
        self.resolution = int(resolution)
        self.prompt_suffix = prompt_suffix
        self.resize_mode = resize_mode
        self.background_color = background_color
        self.to_tensor = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ]
        )

    def __len__(self) -> int:
        raise NotImplementedError

    def _get_raw_example(self, index: int) -> dict[str, Any]:
        raise NotImplementedError

    def _prepare_loaded_image(self, image_value: Any, field_name: str) -> Image.Image:
        pil_image = coerce_image_to_pil(image_value, field_name)
        return prepare_square_image(
            pil_image,
            resolution=self.resolution,
            resize_mode=self.resize_mode,
            background_color=self.background_color,
        )

    def get_visual_example(self, index: int) -> dict[str, Any]:
        raw_example = self._get_raw_example(index)
        original_image = self._prepare_loaded_image(
            raw_example["original_image"],
            field_name="original_image",
        )
        edited_image = self._prepare_loaded_image(
            raw_example["edited_image"],
            field_name="edited_image",
        )
        prompt = build_training_prompt(
            str(raw_example["edit_prompt"]),
            self.prompt_suffix,
        )
        return {
            "original_image": original_image,
            "edited_image": edited_image,
            "prompt": prompt,
            "record": raw_example.get("record"),
        }

    def __getitem__(self, index: int) -> dict[str, Any]:
        example = self.get_visual_example(index)
        item: dict[str, Any] = {
            "original_pixel_values": self.to_tensor(example["original_image"]),
            "edited_pixel_values": self.to_tensor(example["edited_image"]),
            "prompt": example["prompt"],
        }

        if self.tokenizer is not None:
            tokenized = self.tokenizer(
                example["prompt"],
                max_length=self.tokenizer.model_max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            item["input_ids"] = tokenized.input_ids[0]

        return item


class Pix2PixJsonlDataset(BasePix2PixDataset):
    def __init__(
        self,
        metadata_path: str | Path,
        tokenizer: PreTrainedTokenizerBase | None = None,
        resolution: int = 512,
        prompt_suffix: str = DEFAULT_PROMPT_SUFFIX,
        resize_mode: str = "pad",
        background_color: tuple[int, int, int] = (255, 255, 255),
        index_filter_json: str | Path | None = None,
        index_field: str = "original_dataset_index",
        max_records: int | None = None,
        skip_missing_images: bool = False,
        selected_indices: set[int | str] | None = None,
    ) -> None:
        super().__init__(
            tokenizer=tokenizer,
            resolution=resolution,
            prompt_suffix=prompt_suffix,
            resize_mode=resize_mode,
            background_color=background_color,
        )
        self.metadata_path = Path(metadata_path).expanduser().resolve()
        self.records = load_metadata_records(
            self.metadata_path,
            index_filter_json=index_filter_json,
            index_field=index_field,
            max_records=max_records,
            skip_missing_images=skip_missing_images,
            selected_indices=selected_indices,
        )

    def __len__(self) -> int:
        return len(self.records)

    def _get_raw_example(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        return {
            "original_image": record.original_image,
            "edited_image": record.edited_image,
            "edit_prompt": record.edit_prompt,
            "record": record,
        }


class StreamingPix2PixJsonlDataset(IterableDataset):
    def __init__(
        self,
        metadata_path: str | Path,
        tokenizer: PreTrainedTokenizerBase | None = None,
        resolution: int = 512,
        prompt_suffix: str = DEFAULT_PROMPT_SUFFIX,
        resize_mode: str = "pad",
        background_color: tuple[int, int, int] = (255, 255, 255),
        index_filter_json: str | Path | None = None,
        index_field: str = "original_dataset_index",
        max_records: int | None = None,
        skip_missing_images: bool = True,
        shuffle: bool = True,
        seed: int = 42,
        selected_indices: set[int | str] | None = None,
    ) -> None:
        super().__init__()
        self.metadata_path = Path(metadata_path).expanduser().resolve()
        self.tokenizer = tokenizer
        self.resolution = int(resolution)
        self.prompt_suffix = prompt_suffix
        self.resize_mode = resize_mode
        self.background_color = background_color
        self.index_filter_json = index_filter_json
        self.index_field = index_field
        self.max_records = max_records
        self.skip_missing_images = skip_missing_images
        self.shuffle = shuffle
        self.seed = int(seed)
        self.selected_indices = (
            set(selected_indices)
            if selected_indices is not None
            else (
                load_index_filter_set(index_filter_json)
                if index_filter_json is not None
                else None
            )
        )
        self.to_tensor = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ]
        )
        self._iteration = 0

    def available_record_count(self) -> int:
        return count_metadata_records(
            metadata_path=self.metadata_path,
            index_field=self.index_field,
            max_records=self.max_records,
            skip_missing_images=self.skip_missing_images,
            selected_indices=self.selected_indices,
        )

    def _load_records(self) -> list[Pix2PixRecord]:
        return _iter_metadata_records(
            self.metadata_path,
            selection=MetadataSelection(
                selected_indices=self.selected_indices,
                index_field=self.index_field,
                max_records=self.max_records,
                skip_missing_images=self.skip_missing_images,
            ),
        )

    def _prepare_loaded_image(self, image_value: Any, field_name: str) -> Image.Image:
        pil_image = coerce_image_to_pil(image_value, field_name)
        return prepare_square_image(
            pil_image,
            resolution=self.resolution,
            resize_mode=self.resize_mode,
            background_color=self.background_color,
        )

    def _build_item(self, record: Pix2PixRecord) -> dict[str, Any]:
        prompt = build_training_prompt(record.edit_prompt, self.prompt_suffix)
        original_image = self._prepare_loaded_image(record.original_image, "original_image")
        edited_image = self._prepare_loaded_image(record.edited_image, "edited_image")
        item: dict[str, Any] = {
            "original_pixel_values": self.to_tensor(original_image),
            "edited_pixel_values": self.to_tensor(edited_image),
            "prompt": prompt,
        }
        if self.tokenizer is not None:
            tokenized = self.tokenizer(
                prompt,
                max_length=self.tokenizer.model_max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            item["input_ids"] = tokenized.input_ids[0]
        return item

    def __iter__(self):
        records = self._load_records()
        if self.shuffle and len(records) > 1:
            random.Random(self.seed + self._iteration).shuffle(records)
        self._iteration += 1

        for record in records:
            yield self._build_item(record)


class Pix2PixHFDataset(BasePix2PixDataset):
    def __init__(
        self,
        dataset: Any,
        original_image_column: str,
        edited_image_column: str,
        edit_prompt_column: str,
        tokenizer: PreTrainedTokenizerBase | None = None,
        resolution: int = 512,
        prompt_suffix: str = DEFAULT_PROMPT_SUFFIX,
        resize_mode: str = "pad",
        background_color: tuple[int, int, int] = (255, 255, 255),
    ) -> None:
        super().__init__(
            tokenizer=tokenizer,
            resolution=resolution,
            prompt_suffix=prompt_suffix,
            resize_mode=resize_mode,
            background_color=background_color,
        )
        self.dataset = dataset
        self.original_image_column = original_image_column
        self.edited_image_column = edited_image_column
        self.edit_prompt_column = edit_prompt_column

        column_names = set(getattr(dataset, "column_names", []))
        required_columns = {
            self.original_image_column,
            self.edited_image_column,
            self.edit_prompt_column,
        }
        missing_columns = sorted(required_columns - column_names)
        if missing_columns:
            raise ValueError(
                "Missing dataset columns: " + ", ".join(missing_columns)
            )

    def __len__(self) -> int:
        return len(self.dataset)

    def _get_raw_example(self, index: int) -> dict[str, Any]:
        row = self.dataset[index]
        return {
            "original_image": row[self.original_image_column],
            "edited_image": row[self.edited_image_column],
            "edit_prompt": str(row[self.edit_prompt_column]),
            "record": {"index": index},
        }


def collate_fn(examples: list[dict[str, Any]]) -> dict[str, Any]:
    batch: dict[str, Any] = {
        "original_pixel_values": torch.stack(
            [example["original_pixel_values"] for example in examples]
        ).contiguous(),
        "edited_pixel_values": torch.stack(
            [example["edited_pixel_values"] for example in examples]
        ).contiguous(),
        "prompts": [example["prompt"] for example in examples],
    }

    if "input_ids" in examples[0]:
        batch["input_ids"] = torch.stack(
            [example["input_ids"] for example in examples]
        ).contiguous()

    return batch
