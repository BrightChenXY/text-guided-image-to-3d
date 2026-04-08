from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from PIL import Image
from diffusers import StableDiffusionInstructPix2PixPipeline

from config import (
    DEFAULT_SEED,
    INSTRUCT_PIX2PIX_LORA_PATH,
    INSTRUCT_PIX2PIX_LORA_SCALE,
    INSTRUCT_PIX2PIX_MODEL_ID,
)


_PIPELINE_CACHE: dict[str, StableDiffusionInstructPix2PixPipeline] = {}


def _get_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def _get_dtype() -> torch.dtype:
    return torch.float16 if torch.cuda.is_available() else torch.float32


def _build_generator(seed: int) -> torch.Generator:
    if torch.cuda.is_available():
        return torch.Generator(device="cuda").manual_seed(seed)
    return torch.Generator().manual_seed(seed)


def _looks_like_lora_dir(path: Path) -> bool:
    if not path.exists() or not path.is_dir():
        return False

    expected_files = (
        "pytorch_lora_weights.safetensors",
        "pytorch_lora_weights.bin",
        "adapter_model.safetensors",
        "adapter_model.bin",
        "adapter_config.json",
    )
    return any((path / filename).exists() for filename in expected_files)



def _resolve_lora_path(lora_path: str | Path | None) -> Path | None:
    candidate = lora_path if lora_path is not None else INSTRUCT_PIX2PIX_LORA_PATH
    if not candidate:
        return None

    resolved = Path(candidate).expanduser().resolve()
    if resolved.is_file() or _looks_like_lora_dir(resolved):
        return resolved

    if resolved.name == "best_checkpoint":
        best_lora_dir = resolved / "lora"
        if _looks_like_lora_dir(best_lora_dir):
            return best_lora_dir.resolve()

    nested_best_lora_dir = resolved / "best_checkpoint" / "lora"
    if _looks_like_lora_dir(nested_best_lora_dir):
        return nested_best_lora_dir.resolve()

    nested_lora_dir = resolved / "lora"
    if _looks_like_lora_dir(nested_lora_dir):
        return nested_lora_dir.resolve()

    return resolved


def _extract_single_image(result: Any) -> Image.Image:
    if isinstance(result, tuple):
        if not result:
            raise RuntimeError("InstructPix2Pix returned an empty tuple.")
        result = result[0]

    if hasattr(result, "images"):
        images = getattr(result, "images")
        if isinstance(images, (list, tuple)) and images:
            result = images[0]

    if isinstance(result, (list, tuple)):
        if not result:
            raise RuntimeError("InstructPix2Pix returned an empty image list.")
        result = result[0]

    if not isinstance(result, Image.Image):
        raise TypeError(
            "InstructPix2Pix did not return a PIL image. "
            f"Received: {type(result).__name__}"
        )

    return result.convert("RGB")


def load_editor(
    lora_path: str | Path | None = None,
) -> StableDiffusionInstructPix2PixPipeline:
    resolved_lora_path = _resolve_lora_path(lora_path)
    cache_key = str(resolved_lora_path) if resolved_lora_path else "__base__"

    if cache_key in _PIPELINE_CACHE:
        return _PIPELINE_CACHE[cache_key]

    pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
        INSTRUCT_PIX2PIX_MODEL_ID,
        torch_dtype=_get_dtype(),
        safety_checker=None,
        requires_safety_checker=False,
    )

    pipe.set_progress_bar_config(disable=False)

    device = _get_device()
    pipe = pipe.to(device)

    if device == "cuda":
        try:
            pipe.enable_attention_slicing()
        except Exception:
            pass

        try:
            pipe.vae.enable_slicing()
        except Exception:
            pass

    if resolved_lora_path is not None:
        if not resolved_lora_path.exists():
            raise FileNotFoundError(f"LoRA weights not found: {resolved_lora_path}")

        if hasattr(pipe, "load_lora_weights"):
            pipe.load_lora_weights(str(resolved_lora_path))
        else:
            pipe.unet.load_attn_procs(str(resolved_lora_path))

    _PIPELINE_CACHE[cache_key] = pipe
    return pipe


def edit_image_with_prompt(
    image: Image.Image,
    prompt: str,
    negative_prompt: str = (
        "blurry, distorted, messy background, multiple objects, "
        "low quality, duplicate object"
    ),
    num_inference_steps: int = 20,
    guidance_scale: float = 7.5,
    image_guidance_scale: float = 1.5,
    seed: int = DEFAULT_SEED,
    lora_path: str | Path | None = None,
    lora_scale: float = INSTRUCT_PIX2PIX_LORA_SCALE,
) -> Image.Image:
    resolved_lora_path = _resolve_lora_path(lora_path)
    pipe = load_editor(resolved_lora_path)
    source_image = image.convert("RGB")
    generator = _build_generator(seed)
    pipe_kwargs: dict[str, Any] = {}

    if resolved_lora_path is not None:
        pipe_kwargs["cross_attention_kwargs"] = {"scale": float(lora_scale)}

    with torch.inference_mode():
        result = pipe(
            prompt=prompt,
            image=source_image,
            negative_prompt=negative_prompt,
            num_inference_steps=int(num_inference_steps),
            guidance_scale=float(guidance_scale),
            image_guidance_scale=float(image_guidance_scale),
            num_images_per_prompt=1,
            generator=generator,
            **pipe_kwargs,
        )

    edited_image = _extract_single_image(result)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return edited_image

