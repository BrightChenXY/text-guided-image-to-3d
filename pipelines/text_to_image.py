from __future__ import annotations

from typing import Any, Optional

import torch
from PIL import Image
from diffusers import StableDiffusionPipeline

from config import DEFAULT_SEED, PREPROCESS_SIZE, TEXT_TO_IMAGE_MODEL_ID


_PIPELINE: Optional[StableDiffusionPipeline] = None


def _get_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def _get_dtype() -> torch.dtype:
    return torch.float16 if torch.cuda.is_available() else torch.float32


def _build_generator(seed: int) -> torch.Generator:
    if torch.cuda.is_available():
        return torch.Generator(device="cuda").manual_seed(seed)
    return torch.Generator().manual_seed(seed)


def _extract_single_image(result: Any) -> Image.Image:
    if isinstance(result, tuple):
        if not result:
            raise RuntimeError("Text-to-image pipeline returned an empty tuple.")
        result = result[0]

    if hasattr(result, "images"):
        images = getattr(result, "images")
        if isinstance(images, (list, tuple)) and images:
            result = images[0]

    if isinstance(result, (list, tuple)):
        if not result:
            raise RuntimeError("Text-to-image pipeline returned an empty image list.")
        result = result[0]

    if not isinstance(result, Image.Image):
        raise TypeError(
            "Text-to-image pipeline did not return a PIL image. "
            f"Received: {type(result).__name__}"
        )

    return result.convert("RGB")


def load_text_to_image() -> StableDiffusionPipeline:
    global _PIPELINE
    if _PIPELINE is not None:
        return _PIPELINE

    pipe = StableDiffusionPipeline.from_pretrained(
        TEXT_TO_IMAGE_MODEL_ID,
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

    _PIPELINE = pipe
    return _PIPELINE


def generate_image_from_text(
    prompt: str,
    negative_prompt: str = (
        "blurry, cluttered scene, multiple objects, messy background, "
        "low quality, cropped object"
    ),
    num_inference_steps: int = 20,
    guidance_scale: float = 7.5,
    size: int = PREPROCESS_SIZE,
    seed: int = DEFAULT_SEED,
) -> Image.Image:
    pipe = load_text_to_image()
    generator = _build_generator(seed)

    with torch.inference_mode():
        result = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=int(num_inference_steps),
            guidance_scale=float(guidance_scale),
            num_images_per_prompt=1,
            width=int(size),
            height=int(size),
            generator=generator,
        )

    generated_image = _extract_single_image(result)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return generated_image
