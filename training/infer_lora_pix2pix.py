from __future__ import annotations

import argparse
from pathlib import Path

import torch
from PIL import Image
from diffusers import StableDiffusionInstructPix2PixPipeline

from training.dataset import DEFAULT_PROMPT_SUFFIX, build_training_prompt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run InstructPix2Pix inference with optional LoRA weights.",
    )
    parser.add_argument(
        "--model-name-or-path",
        type=str,
        default="timbrooks/instruct-pix2pix",
        help="Base model used for inference.",
    )
    parser.add_argument(
        "--lora-path",
        type=Path,
        required=True,
        help="Directory containing the saved LoRA weights.",
    )
    parser.add_argument(
        "--image",
        type=Path,
        required=True,
        help="Input image path.",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        required=True,
        help="Edit instruction.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Where to save the edited image. Defaults to outputs/edited/<input>_lora.png.",
    )
    parser.add_argument(
        "--prompt-suffix",
        type=str,
        default=DEFAULT_PROMPT_SUFFIX,
        help="Optional suffix appended to every prompt.",
    )
    parser.add_argument(
        "--num-inference-steps",
        type=int,
        default=20,
        help="Number of denoising steps.",
    )
    parser.add_argument(
        "--guidance-scale",
        type=float,
        default=7.5,
        help="Text guidance scale.",
    )
    parser.add_argument(
        "--image-guidance-scale",
        type=float,
        default=1.5,
        help="Image guidance scale.",
    )
    parser.add_argument(
        "--lora-scale",
        type=float,
        default=1.0,
        help="Adapter scale passed through cross_attention_kwargs.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed.",
    )
    return parser.parse_args()


def _get_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def _get_dtype() -> torch.dtype:
    if torch.cuda.is_available():
        return torch.float16
    return torch.float32


def _build_generator(seed: int) -> torch.Generator:
    if torch.cuda.is_available():
        return torch.Generator(device="cuda").manual_seed(seed)
    return torch.Generator().manual_seed(seed)


def _resolve_output_path(image_path: Path, output_path: Path | None) -> Path:
    if output_path is not None:
        return output_path.expanduser().resolve()

    default_dir = Path("outputs") / "edited"
    default_dir.mkdir(parents=True, exist_ok=True)
    return (default_dir / f"{image_path.stem}_lora.png").resolve()


def load_pipeline(
    model_name_or_path: str,
    lora_path: Path,
) -> StableDiffusionInstructPix2PixPipeline:
    pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
        model_name_or_path,
        torch_dtype=_get_dtype(),
        safety_checker=None,
        requires_safety_checker=False,
    )
    pipe = pipe.to(_get_device())
    pipe.set_progress_bar_config(disable=False)

    if torch.cuda.is_available():
        try:
            pipe.enable_attention_slicing()
        except Exception:
            pass

        try:
            pipe.vae.enable_slicing()
        except Exception:
            pass

    if hasattr(pipe, "load_lora_weights"):
        pipe.load_lora_weights(str(lora_path))
    else:
        pipe.unet.load_attn_procs(str(lora_path))

    return pipe


def run_inference(args: argparse.Namespace) -> Path:
    image_path = args.image.expanduser().resolve()
    lora_path = args.lora_path.expanduser().resolve()
    output_path = _resolve_output_path(image_path, args.output)

    if not image_path.exists():
        raise FileNotFoundError(f"Input image not found: {image_path}")
    if not lora_path.exists():
        raise FileNotFoundError(f"LoRA weights not found: {lora_path}")

    prompt = build_training_prompt(args.prompt, args.prompt_suffix)
    generator = _build_generator(args.seed)
    pipe = load_pipeline(args.model_name_or_path, lora_path)

    with Image.open(image_path) as image:
        source_image = image.convert("RGB")

    with torch.inference_mode():
        result = pipe(
            prompt=prompt,
            image=source_image,
            num_inference_steps=int(args.num_inference_steps),
            guidance_scale=float(args.guidance_scale),
            image_guidance_scale=float(args.image_guidance_scale),
            generator=generator,
            cross_attention_kwargs={"scale": float(args.lora_scale)},
        )

    edited_image = result.images[0].convert("RGB")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    edited_image.save(output_path)
    return output_path


def main() -> None:
    args = parse_args()
    output_path = run_inference(args)
    print(f"Saved edited image to: {output_path}")


if __name__ == "__main__":
    main()
