from __future__ import annotations

import argparse
import json
import sys
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    StableDiffusionInstructPix2PixPipeline,
    UNet2DConditionModel,
)
from diffusers.optimization import get_scheduler
from peft import LoraConfig
from PIL import Image
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer

from training.dataset import (
    DEFAULT_PROMPT_SUFFIX,
    BasePix2PixDataset,
    Pix2PixHFDataset,
    Pix2PixJsonlDataset,
    StreamingPix2PixJsonlDataset,
    collate_fn,
)

try:
    from datasets import load_dataset
except ImportError:
    load_dataset = None

try:
    from diffusers.utils import convert_state_dict_to_diffusers
except ImportError:
    from diffusers.utils.state_dict_utils import convert_state_dict_to_diffusers

try:
    from peft.utils import get_peft_model_state_dict
except ImportError:
    from peft.utils.save_and_load import get_peft_model_state_dict

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="LoRA fine-tuning for timbrooks/instruct-pix2pix using local metadata.jsonl or Hugging Face datasets.",
    )
    parser.add_argument(
        "--model-name-or-path",
        type=str,
        default="timbrooks/instruct-pix2pix",
        help="Base InstructPix2Pix model.",
    )
    parser.add_argument(
        "--train-metadata",
        type=Path,
        default=None,
        help="Path to training metadata.jsonl for local mode.",
    )
    parser.add_argument(
        "--val-metadata",
        type=Path,
        default=None,
        help="Optional validation metadata.jsonl for local mode.",
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default=None,
        help="Hugging Face dataset name, for example timbrooks/instructpix2pix-clip-filtered.",
    )
    parser.add_argument(
        "--dataset-config-name",
        type=str,
        default=None,
        help="Optional Hugging Face dataset config name.",
    )
    parser.add_argument(
        "--train-split",
        type=str,
        default="train",
        help="Training split name when using --dataset-name.",
    )
    parser.add_argument(
        "--val-split",
        type=str,
        default=None,
        help="Optional validation split name when using --dataset-name.",
    )
    parser.add_argument(
        "--validation-from-train-ratio",
        type=float,
        default=0.0,
        help=(
            "When using --dataset-name without --val-split, automatically carve out "
            "this fraction of the train split for validation."
        ),
    )
    parser.add_argument(
        "--original-image-column",
        type=str,
        default="original_image",
        help="Column name for the source image in Hugging Face dataset mode.",
    )
    parser.add_argument(
        "--edited-image-column",
        type=str,
        default="edited_image",
        help="Column name for the edited target image in Hugging Face dataset mode.",
    )
    parser.add_argument(
        "--edit-prompt-column",
        type=str,
        default="edit_prompt",
        help="Column name for the edit prompt in Hugging Face dataset mode.",
    )
    parser.add_argument(
        "--train-index-filter-json",
        type=Path,
        default=None,
        help="Optional JSON list of allowed metadata indices for local metadata mode. Use training/data/final_indices.json here.",
    )
    parser.add_argument(
        "--val-index-filter-json",
        type=Path,
        default=None,
        help="Optional JSON list of allowed metadata indices for local validation metadata mode.",
    )
    parser.add_argument(
        "--metadata-index-field",
        type=str,
        default="original_dataset_index",
        help="Metadata field used to match entries from --train-index-filter-json or --val-index-filter-json.",
    )
    parser.add_argument(
        "--max-train-records",
        type=int,
        default=None,
        help="Optionally cap the number of filtered training records.",
    )
    parser.add_argument(
        "--max-val-records",
        type=int,
        default=None,
        help="Optionally cap the number of filtered validation records.",
    )
    parser.add_argument(
        "--stream-train-jsonl",
        action="store_true",
        help="In local metadata mode, rescan metadata.jsonl every epoch and only consume currently available filtered images. Useful for download-while-training workflows.",
    )
    parser.add_argument(
        "--skip-missing-train-images",
        action="store_true",
        help="Skip local training records whose original or edited image files are not available yet.",
    )
    parser.add_argument(
        "--stream-wait-seconds",
        type=float,
        default=15.0,
        help="How long to wait before rescanning metadata when stream mode finds no available training records.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("training/outputs/checkpoints/pix2pix-lora"),
        help="Directory used to store checkpoints and final LoRA weights.",
    )
    parser.add_argument(
        "--sample-dir",
        type=Path,
        default=None,
        help="Directory used to store validation sample images.",
    )
    parser.add_argument(
        "--tensorboard-log-dir",
        type=Path,
        default=None,
        help="TensorBoard log directory. Defaults to <output_dir>/tensorboard.",
    )
    parser.add_argument(
        "--validation-loss-batches",
        type=int,
        default=8,
        help="Number of validation batches used to estimate val/loss for TensorBoard.",
    )
    parser.add_argument(
        "--enable-trellis-rerank",
        action="store_true",
        help="Run black-box TRELLIS evaluation during validation and choose best checkpoints from downstream 3D-friendly score.",
    )
    parser.add_argument(
        "--trellis-eval-samples",
        type=int,
        default=4,
        help="How many validation examples to send through TRELLIS for each rerank pass.",
    )
    parser.add_argument(
        "--trellis-render-size",
        type=int,
        default=256,
        help="Rendered view resolution used by the TRELLIS black-box scorer.",
    )
    parser.add_argument(
        "--trellis-seed",
        type=int,
        default=0,
        help="Seed forwarded to the TRELLIS backend during rerank evaluation.",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=256,
        help="Training resolution.",
    )
    parser.add_argument(
        "--resize-mode",
        type=str,
        choices=("pad", "crop"),
        default="pad",
        help="Image resize strategy before VAE encoding.",
    )
    parser.add_argument(
        "--train-batch-size",
        type=int,
        default=1,
        help="Training batch size per device.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=5e-5,
        help="AdamW learning rate.",
    )
    parser.add_argument(
        "--max-train-steps",
        type=int,
        default=1000,
        help="Total training steps.",
    )
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=4,
        help="Gradient accumulation steps.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Training seed.",
    )
    parser.add_argument(
        "--rank",
        type=int,
        default=8,
        help="LoRA rank.",
    )
    parser.add_argument(
        "--lora-alpha",
        type=int,
        default=16,
        help="LoRA alpha.",
    )
    parser.add_argument(
        "--checkpointing-steps",
        type=int,
        default=500,
        help="Save a checkpoint every N update steps.",
    )
    parser.add_argument(
        "--resume-from-checkpoint",
        type=Path,
        default=None,
        help="Optional checkpoint directory to resume from, for example training/outputs/checkpoints/<run>/checkpoint-000250.",
    )
    parser.add_argument(
        "--validation-steps",
        type=int,
        default=250,
        help="Run validation every N update steps when validation data is provided.",
    )
    parser.add_argument(
        "--num-validation-images",
        type=int,
        default=4,
        help="How many validation rows to render per validation run.",
    )
    parser.add_argument(
        "--validation-num-inference-steps",
        type=int,
        default=20,
        help="Inference steps used during validation renders.",
    )
    parser.add_argument(
        "--validation-guidance-scale",
        type=float,
        default=7.5,
        help="Prompt guidance scale used during validation renders.",
    )
    parser.add_argument(
        "--validation-image-guidance-scale",
        type=float,
        default=1.5,
        help="Image guidance scale used during validation renders.",
    )
    parser.add_argument(
        "--conditioning-dropout-prob",
        type=float,
        default=0.1,
        help="Independent dropout probability for prompt and source image conditioning.",
    )
    parser.add_argument(
        "--lr-scheduler",
        type=str,
        default="constant",
        choices=("constant", "linear", "cosine", "cosine_with_restarts", "polynomial"),
        help="Learning rate scheduler type.",
    )
    parser.add_argument(
        "--lr-warmup-steps",
        type=int,
        default=0,
        help="Learning rate warmup steps.",
    )
    parser.add_argument(
        "--adam-weight-decay",
        type=float,
        default=1e-2,
        help="AdamW weight decay.",
    )
    parser.add_argument(
        "--max-grad-norm",
        type=float,
        default=1.0,
        help="Gradient clipping value.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="Dataloader worker count. Default is 0 for Windows friendliness.",
    )
    parser.add_argument(
        "--prompt-suffix",
        type=str,
        default=DEFAULT_PROMPT_SUFFIX,
        help="Optional suffix appended to every edit prompt.",
    )
    parser.add_argument(
        "--mixed-precision",
        type=str,
        default="auto",
        choices=("auto", "no", "fp16", "bf16"),
        help="Accelerate mixed precision mode.",
    )
    parser.add_argument(
        "--allow-tf32",
        action="store_true",
        help="Enable TF32 on Ampere+ GPUs.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> str:
    use_hf_dataset = bool(args.dataset_name)

    if args.validation_from_train_ratio < 0 or args.validation_from_train_ratio >= 1:
        raise ValueError("--validation-from-train-ratio must be in the range [0, 1).")
    if args.trellis_eval_samples < 1:
        raise ValueError("--trellis-eval-samples must be >= 1.")
    if args.trellis_render_size < 64:
        raise ValueError("--trellis-render-size must be >= 64.")

    if use_hf_dataset:
        if args.train_metadata is not None or args.val_metadata is not None:
            raise ValueError(
                "dataset-name mode cannot be combined with train-metadata / val-metadata."
            )
        if args.train_index_filter_json is not None or args.val_index_filter_json is not None:
            raise ValueError(
                "Index-filter JSON files are only supported in local metadata mode."
            )
        if args.stream_train_jsonl:
            raise ValueError("--stream-train-jsonl can only be used together with --train-metadata.")
        if load_dataset is None:
            raise ImportError(
                "datasets is not installed. Please add it to your environment before using --dataset-name."
            )
        if args.val_split is not None and args.validation_from_train_ratio > 0:
            raise ValueError(
                "Use either --val-split or --validation-from-train-ratio, not both."
            )
        return "hf"

    if args.train_metadata is None:
        raise ValueError(
            "Local mode requires --train-metadata. Alternatively pass --dataset-name to use Hugging Face datasets."
        )

    if args.val_split is not None:
        raise ValueError("--val-split can only be used together with --dataset-name.")
    if args.validation_from_train_ratio > 0:
        raise ValueError(
            "--validation-from-train-ratio can only be used together with --dataset-name."
        )
    if args.max_train_records is not None and args.max_train_records < 1:
        raise ValueError("--max-train-records must be >= 1 when provided.")
    if args.max_val_records is not None and args.max_val_records < 1:
        raise ValueError("--max-val-records must be >= 1 when provided.")
    if args.stream_wait_seconds < 0:
        raise ValueError("--stream-wait-seconds must be >= 0.")

    return "local"

def resolve_mixed_precision(value: str) -> str:
    if value != "auto":
        return value
    if torch.cuda.is_available():
        return "fp16"
    return "no"


def get_weight_dtype(accelerator: Accelerator) -> torch.dtype:
    if accelerator.device.type != "cuda":
        return torch.float32
    if accelerator.mixed_precision == "fp16":
        return torch.float16
    if accelerator.mixed_precision == "bf16":
        return torch.bfloat16
    return torch.float32


def build_seeded_generator(device_type: str, seed: int) -> torch.Generator:
    if device_type == "cuda":
        return torch.Generator(device="cuda").manual_seed(seed)
    return torch.Generator().manual_seed(seed)


def get_runtime_device_summary(accelerator: Accelerator, weight_dtype: torch.dtype) -> dict[str, Any]:
    cuda_available = torch.cuda.is_available()
    device_count = torch.cuda.device_count() if cuda_available else 0
    if cuda_available and accelerator.device.type == "cuda":
        gpu_name = torch.cuda.get_device_name(accelerator.device)
    elif cuda_available and device_count > 0:
        gpu_name = torch.cuda.get_device_name(0)
    else:
        gpu_name = None

    return {
        "accelerator_device": str(accelerator.device),
        "mixed_precision": accelerator.mixed_precision,
        "weight_dtype": str(weight_dtype).replace("torch.", ""),
        "torch_cuda_available": cuda_available,
        "torch_cuda_version": torch.version.cuda,
        "torch_device_count": device_count,
        "gpu_name": gpu_name,
    }


def emit_runtime_device_report(
    accelerator: Accelerator,
    runtime_device: dict[str, Any],
    logger: Any,
) -> None:
    device_message = (
        "Runtime device: accelerator={accelerator_device} | cuda_available={torch_cuda_available} | "
        "cuda_version={torch_cuda_version} | gpu_count={torch_device_count} | gpu={gpu_name} | "
        "mixed_precision={mixed_precision} | weight_dtype={weight_dtype}"
    ).format(**runtime_device)
    logger.info(device_message)
    accelerator.print(device_message)
    print(device_message, flush=True)


def cast_trainable_params_to_float32(model: torch.nn.Module) -> None:
    for parameter in model.parameters():
        if parameter.requires_grad:
            parameter.data = parameter.data.to(torch.float32)


def apply_conditioning_dropout(
    input_ids: torch.Tensor,
    empty_prompt_ids: torch.Tensor,
    dropout_prob: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    prompt_ids = input_ids.clone()
    batch_size = prompt_ids.shape[0]
    device = prompt_ids.device
    image_drop_mask = torch.zeros(batch_size, dtype=torch.bool, device=device)

    if dropout_prob <= 0:
        return prompt_ids, image_drop_mask

    prompt_drop_mask = torch.rand(batch_size, device=device) < dropout_prob
    image_drop_mask = torch.rand(batch_size, device=device) < dropout_prob

    if prompt_drop_mask.any():
        replacement = empty_prompt_ids.unsqueeze(0).expand(
            int(prompt_drop_mask.sum().item()),
            -1,
        )
        prompt_ids[prompt_drop_mask] = replacement

    return prompt_ids, image_drop_mask


def compute_loss_target(
    noise_scheduler: DDPMScheduler,
    latents: torch.Tensor,
    noise: torch.Tensor,
    timesteps: torch.Tensor,
) -> torch.Tensor:
    if noise_scheduler.config.prediction_type == "epsilon":
        return noise
    if noise_scheduler.config.prediction_type == "v_prediction":
        return noise_scheduler.get_velocity(latents, noise, timesteps)
    raise ValueError(
        f"Unsupported prediction type: {noise_scheduler.config.prediction_type}"
    )


def compute_batch_loss(
    batch: dict[str, torch.Tensor],
    unet: UNet2DConditionModel,
    vae: AutoencoderKL,
    text_encoder: CLIPTextModel,
    noise_scheduler: DDPMScheduler,
    accelerator: Accelerator,
    weight_dtype: torch.dtype,
    empty_prompt_ids: torch.Tensor,
    conditioning_dropout_prob: float,
    generator: torch.Generator | None = None,
    requires_grad: bool = True,
) -> torch.Tensor:
    original_pixel_values = batch["original_pixel_values"].to(
        accelerator.device,
        dtype=weight_dtype,
    )
    edited_pixel_values = batch["edited_pixel_values"].to(
        accelerator.device,
        dtype=weight_dtype,
    )
    input_ids = batch["input_ids"].to(accelerator.device)
    prompt_ids, image_drop_mask = apply_conditioning_dropout(
        input_ids=input_ids,
        empty_prompt_ids=empty_prompt_ids,
        dropout_prob=conditioning_dropout_prob,
    )

    with torch.no_grad():
        with accelerator.autocast():
            encoder_hidden_states = text_encoder(
                prompt_ids,
                return_dict=False,
            )[0]
            original_latents = vae.encode(original_pixel_values).latent_dist.sample()
            edited_latents = vae.encode(edited_pixel_values).latent_dist.sample()

        original_latents = original_latents * vae.config.scaling_factor
        edited_latents = edited_latents * vae.config.scaling_factor

        if image_drop_mask.any():
            original_latents[image_drop_mask] = 0

        if generator is None:
            noise = torch.randn_like(edited_latents)
            timesteps = torch.randint(
                0,
                noise_scheduler.config.num_train_timesteps,
                (edited_latents.shape[0],),
                device=edited_latents.device,
            ).long()
        else:
            noise = torch.randn(
                edited_latents.shape,
                device=edited_latents.device,
                dtype=edited_latents.dtype,
                generator=generator,
            )
            timesteps = torch.randint(
                0,
                noise_scheduler.config.num_train_timesteps,
                (edited_latents.shape[0],),
                device=edited_latents.device,
                generator=generator,
            ).long()

        noisy_latents = noise_scheduler.add_noise(
            edited_latents,
            noise,
            timesteps,
        )
        model_input = torch.cat([noisy_latents, original_latents], dim=1)
        target = compute_loss_target(
            noise_scheduler,
            edited_latents,
            noise,
            timesteps,
        )

    grad_context = nullcontext() if requires_grad else torch.no_grad()
    with grad_context:
        with accelerator.autocast():
            model_pred = unet(
                model_input,
                timesteps,
                encoder_hidden_states=encoder_hidden_states,
                return_dict=False,
            )[0]
            loss = F.mse_loss(
                model_pred.float(),
                target.float(),
                reduction="mean",
            )

    return loss


def extract_lora_state_dict(unet: UNet2DConditionModel) -> dict[str, torch.Tensor]:
    state_dict = get_peft_model_state_dict(unet)
    return convert_state_dict_to_diffusers(state_dict)


def save_lora_weights(output_dir: Path, unet: UNet2DConditionModel) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    state_dict = extract_lora_state_dict(unet)

    if hasattr(StableDiffusionInstructPix2PixPipeline, "save_lora_weights"):
        StableDiffusionInstructPix2PixPipeline.save_lora_weights(
            save_directory=str(output_dir),
            unet_lora_layers=state_dict,
        )
        return

    from diffusers.loaders import LoraLoaderMixin

    LoraLoaderMixin.save_lora_weights(
        save_directory=str(output_dir),
        unet_lora_layers=state_dict,
    )


def save_checkpoint(
    accelerator: Accelerator,
    unet: UNet2DConditionModel,
    checkpoint_dir: Path,
) -> None:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    accelerator.save_state(str(checkpoint_dir / "accelerate_state"))
    save_lora_weights(checkpoint_dir / "lora", accelerator.unwrap_model(unet))


def save_best_checkpoint(
    accelerator: Accelerator,
    unet: UNet2DConditionModel,
    output_dir: Path,
    metadata: dict[str, Any],
) -> Path:
    best_dir = output_dir / "best_checkpoint"
    lora_dir = best_dir / "lora"
    save_lora_weights(lora_dir, accelerator.unwrap_model(unet))
    with (best_dir / "best_checkpoint.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, ensure_ascii=False)
    return best_dir


def resolve_resume_checkpoint_path(checkpoint_path: Path | None) -> Path | None:
    if checkpoint_path is None:
        return None

    resolved = checkpoint_path.expanduser().resolve()
    state_dir = resolved / "accelerate_state"
    if not resolved.exists() or not state_dir.exists():
        raise FileNotFoundError(
            f"Resume checkpoint is missing accelerate_state: {resolved}"
        )
    return resolved


def extract_resume_global_step(checkpoint_dir: Path) -> int:
    prefix = "checkpoint-"
    name = checkpoint_dir.name
    if not name.startswith(prefix):
        raise ValueError(
            f"Checkpoint directory name must start with '{prefix}': {checkpoint_dir}"
        )

    step_text = name[len(prefix):]
    if not step_text.isdigit():
        raise ValueError(
            f"Could not parse global step from checkpoint directory: {checkpoint_dir}"
        )
    return int(step_text)


def write_training_config(args: argparse.Namespace, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    with (output_dir / "training_args.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def make_validation_strip(
    source_image: Image.Image,
    generated_image: Image.Image,
    target_image: Image.Image,
) -> Image.Image:
    width, height = generated_image.size
    canvas = Image.new("RGB", (width * 3, height), color=(255, 255, 255))
    canvas.paste(source_image.resize((width, height)), (0, 0))
    canvas.paste(generated_image.resize((width, height)), (width, 0))
    canvas.paste(target_image.resize((width, height)), (width * 2, 0))
    return canvas


def log_preview_to_tensorboard(
    writer: SummaryWriter | None,
    tag: str,
    preview: Image.Image,
    global_step: int,
) -> None:
    if writer is None:
        return

    image_array = np.array(preview, copy=True)
    image_tensor = torch.from_numpy(image_array).permute(2, 0, 1)
    writer.add_image(tag, image_tensor, global_step)


def build_inference_pipeline(
    accelerator: Accelerator,
    args: argparse.Namespace,
    unet: UNet2DConditionModel,
    vae: AutoencoderKL,
    text_encoder: CLIPTextModel,
    weight_dtype: torch.dtype,
) -> StableDiffusionInstructPix2PixPipeline:
    pipeline = StableDiffusionInstructPix2PixPipeline.from_pretrained(
        args.model_name_or_path,
        unet=accelerator.unwrap_model(unet),
        vae=vae,
        text_encoder=text_encoder,
        torch_dtype=weight_dtype,
        safety_checker=None,
        requires_safety_checker=False,
    )
    pipeline = pipeline.to(accelerator.device)
    pipeline.set_progress_bar_config(disable=True)

    if accelerator.device.type == "cuda":
        try:
            pipeline.enable_attention_slicing()
        except Exception:
            pass

        try:
            pipeline.vae.enable_slicing()
        except Exception:
            pass

    return pipeline

def compute_validation_loss(
    accelerator: Accelerator,
    args: argparse.Namespace,
    unet: UNet2DConditionModel,
    vae: AutoencoderKL,
    text_encoder: CLIPTextModel,
    noise_scheduler: DDPMScheduler,
    val_dataloader: DataLoader | None,
    weight_dtype: torch.dtype,
    empty_prompt_ids: torch.Tensor,
) -> float | None:
    if not accelerator.is_main_process or val_dataloader is None:
        return None

    validation_model = accelerator.unwrap_model(unet)
    was_training = validation_model.training
    validation_model.eval()

    total_loss = 0.0
    total_batches = 0
    generator = build_seeded_generator(accelerator.device.type, args.seed)

    for batch_index, batch in enumerate(val_dataloader):
        if batch_index >= args.validation_loss_batches:
            break

        loss = compute_batch_loss(
            batch=batch,
            unet=validation_model,
            vae=vae,
            text_encoder=text_encoder,
            noise_scheduler=noise_scheduler,
            accelerator=accelerator,
            weight_dtype=weight_dtype,
            empty_prompt_ids=empty_prompt_ids,
            conditioning_dropout_prob=0.0,
            generator=generator,
            requires_grad=False,
        )
        total_loss += float(loss.detach().item())
        total_batches += 1

    if was_training:
        validation_model.train()

    if total_batches == 0:
        return None

    return total_loss / total_batches


def run_validation(
    accelerator: Accelerator,
    args: argparse.Namespace,
    unet: UNet2DConditionModel,
    vae: AutoencoderKL,
    text_encoder: CLIPTextModel,
    val_dataset: BasePix2PixDataset,
    sample_dir: Path,
    global_step: int,
    weight_dtype: torch.dtype,
    tensorboard_writer: SummaryWriter | None = None,
) -> None:
    if not accelerator.is_main_process or len(val_dataset) == 0:
        return

    pipeline = build_inference_pipeline(
        accelerator=accelerator,
        args=args,
        unet=unet,
        vae=vae,
        text_encoder=text_encoder,
        weight_dtype=weight_dtype,
    )

    render_dir = sample_dir / f"step_{global_step:06d}"
    render_dir.mkdir(parents=True, exist_ok=True)
    generator = build_seeded_generator(accelerator.device.type, args.seed)

    try:
        for index in range(min(args.num_validation_images, len(val_dataset))):
            example = val_dataset.get_visual_example(index)

            with torch.inference_mode():
                if accelerator.device.type == "cuda":
                    with torch.autocast("cuda", dtype=weight_dtype):
                        result = pipeline(
                            prompt=example["prompt"],
                            image=example["original_image"],
                            num_inference_steps=int(args.validation_num_inference_steps),
                            guidance_scale=float(args.validation_guidance_scale),
                            image_guidance_scale=float(args.validation_image_guidance_scale),
                            generator=generator,
                        )
                else:
                    result = pipeline(
                        prompt=example["prompt"],
                        image=example["original_image"],
                        num_inference_steps=int(args.validation_num_inference_steps),
                        guidance_scale=float(args.validation_guidance_scale),
                        image_guidance_scale=float(args.validation_image_guidance_scale),
                        generator=generator,
                    )

            generated_image = result.images[0].convert("RGB")
            preview = make_validation_strip(
                example["original_image"],
                generated_image,
                example["edited_image"],
            )
            preview.save(render_dir / f"sample_{index:02d}.png")
            log_preview_to_tensorboard(
                writer=tensorboard_writer,
                tag=f"val/previews/sample_{index:02d}",
                preview=preview,
                global_step=global_step,
            )
    finally:
        del pipeline
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

def run_trellis_rerank(
    accelerator: Accelerator,
    args: argparse.Namespace,
    unet: UNet2DConditionModel,
    vae: AutoencoderKL,
    text_encoder: CLIPTextModel,
    val_dataset: BasePix2PixDataset,
    global_step: int,
    weight_dtype: torch.dtype,
    tensorboard_writer: SummaryWriter | None = None,
) -> dict[str, Any] | None:
    if not accelerator.is_main_process or len(val_dataset) == 0:
        return None

    from training.trellis_eval import (
        evaluate_edited_image_with_trellis,
        trellis_rerank_dependencies_available,
    )

    if not trellis_rerank_dependencies_available():
        raise ImportError(
            "TRELLIS rerank requires trimesh and pyrender. Install them before using --enable-trellis-rerank."
        )

    pipeline = build_inference_pipeline(
        accelerator=accelerator,
        args=args,
        unet=unet,
        vae=vae,
        text_encoder=text_encoder,
        weight_dtype=weight_dtype,
    )

    step_dir = args.output_dir / "trellis_eval" / f"step_{global_step:06d}"
    step_dir.mkdir(parents=True, exist_ok=True)
    generator = build_seeded_generator(accelerator.device.type, args.seed)
    sample_payloads: list[dict[str, Any]] = []
    scores: list[float] = []
    front_similarities: list[float] = []
    coverage_scores: list[float] = []
    centering_scores: list[float] = []
    consistency_scores: list[float] = []
    successful_samples = 0
    num_samples = min(args.trellis_eval_samples, len(val_dataset))

    try:
        for index in range(num_samples):
            example = val_dataset.get_visual_example(index)
            sample_dir = step_dir / f"sample_{index:02d}"
            sample_dir.mkdir(parents=True, exist_ok=True)

            with torch.inference_mode():
                if accelerator.device.type == "cuda":
                    with torch.autocast("cuda", dtype=weight_dtype):
                        result = pipeline(
                            prompt=example["prompt"],
                            image=example["original_image"],
                            num_inference_steps=int(args.validation_num_inference_steps),
                            guidance_scale=float(args.validation_guidance_scale),
                            image_guidance_scale=float(args.validation_image_guidance_scale),
                            generator=generator,
                        )
                else:
                    result = pipeline(
                        prompt=example["prompt"],
                        image=example["original_image"],
                        num_inference_steps=int(args.validation_num_inference_steps),
                        guidance_scale=float(args.validation_guidance_scale),
                        image_guidance_scale=float(args.validation_image_guidance_scale),
                        generator=generator,
                    )

            generated_image = result.images[0].convert("RGB")
            generated_image.save(sample_dir / "generated.png")
            example["original_image"].convert("RGB").save(sample_dir / "original.png")
            example["edited_image"].convert("RGB").save(sample_dir / "target.png")

            trellis_result = evaluate_edited_image_with_trellis(
                edited_image=generated_image,
                work_dir=sample_dir,
                seed=args.trellis_seed,
                render_size=args.trellis_render_size,
            )

            sample_score = float(trellis_result.get("score", 0.0) or 0.0)
            metrics = trellis_result.get("metrics", {})
            scores.append(sample_score)
            front_similarities.append(float(metrics.get("front_similarity", 0.0) or 0.0))
            coverage_scores.append(float(metrics.get("mean_coverage_score", 0.0) or 0.0))
            centering_scores.append(float(metrics.get("mean_centering_score", 0.0) or 0.0))
            consistency_scores.append(float(metrics.get("view_consistency_score", 0.0) or 0.0))

            if trellis_result.get("success"):
                successful_samples += 1

            front_render_path = trellis_result.get("render_paths", {}).get("front")
            if front_render_path:
                front_render = Image.open(front_render_path).convert("RGB")
                preview = make_validation_strip(
                    example["original_image"],
                    generated_image,
                    front_render,
                )
                preview.save(sample_dir / "trellis_preview.png")
                log_preview_to_tensorboard(
                    writer=tensorboard_writer,
                    tag=f"trellis/previews/sample_{index:02d}",
                    preview=preview,
                    global_step=global_step,
                )

            sample_payloads.append(
                {
                    "index": index,
                    "prompt": example["prompt"],
                    "score": sample_score,
                    "success": bool(trellis_result.get("success", False)),
                    "message": trellis_result.get("message", ""),
                    "glb_path": trellis_result.get("glb_path"),
                    "metrics": metrics,
                    "render_paths": trellis_result.get("render_paths", {}),
                }
            )
    finally:
        del pipeline
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    summary = {
        "step": global_step,
        "evaluated_samples": num_samples,
        "successful_samples": successful_samples,
        "success_rate": float(successful_samples / num_samples) if num_samples else 0.0,
        "mean_score": float(np.mean(scores)) if scores else 0.0,
        "mean_front_similarity": float(np.mean(front_similarities)) if front_similarities else 0.0,
        "mean_coverage_score": float(np.mean(coverage_scores)) if coverage_scores else 0.0,
        "mean_centering_score": float(np.mean(centering_scores)) if centering_scores else 0.0,
        "mean_view_consistency_score": float(np.mean(consistency_scores)) if consistency_scores else 0.0,
        "samples": sample_payloads,
    }

    with (step_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)

    if tensorboard_writer is not None:
        tensorboard_writer.add_scalar("trellis/mean_score", summary["mean_score"], global_step)
        tensorboard_writer.add_scalar("trellis/success_rate", summary["success_rate"], global_step)
        tensorboard_writer.add_scalar(
            "trellis/front_similarity", summary["mean_front_similarity"], global_step
        )
        tensorboard_writer.add_scalar(
            "trellis/coverage_score", summary["mean_coverage_score"], global_step
        )
        tensorboard_writer.add_scalar(
            "trellis/centering_score", summary["mean_centering_score"], global_step
        )
        tensorboard_writer.add_scalar(
            "trellis/view_consistency_score",
            summary["mean_view_consistency_score"],
            global_step,
        )

    logger.info(
        "TRELLIS rerank step %s: mean_score=%.4f, success_rate=%.2f%% over %s samples",
        global_step,
        summary["mean_score"],
        summary["success_rate"] * 100.0,
        num_samples,
    )
    return summary

def run_full_validation_cycle(
    accelerator: Accelerator,
    args: argparse.Namespace,
    unet: UNet2DConditionModel,
    vae: AutoencoderKL,
    text_encoder: CLIPTextModel,
    noise_scheduler: DDPMScheduler,
    val_dataset: BasePix2PixDataset | None,
    val_dataloader: DataLoader | None,
    sample_dir: Path,
    global_step: int,
    weight_dtype: torch.dtype,
    empty_prompt_ids: torch.Tensor,
    tensorboard_writer: SummaryWriter | None = None,
) -> dict[str, Any]:
    metrics: dict[str, Any] = {"val_loss": None, "trellis_summary": None}
    if val_dataset is None:
        return metrics

    val_loss = compute_validation_loss(
        accelerator=accelerator,
        args=args,
        unet=unet,
        vae=vae,
        text_encoder=text_encoder,
        noise_scheduler=noise_scheduler,
        val_dataloader=val_dataloader,
        weight_dtype=weight_dtype,
        empty_prompt_ids=empty_prompt_ids,
    )
    metrics["val_loss"] = val_loss
    if tensorboard_writer is not None and val_loss is not None:
        tensorboard_writer.add_scalar("val/loss", val_loss, global_step)

    run_validation(
        accelerator=accelerator,
        args=args,
        unet=unet,
        vae=vae,
        text_encoder=text_encoder,
        val_dataset=val_dataset,
        sample_dir=sample_dir,
        global_step=global_step,
        weight_dtype=weight_dtype,
        tensorboard_writer=tensorboard_writer,
    )

    if args.enable_trellis_rerank:
        metrics["trellis_summary"] = run_trellis_rerank(
            accelerator=accelerator,
            args=args,
            unet=unet,
            vae=vae,
            text_encoder=text_encoder,
            val_dataset=val_dataset,
            global_step=global_step,
            weight_dtype=weight_dtype,
            tensorboard_writer=tensorboard_writer,
        )

    if tensorboard_writer is not None:
        tensorboard_writer.flush()

    return metrics


def choose_best_metric(
    args: argparse.Namespace,
    validation_metrics: dict[str, Any],
) -> tuple[str, float, bool] | None:
    trellis_summary = validation_metrics.get("trellis_summary")
    if args.enable_trellis_rerank and trellis_summary is not None:
        return "trellis/mean_score", float(trellis_summary["mean_score"]), True

    val_loss = validation_metrics.get("val_loss")
    if val_loss is not None:
        return "val/loss", float(val_loss), False

    return None


def is_metric_improved(
    metric_value: float,
    best_metric_value: float | None,
    maximize: bool,
) -> bool:
    if best_metric_value is None:
        return True
    if maximize:
        return metric_value > best_metric_value
    return metric_value < best_metric_value


def build_datasets(
    args: argparse.Namespace,
    tokenizer: CLIPTokenizer,
    dataset_mode: str,
) -> tuple[BasePix2PixDataset, BasePix2PixDataset | None]:
    common_kwargs: dict[str, Any] = {
        "resolution": args.resolution,
        "prompt_suffix": args.prompt_suffix,
        "resize_mode": args.resize_mode,
    }

    if dataset_mode == "hf":
        train_source = load_dataset(
            path=args.dataset_name,
            name=args.dataset_config_name,
            split=args.train_split,
        )

        if args.val_split:
            val_source = load_dataset(
                path=args.dataset_name,
                name=args.dataset_config_name,
                split=args.val_split,
            )
        elif args.validation_from_train_ratio > 0:
            split_dataset = train_source.train_test_split(
                test_size=args.validation_from_train_ratio,
                seed=args.seed,
                shuffle=True,
            )
            train_source = split_dataset["train"]
            val_source = split_dataset["test"]
        else:
            val_source = None

        train_dataset = Pix2PixHFDataset(
            dataset=train_source,
            original_image_column=args.original_image_column,
            edited_image_column=args.edited_image_column,
            edit_prompt_column=args.edit_prompt_column,
            tokenizer=tokenizer,
            **common_kwargs,
        )
        val_dataset = (
            Pix2PixHFDataset(
                dataset=val_source,
                original_image_column=args.original_image_column,
                edited_image_column=args.edited_image_column,
                edit_prompt_column=args.edit_prompt_column,
                tokenizer=tokenizer,
                **common_kwargs,
            )
            if val_source is not None
            else None
        )
        return train_dataset, val_dataset

    auto_skip_missing_train = args.skip_missing_train_images or (
        args.train_metadata is not None and "subset" in args.train_metadata.name.lower()
    )

    if args.stream_train_jsonl:
        train_dataset = StreamingPix2PixJsonlDataset(
            metadata_path=args.train_metadata,
            tokenizer=tokenizer,
            index_filter_json=args.train_index_filter_json,
            index_field=args.metadata_index_field,
            max_records=args.max_train_records,
            skip_missing_images=True,
            seed=args.seed,
            **common_kwargs,
        )
    else:
        train_dataset = Pix2PixJsonlDataset(
            metadata_path=args.train_metadata,
            tokenizer=tokenizer,
            index_filter_json=args.train_index_filter_json,
            index_field=args.metadata_index_field,
            max_records=args.max_train_records,
            skip_missing_images=auto_skip_missing_train,
            **common_kwargs,
        )

    val_dataset = (
        Pix2PixJsonlDataset(
            metadata_path=args.val_metadata,
            tokenizer=tokenizer,
            index_filter_json=args.val_index_filter_json,
            index_field=args.metadata_index_field,
            max_records=args.max_val_records,
            skip_missing_images=True,
            **common_kwargs,
        )
        if args.val_metadata is not None
        else None
    )
    return train_dataset, val_dataset

def main() -> None:
    args = parse_args()
    dataset_mode = validate_args(args)

    args.output_dir = args.output_dir.expanduser().resolve()
    args.train_metadata = (
        args.train_metadata.expanduser().resolve()
        if args.train_metadata is not None
        else None
    )
    args.val_metadata = (
        args.val_metadata.expanduser().resolve()
        if args.val_metadata is not None
        else None
    )
    args.train_index_filter_json = (
        args.train_index_filter_json.expanduser().resolve()
        if args.train_index_filter_json is not None
        else None
    )
    args.val_index_filter_json = (
        args.val_index_filter_json.expanduser().resolve()
        if args.val_index_filter_json is not None
        else None
    )
    args.resume_from_checkpoint = resolve_resume_checkpoint_path(
        args.resume_from_checkpoint
    )
    if dataset_mode == "local" and args.train_index_filter_json is None and args.train_metadata is not None:
        default_train_filter = args.train_metadata.parent / "final_indices.json"
        if default_train_filter.exists():
            args.train_index_filter_json = default_train_filter.resolve()
    args.sample_dir = (
        args.sample_dir.expanduser().resolve()
        if args.sample_dir is not None
        else (args.output_dir.parents[1] / "samples" / args.output_dir.name).resolve()
    )
    args.tensorboard_log_dir = (
        args.tensorboard_log_dir.expanduser().resolve()
        if args.tensorboard_log_dir is not None
        else (args.output_dir / "tensorboard").resolve()
    )
    args.mixed_precision = resolve_mixed_precision(args.mixed_precision)

    if args.allow_tf32 and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True

    project_config = ProjectConfiguration(
        project_dir=str(args.output_dir),
        logging_dir=str(args.output_dir / "logs"),
    )
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        project_config=project_config,
    )
    weight_dtype = get_weight_dtype(accelerator)
    runtime_device = get_runtime_device_summary(
        accelerator=accelerator,
        weight_dtype=weight_dtype,
    )
    if accelerator.is_main_process:
        emit_runtime_device_report(
            accelerator=accelerator,
            runtime_device=runtime_device,
            logger=logger,
        )
    tensorboard_writer = None

    if accelerator.is_main_process:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        args.sample_dir.mkdir(parents=True, exist_ok=True)
        args.tensorboard_log_dir.mkdir(parents=True, exist_ok=True)
        write_training_config(args, args.output_dir)
        tensorboard_writer = SummaryWriter(log_dir=str(args.tensorboard_log_dir))

    accelerator.wait_for_everyone()
    set_seed(args.seed)

    tokenizer = CLIPTokenizer.from_pretrained(
        args.model_name_or_path,
        subfolder="tokenizer",
    )
    noise_scheduler = DDPMScheduler.from_pretrained(
        args.model_name_or_path,
        subfolder="scheduler",
    )
    text_encoder = CLIPTextModel.from_pretrained(
        args.model_name_or_path,
        subfolder="text_encoder",
    )
    vae = AutoencoderKL.from_pretrained(
        args.model_name_or_path,
        subfolder="vae",
    )
    unet = UNet2DConditionModel.from_pretrained(
        args.model_name_or_path,
        subfolder="unet",
    )

    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    unet.requires_grad_(False)

    lora_config = LoraConfig(
        r=args.rank,
        lora_alpha=args.lora_alpha,
        init_lora_weights="gaussian",
        target_modules=["to_q", "to_k", "to_v", "to_out.0"],
    )
    unet.add_adapter(lora_config, adapter_name="default")
    cast_trainable_params_to_float32(unet)
    unet.train()
    text_encoder.eval()
    vae.eval()

    train_dataset, val_dataset = build_datasets(
        args=args,
        tokenizer=tokenizer,
        dataset_mode=dataset_mode,
    )
    if args.enable_trellis_rerank and val_dataset is None:
        raise ValueError(
            "--enable-trellis-rerank requires validation data. Pass --val-metadata, --val-split, or --validation-from-train-ratio."
        )

    is_streaming_train = isinstance(train_dataset, StreamingPix2PixJsonlDataset)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=not is_streaming_train,
        num_workers=0 if is_streaming_train else args.num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_fn,
    )
    val_dataloader = (
        DataLoader(
            val_dataset,
            batch_size=args.train_batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=torch.cuda.is_available(),
            collate_fn=collate_fn,
        )
        if val_dataset is not None
        else None
    )

    optimizer = AdamW(
        [parameter for parameter in unet.parameters() if parameter.requires_grad],
        lr=args.learning_rate,
        betas=(0.9, 0.999),
        weight_decay=args.adam_weight_decay,
        eps=1e-8,
    )
    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps,
        num_training_steps=args.max_train_steps,
    )

    unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        unet,
        optimizer,
        train_dataloader,
        lr_scheduler,
    )

    resumed_global_step = 0
    if args.resume_from_checkpoint is not None:
        accelerator.load_state(str(args.resume_from_checkpoint / "accelerate_state"))
        resumed_global_step = extract_resume_global_step(args.resume_from_checkpoint)

    vae.to(accelerator.device, dtype=weight_dtype)
    text_encoder.to(accelerator.device, dtype=weight_dtype)

    empty_prompt_ids = tokenizer(
        "",
        max_length=tokenizer.model_max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    ).input_ids[0].to(accelerator.device)

    if accelerator.is_main_process:
        if dataset_mode == "hf":
            logger.info("Starting training from Hugging Face dataset: %s", args.dataset_name)
            logger.info("Train split: %s", args.train_split)
            if args.val_split is not None:
                logger.info("Validation split: %s", args.val_split)
            elif args.validation_from_train_ratio > 0:
                logger.info(
                    "Validation is auto-split from train with ratio: %.4f",
                    args.validation_from_train_ratio,
                )
        else:
            logger.info("Starting training from local metadata: %s", args.train_metadata)
            if args.val_metadata is not None:
                logger.info("Validation metadata: %s", args.val_metadata)

        if is_streaming_train:
            logger.info(
                "Train records currently available: %s (streaming local metadata mode)",
                train_dataset.available_record_count(),
            )
        else:
            logger.info("Train records available now: %s", len(train_dataset))
        if val_dataset is not None:
            logger.info("Validation records available now: %s", len(val_dataset))
        if dataset_mode == "local" and args.train_index_filter_json is not None:
            logger.info(
                "Training metadata filter: %s via field '%s'",
                args.train_index_filter_json,
                args.metadata_index_field,
            )
        if dataset_mode == "local" and args.max_train_records is not None:
            logger.info("Training record cap: %s", args.max_train_records)
        logger.info("TensorBoard logs: %s", args.tensorboard_log_dir)
        if args.resume_from_checkpoint is not None:
            logger.info(
                "Resuming from checkpoint: %s (global_step=%s)",
                args.resume_from_checkpoint,
                resumed_global_step,
            )
        if torch.cuda.is_available() and accelerator.device.type == "cuda":
            memory_message = (
                f"CUDA memory after model load: allocated={torch.cuda.memory_allocated(accelerator.device) / (1024 ** 2):.1f} MiB | "
                f"reserved={torch.cuda.memory_reserved(accelerator.device) / (1024 ** 2):.1f} MiB"
            )
            logger.info(memory_message)
            accelerator.print(memory_message)
            print(memory_message, flush=True)
        logger.info(
            "This diffusion training does not produce a meaningful accuracy curve. Use train/loss, val/loss, val/previews, and TRELLIS metrics in TensorBoard instead."
        )
        if args.enable_trellis_rerank:
            logger.info(
                "TRELLIS rerank is enabled. Best checkpoint will be chosen by trellis/mean_score and written to %s",
                args.output_dir / "best_checkpoint",
            )

    progress_bar = tqdm(
        range(args.max_train_steps),
        disable=not accelerator.is_local_main_process,
        desc="Training",
        initial=resumed_global_step,
    )

    global_step = resumed_global_step
    best_metric_name: str | None = None
    best_metric_value: float | None = None
    best_metric_maximize = False

    while global_step < args.max_train_steps:
        made_progress = False
        for batch in train_dataloader:
            made_progress = True
            with accelerator.accumulate(unet):
                loss = compute_batch_loss(
                    batch=batch,
                    unet=unet,
                    vae=vae,
                    text_encoder=text_encoder,
                    noise_scheduler=noise_scheduler,
                    accelerator=accelerator,
                    weight_dtype=weight_dtype,
                    empty_prompt_ids=empty_prompt_ids,
                    conditioning_dropout_prob=args.conditioning_dropout_prob,
                    generator=None,
                    requires_grad=True,
                )

                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(
                        [
                            parameter
                            for parameter in unet.parameters()
                            if parameter.requires_grad
                        ],
                        args.max_grad_norm,
                    )

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            if accelerator.sync_gradients:
                global_step += 1
                loss_value = float(loss.detach().item())
                lr_value = float(lr_scheduler.get_last_lr()[0])
                progress_bar.update(1)
                progress_bar.set_postfix(
                    loss=f"{loss_value:.4f}",
                    lr=f"{lr_value:.6f}",
                )

                if tensorboard_writer is not None:
                    tensorboard_writer.add_scalar("train/loss", loss_value, global_step)
                    tensorboard_writer.add_scalar("train/learning_rate", lr_value, global_step)

                if (
                    accelerator.is_main_process
                    and global_step % args.checkpointing_steps == 0
                ):
                    checkpoint_dir = args.output_dir / f"checkpoint-{global_step:06d}"
                    save_checkpoint(accelerator, unet, checkpoint_dir)
                    logger.info("Saved checkpoint to %s", checkpoint_dir)

                if val_dataset is not None and global_step % args.validation_steps == 0:
                    validation_metrics = run_full_validation_cycle(
                        accelerator=accelerator,
                        args=args,
                        unet=unet,
                        vae=vae,
                        text_encoder=text_encoder,
                        noise_scheduler=noise_scheduler,
                        val_dataset=val_dataset,
                        val_dataloader=val_dataloader,
                        sample_dir=args.sample_dir,
                        global_step=global_step,
                        weight_dtype=weight_dtype,
                        empty_prompt_ids=empty_prompt_ids,
                        tensorboard_writer=tensorboard_writer,
                    )
                    metric_info = choose_best_metric(args, validation_metrics)
                    if metric_info is not None:
                        metric_name, metric_value, maximize = metric_info
                        if is_metric_improved(metric_value, best_metric_value, maximize):
                            best_metric_name = metric_name
                            best_metric_value = metric_value
                            best_metric_maximize = maximize
                            best_metadata = {
                                "step": global_step,
                                "metric_name": metric_name,
                                "metric_value": metric_value,
                                "maximize": maximize,
                                "val_loss": validation_metrics.get("val_loss"),
                                "trellis_summary": validation_metrics.get("trellis_summary"),
                            }
                            best_dir = save_best_checkpoint(
                                accelerator=accelerator,
                                unet=unet,
                                output_dir=args.output_dir,
                                metadata=best_metadata,
                            )
                            logger.info(
                                "Updated best checkpoint at step %s using %s=%.4f -> %s",
                                global_step,
                                metric_name,
                                metric_value,
                                best_dir,
                            )

            if global_step >= args.max_train_steps:
                break

        if not made_progress:
            if is_streaming_train:
                if accelerator.is_main_process:
                    logger.info(
                        "No filtered local images are available yet. Waiting %.1f seconds before rescanning metadata.",
                        args.stream_wait_seconds,
                    )
                if args.stream_wait_seconds > 0:
                    time.sleep(args.stream_wait_seconds)
                continue
            raise RuntimeError("Training dataloader produced no batches.")

    accelerator.wait_for_everyone()

    if accelerator.is_main_process:
        final_lora_dir = args.output_dir / "lora"
        save_lora_weights(final_lora_dir, accelerator.unwrap_model(unet))
        logger.info("Saved final LoRA weights to %s", final_lora_dir)

        if val_dataset is not None:
            validation_metrics = run_full_validation_cycle(
                accelerator=accelerator,
                args=args,
                unet=unet,
                vae=vae,
                text_encoder=text_encoder,
                noise_scheduler=noise_scheduler,
                val_dataset=val_dataset,
                val_dataloader=val_dataloader,
                sample_dir=args.sample_dir,
                global_step=global_step,
                weight_dtype=weight_dtype,
                empty_prompt_ids=empty_prompt_ids,
                tensorboard_writer=tensorboard_writer,
            )
            metric_info = choose_best_metric(args, validation_metrics)
            if metric_info is not None:
                metric_name, metric_value, maximize = metric_info
                if is_metric_improved(metric_value, best_metric_value, maximize):
                    best_metric_name = metric_name
                    best_metric_value = metric_value
                    best_metric_maximize = maximize
                    best_metadata = {
                        "step": global_step,
                        "metric_name": metric_name,
                        "metric_value": metric_value,
                        "maximize": maximize,
                        "val_loss": validation_metrics.get("val_loss"),
                        "trellis_summary": validation_metrics.get("trellis_summary"),
                    }
                    best_dir = save_best_checkpoint(
                        accelerator=accelerator,
                        unet=unet,
                        output_dir=args.output_dir,
                        metadata=best_metadata,
                    )
                    logger.info(
                        "Updated best checkpoint at final step %s using %s=%.4f -> %s",
                        global_step,
                        metric_name,
                        metric_value,
                        best_dir,
                    )

        if best_metric_name is not None:
            logger.info(
                "Best checkpoint summary: %s=%.4f (maximize=%s)",
                best_metric_name,
                best_metric_value,
                best_metric_maximize,
            )

        if tensorboard_writer is not None:
            tensorboard_writer.flush()
            tensorboard_writer.close()

    accelerator.end_training()


if __name__ == "__main__":
    main()













