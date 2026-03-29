import hashlib
import io
import uuid
from pathlib import Path
from typing import Any, Dict, Tuple

import gradio as gr
import numpy as np
from PIL import Image

from config import (
    APP_HOST,
    APP_PORT,
    DEFAULT_GUIDANCE_SCALE,
    DEFAULT_IMAGE_GUIDANCE_SCALE,
    DEFAULT_STEPS,
    EDITED_DIR,
    PREPROCESS_SIZE,
)
from pipelines.image_editor import edit_image_with_prompt
from pipelines.preprocess import preprocess_image
from pipelines.text_to_image import generate_image_from_text
from pipelines.trellis_client import request_3d_generation


def build_edit_prompt(user_prompt: str) -> str:
    prompt = (user_prompt or "").strip()
    if not prompt:
        raise ValueError("Please enter a text prompt before generating.")

    return (
        f"{prompt}. "
        "Keep a single centered object, clean background, clear silhouette, "
        "product-style view, suitable for 3D asset generation."
    )


def build_text_to_image_prompt(user_prompt: str) -> str:
    prompt = (user_prompt or "").strip()
    if not prompt:
        raise ValueError("Please enter a text prompt before generating.")

    return (
        f"{prompt}. "
        "Single centered object, clean studio background, clear silhouette, "
        "front three-quarter product render, suitable for 3D asset generation."
    )


def _coerce_int(value: Any, default: int) -> int:
    if value is None:
        return default
    return int(value)


def _coerce_float(value: Any, default: float) -> float:
    if value is None:
        return default
    return float(value)


def _ensure_pil_image(image_value: Any) -> Image.Image:
    if image_value is None:
        raise ValueError("Please upload an image first.")

    if isinstance(image_value, Image.Image):
        return image_value.convert("RGB")

    if isinstance(image_value, np.ndarray):
        return Image.fromarray(image_value).convert("RGB")

    raise TypeError(f"Unsupported image input type: {type(image_value).__name__}")


def _normalize_edited_output(edited_output: Any) -> Image.Image:
    if isinstance(edited_output, tuple):
        if not edited_output:
            raise ValueError("Image editor returned an empty tuple.")
        edited_output = edited_output[0]

    if hasattr(edited_output, "images"):
        images = getattr(edited_output, "images")
        if isinstance(images, (list, tuple)) and images:
            edited_output = images[0]

    if not isinstance(edited_output, Image.Image):
        raise TypeError(
            "Image editor did not return a PIL image. "
            f"Received: {type(edited_output).__name__}"
        )

    return edited_output.convert("RGB")


def _save_prepared_image(image: Image.Image) -> str:
    edited_name = f"{uuid.uuid4().hex}.png"
    edited_path = EDITED_DIR / edited_name
    image.save(edited_path)
    return str(edited_path)


def _compute_request_signature(
    mode: str,
    image: Image.Image | None = None,
    final_prompt: str = "",
    steps: int | None = None,
    guidance: float | None = None,
    image_guidance: float | None = None,
) -> str:
    payload = [mode.encode("utf-8")]

    if image is not None:
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        payload.append(buffer.getvalue())

    if final_prompt:
        payload.append(final_prompt.encode("utf-8"))

    if steps is not None:
        payload.append(f"|{steps}".encode("utf-8"))
    if guidance is not None:
        payload.append(f"|{guidance}".encode("utf-8"))
    if image_guidance is not None:
        payload.append(f"|{image_guidance}".encode("utf-8"))

    return hashlib.md5(b"".join(payload)).hexdigest()


def _detect_generation_mode(input_image: Any, prompt: str) -> str:
    has_image = input_image is not None
    has_prompt = bool((prompt or "").strip())

    if has_image and has_prompt:
        return "image+prompt"
    if has_image:
        return "image"
    if has_prompt:
        return "text"

    raise ValueError("Please upload an image, enter a prompt, or provide both.")


def _maybe_reuse_edited_image(
    edit_state: Any,
    request_signature: str,
) -> Tuple[Image.Image | None, str | None, Dict[str, str], bool]:
    if not isinstance(edit_state, dict):
        return None, None, {}, False

    edited_path = edit_state.get("edited_path")
    saved_signature = edit_state.get("signature")

    if not edited_path or saved_signature != request_signature:
        return None, None, {}, False

    path = Path(edited_path)
    if not path.exists():
        return None, None, {}, False

    with Image.open(path) as cached_image:
        edited_image = cached_image.convert("RGB")

    reused_state = {
        "signature": request_signature,
        "edited_path": str(path),
        "mode": str(edit_state.get("mode", "")),
        "source": str(edit_state.get("source", "")),
    }
    return edited_image, str(path), reused_state, True


def _prepare_image_for_3d(
    input_image: Any,
    prompt: str,
    steps: Any,
    guidance: Any,
    image_guidance: Any,
    edit_state: Any = None,
) -> Tuple[Image.Image, str, Dict[str, str], bool]:
    steps_value = _coerce_int(steps, DEFAULT_STEPS)
    guidance_value = _coerce_float(guidance, DEFAULT_GUIDANCE_SCALE)
    image_guidance_value = _coerce_float(
        image_guidance,
        DEFAULT_IMAGE_GUIDANCE_SCALE,
    )

    mode = _detect_generation_mode(input_image, prompt)
    pil_image = (
        _ensure_pil_image(input_image) if mode in {"image+prompt", "image"} else None
    )

    if mode == "image+prompt":
        final_prompt = build_edit_prompt(prompt)
        request_signature = _compute_request_signature(
            mode=mode,
            image=pil_image,
            final_prompt=final_prompt,
            steps=steps_value,
            guidance=guidance_value,
            image_guidance=image_guidance_value,
        )
        source_label = "Edited uploaded image with prompt."
    elif mode == "text":
        final_prompt = build_text_to_image_prompt(prompt)
        request_signature = _compute_request_signature(
            mode=mode,
            final_prompt=final_prompt,
            steps=steps_value,
            guidance=guidance_value,
        )
        source_label = "Generated image from text prompt."
    else:
        final_prompt = ""
        request_signature = _compute_request_signature(
            mode=mode,
            image=pil_image,
        )
        source_label = "Used uploaded image directly."

    reused_image, reused_path, reused_state, reused = _maybe_reuse_edited_image(
        edit_state,
        request_signature,
    )
    if reused and reused_image is not None and reused_path is not None:
        return reused_image, reused_path, reused_state, True

    if mode == "image+prompt":
        preprocessed = preprocess_image(pil_image, size=PREPROCESS_SIZE)
        prepared_output = edit_image_with_prompt(
            preprocessed,
            final_prompt,
            num_inference_steps=steps_value,
            guidance_scale=guidance_value,
            image_guidance_scale=image_guidance_value,
        )
        prepared_image = _normalize_edited_output(prepared_output)
    elif mode == "text":
        generated_image = generate_image_from_text(
            final_prompt,
            num_inference_steps=steps_value,
            guidance_scale=guidance_value,
            size=PREPROCESS_SIZE,
        )
        prepared_image = preprocess_image(generated_image, size=PREPROCESS_SIZE)
    else:
        prepared_image = preprocess_image(pil_image, size=PREPROCESS_SIZE)

    edited_path = _save_prepared_image(prepared_image)

    state = {
        "signature": request_signature,
        "edited_path": edited_path,
        "mode": mode,
        "source": source_label,
    }
    return prepared_image, edited_path, state, False


def run_edit(
    input_image: Any,
    prompt: str,
    steps: Any,
    guidance: Any,
    image_guidance: Any,
):
    try:
        prepared_image, edited_path, state, reused = _prepare_image_for_3d(
            input_image,
            prompt,
            steps,
            guidance,
            image_guidance,
        )
        source = state.get("source", "Prepared image ready.")
        prefix = "Reused existing prepared image." if reused else source
        return prepared_image, state, f"{prefix}\nSaved to: {edited_path}"
    except Exception as exc:
        return None, {}, f"Preparation failed: {exc}"


def run_full_pipeline(
    input_image: Any,
    prompt: str,
    steps: Any,
    guidance: Any,
    image_guidance: Any,
    edit_state: Any,
):
    try:
        prepared_image, edited_path, state, reused = _prepare_image_for_3d(
            input_image,
            prompt,
            steps,
            guidance,
            image_guidance,
            edit_state=edit_state,
        )
    except Exception as exc:
        return None, {}, None, None, f"Preparation failed: {exc}"

    result = request_3d_generation(edited_path)
    if not result["success"]:
        return (
            prepared_image,
            state,
            None,
            None,
            "3D generation failed: " f"{result['message']}",
        )

    glb_path = result["glb_path"]
    prefix = (
        "Reused existing prepared image."
        if reused
        else state.get("source", "Prepared image ready.")
    )
    status = f"{prefix}\n{result['message']}"
    return prepared_image, state, glb_path, glb_path, status


with gr.Blocks(title="Multimodal 3D Demo") as demo:
    gr.Markdown("# Multimodal Prompt/Image to 3D Demo")
    gr.Markdown(
        "Supports image + prompt, image only, or prompt only. The app always prepares an image locally before sending it to TRELLIS."
    )

    edited_state = gr.State(value={})

    with gr.Row():
        input_image = gr.Image(label="Input Image", type="numpy")
        edited_image = gr.Image(label="Prepared Image", type="pil")

    prompt = gr.Textbox(
        label="Prompt",
        placeholder="Optional. Example: blue fantasy collectible figurine",
    )

    with gr.Row():
        steps = gr.Slider(10, 100, value=DEFAULT_STEPS, step=1, label="Inference Steps")
        guidance = gr.Slider(
            1.0,
            12.0,
            value=DEFAULT_GUIDANCE_SCALE,
            step=0.5,
            label="Prompt Guidance",
        )
        image_guidance = gr.Slider(
            1.0,
            3.0,
            value=DEFAULT_IMAGE_GUIDANCE_SCALE,
            step=0.1,
            label="Image Guidance",
        )

    with gr.Row():
        edit_btn = gr.Button("Preview Prepared Image", variant="secondary")
        gen_btn = gr.Button("Generate 3D", variant="primary")

    with gr.Row():
        if hasattr(gr, "Model3D"):
            model_preview = gr.Model3D(label="3D Preview")
        else:
            model_preview = gr.File(label="3D Preview (.glb)")
        output_model = gr.File(label="Download GLB")

    status_box = gr.Textbox(label="Status", lines=6, interactive=False)

    edit_btn.click(
        fn=run_edit,
        inputs=[input_image, prompt, steps, guidance, image_guidance],
        outputs=[edited_image, edited_state, status_box],
    )

    gen_btn.click(
        fn=run_full_pipeline,
        inputs=[input_image, prompt, steps, guidance, image_guidance, edited_state],
        outputs=[edited_image, edited_state, model_preview, output_model, status_box],
    )


if __name__ == "__main__":
    demo.queue().launch(server_name=APP_HOST, server_port=APP_PORT)
