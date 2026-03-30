import hashlib
import io
import json
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
    DEMO_TEMPLATE_MANIFEST,
    EDITED_DIR,
    PREPROCESS_SIZE,
)
from pipelines.image_editor import edit_image_with_prompt
from pipelines.preprocess import preprocess_image
from pipelines.text_to_image import generate_image_from_text
from pipelines.trellis_client import request_3d_generation


INPUT_PANEL_HEIGHT = 320
PREVIEW_PANEL_HEIGHT = 420
DOWNLOAD_PANEL_HEIGHT = 72
PROJECT_ROOT = Path(__file__).resolve().parent

UI_CSS = f"""
#prompt-box {{
    height: {INPUT_PANEL_HEIGHT}px;
}}

#prompt-box textarea {{
    min-height: calc({INPUT_PANEL_HEIGHT}px - 56px) !important;
    height: calc({INPUT_PANEL_HEIGHT}px - 56px) !important;
    resize: none !important;
}}

#download-glb {{
    min-height: {DOWNLOAD_PANEL_HEIGHT}px;
}}

#download-glb .file-wrap,
#download-glb .file-preview,
#download-glb .empty,
#download-glb .wrap {{
    min-height: {DOWNLOAD_PANEL_HEIGHT}px !important;
}}

#prepared-image,
#model-preview,
#model-preview-file {{
    min-height: {PREVIEW_PANEL_HEIGHT}px;
}}

#prepared-image .image-container,
#prepared-image .image-frame,
#model-preview model-viewer,
#model-preview .wrap,
#model-preview-file .file-wrap,
#model-preview-file .file-preview,
#model-preview-file .empty,
#model-preview-file .wrap {{
    min-height: {PREVIEW_PANEL_HEIGHT}px !important;
    height: {PREVIEW_PANEL_HEIGHT}px !important;
}}

"""


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


def _resolve_template_path(path_value: str | None) -> Path | None:
    if not path_value:
        return None

    candidate = Path(path_value)
    if not candidate.is_absolute():
        candidate = (PROJECT_ROOT / candidate).resolve()
    else:
        candidate = candidate.resolve()

    return candidate


def _is_usable_file(path_value: Path | None) -> bool:
    if path_value is None or not path_value.exists() or not path_value.is_file():
        return False
    return path_value.stat().st_size > 0


def _load_demo_templates() -> dict[str, dict[str, Any]]:
    manifest_path = DEMO_TEMPLATE_MANIFEST
    if not manifest_path.exists():
        return {}

    with manifest_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    items = payload.get("templates", payload) if isinstance(payload, dict) else payload
    templates: dict[str, dict[str, Any]] = {}

    if not isinstance(items, list):
        return templates

    for raw_template in items:
        if not isinstance(raw_template, dict):
            continue

        template_id = str(raw_template.get("id", "")).strip()
        label = str(raw_template.get("label", template_id)).strip()
        if not template_id or not label:
            continue

        input_path = _resolve_template_path(raw_template.get("input_image"))
        edited_path = _resolve_template_path(raw_template.get("edited_image"))
        glb_path = _resolve_template_path(raw_template.get("glb_path"))

        templates[template_id] = {
            "id": template_id,
            "label": label,
            "prompt": str(raw_template.get("prompt", "")).strip(),
            "description": str(raw_template.get("description", "")).strip(),
            "input_image_path": (
                str(input_path) if _is_usable_file(input_path) else None
            ),
            "edited_image_path": (
                str(edited_path) if _is_usable_file(edited_path) else None
            ),
            "glb_path": str(glb_path) if _is_usable_file(glb_path) else None,
            "steps": _coerce_int(raw_template.get("steps"), DEFAULT_STEPS),
            "guidance": _coerce_float(
                raw_template.get("guidance"), DEFAULT_GUIDANCE_SCALE
            ),
            "image_guidance": _coerce_float(
                raw_template.get("image_guidance"),
                DEFAULT_IMAGE_GUIDANCE_SCALE,
            ),
        }

    return templates


def _load_pil_from_path(path_value: str | None) -> Image.Image | None:
    if not path_value:
        return None

    path = Path(path_value)
    if not path.exists():
        return None

    with Image.open(path) as handle:
        return handle.convert("RGB")


def _build_template_state(
    template: dict[str, Any],
    input_pil: Image.Image | None,
) -> dict[str, str]:
    edited_path = template.get("edited_image_path")
    if not edited_path:
        return {}

    prompt = str(template.get("prompt", "")).strip()
    steps = _coerce_int(template.get("steps"), DEFAULT_STEPS)
    guidance = _coerce_float(template.get("guidance"), DEFAULT_GUIDANCE_SCALE)
    image_guidance = _coerce_float(
        template.get("image_guidance"),
        DEFAULT_IMAGE_GUIDANCE_SCALE,
    )

    mode = _detect_generation_mode(input_pil, prompt)
    if mode == "image+prompt":
        final_prompt = build_edit_prompt(prompt)
        signature = _compute_request_signature(
            mode=mode,
            image=input_pil,
            final_prompt=final_prompt,
            steps=steps,
            guidance=guidance,
            image_guidance=image_guidance,
        )
    elif mode == "text":
        final_prompt = build_text_to_image_prompt(prompt)
        signature = _compute_request_signature(
            mode=mode,
            final_prompt=final_prompt,
            steps=steps,
            guidance=guidance,
        )
    else:
        signature = _compute_request_signature(mode=mode, image=input_pil)

    return {
        "signature": signature,
        "edited_path": str(Path(edited_path).resolve()),
        "mode": mode,
        "source": f"Loaded demo template: {template['label']}",
    }


DEMO_TEMPLATE_CLEAR_VALUE = "__clear_template__"


def clear_demo_template():
    return (
        gr.update(value=None),
        None,
        "",
        DEFAULT_STEPS,
        DEFAULT_GUIDANCE_SCALE,
        DEFAULT_IMAGE_GUIDANCE_SCALE,
        None,
        {},
        None,
        None,
        "Cleared demo template. You can now upload an image or enter a new prompt.",
    )


def apply_demo_template(template_id: str):
    if template_id == DEMO_TEMPLATE_CLEAR_VALUE:
        return clear_demo_template()

    template = DEMO_TEMPLATES.get(template_id)
    if template is None:
        return (
            gr.update(value=None),
            None,
            "",
            DEFAULT_STEPS,
            DEFAULT_GUIDANCE_SCALE,
            DEFAULT_IMAGE_GUIDANCE_SCALE,
            None,
            {},
            None,
            None,
            "No demo template selected.",
        )

    input_pil = _load_pil_from_path(template.get("input_image_path"))
    edited_pil = _load_pil_from_path(template.get("edited_image_path"))
    glb_path = template.get("glb_path")
    state = _build_template_state(template, input_pil)

    input_value = np.array(input_pil) if input_pil is not None else None
    model_value = glb_path if glb_path else None
    description = template.get("description") or "Loaded precomputed demo assets."

    status_lines = [
        f"Loaded demo template: {template['label']}",
        description,
    ]
    if template.get("edited_image_path"):
        status_lines.append("Prepared image loaded from disk and ready for reuse.")
    if glb_path:
        status_lines.append(f"Precomputed GLB loaded: {glb_path}")
    else:
        status_lines.append("No precomputed GLB configured for this template.")

    return (
        gr.update(value=template_id),
        input_value,
        template.get("prompt", ""),
        template.get("steps", DEFAULT_STEPS),
        template.get("guidance", DEFAULT_GUIDANCE_SCALE),
        template.get("image_guidance", DEFAULT_IMAGE_GUIDANCE_SCALE),
        edited_pil,
        state,
        model_value,
        model_value,
        "\n".join(status_lines),
    )


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


DEMO_TEMPLATES = _load_demo_templates()
DEMO_TEMPLATE_CHOICES = [("--- Clear Template ---", DEMO_TEMPLATE_CLEAR_VALUE)] + [
    (template["label"], template_id) for template_id, template in DEMO_TEMPLATES.items()
]

with gr.Blocks(title="Multimodal 3D Demo", css=UI_CSS) as demo:
    gr.Markdown("# Multimodal Prompt/Image to 3D Demo")
    gr.Markdown(
        "Supports image + prompt, image only, or prompt only. The app always prepares an image locally before sending it to TRELLIS."
    )

    edited_state = gr.State(value={})

    demo_template_dropdown = gr.Dropdown(
        choices=DEMO_TEMPLATE_CHOICES,
        value=None,
        label="Demo Template",
        info="Choose a precomputed example to populate the input fields and load cached outputs.",
    )

    with gr.Row():
        input_image = gr.Image(
            label="Input Image",
            type="numpy",
            height=INPUT_PANEL_HEIGHT,
            elem_id="input-image",
        )
        prompt = gr.Textbox(
            label="Prompt",
            placeholder="Optional. Example: blue fantasy collectible figurine",
            lines=12,
            max_lines=12,
            scale=1,
            elem_id="prompt-box",
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
        edited_image = gr.Image(
            label="Prepared Image",
            type="pil",
            height=PREVIEW_PANEL_HEIGHT,
            elem_id="prepared-image",
        )
        if hasattr(gr, "Model3D"):
            model_preview = gr.Model3D(
                label="3D Preview",
                height=PREVIEW_PANEL_HEIGHT,
                elem_id="model-preview",
            )
        else:
            model_preview = gr.File(
                label="3D Preview (.glb)",
                elem_id="model-preview-file",
            )

    output_model = gr.File(label="Download GLB", elem_id="download-glb")
    status_box = gr.Textbox(label="Status", lines=6, interactive=False)

    demo_template_dropdown.change(
        fn=apply_demo_template,
        inputs=[demo_template_dropdown],
        outputs=[
            demo_template_dropdown,
            input_image,
            prompt,
            steps,
            guidance,
            image_guidance,
            edited_image,
            edited_state,
            model_preview,
            output_model,
            status_box,
        ],
    )

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
