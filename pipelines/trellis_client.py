import base64
import binascii
import json
import mimetypes
from pathlib import Path

import requests

from config import MESH_DIR, TRELLIS_API_URL, TRELLIS_TIMEOUT_SECONDS


def image_to_data_url(path: str) -> str:
    image_path = Path(path)
    if not image_path.exists():
        raise FileNotFoundError(f"Edited image not found: {image_path}")

    mime_type, _ = mimetypes.guess_type(str(image_path))
    if mime_type is None:
        mime_type = "image/png"

    with image_path.open("rb") as file_obj:
        image_b64 = base64.b64encode(file_obj.read()).decode("utf-8")

    return f"data:{mime_type};base64,{image_b64}"


def save_glb_from_base64(glb_b64: str, filename: str = "result.glb") -> str:
    output_path = MESH_DIR / filename
    try:
        glb_bytes = base64.b64decode(glb_b64)
    except (binascii.Error, ValueError) as exc:
        raise ValueError(f"Invalid GLB base64 payload: {exc}") from exc

    with output_path.open("wb") as file_obj:
        file_obj.write(glb_bytes)

    return str(output_path)


def request_3d_generation(edited_image_path: str, seed: int = 0) -> dict:
    try:
        image_data_url = image_to_data_url(edited_image_path)
    except Exception as exc:
        return {
            "success": False,
            "glb_path": None,
            "message": str(exc),
        }

    payload = {
        "image": image_data_url,
        "seed": int(seed),
    }

    try:
        response = requests.post(
            TRELLIS_API_URL,
            headers={
                "Accept": "application/json",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=TRELLIS_TIMEOUT_SECONDS,
        )
    except requests.RequestException as exc:
        return {
            "success": False,
            "glb_path": None,
            "message": f"Failed to reach TRELLIS server: {exc}",
        }

    try:
        data = response.json()
    except ValueError:
        data = None

    if response.status_code != 200:
        if data is not None:
            error_message = json.dumps(data, ensure_ascii=False)
        else:
            error_message = response.text
        return {
            "success": False,
            "glb_path": None,
            "message": f"TRELLIS API error ({response.status_code}): {error_message}",
        }

    if not isinstance(data, dict):
        return {
            "success": False,
            "glb_path": None,
            "message": "TRELLIS returned a non-JSON response.",
        }

    artifacts = data.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        return {
            "success": False,
            "glb_path": None,
            "message": f"TRELLIS response missing artifacts: {json.dumps(data, ensure_ascii=False)}",
        }

    first_artifact = artifacts[0]
    if not isinstance(first_artifact, dict) or "base64" not in first_artifact:
        return {
            "success": False,
            "glb_path": None,
            "message": (
                "TRELLIS response missing artifacts[0].base64: "
                f"{json.dumps(data, ensure_ascii=False)}"
            ),
        }

    try:
        glb_path = save_glb_from_base64(first_artifact["base64"])
    except Exception as exc:
        return {
            "success": False,
            "glb_path": None,
            "message": f"Failed to save GLB: {exc}",
        }

    return {
        "success": True,
        "glb_path": glb_path,
        "message": f"3D generation successful. GLB saved to: {glb_path}",
    }
