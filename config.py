import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"
EDITED_DIR = OUTPUT_DIR / "edited"
PREVIEW_DIR = OUTPUT_DIR / "previews"
MESH_DIR = OUTPUT_DIR / "meshes"
ASSET_DIR = BASE_DIR / "assets"
DEMO_TEMPLATE_MANIFEST = ASSET_DIR / "demo_templates.json"

for directory in (OUTPUT_DIR, EDITED_DIR, PREVIEW_DIR, MESH_DIR):
    directory.mkdir(parents=True, exist_ok=True)

APP_HOST = "127.0.0.1"
APP_PORT = 7860


def _get_env_float(name: str, default: float) -> float:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default

    try:
        return float(raw_value)
    except ValueError:
        return default


TRELLIS_API_URL = "http://100.66.5.110:8000/v1/infer"
TRELLIS_TIMEOUT_SECONDS = 600
TRELLIS_MAX_IMAGE_SIZE = 1024
TRELLIS_JPEG_QUALITY = 92
TRELLIS_REQUEST_RETRIES = 2

INSTRUCT_PIX2PIX_MODEL_ID = "timbrooks/instruct-pix2pix"
INSTRUCT_PIX2PIX_LORA_PATH = os.getenv("INSTRUCT_PIX2PIX_LORA_PATH") or None
INSTRUCT_PIX2PIX_LORA_SCALE = _get_env_float("INSTRUCT_PIX2PIX_LORA_SCALE", 1.0)
TEXT_TO_IMAGE_MODEL_ID = "runwayml/stable-diffusion-v1-5"
PREPROCESS_SIZE = 512

DEFAULT_STEPS = 20
DEFAULT_GUIDANCE_SCALE = 7.5
DEFAULT_IMAGE_GUIDANCE_SCALE = 1.5
DEFAULT_SEED = 0
