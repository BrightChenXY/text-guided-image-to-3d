from pathlib import Path
from config import ASSET_DIR, MESH_DIR


def generate_mock_3d(prompt: str, edited_image_path: str) -> dict:
    placeholder = ASSET_DIR / "placeholder.glb"
    output_glb = MESH_DIR / "result.glb"

    if placeholder.exists():
        output_glb.write_bytes(placeholder.read_bytes())
        return {
            "success": True,
            "glb_path": str(output_glb),
            "message": f"Mock 3D generated for prompt: {prompt}",
        }

    return {"success": False, "glb_path": None, "message": "placeholder.glb not found"}
