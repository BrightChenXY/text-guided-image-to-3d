from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from pipelines.trellis_client import request_3d_generation
from training.dataset import prepare_square_image

try:
    import pyrender
    import trimesh
except ImportError:
    pyrender = None
    trimesh = None

DEFAULT_VIEW_ORDER = ("front", "left", "right", "back", "top")


def trellis_rerank_dependencies_available() -> bool:
    return pyrender is not None and trimesh is not None


def _require_renderer() -> None:
    if not trellis_rerank_dependencies_available():
        raise ImportError(
            "TRELLIS rerank rendering requires trimesh and pyrender. "
            "Please install them before enabling black-box rerank."
        )


def _look_at_pose(
    eye: np.ndarray,
    target: np.ndarray | None = None,
    up: np.ndarray | None = None,
) -> np.ndarray:
    target = np.zeros(3, dtype=np.float32) if target is None else target.astype(np.float32)
    up = np.array([0.0, 1.0, 0.0], dtype=np.float32) if up is None else up.astype(np.float32)

    forward = target - eye
    forward = forward / np.linalg.norm(forward)

    right = np.cross(forward, up)
    right_norm = np.linalg.norm(right)
    if right_norm < 1e-6:
        up = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        right = np.cross(forward, up)
        right_norm = np.linalg.norm(right)
    right = right / right_norm

    true_up = np.cross(right, forward)
    true_up = true_up / np.linalg.norm(true_up)

    pose = np.eye(4, dtype=np.float32)
    pose[:3, 0] = right
    pose[:3, 1] = true_up
    pose[:3, 2] = -forward
    pose[:3, 3] = eye
    return pose


def _normalize_scene(scene: Any) -> Any:
    scene = scene.copy()
    bounds = scene.bounds
    if bounds is None:
        raise ValueError("Loaded GLB scene does not expose bounds.")

    minimum, maximum = bounds
    extents = maximum - minimum
    scale = float(np.max(extents))
    if scale <= 0:
        raise ValueError("Loaded GLB scene has invalid scale.")

    center = (minimum + maximum) / 2.0
    scene.apply_translation(-center)
    scene.apply_scale(1.0 / scale)
    return scene


def render_glb_views(
    glb_path: str | Path,
    image_size: int = 256,
    view_order: tuple[str, ...] = DEFAULT_VIEW_ORDER,
) -> dict[str, Image.Image]:
    _require_renderer()

    scene = trimesh.load(str(glb_path), force="scene")
    scene = _normalize_scene(scene)
    meshes = scene.dump(concatenate=False)
    if not meshes:
        raise ValueError("The GLB scene does not contain renderable meshes.")

    render_scene = pyrender.Scene(
        bg_color=np.array([255, 255, 255, 255], dtype=np.uint8),
        ambient_light=np.array([0.45, 0.45, 0.45, 1.0], dtype=np.float32),
    )
    for mesh in meshes:
        if hasattr(mesh, "vertices") and len(mesh.vertices) > 0:
            render_scene.add(pyrender.Mesh.from_trimesh(mesh, smooth=False))

    camera = pyrender.PerspectiveCamera(yfov=np.pi / 4.0, aspectRatio=1.0)
    light = pyrender.DirectionalLight(color=np.ones(3), intensity=3.0)
    renderer = pyrender.OffscreenRenderer(image_size, image_size)

    view_positions = {
        "front": np.array([0.0, 0.0, 2.2], dtype=np.float32),
        "left": np.array([-2.2, 0.0, 0.0], dtype=np.float32),
        "right": np.array([2.2, 0.0, 0.0], dtype=np.float32),
        "back": np.array([0.0, 0.0, -2.2], dtype=np.float32),
        "top": np.array([0.0, 1.8, 1.2], dtype=np.float32),
    }

    rendered_views: dict[str, Image.Image] = {}
    try:
        for view_name in view_order:
            eye = view_positions[view_name]
            pose = _look_at_pose(eye)
            camera_node = render_scene.add(camera, pose=pose)
            light_node = render_scene.add(light, pose=pose)
            color, _ = renderer.render(render_scene)
            render_scene.remove_node(camera_node)
            render_scene.remove_node(light_node)
            rendered_views[view_name] = Image.fromarray(color).convert("RGB")
    finally:
        renderer.delete()

    return rendered_views


def _foreground_mask(image: Image.Image) -> np.ndarray:
    array = np.asarray(image.convert("RGB"), dtype=np.uint8)
    return np.any(array < 245, axis=-1)


def _score_coverage(mask: np.ndarray) -> float:
    ratio = float(mask.mean())
    if ratio <= 0:
        return 0.0
    return float(max(0.0, 1.0 - abs(ratio - 0.35) / 0.35))


def _score_centering(mask: np.ndarray) -> float:
    ys, xs = np.where(mask)
    if len(xs) == 0 or len(ys) == 0:
        return 0.0

    width = mask.shape[1]
    height = mask.shape[0]
    center_x = ((xs.min() + xs.max()) / 2.0) / max(width - 1, 1)
    center_y = ((ys.min() + ys.max()) / 2.0) / max(height - 1, 1)
    distance = ((center_x - 0.5) ** 2 + (center_y - 0.5) ** 2) ** 0.5
    normalized = min(distance / 0.70710678, 1.0)
    return float(1.0 - normalized)


def _image_similarity(rendered_image: Image.Image, reference_image: Image.Image) -> float:
    resolution = rendered_image.size[0]
    reference = prepare_square_image(reference_image.convert("RGB"), resolution=resolution)
    rendered = np.asarray(rendered_image, dtype=np.float32) / 255.0
    reference_arr = np.asarray(reference, dtype=np.float32) / 255.0
    l1_distance = float(np.abs(rendered - reference_arr).mean())
    return float(max(0.0, 1.0 - l1_distance))


def score_rendered_views(
    edited_image: Image.Image,
    rendered_views: dict[str, Image.Image],
) -> dict[str, Any]:
    coverage_scores: dict[str, float] = {}
    centering_scores: dict[str, float] = {}
    coverage_ratios: dict[str, float] = {}

    for view_name, rendered_image in rendered_views.items():
        mask = _foreground_mask(rendered_image)
        coverage_ratios[view_name] = float(mask.mean())
        coverage_scores[view_name] = _score_coverage(mask)
        centering_scores[view_name] = _score_centering(mask)

    front_similarity = _image_similarity(rendered_views["front"], edited_image)
    consistency = 1.0 - min(float(np.std(list(coverage_ratios.values()))) / 0.2, 1.0)
    mean_coverage = float(np.mean(list(coverage_scores.values())))
    mean_centering = float(np.mean(list(centering_scores.values())))

    overall_score = (
        0.40 * front_similarity
        + 0.25 * mean_coverage
        + 0.20 * mean_centering
        + 0.15 * consistency
    )

    return {
        "front_similarity": float(front_similarity),
        "mean_coverage_score": mean_coverage,
        "mean_centering_score": mean_centering,
        "view_consistency_score": float(max(0.0, consistency)),
        "coverage_ratios": coverage_ratios,
        "coverage_scores": coverage_scores,
        "centering_scores": centering_scores,
        "overall_score": float(max(0.0, min(overall_score, 1.0))),
    }


def evaluate_edited_image_with_trellis(
    edited_image: Image.Image,
    work_dir: str | Path,
    seed: int = 0,
    render_size: int = 256,
) -> dict[str, Any]:
    output_dir = Path(work_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    edited_image_path = output_dir / "edited_input.png"
    edited_image.convert("RGB").save(edited_image_path)

    trellis_result = request_3d_generation(str(edited_image_path), seed=seed)
    if not trellis_result.get("success"):
        result = {
            "success": False,
            "score": 0.0,
            "message": str(trellis_result.get("message", "TRELLIS generation failed.")),
            "glb_path": trellis_result.get("glb_path"),
            "metrics": {},
            "render_paths": {},
        }
        with (output_dir / "trellis_metrics.json").open("w", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2, ensure_ascii=False)
        return result

    source_glb_path = Path(str(trellis_result["glb_path"])).expanduser().resolve()
    stored_glb_path = output_dir / "trellis_result.glb"
    shutil.copy2(source_glb_path, stored_glb_path)

    result: dict[str, Any] = {
        "success": True,
        "score": 0.0,
        "message": str(trellis_result.get("message", "")),
        "glb_path": str(stored_glb_path),
        "metrics": {"trellis_success": 1.0},
        "render_paths": {},
    }

    try:
        rendered_views = render_glb_views(stored_glb_path, image_size=render_size)
        render_paths: dict[str, str] = {}
        for view_name, rendered_image in rendered_views.items():
            render_path = output_dir / f"render_{view_name}.png"
            rendered_image.save(render_path)
            render_paths[view_name] = str(render_path)

        metrics = score_rendered_views(edited_image, rendered_views)
        result["metrics"].update(metrics)
        result["score"] = float(metrics["overall_score"])
        result["render_paths"] = render_paths
    except Exception as exc:
        result["success"] = False
        result["message"] = f"TRELLIS succeeded, but GLB rendering/scoring failed: {exc}"
        result["metrics"]["render_success"] = 0.0
        result["score"] = 0.0

    with (output_dir / "trellis_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, ensure_ascii=False)

    return result
