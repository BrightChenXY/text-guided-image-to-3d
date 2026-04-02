import os
import sys
import json
import csv
import shutil
import argparse
import subprocess
from pathlib import Path

import numpy as np
import torch
import trimesh
import open3d as o3d
from PIL import Image
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from transformers import CLIPTokenizer, CLIPTextModel

# Make TRELLIS toolkit utils importable
sys.path.append('/workspace/TRELLIS/dataset_toolkits')
from utils import sphere_hammersley_sequence


TRELLIS_ROOT = "/workspace/TRELLIS"
BLENDER_LINK = "https://download.blender.org/release/Blender3.0/blender-3.0.1-linux-x64.tar.xz"
BLENDER_INSTALLATION_PATH = "/tmp"
BLENDER_PATH = f"{BLENDER_INSTALLATION_PATH}/blender-3.0.1-linux-x64/blender"


def run(cmd, env=None):
    print("\n[RUN]", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=TRELLIS_ROOT, env=env)


def ensure_blender():
    if os.path.exists(BLENDER_PATH):
        return
    subprocess.run("apt-get update", shell=True, check=True)
    subprocess.run(
        "apt-get install -y wget libxrender1 libxi6 libxkbcommon-x11-0 libsm6",
        shell=True,
        check=True,
    )
    subprocess.run(
        f"wget -O {BLENDER_INSTALLATION_PATH}/blender-3.0.1-linux-x64.tar.xz {BLENDER_LINK}",
        shell=True,
        check=True,
    )
    subprocess.run(
        f"tar -xf {BLENDER_INSTALLATION_PATH}/blender-3.0.1-linux-x64.tar.xz -C {BLENDER_INSTALLATION_PATH}",
        shell=True,
        check=True,
    )


def load_models(device):
    repo_cache = "/root/.cache/torch/hub/facebookresearch_dinov2_main"
    if os.path.exists(repo_cache):
        dinov2 = torch.hub.load(repo_cache, "dinov2_vitl14_reg", source="local")
    else:
        dinov2 = torch.hub.load("facebookresearch/dinov2", "dinov2_vitl14_reg")
    dinov2 = dinov2.eval().to(device)

    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
    text_model = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14").eval().to(device)

    return dinov2, tokenizer, text_model


def export_mesh_from_glb(src_glb, mesh_out):
    if os.path.exists(mesh_out):
        return

    scene_or_mesh = trimesh.load(src_glb, force="scene")

    if isinstance(scene_or_mesh, trimesh.Scene):
        meshes = []
        for g in scene_or_mesh.geometry.values():
            if isinstance(g, trimesh.Trimesh) and len(g.vertices) > 0 and len(g.faces) > 0:
                meshes.append(g)
        if not meshes:
            raise RuntimeError(f"No valid mesh geometry found inside {src_glb}")
        mesh = trimesh.util.concatenate(meshes) if len(meshes) > 1 else meshes[0]
    else:
        mesh = scene_or_mesh
        if not isinstance(mesh, trimesh.Trimesh):
            raise RuntimeError(f"Loaded object is not a mesh: {type(mesh)}")

    Path(mesh_out).parent.mkdir(parents=True, exist_ok=True)
    mesh.export(mesh_out)


def write_metadata_and_instances(sample_root, sample_id):
    meta_csv = os.path.join(sample_root, "metadata.csv")
    instances_txt = os.path.join(sample_root, "instances.txt")

    if not os.path.exists(meta_csv):
        with open(meta_csv, "w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["sha256", "rendered", "voxelized", "aesthetic_score"]
            )
            writer.writeheader()
            writer.writerow({
                "sha256": sample_id,
                "rendered": True,
                "voxelized": False,
                "aesthetic_score": 10.0
            })

    if not os.path.exists(instances_txt):
        with open(instances_txt, "w") as f:
            f.write(sample_id + "\n")

    return meta_csv, instances_txt


def voxelize_mesh(mesh_ply, voxel_out):
    if os.path.exists(voxel_out):
        return

    mesh_o3d = o3d.io.read_triangle_mesh(mesh_ply)
    if mesh_o3d.is_empty():
        raise RuntimeError(f"Open3D loaded an empty mesh: {mesh_ply}")

    verts = np.asarray(mesh_o3d.vertices)
    if verts.size == 0:
        raise RuntimeError(f"Mesh has zero vertices: {mesh_ply}")

    verts = np.clip(verts, -0.5 + 1e-6, 0.5 - 1e-6)
    mesh_o3d.vertices = o3d.utility.Vector3dVector(verts)

    voxel_grid = o3d.geometry.VoxelGrid.create_from_triangle_mesh_within_bounds(
        mesh_o3d,
        voxel_size=1 / 64,
        min_bound=(-0.5, -0.5, -0.5),
        max_bound=(0.5, 0.5, 0.5),
    )

    grid_pts = np.array([v.grid_index for v in voxel_grid.get_voxels()], dtype=np.float32)
    if len(grid_pts) == 0:
        raise RuntimeError(f"Voxelization produced 0 active voxels: {mesh_ply}")

    assert np.all(grid_pts >= 0) and np.all(grid_pts < 64), "Voxel indices out of bounds"

    grid_pts = (grid_pts + 0.5) / 64.0 - 0.5

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(grid_pts)
    Path(voxel_out).parent.mkdir(parents=True, exist_ok=True)
    ok = o3d.io.write_point_cloud(voxel_out, pcd, write_ascii=True)
    if not ok:
        raise RuntimeError(f"Failed to write voxel point cloud: {voxel_out}")


def render_views(mesh_path, out_dir, num_views):
    tf_path = os.path.join(out_dir, "transforms.json")
    pngs = sorted([p for p in os.listdir(out_dir) if p.endswith(".png")]) if os.path.exists(out_dir) else []
    if os.path.exists(tf_path) and len(pngs) >= num_views:
        return

    ensure_blender()
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    views = []
    offset = (0.12345, 0.67890)
    for i in range(num_views):
        yaw, pitch = sphere_hammersley_sequence(i, num_views, offset)
        views.append({
            "yaw": float(yaw),
            "pitch": float(pitch),
            "radius": 2.0,
            "fov": float(40 / 180 * np.pi),
        })

    args = [
        BLENDER_PATH, "-b",
        "-P", "/workspace/TRELLIS/dataset_toolkits/blender_script/render.py",
        "--",
        "--views", json.dumps(views),
        "--object", mesh_path,
        "--resolution", "512",
        "--output_folder", out_dir,
        "--engine", "CYCLES",
        "--save_mesh",
    ]

    print("\n[RUN BLENDER]", " ".join(args[:6]), "...", flush=True)
    subprocess.run(args, check=True)

    if not os.path.exists(tf_path):
        raise RuntimeError(f"transforms.json not created for {out_dir}")


def make_conditioning(sample_meta_path, top100_root, conditioning_root, dinov2, tokenizer, text_model, device):
    e_img_path = os.path.join(conditioning_root, "e_img.pt")
    e_text_path = os.path.join(conditioning_root, "e_text.pt")
    if os.path.exists(e_img_path) and os.path.exists(e_text_path):
        return

    with open(sample_meta_path, "r") as f:
        meta = json.load(f)

    src_img_path = os.path.join(top100_root, meta["original_image"])
    prompt = meta["edit_prompt"]

    os.makedirs(conditioning_root, exist_ok=True)

    img = Image.open(src_img_path).convert("RGB")
    img_tf = transforms.Compose([
        transforms.Resize(518, interpolation=InterpolationMode.BICUBIC),
        transforms.CenterCrop(518),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406),
                             std=(0.229, 0.224, 0.225)),
    ])
    x = img_tf(img).unsqueeze(0).to(device)

    with torch.no_grad():
        feats = dinov2.forward_features(x)
        if "x_prenorm" in feats:
            e_img = feats["x_prenorm"][:, dinov2.num_register_tokens + 1:, :]
        elif "x_norm_patchtokens" in feats:
            e_img = feats["x_norm_patchtokens"]
        else:
            raise RuntimeError(f"Unexpected DINOv2 output keys: {list(feats.keys())}")

    torch.save(e_img.cpu(), e_img_path)

    tok = tokenizer(
        [prompt],
        padding="max_length",
        truncation=True,
        max_length=77,
        return_tensors="pt",
    )
    tok = {k: v.to(device) for k, v in tok.items()}

    with torch.no_grad():
        txt_out = text_model(**tok)
        e_text = txt_out.last_hidden_state

    torch.save(e_text.cpu(), e_text_path)


def copy_final_outputs(sample_root, sample_id):
    ss_src = os.path.join(sample_root, "ss_latents", "ss_enc_conv3d_16l8_fp16", f"{sample_id}.npz")
    slat_src = os.path.join(sample_root, "latents", "dinov2_vitl14_reg_slat_enc_swin8_B_64l8_fp16", f"{sample_id}.npz")
    e_img_src = os.path.join(sample_root, "conditioning", sample_id, "e_img.pt")
    e_text_src = os.path.join(sample_root, "conditioning", sample_id, "e_text.pt")

    for src, dst_name in [
        (ss_src, "z_ss_target.npz"),
        (slat_src, "z_slat_target.npz"),
        (e_img_src, "e_img.pt"),
        (e_text_src, "e_text.pt"),
    ]:
        if os.path.exists(src):
            dst = os.path.join(sample_root, dst_name)
            if not os.path.exists(dst):
                shutil.copy2(src, dst)


def process_sample(sample_dir, output_root, top100_root, dinov2, tokenizer, text_model, device, num_views):
    sample_id = os.path.basename(sample_dir)
    out_sample_root = os.path.join(output_root, sample_id)
    renders_dir = os.path.join(out_sample_root, "renders", sample_id)
    mesh_ply = os.path.join(renders_dir, "mesh.ply")
    voxel_ply = os.path.join(out_sample_root, "voxels", f"{sample_id}.ply")
    sample_meta_path = os.path.join(sample_dir, "meta.json")
    pseudo_glb = os.path.join(sample_dir, "pseudo.glb")

    os.makedirs(out_sample_root, exist_ok=True)

    meta_copy = os.path.join(out_sample_root, "meta.json")
    if not os.path.exists(meta_copy):
        shutil.copy2(sample_meta_path, meta_copy)

    write_metadata_and_instances(out_sample_root, sample_id)
    export_mesh_from_glb(pseudo_glb, mesh_ply)
    voxelize_mesh(mesh_ply, voxel_ply)

    env = os.environ.copy()
    env["ATTN_BACKEND"] = "xformers"

    ss_target = os.path.join(out_sample_root, "ss_latents", "ss_enc_conv3d_16l8_fp16", f"{sample_id}.npz")
    if not os.path.exists(ss_target):
        run([
            "python", "dataset_toolkits/encode_ss_latent.py",
            "--output_dir", out_sample_root,
            "--model", "ss_vae_conv3d_16l8_fp16",
            "--enc_pretrained", "microsoft/TRELLIS-image-large/ckpts/ss_enc_conv3d_16l8_fp16",
            "--instances", os.path.join(out_sample_root, "instances.txt"),
        ], env=env)

    render_views(mesh_ply, renders_dir, num_views=num_views)

    feat_target = os.path.join(out_sample_root, "features", "dinov2_vitl14_reg", f"{sample_id}.npz")
    if not os.path.exists(feat_target):
        run([
            "python", "dataset_toolkits/extract_feature.py",
            "--output_dir", out_sample_root,
            "--instances", os.path.join(out_sample_root, "instances.txt"),
        ], env=env)

    slat_target = os.path.join(out_sample_root, "latents", "dinov2_vitl14_reg_slat_enc_swin8_B_64l8_fp16", f"{sample_id}.npz")
    if not os.path.exists(slat_target):
        run([
            "python", "dataset_toolkits/encode_latent.py",
            "--output_dir", out_sample_root,
            "--feat_model", "dinov2_vitl14_reg",
            "--enc_pretrained", "microsoft/TRELLIS-image-large/ckpts/slat_enc_swin8_B_64l8_fp16",
            "--instances", os.path.join(out_sample_root, "instances.txt"),
        ], env=env)

    make_conditioning(
        sample_meta_path=sample_meta_path,
        top100_root=top100_root,
        conditioning_root=os.path.join(out_sample_root, "conditioning", sample_id),
        dinov2=dinov2,
        tokenizer=tokenizer,
        text_model=text_model,
        device=device,
    )

    copy_final_outputs(out_sample_root, sample_id)

    print(f"\n[DONE] {sample_id}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_root", type=str, required=True)
    parser.add_argument("--output_root", type=str, required=True)
    parser.add_argument("--top100_root", type=str, required=True)
    parser.add_argument("--num_views", type=int, default=150)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device, flush=True)

    dinov2, tokenizer, text_model = load_models(device)

    sample_dirs = sorted([
        os.path.join(args.input_root, d)
        for d in os.listdir(args.input_root)
        if d.startswith("sample_") and os.path.isdir(os.path.join(args.input_root, d))
    ])

    if args.limit is not None:
        sample_dirs = sample_dirs[:args.limit]

    print("Found sample dirs:", len(sample_dirs), flush=True)
    for s in sample_dirs:
        print(" -", s, flush=True)

    failed = []
    for sample_dir in sample_dirs:
        try:
            process_sample(
                sample_dir=sample_dir,
                output_root=args.output_root,
                top100_root=args.top100_root,
                dinov2=dinov2,
                tokenizer=tokenizer,
                text_model=text_model,
                device=device,
                num_views=args.num_views,
            )
        except Exception as e:
            print(f"\n[FAILED] {sample_dir}: {e}", flush=True)
            failed.append((sample_dir, str(e)))

    print("\n===== SUMMARY =====", flush=True)
    print("Succeeded:", len(sample_dirs) - len(failed), flush=True)
    print("Failed:", len(failed), flush=True)
    for s, e in failed:
        print(" -", s, flush=True)
        print("   ", e, flush=True)


if __name__ == "__main__":
    main()
