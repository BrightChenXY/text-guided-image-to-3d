# Text & Image to 3D — Volkan's Version

A cleaned-up README built from the uploaded notebooks.

![Workflow overview](assets/volkans_workflow_overview.png)

## What this project is

This notebook set is a **Colab-based Text/Image-to-3D workflow** built around **TRELLIS** and **TRELLIS.2**.

In plain language, the project does four things:

1. picks or uploads a 2D image,
2. optionally edits that image with an InstructPix2Pix-style prompt,
3. converts the final image into a 3D asset with TRELLIS or TRELLIS.2,
4. builds a small training pipeline that learns a **feature bridge** from `(original image + edit text)` to the edited-image feature space.

So this is not just one demo notebook. It is really a small pipeline with **data creation**, **3D generation**, **export**, **quality checks**, and **adapter training**.

---

## The notebook set at a glance

### `Text-_3D (7).ipynb`
This is the clearest **single-sample TRELLIS 1 workflow**.

It:
- previews images from the `timbrooks/instructpix2pix-clip-filtered` dataset,
- selects a source image,
- optionally applies a Pix2Pix edit,
- runs TRELLIS image-to-3D,
- exports 3D outputs,
- creates a more printable solid candidate.

A representative example from the notebook:

![Castle to mansion example](assets/trellis1_pix2pix_preview.png)

In that run:
- the source image is a Japanese-style castle,
- the edit prompt is **"Make the castle a mansion"**,
- the edited result is then prepared for TRELLIS.

### `Text-_3D_data_set (5).ipynb`
This notebook is the **dataset-building version** of the pipeline.

It is focused on:
- selecting dataset items or uploaded images,
- deciding whether to use the original or edited image,
- optionally skipping Pix2Pix,
- saving final inputs and metadata,
- building teacher-pair samples under a dataset root,
- appending sample metadata into `index.jsonl`,
- running a sweep over many dataset items.

A sample final input from that notebook:

![Dataset sample preview](assets/trellis1_dataset_final_preview_1.png)

### `Text-_3D_train.ipynb`
This is the **master workflow notebook**.

It combines three modes:
- `single_sample`
- `dataset_sweep`
- `adapter_train`

The important part is `adapter_train`, which trains a **residual feature bridge** using CLIP features.

This notebook:
- builds supervised CLIP features from the dataset,
- trains a residual bridge,
- compares bridge variants,
- selects a checkpoint,
- publishes a deploy package,
- runs smoke tests and sanity checks,
- saves a final deployment summary.

### `Text-_3D_train_v1.ipynb`
This is an earlier training-focused version of the same idea.

It overlaps heavily with `Text-_3D_train.ipynb`, but it stops earlier and has fewer deployment/inference follow-up cells.

### `text-_3D_t2 (4).ipynb`
This is the **TRELLIS.2 environment and testing notebook**.

It:
- creates a fresh `trellis2` conda environment,
- clones `microsoft/TRELLIS.2`,
- installs rendering/build dependencies,
- runs import smoke tests,
- runs TRELLIS.2 on sample images,
- tests an image-edit-to-3D path,
- audits geometry quality,
- attempts mesh repair/export.

The same castle-style example appears again here as an image-pair workflow:

| Source | Edited |
|---|---|
| ![Source](assets/trellis2_ip2p_pair_1.png) | ![Edited](assets/trellis2_ip2p_pair_2.png) |

---


## Installation guide

This project is written like a **Google Colab + GPU + Miniforge** workflow. The commands in the notebooks assume:

- a CUDA-capable NVIDIA GPU,
- `/usr/local/miniforge` exists,
- your working directory is under `/content`,
- and project data is stored in Google Drive under `/content/drive/MyDrive/trellis_project`.

The notebooks were built around **Python 3.10**, **PyTorch 2.8.0**, and **CUDA 12.8 wheels**.

### Before you start

Recommended baseline:
- Google Colab with GPU runtime, or a Linux machine with CUDA
- Miniforge/Conda installed
- Git and Git LFS available
- enough disk for model caches, outputs, and dataset files

If you are using Colab, mount Google Drive first and create the expected project folders:

```python
from google.colab import drive
drive.mount('/content/drive')
```

```bash
mkdir -p /content/drive/MyDrive/trellis_project/{hf_cache,outputs,data,models,logs,notebooks,results,datasets}
```

### Install path A — TRELLIS 1 workflow

Use this for:
- `Text-_3D (7).ipynb`
- `Text-_3D_data_set (5).ipynb`
- `Text-_3D_train.ipynb`
- `Text-_3D_train_v1.ipynb`

Clone the repository:

```bash
cd /content
git clone --recurse-submodules https://github.com/microsoft/TRELLIS.git
```

Create the environment:

```bash
source /usr/local/miniforge/etc/profile.d/conda.sh
mamba create -y -n trellis -c conda-forge python=3.10
conda activate trellis
python -m pip install --upgrade pip
```

Install the pinned PyTorch and Kaolin stack used by the notebooks:

```bash
pip uninstall -y torch torchvision torchaudio kaolin xformers || true
pip install --no-cache-dir   torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0   --index-url https://download.pytorch.org/whl/cu128

pip install --no-cache-dir   kaolin==0.18.0   -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.8.0_cu128.html
```

Install the rest of the Python dependencies found in the notebooks:

```bash
pip install --no-cache-dir   easydict rembg onnxruntime transformers sentencepiece   diffusers accelerate safetensors trimesh scipy open3d plyfile

pip uninstall -y utils3d || true
pip install --no-cache-dir git+https://github.com/EasternJournalist/utils3d.git
```

Set the cache and runtime variables used by the notebooks:

```bash
export HF_HOME=/content/drive/MyDrive/trellis_project/hf_cache
export TORCH_HOME=/content/drive/MyDrive/trellis_project/hf_cache
export SPCONV_ALGO=native
unset ATTN_BACKEND || true
```

Quick verification:

```bash
cd /content/TRELLIS
python - <<'PY'
import torch, kaolin, open3d as o3d
import diffusers, accelerate, trimesh, scipy
print('torch:', torch.__version__)
print('cuda:', torch.version.cuda)
print('cuda available:', torch.cuda.is_available())
print('kaolin ok')
print('open3d ok')
print('diffusers ok:', diffusers.__version__)
PY
```

### Install path B — TRELLIS.2 workflow

Use this for:
- `text-_3D_t2 (4).ipynb`

System packages:

```bash
apt-get update -y
apt-get install -y libjpeg-dev git git-lfs ffmpeg build-essential
```

Create the TRELLIS.2 environment:

```bash
source /usr/local/miniforge/etc/profile.d/conda.sh
conda create -n trellis2 python=3.10 -y
conda activate trellis2
pip install --upgrade pip setuptools wheel packaging ninja
```

Install the pinned PyTorch stack:

```bash
pip install --no-cache-dir   torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0   --index-url https://download.pytorch.org/whl/cu128
```

Clone TRELLIS.2:

```bash
rm -rf /content/TRELLIS2
git clone -b main --recursive https://github.com/microsoft/TRELLIS.2.git /content/TRELLIS2
```

Write the environment file used by the notebook:

```bash
cat > /content/drive/MyDrive/trellis_project/trellis2_env.sh <<'EOF'
export HF_HOME=/content/drive/MyDrive/trellis_project/hf_cache
export TORCH_HOME=/content/drive/MyDrive/trellis_project/hf_cache
export SPCONV_ALGO=native
export SPARSE_CONV_BACKEND=spconv
export SPARSE_ATTN_BACKEND=xformers
export ATTN_BACKEND=xformers
export PATH=/usr/local/nvidia/bin:/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/nvidia/lib:/usr/local/nvidia/lib64:/usr/lib64-nvidia:/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OPENCV_IO_ENABLE_OPENEXR=1
export CC=/usr/bin/gcc
export CXX=/usr/bin/g++
EOF
```

Install the core TRELLIS.2 Python dependencies:

```bash
source /content/drive/MyDrive/trellis_project/trellis2_env.sh
pip install --no-cache-dir   imageio imageio-ffmpeg tqdm easydict opencv-python-headless   trimesh transformers gradio==6.0.1 tensorboard pandas   lpips zstandard kornia timm

pip install --no-cache-dir   git+https://github.com/EasternJournalist/utils3d.git@9a4eb15e4021b67b12c460c7057d642626897ec8

pip uninstall -y pillow || true
CC=cc pip install --no-cache-dir pillow-simd
```

Install the rendering and sparse dependencies used in the notebook:

```bash
git clone -b v0.4.0 https://github.com/NVlabs/nvdiffrast.git /tmp/nvdiffrast
pip install /tmp/nvdiffrast --no-build-isolation

git clone -b renderutils https://github.com/JeffreyXiang/nvdiffrec.git /tmp/nvdiffrec
pip install /tmp/nvdiffrec --no-build-isolation

git clone --recursive https://github.com/JeffreyXiang/CuMesh.git /tmp/CuMesh
pip install /tmp/CuMesh --no-build-isolation

git clone --recursive https://github.com/JeffreyXiang/FlexGEMM.git /tmp/FlexGEMM
pip install /tmp/FlexGEMM --no-build-isolation

pip install /content/TRELLIS2/o-voxel --no-build-isolation
pip install spconv-cu124==2.3.8

pip uninstall -y xformers || true
pip install --no-cache-dir --force-reinstall   torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0   --index-url https://download.pytorch.org/whl/cu128
pip install --no-cache-dir --no-deps   xformers==0.0.32.post2   --index-url https://download.pytorch.org/whl/cu128
```

Quick verification:

```bash
source /usr/local/miniforge/etc/profile.d/conda.sh
conda activate trellis2
source /content/drive/MyDrive/trellis_project/trellis2_env.sh
export PYTHONPATH=/content/TRELLIS2:${PYTHONPATH:-}

python - <<'PY'
import importlib, torch
mods = [
    'xformers',
    'spconv.pytorch',
    'nvdiffrast.torch',
    'cumesh',
    'flex_gemm',
    'o_voxel',
    'trellis2',
    'trellis2.pipelines',
]
print('torch:', torch.__version__)
print('cuda:', torch.version.cuda)
print('cuda available:', torch.cuda.is_available())
for m in mods:
    importlib.import_module(m)
    print('OK:', m)
PY
```

### Installation notes

- These commands are reconstructed from the notebooks, so they match the uploaded workflow closely.
- The setup is **most reliable in Colab/Linux**. If you run locally, you will likely need to adapt paths and CUDA wheel versions.
- TRELLIS.2 is the more fragile environment because it compiles several rendering and sparse-convolution dependencies.
- If you only want the main single-sample and training flow, start with the **TRELLIS 1** environment first.

---

## The core workflow

![Workflow overview](assets/volkans_workflow_overview.png)

### Stage 1 — Input selection
The notebooks support two input modes:
- **upload**: bring your own image,
- **dataset**: pull a sample from the Pix2Pix dataset.

### Stage 2 — Optional image editing
When `RUN_IP2P_EDIT=True`, the selected source image is edited with an InstructPix2Pix pipeline.

Prompt modes include:
- dataset prompt,
- generic cleanup prompt,
- custom prompt.

### Stage 3 — 3D generation
The final RGBA image is passed into:
- **TRELLIS** in the TRELLIS 1 notebooks,
- **TRELLIS.2** in the `t2` notebook.

Typical exported artifacts include:
- `*.glb`
- `*.stl`
- `*.obj`
- `*.ply`
- `*.mp4`
- JSON metadata

### Stage 4 — Dataset creation
The pipeline can save teacher-pair style records such as:
- original 2D image,
- edited 2D image,
- edit instruction text,
- TRELLIS outputs,
- per-sample metadata,
- an `index.jsonl` registry.

### Stage 5 — Feature-bridge training
The training notebook learns a mapping from:

`[original_image_feature || edit_text_feature] -> edited_image_feature`

The selected deployment model is a **residual feature bridge**.

---

## What the training notebook found

### Feature extraction
The training run in `Text-_3D_train.ipynb` builds supervised CLIP features for **1015 samples**.

Shapes reported by the notebook:
- `X`: `(1015, 1536)`
- `Y`: `(1015, 768)`

This implies:
- the input is a concatenation of original-image and text features,
- the target is the edited-image feature.

### Residual bridge result
Across the multi-seed evaluation:
- mean validation cosine improved from about **0.8794** to **0.8874**
- mean cosine gain was about **+0.0080**
- top-1 retrieval slightly decreased on average

So the residual bridge is useful as a **feature-regression improvement**, even though it does not improve top-1 retrieval.

### Retrieval-aware bridge result
The retrieval-aware version showed:
- **top-1 retrieval gain** of about **+0.0392**
- but **cosine loss** of about **-0.1693**

That is a large regression in feature alignment, so it was not chosen as the main deployment model.

### Selected deployment model
The notebook ultimately selects:

- **model**: `residual_feature_bridge`
- **selection mode**: `cosine`

Selected improvement reported by the notebook:
- delta mean cosine: **+0.010820**
- delta top-1 retrieval: **+0.000000**

### Deploy smoke test
On a 12-sample smoke test:
- bridge cosine mean: **0.891326**
- baseline cosine mean: **0.884697**
- bridge top-1 hits: **12 / 12**
- baseline top-1 hits: **12 / 12**

So the deployed bridge improves cosine while leaving top-1 unchanged in the smoke test.

### Batch sanity check
On a 5-sample batch sanity check:
- bridge cosine mean: **0.912457**
- baseline cosine mean: **0.900602**
- cosine delta mean: **+0.011856**
- bridge top-1 hits: **4 / 5**
- baseline top-1 hits: **4 / 5**

The notebook’s own final conclusion is essentially:

> Residual bridge is the current deploy default. It improves mean cosine on sanity samples while keeping top-1 unchanged.

---

## What the TRELLIS.2 notebook found

The TRELLIS.2 notebook is valuable because it is not only a setup notebook. It also documents the practical problems you hit after generation.

### Good news
It successfully:
- builds the environment,
- imports TRELLIS.2,
- runs smoke tests,
- writes `glb` and `mp4` outputs,
- supports image-to-3D testing from edited inputs.

### Geometry warning
In the geometry-quality section, the combined mesh report shows:
- `faces`: **700,449**
- `watertight`: **False**
- `body_count`: **229,038**

That means the generated mesh is heavily fragmented.

### Repair warning
The attempted “keep largest connected component” cleanup reduces the mesh to:
- `faces`: **38**
- `body_count`: **1**

That technically isolates one component, but it destroys almost the whole mesh. So it should be treated as a **diagnostic experiment**, not the final repair strategy.

---

## Expected project structure

The notebooks consistently assume a Google Drive root like this:

```text
/content/drive/MyDrive/trellis_project/
├── hf_cache/
├── outputs/
├── data/
├── models/
├── logs/
├── notebooks/
├── results/
└── datasets/
```

Important subpaths used in the notebooks:

```text
datasets/trellis1_ip2p_teacher_pairs/
models/p2p_trellis_feature_bridge/
outputs/trellis1_ip2p_workflow/
outputs/trellis2_smoke/
outputs/trellis2_ip2p_test/
outputs/trellis2_source_geom_quality/
```

---

## Recommended run order

For a clean project story, the notebooks make the most sense in this order:

1. **`Text-_3D (7).ipynb`**  
   Understand the single-sample TRELLIS 1 workflow.

2. **`Text-_3D_data_set (5).ipynb`**  
   Build teacher-pair samples and dataset metadata.

3. **`Text-_3D_train.ipynb`**  
   Train, compare, select, and deploy the feature bridge.

4. **`text-_3D_t2 (4).ipynb`**  
   Move to TRELLIS.2 setup, testing, and geometry inspection.

`Text-_3D_train_v1.ipynb` looks like a transitional notebook and can be kept as an archive/reference version.

---

## What is strong in this project

- It is more than a demo; it is a real end-to-end workflow.
- It has both **generation** and **training** components.
- It stores outputs in a reproducible folder structure.
- It includes **dataset indexing** and **deployment packaging**.
- It tests both **TRELLIS** and **TRELLIS.2**.
- It records practical geometry issues instead of hiding them.

---

## What still needs cleanup

For a repo or thesis-quality release, I would clean up these parts next:

1. **Split environment setup from workflow logic**  
   The notebooks mix installation and modeling in the same linear flow.

2. **Turn repeated path/config blocks into one config file**  
   The same roots and flags appear in multiple notebooks.

3. **Separate “experiment” code from “production” code**  
   Geometry repair experiments should live in their own notebook or script.

4. **Standardize notebook names**  
   Current names look like working copies. Clear names would help:
   - `trellis1_single_sample.ipynb`
   - `trellis1_dataset_builder.ipynb`
   - `feature_bridge_training.ipynb`
   - `trellis2_setup_and_eval.ipynb`

5. **Document the bridge objective clearly**  
   The bridge is not directly producing 3D. It is improving edited-image feature prediction in CLIP space.

---

## One-sentence summary

This project is a **Volkan-style text/image-to-3D workflow** that starts from 2D images, optionally edits them with Pix2Pix, generates 3D assets with TRELLIS/TRELLIS.2, and then trains a CLIP-space residual bridge to better predict edited-image features from the original image plus edit text.
