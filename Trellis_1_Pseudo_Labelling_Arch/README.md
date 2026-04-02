# TRELLIS 1 Architecture-1: End-to-End README and Scaling Guide

*From filtered 30k edit-triplets to pseudo-3D targets, latent targets, smoke training, and scalable Stage 1 / Stage 2 DiT training*

## 1. What this project is trying to do

Architecture-1 adapts **TRELLIS 1** so that the two original Part-B DiT generators can be trained with **two conditioning streams at the same time**:

- **source image conditioning** from DINOv2
- **edit-prompt conditioning** from CLIP text tokens

The main idea is:

1. start from a filtered InstructPix2Pix-style dataset
2. select an **n-sample subset**
3. run **frozen TRELLIS 1** on each **edited image** to create a pseudo-3D target
4. convert that pseudo-3D target into TRELLIS-native latent targets
5. train the two TRELLIS 1 Part-B DiTs against those latent targets, while conditioning on the **source image** and **edit prompt**

This is not a direct image-editing pipeline. It is an **offline target-generation + latent-training pipeline**.

---

## 2. What is already working in this project

The current workflow has already established the following pieces:

- the original InstructPix2Pix-style data was filtered into a **30k filtered dataset**
- the filtered dataset was reduced to a **Top-100 working subset**
- frozen TRELLIS 1 inference was run on edited images to create **pseudo.glb** and **pseudo.ply**
- the pseudo assets were converted into:
  - `z_ss_target.npz`
  - `z_slat_target.npz`
  - `e_img.pt`
  - `e_text.pt`
- a custom Architecture-1 patch was created so TRELLIS `train.py` could read these new sample folders
- **Stage 1 smoke training** and **Stage 2 smoke training** were both completed successfully
- the next logical step is to **scale the exact same recipe** from 10 samples → 100 samples → larger n

In other words, the engineering path is no longer hypothetical. A working end-to-end proof of concept already exists.

---

## 3. The big picture in one diagram

```text
filtered_dataset (30k)
    ↓
create n-sample subset
    ↓
subset folder with original_images/, edited_images/, subset_metadata.jsonl
    ↓
run frozen TRELLIS 1 on edited images
    ↓
pseudo-3D assets per sample
    ↓
convert pseudo assets into TRELLIS-native training files
    ↓
z_ss_target.npz + z_slat_target.npz + e_img.pt + e_text.pt
    ↓
install Architecture-1 patch into fresh TRELLIS
    ↓
Stage 1 smoke training
    ↓
Stage 2 smoke training
    ↓
proof-of-concept training
    ↓
scale to larger n
```

---

## 4. The clean repo structure

This section has been updated so it matches the **current cleaned repo layout** you assembled locally.

```text
Trellis_1_Pseudo_Labelling_Arch/
├─ README.md
├─ .gitignore
├─ docs/
│  ├─ 01_end_to_end_story.md
│  ├─ 02_dataset_and_pseudolabel_creation.md
│  ├─ 03_stage1_stage2_training.md
│  ├─ 04_arch1_runs_and_debugging_timeline.md
│  ├─ 05_results_presentation_and_reviewer_notes.md
│  ├─ 06_git_and_preservation_guide.md
│  └─ 07_limitations_and_next_steps.md
├─ configs/
│  ├─ arch1_stage1_smoke.json
│  ├─ arch1_stage2_smoke.json
│  ├─ arch1_stage1_poc.json
│  └─ arch1_stage2_poc.json
├─ scripts/
│  ├─ pix2pix_filtering/
│  │  ├─ stage2_3_runpod.py
│  │  ├─ stage2_3_runpod_v2.py
│  │  └─ stage3_fast.py
│  ├─ subset_creation/
│  │  ├─ make_top100.py
│  │  ├─ make_top100_fixed.py
│  │  └─ make_n_subset.py
│  ├─ preprocessing/
│  │  ├─ arch1_generate_pseudo_assets.py
│  │  └─ arch1_batch_driver.py
│  ├─ training/
│  │  ├─ run_stage1_smoke.sh
│  │  ├─ run_stage2_smoke.sh
│  │  ├─ run_stage1_scale.sh
│  │  └─ run_stage2_scale.sh
│  └─ utilities/
│     └─ extract_total_loss_from_tensorboard.py
├─ patches/
│  └─ arch1_training_patch/
│     ├─ README.txt
│     ├─ install_arch1_training_patch.py
│     ├─ configs/
│     │  ├─ arch1_stage1_smoke.json
│     │  ├─ arch1_stage2_smoke.json
│     │  ├─ arch1_stage1_poc.json
│     │  └─ arch1_stage2_poc.json
│     └─ trellis/
│        ├─ datasets/
│        │  └─ arch1_editing.py
│        ├─ models/
│        │  └─ arch1_condition.py
│        └─ trainers/
│           └─ flow_matching/
│              ├─ arch1_flow_matching.py
│              └─ mixins/
│                 └─ arch1_conditioned.py
└─ artifacts/
   ├─ manifests/
   ├─ plots/
   └─ run_metadata/
      ├─ stage1_poc_500/
      └─ stage2_poc_500_fresh/
```

### Why this structure is cleaner

- `README.md` and `docs/` explain the project and the end-to-end workflow
- `configs/` contains the smoke and proof-of-concept training configs used by the project
- `scripts/` contains only scripts a user is expected to run for filtering, subset creation, preprocessing, training, and utility tasks
- `patches/arch1_training_patch/` contains only the custom TRELLIS patch bundle, separated from the rest of the repo
- `artifacts/` contains only small preserved outputs such as plots, manifests, and run metadata
- old machine snapshots are **not** part of the current cleaned repo; if you later want to preserve them, add an `archive/` folder separately instead of mixing them into the active workflow

## 5. What the two DiTs are

### Stage 1 DiT
- **Backbone:** `trellis/models/sparse_structure_flow.py`
- **Predicts:** `z_ss_target.npz["mean"]`
- **Meaning:** sparse structure / occupancy latent
- **Why it matters:** it learns where the object exists in sparse 3D space

### Stage 2 DiT
- **Backbone:** `trellis/models/structured_latent_flow.py`
- **Predicts:** `z_slat_target.npz["coords"]` and `z_slat_target.npz["feats"]`
- **Meaning:** structured latent carrying rich 3D appearance + geometry
- **Why it matters:** this is the latent later decoded by frozen TRELLIS decoders

### Important conceptual point
The DiTs are **not** trained on raw meshes or raw GLBs.

They are trained on **latent targets** produced offline by the frozen TRELLIS Part-A encoders.

---

## 6. What was changed relative to stock TRELLIS 1

Stock TRELLIS 1 supports image-conditioned or text-conditioned generation separately.

Architecture-1 changes the conditioning path so training uses both:

- `e_img.pt` from the **source image**
- `e_text.pt` from the **edit prompt**

The fusion used in the patch is:

1. project CLIP text tokens from **768 → 1024**
2. concatenate projected text tokens with DINOv2 image tokens
3. feed the resulting `e_joint` token sequence into both DiTs

So the target latents remain standard TRELLIS 1 latents, but the conditioning becomes **source image + edit prompt**.

---

## 7. Which files matter most

### Existing TRELLIS repo files used in the pipeline

#### Frozen inference / pseudo-target generation
- `trellis/pipelines/trellis_image_to_3d.py`
- `trellis/models/sparse_structure_flow.py`
- `trellis/models/structured_latent_flow.py`
- `trellis/models/structured_latent_vae/decoder_gs.py`
- `trellis/models/structured_latent_vae/decoder_rf.py`
- `trellis/models/structured_latent_vae/decoder_mesh.py`
- `trellis/utils/postprocessing_utils.py`
- `trellis/utils/render_utils.py`

#### Latent preprocessing
- `dataset_toolkits/voxelize.py`
- `dataset_toolkits/encode_ss_latent.py`
- `dataset_toolkits/render.py`
- `dataset_toolkits/blender_script/render.py`
- `dataset_toolkits/extract_feature.py`
- `dataset_toolkits/encode_latent.py`

#### Training
- `train.py`
- `trellis/models/sparse_structure_flow.py`
- `trellis/models/structured_latent_flow.py`
- `trellis/trainers/flow_matching/flow_matching.py`
- `trellis/trainers/flow_matching/sparse_flow_matching.py`
- `trellis/modules/sparse/basic.py`

### Custom Architecture-1 files

#### Patch installer
- `patches/arch1_training_patch/install_arch1_training_patch.py`

#### Patch usage note
- `patches/arch1_training_patch/README.txt`

#### Custom dataset loader
- `patches/arch1_training_patch/trellis/datasets/arch1_editing.py`

#### Text projection module
- `patches/arch1_training_patch/trellis/models/arch1_condition.py`

#### Custom trainer wrappers
- `patches/arch1_training_patch/trellis/trainers/flow_matching/arch1_flow_matching.py`

#### Conditioning mixin
- `patches/arch1_training_patch/trellis/trainers/flow_matching/mixins/arch1_conditioned.py`

#### User-facing training configs
- `configs/arch1_stage1_smoke.json`
- `configs/arch1_stage2_smoke.json`
- `configs/arch1_stage1_poc.json`
- `configs/arch1_stage2_poc.json`

#### Patch-bundle configs
- `patches/arch1_training_patch/configs/arch1_stage1_smoke.json`
- `patches/arch1_training_patch/configs/arch1_stage2_smoke.json`
- `patches/arch1_training_patch/configs/arch1_stage1_poc.json`
- `patches/arch1_training_patch/configs/arch1_stage2_poc.json`

#### Launch scripts
- `scripts/training/run_stage1_smoke.sh`
- `scripts/training/run_stage2_smoke.sh`
- `scripts/training/run_stage1_scale.sh`
- `scripts/training/run_stage2_scale.sh`

#### Preprocessing drivers
- `scripts/subset_creation/make_top100_fixed.py`
- `scripts/subset_creation/make_n_subset.py`
- `scripts/preprocessing/arch1_generate_pseudo_assets.py`
- `scripts/preprocessing/arch1_batch_driver.py`

---

## 8. Step-by-step workflow for a new user

## Step 1 — Start from the filtered dataset

The filtered dataset is assumed to already exist in a form like:

```text
/workspace/filtered_dataset/
├─ final_indices.json
├─ metadata.jsonl
├─ original_images/
├─ edited_images/
└─ filter_summary.json
```

### Important detail
`final_indices.json` stores **original dataset indices**, not row numbers inside `metadata.jsonl`.

That is why the corrected subset script matches through `original_dataset_index`.

---

## Step 2 — Create an n-sample subset

### For Top-100
Use:
- `scripts/subset_creation/make_top100_fixed.py`

### For arbitrary n
Keep or add:
- `scripts/subset_creation/make_n_subset.py`

### Recommended command
```bash
python scripts/subset_creation/make_n_subset.py   --data_dir /workspace/filtered_dataset   --n 500   --out_dir /workspace/subset_500
```

### Expected output
```text
/workspace/subset_500/
├─ subset_metadata.jsonl
├─ original_images/
└─ edited_images/
```

### What this step does
- reads `final_indices.json`
- takes the first `n` selected indices
- maps them back to `metadata.jsonl` using `original_dataset_index`
- writes a smaller metadata file
- copies only the required original and edited images into the subset folder

This creates a clean physical subset that is easy to hand over.

---

## Step 3 — Generate pseudo-3D assets from the edited images

### Script to use
- `scripts/preprocessing/arch1_generate_pseudo_assets.py`

### What it does
For each row in `subset_metadata.jsonl`, it:

- loads the **edited image**
- runs frozen TRELLIS 1
- saves:
  - `pseudo.glb`
  - `pseudo.ply`
  - `meta.json`

### Why the edited image is used
The pseudo-3D supervision target comes from the **edited target image**, not the source image.

The source image and edit prompt are kept aside for conditioning.

### Recommended command
```bash
cd /workspace/TRELLIS
export ATTN_BACKEND=xformers

python /path/to/scripts/preprocessing/arch1_generate_pseudo_assets.py   --subset_root /workspace/subset_500   --out_root /workspace/arch1_pseudo_500
```

### Expected output
```text
/workspace/arch1_pseudo_500/
├─ sample_000/
│  ├─ meta.json
│  ├─ pseudo.glb
│  └─ pseudo.ply
├─ sample_001/
│  ├─ meta.json
│  ├─ pseudo.glb
│  └─ pseudo.ply
└─ ...
```

### Important note
The DiTs are **not** trained directly on these raw pseudo assets.  
This is only the intermediate supervision stage.

---

## Step 4 — Convert pseudo assets into latent targets and conditioning files

### Script to use
- `scripts/preprocessing/arch1_batch_driver.py`

This is one of the most important project files. It bridges your project-specific `sample_xxx/` pseudo asset folders into the manifest-driven TRELLIS Part-A preprocessing path.

### Inputs
- `--input_root` → pseudo asset folder containing `sample_xxx/`
- `--output_root` → where the training-ready samples should be written
- `--top100_root` or subset root → where original images and metadata live

### What it creates per sample
```text
sample_xxx/
├─ meta.json
├─ z_ss_target.npz
├─ z_slat_target.npz
├─ e_img.pt
├─ e_text.pt
└─ ... intermediate folders used during preprocessing ...
```

### What each output means
- `z_ss_target.npz` → Stage 1 target latent
- `z_slat_target.npz` → Stage 2 target latent
- `e_img.pt` → DINOv2 conditioning tokens from the **source image**
- `e_text.pt` → CLIP text tokens from the **edit prompt**

### Internal pipeline inside `arch1_batch_driver.py`
For each `sample_xxx/`:

1. read `meta.json`
2. read `pseudo.glb`
3. export mesh geometry into `renders/sample_xxx/mesh.ply`
4. create minimal `metadata.csv` and `instances.txt`
5. voxelize mesh
6. run `encode_ss_latent.py` → `z_ss_target.npz`
7. render 150 views with Blender
8. run `extract_feature.py`
9. run `encode_latent.py` → `z_slat_target.npz`
10. load the source image and compute `e_img.pt`
11. tokenize the prompt and compute `e_text.pt`
12. copy the flat final files into the sample root

### Recommended command
```bash
cd /workspace/TRELLIS

python /path/to/scripts/preprocessing/arch1_batch_driver.py   --input_root /workspace/arch1_pseudo_500   --output_root /workspace/arch1_preprocessed_500   --top100_root /workspace/subset_500
```

### Training-ready file contract
The flat files at the root of each sample folder are the files that training actually consumes.

---

## Step 5 — Install the Architecture-1 patch into a fresh TRELLIS checkout

### Correct workflow
Do **not** depend on an old RunPod workspace dump.

Instead:

1. clone fresh upstream `microsoft/TRELLIS`
2. create the TRELLIS environment
3. copy the Architecture-1 patch bundle into the machine
4. run the patch installer

### Script to use
```bash
python patches/arch1_training_patch/install_arch1_training_patch.py
```

### What this installer does
It copies the custom dataset/model/trainer/config files into TRELLIS and patches the package `__init__.py` files so the new classes can be discovered by `train.py`.

### Installed custom files
- `trellis/datasets/arch1_editing.py`
- `trellis/models/arch1_condition.py`
- `trellis/trainers/flow_matching/arch1_flow_matching.py`
- `trellis/trainers/flow_matching/mixins/arch1_conditioned.py`
- the Architecture-1 smoke configs

---

## Step 6 — Understand how the image and text conditioning are combined

The image and text embeddings are **not** combined during preprocessing.

Preprocessing stores them separately:
- `e_img.pt`
- `e_text.pt`

The actual fusion happens during training in:
- `trellis/trainers/flow_matching/mixins/arch1_conditioned.py`

### Fusion rule
```python
e_text_proj = text_proj(e_text)      # 768 -> 1024
e_joint     = concat([e_img, e_text_proj], dim=1)
```

So the design keeps:
- frozen DINOv2 image tokens
- frozen CLIP text tokens
- one small trainable projection layer
- token concatenation instead of pooled-vector fusion

That keeps the architectural surgery small and fits TRELLIS’s token-sequence conditioning design.

---

## Step 7 — Run the smoke test

### Purpose of the smoke test
The smoke test is meant to prove:

- the dataset loader works
- the multimodal conditioning path works
- checkpoints save
- nothing crashes
- both DiT stages can complete a short run

### Existing smoke files
- `run_stage1_smoke.sh`
- `run_stage2_smoke.sh`
- `arch1_stage1_smoke.json`
- `arch1_stage2_smoke.json`

### What to change for a smoke test
Usually only:
- the `--data_dir` path in the shell script
- or the location of the preprocessed sample folders

### Better flexible wrappers
For long-term use, keep:
- `run_stage1_scale.sh`
- `run_stage2_scale.sh`

### Example Stage 1 smoke
```bash
bash scripts/training/run_stage1_scale.sh   /workspace/TRELLIS/configs/arch1_stage1_smoke.json   /workspace/arch1_preprocessed_smoke10   /workspace/arch1_runs/stage1_smoke
```

### Example Stage 2 smoke
```bash
bash scripts/training/run_stage2_scale.sh   /workspace/TRELLIS/configs/arch1_stage2_smoke.json   /workspace/arch1_preprocessed_smoke10   /workspace/arch1_runs/stage2_smoke
```

### What success looks like
- trainer starts
- loss logs appear
- TensorBoard event files are written
- checkpoints save
- both stages finish without crashing

---

## Step 8 — Move from smoke training to proof-of-concept training

Once smoke training works, create or use the proof-of-concept configs:

- `arch1_stage1_poc.json`
- `arch1_stage2_poc.json`

These should run longer than the 50-step smoke configs.

### Typical proof-of-concept settings
- `max_steps = 500`
- less frequent saves
- less frequent logging than smoke tests
- sampling kept disabled or very infrequent

### Example Stage 1 POC
```bash
bash scripts/training/run_stage1_scale.sh   /workspace/TRELLIS/configs/arch1_stage1_poc.json   /workspace/arch1_preprocessed_100   /workspace/arch1_runs/stage1_poc_500
```

### Example Stage 2 POC
```bash
bash scripts/training/run_stage2_scale.sh   /workspace/TRELLIS/configs/arch1_stage2_poc.json   /workspace/arch1_preprocessed_100   /workspace/arch1_runs/stage2_poc_500
```

---

## Step 9 — How the loss is calculated

Architecture-1 does **not** introduce a completely new loss.

It keeps TRELLIS 1’s standard flow-matching objective and changes only the conditioning path.

### What the patch changes
- load `e_img.pt`
- load `e_text.pt`
- project text tokens from 768 → 1024
- concatenate image tokens and projected text tokens
- feed the fused token sequence into the existing TRELLIS training stack

### Where the actual loss still comes from
- `trellis/trainers/flow_matching/flow_matching.py`
- `trellis/trainers/flow_matching/sparse_flow_matching.py`

### Practical meaning
- Stage 1 learns to predict `z_ss_target`
- Stage 2 learns to predict `z_slat_target`
- both are trained under the original TRELLIS flow-matching regime
- the main logged loss in TensorBoard is still the key training metric

### Plotting helper
Keep:
- `scripts/utilities/extract_total_loss_from_tensorboard.py`

---

## Step 10 — What the next user should do to scale the approach

The next user should **not** redesign the architecture first.

They should scale by repeating the same recipe on a larger `n`.

### Recommended scaling order
1. 10-sample smoke test
2. 10-sample proof of concept
3. 100-sample preprocessing
4. 100-sample training
5. only after that, move to 500 / 1000 / more

### What changes when scaling
- larger `n` in the subset creation step
- more pseudo assets
- more preprocessed sample folders
- longer training configs
- more careful storage and backup
- more attention to Stage 2 memory and sparse voxel counts

### What should stay unchanged
- the per-sample file contract
- the patch-based training integration
- the image + text fusion rule
- the two-stage TRELLIS 1 training order
- the use of offline pseudo-3D targets

### Recommended checks before training on larger data
For `N` samples, verify you have:
- `N` `z_ss_target.npz`
- `N` `z_slat_target.npz`
- `N` flat `e_img.pt`
- `N` flat `e_text.pt`

If those counts do not match, fix preprocessing before training.

---

## 11. Minimal runbook for a new user

### A. Create a subset
```bash
python scripts/subset_creation/make_n_subset.py   --data_dir /workspace/filtered_dataset   --n 100   --out_dir /workspace/subset_100
```

### B. Generate pseudo assets
```bash
cd /workspace/TRELLIS
export ATTN_BACKEND=xformers

python /path/to/scripts/preprocessing/arch1_generate_pseudo_assets.py   --subset_root /workspace/subset_100   --out_root /workspace/arch1_pseudo_100
```

### C. Convert pseudo assets into training-ready targets
```bash
cd /workspace/TRELLIS

python /path/to/scripts/preprocessing/arch1_batch_driver.py   --input_root /workspace/arch1_pseudo_100   --output_root /workspace/arch1_preprocessed_100   --top100_root /workspace/subset_100
```

### D. Install the patch into a fresh TRELLIS clone
```bash
python patches/arch1_training_patch/install_arch1_training_patch.py
```

### E. Run Stage 1 smoke
```bash
bash scripts/training/run_stage1_scale.sh   /workspace/TRELLIS/configs/arch1_stage1_smoke.json   /workspace/arch1_preprocessed_100   /workspace/arch1_runs/stage1_smoke
```

### F. Run Stage 2 smoke
```bash
bash scripts/training/run_stage2_scale.sh   /workspace/TRELLIS/configs/arch1_stage2_smoke.json   /workspace/arch1_preprocessed_100   /workspace/arch1_runs/stage2_smoke
```

### G. Move to proof of concept
```bash
bash scripts/training/run_stage1_scale.sh   /workspace/TRELLIS/configs/arch1_stage1_poc.json   /workspace/arch1_preprocessed_100   /workspace/arch1_runs/stage1_poc_500

bash scripts/training/run_stage2_scale.sh   /workspace/TRELLIS/configs/arch1_stage2_poc.json   /workspace/arch1_preprocessed_100   /workspace/arch1_runs/stage2_poc_500
```

---

## 12. Final advice for the cleaned repo

The repo should be a **clean orchestration and patch repo**, not a dead workspace dump.

### Keep in the repo
- README and docs
- patch files
- configs
- scripts users really run
- plotting helpers
- a few representative artifacts

### Do not keep in the repo
- full upstream TRELLIS copy
- giant dataset folders
- giant checkpoint folders
- whole cloud-machine snapshots

### Best handover model
A new user should be able to:
1. clone the clean repo
2. clone fresh upstream TRELLIS
3. run the preprocessing scripts
4. install the patch
5. run smoke training
6. scale to larger `n`

That is the cleanest and most reusable handover path.

---

## 13. One-sentence beginner explanation

This project first turns edited images into frozen pseudo-3D latent targets, then stores those targets together with source-image and edit-prompt conditioning files, then patches TRELLIS 1 so its two original DiT stages can read those files, fuse the two conditioning streams, and train against the latent targets.
