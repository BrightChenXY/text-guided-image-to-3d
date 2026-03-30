# InstructPix2Pix LoRA Training Guide

[English](README_training.md) | [简体中文](README_training_zh.md)

This directory provides a minimal but runnable LoRA fine-tuning and inference pipeline for `timbrooks/instruct-pix2pix`, designed to improve the front-end image editing stage before sending images to TRELLIS `large:image`.

## Suggested Layout

```text
training/
  data/
    custom_train/
      images/
      metadata.jsonl
    custom_val/
      images/
      metadata.jsonl
  outputs/
    checkpoints/
    samples/
  dataset.py
  prepare_metadata.py
  train_lora_pix2pix.py
  infer_lora_pix2pix.py
  README_training.md
  README_training_zh.md
```

## 1. Dataset Layout

Both training and validation use `metadata.jsonl`. Each line should contain:

```json
{"original_image":"images/0001_input.jpg","edited_image":"images/0001_target.jpg","edit_prompt":"make it metallic blue with a clean background"}
```

Recommended structure:

```text
training/data/custom_train/
  images/
    0001_input.jpg
    0001_target.jpg
    0002_input.jpg
    0002_target.jpg
  metadata.jsonl
```

`original_image` and `edited_image` support both relative paths and Windows absolute paths. Relative paths are resolved relative to the corresponding `metadata.jsonl` file.

## 2. Quickly Build metadata.jsonl

If your files are named like this:

```text
chair_01_input.jpg
chair_01_target.jpg
chair_02_input.jpg
chair_02_target.jpg
```

You can scan and build metadata directly:

```bash
python training/prepare_metadata.py ^
  --source-dir training/raw/train ^
  --output-metadata training/data/custom_train/metadata.jsonl ^
  --default-prompt "make it metallic blue with a clean background"
```

You can also use a CSV or JSON manifest:

```bash
python training/prepare_metadata.py ^
  --manifest-csv training/raw/train_manifest.csv ^
  --output-metadata training/data/custom_train/metadata.jsonl
```

The CSV needs at least these three columns:

```text
original_image,edited_image,edit_prompt
```

The script copies paired images into the sibling `images/` directory next to `metadata.jsonl`, then writes relative paths into the metadata file.

## 3. Run Training

After installing dependencies, run the training script directly. Single-GPU and CPU are both supported, though single-GPU is the intended baseline.

Local JSONL mode:

```bash
python training/train_lora_pix2pix.py ^
  --train-metadata training/data/custom_train/metadata.jsonl ^
  --val-metadata training/data/custom_val/metadata.jsonl ^
  --output-dir training/outputs/checkpoints/pix2pix-lora
```

If you already use `accelerate`, you can launch with:

```bash
accelerate launch training/train_lora_pix2pix.py ^
  --train-metadata training/data/custom_train/metadata.jsonl ^
  --val-metadata training/data/custom_val/metadata.jsonl ^
  --output-dir training/outputs/checkpoints/pix2pix-lora
```

Hugging Face dataset mode:

```bash
accelerate launch training/train_lora_pix2pix.py ^
  --dataset-name timbrooks/instructpix2pix-clip-filtered ^
  --train-split train ^
  --original-image-column original_image ^
  --edited-image-column edited_image ^
  --edit-prompt-column edit_prompt ^
  --output-dir training/outputs/checkpoints/hf_baseline
```

If the dataset does not provide a dedicated validation split, you can automatically carve one out from the train split:

```bash
accelerate launch training/train_lora_pix2pix.py ^
  --dataset-name timbrooks/instructpix2pix-clip-filtered ^
  --train-split train ^
  --validation-from-train-ratio 0.05 ^
  --original-image-column original_image ^
  --edited-image-column edited_image ^
  --edit-prompt-column edit_prompt ^
  --output-dir training/outputs/checkpoints/hf_baseline
```

Use either `--val-split` or `--validation-from-train-ratio`, not both.

## 4. Minimal Training Commands

Quick pipeline validation run:

```bash
python training/train_lora_pix2pix.py ^
  --train-metadata training/data/custom_train/metadata.jsonl ^
  --val-metadata training/data/custom_val/metadata.jsonl ^
  --output-dir training/outputs/checkpoints/pix2pix-lora-baseline ^
  --resolution 256 ^
  --train-batch-size 1 ^
  --gradient-accumulation-steps 4 ^
  --learning-rate 5e-5 ^
  --max-train-steps 1000
```

More realistic training suggestion:

```bash
python training/train_lora_pix2pix.py ^
  --train-metadata training/data/custom_train/metadata.jsonl ^
  --val-metadata training/data/custom_val/metadata.jsonl ^
  --output-dir training/outputs/checkpoints/pix2pix-lora-512 ^
  --resolution 512 ^
  --train-batch-size 1 ^
  --gradient-accumulation-steps 8 ^
  --learning-rate 1e-4 ^
  --max-train-steps 3000
```

## 5. Recommended Starting Parameters

Baseline smoke test:

```text
resolution = 256
train_batch_size = 1
gradient_accumulation_steps = 4
learning_rate = 5e-5
max_train_steps = 1000
```

More serious training:

```text
resolution = 512
train_batch_size = 1
gradient_accumulation_steps = 4~8
learning_rate = 5e-5 or 1e-4
max_train_steps = 2000~5000
```

By default, the script appends this target-style suffix to every prompt:

```text
Keep a single centered object, clean background, clear silhouette, product-style view, suitable for 3D asset generation.
```

If you do not want the suffix appended automatically, pass:

```bash
--prompt-suffix ""
```

## 6. Training Outputs

Training outputs are stored by default in:

```text
training/outputs/checkpoints/pix2pix-lora/
```

Inside it:

```text
lora/
```

contains the final LoRA weights.

And:

```text
checkpoint-000500/
checkpoint-001000/
```

contain intermediate checkpoints and their corresponding LoRA weights.

Validation comparison images are stored in:

```text
training/outputs/samples/<run_name>/
```

Each preview image is arranged as:

```text
original | generated | target
```

TensorBoard logs are stored by default in:

```text
training/outputs/checkpoints/<run_name>/tensorboard/
```

You can also override the log directory with:

```bash
--tensorboard-log-dir training/outputs/tensorboard/custom_run
```

Launch TensorBoard with:

```bash
tensorboard --logdir training/outputs/checkpoints/pix2pix-lora/tensorboard
```

If you want to compare multiple runs at once, point TensorBoard to the parent directory:

```bash
tensorboard --logdir training/outputs/checkpoints
```

You will see these visualizations:

```text
train/loss
train/learning_rate
val/loss
val/previews/sample_00 ...
```

### 6.1 What To Watch In TensorBoard

`train/loss`

This is the denoising objective on training batches. In a healthy run, it should trend downward overall, even if it is noisy step to step.

`val/loss`

This is an approximate validation loss. It helps you judge whether the model is improving on unseen examples rather than only memorizing training samples. The script estimates it from the number of batches specified by `--validation-loss-batches`, which defaults to `8`.

`train/learning_rate`

This confirms your scheduler is behaving as expected, whether you use a constant LR, linear decay, cosine schedule, or something else.

`val/previews/sample_00`

These are validation preview strips. Each one is laid out as:

```text
original | generated | target
```

For your use case, these previews are often more important than the scalar losses, because your real goal is better front-end edits for TRELLIS, not just lower denoising loss.

### 6.2 Why There Is No Accuracy Curve

This is diffusion-based image editing, not classification or detection, so there usually is no meaningful accuracy curve to track.

For this task, the more useful signals are:

```text
whether loss trends down overall
whether validation previews get closer to the targets
whether the subject stays more stable
whether the background becomes cleaner
whether prompt control becomes more obvious
```

### 6.3 How To Tell If Training Is Improving

Good signs:

```text
train/loss gradually decreases
val/loss roughly follows or remains stable
validation previews show cleaner silhouettes and more centered subjects
background clutter decreases over time
material, color, and style instructions become easier to control
```

Warning signs:

```text
train/loss goes down while val/loss keeps rising
validation previews collapse toward a repeated training-style template
images become oversharpened, messy, or show duplicated objects
prompt edits get stronger, but subject structure gets worse
```

If that happens, common fixes are:

```text
lower the learning rate
train for fewer steps
add or clean your data
validate more often
check whether prompts are too repetitive
```

### 6.4 Suggested Review Cadence

For a smoke test, checking TensorBoard every `100~250` steps is usually enough.

For a more serious run, compare checkpoints around:

```text
step 500
step 1000
step 2000
step 3000
```

Do not only inspect the final checkpoint. The best LoRA often appears somewhere in the middle of training.

### 6.5 Useful Visualization Parameters

```bash
--validation-steps 250
--validation-loss-batches 8
--tensorboard-log-dir training/outputs/checkpoints/pix2pix-lora/tensorboard
```

If time or memory is tight, you can reduce validation overhead, for example:

```bash
--validation-steps 500
--validation-loss-batches 4
```

### 6.6 Full Example

```bash
accelerate launch training/train_lora_pix2pix.py ^
  --train-metadata training/data/custom_train/metadata.jsonl ^
  --val-metadata training/data/custom_val/metadata.jsonl ^
  --output-dir training/outputs/checkpoints/pix2pix-lora ^
  --resolution 256 ^
  --train-batch-size 1 ^
  --gradient-accumulation-steps 4 ^
  --learning-rate 5e-5 ^
  --max-train-steps 1000 ^
  --validation-steps 250 ^
  --validation-loss-batches 8
```

After training starts, open a second terminal and run:

```bash
tensorboard --logdir training/outputs/checkpoints/pix2pix-lora/tensorboard
```

Then open TensorBoard in your browser to inspect the curves and validation previews.

## 7. Inference Check

After training finishes, you can quickly validate LoRA behavior with:

```bash
python training/infer_lora_pix2pix.py ^
  --lora-path training/outputs/checkpoints/pix2pix-lora/lora ^
  --image assets/example.jpg ^
  --prompt "make it metallic blue with a clean background" ^
  --output outputs/edited/example_lora.png
```

Common inference knobs:

```text
--num-inference-steps 20
--guidance-scale 7.5
--image-guidance-scale 1.5
--lora-scale 1.0
```

## 8. Reconnect To The Demo

The current `pipelines/image_editor.py` already supports a lightweight LoRA extension, so you can switch the demo to the fine-tuned editor through environment variables:

```bash
set INSTRUCT_PIX2PIX_LORA_PATH=training\outputs\checkpoints\pix2pix-lora\lora
set INSTRUCT_PIX2PIX_LORA_SCALE=1.0
```

This keeps the TRELLIS protocol unchanged and only swaps the front-end image editor to the LoRA-tuned version.





## 9. TRELLIS Black-Box Rerank

The training script can now run a downstream TRELLIS-aware validation pass during regular validation.

When `--enable-trellis-rerank` is enabled, each validation cycle does this:

```text
validation example
-> current LoRA editor generates edited image
-> edited image is sent to TRELLIS
-> TRELLIS returns a GLB
-> the GLB is rendered into fixed canonical views
-> proxy metrics are computed
-> the best checkpoint is selected by trellis/mean_score
```

### 9.1 What Gets Saved

Inside your run directory you will now also see:

```text
best_checkpoint/
  lora/
  best_checkpoint.json

trellis_eval/
  step_000250/
    sample_00/
      edited_input.png
      generated.png
      original.png
      target.png
      trellis_result.glb
      render_front.png
      render_left.png
      render_right.png
      render_back.png
      render_top.png
      trellis_preview.png
      trellis_metrics.json
    summary.json
```

`best_checkpoint/lora/`

This is the downstream-best LoRA according to the current rerank score. For frontend testing, this is usually the directory you want to load first.

`best_checkpoint/best_checkpoint.json`

Stores the step, selected metric, validation loss, and TRELLIS summary used to pick the best checkpoint.

### 9.2 TensorBoard Metrics Added By Rerank

With rerank enabled, TensorBoard will also show:

```text
trellis/mean_score
trellis/success_rate
trellis/front_similarity
trellis/coverage_score
trellis/centering_score
trellis/view_consistency_score
trellis/previews/sample_00 ...
```

These are proxy metrics, not differentiable training losses. They are meant for model selection and checkpoint reranking.

### 9.3 Recommended Command

```bash
accelerate launch training/train_lora_pix2pix.py ^
  --dataset-name timbrooks/instructpix2pix-clip-filtered ^
  --train-split train ^
  --validation-from-train-ratio 0.05 ^
  --original-image-column original_image ^
  --edited-image-column edited_image ^
  --edit-prompt-column edit_prompt ^
  --output-dir training/outputs/checkpoints/hf_pix2pix_lora ^
  --resolution 256 ^
  --train-batch-size 1 ^
  --gradient-accumulation-steps 4 ^
  --learning-rate 5e-5 ^
  --max-train-steps 1000 ^
  --validation-steps 250 ^
  --validation-loss-batches 8 ^
  --enable-trellis-rerank ^
  --trellis-eval-samples 4 ^
  --trellis-render-size 256
```

### 9.4 Extra Dependencies

TRELLIS rerank rendering needs these packages in addition to the normal training stack:

```text
trimesh
pyrender
PyOpenGL
pyglet<2
```

### 9.5 Frontend Integration

Once training finishes, you can point the frontend directly at the best downstream checkpoint:

```bash
set INSTRUCT_PIX2PIX_LORA_PATH=training\outputs\checkpoints\hf_pix2pix_lora\best_checkpoint\lora
set INSTRUCT_PIX2PIX_LORA_SCALE=1.0
python app.py
```

If you want the final checkpoint instead, keep using `training/outputs/checkpoints/<run_name>/lora`.
