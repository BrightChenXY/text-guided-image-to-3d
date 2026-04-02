# 02. Dataset Preparation and Pseudo-Label Creation

## 1. Why dataset preparation was necessary

TRELLIS 1 is not naturally trained on arbitrary in-the-wild scene edits. It is much better aligned with **single-object, object-centric** reconstruction. That means the raw 2D edit dataset could not simply be fed into training unchanged.

The dataset preparation stage was therefore one of the most important parts of the project.

## 2. Input data format

The input dataset format was the standard edit triplet:

- `src_img`: the original object image,
- `edit_prompt`: the natural-language editing instruction,
- `tgt_img`: the edited target image.

The architecture report explicitly states that this triplet format is the basis for Architecture 1.

## 3. Filtering strategy

The architecture note describes the intended filtering and preprocessing logic:

1. run background removal on source and target images,
2. keep the object foreground and suppress distracting scene background,
3. reject triplets where the object is too small,
4. reject poor background-removal results,
5. later reject pseudo-3D outputs that collapse into degenerate meshes.

This is important because TRELLIS pseudo-label quality is bounded by TRELLIS reconstruction quality. If the input image is not a clean object-centered image, the pseudo-3D label can become poor or unstable.

## 4. One-example validation

Before scaling, one sample was inspected and processed end-to-end. This single-example pass served several purposes:

- verify that the filtered triplet format was correct;
- confirm that background removal behaved sensibly;
- confirm that frozen TRELLIS inference could run on the edited target image;
- confirm that the resulting pseudo-3D asset could be encoded into the latent targets needed for training;
- verify the naming/layout expected by the dataset loaders.

This step dramatically reduced ambiguity before moving on to subset creation.

## 5. 100-sample subset and 10-sample working subset

After the single-sample check, a **100-sample subset** was created.

Then, for actual early training/debugging, the work intentionally used **10 examples** from that subset.

That means the project had two levels of dataset scale during development:

- a medium small subset (100) for controlled experimentation,
- a tiny operational subset (10) for smoke tests and proof-of-concept training.

This distinction is important for a reviewer, because the existence of 100 prepared samples does **not** mean all 100 were immediately used in the runs shown.

## 6. How the pseudo-3D assets were created for those 10 examples

For each of the small working-set examples, the conceptual preprocessing chain was:

### Step 1 — Prepare the target edited image

The edited target image was foreground-isolated using the project’s background-removal flow.

### Step 2 — Run frozen TRELLIS 1 image-to-3D inference

The architecture report describes this step as:

- DINOv2 encode,
- Stage 1 DiT,
- occupancy prediction,
- Stage 2 DiT,
- latent decoding,
- export of Gaussian / radiance-field / mesh style outputs.

In the context of this project, this step turned each 2D edited target image into a **pseudo-3D asset**.

### Step 3 — Encode that pseudo-3D asset to TRELLIS training latents

The pseudo-3D asset was then encoded into the two latent targets used by training:

- `z_ss_target` for Stage 1,
- `z_slat_target` for Stage 2.

The architecture note explicitly states that the frozen VAE encoders are used here, and that the output tensors are saved in the same style as TRELLIS Part A latent preprocessing.

### Step 4 — Save conditioning embeddings

The architecture note also states that each training example stores:

- `e_img.pt`: DINOv2 embedding of the **source image**,
- `e_text.pt`: CLIP text embedding of the **edit prompt**.

This point is extremely important:

> The image conditioning comes from the **source image**, not the target image.

That avoids leakage. The model must learn “source appearance + edit intent → target 3D latent,” not merely copy information from the target.

## 7. What these files mean conceptually

A single preprocessed training example therefore contained, conceptually:

- the source image,
- the target edited image,
- the edit prompt,
- the source-image embedding `e_img`,
- the prompt embedding `e_text`,
- the Stage 1 latent target,
- the Stage 2 latent target,
- and enough sparse metadata for Stage 2 sampling and batching.

In the actual project code path, `arch1_editing.py` was the dataset loader used for these examples. During debugging, it was clear that the loader consumed `e_img.pt` and `e_text.pt`, and for Stage 2 it also used sparse latent target data whose voxel count was needed for balanced sampling.

## 8. Why the pseudo-labeling strategy was preferable here

This project deliberately avoided supervising the model through direct rendering losses.

Instead, it reused TRELLIS 1’s latent-supervision framework. That choice brought several advantages:

- the training objective stayed close to TRELLIS 1’s native objective;
- the pseudo-label creation was expensive but **offline**;
- the online training loop stayed much simpler than differentiable rendering approaches;
- the proof-of-concept could focus on making the data path and training path work.

## 9. Reviewer summary

The key reviewer message is:

> The 10-example working set was not just “10 images.” Each example was converted into a TRELLIS-compatible latent-supervision package containing source-image conditioning, prompt conditioning, and pseudo-3D latent targets derived from the edited image through frozen TRELLIS reconstruction.

