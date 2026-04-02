# 01. End-to-End Story of What Was Built

## 1. The problem this project tries to solve

The input available in this project is not a clean supervised 3D editing dataset. Instead, the starting point is a **2D editing dataset** with triplets:

- a source image,
- a text edit instruction,
- a target edited image.

The problem is to use this kind of 2D supervision to teach TRELLIS 1 to produce a 3D output that reflects the requested edit.

The central difficulty is that TRELLIS 1 was originally trained using 3D-derived latent supervision, not just image pairs. So the project needed a way to turn each edited 2D target image into something that looks like TRELLIS-style 3D supervision.

## 2. The core solution

The solution adopted here is **Architecture 1: pseudo-label flow-matching fine-tune**.

The architecture report explains that the frozen TRELLIS 1 pipeline can be used as a **3D pseudo-label factory**. The edited target image is reconstructed into a pseudo-3D asset by frozen TRELLIS 1. That pseudo-3D asset is then encoded into:

- a Stage 1 target latent (`z_ss_target`) for sparse structure,
- a Stage 2 target latent (`z_slat_target`) for structured appearance/geometry.

At training time, the model is conditioned on:

- the **source image embedding** (`e_img`),
- the **edit prompt embedding** (`e_text`),
- and, after projection and concatenation, the joint conditioning sequence `e_joint`.

This is the key bridge from 2D edit triplets to TRELLIS-compatible 3D latent supervision.

## 3. What happened in practice in this project

The work was done incrementally.

### Step A — Start from Pix2Pix-style edit triplets

The practical dataset source was a Pix2Pix / InstructPix2Pix-style triplet dataset. The important conceptual format was always:

- source image,
- edit instruction,
- edited target image.

### Step B — Filter it for object-centric TRELLIS use

The architecture plan assumes that TRELLIS 1 works best when the object is isolated and reconstructable. So the dataset had to be filtered so that samples were more likely to survive background removal and later TRELLIS reconstruction.

The architecture note states the intended filtering logic clearly:

- use background removal on source and target,
- reject samples where the foreground object is too small,
- reject poor background-removal cases,
- reject pseudo-3D reconstructions that collapse or degenerate.

In other words, the dataset was not treated as “all samples are equally valid.” It was explicitly pruned down to a more TRELLIS-friendly subset.

### Step C — Check one example first

Before scaling anything, a single example was checked end-to-end. That is an important engineering step because it answers the question:

> Can one triplet go all the way from 2D edit data to pseudo-3D latents without the pipeline breaking?

This single-example validation was used to confirm the logic before attempting larger subset preparation and model training.

### Step D — Create a 100-sample subset

After the one-example validation, a 100-sample subset was prepared. The purpose of the 100-sample subset was not to claim final-scale training, but to create a manageable sandbox for repeated preprocessing, loader validation, checkpointing, and training experiments.

### Step E — Train/debug on only 10 examples first

Although the 100-sample subset existed, the effective training/debugging work used only **10 datasets/examples** first.

This is not a weakness; it is a classic systems-development strategy:

- small enough to preprocess quickly,
- small enough to debug repeatedly,
- small enough to inspect manually,
- small enough that when something fails, the source of failure is easier to localise.

So the 10-example working subset was the first true proof-of-concept training target.

## 4. Why this approach matters

This work matters because it demonstrates the missing bridge between:

- abundant **2D edit supervision**, and
- TRELLIS 1’s **3D latent training framework**.

Instead of trying to invent a brand-new loss with differentiable rendering, this approach stays inside TRELLIS 1’s own training language: latent prediction with flow matching.

That makes the project technically conservative in a good way:

- it changes the data and conditioning,
- but it does not throw away TRELLIS 1’s structure.

## 5. What a reviewer should understand from this file

If a reviewer reads only this file, the main conclusion should be:

> This project transformed 2D edit triplets into TRELLIS-compatible pseudo-3D latent supervision, then fine-tuned TRELLIS 1’s two-stage latent generators with joint image + text conditioning, first on very small subsets to make the pipeline work end-to-end.

