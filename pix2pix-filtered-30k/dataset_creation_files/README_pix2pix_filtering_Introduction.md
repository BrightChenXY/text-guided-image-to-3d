# README: How I Filtered the InstructPix2Pix Dataset for TRELLIS 1

## Introduction

This document explains how I filtered the `timbrooks/instructpix2pix-clip-filtered` dataset to make it more suitable for my TRELLIS 1 training pipeline.

The original InstructPix2Pix dataset is very broad. It contains many different kinds of edits: object edits, scene edits, weather edits, background changes, lighting changes, photography-style effects, and other transformations. That variety is useful for general 2D image editing, but it is not automatically suitable for the kind of object-centric 3D work I want to do with TRELLIS 1.

My goal was not to keep every valid image-editing pair. My goal was to keep the subset that is most useful for an object-focused pipeline, where I eventually want to use:

- a **source image**,
- an **edit prompt**,
- and an **edited image**

as supervision for pseudo-3D target generation and later TRELLIS 1 fine-tuning.

Because of that, I designed the filtering pipeline around three ideas:

1. remove obviously irrelevant prompts early,
2. keep samples where the edit is visually meaningful,
3. keep samples where the edited image still looks like the same object and matches the prompt.

Originally, I planned a three-stage pipeline:

- **Stage 1**: prompt-based filtering,
- **Stage 2**: foreground-object filtering with background removal,
- **Stage 3**: CLIP-based edit quality scoring.

However, in the final workflow I used for the 30k dataset, I kept **Stage 1 + Stage 3** and discarded **Stage 2**.

This README explains exactly what each stage was intended to do, why Stage 2 was dropped, and how the final 30k subset was produced.

---

## Overall Filtering Philosophy

The core idea behind the filtering is simple:

- **Stage 1** uses only the text prompt and is very cheap.
- **Stage 2** tries to verify that the images contain a clear foreground object.
- **Stage 3** uses CLIP to judge whether the edit pair is actually useful.


I wanted Stage 1 to remove obviously bad candidates before I spent any GPU time on them. Then I wanted Stage 3 to be the final quality filter that decides whether a pair is semantically useful for training.

Stage 2 was meant to sit in the middle as a visual objectness check, but in practice it became the bottleneck.

---

## Stage 1: Prompt-Based Filtering

### What Stage 1 does

Stage 1 is a CPU-only filtering step. It reads the `edit_prompt` text from each InstructPix2Pix sample and decides whether the prompt looks like an **object-level edit** or a **scene-level edit**.

This stage does **not** look at the images at all. It only uses the prompt text.

The reason I put this stage first is that it is cheap, fast, and good at removing clearly irrelevant examples before doing any heavier processing.

### Why Stage 1 is needed

The source dataset contains many prompts that are not a good fit for object-centric TRELLIS training, for example:

- weather changes,
- time-of-day changes,
- landscape and environment edits,
- background replacement,
- photography or filter effects,
- adding unrelated scene elements like people, buildings, cars, or text.

Those kinds of edits may be valid for 2D editing, but they are not what I want when the downstream task is much closer to object editing and object-conditioned pseudo-3D generation.

### How Stage 1 works

Stage 1 uses a rule-based heuristic classifier:

- it rejects prompts that are too short or too long,
- it rejects prompts matching scene-level keywords,
- it rejects prompts matching scene-level regex patterns,
- it keeps prompts that match object-level keywords,
- it keeps prompts that match object-level regex patterns,
- and if a prompt is not clearly good or clearly bad, it keeps it as **ambiguous** so later stages can decide.

This design is intentional.

I did **not** want Stage 1 to be overly aggressive. I wanted it to eliminate the clearly bad samples while still passing borderline cases to a stronger semantic filter later.

### What Stage 1 kept and rejected conceptually

Examples of prompts that Stage 1 is meant to keep:

- material changes,
- color changes,
- shape or attribute changes,
- style changes applied to the object,
- condition changes like broken, glowing, dirty, polished, etc.

Examples of prompts that Stage 1 is meant to reject:

- make it rainy,
- make it sunset,
- replace the background,
- add people,
- add text,
- apply blur,
- make it look like a city street.

### What happened after Stage 1

The original dataset has **313,010** samples. After Stage 1, the kept set was **235,496** samples, which is about **75%** of the original dataset. This means Stage 1 removed roughly 78k clearly unsuitable prompt-image pairs before any GPU-heavy processing. 

I consider Stage 1 the broad pruning step. It does not guarantee that every remaining sample is good, but it reduces the search space to a much more reasonable set for later filtering.

---

## Stage 2: Foreground Detection with Background Removal

### What Stage 2 was intended to do

Stage 2 was designed to check whether both the **original image** and the **edited image** contain a reasonably clear foreground object.

This stage uses `rembg` to generate a foreground mask for each image. It then estimates how much of the image is foreground versus background.

The goal was to reject samples where:

- the object is too small,
- the image is mostly background,
- the segmentation mask covers almost the whole image,
- the image is not object-centric enough,
- or the pair looks like a scene-level edit even if the prompt sounded object-related.

### How Stage 2 works conceptually

For each Stage 1 candidate sample:

1. it loads the original image,
2. it loads the edited image,
3. it checks both images are at least `256 x 256`,
4. it resizes each image to `256 x 256` for speed,
5. it runs background removal using `rembg`,
6. it converts the mask to a foreground ratio,
7. it rejects the sample if the foreground is too small,
8. it rejects the sample if the foreground is too large,
9. it keeps the sample only if **both** images pass.

### Stage 2 thresholds

The Stage 2 script used these thresholds:

- `MIN_FG_RATIO = 0.08`
- `MAX_FG_RATIO = 0.92`
- `MIN_IMAGE_SIZE = 256`

This means:

- if less than 8% of the image looks like foreground, the object is probably too tiny,
- if more than 92% of the image looks like foreground, the image is probably not a clean object-centric example,
- if the image itself is too small, it is rejected immediately.

### Why Stage 2 sounded useful in theory

In theory, this stage made a lot of sense.

Stage 1 only understands prompt text. It cannot tell whether the actual image is a clean object shot or a messy scene. Stage 2 was supposed to fill that gap by checking the visual layout directly.

So the intended logic was:

- Stage 1: does the prompt sound object-related?
- Stage 2: do the images actually look object-centric?

### Why I did not keep Stage 2 in the final pipeline

Although Stage 2 was conceptually useful, I did **not** keep it in the final production workflow for the 30k dataset.

The reason is not that the idea was wrong. The reason is that, in practice, it became too slow and too expensive relative to its benefit.

When the combined Stage 2 + 3 pipeline was first run in the original non-streaming script, the job tried to load the entire dataset into memory and crashed at around 69% because the pod ran out of RAM. That is why the streaming rewrite (`stage2_3_runpod_v2.py`) was created. In the notes, this was explicitly described as the fix after the script failed because loading all 313k images into RAM exceeded the pod's available memory. 

However, even after moving to the streaming version, Stage 2 remained the main bottleneck. The rembg foreground filtering in streaming mode was effectively running on CPU and processing images one by one. In the recorded notes, this was described as **painfully slow**, with a rough estimate of **1 sample every 3–5 seconds**, and the conclusion was that it could take **days** to finish because it had to scan through the full dataset while applying background removal to all relevant candidates. 

That is the key reason Stage 2 was dropped.

In other words:

- the idea behind Stage 2 was reasonable,
- the implementation worked in principle,
- but the runtime cost was too high,
- and it blocked the rest of the pipeline.

Given the project constraints, it was better to skip Stage 2 and rely on Stage 1 + Stage 3.

### Why skipping Stage 2 was acceptable

Even without Stage 2, the pipeline still had two meaningful filters:

- Stage 1 removed obviously irrelevant prompt types,
- Stage 3 removed weak, misaligned, or low-value edit pairs using CLIP.

That means the final dataset was still not just random Stage 1 leftovers. It was a CLIP-scored subset with semantic quality control.

So Stage 2 was discarded for **practical reasons**, not because the entire filtering approach was abandoned.

---

## Stage 3: CLIP-Based Edit Quality Scoring

### What Stage 3 does

Stage 3 is the semantic quality filter.

This stage uses CLIP to score how good each edit triplet is. It compares:

- the original image,
- the edited image,
- and the edit prompt.

The purpose is to keep samples where:

- the edited image is still recognizably related to the source image,
- the edit actually produced a visible change,
- and the direction of that visual change matches the prompt.

### Why Stage 3 is important

Stage 1 tells me whether the prompt *sounds* like an object edit.

But that is not enough. A prompt might sound valid while the actual image pair is poor:

- the edit might be barely visible,
- the edit might completely destroy object identity,
- the prompt and the visual result might not match well,
- or the pair might simply be a weak supervision example.

Stage 3 is designed to catch exactly those cases.

### How Stage 3 works

For each surviving candidate sample:

1. CLIP encodes the original image,
2. CLIP encodes the edited image,
3. CLIP encodes the edit prompt,
4. the script computes image similarity between source and edited image,
5. it computes the edit direction in CLIP space,
6. it compares that direction with the prompt embedding,
7. it rejects low-quality samples,
8. it ranks the remaining samples by edit alignment score.

### Stage 3 thresholds

The scripts used the following thresholds:

- `MIN_CLIP_IMG_SIM = 0.70`
- `MAX_CLIP_IMG_SIM = 0.98`
- `MIN_CLIP_DIR_SIM = 0.08`

These thresholds are meant to enforce a balance.

#### 1. Identity preservation

If image similarity is **below 0.70**, the pair is rejected as `identity_lost`.

This means the source and edited images are too different, so the edit may have changed the object too aggressively or produced something unrelated.

#### 2. Visible edit requirement

If image similarity is **above 0.98**, the pair is rejected as `no_visible_change`.

This means the source and edited images are almost identical, so the edit pair is not useful for training because there is little or no real edit signal.

#### 3. Prompt alignment

If directional similarity is **below 0.08**, the pair is rejected as `edit_not_aligned`.

This means that the visual change from source to target does not align well with the prompt semantics.

### Why Stage 3 was kept in the final pipeline

Stage 3 is the strongest semantic filter in the whole pipeline.

Unlike Stage 2, it is GPU-friendly and batchable. That made it much more practical to run at scale.

Once Stage 2 was identified as the bottleneck, the fast alternative was to skip the foreground-removal step and keep Stage 3 as the main visual quality filter.

That is exactly what happened in `stage3_fast.py`.

This fast script:

- loads the Stage 1 kept indices,
- streams the source dataset,
- computes CLIP scores for the candidate samples,
- filters using the same Stage 3 thresholds,
- sorts the surviving samples by directional similarity,
- and then trims to the target size.

So even though Stage 2 was removed, Stage 3 still provided a meaningful second-stage filter instead of just exporting everything Stage 1 had kept.

---

## Why I Ultimately Used Only Stage 1 + Stage 3

The final workflow I used was:

1. **Stage 1** on CPU to remove clearly bad prompt categories.
2. **Stage 3** on GPU to keep semantically strong edit pairs.
3. Export the final subset.

I did **not** use Stage 2 in the final production run because it became the practical bottleneck.

The decision was based on the following tradeoff:

### If I kept Stage 2

- I would get an extra foreground-object check.
- But I would also add a huge runtime cost.
- In streaming mode, that cost became so high that it threatened to stall the project completely.

### If I skipped Stage 2

- I would lose one objectness filter.
- But I would still keep:
  - prompt-level filtering from Stage 1,
  - semantic edit filtering from Stage 3,
  - and later downstream pseudo-labeling quality checks.

So the final choice was not “use less filtering because filtering is unnecessary.”

The choice was:

**keep the filters that give the best quality-per-runtime tradeoff.**

That is why the final dataset was built with **Stage 1 + Stage 3**, not with all three stages.

---

## How the Final 30k Dataset Was Produced

### Step 1: Start from the original dataset

The starting point was the `timbrooks/instructpix2pix-clip-filtered` training split, which contains **313,010** samples.

Each sample contains:

- `original_image`
- `edit_prompt`
- `edited_image`

### Step 2: Apply Stage 1

Stage 1 prompt-based filtering reduced the dataset from **313,010** samples to **235,496** kept candidates.

This removed clearly irrelevant scene-level prompt categories while keeping object-related and ambiguous candidates.

### Step 3: Apply Stage 3 to the Stage 1 candidates

Using the fast CLIP-only version, the script scored the Stage 1 candidates and filtered them using:

- image similarity lower bound,
- image similarity upper bound,
- directional similarity lower bound.

Then it ranked the surviving samples by directional similarity (`dir_sim`) in descending order.

This means that samples with stronger prompt-aligned visual edits were ranked higher.

### Step 4: Trim to the target size

The script uses:

- `TARGET_SIZE = 30000`

After sorting the surviving candidates by edit quality, it keeps only the top **30,000**.

So the final 30k dataset is not just an arbitrary slice. It is the set of samples that:

- passed Stage 1 prompt filtering,
- passed Stage 3 CLIP filtering,
- and ranked high enough by directional similarity to remain in the top 30,000.

### Step 5: Export the dataset

The final export writes:

- `original_images/`
- `edited_images/`
- `metadata.jsonl`
- `filter_summary.json`

Each metadata row contains:

- `id`
- `original_dataset_index`
- `edit_prompt`
- `original_image`
- `edited_image`

That exported folder is the dataset I then use for the next stage of the TRELLIS 1 pipeline.

---

## Final Summary

To summarize the final logic in one place:

- **Stage 1** was used to remove clearly irrelevant prompt types cheaply on CPU.
- **Stage 2** was designed to filter by foreground object quality using rembg, but it was dropped because in practice it became too slow and too expensive in the streaming setup.
- **Stage 3** was kept because it provided strong semantic filtering and ranking using CLIP, while remaining practical to run on GPU.

So the final 30k dataset was produced by:

1. starting with 313,010 InstructPix2Pix samples,
2. reducing them to 235,496 candidates with Stage 1,
3. CLIP-scoring those candidates with Stage 3,
4. ranking by edit alignment quality,
5. trimming to the top 30,000,
6. and exporting the result as the filtered dataset.

That is the filtered subset I use as the training data foundation for the later TRELLIS 1 preprocessing and pseudo-3D supervision pipeline.
