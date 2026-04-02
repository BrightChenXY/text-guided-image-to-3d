
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
stage2_3_runpod_v2.py - GPU filtering using STREAMING mode
===========================================================
Uses HuggingFace streaming so the dataset is never fully loaded into RAM.

Upload stage1_kept_indices.json, then run:
    export HF_HOME=/workspace/hf_cache
    export HF_DATASETS_CACHE=/workspace/hf_cache/datasets
    export TMPDIR=/workspace/tmp
    nohup python stage2_3_runpod_v2.py > /workspace/output.log 2>&1 &

What this script does, in order:
1. Loads the Stage 1 kept indices that came from prompt-based filtering.
2. Runs Stage 2 on those candidates:
      - uses rembg to estimate whether both the source image and the edited image
        contain a reasonably clear foreground object
      - rejects images that are too small, too empty, or almost entirely foreground
3. Runs Stage 3 on the Stage 2 survivors:
      - uses CLIP to measure whether the edit is meaningful and aligned with the prompt
      - rejects examples where identity is lost, nothing visibly changed, or the edit
        does not match the text instruction well
4. Exports the final filtered dataset as:
      - original_images/
      - edited_images/
      - metadata.jsonl
      - filter_summary.json

Why this v2 script exists:
- 
- This version uses Hugging Face streaming mode, so it scans the dataset row by row
  instead of materializing everything in RAM at once.
"""

import json
import os
import sys
import time
import gc
import numpy as np
from pathlib import Path
from collections import Counter
from PIL import Image
from io import BytesIO

# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------
# STAGE1_FILE:
#   This is the JSON file produced by Stage 1 on the Windows machine.
#   It contains the list of original InstructPix2Pix dataset indices that
#   survived the prompt-based filtering step.
#
# OUTPUT_DIR:
#   The folder where the final filtered dataset will be written.
#
# TARGET_SIZE:
#   Even if more than this many samples survive Stage 3, keep only the best
#   TARGET_SIZE samples ranked by CLIP directional similarity.
# ---------------------------------------------------------------------
STAGE1_FILE = "stage1_kept_indices.json"
OUTPUT_DIR = Path("/workspace/filtered_dataset")
TARGET_SIZE = 30000

# ---------------------------------------------------------------------
# Stage 2 thresholds
# ---------------------------------------------------------------------
# MIN_FG_RATIO / MAX_FG_RATIO:
#   After running rembg, I estimate what fraction of the image belongs to
#   foreground. I want a real object to occupy a reasonable amount of the frame:
#   - too small  -> mostly background, probably not a clean object image
#   - too large  -> often a crop/texture/failed mask, not a clean object silhouette
#
# MIN_IMAGE_SIZE:
#   Reject images whose width or height is below 256 pixels.
# ---------------------------------------------------------------------
MIN_FG_RATIO = 0.08
MAX_FG_RATIO = 0.92
MIN_IMAGE_SIZE = 256

# ---------------------------------------------------------------------
# Stage 3 thresholds
# ---------------------------------------------------------------------
# MIN_CLIP_IMG_SIM:
#   If source and target image CLIP embeddings are too dissimilar, the edit
#   likely changed the object too much and identity may be lost.
#
# MAX_CLIP_IMG_SIM:
#   If source and target are too similar, almost no visible edit happened.
#
# MIN_CLIP_DIR_SIM:
#   Measures how well the visual change direction matches the text prompt.
#   If it is too low, the edit likely does not align with the instruction.
# ---------------------------------------------------------------------
MIN_CLIP_IMG_SIM = 0.70
MAX_CLIP_IMG_SIM = 0.98
MIN_CLIP_DIR_SIM = 0.08


def main():
    """
    Run the full streaming Stage 2 + Stage 3 pipeline.

    High-level flow:
    1. Load Stage 1 candidate indices.
    2. Stream the public InstructPix2Pix dataset and run Stage 2 foreground checks.
    3. Save Stage 2 survivors as a checkpoint.
    4. Stream the dataset again and run Stage 3 CLIP scoring on those survivors.
    5. Rank, trim, and save the final selected indices.
    6. Stream the dataset one last time and export the actual image files +
       metadata manifest.
    7. Print a summary and total runtime.

    Note:
    Because the dataset is streamed three times (once for Stage 2, once for
    Stage 3, once for export), the script is slower than the in-memory version,
    but it is much safer on limited-RAM machines.
    """
    start_time = time.time()
    print("=" * 60)
    print("  InstructPix2Pix Filtering: Stages 2 + 3 (streaming)")
    print("  For TRELLIS 1 Fine-Tuning")
    print("=" * 60)
    sys.stdout.flush()

    # -----------------------------------------------------------------
    # Load Stage 1 indices
    # -----------------------------------------------------------------
    # I convert the JSON list into a set for fast membership checks:
    #     if idx not in s1_indices: continue
    # This matters because the streaming dataset iterates through all rows
    # sequentially, so I need membership tests to be O(1).
    # -----------------------------------------------------------------
    print("\nLoading Stage 1 indices...")
    with open(STAGE1_FILE) as f:
        s1_indices = set(json.load(f))
    print("  {:,} indices to check".format(len(s1_indices)))
    sys.stdout.flush()

    # ===============================================================
    # STAGE 2: Foreground detection via rembg (streaming)
    # ===============================================================
    # Purpose:
    #   Keep only samples where BOTH the original image and the edited image
    #   appear to contain a reasonably clear foreground object.
    #
    # Logic:
    #   For each candidate index:
    #   - load the source and edited images
    #   - reject if either image is too small
    #   - resize to 256x256 for fast masking
    #   - run rembg to get a foreground mask
    #   - compute foreground ratio = fraction of pixels marked as foreground
    #   - reject if foreground is too small or too large
    #
    # Output:
    #   s2_kept = list of original dataset indices that pass Stage 2
    # ===============================================================
    print("\n" + "=" * 60)
    print("  STAGE 2: Foreground Detection (streaming)")
    print("=" * 60)
    sys.stdout.flush()

    from rembg import remove, new_session
    try:
        # Try the stronger BiRefNet model first.
        session = new_session("birefnet-general")
        print("  Using birefnet-general model")
    except Exception:
        # If that fails, fall back to u2net instead of stopping the whole run.
        session = new_session("u2net")
        print("  Using u2net model (fallback)")
    sys.stdout.flush()

    from datasets import load_dataset

    print("  Streaming dataset (no full download needed)...")
    sys.stdout.flush()

    s2_kept = []
    s2_reject = Counter()
    processed = 0
    total = len(s1_indices)

    # Stream through the source dataset row by row.
    dataset = load_dataset(
        "timbrooks/instructpix2pix-clip-filtered",
        split="train",
        streaming=True,
    )

    for idx, sample in enumerate(dataset):
        # Skip rows that Stage 1 already rejected.
        if idx not in s1_indices:
            continue

        processed += 1

        try:
            passed = True

            # Both images must pass the foreground test.
            for img_key in ["original_image", "edited_image"]:
                img = sample[img_key]

                # Ensure we are working with a PIL image object.
                if not isinstance(img, Image.Image):
                    img = Image.open(img).convert("RGB")

                # Basic resolution sanity check.
                w, h = img.size
                if w < MIN_IMAGE_SIZE or h < MIN_IMAGE_SIZE:
                    s2_reject["too_small"] += 1
                    passed = False
                    break

                # Resize before rembg to save time.
                # Stage 2 only needs a coarse foreground estimate, not full-res masking.
                img_small = img.resize((256, 256))

                # only_mask=True returns the foreground mask instead of a composited RGBA image.
                mask = remove(img_small, session=session, only_mask=True)
                mask_arr = np.array(mask)

                # Estimate what fraction of the image is foreground.
                fg_ratio = (mask_arr > 128).sum() / mask_arr.size

                # Reject if object is too tiny in the frame.
                if fg_ratio < MIN_FG_RATIO:
                    s2_reject["fg_too_small"] += 1
                    passed = False
                    break

                # Reject if almost the whole frame is foreground.
                if fg_ratio > MAX_FG_RATIO:
                    s2_reject["fg_too_large"] += 1
                    passed = False
                    break

            # Keep only samples where BOTH images passed.
            if passed:
                s2_kept.append(idx)

        except Exception as e:
            # Any failure on this sample is counted and skipped.
            s2_reject["error"] += 1

        # Print progress every 1000 processed candidates.
        if processed % 1000 == 0:
            print("  Stage 2: {:,}/{:,} processed, {:,} kept".format(
                processed, total, len(s2_kept)))
            sys.stdout.flush()

            # Encourage Python to release temporary memory during long runs.
            gc.collect()

    print("\n  Stage 2 RESULTS:")
    print("    Input:    {:,}".format(total))
    print("    Kept:     {:,}  ({:.1f}%)".format(len(s2_kept), 100*len(s2_kept)/total))
    print("    Rejected: {:,}".format(total - len(s2_kept)))
    for reason, count in s2_reject.most_common(10):
        print("      {:20s}  {:>7,}".format(reason, count))
    sys.stdout.flush()

    # Save Stage 2 checkpoint so I have the post-rembg survivors on disk.
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_DIR / "stage2_kept_indices.json", "w") as f:
        json.dump(s2_kept, f)
    print("  Saved Stage 2 indices")
    sys.stdout.flush()

    # Free the rembg session before moving to Stage 3.
    del session
    gc.collect()

    # ===============================================================
    # STAGE 3: CLIP scoring (streaming)
    # ===============================================================
    # Purpose:
    #   Measure whether the edit pair is actually good training data.
    #
    # What is scored:
    #   src_feat = CLIP embedding of original image
    #   tgt_feat = CLIP embedding of edited image
    #   txt_feat = CLIP embedding of edit prompt
    #
    # Signals:
    #   img_sim = similarity(source, target)
    #       - too low  -> identity likely lost
    #       - too high -> almost no visible change happened
    #
    #   direction = target - source in CLIP space
    #   dir_sim = similarity(direction, prompt)
    #       - low -> the visual change does not match the text instruction
    #
    # Output:
    #   kept = list of scored samples that pass all Stage 3 thresholds
    # ===============================================================
    print("\n" + "=" * 60)
    print("  STAGE 3: CLIP Quality Scoring (streaming)")
    print("=" * 60)
    sys.stdout.flush()

    import torch
    import open_clip

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("  Device: {}".format(device))

    # Load CLIP model + preprocessing.
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="openai"
    )
    tokenizer = open_clip.get_tokenizer("ViT-B-32")
    model = model.to(device).eval()

    s2_set = set(s2_kept)
    scored = []
    processed = 0
    total_s3 = len(s2_kept)

    # --------------------------------------------------------------
    # Batch buffers
    # --------------------------------------------------------------
    # CLIP runs much faster in GPU batches than one sample at a time.
    # So I collect source images, target images, and prompts into small
    # batches and score them together.
    # --------------------------------------------------------------
    batch_indices = []
    batch_src = []
    batch_tgt = []
    batch_prompts = []
    BATCH_SIZE = 32

    def process_batch():
        """
        Score one accumulated CLIP batch and append results to `scored`.

        For each sample in the batch, compute:
        - img_sim: similarity between source and target images
        - dir_sim: similarity between visual edit direction and prompt text

        The results are stored as dictionaries so the later filtering step
        can inspect both metrics and keep the original dataset index.
        """
        nonlocal scored
        if not batch_src:
            return
        try:
            with torch.no_grad():
                src_feat = model.encode_image(torch.stack(batch_src).to(device))
                tgt_feat = model.encode_image(torch.stack(batch_tgt).to(device))
                txt_feat = model.encode_text(tokenizer(batch_prompts).to(device))

                # Normalize features so dot products behave like cosine similarity.
                src_feat = src_feat / src_feat.norm(dim=-1, keepdim=True)
                tgt_feat = tgt_feat / tgt_feat.norm(dim=-1, keepdim=True)
                txt_feat = txt_feat / txt_feat.norm(dim=-1, keepdim=True)

                # Similarity between source and edited image.
                img_sim = (src_feat * tgt_feat).sum(dim=-1)

                # Direction of change in CLIP space.
                direction = tgt_feat - src_feat
                direction = direction / (direction.norm(dim=-1, keepdim=True) + 1e-8)

                # How well that visual change matches the text edit prompt.
                dir_sim = (direction * txt_feat).sum(dim=-1)

            for i, bidx in enumerate(batch_indices):
                scored.append({
                    "index": bidx,
                    "img_sim": img_sim[i].item(),
                    "dir_sim": dir_sim[i].item(),
                    "prompt": batch_prompts[i],
                })
        except Exception:
            # If a whole batch fails, just skip it and continue.
            pass

    print("  Streaming for CLIP scoring...")
    sys.stdout.flush()

    # Stream the dataset again, but now only keep Stage 2 survivors.
    dataset2 = load_dataset(
        "timbrooks/instructpix2pix-clip-filtered",
        split="train",
        streaming=True,
    )

    for idx, sample in enumerate(dataset2):
        if idx not in s2_set:
            continue

        processed += 1

        try:
            src = sample["original_image"]
            tgt = sample["edited_image"]
            if not isinstance(src, Image.Image):
                src = Image.open(src).convert("RGB")
            if not isinstance(tgt, Image.Image):
                tgt = Image.open(tgt).convert("RGB")

            batch_indices.append(idx)
            batch_src.append(preprocess(src))
            batch_tgt.append(preprocess(tgt))
            batch_prompts.append(sample["edit_prompt"])

            # Once the batch is full, score it and reset the buffers.
            if len(batch_indices) >= BATCH_SIZE:
                process_batch()
                batch_indices.clear()
                batch_src.clear()
                batch_tgt.clear()
                batch_prompts.clear()

        except Exception:
            continue

        if processed % 1000 == 0:
            print("  Stage 3: {:,}/{:,} scored".format(processed, total_s3))
            sys.stdout.flush()

    # Handle the final partial batch.
    process_batch()

    # --------------------------------------------------------------
    # Apply Stage 3 thresholds
    # --------------------------------------------------------------
    # Reject reasons:
    #   identity_lost    -> source and target too different
    #   no_visible_change-> source and target too similar
    #   edit_not_aligned -> target-source change does not match prompt
    # --------------------------------------------------------------
    s3_reject = Counter()
    kept = []
    for s in scored:
        if s["img_sim"] < MIN_CLIP_IMG_SIM:
            s3_reject["identity_lost"] += 1
        elif s["img_sim"] > MAX_CLIP_IMG_SIM:
            s3_reject["no_visible_change"] += 1
        elif s["dir_sim"] < MIN_CLIP_DIR_SIM:
            s3_reject["edit_not_aligned"] += 1
        else:
            kept.append(s)

    print("\n  Stage 3 RESULTS:")
    print("    Scored:   {:,}".format(len(scored)))
    print("    Passed:   {:,}".format(len(kept)))
    print("    Rejected: {:,}".format(len(scored) - len(kept)))
    for reason, count in s3_reject.most_common():
        print("      {:25s}  {:>7,}".format(reason, count))
    sys.stdout.flush()

    # --------------------------------------------------------------
    # Rank and trim
    # --------------------------------------------------------------
    # Among the surviving samples, keep the ones whose visual edit direction
    # aligns best with the prompt. Higher dir_sim means better edit-text match.
    # --------------------------------------------------------------
    kept.sort(key=lambda x: x["dir_sim"], reverse=True)
    if len(kept) > TARGET_SIZE:
        kept = kept[:TARGET_SIZE]
        print("  Trimmed to top {:,}".format(TARGET_SIZE))

    # NOTE:
    # The script stores final_indices as a set before saving.
    # That is convenient for fast lookup during export, but it also destroys
    # the ranking order in the saved JSON.
    final_indices = set(s["index"] for s in kept)
    final_prompts = {s["index"]: s["prompt"] for s in kept}

    # Save Stage 3 index list to disk.
    with open(OUTPUT_DIR / "stage3_final_indices.json", "w") as f:
        json.dump(list(final_indices), f)
    print("  Saved Stage 3 indices")
    sys.stdout.flush()

    # Free CLIP GPU memory before export.
    del model
    torch.cuda.empty_cache()
    gc.collect()

    # ===============================================================
    # EXPORT: Save filtered images (streaming)
    # ===============================================================
    # Purpose:
    #   Turn the final selected dataset indices back into a physical dataset:
    #   - save original image files
    #   - save edited image files
    #   - save metadata.jsonl that maps file paths + prompt + original index
    #
    # Because the dataset is streamed, the export step scans the source dataset
    # once more and saves only the final selected rows.
    # ===============================================================
    print("\n" + "=" * 60)
    print("  EXPORTING FILTERED DATASET")
    print("=" * 60)
    sys.stdout.flush()

    img_src_dir = OUTPUT_DIR / "original_images"
    img_tgt_dir = OUTPUT_DIR / "edited_images"
    img_src_dir.mkdir(parents=True, exist_ok=True)
    img_tgt_dir.mkdir(parents=True, exist_ok=True)

    metadata = []
    new_idx = 0
    exported = 0

    dataset3 = load_dataset(
        "timbrooks/instructpix2pix-clip-filtered",
        split="train",
        streaming=True,
    )

    for idx, sample in enumerate(dataset3):
        if idx not in final_indices:
            continue

        try:
            src = sample["original_image"]
            tgt = sample["edited_image"]
            if not isinstance(src, Image.Image):
                src = Image.open(src).convert("RGB")
            if not isinstance(tgt, Image.Image):
                tgt = Image.open(tgt).convert("RGB")

            src_path = "original_images/{:06d}.png".format(new_idx)
            tgt_path = "edited_images/{:06d}.png".format(new_idx)

            src.save(OUTPUT_DIR / src_path)
            tgt.save(OUTPUT_DIR / tgt_path)

            metadata.append({
                "id": new_idx,
                "original_dataset_index": idx,
                "edit_prompt": final_prompts.get(idx, sample["edit_prompt"]),
                "original_image": src_path,
                "edited_image": tgt_path,
            })
            new_idx += 1
            exported += 1

        except Exception:
            continue

        if exported % 1000 == 0:
            print("  Exported {:,}/{:,}".format(exported, len(final_indices)))
            sys.stdout.flush()

    # Save metadata manifest.
    with open(OUTPUT_DIR / "metadata.jsonl", "w") as f:
        for entry in metadata:
            f.write(json.dumps(entry) + "\n")

    # Save a compact summary for later reference.
    with open(OUTPUT_DIR / "filter_summary.json", "w") as f:
        json.dump({
            "total_samples": len(metadata),
            "source": "timbrooks/instructpix2pix-clip-filtered",
            "filtering": "Stage1(prompt) + Stage2(rembg) + Stage3(CLIP)",
        }, f, indent=2)

    elapsed = time.time() - start_time
    hours = int(elapsed // 3600)
    mins = int((elapsed % 3600) // 60)

    print("\n" + "=" * 60)
    print("  ALL DONE!")
    print("=" * 60)
    print("  Total time: {}h {}m".format(hours, mins))
    print("  Final dataset: {:,} triplets".format(len(metadata)))
    print("  Location: {}".format(OUTPUT_DIR))
    print("")
    print("  Next step: TRELLIS 1 pseudo-labeling")
    print("")
    sys.stdout.flush()


if __name__ == "__main__":
    main()
