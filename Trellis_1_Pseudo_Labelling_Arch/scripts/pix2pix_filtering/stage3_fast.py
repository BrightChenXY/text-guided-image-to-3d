
import json, sys, os, time, gc
import numpy as np
from collections import Counter
from PIL import Image

# ---------------------------------------------------------------------
# Force line-buffered output
# ---------------------------------------------------------------------
# When running long jobs through nohup / output.log, Python can buffer prints
# for a long time. This line forces line-buffered stdout so progress messages
# appear in the log file quickly and the run feels less "stuck".
# ---------------------------------------------------------------------
sys.stdout.reconfigure(line_buffering=True)

# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------
# STAGE1_FILE:
#   The Stage 1 output JSON containing the original dataset indices that
#   survived prompt filtering.
#
# OUTPUT_DIR:
#   Where the final exported filtered dataset will be written.
#
# TARGET_SIZE:
#   After CLIP filtering, keep at most the top TARGET_SIZE samples.
# ---------------------------------------------------------------------
STAGE1_FILE = "stage1_kept_indices.json"
OUTPUT_DIR = "/workspace/filtered_dataset"
TARGET_SIZE = 30000

# ---------------------------------------------------------------------
# Stage 3 thresholds
# ---------------------------------------------------------------------
# This script intentionally skips Stage 2 foreground masking completely.
# It only uses CLIP to score edit quality.
#
# MIN_CLIP_IMG_SIM:
#   Reject if source and target are too different -> identity likely lost.
#
# MAX_CLIP_IMG_SIM:
#   Reject if source and target are too similar -> almost no visible edit.
#
# MIN_CLIP_DIR_SIM:
#   Reject if the source->target visual change does not align with the prompt.
# ---------------------------------------------------------------------
MIN_CLIP_IMG_SIM = 0.70
MAX_CLIP_IMG_SIM = 0.98
MIN_CLIP_DIR_SIM = 0.08

# ---------------------------------------------------------------------
# Load Stage 1 candidate indices
# ---------------------------------------------------------------------
# I convert the list to a set so membership checks are fast while I stream
# through the whole source dataset:
#     if idx not in s1_indices: continue
# ---------------------------------------------------------------------
print("Loading Stage 1 indices...", flush=True)
with open(STAGE1_FILE) as f:
    s1_indices = set(json.load(f))
print("  {} indices".format(len(s1_indices)), flush=True)

# ---------------------------------------------------------------------
# Load CLIP model
# ---------------------------------------------------------------------
# This script uses OpenCLIP ViT-B/32 (OpenAI weights) to score edit quality.
# It runs entirely on GPU for the scoring stage, which is why it is much faster
# than the rembg-based Stage 2 foreground filtering.
# ---------------------------------------------------------------------
print("Loading CLIP...", flush=True)
import torch
import open_clip
device = "cuda"
model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")
tokenizer = open_clip.get_tokenizer("ViT-B-32")
model = model.to(device).eval()
print("  CLIP loaded on {}".format(device), flush=True)

# ---------------------------------------------------------------------
# Stream the source dataset
# ---------------------------------------------------------------------
# This script also uses Hugging Face streaming mode, so it does not try to
# fully load the dataset into RAM.
# ---------------------------------------------------------------------
print("Starting CLIP scoring (streaming)...", flush=True)
from datasets import load_dataset
os.environ["HF_HOME"] = "/workspace/hf_cache"
os.environ["HF_DATASETS_CACHE"] = "/workspace/hf_cache/datasets"

dataset = load_dataset("timbrooks/instructpix2pix-clip-filtered", split="train", streaming=True)

# ---------------------------------------------------------------------
# Runtime buffers and counters
# ---------------------------------------------------------------------
# scored:
#   Will hold one dictionary per successfully scored sample:
#       {"index": ..., "img_sim": ..., "dir_sim": ..., "prompt": ...}
#
# processed:
#   How many Stage 1 candidates I have actually tried to score.
#
# skipped:
#   How many source dataset rows I skipped because they were not in Stage 1.
#
# total:
#   Number of Stage 1 candidate samples.
#
# batch_*:
#   Small buffers used to build GPU batches for CLIP scoring.
# ---------------------------------------------------------------------
scored = []
processed = 0
skipped = 0
total = len(s1_indices)
batch_idx = []
batch_src = []
batch_tgt = []
batch_prompts = []
BATCH = 32

def process_batch():
    """
    Score one CLIP batch and append results to `scored`.

    For each sample in the batch:
    - encode source image with CLIP
    - encode target image with CLIP
    - encode prompt text with CLIP
    - normalize all embeddings
    - compute img_sim = similarity(source, target)
    - compute direction = target - source
    - compute dir_sim = similarity(direction, prompt)

    Why these scores matter:
    - img_sim too low  -> identity lost
    - img_sim too high -> nothing visibly changed
    - dir_sim too low  -> edit does not match the prompt
    """
    global scored
    if not batch_src:
        return
    try:
        with torch.no_grad():
            sf = model.encode_image(torch.stack(batch_src).to(device))
            tf = model.encode_image(torch.stack(batch_tgt).to(device))
            xf = model.encode_text(tokenizer(batch_prompts).to(device))

            # Normalize to unit vectors so dot products behave like cosine similarity.
            sf = sf / sf.norm(dim=-1, keepdim=True)
            tf = tf / tf.norm(dim=-1, keepdim=True)
            xf = xf / xf.norm(dim=-1, keepdim=True)

            # Similarity between original and edited image embeddings.
            img_sim = (sf * tf).sum(dim=-1)

            # Edit direction in CLIP space.
            d = tf - sf
            d = d / (d.norm(dim=-1, keepdim=True) + 1e-8)

            # How well the visual edit direction matches the prompt.
            dir_sim = (d * xf).sum(dim=-1)

        for i, idx in enumerate(batch_idx):
            scored.append({"index": idx, "img_sim": img_sim[i].item(),
                          "dir_sim": dir_sim[i].item(), "prompt": batch_prompts[i]})
    except Exception as e:
        print("  Batch error: {}".format(e), flush=True)

# ---------------------------------------------------------------------
# Main scoring loop
# ---------------------------------------------------------------------
# I stream through the entire source dataset, but only process rows whose
# original index is in the Stage 1 candidate set.
#
# Important:
# This script does NOT perform Stage 2 foreground filtering.
# It goes directly from Stage 1 prompt filtering -> Stage 3 CLIP scoring.
# ---------------------------------------------------------------------
start = time.time()
for idx, sample in enumerate(dataset):
    if idx not in s1_indices:
        skipped += 1
        if skipped % 50000 == 0:
            print("  Skipped {} non-matching samples...".format(skipped), flush=True)
        continue

    processed += 1
    try:
        src = sample["original_image"]
        tgt = sample["edited_image"]
        if not isinstance(src, Image.Image):
            src = Image.open(src).convert("RGB")
        if not isinstance(tgt, Image.Image):
            tgt = Image.open(tgt).convert("RGB")

        batch_idx.append(idx)
        batch_src.append(preprocess(src))
        batch_tgt.append(preprocess(tgt))
        batch_prompts.append(sample["edit_prompt"])

        # Once the batch reaches size BATCH, send it through CLIP.
        if len(batch_idx) >= BATCH:
            process_batch()
            batch_idx.clear()
            batch_src.clear()
            batch_tgt.clear()
            batch_prompts.clear()
    except Exception:
        continue

    # Print useful progress every 500 successfully processed candidates.
    if processed % 500 == 0:
        elapsed = time.time() - start
        rate = processed / elapsed
        eta = (total - processed) / rate / 3600
        print("  Scored {:,}/{:,} | {:.1f}/sec | ETA {:.1f}h".format(
            processed, total, rate, eta), flush=True)

# Score any leftover partial batch.
process_batch()

# ---------------------------------------------------------------------
# Apply Stage 3 filtering thresholds
# ---------------------------------------------------------------------
# Keep only samples that satisfy all three conditions:
# - image similarity is not too low
# - image similarity is not too high
# - edit direction matches the prompt sufficiently well
# ---------------------------------------------------------------------
reject = Counter()
kept = []
for s in scored:
    if s["img_sim"] < MIN_CLIP_IMG_SIM:
        reject["identity_lost"] += 1
    elif s["img_sim"] > MAX_CLIP_IMG_SIM:
        reject["no_visible_change"] += 1
    elif s["dir_sim"] < MIN_CLIP_DIR_SIM:
        reject["edit_not_aligned"] += 1
    else:
        kept.append(s)

print("", flush=True)
print("Scored:   {:,}".format(len(scored)), flush=True)
print("Passed:   {:,}".format(len(kept)), flush=True)
print("Rejected: {:,}".format(len(scored)-len(kept)), flush=True)
for r, c in reject.most_common():
    print("  {:25s} {:>7,}".format(r, c), flush=True)

# ---------------------------------------------------------------------
# Rank by dir_sim and trim to the target size
# ---------------------------------------------------------------------
# Higher dir_sim means the visual edit direction matches the prompt better.
# ---------------------------------------------------------------------
kept.sort(key=lambda x: x["dir_sim"], reverse=True)
if len(kept) > TARGET_SIZE:
    kept = kept[:TARGET_SIZE]
    print("Trimmed to top {:,}".format(TARGET_SIZE), flush=True)

# NOTE:
# final_indices is converted to a set, which is convenient for export-time
# lookup, but it also destroys the ranking order when saved to disk.
final_indices = set(s["index"] for s in kept)
final_prompts = {s["index"]: s["prompt"] for s in kept}

# Save the selected original dataset indices.
os.makedirs(OUTPUT_DIR, exist_ok=True)
with open("{}/final_indices.json".format(OUTPUT_DIR), "w") as f:
    json.dump(list(final_indices), f)
print("Saved {} indices".format(len(final_indices)), flush=True)

# ---------------------------------------------------------------------
# Export actual image files
# ---------------------------------------------------------------------
# Now that I know which original dataset rows survived Stage 3, I stream the
# source dataset again and physically save the original and edited image files.
# I also build metadata.jsonl so later steps know:
# - the exported id
# - the original dataset index
# - the edit prompt
# - the file paths of original and edited images
# ---------------------------------------------------------------------
print("", flush=True)
print("Exporting images (streaming)...", flush=True)
os.makedirs("{}/original_images".format(OUTPUT_DIR), exist_ok=True)
os.makedirs("{}/edited_images".format(OUTPUT_DIR), exist_ok=True)

# Free the CLIP model before export to save GPU memory.
del model
torch.cuda.empty_cache()
gc.collect()

dataset2 = load_dataset("timbrooks/instructpix2pix-clip-filtered", split="train", streaming=True)
metadata = []
new_idx = 0

for idx, sample in enumerate(dataset2):
    if idx not in final_indices:
        continue
    try:
        src = sample["original_image"]
        tgt = sample["edited_image"]
        if not isinstance(src, Image.Image):
            src = Image.open(src).convert("RGB")
        if not isinstance(tgt, Image.Image):
            tgt = Image.open(tgt).convert("RGB")

        src.save("{}/original_images/{:06d}.png".format(OUTPUT_DIR, new_idx))
        tgt.save("{}/edited_images/{:06d}.png".format(OUTPUT_DIR, new_idx))

        metadata.append({"id": new_idx, "original_dataset_index": idx,
                        "edit_prompt": final_prompts.get(idx, sample["edit_prompt"]),
                        "original_image": "original_images/{:06d}.png".format(new_idx),
                        "edited_image": "edited_images/{:06d}.png".format(new_idx)})
        new_idx += 1

        if new_idx % 1000 == 0:
            print("  Exported {:,}/{:,}".format(new_idx, len(final_indices)), flush=True)
    except Exception:
        continue

# Save metadata manifest.
with open("{}/metadata.jsonl".format(OUTPUT_DIR), "w") as f:
    for m in metadata:
        f.write(json.dumps(m) + "\n")

print("", flush=True)
print("DONE! {:,} triplets saved to {}".format(len(metadata), OUTPUT_DIR), flush=True)
