# -*- coding: utf-8 -*-
"""
stage1_filter.py - Prompt-Based Filtering for TRELLIS 1

What this script does:
1. Loads the InstructPix2Pix training split from the local Hugging Face cache.
2. Looks only at the edit_prompt text for each sample.
3. Rejects prompts that look like scene/background/weather/photography edits.
4. Keeps prompts that look like object-level edits relevant to TRELLIS 1.
5. Saves the kept dataset indices so I can use them later on RunPod for further filtering.

Author: Sambit Hore, Trinity College Dublin
"""

import json
import re
import os
import sys
from collections import Counter
from pathlib import Path

# --- Check dependencies ---

# ---------------------------------------------------------------------
# Dependency checks
# ---------------------------------------------------------------------
# I am checking for the required packages up front so the script fails early
# with a clear message instead of crashing later in a confusing way.
#
# datasets:
#   Needed to load the InstructPix2Pix dataset from Hugging Face cache.
#
# tqdm:
#   Used only for a nice progress bar while I loop through all prompts.
#   If tqdm is missing, I fall back to a simple custom progress printer.
# ---------------------------------------------------------------------
try:
    from datasets import load_dataset
except ImportError:
    print("ERROR: 'datasets' library not installed.")
    print("Fix: pip install datasets")
    sys.exit(1)

try:
    from tqdm import tqdm
except ImportError:
    print("WARNING: 'tqdm' not installed, using simple progress.")
    def tqdm(iterable, **kwargs):
        desc = kwargs.get("desc", "")
        total = kwargs.get("total", None)
        for i, item in enumerate(iterable):
            if i % 10000 == 0:
                print("  {}: {:,}".format(desc, i) + (" / {:,}".format(total) if total else ""))
            yield item



# ---------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------
# OUTPUT_DIR:
#   This is where I save all Stage 1 results.
#
# MIN_PROMPT_WORDS / MAX_PROMPT_WORDS:
#   These are very simple sanity checks.
#   - Extremely short prompts are often too vague to be useful.
#   - Extremely long prompts often describe complex scenes or multi-step edits.
# ---------------------------------------------------------------------
OUTPUT_DIR = Path("./stage1_output")
MIN_PROMPT_WORDS = 3
MAX_PROMPT_WORDS = 25


# ---------------------------------------------------------------------
# FILTER RULES
# ---------------------------------------------------------------------
# This stage is intentionally text-only and heuristic-based.
# I am not looking at images here yet. I only inspect the edit_prompt text.
#
# The overall idea:
# - Reject prompts that look scene-level.
# - Keep prompts that look object-level.
# - If a prompt is not clearly good or clearly bad, I keep it as "ambiguous"
#   so later GPU stages can make a better decision.
# ---------------------------------------------------------------------

# ---------------------------------------------------------------------
# REJECT these keywords
# ---------------------------------------------------------------------
# These words usually indicate that the user is editing:
# - weather
# - time of day
# - landscape/background
# - full-scene atmosphere
# - photographic effects
# - adding/removing unrelated scene entities
#
# These are usually bad for TRELLIS 1 because TRELLIS 1 is much better
# suited to single-object edits, not whole-scene transformations.
# ---------------------------------------------------------------------
SCENE_KEYWORDS = {
    "weather", "rain", "rainy", "raining", "snow", "snowing", "snowy",
    "fog", "foggy", "mist", "misty", "cloud", "cloudy", "storm", "stormy",
    "thunder", "lightning", "hail", "tornado", "hurricane",
    "sunset", "sunrise", "dawn", "dusk", "twilight", "midnight",
    "night", "nighttime", "daytime", "morning", "evening", "afternoon",
    "overcast", "sunny", "windy",
    "season", "spring", "summer", "autumn", "fall", "winter",
    "christmas", "halloween", "holiday",
    "background", "backdrop", "scenery", "landscape", "skyline",
    "horizon", "sky", "ocean", "sea", "beach", "mountain", "forest",
    "field", "meadow", "desert", "canyon", "valley", "river", "lake",
    "waterfall", "volcano", "island", "cliff",
    "blur", "bokeh", "exposure", "contrast", "brightness", "saturation",
    "hdr", "filter", "vignette", "sepia", "black and white", "grayscale",
    "panorama", "wide angle", "zoom",
    "depth of field", "lens flare", "motion blur",
    "add people", "add person", "add crowd", "add animals",
    "add trees", "add buildings", "add cars",
    "remove people", "remove person", "remove background",
    "change background", "replace background",
    "add text", "add caption", "add watermark",
}

# ---------------------------------------------------------------------
# KEEP these keywords
# ---------------------------------------------------------------------
# These words usually point toward edits applied to the object itself:
# - material changes
# - color changes
# - shape/size changes
# - artistic style changes
# - condition/state changes
#
# These are exactly the kinds of edits I want to keep for an object-centric
# dataset that will later be used for TRELLIS-style 3D editing supervision.
# ---------------------------------------------------------------------
OBJECT_KEYWORDS = {
    "wooden", "metal", "metallic", "glass", "crystal", "stone",
    "marble", "brick", "ceramic", "plastic", "rubber", "leather",
    "fabric", "silk", "velvet", "wool", "cotton", "fur", "furry",
    "chrome", "gold", "golden", "silver", "bronze", "copper",
    "rusty", "shiny", "matte", "glossy", "transparent", "translucent",
    "red", "blue", "green", "yellow", "orange", "purple", "pink",
    "black", "white", "gray", "grey", "brown", "cyan", "magenta",
    "turquoise", "teal", "navy", "crimson", "scarlet",
    "color", "colour", "paint", "painted",
    "bigger", "smaller", "larger", "taller", "shorter", "wider",
    "thinner", "rounder", "flatter", "pointy", "angular",
    "stretched", "compressed", "twisted", "bent", "curved",
    "elongated", "inflated", "deflated",
    "cartoon", "anime", "pixel", "pixelated", "watercolor",
    "oil painting", "sketch", "drawing", "stylized",
    "realistic", "photorealistic", "futuristic", "steampunk",
    "cyberpunk", "medieval", "ancient", "modern", "vintage",
    "retro", "minimalist", "ornate", "decorated",
    "lego", "clay", "origami", "knitted", "crocheted",
    "on fire", "burning", "frozen", "ice", "icy", "melting",
    "wet", "dry", "dirty", "clean", "broken", "cracked",
    "new", "old", "aged", "worn", "polished",
    "striped", "spotted", "checkered", "camouflage",
    "glowing", "neon", "luminous", "sparkling",
    "make it", "turn it", "make the", "turn the",
}

# ---------------------------------------------------------------------
# Scene-level regex patterns
# ---------------------------------------------------------------------
# Keywords alone are not enough, because some prompts are more about
# sentence structure than a single word.
#
# These patterns catch prompts like:
# - "add a tree to the background"
# - "replace the scene with a beach"
# - "make the image look like night"
# - "turn this photo into a city street scene"
#
# If a prompt matches one of these, I reject it immediately.
#
SCENE_PATTERNS = [
    r"\b(add|place|put)\b.*(to|in|on)\b.*(scene|background|sky|ground)",
    r"\b(change|replace|swap)\b.*\b(background|scene|setting|environment)\b",
    r"\b(make|turn)\b.*\b(scene|image|photo|picture|photograph)\b",
    r"\bmake it (look like|seem like)\b.*\b(night|day|morning|evening)\b",
    r"\b(indoors?|outdoors?)\b",
    r"\b(city|town|village|street|road|highway)\b",
]


# ---------------------------------------------------------------------
# Object-level regex patterns
# ---------------------------------------------------------------------
# These patterns catch common object-edit phrasing even if the prompt
# does not contain one of my exact object keywords.
#
# Examples:
# - "make it into glass"
# - "turn the vase blue"
# - "give it wings"
# - "add stripes to it"
# - "remove the handle from the mug"
#
# If a prompt matches one of these, I keep it.
# 
OBJECT_PATTERNS = [
    r"\b(make|turn|change|paint|color)\b.*\b(it|the \w+)\b.*\b(into|to|from)\b",
    r"\b(make|turn)\b.*\b(it|the \w+)\b\s+\w+$",
    r"\bgive (it|the \w+)\b",
    r"\b(add|put|give)\b.*\b(to|on) (it|the \w+)\b",
    r"\b(remove|take off|delete)\b.*\bfrom (it|the \w+)\b",
]



# ---------------------------------------------------------------------
# classify_prompt(prompt)
# ---------------------------------------------------------------------
# This is the core decision function for Stage 1.
#
# Input:
#   A single edit prompt string from InstructPix2Pix.
#
# Output:
#   A tuple:
#     (keep_boolean, reason_string)
#
# Decision flow:
#   1. Normalize the prompt.
#   2. Reject it if it is too short.
#   3. Reject it if it is too long.
#   4. Reject it if it matches a scene-level regex pattern.
#   5. Reject it if it contains a scene-level keyword.
#   6. Keep it if it contains an object-level keyword.
#   7. Keep it if it matches an object-level regex pattern.
#   8. Otherwise keep it as "ambiguous".
#
# Why I keep ambiguous prompts:
#   Stage 1 is meant to be a broad, cheap first-pass filter.
#   I do not want to be too aggressive here and accidentally throw away
#   useful data. Later GPU stages can do stricter filtering.
# ---------------------------------------------------------------------
def classify_prompt(prompt):
    # Lowercase and strip whitespace so matching is case-insensitive
    # and not affected by accidental spaces at the start/end.
    p = prompt.lower().strip()
    
    # Split into words for the basic length checks below.
    words = p.split()

    # Reject very short prompts because they are often too vague.
    if len(words) < MIN_PROMPT_WORDS:
        return False, "too_short"
        
    # Reject very long prompts because they are often noisy or scene-heavy.    
    if len(words) > MAX_PROMPT_WORDS:
        return False, "too_long"
    
    # First, check strong scene-level sentence patterns.
    # If one matches, I reject immediately.
    for pattern in SCENE_PATTERNS:
        if re.search(pattern, p):
            return False, "scene_pattern"
    
    # Then check scene-level keywords.
    # I use word boundaries (\b) so I match whole words/phrases more safely.
    # re.escape() protects against regex issues inside the keyword text.
    for kw in SCENE_KEYWORDS:
        if re.search(r"\b{}\b".format(re.escape(kw)), p):
            return False, "scene_kw:{}".format(kw)
    
    # If the prompt contains a strong object-level keyword,
    # I keep it as likely useful for object editing.
    for kw in OBJECT_KEYWORDS:
        if re.search(r"\b{}\b".format(re.escape(kw)), p):
            return True, "object_keyword"
    
    # If the prompt matches a common object-edit grammar pattern,
    # I also keep it.
    for pattern in OBJECT_PATTERNS:
        if re.search(pattern, p):
            return True, "object_pattern"
    
    # Default fallback:
    # If the prompt is not clearly bad and not clearly good,
    # I still keep it for later stages to judge.
    return True, "ambiguous"


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------
# This function runs the full Stage 1 pipeline:
#   Step 1: Load dataset
#   Step 2: Preview a few prompts
#   Step 3: Filter every prompt
#   Step 4: Save outputs and print summary
# ---------------------------------------------------------------------

def main():
    print("=" * 60)
    print("  STAGE 1: Prompt-Based Filtering")
    print("  InstructPix2Pix -> TRELLIS 1 Object Edits")
    print("=" * 60)

    # -----------------------------------------------------------------
    # Step 1: Load dataset
    # -----------------------------------------------------------------
    # I expect the dataset to already exist in the local Hugging Face cache.
    # On my Windows machine this should read from:
    #   C:\Users\sambi\.cache\huggingface\
    #
    # It should not re-download if the cache is already present.
    # If it unexpectedly starts downloading again, I should stop and check why.
    # -----------------------------------------------------------------
    print("")
    print("[Step 1/4] Loading dataset from local HuggingFace cache...")
    print("  Reading from C:\\Users\\sambi\\.cache\\huggingface\\")
    print("  Should NOT re-download. If it does, press Ctrl+C.")
    print("")

    try:
        dataset = load_dataset(
            "timbrooks/instructpix2pix-clip-filtered",
            split="train",
        )
    except Exception as e:
        print("ERROR loading dataset: {}".format(e))
        sys.exit(1)

    total = len(dataset)
    print("  Loaded {:,} samples".format(total))
    print("  Columns: {}".format(dataset.column_names))

    # -----------------------------------------------------------------
    # Step 2: Preview
    # -----------------------------------------------------------------
    # Before I filter the whole dataset, I print the first few prompts
    # just to sanity-check that the right dataset loaded and the field
    # name is what I expect ("edit_prompt").
    # -----------------------------------------------------------------
    print("")
    print("[Step 2/4] Previewing first 10 edit prompts...")
    for i in range(min(10, total)):
        print('  [{}] "{}"'.format(i, dataset[i]["edit_prompt"]))

    # -----------------------------------------------------------------
    # Step 3: Filter
    # -----------------------------------------------------------------
    # Here I loop through every dataset row, classify the edit prompt,
    # and store:
    #
    # kept_indices:
    #   Original dataset row numbers that passed Stage 1.
    #
    # kept_prompts:
    #   The prompt text for prompts I kept, used only for preview output.
    #
    # rejected_samples:
    #   A small sample of rejected prompts saved for manual inspection.
    #
    # reject_reasons / keep_reasons:
    #   Counters that help me understand what the filter is doing.
    # -----------------------------------------------------------------
    print("")
    print("[Step 3/4] Filtering {:,} prompts...".format(total))
    print("  This takes about 2-5 minutes. Please wait.")
    print("")

    kept_indices = []
    kept_prompts = []
    rejected_samples = []

    reject_reasons = Counter()
    keep_reasons = Counter()

    for idx in tqdm(range(total), desc="  Filtering", total=total):
        
        # Pull the prompt text from the current dataset row.
        prompt = dataset[idx]["edit_prompt"]
        
        # Run my rule-based classifier on that prompt.
        keep, reason = classify_prompt(prompt)

        if keep:
            # If I keep the sample:
            # - store its original dataset index
            # - store the prompt text for preview purposes
            # - count the keep reason
            kept_indices.append(idx)
            kept_prompts.append(prompt)
            keep_reasons[reason] += 1
        else:
            # If I reject the sample:
            # - count the reject reason
            # - store up to 5 example prompts per reject reason
            #   so I can inspect them later without creating a huge file
            reject_reasons[reason] += 1
            if reject_reasons[reason] <= 5:
                rejected_samples.append({
                    "index": idx,
                    "prompt": prompt,
                    "reason": reason,
                })

    kept = len(kept_indices)
    rejected = total - kept

    # -----------------------------------------------------------------
    # Step 4: Save
    # -----------------------------------------------------------------
    # I now write all outputs to disk.
    #
    # The most important output is:
    #   stage1_kept_indices.json
    #
    # That file is what I pass to the later RunPod GPU stages.
    # -----------------------------------------------------------------
    print("")
    print("[Step 4/4] Saving results...")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Save indices
    # This is the main artifact from Stage 1.
    # It contains the original dataset row indices that passed the filter.
    indices_path = OUTPUT_DIR / "stage1_kept_indices.json"
    with open(indices_path, "w") as f:
        json.dump(kept_indices, f)

    # Save kept prompts preview
    # I write the first 200 kept prompts so I can manually inspect whether
    # the filter kept the sort of edits I actually want.
    kept_preview_path = OUTPUT_DIR / "kept_prompts_preview.txt"
    with open(kept_preview_path, "w", encoding="utf-8") as f:
        f.write("KEPT PROMPTS - {:,} total\n".format(kept))
        f.write("=" * 60 + "\n\n")
        for i, prompt in enumerate(kept_prompts[:200]):
            f.write("[{:4d}] {}\n".format(i, prompt))
        if kept > 200:
            f.write("\n... and {:,} more\n".format(kept - 200))

    # Save rejected samples preview
    # I save a small set of rejected prompts plus their reasons so I can
    # check whether my rules were too harsh or made obvious mistakes.
    rejected_path = OUTPUT_DIR / "rejected_samples_preview.txt"
    with open(rejected_path, "w", encoding="utf-8") as f:
        f.write("REJECTED PROMPTS - {:,} total\n".format(rejected))
        f.write("=" * 60 + "\n\n")
        for s in rejected_samples:
            f.write('  [{:6d}] reason={:30s} "{}"\n'.format(
                s["index"], s["reason"], s["prompt"]))

    # Save summary
    # This gives me a compact machine-readable summary of the run:
    # - total samples
    # - kept vs rejected counts
    # - kept percentage
    # - the breakdown of keep reasons
    # - the top reject reasons
    summary = {
        "total_samples": total,
        "kept_samples": kept,
        "rejected_samples": rejected,
        "kept_percentage": round(100 * kept / total, 1),
        "keep_reasons": dict(keep_reasons.most_common()),
        "reject_reasons": dict(reject_reasons.most_common(30)),
    }
    summary_path = OUTPUT_DIR / "stage1_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    # Print results
    print("")
    print("=" * 60)
    print("  RESULTS")
    print("=" * 60)
    print("")
    print("  Total samples:     {:>10,}".format(total))
    print("  KEPT:              {:>10,}  ({:.1f}%)".format(kept, 100*kept/total))
    print("  Rejected:          {:>10,}  ({:.1f}%)".format(rejected, 100*rejected/total))

    print("")
    print("  -- Why samples were KEPT --")
    for reason, count in keep_reasons.most_common(5):
        bar = "#" * min(40, int(40 * count / max(keep_reasons.values())))
        print("    {:20s}  {:>7,}  {}".format(reason, count, bar))

    print("")
    print("  -- Why samples were REJECTED (top 15) --")
    for reason, count in reject_reasons.most_common(15):
        bar = "#" * min(40, int(40 * count / max(reject_reasons.values())))
        print("    {:30s}  {:>7,}  {}".format(reason, count, bar))

    print("")
    print("  -- Files saved in {} --".format(OUTPUT_DIR))
    print("    - stage1_kept_indices.json   (upload to RunPod for Stage 2)")
    print("    - kept_prompts_preview.txt   (review what was kept)")
    print("    - rejected_samples_preview.txt (review what was rejected)")
    print("    - stage1_summary.json        (full statistics)")

    print("")
    print("  -- What to do next --")
    print("    1. Open kept_prompts_preview.txt to check kept prompts look right")
    print("    2. Open rejected_samples_preview.txt to check nothing good was lost")
    print("    3. Upload stage1_kept_indices.json to RunPod for Stage 2+3")
    print("")
    print("  Done!")
    print("")


if __name__ == "__main__":
    main()
