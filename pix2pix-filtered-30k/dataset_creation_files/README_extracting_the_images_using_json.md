# Dataset Recovery Kit for `pix2pix-filtered-30k`

This folder helps any user rebuild a usable copy of the filtered InstructPix2Pix dataset when the image folders are missing or incomplete.

It is designed for datasets that came from:
- `metadata.jsonl`
- `final_indices.json`

and the public source dataset:
- `timbrooks/instructpix2pix-clip-filtered`

---

## What this kit does

The recovery script can recreate:
- `original_images/`
- `edited_images/`
- `metadata.jsonl`

using one or both of these files:
- `metadata.jsonl`
- `final_indices.json`

### Important note

`final_indices.json` contains **original InstructPix2Pix dataset indices**.

That means:
- it is **not** a row-number list for `metadata.jsonl`
- it is **not** the same thing as the exported `id`
- if you match metadata to final indices, match using:

```python
row["original_dataset_index"]
```

---






## Run on Google Colab

If you want the easiest guided workflow, use:

- `recover_dataset_colab.ipynb` — step-by-step Colab notebook
- `recover_dataset.py` — main recovery script

### Colab workflow

1. Open `recover_dataset_colab.ipynb` in Google Colab.
2. Upload:
   - `recover_dataset.py`
   - `final_indices.json`
   - `metadata.jsonl` (if available)
3. Run the install cells.
4. Run the recovery cell.
5. Verify that:
   - `original_images/` was created
   - `edited_images/` was created
   - `metadata.jsonl` was written
6. Optionally save the recovered dataset to Google Drive.

### Recommended mode

Use both:
- `metadata.jsonl`
- `final_indices.json`

This preserves the exported file layout and filenames.









## Files in this folder

- `recover_dataset.py` — main recovery script
- `README.md` — this guide

---


## When to use which mode

### Mode A — Best option
Use this when you have both:
- `metadata.jsonl`
- `final_indices.json`

This preserves the exported file structure and file names from the metadata.

### Mode B — Still works
Use this when you have only:
- `final_indices.json`

This creates a fresh dataset folder with new numbered file names and a new `metadata.jsonl`.

---

## Step-by-step instructions

## 1. Create a working folder

Make a folder on your machine or pod, for example:

```bash
mkdir -p /workspace/recovery_job
```

Copy these files into that folder:
- `recover_dataset.py`
- `metadata.jsonl` (if available)
- `final_indices.json` (if available)

Example layout:

```text
/workspace/recovery_job/
  recover_dataset.py
  metadata.jsonl
  final_indices.json
```

---

## 2. Install the required packages

Run:

```bash
python -m pip install -U datasets pillow
```

---

## 3. Run the recovery

### Option A — If you have both metadata and final indices

```bash
python /workspace/recovery_job/recover_dataset.py \
  --metadata /workspace/recovery_job/metadata.jsonl \
  --final-indices /workspace/recovery_job/final_indices.json \
  --output-dir /workspace/recovered_dataset \
  --skip-existing
```

### Option B — If you only have final indices

```bash
python /workspace/recovery_job/recover_dataset.py \
  --final-indices /workspace/recovery_job/final_indices.json \
  --output-dir /workspace/recovered_dataset \
  --skip-existing
```

### Option C — If you only have metadata

```bash
python /workspace/recovery_job/recover_dataset.py \
  --metadata /workspace/recovery_job/metadata.jsonl \
  --output-dir /workspace/recovered_dataset \
  --skip-existing
```

---

## 4. What the script will create

After the script finishes, the output folder will look like this:

```text
/workspace/recovered_dataset/
  original_images/
  edited_images/
  metadata.jsonl
  recovery_summary.json
```

---

## 5. Verify that the recovery worked

Run these checks:

```bash
ls /workspace/recovered_dataset/original_images | wc -l
ls /workspace/recovered_dataset/edited_images | wc -l
```

Then verify one metadata row:

```bash
python - << 'PY'
import json, os
meta = "/workspace/recovered_dataset/metadata.jsonl"
with open(meta, "r", encoding="utf-8") as f:
    row = json.loads(next(f))
print("First row:", row)
print("Original exists:", os.path.exists("/workspace/recovered_dataset/" + row["original_image"]))
print("Edited exists:", os.path.exists("/workspace/recovered_dataset/" + row["edited_image"]))
PY
```

If both print `True`, the dataset is ready to use.

---

## 6. Expected metadata format

A normal recovered row looks like this:

```json
{
  "id": 0,
  "original_dataset_index": 262144,
  "edit_prompt": "make it red",
  "original_image": "original_images/000000.png",
  "edited_image": "edited_images/000000.png"
}
```

---

## 7. How to use the recovered dataset

For each sample, use:
- `original_image` → source image
- `edit_prompt` → text instruction
- `edited_image` → edited target image

This is the triplet needed for image-edit supervision.

---

## 8. Common mistake to avoid

Do **not** do this:

```python
selected = [rows[i] for i in final_indices]
```

That is wrong because `final_indices.json` stores **original source-dataset indices**, not metadata row positions.

If you need to match metadata rows to final indices, do this instead:

```python
idx_map = {row["original_dataset_index"]: row for row in rows}
selected = [idx_map[i] for i in final_indices if i in idx_map]
```

---

## 9. Helpful notes

- The script uses Hugging Face `streaming=True`, so it does not need to fully download the entire source dataset first.
- If some images already exist in the output folder, `--skip-existing` avoids re-saving them.
- If you want a small test run first, add `--limit 100`.

Example:

```bash
python /workspace/recovery_job/recover_dataset.py \
  --metadata /workspace/recovery_job/metadata.jsonl \
  --final-indices /workspace/recovery_job/final_indices.json \
  --output-dir /workspace/recovered_dataset_test \
  --skip-existing \
  --limit 100
```

---

## 10. Recommended workflow for teammates

1. Copy this recovery kit folder.
2. Put `metadata.jsonl` and `final_indices.json` beside the script.
3. Run the script.
4. Verify that both image folders were rebuilt.
5. Use `/workspace/recovered_dataset/` as the dataset root.

---

## 11. One-line summary

If the image folders are missing, this kit rebuilds the dataset from the public source dataset using the saved original dataset indices.
