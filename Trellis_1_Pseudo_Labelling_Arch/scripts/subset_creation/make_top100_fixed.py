import json

DATA_DIR = "/workspace/filtered_dataset"

with open(f"{DATA_DIR}/final_indices.json", "r") as f:
    indices = json.load(f)

top_100 = indices[:100]

with open(f"{DATA_DIR}/metadata.jsonl", "r") as f:
    rows = [json.loads(line) for line in f]

# Match using original_dataset_index
idx_map = {row["original_dataset_index"]: row for row in rows}
selected = [idx_map[i] for i in top_100 if i in idx_map]

with open(f"{DATA_DIR}/top100_metadata.jsonl", "w") as f:
    for row in selected:
        f.write(json.dumps(row) + "\n")

print("Saved:", f"{DATA_DIR}/top100_metadata.jsonl")
print("Count:", len(selected))
print("First 5 prompts:")
for row in selected[:5]:
    print("-", row["edit_prompt"])
