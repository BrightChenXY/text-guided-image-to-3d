import json

DATA_DIR = "/workspace/filtered_dataset"

with open(f"{DATA_DIR}/final_indices.json", "r") as f:
    indices = json.load(f)

top_100 = indices[:100]

with open(f"{DATA_DIR}/metadata.jsonl", "r") as f:
    rows = [json.loads(line) for line in f]

if isinstance(top_100[0], int):
    selected = [rows[i] for i in top_100]
else:
    id_map = {str(r["id"]): r for r in rows}
    selected = [id_map[str(i)] for i in top_100 if str(i) in id_map]

with open(f"{DATA_DIR}/top100_metadata.jsonl", "w") as f:
    for row in selected:
        f.write(json.dumps(row) + "\n")

print("Saved:", f"{DATA_DIR}/top100_metadata.jsonl")
print("Count:", len(selected))
print("First 5 samples:")
for row in selected[:5]:
    print(row)
