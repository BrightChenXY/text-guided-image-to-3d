import argparse
import json
import os
import shutil
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description='Create an n-sample subset from filtered_dataset using final_indices.json + metadata.jsonl')
    p.add_argument('--data_dir', required=True, help='Path to filtered_dataset root')
    p.add_argument('--n', type=int, required=True, help='Number of samples to keep')
    p.add_argument('--out_dir', required=True, help='Output folder for the physical subset')
    p.add_argument('--metadata_name', default='subset_metadata.jsonl', help='Name of output metadata file')
    args = p.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / 'original_images').mkdir(exist_ok=True)
    (out_dir / 'edited_images').mkdir(exist_ok=True)

    with open(data_dir / 'final_indices.json', 'r', encoding='utf-8') as f:
        indices = json.load(f)
    top_n = indices[:args.n]

    with open(data_dir / 'metadata.jsonl', 'r', encoding='utf-8') as f:
        rows = [json.loads(line) for line in f]

    idx_map = {row['original_dataset_index']: row for row in rows}
    selected = [idx_map[i] for i in top_n if i in idx_map]

    with open(out_dir / args.metadata_name, 'w', encoding='utf-8') as f:
        for row in selected:
            f.write(json.dumps(row) + '\n')

    for row in selected:
        orig = data_dir / row['original_image']
        edit = data_dir / row['edited_image']
        shutil.copy2(orig, out_dir / row['original_image'])
        shutil.copy2(edit, out_dir / row['edited_image'])

    print('Saved metadata:', out_dir / args.metadata_name)
    print('Selected rows:', len(selected))
    print('Copied original images:', len(list((out_dir / 'original_images').glob('*'))))
    print('Copied edited images:', len(list((out_dir / 'edited_images').glob('*'))))


if __name__ == '__main__':
    main()
