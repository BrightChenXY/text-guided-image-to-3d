import argparse
import json
import os
from pathlib import Path

from PIL import Image

from trellis.pipelines import TrellisImageTo3DPipeline
from trellis.utils import postprocessing_utils


def main():
    p = argparse.ArgumentParser(description='Generate pseudo-3D assets from edited images in a subset metadata file.')
    p.add_argument('--subset_root', required=True, help='Folder containing original_images/, edited_images/, subset_metadata.jsonl')
    p.add_argument('--metadata', default='subset_metadata.jsonl', help='Metadata file inside subset_root')
    p.add_argument('--out_root', required=True, help='Folder to save sample_XXX pseudo assets')
    p.add_argument('--seed', type=int, default=1)
    p.add_argument('--limit', type=int, default=None)
    p.add_argument('--start', type=int, default=0)
    p.add_argument('--texture_size', type=int, default=1024)
    p.add_argument('--simplify', type=float, default=0.95)
    args = p.parse_args()

    subset_root = Path(args.subset_root)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    rows = []
    with open(subset_root / args.metadata, 'r', encoding='utf-8') as f:
        for line in f:
            rows.append(json.loads(line))

    rows = rows[args.start:]
    if args.limit is not None:
        rows = rows[:args.limit]

    pipeline = TrellisImageTo3DPipeline.from_pretrained('microsoft/TRELLIS-image-large')
    pipeline.cuda()

    for i, row in enumerate(rows, start=args.start):
        sample_dir = out_root / f'sample_{i:03d}'
        sample_dir.mkdir(parents=True, exist_ok=True)
        meta_path = sample_dir / 'meta.json'
        edited_abs = subset_root / row['edited_image']

        if not edited_abs.exists():
            raise FileNotFoundError(f'Edited image not found: {edited_abs}')

        with open(meta_path, 'w', encoding='utf-8') as f:
            json.dump(row, f, indent=2)

        glb_path = sample_dir / 'pseudo.glb'
        ply_path = sample_dir / 'pseudo.ply'
        if glb_path.exists() and ply_path.exists():
            print(f'[SKIP] sample_{i:03d} already exists')
            continue

        image = Image.open(edited_abs).convert('RGBA')
        outputs = pipeline.run(image, seed=args.seed)

        glb = postprocessing_utils.to_glb(
            outputs['gaussian'][0],
            outputs['mesh'][0],
            simplify=args.simplify,
            texture_size=args.texture_size,
        )
        glb.export(glb_path)
        outputs['gaussian'][0].save_ply(ply_path)
        print(f'[DONE] sample_{i:03d} -> {sample_dir}')

    print('Finished pseudo-asset generation.')


if __name__ == '__main__':
    main()
