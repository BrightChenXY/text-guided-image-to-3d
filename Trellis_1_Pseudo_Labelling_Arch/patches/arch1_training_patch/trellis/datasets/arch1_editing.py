import os
from typing import *
import numpy as np
import torch
from torch.utils.data import Dataset

from ..modules.sparse.basic import SparseTensor
from ..utils.data_utils import load_balanced_group_indices


class _Arch1Base(Dataset):
    """
    Base dataset for preprocessed Architecture-1 samples.
    Expects per-sample folders containing:
      z_ss_target.npz
      z_slat_target.npz
      e_img.pt
      e_text.pt
    """
    def __init__(self, roots: str):
        super().__init__()
        self.roots = roots.split(',')
        self.sample_dirs = []
        for root in self.roots:
            root = root.strip()
            if not root:
                continue

            # Either a directory of sample_* dirs or a single sample dir
            if os.path.exists(os.path.join(root, 'z_ss_target.npz')):
                self.sample_dirs.append(root)
            else:
                for name in sorted(os.listdir(root)):
                    path = os.path.join(root, name)
                    if name.startswith('sample_') and os.path.isdir(path):
                        self.sample_dirs.append(path)

        if len(self.sample_dirs) == 0:
            raise RuntimeError(f'No sample directories found in roots={roots}')

        self.value_range = (0, 1)

    def __len__(self):
        return len(self.sample_dirs)

    def _load_cond(self, sample_dir: str) -> Dict[str, torch.Tensor]:
        e_img = torch.load(os.path.join(sample_dir, 'e_img.pt'), map_location='cpu')
        e_text = torch.load(os.path.join(sample_dir, 'e_text.pt'), map_location='cpu')
        # saved as [1, T, D]
        if e_img.ndim == 3 and e_img.shape[0] == 1:
            e_img = e_img[0]
        if e_text.ndim == 3 and e_text.shape[0] == 1:
            e_text = e_text[0]
        return {
            'img': e_img.float(),
            'text': e_text.float(),
        }


class Arch1ConditionedSparseStructureLatent(_Arch1Base):
    """
    Stage-1 dataset:
      x_0 <- z_ss_target.npz['mean']
      cond <- {'img': e_img, 'text': e_text}
    """
    def __init__(self, roots: str):
        super().__init__(roots)

    def __getitem__(self, index):
        sample_dir = self.sample_dirs[index]
        z = np.load(os.path.join(sample_dir, 'z_ss_target.npz'))['mean']
        return {
            'x_0': torch.tensor(z).float(),
            'cond': self._load_cond(sample_dir),
        }

    @staticmethod
    def collate_fn(batch):
        return {
            'x_0': torch.stack([b['x_0'] for b in batch]),
            'cond': {
                'img': torch.stack([b['cond']['img'] for b in batch]),
                'text': torch.stack([b['cond']['text'] for b in batch]),
            },
        }


class Arch1ConditionedSLat(_Arch1Base):
    """
    Stage-2 dataset:
      x_0 <- z_slat_target.npz['coords'], ['feats']
      cond <- {'img': e_img, 'text': e_text}
    """
    def __init__(
        self,
        roots: str,
        *,
        normalization: Optional[dict] = None,
        max_num_voxels: int = 32768,
    ):
        super().__init__(roots)
        self.normalization = normalization
        self.max_num_voxels = max_num_voxels

        if self.normalization is not None:
            self.mean = torch.tensor(self.normalization['mean']).reshape(1, -1)
            self.std = torch.tensor(self.normalization['std']).reshape(1, -1)

        # Filter by voxel count if desired
        filtered = []
        for d in self.sample_dirs:
            z = np.load(os.path.join(d, 'z_slat_target.npz'))
            if z['coords'].shape[0] <= self.max_num_voxels:
                filtered.append(d)
        self.sample_dirs = filtered
        if len(self.sample_dirs) == 0:
            raise RuntimeError('No valid Stage-2 samples left after max_num_voxels filter')

        self.loads = []
        for d in self.sample_dirs:
            z = np.load(os.path.join(d, 'z_slat_target.npz'))
            self.loads.append(int(z['coords'].shape[0]))

    def __getitem__(self, index):
        sample_dir = self.sample_dirs[index]
        data = np.load(os.path.join(sample_dir, 'z_slat_target.npz'))
        coords = torch.tensor(data['coords']).int()
        feats = torch.tensor(data['feats']).float()
        if self.normalization is not None:
            feats = (feats - self.mean) / self.std
        return {
            'coords': coords,
            'feats': feats,
            'cond': self._load_cond(sample_dir),
        }

    @staticmethod
    def collate_fn(batch, split_size=None):
        if split_size is None:
            group_idx = [list(range(len(batch)))]
        else:
            group_idx = load_balanced_group_indices([b['coords'].shape[0] for b in batch], split_size)

        packs = []
        for group in group_idx:
            sub_batch = [batch[i] for i in group]
            pack = {}

            coords = []
            feats = []
            layout = []
            start = 0

            for i, b in enumerate(sub_batch):
                coords.append(torch.cat([
                    torch.full((b['coords'].shape[0], 1), i, dtype=torch.int32),
                    b['coords']
                ], dim=-1))
                feats.append(b['feats'])
                layout.append(slice(start, start + b['coords'].shape[0]))
                start += b['coords'].shape[0]

            coords = torch.cat(coords)
            feats = torch.cat(feats)

            pack['x_0'] = SparseTensor(coords=coords, feats=feats)
            pack['x_0']._shape = torch.Size([len(group), *sub_batch[0]['feats'].shape[1:]])
            pack['x_0'].register_spatial_cache('layout', layout)

            pack['cond'] = {
                'img': torch.stack([b['cond']['img'] for b in sub_batch]),
                'text': torch.stack([b['cond']['text'] for b in sub_batch]),
            }

            packs.append(pack)

        if split_size is None:
            return packs[0]
        return packs
