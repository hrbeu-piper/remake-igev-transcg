"""Independent reference-only TransCG loader. Never opens K/T or source RGB.

Only pure PNG/provider/spatial helpers are reused from dataset.py; this class
does not inherit or instantiate TransCGPairsDataset. GT depth is always a
target. The declared gt_privileged binary mask is the sole privileged input.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .dataset import DatasetError, SpatialTransform, _read_png, _relative_provider_map
from .single_manifest import CAMERA_SCALES, DATA_PROTOCOL, MASK_PROTOCOL, load_manifest

INPUT_KEYS = frozenset(('rgb_ref', 'raw_depth', 'raw_valid', 'relative_depth', 'mask'))


class TransCGSingleDataset(Dataset):
    def __init__(self, manifest, split, *, relative_depth_provider, mask_protocol=MASK_PROTOCOL,
                 output_size=(480, 640), diagnostic_only=False, allow_test=False):
        if mask_protocol != MASK_PROTOCOL:
            raise DatasetError('This protocol requires explicit gt_privileged; predicted masks need a separate protocol')
        if not callable(relative_depth_provider):
            raise DatasetError('An external RGB-only relative-depth provider is required')
        if split not in ('train', 'val', 'test') or (split == 'test' and not allow_test):
            raise DatasetError('Test loading requires explicit allow_test=True after model/source freeze')
        document = load_manifest(manifest, diagnostic_only=diagnostic_only)
        self.root = Path(document['dataset_root']).resolve(strict=True)
        if not self.root.is_dir():
            raise DatasetError('Dataset root must be a directory')
        self.rows = tuple(row for row in document['samples'] if row['split'] == split)
        if not self.rows:
            raise DatasetError(f'No {split} reference rows')
        self.split, self.mask_protocol = split, mask_protocol
        self.relative_depth_provider = relative_depth_provider
        self.spatial = SpatialTransform(output_size=output_size, crop=None)
        self.data_protocol = DATA_PROTOCOL
        self.diagnostic_only = diagnostic_only

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        try:
            return self._load(row)
        except DatasetError:
            raise
        except (OSError, KeyError, TypeError, ValueError, RuntimeError) as exc:
            raise DatasetError(f'{row["sample_id"]}: invalid reference: {exc}') from exc

    def _load(self, row):
        sample_id = row['sample_id']
        observation, supervision = row['inputs'], row['supervision']
        scale = CAMERA_SCALES[row['reference_camera_id']]
        rgb_np = _read_png(self.root, observation['rgb'], kind='rgb', label=f'{sample_id} RGB')
        raw_np = _read_png(self.root, observation['raw_depth'], kind='depth', label=f'{sample_id} raw')
        mask_np = _read_png(self.root, observation['transparent_mask'], kind='mask', label=f'{sample_id} privileged mask')
        native_hw = tuple(rgb_np.shape[:2])
        if raw_np.shape != native_hw or mask_np.shape != native_hw:
            raise DatasetError(f'{sample_id}: reference RGB/raw/mask dimensions differ')
        relative = _relative_provider_map(self.relative_depth_provider, sample_id, rgb_np, self.spatial)
        rgb = torch.from_numpy(rgb_np.copy()).permute(2, 0, 1).float() / 255.
        raw = torch.from_numpy(raw_np.astype(np.float32)).unsqueeze(0) * scale
        raw_valid = torch.from_numpy(raw_np > 0).unsqueeze(0)
        mask = torch.from_numpy(mask_np > 0).unsqueeze(0)
        # Build every forward input before opening GT depth. This is not a
        # security boundary by itself; invariance under changed GT is tested.
        inputs = {'rgb_ref': self.spatial.apply(rgb, nearest=False),
                  'raw_depth': self.spatial.apply(raw, nearest=True),
                  'raw_valid': self.spatial.apply(raw_valid, nearest=True),
                  'relative_depth': relative,
                  'mask': self.spatial.apply(mask, nearest=True).float()}
        gt_np = _read_png(self.root, supervision['depth_gt'], kind='depth', label=f'{sample_id} GT target')
        if gt_np.shape != native_hw:
            raise DatasetError(f'{sample_id}: GT and reference dimensions differ')
        gt = torch.from_numpy(gt_np.astype(np.float32)).unsqueeze(0) * scale
        valid = torch.from_numpy(gt_np > 0).unsqueeze(0)
        return {'inputs': inputs,
                'targets': {'depth_gt': self.spatial.apply(gt, nearest=True),
                            'gt_valid': self.spatial.apply(valid, nearest=True),
                            'transparent_mask': self.spatial.apply(mask, nearest=True),
                            'depth_gt_original': gt, 'gt_valid_original': valid,
                            'transparent_mask_original': mask},
                'metadata': {'sample_id': sample_id, 'split': row['split'], 'scene_id': row['scene_id'],
                             'view_id': row['view_id'], 'scene_type': row['scene_type'],
                             'reference_camera_id': row['reference_camera_id'],
                             'data_protocol': DATA_PROTOCOL, 'mask_protocol': MASK_PROTOCOL,
                             'privileged_mask_input': True, 'reference_native_hw': torch.tensor(native_hw),
                             'raw_valid_original': raw_valid, 'depth_scale_m': scale}}
