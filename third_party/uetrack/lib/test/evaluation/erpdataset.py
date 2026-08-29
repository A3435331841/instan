# -*- coding: utf-8 -*-
"""UETrack dataset adapter for extracted 360VOT ERP sequences."""
import json
import os
import re
from pathlib import Path

import numpy as np

from lib.test.evaluation.data import BaseDataset, Sequence, SequenceList


_DEFAULT_DATA_ROOT = '/data/projects/instan/data360'
_FRAME_DIRS = ('image', 'img', 'frames', 'imgs', 'JPEGImages')
_IMAGE_SUFFIXES = ('.jpg', '.jpeg', '.png', '.bmp')
_NUM_PART = re.compile(r'(\d+)')


def _natural_key(name):
    return [int(value) if value.isdigit() else value.lower()
            for value in _NUM_PART.split(name)]


def _find_payload(sequence_root):
    """Return ``(label.json, frame_dir)`` for normal or nested archives."""
    sequence_root = Path(sequence_root)
    labels = sorted(sequence_root.rglob('label.json'),
                    key=lambda path: (len(path.parts), str(path)))
    for label_path in labels:
        parent = label_path.parent
        for name in _FRAME_DIRS:
            frame_dir = parent / name
            if frame_dir.is_dir() and any(
                    item.is_file() and item.suffix.lower() in _IMAGE_SUFFIXES
                    for item in frame_dir.iterdir()):
                return label_path, frame_dir
    return None, None


class ERPDataset(BaseDataset):
    """360VOT full-frame ERP dataset with first-frame initialization boxes."""

    def __init__(self):
        super().__init__()
        self.base_path = Path(os.environ.get(
            'GRT360_DATA_ROOT', _DEFAULT_DATA_ROOT)).resolve()
        requested = os.environ.get('GRT360_SEQUENCES', '').strip()
        self.requested = ({value.strip().zfill(4)
                           for value in requested.split(',') if value.strip()}
                          if requested else None)
        self.sequence_list = self._get_sequence_list()

    def __len__(self):
        return len(self.sequence_list)

    def _get_sequence_list(self):
        if not self.base_path.is_dir():
            raise FileNotFoundError(f'ERP data root not found: {self.base_path}')
        sequences = []
        for child in sorted(self.base_path.iterdir(), key=lambda path: _natural_key(path.name)):
            if not child.is_dir() or not child.name.isdigit():
                continue
            name = child.name.zfill(4)
            if self.requested is not None and name not in self.requested:
                continue
            label_path, frame_dir = _find_payload(child)
            if label_path is not None and frame_dir is not None:
                sequences.append((name, label_path, frame_dir))
        if self.requested is not None:
            found = {name for name, _, _ in sequences}
            missing = sorted(self.requested - found)
            if missing:
                raise FileNotFoundError(
                    'requested ERP sequences are incomplete: ' + ','.join(missing))
        return sequences

    def get_sequence_list(self):
        return SequenceList([self._construct_sequence(*item)
                             for item in self.sequence_list])

    def _construct_sequence(self, sequence_name, label_path, frame_dir):
        with open(label_path, 'r', encoding='utf-8') as handle:
            labels = json.load(handle)

        gt_by_name = {}
        for frame_name, item in labels.items():
            box = item.get('bbox') if isinstance(item, dict) else None
            if not box:
                continue
            cx = float(box['cx'])
            cy = float(box['cy'])
            width = float(box['w'])
            height = float(box['h'])
            gt_by_name[frame_name] = [
                cx - width / 2.0, cy - height / 2.0, width, height,
            ]

        frame_names = sorted(
            (path.name for path in frame_dir.iterdir()
             if path.is_file() and path.suffix.lower() in _IMAGE_SUFFIXES),
            key=_natural_key,
        )
        frames, ground_truth = [], []
        for frame_name in frame_names:
            if frame_name in gt_by_name:
                frames.append(str(frame_dir / frame_name))
                ground_truth.append(gt_by_name[frame_name])
        if not frames:
            raise ValueError(f'no frames matched GT in {label_path.parent}')
        return Sequence(
            sequence_name,
            frames,
            'erp',
            np.asarray(ground_truth, dtype=np.float64).reshape(-1, 4),
        )
