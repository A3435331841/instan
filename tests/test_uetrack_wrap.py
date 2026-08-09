# -*- coding: utf-8 -*-
"""Tests for UETrack's opt-in ERP seam primitives."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from integrations.uetrack.erp_wrap import clip_box_erp, sample_target_erp  # noqa: E402


def column_image(height=6, width=8):
    image = np.zeros((height, width, 3), dtype=np.uint8)
    for column in range(width):
        image[:, column, :] = column
    return image


def test_horizontal_wrap():
    image = column_image()
    crop, factor = sample_target_erp(
        image, [-1.0, 2.0, 2.0, 2.0], 2.0, output_sz=None)
    assert factor == 1.0 and crop.shape == (4, 4, 3)
    assert crop[1, :, 0].tolist() == [6, 7, 0, 1]


def test_vertical_padding_only():
    image = column_image()
    crop, _ = sample_target_erp(
        image, [2.0, -1.0, 2.0, 2.0], 2.0, output_sz=None)
    assert np.all(crop[:2] == 0)
    assert crop[2, :, 0].tolist() == [1, 2, 3, 4]


def test_interior_crop_matches_slice():
    image = column_image(height=10, width=12)
    crop, _ = sample_target_erp(
        image, [4.0, 4.0, 2.0, 2.0], 2.0, output_sz=None)
    assert np.array_equal(crop, image[3:7, 3:7, :])


def test_box_wrap_retains_extent():
    box = clip_box_erp([-3.0, 2.0, 15.0, 5.0], 10, 100, margin=1)
    assert box == [97.0, 2.0, 15.0, 5.0]
    crossing = clip_box_erp([98.0, 2.0, 8.0, 5.0], 10, 100, margin=1)
    assert crossing == [98.0, 2.0, 8.0, 5.0]


def main():
    tests = [test_horizontal_wrap, test_vertical_padding_only,
             test_interior_crop_matches_slice, test_box_wrap_retains_extent]
    for test in tests:
        test()
        print('PASS', test.__name__)
    print(f'{len(tests)}/{len(tests)} passed')
    return 0


if __name__ == '__main__':
    sys.exit(main())
