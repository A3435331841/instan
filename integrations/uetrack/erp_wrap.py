# -*- coding: utf-8 -*-
"""Seam-aware crop and box clipping primitives for UETrack ERP inference."""
import math

import cv2 as cv
import numpy as np


def sample_target_erp(image, target_box, search_area_factor, output_sz=None):
    """Crop with horizontal circular wrap and vertical constant padding."""
    if isinstance(target_box, list):
        x, y, width, height = target_box
    else:
        x, y, width, height = target_box.tolist()
    crop_size = math.ceil(math.sqrt(width * height) * search_area_factor)
    if crop_size < 1:
        raise ValueError('Too small bounding box.')

    x1 = round(x + 0.5 * width - crop_size * 0.5)
    y1 = round(y + 0.5 * height - crop_size * 0.5)
    x2 = x1 + crop_size
    y2 = y1 + crop_size
    image_height, image_width = image.shape[:2]

    source_y1 = max(0, y1)
    source_y2 = min(image_height, y2)
    strip = image[source_y1:source_y2, :, :]
    # Keep the overwhelmingly common interior case on NumPy's zero-copy slice
    # path.  Seam crops concatenate at most two contiguous pieces; only crops
    # wider than the panorama need general modular advanced indexing.
    if 0 <= x1 and x2 <= image_width:
        crop = strip[:, x1:x2, :]
    elif crop_size <= image_width:
        start = x1 % image_width
        first_width = min(crop_size, image_width - start)
        pieces = [strip[:, start:start + first_width, :]]
        if first_width < crop_size:
            pieces.append(strip[:, :crop_size - first_width, :])
        crop = np.concatenate(pieces, axis=1)
    else:
        x_indices = np.mod(np.arange(x1, x2, dtype=np.int64), image_width)
        crop = strip[:, x_indices, :]
    top = max(0, -y1)
    bottom = max(0, y2 - image_height)
    crop = cv.copyMakeBorder(crop, top, bottom, 0, 0, cv.BORDER_CONSTANT)
    if crop.shape[:2] != (crop_size, crop_size):
        raise RuntimeError(
            f'ERP crop shape {crop.shape[:2]} != {(crop_size, crop_size)}')

    if output_sz is None:
        return crop, 1.0
    resize_factor = output_sz / crop_size
    return cv.resize(crop, (output_sz, output_sz)), resize_factor


def clip_box_erp(box, image_height, image_width, margin=0):
    """Keep latitude valid while retaining seam-crossing horizontal extent."""
    x, y, width, height = (float(value) for value in box)
    width = min(float(image_width), max(float(margin), width))
    x = x % float(image_width)
    y2 = min(max(float(margin), y + height), float(image_height))
    y = min(max(0.0, y), float(image_height - margin))
    height = max(float(margin), y2 - y)
    return [x, y, width, height]
