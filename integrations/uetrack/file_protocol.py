#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run seam-aware UETrack with a simple file-based tracking protocol."""
import argparse
import os
import re
import sys
import time
from pathlib import Path


IMAGE_SUFFIXES = {'.bmp', '.jpeg', '.jpg', '.png', '.tif', '.tiff', '.webp'}


def parse_box(value):
    """Parse x,y,w,h from a file path or an inline string."""
    candidate = Path(value)
    text = candidate.read_text(encoding='utf-8') if candidate.is_file() else value
    first_line = next((line for line in text.splitlines() if line.strip()), '')
    fields = re.split(r'[\s,;]+', first_line.strip())
    if len(fields) != 4:
        raise ValueError('initial box must contain exactly four values: x,y,w,h')
    try:
        box = [float(field) for field in fields]
    except ValueError as exc:
        raise ValueError('initial box contains a non-numeric value') from exc
    if not all(value == value and abs(value) != float('inf') for value in box):
        raise ValueError('initial box contains a non-finite value')
    if box[2] <= 0.0 or box[3] <= 0.0:
        raise ValueError('initial box width and height must be positive')
    return box


def natural_key(path):
    return [int(part) if part.isdigit() else part.lower()
            for part in re.split(r'(\d+)', path.name)]


def frame_paths(root):
    """Return naturally sorted image files from one non-recursive directory."""
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f'frame directory does not exist: {root}')
    frames = sorted(
        (path for path in root.iterdir()
         if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES),
        key=natural_key,
    )
    if not frames:
        raise FileNotFoundError(f'no supported images found in {root}')
    return frames


def write_rows(path, rows):
    """Atomically write numeric rows as comma-separated text."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    with open(temporary, 'w', encoding='utf-8', newline='\n') as handle:
        for row in rows:
            # Twelve fractional digits preserve the official first-frame box
            # precision while remaining accepted by common VOT/OPE readers.
            handle.write(','.join(f'{float(value):.12f}' for value in row) + '\n')
    temporary.replace(path)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--frames', required=True, help='directory of ordered frames')
    parser.add_argument('--init', required=True,
                        help='text file or inline x,y,w,h initial box')
    parser.add_argument('--out', required=True, help='output x,y,w,h result file')
    parser.add_argument('--timing', default=None,
                        help='optional per-frame timing output file')
    parser.add_argument('--workspace', default='/opt/uetrack',
                        help='installed UETrack repository root')
    parser.add_argument('--parameter', default='uetrack_base')
    parser.add_argument('--gpu', default='0', help='CUDA_VISIBLE_DEVICES value')
    parser.add_argument('--no-erp-wrap', action='store_true',
                        help='use upstream zero padding instead of ERP wrapping')
    args = parser.parse_args(argv)

    # CUDA visibility must be set before importing torch through UETrack.
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    workspace = Path(args.workspace).resolve()
    if not workspace.is_dir():
        raise FileNotFoundError(f'UETrack workspace does not exist: {workspace}')
    os.chdir(workspace)
    sys.path.insert(0, str(workspace))

    import cv2 as cv
    from lib.test.evaluation import Tracker

    if not args.no_erp_wrap:
        from erp_wrap import clip_box_erp, sample_target_erp
        import lib.test.tracker.uetrack as tracker_module
        tracker_module.sample_target = sample_target_erp
        tracker_module.clip_box = clip_box_erp

    frames = frame_paths(args.frames)
    initial_box = parse_box(args.init)
    wrapper = Tracker('uetrack', args.parameter, 'erp', None)
    params = wrapper.get_parameters()
    params.debug = 0
    tracker = wrapper.create_tracker(params)

    boxes = [initial_box]
    timings = []
    first = cv.imread(str(frames[0]), cv.IMREAD_COLOR)
    if first is None:
        raise ValueError(f'failed to decode frame: {frames[0]}')
    first = cv.cvtColor(first, cv.COLOR_BGR2RGB)
    start = time.perf_counter()
    initialized = tracker.initialize(
        first, {'init_bbox': initial_box, 'seq_name': Path(args.frames).name})
    timings.append(time.perf_counter() - start)
    previous = initialized or {}

    for frame_path in frames[1:]:
        image = cv.imread(str(frame_path), cv.IMREAD_COLOR)
        if image is None:
            raise ValueError(f'failed to decode frame: {frame_path}')
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        start = time.perf_counter()
        output = tracker.track(image, {'previous_output': previous})
        timings.append(time.perf_counter() - start)
        box = output.get('target_bbox')
        if box is None or len(box) != 4:
            raise RuntimeError(f'tracker returned an invalid box for {frame_path}')
        boxes.append([float(value) for value in box])
        previous = output

    write_rows(args.out, boxes)
    if args.timing:
        write_rows(args.timing, ([value] for value in timings))
    elapsed = sum(timings[1:])
    fps = (len(timings) - 1) / elapsed if elapsed > 0.0 else 0.0
    print(f'COMPLETE frames={len(frames)} fps={fps:.4f} output={Path(args.out)}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
