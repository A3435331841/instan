#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run an installed UETrack adapter on selected 360VOT ERP sequences."""
import argparse
import os
import sys
from pathlib import Path


def _line_count(path):
    with open(path, 'rb') as handle:
        return sum(1 for line in handle if line.strip())


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace', required=True, help='UETrack repository root')
    parser.add_argument('--data', required=True, help='extracted 360VOT root')
    parser.add_argument('--seqs', required=True, help='comma-separated sequence ids')
    parser.add_argument('--gpu', default='0', help='CUDA_VISIBLE_DEVICES value')
    parser.add_argument('--tracker', default='uetrack')
    parser.add_argument('--parameter', default='uetrack_base')
    parser.add_argument('--skip-existing', action='store_true')
    args = parser.parse_args(argv)

    workspace = Path(args.workspace).resolve()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    os.environ['GRT360_DATA_ROOT'] = str(Path(args.data).resolve())
    os.environ['GRT360_SEQUENCES'] = ','.join(
        value.strip().zfill(4) for value in args.seqs.split(',') if value.strip())
    os.chdir(workspace)
    sys.path.insert(0, str(workspace))

    from lib.test.evaluation import Tracker, get_dataset
    from lib.test.evaluation.running import run_sequence

    dataset = get_dataset('erp')
    tracker = Tracker(args.tracker, args.parameter, 'erp', None)
    print('WORKSPACE', workspace, flush=True)
    print('DATA', os.environ['GRT360_DATA_ROOT'], flush=True)
    print('SEQUENCES', [sequence.name for sequence in dataset], flush=True)
    print('RESULTS', tracker.results_dir, flush=True)

    for sequence in dataset:
        result_path = Path(tracker.results_dir) / f'{sequence.name}.txt'
        timing_path = Path(tracker.results_dir) / f'{sequence.name}_time.txt'
        complete = (result_path.is_file() and timing_path.is_file()
                    and _line_count(result_path) == len(sequence.frames)
                    and _line_count(timing_path) == len(sequence.frames))
        if args.skip_existing and complete:
            print('SKIP_COMPLETE', sequence.name, len(sequence.frames), flush=True)
            continue
        print('RUN', sequence.name, len(sequence.frames), flush=True)
        run_sequence(sequence, tracker, debug=False)
        if (_line_count(result_path) != len(sequence.frames)
                or _line_count(timing_path) != len(sequence.frames)):
            raise RuntimeError(f'incomplete UETrack result: {sequence.name}')
        print('DONE', sequence.name, flush=True)
    print('ALL_DONE', flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
