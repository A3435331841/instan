# -*- coding: utf-8 -*-
"""Run UETrack RGB tracking on the ERP dataset (3 sequences), print AUC/SR/FPS per sequence."""
import sys
import os

WORKSPACE = '/data/projects/instan_check/UETrack'
sys.path.insert(0, WORKSPACE)
os.chdir(WORKSPACE)
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '0')

from lib.test.evaluation import get_dataset, Tracker
from lib.test.evaluation.running import run_sequence


def main():
    dataset = get_dataset('erp')
    names = [s.name for s in dataset]
    print('ERP sequences:', names)

    tracker = Tracker('uetrack', 'uetrack_base', 'erp', None)
    print('Tracker:', tracker.name, tracker.parameter_name)
    print('results_dir:', tracker.results_dir)

    for seq in dataset:
        print('==== running', seq.name, 'frames=', len(seq.frames), '====')
        run_sequence(seq, tracker, debug=False)
        sys.stdout.flush()

    print('ALL DONE')


if __name__ == '__main__':
    main()