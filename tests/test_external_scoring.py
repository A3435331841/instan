# -*- coding: utf-8 -*-
"""Regression tests for strict external-result OPE scoring."""
import json
import sys
import tempfile
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.score_external_results import (  # noqa: E402
    aggregate,
    score_sequence,
    write_outputs,
)


def make_sequence(root):
    seq = Path(root) / 'data' / '0001'
    images = seq / 'image'
    images.mkdir(parents=True)
    labels = {}
    for index in range(3):
        name = f'{index + 1:06d}.jpg'
        Image.new('RGB', (100, 50), (index * 20, 0, 0)).save(images / name)
        labels[name] = {
            'bbox': {'cx': 15 + index, 'cy': 15, 'w': 10, 'h': 10},
        }
    (seq / 'label.json').write_text(
        json.dumps(labels), encoding='utf-8')
    return seq


def make_results(root, rows):
    result_root = Path(root) / 'results'
    result_root.mkdir()
    (result_root / '0001.txt').write_text(
        '\n'.join(','.join(str(value) for value in row) for row in rows) + '\n',
        encoding='utf-8')
    (result_root / '0001_time.txt').write_text('0\n0.1\n0.2\n', encoding='utf-8')
    return result_root


def test_perfect_score_and_timing():
    with tempfile.TemporaryDirectory() as tmp:
        seq = make_sequence(tmp)
        results = make_results(tmp, [(10, 10, 10, 10),
                                     (11, 10, 10, 10),
                                     (12, 10, 10, 10)])
        row = score_sequence('perfect', results, seq)
        assert row['sr'] == 1.0 and row['auc'] == 1.0
        assert row['first_frame_linf'] == 0.0
        assert abs(row['fps'] - (2.0 / 0.3)) < 1e-9


def test_strict_length_rejected():
    with tempfile.TemporaryDirectory() as tmp:
        seq = make_sequence(tmp)
        results = make_results(tmp, [(10, 10, 10, 10), (11, 10, 10, 10)])
        try:
            score_sequence('short', results, seq)
        except ValueError as exc:
            assert 'prediction rows 2 != GT rows 3' in str(exc)
        else:
            raise AssertionError('short external result must be rejected')


def test_summary_and_manifest():
    with tempfile.TemporaryDirectory() as tmp:
        rows = [
            {'tracker': 'A', 'sequence': '0001', 'n_frames': 3,
             'sr': 1.0, 'sr_dual': 1.0, 'auc': 1.0, 'auc_dual': 1.0,
             'fps': 5.0, 'first_frame_linf': 0.0, 'result_path': 'a'},
            {'tracker': 'B', 'sequence': '0001', 'n_frames': 3,
             'sr': 0.0, 'sr_dual': 0.0, 'auc': 0.1, 'auc_dual': 0.1,
             'fps': 10.0, 'first_frame_linf': 0.0, 'result_path': 'b'},
        ]
        assert aggregate(rows)[0]['tracker'] == 'A'
        payload = write_outputs(
            rows, Path(tmp) / 'out', {'A': '.', 'B': '.'}, ['0001'])
        assert payload['winner_by_ordinary_auc'] == 'A'
        assert payload['protocol']['first_frame_excluded'] is True


def main():
    tests = [test_perfect_score_and_timing, test_strict_length_rejected,
             test_summary_and_manifest]
    for test in tests:
        test()
        print('PASS', test.__name__)
    print(f'{len(tests)}/{len(tests)} passed')
    return 0


if __name__ == '__main__':
    sys.exit(main())
