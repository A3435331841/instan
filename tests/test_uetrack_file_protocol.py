# -*- coding: utf-8 -*-
"""Unit tests for UETrack's standalone file protocol helpers."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from integrations.uetrack.file_protocol import (  # noqa: E402
    frame_paths,
    parse_box,
    write_rows,
)


def test_parse_inline_and_file(tmp_path):
    assert parse_box('1, 2, 3, 4') == [1.0, 2.0, 3.0, 4.0]
    path = tmp_path / 'init.txt'
    path.write_text('\n5\t6\t7\t8\n', encoding='utf-8')
    assert parse_box(str(path)) == [5.0, 6.0, 7.0, 8.0]


def test_natural_frame_order_and_filter(tmp_path):
    for name in ('10.jpg', '2.jpg', '1.png', 'ignore.txt'):
        (tmp_path / name).write_bytes(b'x')
    assert [path.name for path in frame_paths(tmp_path)] == [
        '1.png', '2.jpg', '10.jpg']


def test_atomic_numeric_output(tmp_path):
    path = tmp_path / 'nested' / 'results.txt'
    write_rows(path, [[1, 2, 3, 4], [5.25, 6, 7, 8]])
    assert path.read_text(encoding='utf-8').splitlines() == [
        '1.000000000000,2.000000000000,3.000000000000,4.000000000000',
        '5.250000000000,6.000000000000,7.000000000000,8.000000000000',
    ]
    assert not path.with_name(path.name + '.tmp').exists()
