#!/usr/bin/env python3
"""Install the pinned GRT-360 ERP adapter into an upstream UETrack checkout."""
import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path


HERE = Path(__file__).resolve().parent


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace', required=True, help='UETrack repository root')
    parser.add_argument('--output-root', required=True,
                        help='UETrack output root used by environment.local')
    parser.add_argument('--checkpoint', required=True,
                        help='uetrack_base.tar checkpoint path')
    args = parser.parse_args(argv)

    workspace = Path(args.workspace).resolve()
    output_root = Path(args.output_root).resolve()
    checkpoint = Path(args.checkpoint).resolve()
    anchor = json.loads((HERE / 'UPSTREAM.json').read_text(encoding='utf-8'))
    if sha256(checkpoint) != anchor['checkpoint_sha256']:
        raise ValueError('UETrack checkpoint SHA-256 does not match UPSTREAM.json')

    dataset_target = workspace / 'lib/test/evaluation/erpdataset.py'
    dataset_target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(HERE / 'erpdataset.py', dataset_target)
    shutil.copy2(HERE / 'run_erp.py', workspace / '_grt360_run_erp.py')
    shutil.copy2(HERE / 'file_protocol.py', workspace / '_grt360_file_protocol.py')
    shutil.copy2(HERE / 'erp_wrap.py', workspace / 'erp_wrap.py')

    registry = workspace / 'lib/test/evaluation/datasets.py'
    registry_text = registry.read_text(encoding='utf-8')
    entry = ('    erp=DatasetInfo(module=pt % "erp", '
             'class_name="ERPDataset", kwargs=dict()),\n')
    if 'erp=DatasetInfo(' not in registry_text:
        marker = '    tnl2k=DatasetInfo('
        index = registry_text.find(marker)
        if index < 0:
            raise ValueError('cannot locate tnl2k registry entry')
        line_end = registry_text.find('\n', index) + 1
        registry_text = registry_text[:line_end] + entry + registry_text[line_end:]
        registry.write_text(registry_text, encoding='utf-8')

    backbone = workspace / 'lib/models/uetrack/fastitpn.py'
    backbone_text = backbone.read_text(encoding='utf-8')
    unguarded = '    if pretrained:\n        checkpoint = torch.load(pretrain_type, map_location="cpu")'
    guarded = '    if pretrained and pretrain_type:\n        checkpoint = torch.load(pretrain_type, map_location="cpu")'
    replacements = backbone_text.count(unguarded)
    if replacements not in (0, 3):
        raise ValueError(f'unexpected fastitpn pretrain guard count: {replacements}')
    if replacements:
        backbone.write_text(backbone_text.replace(unguarded, guarded), encoding='utf-8')

    local_env = workspace / 'lib/test/evaluation/local.py'
    local_env.write_text(
        'from lib.test.evaluation.environment import EnvSettings\n\n'
        'def local_env_settings():\n'
        '    settings = EnvSettings()\n'
        f'    settings.prj_dir = {str(workspace)!r}\n'
        f'    settings.save_dir = {str(output_root)!r}\n'
        f'    settings.results_path = {str(output_root / "test/tracking_results")!r}\n'
        f'    settings.network_path = {str(output_root / "test/networks")!r}\n'
        f'    settings.segmentation_path = {str(output_root / "test/segmentation_results")!r}\n'
        f'    settings.result_plot_path = {str(output_root / "test/result_plots")!r}\n'
        '    return settings\n',
        encoding='utf-8',
    )

    checkpoint_target = (output_root / 'checkpoints/train/uetrack/uetrack_base'
                         / 'UETrack_ep0500.pth.tar')
    checkpoint_target.parent.mkdir(parents=True, exist_ok=True)
    if checkpoint_target.exists() or checkpoint_target.is_symlink():
        if checkpoint_target.resolve() != checkpoint:
            raise FileExistsError(f'refusing to replace {checkpoint_target}')
    else:
        checkpoint_target.symlink_to(checkpoint)
    print('INSTALLED', workspace)
    print('CHECKPOINT', checkpoint_target, sha256(checkpoint_target))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
