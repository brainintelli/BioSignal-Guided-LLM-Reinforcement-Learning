"""Turnkey campaign runner for existing-data wetware PPO training (Stage 0 -> Stage 3 -> Stage 4).

Uses ONLY existing workspace recordings (zero new MEA collection):
- Primary Organoid (Stage 3): `22021_1123_oddball_clean` (`rho_lib = 0.441`)
- Independent Organoid Replication (Stage 4): `28080_1116_long_spontaneous` (`rho_lib = 0.350`)

Supports `--dry-run` for local end-to-end verification on extracted real library JSONs, and full GPU
execution when `--model-config` and `--godot` are supplied on the training node.
"""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools import build_existing_libraries
from tools import run_wetware_stage0 as s0
from tools import run_wetware_stage3 as s3


def run_campaign(args):
    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)
    lib_dir = Path(args.libraries_dir)
    if not (lib_dir / 'gate_manifest.json').is_file():
        build_existing_libraries.build_all(lib_dir)

    manifest = json.loads((lib_dir / 'gate_manifest.json').read_text(encoding='utf-8'))
    p_ds = manifest['datasets'][manifest['primary_dataset']]
    r_ds = manifest['datasets'][manifest['replication_dataset']]
    if not p_ds['real_gate']['dual_drive_unlocked'] or not r_ds['real_gate']['dual_drive_unlocked']:
        raise RuntimeError('Primary or replication existing dataset did not pass offline dual gates')

    # Stage 0
    s0_dir = out_root / 'stage0'
    s0_argv = ['--mode', args.mode, '--out', str(s0_dir), '--n-boot', str(args.n_boot)]
    if args.dry_run:
        s0_argv.extend(['--dry-run', '--tune-seeds', str(args.seeds or 16),
                        '--confirm-seeds', str(args.seeds or 32), '--eval-worlds', str(args.eval_worlds)])
    else:
        if not args.model_config or not args.godot:
            raise ValueError('--model-config and --godot are required for full GPU execution')
        s0_argv.extend(['--model-config', str(args.model_config), '--godot', str(args.godot),
                        '--project', str(args.project)])
    s0_rep = s0.run_stage0(s0.build_parser().parse_args(s0_argv))

    # Stage 3 (22021_1123) + Stage 4 (28080_1116_long)
    p_base = lib_dir / manifest['primary_dataset']
    r_base = lib_dir / manifest['replication_dataset']
    s3_dir = out_root / 'stage3_and_stage4'
    s3_argv = [
        '--mode', args.mode,
        '--stage0-report', str(s0_dir / 'stage0_report.json'),
        '--out', str(s3_dir),
        '--habituation-library', str(p_base / 'habituation_library.json'),
        '--evoked-library', str(p_base / 'evoked_library.json'),
        '--ctime-habituation-library', str(p_base / 'ctime_habituation_library.json'),
        '--ctime-evoked-library', str(p_base / 'ctime_evoked_library.json'),
        '--replication-habituation-library', str(r_base / 'habituation_library.json'),
        '--replication-evoked-library', str(r_base / 'evoked_library.json'),
        '--replication-ctime-habituation-library', str(r_base / 'ctime_habituation_library.json'),
        '--replication-ctime-evoked-library', str(r_base / 'ctime_evoked_library.json'),
        '--rho-lib', f"{p_ds['real_gate']['rho_lib']:.4f}",
        '--replication-rho-lib', f"{r_ds['real_gate']['rho_lib']:.4f}",
        '--eval-worlds', str(args.eval_worlds),
        '--n-boot', str(args.n_boot),
    ]
    if args.seeds:
        s3_argv.extend(['--seeds', str(args.seeds)])
    if args.dry_run:
        s3_argv.append('--dry-run')
    else:
        s3_argv.extend(['--model-config', str(args.model_config), '--godot', str(args.godot),
                        '--project', str(args.project)])
    s3_rep = s3.run_stage3(s3.build_parser().parse_args(s3_argv))
    return dict(stage0=s0_rep, stage3_stage4=s3_rep)


def build_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode', choices=('explore', 'homeo', 'dual'), default='dual')
    p.add_argument('--libraries-dir', type=Path, default=ROOT / 'output/existing_wetware_libraries')
    p.add_argument('--out', type=Path, default=ROOT / 'output/existing_wetware_campaign_20261123')
    p.add_argument('--dry-run', action='store_true')
    p.add_argument('--model-config', type=Path)
    p.add_argument('--godot', type=Path)
    p.add_argument('--project', type=Path, default=ROOT)
    p.add_argument('--seeds', type=int)
    p.add_argument('--eval-worlds', type=int, default=48)
    p.add_argument('--n-boot', type=int, default=2000)
    return p


if __name__ == '__main__':
    res = run_campaign(build_parser().parse_args())
    s3r = res['stage3_stage4']
    print(json.dumps(dict(
        stage0_confirmed=res['stage0']['confirmed'],
        frozen_tier=res['stage0']['frozen_tier'],
        final_verdict=s3r['outcome']['verdict'],
        reason=s3r['outcome']['reason'],
        stage3_B_minus_C=s3r['comparisons']['B_minus_C'],
        stage3_B_minus_Ctime=s3r['comparisons']['B_minus_Ctime'],
        stage4_verdict=s3r['stage4_replication']['outcome']['verdict'] if s3r.get('stage4_replication') else None,
    ), indent=2, ensure_ascii=False))
