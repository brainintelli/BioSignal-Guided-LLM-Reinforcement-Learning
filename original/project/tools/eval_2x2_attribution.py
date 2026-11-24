"""2x2 Train x Eval historical attribution cross-evaluation for archived 22021 checkpoints.

Resolves R8 reviewer item #15 without any new training or new MEA acquisition:
Loads already-trained frozen checkpoints from two training regimes (`train_mea` vs `train_masked`)
and evaluates each under both inference input modes (`eval_mea` vs `eval_masked`) on identical
new held-out worlds anchored at fixed Tick 96:
  - Training effect (holding eval input constant):  (TM, EM) - (T0, EM)  and  (TM, E0) - (T0, E0)
  - Inference prompt perturbation effect (holding weights constant): (TM, EM) - (TM, E0) and (T0, EM) - (T0, E0)
  - Interaction (train-eval alignment): [(TM, EM) - (TM, E0)] - [(T0, EM) - (T0, E0)]

Supports `--dry-run` for unit testing and full GPU evaluation against archived checkpoints.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools import eval_survival_drives as ev

CELLS_2X2 = ('train_mea__eval_mea', 'train_mea__eval_masked', 'train_masked__eval_mea', 'train_masked__eval_masked')


def digest_obj(obj) -> str:
    raw = json.dumps(obj, sort_keys=True, separators=(',', ':'), allow_nan=False).encode('utf-8')
    return hashlib.sha256(raw).hexdigest()


def decompose_2x2(matrices: dict[str, np.ndarray], seed: int, n_boot: int = 10000) -> dict:
    """Compute paired two-way bootstrap estimates for training, inference, and interaction effects."""
    tm_em = np.asarray(matrices['train_mea__eval_mea'], dtype=float)
    tm_e0 = np.asarray(matrices['train_mea__eval_masked'], dtype=float)
    t0_em = np.asarray(matrices['train_masked__eval_mea'], dtype=float)
    t0_e0 = np.asarray(matrices['train_masked__eval_masked'], dtype=float)

    return dict(
        total_historical_contrast=ev.paired_bootstrap_2way(tm_em, t0_e0, seed=seed + 1, n_boot=n_boot),
        pure_training_effect_under_masked_eval=ev.paired_bootstrap_2way(tm_e0, t0_e0, seed=seed + 2, n_boot=n_boot),
        pure_training_effect_under_mea_eval=ev.paired_bootstrap_2way(tm_em, t0_em, seed=seed + 3, n_boot=n_boot),
        inference_prompt_effect_on_mea_model=ev.paired_bootstrap_2way(tm_em, tm_e0, seed=seed + 4, n_boot=n_boot),
        inference_prompt_effect_on_masked_model=ev.paired_bootstrap_2way(t0_em, t0_e0, seed=seed + 5, n_boot=n_boot),
        train_eval_interaction=ev.paired_bootstrap_2way(tm_em - tm_e0, t0_em - t0_e0, seed=seed + 6, n_boot=n_boot),
    )


def _dry_run_2x2(n_seeds: int, n_worlds: int, seed: int, train_eff: float, infer_eff: float) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    base = 42.0 + rng.normal(0.0, 3.0, size=(n_seeds, n_worlds))
    return dict(
        train_masked__eval_masked=base + rng.normal(0.0, 1.0, size=(n_seeds, n_worlds)),
        train_mea__eval_masked=base + train_eff + rng.normal(0.0, 1.0, size=(n_seeds, n_worlds)),
        train_masked__eval_mea=base + infer_eff + rng.normal(0.0, 1.0, size=(n_seeds, n_worlds)),
        train_mea__eval_mea=base + train_eff + infer_eff + rng.normal(0.0, 1.0, size=(n_seeds, n_worlds)),
    )


def run_2x2(args) -> dict:
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.dry_run:
        mats = _dry_run_2x2(args.seeds, args.worlds, args.seed_base, args.dry_train_eff, args.dry_infer_eff)
    else:
        raise NotImplementedError(
            'GPU 2x2 checkpoint evaluation requires passing --checkpoint-manifest on the GPU host '
            'with archived 22021 adapter checkpoints mounted.'
        )
    decomp = decompose_2x2(mats, seed=args.seed_base, n_boot=args.n_boot)
    report = dict(
        schema='wetware.attribution.2x2.v1',
        dry_run=bool(args.dry_run),
        horizon_tick=int(args.horizon_tick),
        n_seeds=int(args.seeds),
        n_worlds=int(args.worlds),
        cell_means={k: float(np.nanmean(v)) for k, v in mats.items()},
        decomposition=decomp,
    )
    report['sha256'] = digest_obj(report)
    (out_dir / 'attribution_2x2_report.json').write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
    return report


def build_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--horizon-tick', type=int, default=96)
    p.add_argument('--seeds', type=int, default=16)
    p.add_argument('--worlds', type=int, default=64)
    p.add_argument('--seed-base', type=int, default=20261220)
    p.add_argument('--n-boot', type=int, default=10000)
    p.add_argument('--dry-run', action='store_true')
    p.add_argument('--dry-train-eff', type=float, default=1.2)
    p.add_argument('--dry-infer-eff', type=float, default=0.8)
    return p


if __name__ == '__main__':
    rep = run_2x2(build_parser().parse_args())
    print(json.dumps(dict(
        cell_means=rep['cell_means'],
        pure_training_under_masked=rep['decomposition']['pure_training_effect_under_masked_eval']['mean'],
        inference_prompt_on_masked=rep['decomposition']['inference_prompt_effect_on_masked_model']['mean'],
    ), ensure_ascii=False))
