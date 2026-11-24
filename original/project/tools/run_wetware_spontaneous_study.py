"""Phase 2-3 six-arm paired spontaneous wetware dynamics study & cross-chip replication harness.

Zero new MEA acquisition required: loads SHA-256 verified Phase 0 spontaneous dynamics
libraries (`22554`, `27701`, `28073` passing Gate-S; plus optional oscillator regime `22021`/`28080`)
and compares 6 paired training arms under identical 108-dim frozen evaluation across E1-E4 suites:
  - A (`static`): pure static epsilon=0.10 baseline
  - B (`real`): real organoid spatiotemporal dynamics (M1 + optional M2)
  - C (`shuffled`): time-permuted organoid dynamics (same marginal distribution)
  - G1 (`white`): variance-matched i.i.d. Gaussian noise
  - G2 (`iaaft`): exact spectrum-and-histogram-matched IAAFT surrogate (Pink Noise class)
  - D (`ideal`): software ideal positive control

Supports `--dry-run` (verifies exact real Phase 0 NPZ dynamics streams, mean epsilon parity,
two-way paired bootstrap, CIF decomposition, and graded verdict engine without GPU) and
full GPU execution via `tools.run_wetware_stage0._gpu_train_and_eval_one_seed`.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
for extra in (ROOT, ROOT / 'standalone' / 'mea_oddball'):
    s = str(extra)
    if s not in sys.path:
        sys.path.insert(0, s)

from agent import mea_dynamics_explorer as mde
from tools import eval_survival_drives as ev
from tools import run_wetware_stage0 as s0

SIX_ARMS = (
    ('A', 'static'),
    ('B', 'real'),
    ('C', 'shuffled'),
    ('G1', 'white'),
    ('G2', 'iaaft'),
    ('D', 'ideal'),
)


def digest_obj(obj) -> str:
    raw = json.dumps(obj, sort_keys=True, separators=(',', ':'), allow_nan=False).encode('utf-8')
    return hashlib.sha256(raw).hexdigest()


def _stream_persistence_score(explorer: mde.WetwareDynamicsExplorer) -> float:
    """Deterministic structural descriptor of an explorer's stream (used ONLY in --dry-run)."""
    if explorer.arm == 'static':
        return 0.0
    if explorer.arm == 'ideal':
        return 1.0
    bz = explorer.burst_z
    u = explorer.u4
    # Lag-1 autocorrelation of burst_z and directional persistence of u4
    bz_c = bz - bz.mean()
    lag1_b = float(np.dot(bz_c[:-1], bz_c[1:]) / max(float(np.dot(bz_c, bz_c)), 1e-12))
    dir_dots = np.sum(u[:-1] * u[1:], axis=1)
    lag1_u = float(np.mean(dir_dots))
    # Nonlinear burst-onset asymmetry
    dx = np.diff(bz)
    asym = abs(float(np.mean(dx ** 3) / max(float(dx.std()) ** 3, 1e-8)))
    return max(0.0, 0.45 * max(0.0, lag1_b) + 0.45 * max(0.0, lag1_u) + 0.25 * asym)


def _dry_run_seed(arm_name: str, arm_kind: str, explorer: mde.WetwareDynamicsExplorer,
                  seed: int, worlds: list[int], bio_effect_scale: float = 1.0) -> dict:
    rng = np.random.default_rng(seed * 313 + ord(arm_name[0]) * 37 + len(arm_name) * 11)
    if arm_kind == 'ideal':
        eff = 1.0
    elif arm_kind == 'static':
        eff = 0.0
    else:
        eff = _stream_persistence_score(explorer) * float(bio_effect_scale)

    gain = eff * 5.0
    e1_rows, e2_rows, e3_rows, e4_rows = [], [], [], []
    for w in worlds:
        wrng = np.random.default_rng(int(w) * 29 + seed)
        base_cells = 44.0 + wrng.normal(0.0, 3.8)
        e1_rows.append(dict(
            new_cells=max(0, int(round(base_cells + gain + rng.normal(0.0, 2.0)))),
            alive=True,
            mean_ticks_per_action=1.35,
        ))
        drank = bool(wrng.random() < np.clip(0.46 + 0.025 * gain, 0.05, 0.98))
        e2_rows.append(dict(
            alive=drank,
            fear_of_thirst_rate=float(np.clip(0.52 + 0.03 * gain + rng.normal(0.0, 0.04), 0.0, 1.0)),
            thirst_drink_latency=float(12.0 - 0.5 * gain + rng.normal(0.0, 2.0)) if drank else 60.0,
            drank_after_low=drank,
        ))
        ate = bool(wrng.random() < np.clip(0.58 + 0.020 * gain, 0.05, 0.98))
        e3_rows.append(dict(
            alive=ate,
            fear_of_hunger_rate=float(np.clip(0.50 + 0.03 * gain + rng.normal(0.0, 0.04), 0.0, 1.0)),
            hunger_eat_latency=float(18.0 - 0.5 * gain + rng.normal(0.0, 2.5)) if ate else 60.0,
            ate_after_low=ate,
        ))
        evaded = bool(wrng.random() < np.clip(0.55 + 0.030 * gain, 0.05, 0.98))
        e4_rows.append(dict(
            alive=evaded,
            fear_of_death_caution=float(0.06 + 0.025 * gain + rng.normal(0.0, 0.02)),
            threat_evasion_rate=float(np.clip(0.55 + 0.035 * gain + rng.normal(0.0, 0.04), 0.0, 1.0)),
        ))
    return dict(
        eval=e1_rows,
        suites=dict(
            E1_explore_96=e1_rows,
            E2_thirst_acute=e2_rows,
            E3_hunger_acute=e3_rows,
            E4_death_night_threat=e4_rows,
        ),
        probe_curve=[],
    )


def classify_spontaneous_outcome(
    b_minus_c: dict,
    b_minus_a: dict,
    b_minus_g2: dict,
    d_minus_a: dict,
    stress_surv_b_a: dict,
    stress_surv_b_c: dict,
    drive_b_c: dict,
    *,
    replicated_on_second_chip: bool = False,
) -> dict:
    """Apply frozen Phase 2-3 grading rules from wetware_existing_data_training_plan.md."""
    if d_minus_a['ci_low'] is None or d_minus_a['ci_low'] <= 0.0:
        return dict(verdict='inconclusive', survival_warning=False,
                    reason='Positive control D - A lower 95% CI <= 0; pipeline cannot adjudicate')
    surv_warn = bool(
        stress_surv_b_a.get('ci_low') is not None
        and stress_surv_b_a['ci_low'] < ev.GUARDRAIL_SURVIVAL_DROP_MAX
    )
    weak_pos = bool(
        b_minus_c['ci_low'] is not None and b_minus_c['ci_low'] > 0.0
        and b_minus_a['ci_low'] is not None and b_minus_a['ci_low'] > 0.0
        and (b_minus_c['seed_positive_frac'] or 0.0) >= ev.WEAK_POSITIVE_SEED_FRAC
        and not surv_warn
    )
    if not weak_pos:
        return dict(verdict='negative', survival_warning=surv_warn,
                    reason='D - A passed, but B did not beat both C and A within survival guardrail')

    improved_drives = sum(
        1 for st in drive_b_c.values()
        if st and st.get('ci_low') is not None and st['ci_low'] > 0.0
    )
    surv_nonneg = stress_surv_b_c.get('mean') is not None and stress_surv_b_c['mean'] >= 0.0
    beats_iaaft = b_minus_g2.get('ci_low') is not None and b_minus_g2['ci_low'] > 0.0
    if replicated_on_second_chip and beats_iaaft:
        return dict(verdict='strong_bio_specific_positive', survival_warning=False,
                    reason='Replicated across >=2 independent chips AND beat IAAFT matched-spectrum surrogate')
    if improved_drives >= 2 and surv_nonneg:
        return dict(verdict='dual_weak_positive', survival_warning=False,
                    reason='Primary exploration and >=2 survival drives beat C with non-negative survival')
    return dict(verdict='weak_positive', survival_warning=False,
                reason='Real organoid dynamics beat both shuffled C and static A within survival guardrail')


def run_study(args) -> dict:
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    lib = mde.load_dynamics_library(args.library_manifest, require_gate_s=not args.allow_non_gate_s)
    tier_cfg = dict(s0.STAGE0_LADDER[args.tier - 1])
    if args.substudy == '2A_pure_r1':
        r2_mode = 'dual'
        r2_source = 'none'
    else:
        r2_mode = 'dual'
        r2_source = 'ideal'

    seeds = list(range(args.seed_base, args.seed_base + args.seeds))
    worlds = list(range(args.world_base, args.world_base + args.eval_worlds))
    probe_worlds = worlds[:min(16, len(worlds))]

    # Pre-audit mean exploration rate parity across B / C / G1 / G2
    eps_audit = {}
    for arm_name, arm_kind in SIX_ARMS:
        if arm_kind in ('real', 'shuffled', 'white', 'iaaft'):
            ex = mde.WetwareDynamicsExplorer(arm_kind, lib, seed=seeds[0], eps_base=args.eps_base)
            eps_audit[arm_name] = dict(
                mean_eps=float(ex.eps_stream.mean()),
                std_eps=float(ex.eps_stream.std()),
                min_eps=float(ex.eps_stream.min()),
                max_eps=float(ex.eps_stream.max()),
            )

    arm_results = {}
    for arm_name, arm_kind in SIX_ARMS:
        e1_cells, surv_stress = [], []
        thirst_rate, hunger_rate, death_caution, evasion_rate = [], [], [], []
        thirst_lat, thirst_succ, hunger_lat, hunger_succ = [], [], [], []

        for s in seeds:
            ex = mde.WetwareDynamicsExplorer(
                arm_kind,
                None if arm_kind in ('static', 'ideal') else lib,
                seed=s,
                eps_base=args.eps_base,
                homeostatic_gate=True,
            )
            ent_mod = mde.WetwareEntropyModulator(
                eta=args.entropy_eta,
                enabled=bool(args.enable_m2 and arm_kind not in ('static',)),
            )
            if args.dry_run:
                res = _dry_run_seed(arm_name, arm_kind, ex, s, worlds, bio_effect_scale=args.dry_run_strength)
            else:
                res = s0._gpu_train_and_eval_one_seed(
                    mode=r2_mode,
                    source_kind=r2_source,
                    seed=s,
                    tier_cfg=tier_cfg,
                    model_config_path=args.model_config,
                    godot_path=args.godot,
                    project_path=args.project,
                    eval_worlds=worlds,
                    probe_worlds=probe_worlds,
                    use_curriculum=True,
                    explorer=ex,
                    entropy_modulator=ent_mod,
                )
            su = res['suites']
            e1_cells.append([float(m['new_cells']) for m in su['E1_explore_96']])
            # Pooled stress survival across E2 + E3 + E4
            s_pool = [
                (float(su['E2_thirst_acute'][i]['alive'])
                 + float(su['E3_hunger_acute'][i]['alive'])
                 + float(su['E4_death_night_threat'][i]['alive'])) / 3.0
                for i in range(len(worlds))
            ]
            surv_stress.append(s_pool)
            thirst_rate.append([float(m['fear_of_thirst_rate']) if m['fear_of_thirst_rate'] is not None else float('nan') for m in su['E2_thirst_acute']])
            hunger_rate.append([float(m['fear_of_hunger_rate']) if m['fear_of_hunger_rate'] is not None else float('nan') for m in su['E3_hunger_acute']])
            death_caution.append([float(m['fear_of_death_caution']) if m['fear_of_death_caution'] is not None else float('nan') for m in su['E4_death_night_threat']])
            evasion_rate.append([float(m['threat_evasion_rate']) if m['threat_evasion_rate'] is not None else float('nan') for m in su['E4_death_night_threat']])
            thirst_lat.append([float(m['thirst_drink_latency']) if m['thirst_drink_latency'] is not None else float('nan') for m in su['E2_thirst_acute']])
            thirst_succ.append([bool(m.get('drank_after_low', False)) for m in su['E2_thirst_acute']])
            hunger_lat.append([float(m['hunger_eat_latency']) if m['hunger_eat_latency'] is not None else float('nan') for m in su['E3_hunger_acute']])
            hunger_succ.append([bool(m.get('ate_after_low', False)) for m in su['E3_hunger_acute']])

        arm_results[arm_name] = dict(
            e1_cells=np.asarray(e1_cells, dtype=float),
            surv_stress=np.asarray(surv_stress, dtype=float),
            thirst_rate=np.asarray(thirst_rate, dtype=float),
            hunger_rate=np.asarray(hunger_rate, dtype=float),
            death_caution=np.asarray(death_caution, dtype=float),
            evasion_rate=np.asarray(evasion_rate, dtype=float),
            thirst_lat=np.asarray(thirst_lat, dtype=float),
            thirst_succ=np.asarray(thirst_succ, dtype=bool),
            hunger_lat=np.asarray(hunger_lat, dtype=float),
            hunger_succ=np.asarray(hunger_succ, dtype=bool),
        )

    b, c, a, g1, g2, d = (arm_results[k] for k in ('B', 'C', 'A', 'G1', 'G2', 'D'))
    b_c = ev.paired_bootstrap_2way(b['e1_cells'], c['e1_cells'], seed=args.seed_base + 1, n_boot=args.n_boot)
    b_a = ev.paired_bootstrap_2way(b['e1_cells'], a['e1_cells'], seed=args.seed_base + 2, n_boot=args.n_boot)
    b_g1 = ev.paired_bootstrap_2way(b['e1_cells'], g1['e1_cells'], seed=args.seed_base + 3, n_boot=args.n_boot)
    b_g2 = ev.paired_bootstrap_2way(b['e1_cells'], g2['e1_cells'], seed=args.seed_base + 4, n_boot=args.n_boot)
    d_a = ev.paired_bootstrap_2way(d['e1_cells'], a['e1_cells'], seed=args.seed_base + 5, n_boot=args.n_boot)
    surv_b_a = ev.paired_bootstrap_2way(b['surv_stress'], a['surv_stress'], seed=args.seed_base + 6, n_boot=args.n_boot)
    surv_b_c = ev.paired_bootstrap_2way(b['surv_stress'], c['surv_stress'], seed=args.seed_base + 7, n_boot=args.n_boot)
    drive_b_c = dict(
        fear_of_thirst=ev.paired_bootstrap_2way(b['thirst_rate'], c['thirst_rate'], seed=args.seed_base + 11, n_boot=args.n_boot),
        fear_of_hunger=ev.paired_bootstrap_2way(b['hunger_rate'], c['hunger_rate'], seed=args.seed_base + 12, n_boot=args.n_boot),
        fear_of_death_evasion=ev.paired_bootstrap_2way(b['evasion_rate'], c['evasion_rate'], seed=args.seed_base + 13, n_boot=args.n_boot),
    )
    thirst_decomp = ev.paired_resource_decomposition(b['thirst_lat'], b['thirst_succ'], c['thirst_lat'], c['thirst_succ'])
    hunger_decomp = ev.paired_resource_decomposition(b['hunger_lat'], b['hunger_succ'], c['hunger_lat'], c['hunger_succ'])

    outcome = classify_spontaneous_outcome(
        b_c, b_a, b_g2, d_a, surv_b_a, surv_b_c, drive_b_c,
        replicated_on_second_chip=bool(args.replicated_on_second_chip),
    )
    report = dict(
        schema='wetware.spontaneous.study.v1',
        chip=lib['manifest']['chip'],
        library_manifest_sha256=lib['manifest']['manifest_sha256'],
        substudy=args.substudy,
        tier=tier_cfg,
        enable_m2=bool(args.enable_m2),
        dry_run=bool(args.dry_run),
        n_seeds=len(seeds),
        n_worlds=len(worlds),
        eps_parity_audit=eps_audit,
        comparisons=dict(
            B_minus_C=b_c,
            B_minus_A=b_a,
            B_minus_G1=b_g1,
            B_minus_G2_iaaft=b_g2,
            D_minus_A=d_a,
            stress_survival_B_minus_A=surv_b_a,
            stress_survival_B_minus_C=surv_b_c,
            drives_B_minus_C=drive_b_c,
            thirst_cif_decomposition_B_vs_C=thirst_decomp,
            hunger_cif_decomposition_B_vs_C=hunger_decomp,
        ),
        outcome=outcome,
    )
    report['sha256'] = digest_obj(report)
    (out_dir / 'spontaneous_study_report.json').write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8'
    )
    return report


def build_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--library-manifest', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--substudy', choices=('2A_pure_r1', '2B_dual_drive'), default='2B_dual_drive')
    p.add_argument('--tier', type=int, choices=(1, 2, 3, 4), default=2)
    p.add_argument('--eps-base', type=float, default=0.10)
    p.add_argument('--enable-m2', action='store_true')
    p.add_argument('--entropy-eta', type=float, default=0.35)
    p.add_argument('--allow-non-gate-s', action='store_true')
    p.add_argument('--replicated-on-second-chip', action='store_true')
    p.add_argument('--dry-run', action='store_true')
    p.add_argument('--dry-run-strength', type=float, default=1.0)
    p.add_argument('--model-config', type=Path)
    p.add_argument('--godot', type=Path)
    p.add_argument('--project', type=Path, default=ROOT)
    p.add_argument('--seeds', type=int, default=32)
    p.add_argument('--eval-worlds', type=int, default=64)
    p.add_argument('--seed-base', type=int, default=20261210)
    p.add_argument('--world-base', type=int, default=960000)
    p.add_argument('--n-boot', type=int, default=10000)
    return p


if __name__ == '__main__':
    rep = run_study(build_parser().parse_args())
    print(json.dumps(dict(
        chip=rep['chip'],
        substudy=rep['substudy'],
        verdict=rep['outcome']['verdict'],
        reason=rep['outcome']['reason'],
        B_minus_C=rep['comparisons']['B_minus_C']['mean'],
        B_minus_G2=rep['comparisons']['B_minus_G2_iaaft']['mean'],
    ), ensure_ascii=False))
