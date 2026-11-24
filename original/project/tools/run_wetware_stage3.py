"""Stage 3 & Stage 4 seven-arm paired wetware training, channel bound audit, and cross-organoid replication.

Seven paired arms on identical seeds, identical unseen maps, and identical 108-dim model inputs:
- A: pure R1 baseline (`none`)
- B: real timeline organoid dual library (`wetware_real`)
- C: within-batch context-shuffled organoid library (`wetware_shuffled`)
- C_time: pre-build timeline-permuted organoid library (`wetware_time_shuffled`)
- D: noise-free software ideal positive control (`ideal`)
- E: sign-inverted organoid library (`wetware_inverted`)
- F: noise-matched software control at `rho_lib` (`noisy`)

When `--replication-habituation-library` / `--replication-evoked-library` (e.g., independent chip
`28080_1116_long`) are supplied, automatically executes Stage 4 on fresh seeds and fresh maps to
verify cross-organoid `stable_positive` replication without any new data collection.
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
for extra in (ROOT, ROOT / 'standalone' / 'mea_oddball'):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

from oddball import library_v2
from oddball.signals import EpisodeContext, MODES, SensoryState, WetwareSignalSource
from tools import eval_survival_drives as ev
from tools import r2_guard as guard
from tools import run_wetware_stage0 as s0

ARMS = (
    ('A', 'none'),
    ('B', 'wetware_real'),
    ('C', 'wetware_shuffled'),
    ('C_time', 'wetware_time_shuffled'),
    ('D', 'ideal'),
    ('E', 'wetware_inverted'),
    ('F', 'noisy'),
)


def _measure_library_effective_alignment(
    mode, source_kind, seed, rho_lib, hab_lib, evk_lib, ctime_hab_lib, ctime_evk_lib
):
    """When real library JSONs are supplied in dry-run, sample actual library draws via WetwareSignalSource."""
    ctx = EpisodeContext()
    contexts = []
    rng = np.random.default_rng(seed + 17)
    px, py = 0, 0
    for step in range(384):
        prev = SensoryState(x=px, y=py, health=90.0, thirst=85.0, hunger=85.0, nearest_water_dist=8.0, nearest_food_dist=8.0)
        if step % 6 == 0:
            # Thirst or hunger crisis transition
            relieved = bool(step % 12 == 0)
            prev = SensoryState(x=px, y=py, health=90.0, thirst=22.0, hunger=80.0, nearest_water_dist=8.0)
            cur = SensoryState(x=px, y=py, health=90.0, thirst=55.0 if relieved else 21.0,
                               hunger=80.0, nearest_water_dist=4.0 if relieved else 9.0)
        else:
            if rng.random() < 0.65:
                px += 4
            cur = SensoryState(x=px, y=py, health=90.0, thirst=85.0, hunger=85.0, nearest_water_dist=8.0, nearest_food_dist=8.0)
        contexts.append(ctx.encode_step(cur, mode, prev_state=prev))

    ideal_vec = WetwareSignalSource(mode, 'ideal', seed).score_batch(contexts)
    src_vec = WetwareSignalSource(
        mode, source_kind, seed, rho=rho_lib,
        habituation_library=hab_lib, evoked_library=evk_lib,
        ctime_habituation_library=ctime_hab_lib, ctime_evoked_library=ctime_evk_lib,
    ).score_batch(contexts)
    if src_vec.std() < 1e-12 or ideal_vec.std() < 1e-12:
        return 0.0
    return float(np.corrcoef(src_vec, ideal_vec)[0, 1])


def pretrain_channel_audit(mode, seed, rho_lib, hab_lib, evk_lib, ctime_hab_lib, ctime_evk_lib):
    """Verify `channel_bound_check` on synthetic pre-training rollout contexts before Stage 3/4."""
    if hab_lib is None and evk_lib is None:
        return None
    ctx = EpisodeContext()
    contexts = []
    labels = []
    groups = np.repeat(np.arange(8), 64)
    rng = np.random.default_rng(seed + 91)
    px, py = 0, 0
    for i in range(len(groups)):
        if i % 64 == 0:
            ctx.reset()
        explore_move = bool(rng.random() < 0.5)
        prev = SensoryState(x=px, y=py, health=90.0, thirst=80.0, hunger=80.0)
        if explore_move:
            px += 4
        cur = SensoryState(x=px, y=py, health=90.0, thirst=80.0, hunger=80.0)
        c = ctx.encode_step(cur, mode, prev_state=prev)
        contexts.append(c)
        labels.append(c['habituation_bin'] == 'low' if c['channel'] == 'habituation' else c['deficit_relieved'])

    kw = dict(
        habituation_library=hab_lib, evoked_library=evk_lib,
        ctime_habituation_library=ctime_hab_lib, ctime_evoked_library=ctime_evk_lib,
    )
    r_real = WetwareSignalSource(mode, 'wetware_real', seed + 1, **kw).score_batch(contexts)
    r_ideal = WetwareSignalSource(mode, 'ideal', seed + 2).score_batch(contexts)
    r_shuf = WetwareSignalSource(mode, 'wetware_shuffled', seed + 3, **kw).score_batch(contexts)
    r_ctime = (
        WetwareSignalSource(mode, 'wetware_time_shuffled', seed + 4, **kw).score_batch(contexts)
        if (ctime_hab_lib is not None or ctime_evk_lib is not None) else None
    )
    return guard.channel_bound_check(
        r_real, r_ideal, r_shuf, np.asarray(labels, bool), groups, range(4, 8),
        seed=seed + 100, shuffles=200, r2_ctime=r_ctime,
    )


def _dry_run_arm(
    mode, arm_name, source_kind, seed, tier_cfg, worlds, rho_lib, *,
    organoid_signal_strength=1.0, hab_lib=None, evk_lib=None, ctime_hab_lib=None, ctime_evk_lib=None,
):
    rng = np.random.default_rng(seed * 257 + sum(ord(ch) for ch in arm_name) * 31)
    has_libs = (
        (mode in ('explore', 'dual') and hab_lib is not None)
        or (mode in ('homeo', 'dual') and evk_lib is not None)
    )
    if has_libs and source_kind in ('wetware_real', 'wetware_shuffled', 'wetware_time_shuffled', 'wetware_inverted'):
        emp_corr = _measure_library_effective_alignment(
            mode, source_kind, seed, rho_lib, hab_lib, evk_lib, ctime_hab_lib, ctime_evk_lib
        )
        eff = emp_corr * float(organoid_signal_strength)
    elif source_kind in ('none', 'wetware_shuffled', 'wetware_time_shuffled'):
        eff = 0.0
    elif source_kind == 'ideal':
        eff = 1.0
    elif source_kind == 'noisy':
        eff = float(rho_lib)
    elif source_kind == 'wetware_real':
        eff = float(rho_lib) * float(organoid_signal_strength)
    elif source_kind == 'wetware_inverted':
        eff = -float(rho_lib) * float(organoid_signal_strength)
    else:
        raise ValueError(source_kind)

    beta_scale = float(tier_cfg.get('beta_adv', 0.15)) / 0.15
    gain = eff * np.sqrt(tier_cfg['updates'] / 4.0) * beta_scale * 3.2
    eval_rows = []
    for w in worlds:
        wrng = np.random.default_rng(int(w) * 19 + seed)
        base_cells = 42.0 + wrng.normal(0.0, 3.5)
        cells = max(0, int(round(base_cells + gain + rng.normal(0.0, 2.0))))
        thirst_r = float(np.clip(0.55 + 0.045 * gain + rng.normal(0.0, 0.04), 0.0, 1.0))
        hunger_r = float(np.clip(0.52 + 0.045 * gain + rng.normal(0.0, 0.04), 0.0, 1.0))
        caution = float(0.05 + 0.025 * gain + rng.normal(0.0, 0.02))
        alive = bool(wrng.random() < np.clip(0.82 + 0.02 * gain, 0.1, 0.99))
        eval_rows.append(dict(
            new_cells=cells,
            alive=alive,
            fear_of_thirst_rate=thirst_r,
            fear_of_hunger_rate=hunger_r,
            fear_of_death_caution=caution,
            mean_ticks_per_action=1.35,
        ))
    probe = [
        dict(update=u, new_cells=42.0 + gain * (u / tier_cfg['updates']) + rng.normal(0.0, 0.2))
        for u in range(s0.PROBE_EVERY_UPDATES, tier_cfg['updates'] + 1, s0.PROBE_EVERY_UPDATES)
    ]
    return dict(eval=eval_rows, probe_curve=probe)


def _load_optional_lib(path, schema):
    if not path:
        return None
    p = Path(path)
    if not p.is_file():
        return None
    return library_v2.check_library(json.loads(p.read_text(encoding='utf-8')), schema)


def _auto_sibling_ctime(lib_path, prefix_name):
    if not lib_path:
        return None
    cand = Path(lib_path).parent / prefix_name
    return cand if cand.is_file() else None


def _execute_cohort(
    *,
    cohort_tag,
    args,
    tier_cfg,
    seeds,
    worlds,
    rho_lib,
    hab_lib,
    evk_lib,
    ctime_hab_lib,
    ctime_evk_lib,
    second_organoid_replicated=False,
):
    probe_worlds = worlds[:min(16, len(worlds))]
    primary_key = 'new_cells' if args.mode in ('explore', 'dual') else 'fear_of_thirst_rate'

    bound_audit = pretrain_channel_audit(
        args.mode, seeds[0], rho_lib, hab_lib, evk_lib, ctime_hab_lib, ctime_evk_lib
    )
    if bound_audit is not None and bound_audit.get('passed') is False:
        raise RuntimeError(f'Pre-training channel_bound_check failed on {cohort_tag}: {bound_audit}')

    arm_data = {}
    arm_probes = {}
    for arm_name, source_kind in ARMS:
        primary_mat = []
        surv_mat = []
        drive_mats = {'fear_of_thirst': [], 'fear_of_hunger': [], 'fear_of_death': []}
        probes = []
        for s in seeds:
            if args.dry_run:
                res = _dry_run_arm(
                    args.mode, arm_name, source_kind, s, tier_cfg, worlds, rho_lib,
                    organoid_signal_strength=args.dry_run_strength,
                    hab_lib=hab_lib, evk_lib=evk_lib,
                    ctime_hab_lib=ctime_hab_lib, ctime_evk_lib=ctime_evk_lib,
                )
            else:
                res = s0._gpu_train_and_eval_one_seed(
                    mode=args.mode, source_kind=source_kind, seed=s, tier_cfg=tier_cfg,
                    model_config_path=args.model_config, godot_path=args.godot,
                    project_path=args.project, eval_worlds=worlds, probe_worlds=probe_worlds,
                    rho=rho_lib if source_kind == 'noisy' else 1.0,
                    habituation_library=hab_lib, evoked_library=evk_lib,
                    ctime_habituation_library=ctime_hab_lib, ctime_evoked_library=ctime_evk_lib,
                )
            primary_mat.append([float(m[primary_key]) if m[primary_key] is not None else float('nan') for m in res['eval']])
            surv_mat.append([1.0 if m['alive'] else 0.0 for m in res['eval']])
            drive_mats['fear_of_thirst'].append([float(m['fear_of_thirst_rate']) if m['fear_of_thirst_rate'] is not None else float('nan') for m in res['eval']])
            drive_mats['fear_of_hunger'].append([float(m['fear_of_hunger_rate']) if m['fear_of_hunger_rate'] is not None else float('nan') for m in res['eval']])
            drive_mats['fear_of_death'].append([float(m['fear_of_death_caution']) if m['fear_of_death_caution'] is not None else float('nan') for m in res['eval']])
            probes.append(res['probe_curve'])
        arm_data[arm_name] = dict(
            primary=np.asarray(primary_mat, dtype=float),
            survival=np.asarray(surv_mat, dtype=float),
            drives={k: np.asarray(v, dtype=float) for k, v in drive_mats.items()},
        )
        arm_probes[arm_name] = probes

    sb = seeds[0]
    b_minus_c = ev.paired_bootstrap_2way(arm_data['B']['primary'], arm_data['C']['primary'], seed=sb + 1, n_boot=args.n_boot)
    b_minus_ctime = ev.paired_bootstrap_2way(arm_data['B']['primary'], arm_data['C_time']['primary'], seed=sb + 8, n_boot=args.n_boot)
    b_minus_a = ev.paired_bootstrap_2way(arm_data['B']['primary'], arm_data['A']['primary'], seed=sb + 2, n_boot=args.n_boot)
    d_minus_a = ev.paired_bootstrap_2way(arm_data['D']['primary'], arm_data['A']['primary'], seed=sb + 3, n_boot=args.n_boot)
    e_minus_a = ev.paired_bootstrap_2way(arm_data['E']['primary'], arm_data['A']['primary'], seed=sb + 4, n_boot=args.n_boot)
    b_minus_f = ev.paired_bootstrap_2way(arm_data['B']['primary'], arm_data['F']['primary'], seed=sb + 5, n_boot=args.n_boot)
    surv_b_a = ev.paired_bootstrap_2way(arm_data['B']['survival'], arm_data['A']['survival'], seed=sb + 6, n_boot=args.n_boot)
    surv_b_c = ev.paired_bootstrap_2way(arm_data['B']['survival'], arm_data['C']['survival'], seed=sb + 7, n_boot=args.n_boot)
    drive_b_c = {
        k: ev.paired_bootstrap_2way(arm_data['B']['drives'][k], arm_data['C']['drives'][k],
                                    seed=sb + 10 + i, n_boot=args.n_boot)
        for i, k in enumerate(('fear_of_thirst', 'fear_of_hunger', 'fear_of_death'))
    }
    drive_b_ctime = {
        k: ev.paired_bootstrap_2way(arm_data['B']['drives'][k], arm_data['C_time']['drives'][k],
                                    seed=sb + 20 + i, n_boot=args.n_boot)
        for i, k in enumerate(('fear_of_thirst', 'fear_of_hunger', 'fear_of_death'))
    }

    ckpt_b_gt_c = []
    if arm_probes['B'] and arm_probes['C'] and arm_probes['B'][0]:
        n_ckpts = len(arm_probes['B'][0])
        for c_idx in range(n_ckpts):
            b_mean = float(np.mean([p[c_idx]['new_cells'] for p in arm_probes['B']]))
            c_mean = float(np.mean([p[c_idx]['new_cells'] for p in arm_probes['C']]))
            ckpt_b_gt_c.append(b_mean > c_mean)

    outcome = ev.classify_stage3_outcome(
        b_minus_c, b_minus_a, d_minus_a, e_minus_a, surv_b_a,
        b_minus_ctime=b_minus_ctime,
        drive_b_minus_c=drive_b_c,
        survival_b_minus_c=surv_b_c,
        checkpoint_b_gt_c=ckpt_b_gt_c,
        second_organoid_replicated=second_organoid_replicated,
    )
    return dict(
        cohort=cohort_tag,
        channel_bound_audit=bound_audit,
        arm_means={a: float(np.nanmean(arm_data[a]['primary'])) for a, _ in ARMS},
        arm_survival_means={a: float(np.nanmean(arm_data[a]['survival'])) for a, _ in ARMS},
        comparisons=dict(
            B_minus_C=b_minus_c,
            B_minus_Ctime=b_minus_ctime,
            B_minus_A=b_minus_a,
            D_minus_A=d_minus_a,
            E_minus_A=e_minus_a,
            B_minus_F=b_minus_f,
            survival_B_minus_A=surv_b_a,
            survival_B_minus_C=surv_b_c,
            drives_B_minus_C=drive_b_c,
            drives_B_minus_Ctime=drive_b_ctime,
        ),
        checkpoint_B_gt_C=ckpt_b_gt_c,
        outcome=outcome,
    )


def run_stage3(args):
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    stage0 = json.loads(Path(args.stage0_report).read_text(encoding='utf-8'))
    if not stage0.get('confirmed') or not stage0.get('frozen_tier'):
        raise ValueError('Stage 0 did not pass confirmation; Stage 3 training is blocked')
    tier_cfg = stage0['frozen_tier']

    hab_lib = _load_optional_lib(args.habituation_library, library_v2.HABITUATION_SCHEMA)
    evk_lib = _load_optional_lib(args.evoked_library, library_v2.EVOKED_SCHEMA)
    ctime_hab_path = args.ctime_habituation_library or _auto_sibling_ctime(args.habituation_library, 'ctime_habituation_library.json')
    ctime_evk_path = args.ctime_evoked_library or _auto_sibling_ctime(args.evoked_library, 'ctime_evoked_library.json')
    ctime_hab_lib = _load_optional_lib(ctime_hab_path, library_v2.HABITUATION_SCHEMA)
    ctime_evk_lib = _load_optional_lib(ctime_evk_path, library_v2.EVOKED_SCHEMA)

    if not args.dry_run:
        if args.mode in ('explore', 'dual') and hab_lib is None:
            raise ValueError(f'Mode {args.mode} requires a verified habituation_library.json')
        if args.mode in ('homeo', 'dual') and evk_lib is None:
            raise ValueError(f'Mode {args.mode} requires a verified evoked_library.json')

    n_seeds = int(args.seeds or stage0.get('recommended_stage3_seeds') or 32)
    seeds = list(range(args.seed_base, args.seed_base + n_seeds))
    worlds = list(range(args.world_base, args.world_base + args.eval_worlds))
    rho_lib = float(args.rho_lib)

    stage3_res = _execute_cohort(
        cohort_tag='stage3_primary',
        args=args,
        tier_cfg=tier_cfg,
        seeds=seeds,
        worlds=worlds,
        rho_lib=rho_lib,
        hab_lib=hab_lib,
        evk_lib=evk_lib,
        ctime_hab_lib=ctime_hab_lib,
        ctime_evk_lib=ctime_evk_lib,
        second_organoid_replicated=bool(args.second_organoid_replicated),
    )

    # Optional Stage 4 independent second-organoid replication on fresh seeds and fresh maps
    rep_hab = _load_optional_lib(args.replication_habituation_library, library_v2.HABITUATION_SCHEMA)
    rep_evk = _load_optional_lib(args.replication_evoked_library, library_v2.EVOKED_SCHEMA)
    stage4_res = None
    final_outcome = stage3_res['outcome']

    if rep_hab is not None or rep_evk is not None:
        rep_ctime_hab = _load_optional_lib(
            args.replication_ctime_habituation_library or _auto_sibling_ctime(args.replication_habituation_library, 'ctime_habituation_library.json'),
            library_v2.HABITUATION_SCHEMA,
        )
        rep_ctime_evk = _load_optional_lib(
            args.replication_ctime_evoked_library or _auto_sibling_ctime(args.replication_evoked_library, 'ctime_evoked_library.json'),
            library_v2.EVOKED_SCHEMA,
        )
        rep_seeds = list(range(args.seed_base + 50000, args.seed_base + 50000 + n_seeds))
        rep_worlds = list(range(args.world_base + 50000, args.world_base + 50000 + args.eval_worlds))
        stage4_res = _execute_cohort(
            cohort_tag='stage4_independent_organoid_replication',
            args=args,
            tier_cfg=tier_cfg,
            seeds=rep_seeds,
            worlds=rep_worlds,
            rho_lib=float(args.replication_rho_lib or rho_lib),
            hab_lib=rep_hab,
            evk_lib=rep_evk,
            ctime_hab_lib=rep_ctime_hab,
            ctime_evk_lib=rep_ctime_evk,
            second_organoid_replicated=False,
        )
        s3_pos = stage3_res['outcome']['verdict'] in ('weak_positive', 'dual_weak_positive', 'stable_positive')
        s4_pos = stage4_res['outcome']['verdict'] in ('weak_positive', 'dual_weak_positive', 'stable_positive')
        if s3_pos and s4_pos:
            final_outcome = dict(
                verdict='stable_positive',
                survival_warning=False,
                reason='Confirmed positive replicated across two independent archived organoids (Stage 3 + Stage 4) beating A, C, and C_time',
                stage3_verdict=stage3_res['outcome']['verdict'],
                stage4_verdict=stage4_res['outcome']['verdict'],
            )

    primary_key = 'new_cells' if args.mode in ('explore', 'dual') else 'fear_of_thirst_rate'
    report = dict(
        schema='wetware.stage3.v2',
        mode=args.mode,
        arms=[a for a, _ in ARMS],
        primary_metric=primary_key,
        dry_run=bool(args.dry_run),
        rho_lib=rho_lib,
        n_seeds=n_seeds,
        n_worlds=len(worlds),
        stage0_sha256=stage0.get('sha256'),
        frozen_tier=tier_cfg,
        channel_bound_audit=stage3_res['channel_bound_audit'],
        arm_means=stage3_res['arm_means'],
        arm_survival_means=stage3_res['arm_survival_means'],
        comparisons=stage3_res['comparisons'],
        checkpoint_B_gt_C=stage3_res['checkpoint_B_gt_C'],
        stage4_replication=stage4_res,
        outcome=final_outcome,
    )
    report['sha256'] = s0.digest_obj(report)
    (out_dir / 'stage3_report.json').write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
    return report


def build_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode', choices=MODES, default='dual')
    p.add_argument('--stage0-report', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--habituation-library', type=Path)
    p.add_argument('--evoked-library', type=Path)
    p.add_argument('--ctime-habituation-library', type=Path)
    p.add_argument('--ctime-evoked-library', type=Path)
    p.add_argument('--replication-habituation-library', type=Path)
    p.add_argument('--replication-evoked-library', type=Path)
    p.add_argument('--replication-ctime-habituation-library', type=Path)
    p.add_argument('--replication-ctime-evoked-library', type=Path)
    p.add_argument('--rho-lib', type=float, default=0.44)
    p.add_argument('--replication-rho-lib', type=float)
    p.add_argument('--dry-run', action='store_true')
    p.add_argument('--dry-run-strength', type=float, default=1.0)
    p.add_argument('--second-organoid-replicated', action='store_true')
    p.add_argument('--model-config', type=Path)
    p.add_argument('--godot', type=Path)
    p.add_argument('--project', type=Path, default=ROOT)
    p.add_argument('--seeds', type=int)
    p.add_argument('--eval-worlds', type=int, default=64)
    p.add_argument('--seed-base', type=int, default=20261201)
    p.add_argument('--world-base', type=int, default=950000)
    p.add_argument('--n-boot', type=int, default=10000)
    return p


if __name__ == '__main__':
    res = run_stage3(build_parser().parse_args())
    print(json.dumps(dict(
        verdict=res['outcome']['verdict'],
        reason=res['outcome']['reason'],
        B_minus_C=res['comparisons']['B_minus_C']['mean'],
        B_minus_C_ci=[res['comparisons']['B_minus_C']['ci_low'], res['comparisons']['B_minus_C']['ci_high']],
        B_minus_Ctime=res['comparisons']['B_minus_Ctime']['mean'],
        B_minus_Ctime_ci=[res['comparisons']['B_minus_Ctime']['ci_low'], res['comparisons']['B_minus_Ctime']['ci_high']],
    ), ensure_ascii=False))
