"""Zero-GPU historical trajectory audit: clock-deconfounded R1_task & fixed-tick baseline ledger.

Completes the pending historical diagnostic from `output/下一步决策_20261119.md` (Step 1) using
archived evaluation logs on disk (zero GPU, zero hardware):
1) Decomposes `Decision-6` vs `Tick-6` R1 & exploration on `22021` and `22554` (`fixed_tick_worlds.csv`
   + `r1_reference_worlds.csv`), quantifying exact survival-clock inflation (`0.01 * sum gamma^t`).
2) Decomposes `Tick-60` total R1 into survival-clock `R1_clock` and pure task potential `R1_task`
   across all 18,432 episodes of `22021` 32-seed x 64-map x 3-scene (`low_health`, `thirst`, `hunger`)
   confirmatory archive (`episodes.csv`), including strict common-survivor (`end_tick == 60` across
   all arms) paired two-way bootstrap intervals.
"""
import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools import eval_survival_drives as ev

GAMMA = 0.99985


def living_clock_r1(ticks: int, gamma: float = GAMMA) -> float:
    t = max(0, int(ticks))
    if t == 0:
        return 0.0
    return 0.01 * (1.0 - (gamma ** t)) / (1.0 - gamma)


def _audit_beta_sweep_clock(output_root: Path, n_boot: int):
    ft_path = output_root / 'mea_r2_beta_sweep_20261120/fixed_tick_worlds.csv'
    ref_path = output_root / 'mea_r2_beta_sweep_20261120/r1_reference_worlds.csv'
    if not ft_path.is_file() or not ref_path.is_file():
        return {}

    # Group by (chip, arm, beta=0.0) -> {(train_seed, world_seed): row}
    grouped = {}
    for p in (ft_path, ref_path):
        with p.open('r', encoding='utf-8-sig') as f:
            for row in csv.DictReader(f):
                beta = float(row.get('beta', 0.0))
                if abs(beta) > 1e-9:
                    continue
                key = (row['chip'], row['arm'])
                sw = (int(row['train_seed']), int(row['world_seed']))
                grouped.setdefault(key, {})[sw] = row

    results = {}
    # Compare 22021 baseline (pre-stim spontaneous) & sham (zero-output) against masked
    for chip, arm, base_arm in [
        ('22021', 'baseline', 'masked'),
        ('22021', 'sham', 'masked'),
        ('22554', 'ordered', 'shuffled'),
    ]:
        if (chip, arm) not in grouped or (chip, base_arm) not in grouped:
            continue
        a_map = grouped[(chip, arm)]
        b_map = grouped[(chip, base_arm)]
        common_seeds = sorted({s for s, _ in a_map} & {s for s, _ in b_map})
        common_worlds = sorted({w for _, w in a_map} & {w for _, w in b_map})
        if not common_seeds or not common_worlds:
            continue

        def build_mats(extractor):
            ma = np.full((len(common_seeds), len(common_worlds)), np.nan)
            mb = np.full((len(common_seeds), len(common_worlds)), np.nan)
            for si, s in enumerate(common_seeds):
                for wi, w in enumerate(common_worlds):
                    if (s, w) in a_map and (s, w) in b_map:
                        ma[si, wi] = extractor(a_map[(s, w)])
                        mb[si, wi] = extractor(b_map[(s, w)])
            return ma, mb

        # 1) Decision-6 raw R1 vs clock-stripped R1_task
        d6_r1_a, d6_r1_b = build_mats(lambda r: float(r['decision6_r1']))
        d6_task_a, d6_task_b = build_mats(
            lambda r: float(r['decision6_r1']) - living_clock_r1(int(float(r['end_ticks'])))
        )
        d6_ticks_a, d6_ticks_b = build_mats(lambda r: float(r['end_ticks']))
        # 2) Exact Tick-6 R1_task (clock is identical 6 ticks on both arms!) and cells
        t6_r1_a, t6_r1_b = build_mats(lambda r: float(r['r1']) - living_clock_r1(int(float(r['tick']))))
        t6_cells_a, t6_cells_b = build_mats(lambda r: float(r['cells']))

        tag = f'{chip}_{arm}_minus_{base_arm}'
        results[tag] = dict(
            n_seeds=len(common_seeds),
            n_worlds=len(common_worlds),
            decision6_raw_r1_diff=ev.paired_bootstrap_2way(d6_r1_a, d6_r1_b, seed=101, n_boot=n_boot),
            decision6_clock_stripped_r1_diff=ev.paired_bootstrap_2way(d6_task_a, d6_task_b, seed=102, n_boot=n_boot),
            decision6_elapsed_ticks_diff=ev.paired_bootstrap_2way(d6_ticks_a, d6_ticks_b, seed=103, n_boot=n_boot),
            tick6_clock_stripped_r1_diff=ev.paired_bootstrap_2way(t6_r1_a, t6_r1_b, seed=104, n_boot=n_boot),
            tick6_new_cells_diff=ev.paired_bootstrap_2way(t6_cells_a, t6_cells_b, seed=105, n_boot=n_boot),
        )
    return results


def _audit_tick60_confirmatory(output_root: Path, n_boot: int):
    ep_path = output_root / 'mea_predictability_22021_20261119T042416Z/offline-confirmatory-results/episodes.csv'
    if not ep_path.is_file():
        return {}

    by_scene = {}
    with ep_path.open('r', encoding='utf-8-sig') as f:
        for row in csv.DictReader(f):
            sc = row['scene']
            sd = int(row['seed'])
            wd = int(row['world'])
            arm = row['arm']
            by_scene.setdefault(sc, {}).setdefault(arm, {})[(sd, wd)] = row

    out = {}
    for sc, arm_dict in sorted(by_scene.items()):
        if not {'ordered', 'masked', 'shuffled'}.issubset(arm_dict.keys()):
            continue
        seeds = sorted({s for s, _ in arm_dict['ordered']})
        worlds = sorted({w for _, w in arm_dict['ordered']})

        def mats(arm_x, arm_y, fn, common_alive_only=False):
            mx = np.full((len(seeds), len(worlds)), np.nan)
            my = np.full((len(seeds), len(worlds)), np.nan)
            for si, s in enumerate(seeds):
                for wi, w in enumerate(worlds):
                    rx = arm_dict[arm_x].get((s, w))
                    ry = arm_dict[arm_y].get((s, w))
                    if rx is None or ry is None:
                        continue
                    if common_alive_only and (int(rx['survived']) != 1 or int(ry['survived']) != 1):
                        continue
                    mx[si, wi] = fn(rx)
                    my[si, wi] = fn(ry)
            return mx, my

        scene_res = {}
        for ref in ('masked', 'shuffled'):
            surv_x, surv_y = mats('ordered', ref, lambda r: float(r['survived']))
            cells_x, cells_y = mats('ordered', ref, lambda r: float(r['cells']), common_alive_only=True)
            raw_x, raw_y = mats('ordered', ref, lambda r: float(r['r1']), common_alive_only=True)
            task_x, task_y = mats(
                'ordered', ref,
                lambda r: float(r['r1']) - living_clock_r1(int(r['end_tick'])),
                common_alive_only=True,
            )
            tpa_x, tpa_y = mats(
                'ordered', ref,
                lambda r: float(r['end_tick']) / max(1, int(r['decisions'])),
                common_alive_only=True,
            )
            dwait_x, dwait_y = mats('ordered', ref, lambda r: float(r['drink_wait']))
            ewait_x, ewait_y = mats('ordered', ref, lambda r: float(r['eat_wait']))
            scene_res[f'ordered_minus_{ref}'] = dict(
                survival_rate_diff=ev.paired_bootstrap_2way(surv_x, surv_y, seed=201, n_boot=n_boot),
                common_alive_pairs=int(np.sum(~np.isnan(task_x))),
                common_alive_tick60_cells_diff=ev.paired_bootstrap_2way(cells_x, cells_y, seed=202, n_boot=n_boot),
                common_alive_tick60_raw_r1_diff=ev.paired_bootstrap_2way(raw_x, raw_y, seed=203, n_boot=n_boot),
                common_alive_tick60_task_r1_diff=ev.paired_bootstrap_2way(task_x, task_y, seed=204, n_boot=n_boot),
                common_alive_ticks_per_action_diff=ev.paired_bootstrap_2way(tpa_x, tpa_y, seed=205, n_boot=n_boot),
                drink_wait_diff=ev.paired_bootstrap_2way(dwait_x, dwait_y, seed=206, n_boot=n_boot),
                eat_wait_diff=ev.paired_bootstrap_2way(ewait_x, ewait_y, seed=207, n_boot=n_boot),
            )
        out[sc] = scene_res
    return out


def run_audit(out_dir: Path, n_boot: int = 5000):
    out_dir.mkdir(parents=True, exist_ok=True)
    output_root = ROOT / 'output'
    short_clock = _audit_beta_sweep_clock(output_root, n_boot=n_boot)
    tick60_stress = _audit_tick60_confirmatory(output_root, n_boot=n_boot)

    audit = dict(
        schema='wetware.historical_clock_deconfounded_audit.v1',
        gamma_tick=GAMMA,
        short_horizon_clock_audit=short_clock,
        tick60_confirmatory_stress_audit=tick60_stress,
    )
    (out_dir / 'audit.json').write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding='utf-8')

    lines = [
        '# 历史归档轨迹等时刻与剥离存活计时项（`R1_task`）基线复算报告',
        '',
        '本报告零 GPU、零新训练，直接对工作区现有归档评估表执行“剥离存活计时项 `0.01 * sum(gamma^t)`”与共同存活刻对齐复算。',
        '',
        '## 1. 短程六决策 vs 固定第 6 刻（剥离动作耗时混杂前后对比）',
        '',
        '| 对比组 | 六决策原始 `R1` 差 [95% CI] | 六决策耗时差(刻) | 六决策剥离计时后 `R1_task` 差 [95% CI] | 第 6 刻 `R1_task` 差 [95% CI] | 第 6 刻新格差 [95% CI] |',
        '|---|---|---|---|---|---|',
    ]
    for k, v in short_clock.items():
        r_raw = v['decision6_raw_r1_diff']
        r_dt = v['decision6_elapsed_ticks_diff']
        r_dtask = v['decision6_clock_stripped_r1_diff']
        r_t6 = v['tick6_clock_stripped_r1_diff']
        r_c6 = v['tick6_new_cells_diff']
        lines.append(
            f"| `{k}` | `{r_raw['mean']:+.6f}` [{r_raw['ci_low']:+.6f}, {r_raw['ci_high']:+.6f}] "
            f"| `{r_dt['mean']:+.3f}` | `{r_dtask['mean']:+.6f}` [{r_dtask['ci_low']:+.6f}, {r_dtask['ci_high']:+.6f}] "
            f"| `{r_t6['mean']:+.6f}` [{r_t6['ci_low']:+.6f}, {r_t6['ci_high']:+.6f}] "
            f"| `{r_c6['mean']:+.2f}` [{r_c6['ci_low']:+.2f}, {r_c6['ci_high']:+.2f}] |"
        )

    lines.extend([
        '',
        '## 2. 第 60 刻三类高压场景（18,432 局：`low_health / thirst / hunger`）共同存活对扣时基线',
        '',
        '| 场景 | 对比 | 全样本存活率差 [95% CI] | 共同存活配对数 | 第 60 刻 `R1_task` 差 [95% CI] | 第 60 刻新格差 [95% CI] | 首次饮水等待差(刻) | 首次进食等待差(刻) |',
        '|---|---|---|---:|---|---|---|---|',
    ])
    for sc, sc_dict in tick60_stress.items():
        for comp, v in sc_dict.items():
            sv = v['survival_rate_diff']
            tk = v['common_alive_tick60_task_r1_diff']
            cl = v['common_alive_tick60_cells_diff']
            dw = v['drink_wait_diff']
            ew = v['eat_wait_diff']
            lines.append(
                f"| `{sc}` | `{comp}` | `{sv['mean']*100:+.2f}pp` [{sv['ci_low']*100:+.2f}, {sv['ci_high']*100:+.2f}] "
                f"| {v['common_alive_pairs']} | `{tk['mean']:+.6f}` [{tk['ci_low']:+.6f}, {tk['ci_high']:+.6f}] "
                f"| `{cl['mean']:+.2f}` [{cl['ci_low']:+.2f}, {cl['ci_high']:+.2f}] "
                f"| `{dw['mean']:+.2f}` | `{ew['mean']:+.2f}` |"
            )

    (out_dir / 'report.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    return audit


def build_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, default=ROOT / 'output/existing_clock_audit_20261123')
    p.add_argument('--n-boot', type=int, default=5000)
    return p


if __name__ == '__main__':
    args = build_parser().parse_args()
    run_audit(args.out, n_boot=args.n_boot)
    print((args.out / 'report.md').read_text(encoding='utf-8'))
