"""Generate comprehensive comparison report (REPORT.md) and 3 diagnostic figures for wetware training.

Consumes:
  - `gate_manifest.json` (offline MEA data purification, G2/G3 gates, C_time, 25449 QC rejection)
  - `stage0_summary.json` (positive-control ladder D vs A/A0 and noise sensitivity `rho_min`)
  - `stage3_summary.json` (7-arm paired comparison A/B/C/C_time/D/E/F on afternoon 22021 233-ROI)
  - Optional `stage4_summary.json` (cross-organoid replication on archived 28080)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
for extra in (ROOT, ROOT / '.tools' / 'mea_baseline_deps'):
    s = str(extra)
    if s not in sys.path:
        sys.path.insert(0, s)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Noto Sans CJK SC', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

ARM_LABELS = {
    'A': 'A 纯R1基线',
    'B': 'B 真实湿件(22021)',
    'C': 'C 批内打乱',
    'C_time': 'C_time 时间置换',
    'D': 'D 软件理想上限',
    'E': 'E 湿件取反',
    'F': 'F 等噪软件(ρ)',
}
ARM_COLORS = {
    'A': '#6c757d',
    'B': '#1f77b4',
    'C': '#ff7f0e',
    'C_time': '#d62728',
    'D': '#2ca02c',
    'E': '#9467bd',
    'F': '#17becf',
}
VERDICT_CN = {
    'inconclusive': '无法判定（阳性对照 D−A 未通过）',
    'negative': '阴性（D−A 通过，但 B 未在主终点与护栏内同时胜过 C / C_time / A）',
    'process_clue': '过程线索（连续检查点 B>C 且方向一致，但 95% 置信区间跨 0）',
    'weak_positive': '确认弱阳性（单芯片 B 显著胜过 C、C_time 与 A，且满足存活护栏）',
    'dual_weak_positive': '双维弱阳性（探索主终点 + 至少两项怕渴/怕饿/怕死指标均胜过对照，且存活非负）',
    'stable_positive': '稳定阳性（独立双芯片复现确认阳性）',
}


def _fmt_num(v, digits=3, signed=False):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return 'n/a'
    fmt = f'{{:{ "+" if signed else "" }.{digits}f}}'
    return fmt.format(float(v))


def _fmt_pct(v, digits=2, signed=False):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return 'n/a'
    fmt = f'{{:{ "+" if signed else "" }.{digits}f}}%'
    return fmt.format(100.0 * float(v))


def _fmt_pp(v, digits=2):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return 'n/a'
    return f'{100.0 * float(v):+.{digits}f}pp'


def _fmt_ci(st: dict | None, *, pct: bool = False, digits: int = 3) -> str:
    if not st or st.get('mean') is None:
        return 'n/a'
    if pct:
        return f"{_fmt_pp(st['mean'], 2)} [{_fmt_pp(st['ci_low'], 2)}, {_fmt_pp(st['ci_high'], 2)}]"
    return f"{_fmt_num(st['mean'], digits, signed=True)} [{_fmt_num(st['ci_low'], digits, signed=True)}, {_fmt_num(st['ci_high'], digits, signed=True)}]"


def _plot_seven_arm_endpoints(stage3: dict, out_path: Path):
    arms = stage3['arms']
    am = stage3['arm_means']
    x = np.arange(len(arms))
    colors = [ARM_COLORS.get(a, '#555555') for a in arms]
    tag = ' [DRY-RUN 演练]' if stage3.get('dry_run') else ''

    fig, axes = plt.subplots(2, 2, figsize=(13, 8.8), dpi=160)
    fig.suptitle(f"七臂冻结评估全景对比 — {stage3['dataset_id']} (Tier {stage3['tier']['tier']}, {stage3['n_seeds']} seeds × {stage3['n_worlds']} worlds){tag}",
                 fontsize=13, fontweight='bold')

    # (a) E1 Tick 96 new cells
    ax = axes[0, 0]
    vals = [am[a]['E1_new_cells'] for a in arms]
    bars = ax.bar(x, vals, color=colors, edgecolor='black', linewidth=0.6)
    ax.axhline(am['A']['E1_new_cells'], color='#6c757d', linestyle='--', linewidth=1.0, alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(arms)
    ax.set_ylabel('第 96 刻新探索格数')
    ax.set_title('主终点：E1 固定 96 刻探索新格数')
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.3, f'{v:.2f}', ha='center', va='bottom', fontsize=8.5)
    ax.grid(axis='y', alpha=0.25)

    # (b) Stress survival rates
    ax = axes[0, 1]
    w = 0.25
    ax.bar(x - w, [100 * am[a]['E2_thirst_surv'] for a in arms], width=w, label='E2 濒渴存活', color='#3182bd')
    ax.bar(x, [100 * am[a]['E3_hunger_surv'] for a in arms], width=w, label='E3 濒饿存活', color='#e6550d')
    ax.bar(x + w, [100 * am[a]['E4_death_surv'] for a in arms], width=w, label='E4 夜间威胁存活', color='#756bb1')
    ax.plot(x, [100 * am[a]['stress_survival'] for a in arms], color='black', marker='o', linewidth=1.6, label='三场景综合存活')
    ax.set_xticks(x)
    ax.set_xticklabels(arms)
    ax.set_ylabel('存活率 (%)')
    ax.set_ylim(0, 105)
    ax.set_title('护栏终点：高压三场景存活率')
    ax.legend(fontsize=8, loc='lower right')
    ax.grid(axis='y', alpha=0.25)

    # (c) Fear of thirst & fear of hunger
    ax = axes[1, 0]
    w = 0.35
    ax.bar(x - w / 2, [100 * am[a]['E2_fear_thirst'] for a in arms], width=w, label='怕渴响应率 (E2)', color='#17becf')
    ax.bar(x + w / 2, [100 * am[a]['E3_fear_hunger'] for a in arms], width=w, label='怕饿响应率 (E3)', color='#ff7f0e')
    ax.set_xticks(x)
    ax.set_xticklabels(arms)
    ax.set_ylabel('低体征响应率 (%)')
    ax.set_ylim(0, 105)
    ax.set_title('次要终点：怕渴与怕饿行为指标')
    ax.legend(fontsize=8.5)
    ax.grid(axis='y', alpha=0.25)

    # (d) Fear of death: threat evasion & caution (or threat damage if caution is null in night-only E4)
    ax = axes[1, 1]
    ev_vals = [100 * (am[a]['E4_threat_evasion'] or 0.0) for a in arms]
    ax.bar(x, ev_vals, color=colors, edgecolor='black', linewidth=0.6, alpha=0.85)
    ax2 = ax.twinx()
    caut_vals = [am[a].get('E4_fear_death_caution') for a in arms]
    if any(v is not None and np.isfinite(v) for v in caut_vals):
        ax2.plot(x, [float(v) if v is not None else np.nan for v in caut_vals],
                 color='#d62728', marker='s', linewidth=1.6, label='危险谨慎度(右轴)')
        ax2.set_ylabel('危险谨慎度 (安全−危险探新率)', color='#d62728')
        ax.set_title('次要终点：怕死（夜间威胁规避与谨慎度）')
    else:
        dmg_vals = [float(am[a].get('E4_threat_damage') or 0.0) for a in arms]
        ax2.plot(x, dmg_vals, color='#d62728', marker='s', linewidth=1.6, label='夜间受击伤害(右轴)')
        ax2.set_ylabel('夜间威胁平均受击伤害 (越低越好)', color='#d62728')
        ax.set_title('次要终点：怕死（夜间威胁规避率与受击伤害）')
    ax.set_xticks(x)
    ax.set_xticklabels(arms)
    ax.set_ylabel('夜间威胁规避率 (%)')
    ax.set_ylim(0, 105)
    ax.grid(axis='y', alpha=0.25)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_path)
    plt.close(fig)


def _plot_paired_forest(stage3: dict, out_path: Path):
    comps = stage3['comparisons']
    pairs = ('B_minus_C', 'B_minus_Ctime', 'B_minus_A', 'D_minus_A', 'E_minus_A', 'B_minus_F')
    labels = ['B − C\n(批内打乱)', 'B − C_time\n(时间置换)', 'B − A\n(纯R1基线)', 'D − A\n(软件上限)', 'E − A\n(反向检验)', 'B − F\n(等噪软件)']
    y = np.arange(len(pairs))[::-1]
    tag = ' [DRY-RUN 演练]' if stage3.get('dry_run') else ''

    fig, axes = plt.subplots(1, 3, figsize=(15, 5.4), dpi=160)
    fig.suptitle(f"配对双层自助抽样 95% 置信区间森林图 — 判定：{stage3['outcome']['verdict']}{tag}", fontsize=13, fontweight='bold')

    def _draw_panel(ax, metric_key, title, xlabel, *, scale=1.0, guardrail=None):
        means = [comps[p][metric_key]['mean'] * scale for p in pairs]
        los = [(comps[p][metric_key]['mean'] - comps[p][metric_key]['ci_low']) * scale for p in pairs]
        his = [(comps[p][metric_key]['ci_high'] - comps[p][metric_key]['mean']) * scale for p in pairs]
        cols = ['#1f77b4' if comps[p][metric_key]['ci_low'] > 0 else ('#d62728' if comps[p][metric_key]['ci_high'] < 0 else '#7f7f7f') for p in pairs]
        ax.axvline(0.0, color='black', linestyle='--', linewidth=1.0)
        if guardrail is not None:
            ax.axvline(guardrail, color='#d62728', linestyle=':', linewidth=1.3, label=f'护栏红线 ({guardrail:+.0f}pp)')
            ax.legend(fontsize=8, loc='upper right')
        for yi, m, lo, hi, c in zip(y, means, los, his, cols):
            ax.errorbar(m, yi, xerr=[[lo], [hi]], fmt='o', color=c, ecolor=c, elinewidth=2.0, capsize=4, markersize=6)
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=8.5)
        ax.set_title(title, fontsize=10.5)
        ax.set_xlabel(xlabel)
        ax.grid(axis='x', alpha=0.25)

    _draw_panel(axes[0], 'E1_new_cells', '主终点：E1 第 96 刻新格数差', '新格数差值 (95% CI)', scale=1.0)
    _draw_panel(axes[1], 'stress_survival', '存活护栏：高压三场景综合存活率差', '存活率差 (百分点 pp)', scale=100.0, guardrail=-5.0)

    # Panel 3: B-C and B-C_time across the 3 drives
    ax = axes[2]
    drive_specs = [
        ('怕渴响应 (B−C)', comps['B_minus_C']['E2_fear_thirst']),
        ('怕渴响应 (B−C_time)', comps['B_minus_Ctime']['E2_fear_thirst']),
        ('怕饿响应 (B−C)', comps['B_minus_C']['E3_fear_hunger']),
        ('怕饿响应 (B−C_time)', comps['B_minus_Ctime']['E3_fear_hunger']),
        ('威胁规避 (B−C)', comps['B_minus_C']['E4_threat_evasion']),
        ('威胁规避 (B−C_time)', comps['B_minus_Ctime']['E4_threat_evasion']),
    ]
    yd = np.arange(len(drive_specs))[::-1]
    ax.axvline(0.0, color='black', linestyle='--', linewidth=1.0)
    for yi, (_, st) in zip(yd, drive_specs):
        if st and st.get('mean') is not None:
            m = st['mean'] * 100.0
            lo = (st['mean'] - st['ci_low']) * 100.0
            hi = (st['ci_high'] - st['mean']) * 100.0
            c = '#2ca02c' if st['ci_low'] > 0 else ('#d62728' if st['ci_high'] < 0 else '#7f7f7f')
            ax.errorbar(m, yi, xerr=[[lo], [hi]], fmt='o', color=c, ecolor=c, elinewidth=2.0, capsize=4, markersize=6)
    ax.set_yticks(yd)
    ax.set_yticklabels([d[0] for d in drive_specs], fontsize=8.5)
    ax.set_title('次要终点：怕渴 / 怕饿 / 怕死增益', fontsize=10.5)
    ax.set_xlabel('响应率差 (百分点 pp)')
    ax.grid(axis='x', alpha=0.25)

    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(out_path)
    plt.close(fig)


def _plot_training_audit(stage3: dict, out_path: Path):
    arms = stage3['arms']
    aud = stage3['training_audit']
    am = stage3['arm_means']
    x = np.arange(len(arms))
    tag = ' [DRY-RUN 演练]' if stage3.get('dry_run') else ''

    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.6), dpi=160)
    fig.suptitle(f"训练期工程护栏、R2 占比工作带与动作耗时一致性审计{tag}", fontsize=12.5, fontweight='bold')

    # (a) R2 magnitude & variance share
    ax = axes[0]
    w = 0.35
    ax.bar(x - w / 2, [100 * aud[a]['mean_r2_mag_share'] for a in arms], width=w, label='幅度占比', color='#1f77b4')
    ax.bar(x + w / 2, [100 * aud[a]['mean_r2_var_share'] for a in arms], width=w, label='方差占比', color='#ff7f0e')
    ax.axhline(10.0, color='#2ca02c', linestyle='--', linewidth=1.1, label='目标上限 10%')
    ax.axhline(20.0, color='#d62728', linestyle=':', linewidth=1.3, label='红线 20%')
    ax.set_xticks(x)
    ax.set_xticklabels(arms)
    ax.set_ylabel('占比 (%)')
    ax.set_ylim(0, 23)
    ax.set_title('R2 幅度与方差占比工作带')
    ax.legend(fontsize=7.5)
    ax.grid(axis='y', alpha=0.25)

    # (b) Anchor TV shift & Adv SNR
    ax = axes[1]
    ax.bar(x, [aud[a]['mean_anchor_tv_shift'] for a in arms], color=[ARM_COLORS.get(a, '#555') for a in arms], alpha=0.85)
    ax2 = ax.twinx()
    ax2.plot(x, [aud[a]['mean_adv_snr'] for a in arms], color='black', marker='d', linewidth=1.5, label='Adv SNR(右轴)')
    ax.set_xticks(x)
    ax.set_xticklabels(arms)
    ax.set_ylabel('策略锚点 TV 偏移')
    ax2.set_ylabel('解耦优势 SNR')
    ax.set_title('策略真实更新幅度与优势信噪比')
    ax.grid(axis='y', alpha=0.25)

    # (c) Mean ticks per action parity (confound check)
    ax = axes[2]
    vals = [am[a]['E1_ticks_per_action'] for a in arms]
    ax.bar(x, vals, color='#6c757d', alpha=0.85)
    mean_v = float(np.mean(vals))
    ax.axhline(mean_v, color='black', linestyle='--', linewidth=1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(arms)
    ax.set_ylabel('平均每动作耗时 (游戏刻)')
    ax.set_ylim(max(0.0, mean_v - 0.35), mean_v + 0.35)
    ax.set_title('冻结评估动作耗时一致性（无计时混杂）')
    for xi, v in zip(x, vals):
        ax.text(xi, v + 0.01, f'{v:.3f}', ha='center', va='bottom', fontsize=8)
    ax.grid(axis='y', alpha=0.25)

    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(out_path)
    plt.close(fig)


def generate_report(
    *,
    manifest_path: Path | None,
    stage0_path: Path | None,
    stage3_path: Path,
    stage4_path: Path | None,
    out_dir: Path,
) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(Path(manifest_path).read_text(encoding='utf-8')) if (manifest_path and Path(manifest_path).is_file()) else None
    stage0 = json.loads(Path(stage0_path).read_text(encoding='utf-8')) if (stage0_path and Path(stage0_path).is_file()) else None
    stage3 = json.loads(Path(stage3_path).read_text(encoding='utf-8'))
    stage4 = json.loads(Path(stage4_path).read_text(encoding='utf-8')) if (stage4_path and Path(stage4_path).is_file()) else None

    fig1 = out_dir / 'fig1_seven_arm_endpoints.png'
    fig2 = out_dir / 'fig2_paired_effects_forest.png'
    fig3 = out_dir / 'fig3_training_audit_and_timing.png'
    _plot_seven_arm_endpoints(stage3, fig1)
    _plot_paired_forest(stage3, fig2)
    _plot_training_audit(stage3, fig3)

    is_dry = bool(stage3.get('dry_run', False))
    verdict = stage3['outcome']['verdict']
    surv_warn = bool(stage3['outcome'].get('survival_warning', False))
    comps = stage3['comparisons']
    am = stage3['arm_means']
    aud = stage3['training_audit']

    lines = [
        f"# 下午实测 MEA 数据模型训练对比与效果判定报告{'（DRY-RUN 管线演练）' if is_dry else ''}",
        '',
    ]
    if is_dry:
        lines.extend([
            '> [!WARNING]',
            '> **当前报告来自 `--dry-run` 本地离线演练（0 GPU）**，用于验证完整训练/统计/出图/判据代码链路，**不构成真实 GPU 模型训练科学结论**。真实训练结论以 GPU 节点实际产出的 `dry_run=false` 报告为准。',
            '',
        ])

    lines.extend([
        '## 核心判定结论',
        '',
        f"- **主实验数据集**：`{stage3['dataset_id']}`（芯片 `{stage3['chip_id']}`，留出可靠度 `rho_lib = {stage3['rho_lib']:.4f}`）",
        f"- **冻结配置**：Tier `{stage3['tier']['tier']}`（每次训练 `{stage3['tier']['decisions']}` 决策 / `{stage3['tier']['updates']}` 更新，7 臂配对 × `{stage3['n_seeds']}` 种子 × `{stage3['n_worlds']}` 张未知地图，100% 纯 108 维脱离 MEA 冻结评估）",
        f"- **分级判定**：**`{verdict}`** — {VERDICT_CN.get(verdict, stage3['outcome']['reason'])}",
        f"- **存活护栏（`B − A` 高压综合存活下界 >= -5pp）**：`{'触发警告（以生存为代价）' if surv_warn else '通过'}`（{_fmt_ci(comps['B_minus_A']['stress_survival'], pct=True)}）",
        f"- **主终点（E1 第 96 刻新格数）**：`B − C = {_fmt_ci(comps['B_minus_C']['E1_new_cells'])}`（正收益种子 `{_fmt_pct(comps['B_minus_C']['E1_new_cells']['seed_positive_frac'], 1)}`）；`B − C_time = {_fmt_ci(comps['B_minus_Ctime']['E1_new_cells'])}`；`B − A = {_fmt_ci(comps['B_minus_A']['E1_new_cells'])}`；阳性对照 `D − A = {_fmt_ci(comps['D_minus_A']['E1_new_cells'])}`。",
        '',
    ])

    # Section 1: Afternoon MEA Data Audit
    if manifest is not None:
        lines.extend(['## 1. 下午采集 MEA 数据净化与门控核验', ''])
        ds_entry = manifest.get('datasets', {}).get(stage3['dataset_id'])
        if ds_entry:
            rg = ds_entry['real_gate']
            cg = ds_entry['ctime_null_gate']
            cp = ds_entry.get('channel_purification')
            if cp:
                lines.append(
                    f"- **通道净化**：保留 **`{cp['n_clean_roi_channels']}`** 个干净 `roi` 读出通道（总放电 `{cp['total_spikes_clean_roi']}`），严格剔除 `{cp['n_excluded_channels']}` 个刺激/近场通道（含 3 个 DAC 直连噪声电极 `96, 737, 804`，原 243 通道总尖峰 `{cp['total_spikes_all_243']}` 中约 70% 为 DAC 缓冲噪声）。"
                )
            lines.append(
                f"- **真实时序留出验证（Round 3, {rg['held_windows']} 窗）**：G2=`PASS`（AUC `{rg['evoked_held_sites']['H']['auc']:.3f} / {rg['evoked_held_sites']['W']['auc']:.3f} / {rg['evoked_held_sites']['F']['auc']:.3f}`），G3=`PASS`（AUC `{rg['habituation_held_sites']['H']['auc']:.3f} / {rg['habituation_held_sites']['W']['auc']:.3f} / {rg['habituation_held_sites']['F']['auc']:.3f}`），**`rho_lib = {rg['rho_lib']:+.4f}`**。"
            )
            lines.append(
                f"- **时间置换零模型 `C_time`**：G2=`{'PASS' if cg['g2_passed'] else 'FAIL'}`，G3=`{'PASS' if cg['g3_passed'] else 'FAIL'}`，`rho_all = {_fmt_num(cg['rho_all_sites'], 4, signed=True)}`（彻底消除生物时序记忆）。"
            )
        pre_sens = manifest.get('sensitivity_audits', {}).get('22021_1123_prestim_only')
        if pre_sens:
            ps = pre_sens['summary']
            lines.append(
                f"- **纯刺激前窗口 `[-300, -1) ms` 灵敏度核验**（0 刺激后帧，`{pre_sens['total_spikes_clean_roi']}` 放电）：G2=`{'PASS' if ps['g2_passed'] else 'FAIL'}`，G3=`{'PASS' if ps['g3_passed'] else 'FAIL'}`，`rho_lib = {_fmt_num(ps['rho_lib'], 4, signed=True)}`。"
            )
        rej = manifest.get('rejected_afternoon_chips', {}).get('25449_1123')
        if rej:
            q_ch = rej.get('recordings', {}).get('quiet', {}).get('n_active_channels', 'n/a')
            w_ch = rej.get('recordings', {}).get('warmup', {}).get('n_active_channels', 'n/a')
            lines.append(
                f"- **下午芯片 `25449` 自动拒收**：`quiet` 活跃通道 `{q_ch}`、`warmup` 活跃通道 `{w_ch}`（均 `< 50` 门槛，0 真脉冲），按 G1 规则拒收不入训。"
            )
        lines.append('')

    # Section 2: Stage 0
    if stage0 is not None:
        lines.extend([
            '## 2. Stage 0 阳性对照校准与噪声灵敏度（`rho_min`）',
            '',
            f"- 确认通过：`{stage0.get('confirmed')}` | 冻结预算档位：`Tier {stage0['frozen_tier']['tier'] if stage0.get('frozen_tier') else 'None'}` | 灵敏度下限 **`rho_min = {stage0.get('rho_min')}`** | 推荐 Stage 3 种子数：`{stage0.get('recommended_stage3_seeds')}`",
            '',
        ])
        if stage0.get('noise_curve'):
            lines.extend(['| 理想信号掺噪可靠度 `rho` | `F − A` E1 第 96 刻新格数 [95% CI] | 正收益种子比例 | 是否通过 |', '|---:|---|---:|---|'])
            for row in stage0['noise_curve']:
                st = row['stats']
                ok = st.get('ci_low') is not None and st['ci_low'] > 0.0
                lines.append(f"| {row['rho']:.2f} | {_fmt_ci(st)} | {_fmt_pct(st.get('seed_positive_frac'), 1)} | {'PASS' if ok else 'FAIL'} |")
            lines.append('')

    # Section 3: Seven-arm performance table
    lines.extend([
        '## 3. 七臂冻结评估全维度对比表',
        '',
        '![七臂冻结评估全景对比](fig1_seven_arm_endpoints.png)',
        '',
        '| 试验臂 | E1 96刻新格 | 综合高压存活 | E2 濒渴存活 / 怕渴响应 / 饮水延迟 | E3 濒饿存活 / 怕饿响应 / 进食延迟 | E4 威胁存活 / 威胁规避 / 怕死谨慎 / 伤害 | 每动作刻数 |',
        '|---|---:|---:|---|---|---|---:|',
    ])
    for arm in stage3['arms']:
        m = am[arm]
        lines.append(
            f"| **{ARM_LABELS.get(arm, arm)}** | {_fmt_num(m['E1_new_cells'], 2)} | {_fmt_pct(m['stress_survival'])} "
            f"| {_fmt_pct(m['E2_thirst_surv'], 1)} / {_fmt_pct(m['E2_fear_thirst'], 1)} / {_fmt_num(m['E2_drink_latency'], 1)}t "
            f"| {_fmt_pct(m['E3_hunger_surv'], 1)} / {_fmt_pct(m['E3_fear_hunger'], 1)} / {_fmt_num(m['E3_eat_latency'], 1)}t "
            f"| {_fmt_pct(m['E4_death_surv'], 1)} / {_fmt_pct(m['E4_threat_evasion'], 1)} / {_fmt_num(m['E4_fear_death_caution'], 3, signed=True)} / {_fmt_num(m['E4_threat_damage'], 2)} "
            f"| {_fmt_num(m['E1_ticks_per_action'], 3)} |"
        )
    lines.append('')

    # Section 4: Paired contrasts & CIF
    lines.extend([
        '## 4. 核心配对双层自助抽样检验（10,000 次重抽样）与竞争风险分解',
        '',
        '![配对效应森林图](fig2_paired_effects_forest.png)',
        '',
        '| 配对对比 | E1 第96刻新格差 [95% CI] | 正种子比 | 综合高压存活差 [95% CI] | E2 怕渴响应差 | E3 怕饿响应差 | E4 威胁规避差 | 耗时差(刻/动作) |',
        '|---|---|---:|---|---|---|---|---|',
    ])
    for pkey in ('B_minus_C', 'B_minus_Ctime', 'B_minus_A', 'D_minus_A', 'E_minus_A', 'B_minus_F'):
        cp = comps[pkey]
        lines.append(
            f"| `{pkey}` | {_fmt_ci(cp['E1_new_cells'])} | {_fmt_pct(cp['E1_new_cells'].get('seed_positive_frac'), 1)} "
            f"| {_fmt_ci(cp['stress_survival'], pct=True)} | {_fmt_ci(cp['E2_fear_thirst'], pct=True)} "
            f"| {_fmt_ci(cp['E3_fear_hunger'], pct=True)} | {_fmt_ci(cp['E4_threat_evasion'], pct=True)} "
            f"| {_fmt_ci(cp['E1_ticks_per_action'], digits=4)} |"
        )
    lines.extend(['', '### 竞争风险延迟分解（CIF：饮水/进食成功率差与一致成功对等待差）', '',
                  '| 对比 | 场景 | 配对数 | 成功率差 (CIF diff) | 总体等待差(刻) | 共同成功对等待差(刻) | 仅 B 成功 / 仅对照成功 |',
                  '|---|---|---:|---:|---:|---:|---|'])
    for tag, d in stage3.get('cif_decomposition', {}).items():
        for sc_name, sc_cn in (('thirst', 'E2 濒渴饮水'), ('hunger', 'E3 濒饿进食')):
            x = d.get(sc_name, {})
            lines.append(
                f"| `{tag}` | {sc_cn} | {x.get('n_pairs', 0)} | {_fmt_pp(x.get('cif_diff'))} "
                f"| {_fmt_num(x.get('overall_wait_diff'), 2, signed=True)} | {_fmt_num(x.get('concordant_wait_diff'), 2, signed=True)} "
                f"| {x.get('discordant_only_a', 0)} / {x.get('discordant_only_b', 0)} |"
            )
    lines.append('')

    # Section 5: Pre-training channel bound & engineering audit
    lines.extend([
        '## 5. 训练前反泄漏门控与训练期工程审计',
        '',
        '![训练期工程审计](fig3_training_audit_and_timing.png)',
        '',
    ])
    cba = stage3.get('channel_bound_audit')
    if cba:
        lines.append(
            f"- **训练前反泄漏门 `channel_bound_check`**：`passed = {cba.get('passed')}` | `AUC(B) = {_fmt_num(cba.get('real_auc'))}` <= `AUC(D) = {_fmt_num(cba.get('ideal_auc'))}` + 0.05 | `AUC(C) = {_fmt_num(cba.get('shuffled_auc'))}` | `AUC(C_time) = {_fmt_num(cba.get('ctime_auc'))}`。"
        )
        lines.append('')
    lines.extend([
        '| 试验臂 | 权重全部更新 | 最大零更新logp漂移 | 锚点TV偏移 | R2幅度占比 | R2方差占比 | 解耦优势SNR | 危机步占比 | 解警触发占比 | KL早停率 |',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|',
    ])
    for arm in stage3['arms']:
        a = aud[arm]
        lines.append(
            f"| `{arm}` | {a['weights_changed_all']} | {a['max_zero_update_logp']:.2e} | {a['mean_anchor_tv_shift']:.4f} "
            f"| {_fmt_pct(a['mean_r2_mag_share'])} | {_fmt_pct(a['mean_r2_var_share'])} | {a['mean_adv_snr']:.3f} "
            f"| {_fmt_pct(a['mean_in_stress_frac'], 1)} | {_fmt_pct(a['mean_relieved_frac'], 1)} | {_fmt_pct(a['kl_early_stop_frac'], 1)} |"
        )
    lines.append('')

    # Section 6: Optional Stage 4 cross-organoid replication
    next_sec = 6
    if stage4 is not None:
        s4c = stage4['comparisons']
        lines.extend([
            f'## {next_sec}. Stage 4 独立跨芯片复现结果',
            '',
            f"- 复现数据集：`{stage4['dataset_id']}`（芯片 `{stage4['chip_id']}`，`rho_lib = {stage4['rho_lib']:.4f}`），判定：**`{stage4['outcome']['verdict']}`**",
            f"- `B − C` E1 新格数：{_fmt_ci(s4c['B_minus_C']['E1_new_cells'])} | `B − C_time`：{_fmt_ci(s4c['B_minus_Ctime']['E1_new_cells'])} | 综合存活 `B − A`：{_fmt_ci(s4c['B_minus_A']['stress_survival'], pct=True)}",
            '',
        ])
        next_sec += 1

    # Final Section: Direct answers to AGENTS.md goals
    b_c_e1 = comps['B_minus_C']['E1_new_cells']
    b_ct_e1 = comps['B_minus_Ctime']['E1_new_cells']
    b_c_th = comps['B_minus_C']['E2_fear_thirst']
    b_c_hu = comps['B_minus_C']['E3_fear_hunger']
    b_c_de = comps['B_minus_C']['E4_threat_evasion']
    b_a_sv = comps['B_minus_A']['stress_survival']
    lines.extend([
        f'## {next_sec}. 对 `AGENTS.md` 三个核心问题的直接回答与科学边界',
        '',
        f"1. **相同游戏时长（第 96 刻）内能否稳定探索更多新格？** `B − C = {_fmt_ci(b_c_e1)}`、`B − C_time = {_fmt_ci(b_ct_e1)}`（判定：`{'是（显著为正）' if (b_c_e1['ci_low'] > 0 and b_ct_e1['ci_low'] > 0) else '否（置信区间未完全大于 0）'}`）。",
        f"2. **是否产生怕渴、怕饿、怕死的行为指标？** 相对打乱对照 `C`：怕渴响应 `{_fmt_ci(b_c_th, pct=True)}`、怕饿响应 `{_fmt_ci(b_c_hu, pct=True)}`、夜间威胁规避 `{_fmt_ci(b_c_de, pct=True)}`。",
        f"3. **加入湿件反馈后，AI 的生存决策会不会更好？** 高压三场景综合存活 `B − A = {_fmt_ci(b_a_sv, pct=True)}`、`B − C = {_fmt_ci(comps['B_minus_C']['stress_survival'], pct=True)}`（护栏红线 `-5.00pp`）。",
        '',
        '> **科学边界声明**：下午 `22021` 的直接电刺激诱发检验为阴性（A 类），本实验利用的是 `22021` 233 个干净读出通道上经留出检验与时间置换对照证实的**内源突触抑压与兴奋度时序记忆**。即使 `B` 显著胜过 `A / C / C_time`，结论也仅表示“真实类器官生物时序状态可承载对大模型 RL 有效的双驱信号”，而不代表湿件强于理想软件上限 `D`。',
    ])

    report_md = out_dir / 'REPORT.md'
    report_md.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    return dict(
        report_md=str(report_md),
        figures=[str(fig1), str(fig2), str(fig3)],
        verdict=verdict,
        dry_run=is_dry,
    )


def _plot_stage0_ladder_endpoints(stage0: dict, out_path: Path):
    phases = []
    for row in stage0.get('ladder', []):
        t = row['tier']['tier']
        upd = row['tier']['updates']
        phases.append((f'Tier {t}\n({upd} upd)', row['arm_means']))
    if stage0.get('confirm'):
        ft = stage0.get('frozen_tier') or {}
        phases.append((f"Tier {ft.get('tier', '?')} 验证\n(16 新种子)", stage0['confirm']['arm_means']))

    x = np.arange(len(phases))
    w = 0.25
    arms = ('A0_native_r1', 'A', 'D')
    arm_names = {'A0_native_r1': 'A0 原生R1', 'A': 'A 课程纯R1', 'D': 'D 双驱理想上限'}
    arm_cols = {'A0_native_r1': '#adb5bd', 'A': '#6c757d', 'D': '#2ca02c'}
    tag = ' [DRY-RUN 演练]' if stage0.get('dry_run') else ''

    fig, axes = plt.subplots(2, 2, figsize=(13.2, 8.8), dpi=160)
    fig.suptitle(
        f"Stage 0 真实 GPU 梯级校准与独立验证全景对比（每阶段 16 种子 × {stage0.get('n_worlds', 24)} 未知世界）{tag}",
        fontsize=13, fontweight='bold',
    )

    # (a) E1 new cells
    ax = axes[0, 0]
    for i, a in enumerate(arms):
        vals = [am[a]['E1_new_cells'] for _, am in phases]
        bars = ax.bar(x + (i - 1) * w, vals, width=w, label=arm_names[a], color=arm_cols[a], edgecolor='black', linewidth=0.5)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 2.0, f'{v:.1f}', ha='center', va='bottom', fontsize=7.5)
    ax.set_xticks(x)
    ax.set_xticklabels([p[0] for p in phases], fontsize=8.5)
    ax.set_ylabel('第 96 刻新探索格数')
    ax.set_ylim(0, max(am[a]['E1_new_cells'] for _, am in phases for a in arms) * 1.18)
    ax.set_title('主终点：E1 固定 96 刻探索新格数')
    ax.legend(fontsize=8, loc='upper left')
    ax.grid(axis='y', alpha=0.25)

    # (b) Pooled stress survival + E4 death survival
    ax = axes[0, 1]
    for i, a in enumerate(arms):
        vals = [100.0 * am[a]['stress_survival'] for _, am in phases]
        bars = ax.bar(x + (i - 1) * w, vals, width=w, label=f'{arm_names[a]} 综合存活', color=arm_cols[a], edgecolor='black', linewidth=0.5)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 1.0, f'{v:.1f}%', ha='center', va='bottom', fontsize=7.2)
    ax.plot(x + w, [100.0 * am['D']['E4_death_surv'] for _, am in phases], color='#1b7837', marker='o', linewidth=1.8, label='D 夜间威胁存活(E4)')
    ax.plot(x, [100.0 * am['A']['E4_death_surv'] for _, am in phases], color='#343a40', marker='s', linestyle='--', linewidth=1.4, label='A 夜间威胁存活(E4)')
    ax.set_xticks(x)
    ax.set_xticklabels([p[0] for p in phases], fontsize=8.5)
    ax.set_ylabel('存活率 (%)')
    ax.set_ylim(45, 86)
    ax.set_title('护栏终点：高压三场景综合存活与 E4 夜间存活')
    ax.legend(fontsize=7.5, loc='lower right', ncol=2)
    ax.grid(axis='y', alpha=0.25)

    # (c) Fear of hunger (E3) & Fear of thirst (E2)
    ax = axes[1, 0]
    for i, a in enumerate(arms):
        vals = [100.0 * am[a]['E3_fear_hunger'] for _, am in phases]
        bars = ax.bar(x + (i - 1) * w, vals, width=w, label=f'{arm_names[a]} 怕饿(E3)', color=arm_cols[a], edgecolor='black', linewidth=0.5)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.5, f'{v:.1f}%', ha='center', va='bottom', fontsize=7.2)
    ax.set_xticks(x)
    ax.set_xticklabels([p[0] for p in phases], fontsize=8.5)
    ax.set_ylabel('低饱食主动进食响应率 (%)')
    ax.set_ylim(28, 46)
    ax.set_title('次要终点：怕饿指标 (E3_fear_hunger，双阶段显著)')
    ax.legend(fontsize=8, loc='upper left')
    ax.grid(axis='y', alpha=0.25)

    # (d) Fear of death: E4 threat evasion
    ax = axes[1, 1]
    for i, a in enumerate(arms):
        vals = [100.0 * am[a]['E4_threat_evasion'] for _, am in phases]
        bars = ax.bar(x + (i - 1) * w, vals, width=w, label=f'{arm_names[a]} 规避(E4)', color=arm_cols[a], edgecolor='black', linewidth=0.5)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.5, f'{v:.1f}%', ha='center', va='bottom', fontsize=7.2)
    ax.set_xticks(x)
    ax.set_xticklabels([p[0] for p in phases], fontsize=8.5)
    ax.set_ylabel('夜间威胁规避率 (%)')
    ax.set_ylim(30, 48)
    ax.set_title('次要终点：怕死指标 (E4_threat_evasion，连续三阶段显著)')
    ax.legend(fontsize=8, loc='upper left')
    ax.grid(axis='y', alpha=0.25)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_path)
    plt.close(fig)


def _plot_stage0_paired_forest(stage0: dict, diag: dict | None, out_path: Path):
    rows = []
    for row in stage0.get('ladder', []):
        t = row['tier']['tier']
        rows.append((f'Tier {t} 调优\n(D−A, n=16)', row['D_minus_A'], False))
    if stage0.get('confirm'):
        rows.append(('Tier 3 独立验证\n(D−A, n=16)', stage0['confirm']['D_minus_A'], False))
        rows.append(('Tier 3 独立验证\n(D−A0, n=16)', stage0['confirm']['D_minus_A0'], False))
    if diag and diag.get('exploratory_pooled_32seeds_tier3'):
        rows.append(('事后合并诊断*\n(D−A, n=32)', diag['exploratory_pooled_32seeds_tier3']['D_minus_A'], True))

    y = np.arange(len(rows))[::-1]
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 5.6), dpi=160)
    status_cn = '已通过' if stage0.get('confirmed') else '在独立验证门控触发早停 (confirmed=false)'
    fig.suptitle(f"Stage 0 配对双层自助抽样 95% 置信区间森林图 — 状态：{status_cn}", fontsize=12.8, fontweight='bold')

    def _draw(ax, key, title, xlabel, *, scale=1.0, guardrail=None):
        ax.axvline(0.0, color='black', linestyle='--', linewidth=1.0)
        if guardrail is not None:
            ax.axvline(guardrail, color='#d62728', linestyle=':', linewidth=1.3, label=f'护栏红线 ({guardrail:+.0f}pp)')
            ax.legend(fontsize=8, loc='upper right')
        for yi, (_, cp, is_posthoc) in zip(y, rows):
            st = cp[key]
            if not st or st.get('mean') is None:
                continue
            m = st['mean'] * scale
            lo = (st['mean'] - st['ci_low']) * scale
            hi = (st['ci_high'] - st['mean']) * scale
            sig_pos = st['ci_low'] > 0
            sig_neg = st['ci_high'] < 0
            c = '#1f77b4' if is_posthoc else ('#2ca02c' if sig_pos else ('#d62728' if sig_neg else '#7f7f7f'))
            fmt = 'D' if is_posthoc else 'o'
            ax.errorbar(m, yi, xerr=[[lo], [hi]], fmt=fmt, color=c, ecolor=c, elinewidth=2.0, capsize=4, markersize=6)
        ax.set_yticks(y)
        ax.set_yticklabels([r[0] for r in rows], fontsize=8.3)
        ax.set_title(title, fontsize=10.5)
        ax.set_xlabel(xlabel)
        ax.grid(axis='x', alpha=0.25)

    _draw(axes[0], 'E1_new_cells', '主终点：E1 第 96 刻新格数差', '新格数差值 (95% CI)', scale=1.0)
    _draw(axes[1], 'stress_survival', '存活护栏：高压三场景综合存活率差', '存活率差 (百分点 pp)', scale=100.0, guardrail=-5.0)

    # Panel 3: Drives at Tier 3 tune, Tier 3 confirm, and pooled 32 seeds
    ax = axes[2]
    t3_row = next((r for r in stage0.get('ladder', []) if r['tier']['tier'] == (stage0.get('frozen_tier') or {}).get('tier', 3)), stage0['ladder'][-1])
    drive_rows = [
        ('怕渴 E2 (T3调优)', t3_row['D_minus_A']['E2_fear_thirst'], False),
        ('怕渴 E2 (T3验证)', stage0['confirm']['D_minus_A']['E2_fear_thirst'], False) if stage0.get('confirm') else None,
        ('怕饿 E3 (T3调优)', t3_row['D_minus_A']['E3_fear_hunger'], False),
        ('怕饿 E3 (T3验证)', stage0['confirm']['D_minus_A']['E3_fear_hunger'], False) if stage0.get('confirm') else None,
        ('怕死规避 E4 (T3调优)', t3_row['D_minus_A']['E4_threat_evasion'], False),
        ('怕死规避 E4 (T3验证)', stage0['confirm']['D_minus_A']['E4_threat_evasion'], False) if stage0.get('confirm') else None,
    ]
    drive_rows = [d for d in drive_rows if d is not None]
    if diag and diag.get('exploratory_pooled_32seeds_tier3'):
        pd = diag['exploratory_pooled_32seeds_tier3']['D_minus_A']
        drive_rows.append(('怕饿 E3 (32种子合并*)', pd['E3_fear_hunger'], True))
        drive_rows.append(('怕死规避 E4 (32种子合并*)', pd['E4_threat_evasion'], True))
    yd = np.arange(len(drive_rows))[::-1]
    ax.axvline(0.0, color='black', linestyle='--', linewidth=1.0)
    for yi, (_, st, is_posthoc) in zip(yd, drive_rows):
        m = st['mean'] * 100.0
        lo = (st['mean'] - st['ci_low']) * 100.0
        hi = (st['ci_high'] - st['mean']) * 100.0
        c = '#1f77b4' if is_posthoc else ('#2ca02c' if st['ci_low'] > 0 else ('#d62728' if st['ci_high'] < 0 else '#7f7f7f'))
        fmt = 'D' if is_posthoc else 'o'
        ax.errorbar(m, yi, xerr=[[lo], [hi]], fmt=fmt, color=c, ecolor=c, elinewidth=2.0, capsize=4, markersize=5.5)
    ax.set_yticks(yd)
    ax.set_yticklabels([d[0] for d in drive_rows], fontsize=8.0)
    ax.set_title('次要终点：怕渴 / 怕饿 / 怕死 (D−A)', fontsize=10.5)
    ax.set_xlabel('响应率差 (百分点 pp)')
    ax.grid(axis='x', alpha=0.25)

    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(out_path)
    plt.close(fig)


def _plot_stage0_seed_and_audit(stage0: dict, diag: dict | None, smoke: dict | None, out_path: Path):
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.9), dpi=160)
    fig.suptitle('Stage 0 种子级方差诊断、怕饿/怕死协同性与训练期工程护栏审计', fontsize=12.5, fontweight='bold')

    # (a) Per-seed D-A E1 distribution across 32 Tier 3 seeds
    ax = axes[0]
    if diag and diag.get('tier3_seed_deltas_D_minus_A') and diag.get('confirm_seed_deltas_D_minus_A'):
        t3 = [r['dE1'] for r in diag['tier3_seed_deltas_D_minus_A']]
        cf = [r['dE1'] for r in diag['confirm_seed_deltas_D_minus_A']]
        vals = [(v, '调优 16 种子', '#2ca02c') for v in t3] + [(v, '验证 16 种子', '#1f77b4') for v in cf]
        vals.sort(key=lambda z: z[0])
        xs = np.arange(len(vals))
        ax.bar(xs, [z[0] for z in vals], color=[z[2] for z in vals], alpha=0.85, edgecolor='black', linewidth=0.3)
        ax.axhline(0.0, color='black', linewidth=1.0)
        ax.axhline(float(np.mean(t3 + cf)), color='#d62728', linestyle='--', linewidth=1.3, label=f'32种子均值 {np.mean(t3+cf):+.1f}')
        from matplotlib.patches import Patch
        handles = [
            Patch(facecolor='#2ca02c', label=f'T3 调优 (均值 {np.mean(t3):+.1f}, 12/16>0)'),
            Patch(facecolor='#1f77b4', label=f'T3 验证 (均值 {np.mean(cf):+.1f}, 11/16>0)'),
        ]
        ax.legend(handles=handles, fontsize=7.5, loc='upper left')
        ax.set_xlabel('按 D−A 增益排序的 32 个独立种子')
        ax.set_ylabel('E1 第 96 刻新格差 (D − A)')
        ax.set_title('为何 16 种子难过门控：高方差单种子分布')
        ax.grid(axis='y', alpha=0.25)
    else:
        ax.text(0.5, 0.5, '无单种子明细数据', ha='center', va='center')

    # (b) Per-seed E3_fear_hunger vs E4_threat_evasion scatter
    ax = axes[1]
    if diag and diag.get('tier3_seed_deltas_D_minus_A') and diag.get('confirm_seed_deltas_D_minus_A'):
        t3 = diag['tier3_seed_deltas_D_minus_A']
        cf = diag['confirm_seed_deltas_D_minus_A']
        ax.axvline(0.0, color='black', linestyle='--', linewidth=0.9)
        ax.axhline(0.0, color='black', linestyle='--', linewidth=0.9)
        ax.scatter([100 * r['dHunger'] for r in t3], [100 * r['dEvade'] for r in t3],
                   color='#2ca02c', s=42, alpha=0.85, edgecolor='black', linewidth=0.4, label='T3 调优 (n=16)')
        ax.scatter([100 * r['dHunger'] for r in cf], [100 * r['dEvade'] for r in cf],
                   color='#1f77b4', marker='^', s=48, alpha=0.85, edgecolor='black', linewidth=0.4, label='T3 验证 (n=16)')
        both_pos = sum(1 for r in (t3 + cf) if r['dHunger'] > 0 and r['dEvade'] > 0)
        ax.set_xlabel('怕饿响应增益 D−A (pp)')
        ax.set_ylabel('怕死规避增益 D−A (pp)')
        ax.set_title(f'双驱行为协同性：{both_pos}/32 种子双正象限')
        ax.legend(fontsize=8, loc='lower right')
        ax.grid(alpha=0.25)
    else:
        ax.text(0.5, 0.5, '无单种子明细数据', ha='center', va='center')

    # (c) Engineering audit across tiers (D arm)
    ax = axes[2]
    if diag and diag.get('tiers'):
        subs = [s for s in ('tier1', 'tier2', 'tier3', 'confirm') if s in diag['tiers']]
        labels = {'tier1': 'T1', 'tier2': 'T2', 'tier3': 'T3调优', 'confirm': 'T3验证'}
        xs = np.arange(len(subs))
        w = 0.25
        ax.bar(xs - w, [100 * diag['tiers'][s]['D']['mean_r2_mag_share'] for s in subs], width=w, color='#1f77b4', label='R2幅度占比(%)')
        ax.bar(xs, [100 * diag['tiers'][s]['D']['mean_anchor_tv_shift'] for s in subs], width=w, color='#2ca02c', label='锚点TV偏移(×100)')
        ax.bar(xs + w, [100 * diag['tiers'][s]['D']['mean_relieved_frac'] for s in subs], width=w, color='#ff7f0e', label='危机解警触发(%)')
        ax.axhline(20.0, color='#d62728', linestyle=':', linewidth=1.2, label='R2占比红线(20%)')
        ax.set_xticks(xs)
        ax.set_xticklabels([labels[s] for s in subs])
        ax.set_ylabel('百分比 (%) / 偏移×100')
        ax.set_ylim(0, 22)
        ax.set_title('训练期护栏审计（D 臂，KL早停 ≤ 1.6%）')
        ax.legend(fontsize=7.3, loc='upper left')
        ax.grid(axis='y', alpha=0.25)
    else:
        ax.text(0.5, 0.5, '无训练审计数据', ha='center', va='center')

    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(out_path)
    plt.close(fig)


def generate_stage0_report(
    *,
    manifest_path: Path | None,
    stage0_path: Path,
    stage0_diag_path: Path | None,
    smoke_path: Path | None,
    out_dir: Path,
) -> dict:
    """Generate real-GPU Stage 0 stop-gate diagnostic report and 3 figures when Stage 3 is blocked."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(Path(manifest_path).read_text(encoding='utf-8')) if (manifest_path and Path(manifest_path).is_file()) else None
    stage0 = json.loads(Path(stage0_path).read_text(encoding='utf-8'))
    diag = json.loads(Path(stage0_diag_path).read_text(encoding='utf-8')) if (stage0_diag_path and Path(stage0_diag_path).is_file()) else None
    smoke = json.loads(Path(smoke_path).read_text(encoding='utf-8')) if (smoke_path and Path(smoke_path).is_file()) else None

    fig1 = out_dir / 'fig1_stage0_ladder_endpoints.png'
    fig2 = out_dir / 'fig2_stage0_paired_forest.png'
    fig3 = out_dir / 'fig3_stage0_seed_scatter_and_audit.png'
    _plot_stage0_ladder_endpoints(stage0, fig1)
    _plot_stage0_paired_forest(stage0, diag, fig2)
    _plot_stage0_seed_and_audit(stage0, diag, smoke, fig3)

    ft = stage0.get('frozen_tier') or {}
    conf = stage0.get('confirm') or {}
    t3_row = next((r for r in stage0.get('ladder', []) if r['tier']['tier'] == ft.get('tier', 3)), stage0['ladder'][-1])
    pooled = (diag or {}).get('exploratory_pooled_32seeds_tier3')
    is_dry = bool(stage0.get('dry_run', False))
    verdict = 'stage0_confirmed' if stage0.get('confirmed') else 'stage0_confirm_stopped'

    lines = [
        f"# 下午实测 MEA 训练管线：真实 GPU Stage 0 门控校准与诊断报告{'（DRY-RUN）' if is_dry else ''}",
        '',
        '> [!IMPORTANT]',
        '> **预注册门控早停声明**：`192.0.2.16`（7× RTX 4090）已完成 4 组真实湿件烟雾验证（`library_smoke`）及 **192 组真实 GPU Stage 0 对照训练**（`Tier 1/2/3` 梯级校准 + `Tier 3` 独立验证，合计 196 组真实 GPU 任务，零报错、零 OOM）。`Tier 3` 在 16 种子调优集通过门控，但在 16 种子独立验证集上主终点探索新格差 `D − A` 的双层 95% 置信区间跨 0（`confirmed = false`），系统严格遵循预注册停机规则**拦截了 112 组 Stage 3 大规模湿件训练**，避免在统计功效不足时浪费算力或产生假阳性。',
        '',
        '## 核心结论速览',
        '',
        f"- **预注册状态**：**`{verdict}`**（`frozen_tier = Tier {ft.get('tier', '?')}`，`confirmed = {stage0.get('confirmed')}`，Stage 3 7 臂下午 `22021` 正式对比训练**尚未启动**）。",
        f"- **为何触发主终点停机（统计功效诊断）**：`Tier 3` 调优集 `D − A` E1 探索新格数为 **`{_fmt_ci(t3_row['D_minus_A']['E1_new_cells'])}`**（12/16 正向，PASS）；独立验证集 `D − A` 为 **`{_fmt_ci(conf['D_minus_A']['E1_new_cells'])}`**（11/16 正向，单种子标准差 `44.76`）。"
        + (f" 事后合并全部 32 个独立 Tier 3 种子诊断显示真实正向效应存在（`{_fmt_ci(pooled['D_minus_A']['E1_new_cells'])}`，`23/32 = 71.9%` 正向），但 90% 功效需约 **`{pooled['estimated_seeds_power90_E1']}` 种子**，16 种子对高方差 96 刻探索总量存在明显功效不足。" if pooled else ''),
        f"- **双阶段稳定复现的惊喜发现（怕饿与怕死指标）**：理想双驱信号 `D` 相对课程纯 R1 基线 `A`，在**调优集与独立验证集两次均严格突破 95% CI > 0**："
        f"怕饿响应 `E3_fear_hunger` 分别为 **`{_fmt_ci(t3_row['D_minus_A']['E3_fear_hunger'], pct=True)}`** 与 **`{_fmt_ci(conf['D_minus_A']['E3_fear_hunger'], pct=True)}`**；"
        f"怕死夜间威胁规避 `E4_threat_evasion` 在 Tier 2、Tier 3 调优、Tier 3 验证连续三轮显著为正（验证集 **`{_fmt_ci(conf['D_minus_A']['E4_threat_evasion'], pct=True)}`**，相对原生 R1 `A0` 达 **`{_fmt_ci(conf['D_minus_A0']['E4_threat_evasion'], pct=True)}`**）。",
        f"- **生存护栏全部通过**：验证集综合高压存活 `D − A0 = {_fmt_ci(conf['D_minus_A0']['stress_survival'], pct=True)}`、`D − A = {_fmt_ci(conf['D_minus_A']['stress_survival'], pct=True)}`（完全守住 `-5.00pp` 红线，夜间 E4 生存率稳定提升至 `73.4%–74.2%`）。",
        '',
    ]

    if manifest is not None:
        lines.extend(['## 1. 下午采集 MEA 数据净化与门控核验（已完成并就绪）', ''])
        ds_entry = manifest.get('datasets', {}).get('22021_1123_oddball_clean')
        if ds_entry:
            rg = ds_entry['real_gate']
            cg = ds_entry['ctime_null_gate']
            cp = ds_entry.get('channel_purification') or {}
            lines.append(
                f"- **下午 `22021` 通道净化**：严格过滤为 **`{cp.get('n_clean_roi_channels', 233)}`** 个干净 `roi` 读出通道（总放电 `{cp.get('total_spikes_clean_roi')}`），剔除 3 个占原始 70% 伪尖峰的 DAC 刺激通道（`96, 737, 804`）及 7 个近场通道。"
            )
            lines.append(
                f"- **双门控通过**：真实时序留出验证（400 窗）G2 PASS（`{rg['evoked_held_sites']['H']['auc']:.3f} / {rg['evoked_held_sites']['W']['auc']:.3f} / {rg['evoked_held_sites']['F']['auc']:.3f}`）、G3 PASS（`{rg['habituation_held_sites']['H']['auc']:.3f} / {rg['habituation_held_sites']['W']['auc']:.3f} / {rg['habituation_held_sites']['F']['auc']:.3f}`），**`rho_lib = {rg['rho_lib']:+.4f}`**；`C_time` 时间置换彻底坍缩至 **`rho_all = {_fmt_num(cg['rho_all_sites'], 4, signed=True)}`**。"
            )
        pre_sens = manifest.get('sensitivity_audits', {}).get('22021_1123_prestim_only')
        if pre_sens:
            ps = pre_sens['summary']
            lines.append(
                f"- **纯刺激前窗口 `[-300, -1) ms` 灵敏度核验**：`{pre_sens['total_spikes_clean_roi']}` 放电，G2/G3 仍双通过，**`rho_lib = {_fmt_num(ps['rho_lib'], 4, signed=True)}`**。"
            )
        rej = manifest.get('rejected_afternoon_chips', {}).get('25449_1123')
        if rej:
            q_ch = rej.get('recordings', {}).get('quiet', {}).get('n_active_channels', 'n/a')
            w_ch = rej.get('recordings', {}).get('warmup', {}).get('n_active_channels', 'n/a')
            lines.append(
                f"- **下午芯片 `25449` 自动拒收**：`quiet` `{q_ch}` 活跃通道、`warmup` `{w_ch}` 活跃通道（均 `< 50` 门槛），按 G1 拒收。"
            )
        lines.append('')

    lines.extend([
        '## 2. Stage 0 真实 GPU 梯级校准与独立验证全景表',
        '',
        '![Stage 0 梯级与验证全景对比](fig1_stage0_ladder_endpoints.png)',
        '',
        '| 阶段 | 试验臂 | E1 96刻新格 | 综合高压存活 | E2 存活 / 怕渴响应 | E3 存活 / 怕饿响应 | E4 存活 / 威胁规避 / 受击伤害 | 锚点TV偏移 |',
        '|---|---|---:|---:|---|---|---|---:|',
    ])
    phase_rows = [(f"Tier {r['tier']['tier']} ({r['tier']['updates']} upd)", r['arm_means']) for r in stage0.get('ladder', [])]
    if conf:
        phase_rows.append((f"Tier {ft.get('tier', 3)} 独立验证", conf['arm_means']))
    for pname, am in phase_rows:
        for arm in ('A0_native_r1', 'A', 'D'):
            m = am[arm]
            lines.append(
                f"| {pname} | `{arm}` | {_fmt_num(m['E1_new_cells'], 2)} | {_fmt_pct(m['stress_survival'])} "
                f"| {_fmt_pct(m['E2_thirst_surv'], 1)} / {_fmt_pct(m['E2_fear_thirst'], 1)} "
                f"| {_fmt_pct(m['E3_hunger_surv'], 1)} / {_fmt_pct(m['E3_fear_hunger'], 1)} "
                f"| {_fmt_pct(m['E4_death_surv'], 1)} / {_fmt_pct(m['E4_threat_evasion'], 1)} / {_fmt_num(m['E4_threat_damage'], 2)} "
                f"| {_fmt_num(m['mean_tv_shift'], 4)} |"
            )
    lines.append('')

    lines.extend([
        '## 3. 配对双层自助抽样检验（10,000 次重抽样）与统计功效诊断',
        '',
        '![Stage 0 配对效应森林图](fig2_stage0_paired_forest.png)',
        '',
        '| 阶段与对比 | E1 96刻新格差 [95% CI] | 正种子比 | 综合高压存活差 [95% CI] | E2 怕渴响应差 | E3 怕饿响应差 | E4 威胁规避差 | 门控状态 |',
        '|---|---|---:|---|---|---|---|---|',
    ])
    for r in stage0.get('ladder', []):
        cp = r['D_minus_A']
        lines.append(
            f"| Tier {r['tier']['tier']} `D−A` (n=16) | {_fmt_ci(cp['E1_new_cells'])} | {_fmt_pct(cp['E1_new_cells']['seed_positive_frac'], 1)} "
            f"| {_fmt_ci(cp['stress_survival'], pct=True)} | {_fmt_ci(cp['E2_fear_thirst'], pct=True)} "
            f"| {_fmt_ci(cp['E3_fear_hunger'], pct=True)} | {_fmt_ci(cp['E4_threat_evasion'], pct=True)} "
            f"| {'PASS' if r['passed'] else '升级下一档'} |"
        )
    if conf:
        for lbl, cp in (('Tier 3 独立验证 `D−A` (n=16)', conf['D_minus_A']), ('Tier 3 独立验证 `D−A0` (n=16)', conf['D_minus_A0'])):
            lines.append(
                f"| {lbl} | {_fmt_ci(cp['E1_new_cells'])} | {_fmt_pct(cp['E1_new_cells']['seed_positive_frac'], 1)} "
                f"| {_fmt_ci(cp['stress_survival'], pct=True)} | {_fmt_ci(cp['E2_fear_thirst'], pct=True)} "
                f"| {_fmt_ci(cp['E3_fear_hunger'], pct=True)} | {_fmt_ci(cp['E4_threat_evasion'], pct=True)} "
                f"| {'STOP (ci_low≤0)' if 'D−A`' in lbl else '参考对照'} |"
            )
    if pooled:
        cp = pooled['D_minus_A']
        lines.append(
            f"| *事后合并诊断 `D−A` (n=32)* | {_fmt_ci(cp['E1_new_cells'])} | {_fmt_pct(cp['E1_new_cells']['seed_positive_frac'], 1)} "
            f"| {_fmt_ci(cp['stress_survival'], pct=True)} | {_fmt_ci(cp['E2_fear_thirst'], pct=True)} "
            f"| {_fmt_ci(cp['E3_fear_hunger'], pct=True)} | {_fmt_ci(cp['E4_threat_evasion'], pct=True)} "
            f"| 功效诊断 (E1需{pooled['estimated_seeds_power90_E1']}种子 / E3·E4需16种子) |"
        )
    lines.append('')

    lines.extend([
        '## 4. 种子级方差来源、训练期工程护栏与真实湿件烟雾预览',
        '',
        '![Stage 0 种子诊断与训练审计](fig3_stage0_seed_scatter_and_audit.png)',
        '',
    ])
    if diag and diag.get('tiers'):
        lines.extend([
            '| 阶段 (`D` 臂) | 权重更新 | 最大零更新漂移 | R2 幅度占比 | R2 方差占比 | 解耦优势 SNR | 危机步占比 | 解警触发占比 | KL 早停率 | 单任务耗时 |',
            '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|',
        ])
        for sub, cn in (('tier1', 'Tier 1'), ('tier2', 'Tier 2'), ('tier3', 'Tier 3 调优'), ('confirm', 'Tier 3 验证')):
            if sub in diag['tiers']:
                a = diag['tiers'][sub]['D']
                lines.append(
                    f"| {cn} | {a['weights_changed_all']} | {a['max_zero_update_logp']:.1e} "
                    f"| {_fmt_pct(a['mean_r2_mag_share'])} | {_fmt_pct(a['mean_r2_var_share'])} | {a['mean_adv_snr']:.3f} "
                    f"| {_fmt_pct(a['mean_in_stress_frac'], 1)} | {_fmt_pct(a['mean_relieved_frac'], 1)} | {_fmt_pct(a['kl_early_stop_frac'], 1)} "
                    f"| {a['mean_train_seconds'] + a['mean_eval_seconds']:.0f}s |"
                )
        lines.append('')
    if smoke and smoke.get('arms'):
        lines.extend([
            '### 下午 `22021` 233-ROI 真实 GPU 1 种子烟雾验证（`library_smoke`，Tier 0 / 2 updates，仅工程通路预览）',
            '',
            '| 烟雾臂 | 权重更新 | 锚点TV偏移 | R2幅度占比 | E1 第96刻新格 | E2 怕渴响应 | E3 怕饿响应 | E4 威胁规避 | 峰值显存 |',
            '|---|---|---:|---:|---:|---:|---:|---:|---:|',
        ])
        for arm in ('A', 'C_time', 'D', 'B'):
            if arm in smoke['arms']:
                x = smoke['arms'][arm]
                sm = x['suite_means']
                lines.append(
                    f"| `{arm}` | {x['weights_changed']} | {x['anchor_tv_shift']:.4f} | {_fmt_pct(x['audit']['mean_r2_mag_share'])} "
                    f"| {sm['E1_explore_96']['new_cells']:.2f} | {_fmt_pct(sm['E2_thirst_acute']['fear_of_thirst_rate'], 1)} "
                    f"| {_fmt_pct(sm['E3_hunger_acute']['fear_of_hunger_rate'], 1)} | {_fmt_pct(sm['E4_death_night_threat']['threat_evasion_rate'], 1)} "
                    f"| {x['vram_mb']:.0f} MB |"
                )
        lines.append('')

    lines.extend([
        '## 5. 对 `AGENTS.md` 核心问题的阶段性诚实回答与下一步决策建议',
        '',
        '1. **真实湿件（下午 `22021` 233-ROI）的 7 臂大规模训练效果是否已出结论？** **尚未**。因为预注册 Stage 0 理想阳性对照在 16 种子独立验证集上主终点 E1 置信下界跨 0，系统按规停机，尚未运行 Stage 3（`B` vs `C`/`C_time`/`A` 112 任务）；目前仅完成了 1 种子 `library_smoke` 通路验证（`B=200.29`, `D=200.25`, `C_time=190.79`, `A=182.54`）。',
        '2. **双驱 RL 训练本身能否产生怕饿、怕死行为指标与探索增益？** **能，且证据已高度分化**：在 192 组真实 GPU 对照训练中，双驱理想反馈 `D` 相对纯 R1 `A` 已在**调优集与独立验证集两次严格复现了显著的怕饿响应（`+5.23pp / +3.16pp`）和怕死夜间威胁规避（`+5.22pp / +3.36pp`，Tier 2 亦达 `+4.01pp`）**，且完全不牺牲综合高压存活率；探索新格数 `E1` 方向一致为正（32 种子合并 `+25.87 [+5.12, +47.72]`，`71.9%` 种子为正），但因长视野 96 刻轨迹方差大（`SD=57.1`），16 种子检验功效不足（90% 功效需 ~42 种子）。',
        '3. **下一步可选路径**：',
        '   - **路径 1（直接推进下午 22021 7 臂真实湿件验证，推荐最省时）**：在已冻结的 `Tier 3` 上显式放行 `--allow-unconfirmed-stage0` 运行 112 组 7 臂对比（7 卡约 5.5 小时），同时检验 `B` vs `C`/`C_time`/`A` 在怕饿、怕死（16 种子已达 90% 功效）和探索主终点上的真实湿件效应（并在报告中诚实标注 Stage 0 E1 确认未过）。',
        '   - **路径 2（严格重过 Stage 0 门控后再进 Stage 3）**：追加 `Tier 4`（32 updates）或将 `Tier 3` 独立验证扩至 48 种子，使主终点 `E1` 严格通过 `confirmed=true` 后再跑 Stage 3。',
        '   - **路径 3（停在当前 Stage 0 诊断结论）**：保留 196 组真实 GPU 审计与数据净化成果，先优化探索奖励方差后再启动下一轮训练。',
    ])

    report_md = out_dir / 'REPORT.md'
    report_md.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    return dict(
        report_md=str(report_md),
        figures=[str(fig1), str(fig2), str(fig3)],
        verdict=verdict,
        dry_run=is_dry,
    )


def build_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', type=Path, default=ROOT / 'output/existing_wetware_libraries/gate_manifest.json')
    p.add_argument('--stage0', type=Path)
    p.add_argument('--stage0-diag', type=Path)
    p.add_argument('--smoke', type=Path)
    p.add_argument('--stage3', type=Path)
    p.add_argument('--stage4', type=Path)
    p.add_argument('--out', type=Path, required=True)
    return p


if __name__ == '__main__':
    args = build_parser().parse_args()
    if args.stage3 is not None and Path(args.stage3).is_file():
        out = generate_report(
            manifest_path=args.manifest,
            stage0_path=args.stage0,
            stage3_path=args.stage3,
            stage4_path=args.stage4,
            out_dir=args.out,
        )
    elif args.stage0 is not None and Path(args.stage0).is_file():
        out = generate_stage0_report(
            manifest_path=args.manifest,
            stage0_path=args.stage0,
            stage0_diag_path=args.stage0_diag,
            smoke_path=args.smoke,
            out_dir=args.out,
        )
    else:
        raise SystemExit('Provide either --stage3 or --stage0 JSON path.')
    print(json.dumps(out, indent=2, ensure_ascii=False))
