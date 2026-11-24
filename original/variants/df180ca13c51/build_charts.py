"""Plot the sealed 28080 experiment; do not retrain or alter registered statistics."""
from pathlib import Path
from datetime import datetime, timezone
from statistics import mean
import hashlib
import json
import math
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'chart_dependencies'))
sys.path.insert(0, str(ROOT / 'report_frozen_project'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D
import numpy as np
from agent.mea_long_ablation import analyze, PIN

OUT = ROOT / 'charts'
CAMPAIGN = ROOT / 'delivery/experiment01'
ARMS = ('none', 'real', 'shuffled')
LABELS = {'none': '无 state2', 'real': '真实 state2', 'shuffled': '打乱 state2'}
COLORS = {'none': '#64748B', 'real': '#007F86', 'shuffled': '#C77920'}
INK, MUTED, GRID = '#162C3C', '#607482', '#E4EBEF'
SOURCE_HASHES = {}


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def verified_path(name):
    path = (CAMPAIGN / name).resolve()
    assert path.is_relative_to(CAMPAIGN.resolve())
    expected = manifest['files'][name]
    assert path.stat().st_size == expected['size'] and sha(path) == expected['sha256'], name
    SOURCE_HASHES[name] = expected['sha256']
    return path


def axis_style(ax, horizontal=False):
    ax.set_axisbelow(True)
    ax.grid(axis='x' if horizontal else 'y', color=GRID, lw=.8)
    ax.spines[['top', 'right']].set_visible(False)
    for side in ('left', 'bottom'):
        ax.spines[side].set_color('#C8D5DE')
    ax.tick_params(colors=MUTED, length=0, pad=9)


def heading(fig, title, subtitle):
    fig.text(.055, .955, title, fontsize=22, weight='bold', color=INK, va='top')
    fig.text(.055, .897, subtitle, fontsize=12, color=MUTED, va='top')


def footer(fig, text):
    fig.text(.055, .035, text, fontsize=10.5, color=MUTED, va='bottom')


def save(fig, stem):
    for suffix in ('png', 'svg'):
        fig.savefig(OUT / f'{stem}.{suffix}', dpi=170, facecolor='white', bbox_inches='tight')
    plt.close(fig)


def phase_values(arm, phase, key='discounted_r1'):
    return [indexed[b, arm]['metrics'][phase][key] for b in range(8)]


def score_figure():
    fig, axes = plt.subplots(1, 3, figsize=(15, 6.5), sharey=True)
    fig.subplots_adjust(left=.07, right=.97, bottom=.19, top=.80, wspace=.18)
    heading(fig, '训练前后 R1：真实组均值从 7.645 变为 7.591',
            '每组 8 个配对数字种子 · 相同初始权重 · 每次训练 1,024 决策 · 留存评估 600 决策')
    for ax, arm in zip(axes, ARMS):
        pre, post = phase_values(arm, 'pre'), phase_values(arm, 'post')
        for x, y in zip(pre, post):
            ax.plot([0, 1], [x, y], color=COLORS[arm], alpha=.24, lw=1.4, marker='o', ms=5)
        av = [mean(pre), mean(post)]
        ax.plot([0, 1], av, color=COLORS[arm], marker='D', ms=9, lw=3.5, zorder=5)
        for x, y in enumerate(av):
            ax.annotate(f'{y:.3f}', (x, y), xytext=(0, -24), textcoords='offset points',
                        ha='center', weight='bold', color=COLORS[arm], fontsize=13,
                        bbox=dict(fc='white', ec='none', alpha=.85, pad=1.5))
        ax.set_title(f'{LABELS[arm]}  |  平均变化 {av[1]-av[0]:+.3f}', fontsize=14,
                     color=COLORS[arm], weight='bold', pad=18)
        ax.set_xticks([0, 1], ['前测：初始模型', '后测：训练后模型'])
        ax.set_xlim(-.22, 1.22)
        ax.set_ylim(6.9, 8.6)
        ax.set_yticks(np.arange(7., 8.6, .25))
        axis_style(ax)
    axes[0].set_ylabel('留存评估的折扣累计 R1（越高越好）', color=INK)
    fig.legend(handles=[Line2D([], [], color=MUTED, alpha=.4, marker='o', lw=1, label='细线：单个种子'),
                        Line2D([], [], color=INK, marker='D', lw=3, label='粗线：8 个种子的均值')],
               loc='lower center', bbox_to_anchor=(.5, .075), ncol=2, frameon=False, fontsize=11)
    footer(fig, '三组均保留 state1；纵轴相同，按数据范围显示。细线展示全部运行；前后只读评估分别加载初始和最终权重。')
    save(fig, '01_pre_post_scores')


def effect_figure():
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 7))
    fig.subplots_adjust(left=.07, right=.97, bottom=.18, top=.78, wspace=.14)
    heading(fig, '真实 state2：后测高于无 state2，未显示优于打乱组',
            '点为配对均值差，横线为 95% 配对 bootstrap 区间；区间跨 0 表示方向仍不明确')
    for ax, phase, title in zip(axes, ('post', 'gain'), ('后测分数差（主要比较）', '前后增益差（辅助比较）')):
        ax.set_title(title, fontsize=16, weight='bold', color=INK, pad=20)
        ax.axvspan(0, .72, color='#F3FAF8', zorder=0)
        ax.axvline(0, color='#93A5B1', ls='--', lw=1.5)
        for y, control in ((1, 'none'), (0, 'shuffled')):
            c = result['contrasts']['real_minus_' + control][phase]
            m, (lo, hi) = c['mean_difference'], c['bootstrap_95']
            ax.errorbar(m, y, xerr=np.array([[m-lo], [hi-m]]), fmt='D', ms=9,
                        color=COLORS['real'], lw=3, capsize=6, zorder=5)
            ax.text(-.39, y+.28, f'真实 − {LABELS[control]}', weight='bold', color=INK, fontsize=13)
            ax.text(-.39, y-.27, f'{m:+.4f}  [{lo:+.4f}, {hi:+.4f}]', fontsize=12, color=MUTED)
            if phase == 'post':
                ax.text(.70, y+.28, f"校正 p = {c['p_bonferroni_two_contrasts']:.4f}",
                        ha='right', color=MUTED, fontsize=11)
        ax.set_xlim(-.40, .72)
        ax.set_ylim(-.5, 1.58)
        ax.set_yticks([])
        ax.set_xticks([-.4, -.2, 0, .2, .4, .6])
        ax.set_xlabel('← 真实组更低                 真实组更高 →', color=MUTED, labelpad=16)
        axis_style(ax, horizontal=True)
    footer(fig, '8 个配对数字种子；20,000 次配对 bootstrap。主要检验为双侧符号翻转，两比较 Bonferroni 修正。辅助比较用于解释训练增益。')
    save(fig, '02_paired_effects')


def decomposition_figure():
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 6.7), sharey=True)
    fig.subplots_adjust(left=.08, right=.97, bottom=.21, top=.78, wspace=.17)
    heading(fig, 'R1 差异来自哪里：拆开时长项与资源状态项',
            '总 R1 差值 = 存活时长项差值 + 资源势函数项差值；以下是描述性分解，不新增显著性检验')
    for ax, control in zip(axes, ('none', 'shuffled')):
        vals = [mean(decomp['real'][k]) - mean(decomp[control][k]) for k in ('survival', 'potential')]
        vals.append(result['contrasts']['real_minus_' + control]['post']['mean_difference'])
        assert math.isclose(vals[0]+vals[1], vals[2], abs_tol=1e-10)
        bars = ax.bar(range(3), vals, width=.52, color=['#4F7FA8', '#D09B45', COLORS['real']], zorder=3)
        for bar, value in zip(bars, vals):
            ax.annotate(f'{value:+.4f}', (bar.get_x()+bar.get_width()/2, value),
                        xytext=(0, 9 if value>=0 else -10), textcoords='offset points',
                        ha='center', va='bottom' if value>=0 else 'top', fontsize=14, weight='bold', color=INK)
        ax.axhline(0, color='#93A5B1', lw=1.2)
        ax.set_title(f'真实 − {LABELS[control]}', fontsize=16, weight='bold', color=INK, pad=18)
        ax.set_xticks(range(3), ['存活时长项\n差值', '资源势函数项\n差值', '总 R1\n差值'])
        ax.set_ylim(-.17, .38)
        axis_style(ax)
    axes[0].set_ylabel('后测平均折扣 R1 的差值', color=INK)
    footer(fig, '固定 600 决策可对应不同游戏 tick。真实组相对无 state2 的 R1 优势主要来自时长项，不能直接解释为完整生存期或资源管理改善。')
    save(fig, '03_reward_decomposition')


def outcomes_figure():
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.subplots_adjust(left=.08, right=.97, bottom=.13, top=.82, hspace=.44, wspace=.22)
    heading(fig, '游戏内的具体表现：时长、生命与资源',
            '三组后测均为 8/8 存活至 600 决策截止；灰线连接同一个数字种子，圆点为单次运行，菱形为均值')
    fields = [('elapsed_ticks', '实际经过的游戏 tick', (740, 960)),
              ('final_health', '最终生命值', (0, 110)),
              ('final_hunger', '最终饱食度（高值代表储备较多）', (0, 110)),
              ('final_thirst', '最终水分（高值代表储备较多）', (0, 110))]
    offsets = np.linspace(-.07, .07, 8)
    for ax, (key, title, bounds) in zip(axes.flat, fields):
        for b in range(8):
            vals = [indexed[b, arm]['metrics']['post'][key] for arm in ARMS]
            ax.plot(np.arange(3)+offsets[b], vals, color='#ADBCC7', lw=1, alpha=.5, zorder=1)
        ticks = []
        for x, arm in enumerate(ARMS):
            values = phase_values(arm, 'post', key)
            ax.scatter(np.full(8, x)+offsets, values, color=COLORS[arm], s=28, alpha=.62, zorder=3)
            ax.scatter([x], [mean(values)], color=COLORS[arm], s=110, marker='D', ec='white', lw=1.2, zorder=5)
            precision = 1 if key == 'elapsed_ticks' else 2
            ticks.append(f'{LABELS[arm]}\n均值 {mean(values):.{precision}f}')
        ax.set_xticks(range(3), ticks)
        ax.set_xlim(-.38, 2.38)
        ax.set_ylim(*bounds)
        ax.set_title(title, fontsize=14, weight='bold', color=INK, pad=13)
        axis_style(ax)
    footer(fig, '这些为辅助描述性指标，未对其新增显著性筛选。全部运行存活至截止，本次观察窗口不能区分完整生存期；8 个数字种子不是 8 个生物样本。')
    save(fig, '04_game_outcomes')


def write_report():
    plots = [('01_pre_post_scores', '训练前后：真实组没有平均分上升',
              '真实组平均 R1 从 7.6447 变为 7.5907，变化为 −0.0540；打乱组变化为 +0.0362。细线显示各个种子的结果，均值变化本身不等于显著训练收益。'),
             ('02_paired_effects', '主要比较与训练增益',
              '真实−无 state2 后测差为 +0.3026，95% 区间 [+0.0963, +0.5878]，两比较修正 p=0.046875；真实−打乱差为 −0.0156，区间 [−0.1624, +0.1320]，修正 p=1。两项前后增益差区间也都跨 0，整体阳性判据未达到。'),
             ('03_reward_decomposition', 'R1 优势主要来自时长项',
              '真实−无 state2 的总 R1 差 +0.3026，由时长项 +0.3103 和势函数项 −0.0077 构成。固定决策次数下，不同动作会经历不同游戏时间，因此这项分数差不能直接称为生存能力提升。'),
             ('04_game_outcomes', '具体游戏指标与结果分布',
              '三组全部存活到 600 决策截止；真实组平均生命值较高，但饱食度和水分没有同时优于无 state2。此处只作描述，不以这些事后观察替代预先确定的主要指标。')]
    lines = ['# 芯片 28080：离线对照图表', '',
             '**结论：真实 state2 的后测 R1 高于无 state2，但未优于打乱 state2；未获得真实信号时间顺序带来额外收益的证据。**', '',
             '数据来自已经封存核验的 24 次运行（3 组 × 8 个配对数字种子），每次训练 1,024 决策、留存前后测各 600 决策。未新增训练，未修改检验或筛选种子。全程无刺激、R2 只记录、α=1、实时训练关闭。', '']
    for stem, title, description in plots:
        lines += [f'## {title}', '', description, '', f'![{title}]({(OUT/(stem+".png")).as_posix()})', '',
                  f'[下载 PNG]({(OUT/(stem+".png")).as_posix()}) · [下载 SVG 矢量图]({(OUT/(stem+".svg")).as_posix()})', '']
    lines += ['## 解释范围', '',
              '所有区间和检验沿用原登记分析。8 个数字种子反映同一次生物记录下的软件随机性，不是 8 次独立类器官实验；预录信号与当前游戏动作没有因果对应。未检出真实与打乱的差异，也不等于证明两组完全等效。', '',
              '固定决策预算与优化步骤相同；含神经观测的输入较长，因此不声称 token、FLOPs 或运行时间完全相同。', '',
              f'[原始完整报告]({(ROOT/"ABLATION_28080_report.md").as_posix()}) · [统一统计 JSON]({(CAMPAIGN/"analysis.json").as_posix()}) · [绘图数据]({(OUT/"chart_data.json").as_posix()}) · [绘图校验]({(OUT/"chart_verification.json").as_posix()})', '']
    (OUT/'CHART_REPORT.md').write_text('\n'.join(lines), encoding='utf-8')


if __name__ == '__main__':
    OUT.mkdir(exist_ok=True)
    archive_receipt = read(ROOT/'archive_receipt.json')
    assert read(ROOT/'local_verification.json')['passed'] is True
    assert read(ROOT/'transport_verification.json')['passed'] is True
    manifest_path = ROOT/'delivery/accepted_campaign_manifest.json'
    assert sha(manifest_path) == archive_receipt['accepted_manifest_sha256']
    manifest = read(manifest_path)
    assert manifest['source_manifest_sha256'] == PIN and manifest['accepted_cells'] == 24
    result = read(verified_path('analysis.json'))
    assert result['status'] == 'complete' and result['positive_evidence'] is False
    indexed = {(c['block'], c['arm']): c for c in result['cells']}
    assert len(indexed) == 24 and set(indexed) == {(b, a) for b in range(8) for a in ARMS}
    cells = [dict(block=b, arm=a, **{p:indexed[b,a]['metrics'][p]['discounted_r1'] for p in ('pre','post')})
             for b in range(8) for a in ARMS]
    for key, value in analyze(cells).items():
        assert result[key] == value, key
    decomp = {a:dict(survival=[], potential=[]) for a in ARMS}
    for b in range(8):
        for arm in ARMS:
            c = indexed[b, arm]
            assert c['alpha'] == 1 and not c['r2_enabled'] and not c['stimulation'] and not c['realtime_training']
            assert all(c['metrics'][p]['decisions'] == 600 for p in ('pre','post'))
            assert not c['metrics']['post']['terminated']
            elapsed = count = 0
            sums = dict(survival=0., potential=0.)
            with verified_path(f'b{b:02d}_{arm}/post/rollout.jsonl').open(encoding='utf-8') as stream:
                for line in stream:
                    row = json.loads(line)
                    assert row['reward'] == row['r1_score']['reward'] and row['reward_record']['alpha'] == 1
                    for key in sums:
                        sums[key] += .99985**elapsed * row['r1_score']['components'][key]
                    elapsed += row['duration']
                    count += 1
            assert count == 600 and elapsed == c['metrics']['post']['elapsed_ticks']
            assert math.isclose(sum(sums.values()), c['metrics']['post']['discounted_r1'], abs_tol=1e-10)
            for key, value in sums.items():
                decomp[arm][key].append(value)
    font_manager.fontManager.addfont('C:/Windows/Fonts/msyh.ttc')
    font_manager.fontManager.addfont('C:/Windows/Fonts/msyhbd.ttc')
    plt.rcParams.update({'font.family': 'Microsoft YaHei', 'font.size': 12,
                         'axes.unicode_minus': False, 'svg.fonttype': 'path',
                         'text.color': INK, 'axes.labelcolor': INK})
    data = dict(source_manifest_sha256=PIN, analysis_sha256=SOURCE_HASHES['analysis.json'],
                seed_cells=cells, all_metrics={f'b{b:02d}_{a}':indexed[b,a]['metrics'] for b,a in indexed},
                contrasts=result['contrasts'], decomposition_by_seed=decomp,
                decomposition_means={a:{k:mean(v) for k,v in decomp[a].items()} for a in ARMS})
    (OUT/'chart_data.json').write_text(json.dumps(data, indent=2)+'\n', encoding='utf-8')
    score_figure(); effect_figure(); decomposition_figure(); outcomes_figure(); write_report()
    outputs = {p.name:sha(p) for p in sorted(OUT.iterdir()) if p.is_file() and p.name != 'chart_verification.json'}
    verified = dict(passed=True, verified_utc=datetime.now(timezone.utc).isoformat(),
                    biological_recordings=1, cells=24, paired_seed_blocks=8, post_rows_recomputed=14400,
                    original_analysis_recomputed=True, reward_decomposition_verified=True,
                    sources=SOURCE_HASHES, outputs=outputs, script_sha256=sha(Path(__file__)),
                    matplotlib_version=matplotlib.__version__, numpy_version=np.__version__,
                    no_new_training=True, no_new_significance_tests=True)
    (OUT/'chart_verification.json').write_text(json.dumps(verified, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(dict(passed=True, cells=24, source_files=len(SOURCE_HASHES), outputs=list(outputs))))
