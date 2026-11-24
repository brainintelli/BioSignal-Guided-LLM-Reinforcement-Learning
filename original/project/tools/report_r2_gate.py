"""Run the R2 leak gate and the share checks on archived training decisions. Reads only.

    python tools/report_r2_gate.py [--output DIR]

Two recorded R2 sources are examined:
  22021  sign of a game-outcome rule, every third decision (offline-r2-bounded-005)
  22554  sign of the recorded stimulation phase, every decision (mea_r2_ratio_22554_20261120)
Writes gate.json and report.md. No policy, simulator or device is started.
"""
import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT)] + [str(ROOT / 'standalone' / name) for name in
                               ('mea_live_pair', 'mea_binary_r2', 'mea_state2_r2')]
from agent.rl_contract import gae
from livepair.predictability import outcome
from tools import r2_guard as g

GAME_LABEL = ROOT / 'output/mea_predictability_22021_20261118T093720Z/offline-r2-bounded-005'
PHASE_LABEL = ROOT / 'output/mea_r2_ratio_22554_20261120/runs'
GAMMA, LAMBDA = .99985, .999
REFERENCE_PER_TICK = .009775
FIXED_PER_TICK = .001          # the 18 November form: beta * sign * discounted ticks
SCALE_FRACTION = .10
BATCH = 12
SEED = 2026112301
BETAS = (.02, .05, .10)


def ticks(duration):
    return math.expm1(duration * math.log(GAMMA)) / math.expm1(math.log(GAMMA))


def game_label_rows():
    """22021: one row per training decision of the R1-only arm and of the arm trained with R2."""
    design = json.loads((GAME_LABEL / 'design.json').read_bytes())
    rows = []
    for seed in design['training_seeds']:
        for arm in ('r1', 'ordered'):
            folder = GAME_LABEL / f'models/{seed}/{arm}'
            pins = json.loads((folder / 'verified-transfer.json').read_bytes())
            for batch in range(1, 5):
                path = folder / f'train-{batch}.jsonl'
                raw = path.read_bytes()
                if hashlib.sha256(raw).hexdigest() != pins[path.relative_to(GAME_LABEL).as_posix()]:
                    raise ValueError('Archived batch differs from its recorded hash: ' + str(path))
                for line in raw.splitlines():
                    r = json.loads(line)
                    t, before = r['transition'], r['transition']['before']
                    result = outcome(t)
                    eligible = (r['index'] + 1) % design['feedback_stride'] == 0 and not r['terminal']
                    expected = (1 if result['correct'] else -1) if eligible else 0
                    if r['r2'] != expected:
                        raise ValueError('Recorded R2 differs from the game-outcome rule')
                    rows.append(dict(seed=seed, arm=arm, batch=batch, r1=r['r1_score']['reward'], sign=r['r2'],
                                     contribution=r['contribution'], duration=r['duration'], terminal=r['terminal'],
                                     boundary=r['boundary'], advantage=r['advantage'], verb=r['action']['verb'],
                                     new_cell=result['new_cells'] > 0, correct=result['correct'],
                                     state=[before[k] for k in ('health', 'hunger', 'thirst', 'visited_count', 'tick')]
                                           + [before['inventory']['berries']]))
    return rows, design['training_seeds']


def phase_label_rows():
    """22554: one row per training decision of the four conditions."""
    rows, seeds = [], None
    for condition in ('off-ordered', 'off-shuffled', 'cap02-ordered', 'cap02-shuffled'):
        folders = sorted((PHASE_LABEL / condition).iterdir())
        seeds = [int(f.name) for f in folders]
        for folder in folders:
            trace = json.loads((folder / 'training-trace.json').read_bytes())
            updates = json.loads((folder / 'updates.json').read_bytes())
            for i, r in enumerate(trace):
                if r['r2_event_sign'] != (1 if r['recorded_phase'] == 'pre' else -1):
                    raise ValueError('Recorded sign differs from the recorded phase')
                rows.append(dict(seed=int(folder.name), arm=condition, batch=i // BATCH + 1, r1=r['r1'],
                                 sign=r['r2_event_sign'], contribution=r['r2_contribution'], duration=r['duration'],
                                 terminal=r['terminal'], boundary=r['terminal'] or i % BATCH == BATCH - 1,
                                 advantage_std=updates[i // BATCH]['advantage_std'], verb=r['action']['verb'],
                                 decision=r['decision'], state=[r['value'], r['decision']]))
    return rows, seeds


def column(rows, key):
    return np.asarray([r[key] for r in rows])


def one_hot(sign):
    return np.column_stack([sign == 1, sign == -1]).astype(float)


def gates(rows, seeds, labels):
    """Leak gate and the input-controlled check for each label, on all decisions and on labelled ones.

    The input is what was recorded of the game state before the action: reserves, explored cells,
    clock and inventory for 22021; the critic's value and the decision number for 22554.
    """
    sign, group = column(rows, 'sign'), column(rows, 'seed')
    held_out = seeds[len(seeds) // 2:]
    inputs = np.asarray([r['state'] for r in rows], float)
    result = {}
    for name, label in labels.items():
        entry = dict(all_decisions=g.leak_gate(one_hot(sign), label, group, held_out, SEED),
                     input_controlled=g.partial_correlation(sign, label, inputs, SEED))
        if (sign == 0).any():
            keep = sign != 0
            entry['decisions_with_feedback'] = g.leak_gate(one_hot(sign[keep]), label[keep], group[keep], held_out, SEED)
        result[name] = entry
    return result


def batches(rows):
    keyed = {}
    for r in rows:
        keyed.setdefault((r['arm'], r['seed'], r['batch']), []).append(r)
    return [keyed[k] for k in sorted(keyed)]


def forms(batch, scale):
    """Every way R2 has been or could be added, on one recorded batch. Returns name -> per-decision R2."""
    r1, sign = column(batch, 'r1'), column(batch, 'sign').astype(float)
    d = np.asarray([ticks(r['duration']) for r in batch])
    bounded = sign * SCALE_FRACTION * np.minimum(np.abs(r1), REFERENCE_PER_TICK * d)
    out = {'raw_sign': sign, 'fixed_per_tick': FIXED_PER_TICK * sign * d,
           'cap_2pct': .20 * bounded, 'cap_5pct': .50 * bounded}
    for beta in BETAS:
        out['relative_beta_%.2f' % beta] = g.shape(r1, sign, scale, beta, eligible=sign != 0)
    return out


def advantage_change(batch, r2):
    """GAE is linear in the reward, so the change in advantage needs no value estimate."""
    zeros = [0.] * len(batch)
    change, _ = gae(list(r2), zeros, zeros, [r['duration'] for r in batch], [r['terminal'] for r in batch],
                    [r['boundary'] for r in batch], GAMMA, LAMBDA)
    baseline = batch[0]['advantage_std'] if 'advantage_std' in batch[0] else float(np.std([r['advantage'] for r in batch]))
    return float(np.std(change) / baseline)


def proportions(rows):
    """Shares per form on the trajectories that were collected without R2 in the reward."""
    pooled, per_batch, advantage, scale, previous = {}, {}, {}, None, None
    for batch in batches(rows):
        run = (batch[0]['arm'], batch[0]['seed'])
        scale = g.track_scale(scale if run == previous else None, column(batch, 'r1'))
        previous = run
        for name, r2 in forms(batch, scale).items():
            pooled.setdefault(name, []).append((column(batch, 'r1'), r2))
            per_batch.setdefault(name, []).append(g.shares(column(batch, 'r1'), r2))
            advantage.setdefault(name, []).append(advantage_change(batch, r2))
    result = {}
    for name, parts in pooled.items():
        share = g.shares(np.concatenate([p[0] for p in parts]), np.concatenate([p[1] for p in parts]))
        level, flags = g.verdict(share)
        result[name] = share | dict(level=level, flags=flags,
            worst_batch_magnitude=max(b['magnitude'] for b in per_batch[name]),
            worst_batch_variance=max(b['variance'] for b in per_batch[name]),
            advantage_change_over_std_median=float(np.median(advantage[name])),
            advantage_change_over_std_max=float(np.max(advantage[name])))
    return result


def as_trained(rows):
    share = g.shares(column(rows, 'r1'), column(rows, 'contribution'))
    level, flags = g.verdict(share)
    return share | dict(level=level, flags=flags)


def move_rate(rows, seeds):
    """How often the model moved on decisions that carry each sign, with R2 switched off."""
    sign, moved, seed = column(rows, 'sign'), column(rows, 'verb') == 'move', column(rows, 'seed')
    gaps = [moved[(seed == s) & (sign == -1)].mean() - moved[(seed == s) & (sign == 1)].mean() for s in seeds]
    return dict(positive_sign=float(moved[sign == 1].mean()), negative_sign=float(moved[sign == -1].mean()),
                seeds_moving_more_on_negative=int(sum(x > 0 for x in gaps)), seeds=len(seeds))


def r1_scale_by_update(rows):
    return {str(b): float(np.median([abs(r['r1']) / ticks(r['duration']) for r in rows if r['batch'] == b]))
            for b in range(1, 5)}


def analyse():
    game, game_seeds = game_label_rows()
    phase, phase_seeds = phase_label_rows()
    game_r1 = [r for r in game if r['arm'] == 'r1' and not r['terminal']]
    phase_off = [r for r in phase if r['arm'].startswith('off') and not r['terminal']]
    per_tick = lambda rows: np.asarray([r['r1'] / ticks(r['duration']) for r in rows])
    # "Below" because most 22021 decisions sit exactly on the median, the reward of an uneventful tick.
    below = lambda rows: per_tick(rows) < np.median(per_tick(rows))
    parity = column(phase_off, 'decision') % 2
    return dict(
        constants={k: getattr(g, k) for k in ('LEAK_MARGIN', 'LEAK_P_MAX', 'SHUFFLES', 'PARTIAL_P_MAX', 'BETA_DEFAULT', 'BETA_MAX',
                   'KAPPA', 'DELTA_FRAC', 'SHARE_TARGET', 'SHARE_REDLINE', 'REDUNDANT_RHO', 'NOISE_RHO', 'EMA_DECAY')},
        game_label=dict(
            source='22021 offline-r2-bounded-005', decisions=len(game_r1), feedback=int((column(game_r1, 'sign') != 0).sum()),
            recorded_sign_equals_game_rule=True,
            gate=gates(game_r1, game_seeds, dict(new_cell=column(game_r1, 'new_cell'), safe_new_cell=column(game_r1, 'correct'),
                                                 r1_below_median=below(game_r1))),
            proportions=proportions([r for r in game if r['arm'] == 'r1']),
            as_trained_cap_5pct=as_trained([r for r in game if r['arm'] == 'ordered']),
            median_abs_r1_per_tick_by_update=r1_scale_by_update(game)),
        phase_label=dict(
            source='22554 mea_r2_ratio_22554_20261120', decisions=len(phase_off),
            sign_equals_decision_parity=bool(np.all((column(phase_off, 'sign') == 1) == (parity == 1))),
            move_rate=move_rate(phase_off, phase_seeds),
            gate=gates(phase_off, phase_seeds, dict(move=column(phase_off, 'verb') == 'move',
                                                    multi_tick=column(phase_off, 'duration') >= 2,
                                                    r1_below_median=below(phase_off))),
            proportions=proportions([r for r in phase if r['arm'].startswith('off')]),
            as_trained_cap_2pct=as_trained([r for r in phase if r['arm'].startswith('cap02')]),
            median_abs_r1_per_tick_by_update=r1_scale_by_update(phase)))


def fmt(value, digits=3):
    return 'n/a' if value is None else ('%.' + str(digits) + 'f') % value


LABELS = dict(new_cell='这一步发现新格', safe_new_cell='发现新格且三项储备不低于 85', r1_below_median='每刻 R1 低于中位数',
              move='动作是移动', multi_tick='动作历时不少于 2 刻')


def leaked_labels(section):
    return [LABELS[name] for name, entry in section['gate'].items()
            if any(entry[scope]['leaked'] for scope in ('all_decisions', 'decisions_with_feedback') if scope in entry)]


def gate_verdict(section):
    found = leaked_labels(section)
    return '判定：**泄漏**（%s）。按规则这个 R2 不得进入奖励。' % '；'.join(found) if found else '判定：通过泄漏门。'


def gate_table(section):
    lines = ['| 标签 | 范围 | AUC_real | AUC_shuf | 差（置换 p） | 泄漏门 | 扣除行动前状态后 r（p） | 状态之外仍相关 |',
             '|---|---|---:|---:|---:|---|---:|---|']
    word = {True: '泄漏', False: '通过', None: '无法判定'}
    for label, entry in section['gate'].items():
        partial = entry['input_controlled']
        tail = '%s（%s）' % (fmt(partial['r']), fmt(partial['p'], 4))
        for scope, title in (('all_decisions', '全部决策'), ('decisions_with_feedback', '有反馈的决策')):
            if scope in entry:
                e = entry[scope]
                lines.append('| %s | %s | %s | %s | %s（%s） | %s | %s | %s |' % (LABELS[label], title, fmt(e['auc_real']),
                             fmt(e['auc_shuffled']), fmt(e['margin']), fmt(e.get('p'), 4), word[e['leaked']], tail,
                             {True: '是', False: '否', None: '无法判定'}[partial['leaked']]))
    return lines


def share_table(section, trained_key, trained_title):
    names = {'raw_sign': '直接加 ±1（本工作区没有这样训练的记录）', 'fixed_per_tick': '0.001 × 符号 × 游戏刻（11 月 18 日）',
             'cap_2pct': '单步上限 2%（11 月 20 日）', 'cap_5pct': '单步上限 5%（11 月 19 日）'}
    names.update({'relative_beta_%.2f' % b: '相对尺度加逐样本封顶，β=%.2f' % b for b in BETAS})
    level = {'ok': '达标', 'above_target': '超目标', 'redline': '越红线'}
    flag = {'redundant': '与 R1 冗余', 'noise': '与 R1 无关的噪声'}
    lines = ['| R2 的加法 | 幅度占比 | 方差占比 | ρ(R1,R2) | 最大单步 R2/R1 | 压过 R1 的步数 | 优势改变量/优势标准差（中位） | 判定 |',
             '|---|---:|---:|---:|---:|---:|---:|---|']
    for key, title in names.items():
        s = section['proportions'][key]
        lines.append('| %s | %s | %s | %s | %s | %d / %d | %s | %s |' % (title, fmt(s['magnitude'], 4), fmt(s['variance'], 4),
                     fmt(s['rho']), fmt(s['max_step_ratio']), s['larger_than_r1'], s['steps'],
                     fmt(s['advantage_change_over_std_median'], 4), level[s['level']] + ''.join('，' + flag[f] for f in s['flags'])))
    s = section[trained_key]
    lines.append('| %s | %s | %s | %s | %s | %d / %d | 未重算 | %s |' % (trained_title, fmt(s['magnitude'], 4), fmt(s['variance'], 4),
                 fmt(s['rho']), fmt(s['max_step_ratio']), s['larger_than_r1'], s['steps'], level[s['level']]))
    return lines


def report(result):
    a, b = result['game_label'], result['phase_label']
    c = result['constants']
    lines = ['# R2 泄漏门与占比核查', '',
             '只读分析：重算已归档的训练决策，没有训练模型，没有接触设备。数字全部来自 `gate.json`，阈值是 `tools/r2_guard.py` 里的具名常量。', '',
             '判定规则：只用 R2 预测标签，留出一半训练种子，AUC_real 减 AUC_shuf 大于 %.2f 且置换 p ≤ %.2f 判为泄漏；'
             '把记录到的行动前游戏状态线性扣除后 R2 与标签的相关，置换 p ≤ %.2f 判为状态之外仍相关。' % (
                 c['LEAK_MARGIN'], c['LEAK_P_MAX'], c['PARTIAL_P_MAX']),
             '占比目标 < %.2f，红线 %.2f。' % (c['SHARE_TARGET'], c['SHARE_REDLINE']), '',
             '## 22021：R2 的正负号来自游戏结果规则', '',
             '%d 个纯 R1 训练决策，其中 %d 个有反馈（每三个决策一次）。逐步核对：记录的正负号与“发现新格且生命、水分、饱食都不低于 85”这条规则完全一致。' % (a['decisions'], a['feedback']), '',
             gate_verdict(a), '', *gate_table(a), '', *share_table(a, 'as_trained_cap_5pct', '实际训练记录（上限 5% 组）'), '',
             '## 22554：R2 的正负号来自记录的刺激时相', '',
             '%d 个关闭 R2 的训练决策，每个决策都有正负号。正负号与决策序号的奇偶完全一致：%s。' % (b['decisions'], '是' if b['sign_equals_decision_parity'] else '否'),
             '模型在正号决策上有 %.1f%% 选择移动，在负号决策上有 %.1f%%；%d/%d 个训练种子在负号决策上移动得更多。' % (
                 100 * b['move_rate']['positive_sign'], 100 * b['move_rate']['negative_sign'],
                 b['move_rate']['seeds_moving_more_on_negative'], b['move_rate']['seeds']), '',
             gate_verdict(b), '', *gate_table(b), '', *share_table(b, 'as_trained_cap_2pct', '实际训练记录（上限 2% 组）'), '',
             '## R1 的尺度是否随训练下降', '',
             '| 数据 | 第 1 次更新 | 第 2 次 | 第 3 次 | 第 4 次 |', '|---|---:|---:|---:|---:|',
             '| 22021 每折扣游戏刻 \\|R1\\| 中位数 | ' + ' | '.join(fmt(a['median_abs_r1_per_tick_by_update'][str(i)], 6) for i in range(1, 5)) + ' |',
             '| 22554 每折扣游戏刻 \\|R1\\| 中位数 | ' + ' | '.join(fmt(b['median_abs_r1_per_tick_by_update'][str(i)], 6) for i in range(1, 5)) + ' |', '']
    return '\n'.join(lines)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--output', type=Path, default=ROOT / 'output/r2_gate_20261123')
    out = parser.parse_args().output
    out.mkdir(parents=True, exist_ok=False)
    result = analyse()
    (out / 'gate.json').write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    (out / 'report.md').write_text(report(result), encoding='utf-8')
    print(report(result))
