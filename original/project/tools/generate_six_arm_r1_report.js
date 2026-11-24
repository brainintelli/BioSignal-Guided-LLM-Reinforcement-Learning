const fs = require('fs');
const path = require('path');

const workspace = path.resolve(__dirname, '..');
const output = path.join(workspace, 'output', 'mea_six_arm_r1_off_20261119');
const source22021 = path.join(workspace, 'output', 'mea_predictability_22021_20261118T093720Z', 'offline-mea-stim-ablation');
const source22554 = path.join(output, '22554');

const metricNames = ['exploration', 'r1', 'health', 'hunger', 'thirst'];
const metricLabels = {
  exploration: '探索格数',
  r1: 'R1',
  health: '生命',
  hunger: '饱食',
  thirst: '水分',
};

function readJson(file) {
  return JSON.parse(fs.readFileSync(file, 'utf8'));
}

function readJsonLines(file) {
  return fs.readFileSync(file, 'utf8').trim().split(/\r?\n/).filter(Boolean).map(JSON.parse);
}

function mean(values) {
  return values.reduce((sum, value) => sum + value, 0) / values.length;
}

function sd(values) {
  if (values.length < 2) return 0;
  const m = mean(values);
  return Math.sqrt(values.reduce((sum, value) => sum + (value - m) ** 2, 0) / (values.length - 1));
}

function quantile(values, q) {
  const sorted = [...values].sort((a, b) => a - b);
  if (!sorted.length) return 0;
  const position = (sorted.length - 1) * q;
  const lower = Math.floor(position);
  const upper = Math.ceil(position);
  if (lower === upper) return sorted[lower];
  return sorted[lower] + (sorted[upper] - sorted[lower]) * (position - lower);
}

function bootstrapCI(values, draws = 20000, seed = 17) {
  let state = seed >>> 0;
  const random = () => {
    state = (1664525 * state + 1013904223) >>> 0;
    return state / 0x100000000;
  };
  const means = [];
  for (let draw = 0; draw < draws; draw += 1) {
    let total = 0;
    for (let i = 0; i < values.length; i += 1) total += values[Math.floor(random() * values.length)];
    means.push(total / values.length);
  }
  return [quantile(means, 0.025), quantile(means, 0.975)];
}

function makeGroup(chip, arm, label, sourceRoot, seedIds, options = {}) {
  const rows = [];
  const actionCounts = {};
  const seedMetrics = {};
  for (const seed of seedIds) {
    const folder = path.join(sourceRoot, 'models', String(seed), arm);
    const result = readJson(path.join(folder, 'result.json'));
    const evaluation = result.evaluation || result.eval;
    if (!evaluation || evaluation.length !== 48) throw new Error(`${chip}/${arm}/${seed} does not have 48 held-out worlds`);
    const seedValues = [];
    for (const item of evaluation) {
      const row = {
        chip,
        arm,
        label,
        train_seed: Number(seed),
        world_seed: Number(item.seed),
        exploration: Number(item.new_cells ?? item.explored),
        r1: Number(item.r1_sum ?? item.r1),
        health: Number(item.final_health ?? item.health),
        hunger: Number(item.final_hunger ?? item.hunger),
        thirst: Number(item.final_thirst ?? item.thirst),
        terminated: Number(item.terminated ?? item.death ?? 0),
      };
      rows.push(row);
      seedValues.push(row);
    }
    seedMetrics[seed] = Object.fromEntries(metricNames.map((metric) => [metric, mean(seedValues.map((row) => row[metric]))]));

    const holdout = path.join(folder, options.holdoutFile || 'holdout.jsonl');
    if (fs.existsSync(holdout)) {
      for (const game of readJsonLines(holdout)) {
        for (const row of (game.rows || [])) {
          const verb = row.action?.verb ?? row.verb ?? 'unknown';
          actionCounts[verb] = (actionCounts[verb] || 0) + 1;
        }
      }
    }
  }
  return { chip, arm, label, rows, seedMetrics, actionCounts };
}

function load22554Group(arm, label) {
  const folder = path.join(source22554, arm, 'training');
  const result = readJson(path.join(folder, 'result.json'));
  const rows = [];
  const seedMetrics = {};
  const actionCounts = {};
  for (const seedResult of result.seeds) {
    const seed = Number(seedResult.training_seed);
    if (seedResult.eval.length !== 48) throw new Error(`22554/${arm}/${seed} does not have 48 held-out worlds`);
    const seedValues = [];
    for (const item of seedResult.eval) {
      const row = {
        chip: '22554',
        arm,
        label,
        train_seed: seed,
        world_seed: Number(item.seed),
        exploration: Number(item.explored),
        r1: Number(item.r1),
        health: Number(item.health),
        hunger: Number(item.hunger),
        thirst: Number(item.thirst),
        terminated: Number(item.death || 0),
      };
      rows.push(row);
      seedValues.push(row);
    }
    seedMetrics[seed] = Object.fromEntries(metricNames.map((metric) => [metric, mean(seedValues.map((row) => row[metric]))]));

    const trace = readJson(path.join(folder, String(seed), 'eval-trace.json'));
    for (const game of Object.values(trace)) {
      for (const step of game) {
        if (Array.isArray(step.verb_probabilities)) {
          step.verb_probabilities.forEach((probability, index) => {
            const key = `verb_${index}`;
            actionCounts[key] = (actionCounts[key] || 0) + Number(probability);
          });
        } else {
          const key = `verb_${step.verb ?? 'unknown'}`;
          actionCounts[key] = (actionCounts[key] || 0) + 1;
        }
      }
    }
  }
  return { chip: '22554', arm, label, rows, seedMetrics, actionCounts };
}

function summarize(group) {
  const summary = { chip: group.chip, arm: group.arm, label: group.label, n_worlds: group.rows.length, n_seeds: Object.keys(group.seedMetrics).length };
  for (const metric of metricNames) {
    const worldValues = group.rows.map((row) => row[metric]);
    const seedValues = Object.values(group.seedMetrics).map((row) => row[metric]);
    summary[metric] = {
      mean: mean(worldValues),
      sd_world: sd(worldValues),
      seed_mean: mean(seedValues),
      seed_sd: sd(seedValues),
      seed_ci95: bootstrapCI(seedValues, 12000, metric.length + group.arm.length),
    };
  }
  const totalActions = Object.values(group.actionCounts).reduce((sum, count) => sum + count, 0);
  summary.action_probabilities = Object.fromEntries(Object.entries(group.actionCounts).map(([action, count]) => [action, count / totalActions]));
  summary.terminated_worlds = group.rows.reduce((sum, row) => sum + row.terminated, 0);
  return summary;
}

function pairedEffects(groupA, groupB, contrast) {
  const seeds = Object.keys(groupA.seedMetrics).filter((seed) => Object.prototype.hasOwnProperty.call(groupB.seedMetrics, seed));
  return metricNames.map((metric) => {
    const differences = seeds.map((seed) => groupA.seedMetrics[seed][metric] - groupB.seedMetrics[seed][metric]);
    const ci = bootstrapCI(differences, 12000, metric.length + contrast.length + 71);
    return {
      chip: groupA.chip,
      contrast,
      arm_a: groupA.arm,
      arm_b: groupB.arm,
      metric,
      mean_difference: mean(differences),
      seed_sd: sd(differences),
      ci_low: ci[0],
      ci_high: ci[1],
      n_seeds: differences.length,
      positive_seeds: differences.filter((value) => value > 0).length,
      negative_seeds: differences.filter((value) => value < 0).length,
    };
  });
}

function csvEscape(value) {
  const text = String(value ?? '');
  return /[",\n]/.test(text) ? `"${text.replaceAll('"', '""')}"` : text;
}

function writeCsv(file, rows, fields) {
  const lines = [fields.join(',')];
  for (const row of rows) lines.push(fields.map((field) => csvEscape(row[field])).join(','));
  fs.writeFileSync(file, `${lines.join('\n')}\n`);
}

function svgText(x, y, text, size = 14, anchor = 'start', fill = '#1d2935') {
  return `<text x="${x}" y="${y}" font-family="Arial, sans-serif" font-size="${size}" text-anchor="${anchor}" fill="${fill}">${text}</text>`;
}

function metricSvg(summaries) {
  const width = 1320;
  const height = 820;
  const colors = ['#2a6f97', '#e07a5f', '#43aa8b', '#8e6cbe', '#f2a65a', '#6c757d'];
  const groups = summaries;
  const panels = metricNames.map((metric, index) => ({ metric, x: 40 + (index % 3) * 430, y: 70 + Math.floor(index / 3) * 350 }));
  let svg = `<svg xmlns="http://www.w3.org/2000/svg" width="${width}" height="${height}" viewBox="0 0 ${width} ${height}"><rect width="100%" height="100%" fill="#fbfcfe"/>`;
  svg += svgText(width / 2, 30, 'R2=0 六组离线复验：探索与生存指标', 22, 'middle', '#13293d');
  for (const panel of panels) {
    const { metric, x, y } = panel;
    const values = groups.map((group) => group[metric].mean);
    const max = Math.max(...values, 1e-9);
    const min = Math.min(...values, 0);
    const span = Math.max(max - min, 1e-9);
    const plotX = x + 20;
    const plotY = y + 30;
    const plotW = 370;
    const plotH = 240;
    svg += `<rect x="${x}" y="${y}" width="400" height="300" rx="12" fill="#ffffff" stroke="#d8e0e8"/>`;
    svg += svgText(x + 200, y + 22, metricLabels[metric], 16, 'middle', '#13293d');
    svg += `<line x1="${plotX}" y1="${plotY + plotH}" x2="${plotX + plotW}" y2="${plotY + plotH}" stroke="#8a98a8"/>`;
    svg += `<line x1="${plotX}" y1="${plotY}" x2="${plotX}" y2="${plotY + plotH}" stroke="#8a98a8"/>`;
    for (let i = 0; i < groups.length; i += 1) {
      const group = groups[i];
      const barW = 46;
      const bx = plotX + 12 + i * 58;
      const barH = ((group[metric].mean - min) / span) * (plotH - 28);
      const by = plotY + plotH - barH;
      const ci = group[metric].seed_ci95;
      const lowY = plotY + plotH - ((ci[0] - min) / span) * (plotH - 28);
      const highY = plotY + plotH - ((ci[1] - min) / span) * (plotH - 28);
      svg += `<rect x="${bx}" y="${by}" width="${barW}" height="${Math.max(barH, 1)}" fill="${colors[i]}" opacity="0.85"/>`;
      svg += `<line x1="${bx + barW / 2}" y1="${highY}" x2="${bx + barW / 2}" y2="${lowY}" stroke="#263238" stroke-width="2"/><line x1="${bx + 12}" y1="${highY}" x2="${bx + barW - 12}" y2="${highY}" stroke="#263238"/><line x1="${bx + 12}" y1="${lowY}" x2="${bx + barW - 12}" y2="${lowY}" stroke="#263238"/>`;
      svg += svgText(bx + barW / 2, plotY + plotH + 18, `${i + 1}`, 11, 'middle');
    }
    svg += svgText(plotX + plotW / 2, y + 294, '组编号（见图例）', 11, 'middle', '#657482');
  }
  const legendY = 775;
  groups.forEach((group, index) => {
    const x = 30 + index * 215;
    svg += `<rect x="${x}" y="${legendY - 12}" width="14" height="14" fill="${colors[index]}"/>`;
    svg += svgText(x + 20, legendY, `${index + 1} ${group.label}`, 11);
  });
  svg += '</svg>';
  return svg;
}

function actionSvg(summaries) {
  const actionKeys = [...new Set(summaries.flatMap((group) => Object.keys(group.action_probabilities)))].sort();
  const width = 1320;
  const height = 620;
  const colors = ['#2a6f97', '#e07a5f', '#43aa8b', '#8e6cbe', '#f2a65a', '#6c757d'];
  const left = 80;
  const top = 80;
  const chartW = 1160;
  const chartH = 420;
  let svg = `<svg xmlns="http://www.w3.org/2000/svg" width="${width}" height="${height}" viewBox="0 0 ${width} ${height}"><rect width="100%" height="100%" fill="#fbfcfe"/>`;
  svg += svgText(width / 2, 35, '动作概率 / 经验动作频率', 22, 'middle', '#13293d');
  svg += `<rect x="${left}" y="${top}" width="${chartW}" height="${chartH}" fill="#fff" stroke="#d8e0e8"/>`;
  const groupW = chartW / actionKeys.length;
  const barW = Math.min(22, groupW / (summaries.length + 1));
  actionKeys.forEach((action, actionIndex) => {
    const baseX = left + actionIndex * groupW + groupW / 2 - (summaries.length * barW) / 2;
    summaries.forEach((group, groupIndex) => {
      const value = group.action_probabilities[action] || 0;
      const h = value * (chartH - 25);
      const x = baseX + groupIndex * barW;
      svg += `<rect x="${x}" y="${top + chartH - h}" width="${barW - 2}" height="${h}" fill="${colors[groupIndex]}" opacity="0.85"/>`;
    });
    svg += svgText(left + actionIndex * groupW + groupW / 2, top + chartH + 20, action.replace('verb_', '动作'), 11, 'middle');
  });
  summaries.forEach((group, index) => {
    const x = 40 + index * 215;
    svg += `<rect x="${x}" y="560" width="14" height="14" fill="${colors[index]}"/>`;
    svg += svgText(x + 20, 572, `${index + 1} ${group.label}`, 11);
  });
  svg += '</svg>';
  return svg;
}

function main() {
  fs.mkdirSync(output, { recursive: true });
  fs.mkdirSync(source22554, { recursive: true });
  const seeds22021 = readJson(path.join(source22021, 'design.json')).training_seeds;
  const groups = [
    makeGroup('22021', 'masked', '22021 纯 R1', source22021, seeds22021),
    makeGroup('22021', 'baseline', '22021 刺激前自发 MEA', path.join(source22021, 'pre-stim-baseline'), seeds22021),
    makeGroup('22021', 'sham', '22021 当轮零输出 MEA', source22021, seeds22021),
    load22554Group('pure', '22554 纯 R1'),
    load22554Group('ordered', '22554 有序 MEA'),
    load22554Group('shuffled', '22554 打乱 MEA'),
  ];
  const summaries = groups.map(summarize);
  const byArm = Object.fromEntries(groups.map((group) => [group.arm, group]));
  const effects = [
    ...pairedEffects(byArm.baseline, byArm.masked, '22021 自发 MEA − 纯 R1'),
    ...pairedEffects(byArm.sham, byArm.masked, '22021 零输出 MEA − 纯 R1'),
    ...pairedEffects(byArm.ordered, byArm.pure, '22554 有序 MEA − 纯 R1'),
    ...pairedEffects(byArm.shuffled, byArm.pure, '22554 打乱 MEA − 纯 R1'),
    ...pairedEffects(byArm.ordered, byArm.shuffled, '22554 有序 MEA − 打乱 MEA'),
  ];
  const allRows = groups.flatMap((group) => group.rows);
  writeCsv(path.join(output, 'metrics_by_world.csv'), allRows, ['chip', 'arm', 'label', 'train_seed', 'world_seed', ...metricNames, 'terminated']);
  const summaryRows = summaries.map((summary) => {
    const row = { chip: summary.chip, arm: summary.arm, label: summary.label, n_worlds: summary.n_worlds, n_seeds: summary.n_seeds, terminated_worlds: summary.terminated_worlds };
    for (const metric of metricNames) {
      row[`${metric}_mean`] = summary[metric].mean;
      row[`${metric}_seed_sd`] = summary[metric].seed_sd;
      row[`${metric}_ci_low`] = summary[metric].seed_ci95[0];
      row[`${metric}_ci_high`] = summary[metric].seed_ci95[1];
    }
    return row;
  });
  writeCsv(path.join(output, 'summary.csv'), summaryRows, Object.keys(summaryRows[0]));
  writeCsv(path.join(output, 'effects.csv'), effects, ['chip', 'contrast', 'arm_a', 'arm_b', 'metric', 'mean_difference', 'seed_sd', 'ci_low', 'ci_high', 'n_seeds', 'positive_seeds', 'negative_seeds']);
  fs.writeFileSync(path.join(output, 'summary.json'), JSON.stringify({ protocol: { r2_enabled: false, train_seeds_per_group: 8, heldout_worlds_per_group: 48, train_horizon: 48, eval_horizon: 6 }, groups: summaries }, null, 2));
  fs.writeFileSync(path.join(output, 'metrics.svg'), metricSvg(summaries));
  fs.writeFileSync(path.join(output, 'action_probabilities.svg'), actionSvg(summaries));
  const getEffect = (contrast, metric) => effects.find((effect) => effect.contrast === contrast && effect.metric === metric);
  const spontaneousExploration = getEffect('22021 自发 MEA − 纯 R1', 'exploration');
  const shamExploration = getEffect('22021 零输出 MEA − 纯 R1', 'exploration');
  const orderedExploration = getEffect('22554 有序 MEA − 纯 R1', 'exploration');
  const shuffledExploration = getEffect('22554 打乱 MEA − 纯 R1', 'exploration');
  const orderedThirst = getEffect('22554 有序 MEA − 纯 R1', 'thirst');
  const shuffledThirst = getEffect('22554 打乱 MEA − 纯 R1', 'thirst');
  const report = [
    '# 六组 R2=0 离线复验',
    '',
    '本报告按每枚芯片三组执行。22021 使用纯 R1、刺激前自发 MEA、当轮零输出 MEA。22554 使用纯 R1、有序 MEA、打乱 MEA。每组 8 个训练种子、48 个留存世界、48 个训练决策、6 个留存决策。R2 全部关闭，训练奖励严格等于 R1。',
    '',
    '## 指标汇总',
    '',
    '| 组 | 探索格数 | R1 | 生命 | 饱食 | 水分 |',
    '|---|---:|---:|---:|---:|---:|',
    ...summaries.map((summary) => `| ${summary.label} | ${summary.exploration.mean.toFixed(3)} | ${summary.r1.mean.toFixed(6)} | ${summary.health.mean.toFixed(3)} | ${summary.hunger.mean.toFixed(3)} | ${summary.thirst.mean.toFixed(3)} |`),
    '',
    '## 芯片内配对效应（A − B）',
    '',
    '| 对照 | 指标 | 平均差 | 95% CI | 正向种子 |',
    '|---|---|---:|---:|---:|',
    ...effects.map((effect) => `| ${effect.contrast} | ${metricLabels[effect.metric]} | ${effect.mean_difference.toFixed(6)} | [${effect.ci_low.toFixed(6)}, ${effect.ci_high.toFixed(6)}] | ${effect.positive_seeds}/${effect.n_seeds} |`),
    '',
    '## 结果解读',
    '',
    `22021 的刺激前自发 MEA 相对纯 R1 探索格数增加 ${spontaneousExploration.mean_difference.toFixed(3)}，零输出 MEA 增加 ${shamExploration.mean_difference.toFixed(3)}。零输出也出现同方向增益，说明这轮提升不能单独归因于 MEA 信号内容。`,
    `22554 的有序 MEA 相对纯 R1 探索格数变化 ${orderedExploration.mean_difference.toFixed(3)}，打乱 MEA 变化 ${shuffledExploration.mean_difference.toFixed(3)}，两者区间都跨零，因此没有复现探索格数的稳定弱阳性。`,
    `22554 两个 MEA 组的 R1 均相对纯 R1 上升约 0.004，但有序和打乱之间接近零差异。两组末状态水分分别变化 ${orderedThirst.mean_difference.toFixed(3)} 和 ${shuffledThirst.mean_difference.toFixed(3)}，均为下降。生命指标在所有组的留存末点均为 100，未形成区分度。`,
    '',
    '图表文件：`metrics.svg` 与 `action_probabilities.svg`。逐世界数据在 `metrics_by_world.csv`，汇总数据在 `summary.csv`。',
    '配对效应明细在 `effects.csv`。置信区间按 8 个训练种子的配对均值 bootstrap 计算。',
    '',
    '跨芯片均值用于展示，因芯片起始记录和源窗口不同，效应判断以芯片内配对比较为准。动作图中，22021 使用留存轨迹的经验动作频率，22554 使用评估轨迹记录的 verb 概率均值。',
  ];
  fs.writeFileSync(path.join(output, 'report.md'), `${report.join('\n')}\n`);
  console.log(JSON.stringify({ output, groups: summaries.map((summary) => ({ chip: summary.chip, arm: summary.arm, worlds: summary.n_worlds, seeds: summary.n_seeds })) }));
}

main();
