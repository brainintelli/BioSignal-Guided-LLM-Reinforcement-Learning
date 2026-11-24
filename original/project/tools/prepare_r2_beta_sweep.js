const fs = require('fs');
const path = require('path');

const workspace = path.resolve(__dirname, '..');
const output = path.join(workspace, 'output', 'mea_r2_beta_sweep_20261120');
const old22021 = JSON.parse(fs.readFileSync(path.join(workspace, 'output', 'mea_predictability_22021_20261118T093720Z', 'offline-mea-stim-ablation', 'design.json')));
const new22021Main = JSON.parse(fs.readFileSync(path.join(workspace, 'output', 'mea_six_arm_r1_off_20261120', 'main', 'design.json')));
const new22021Baseline = JSON.parse(fs.readFileSync(path.join(workspace, 'output', 'mea_six_arm_r1_off_20261120', 'baseline', 'design.json')));
const oldSeeds = old22021.training_seeds.map(Number);
const newSeeds = new22021Main.training_seeds.map(Number);

function writeJson(file, value) {
  fs.mkdirSync(path.dirname(file), { recursive: true });
  fs.writeFileSync(file, `${JSON.stringify(value, null, 2)}\n`);
}

const combined22021 = {
  ...new22021Main,
  study: '22021_r2_beta_sweep_20261120',
  training_seeds: [...oldSeeds, ...newSeeds],
  episodes: { ...old22021.episodes, ...new22021Main.episodes },
  arms: ['baseline', 'sham'],
  contexts: {
    baseline: new22021Baseline.contexts.baseline,
    sham: new22021Main.contexts.sham,
  },
  reward_mix: {
    schema: 'bounded_native_r1.v1',
    reference_per_tick: 0.00977500000000008,
    scale_fraction: 0.1,
    alpha: 1,
    gamma_tick: 0.99985,
    source: 'same-sample calibrated R1 scale, beta supplied per run',
  },
  beta_sweep: [0, 0.02, 0.05, 0.10],
  seed_cohorts: {
    prior_8: oldSeeds,
    added_8: newSeeds,
  },
};

writeJson(path.join(output, 'configs', '22021', 'design.json'), combined22021);

const baseBudget = JSON.parse(fs.readFileSync(path.join(workspace, 'output', 'mea_six_arm_r1_off_20261120', '22554_extra', 'budget.json')));
const combined22554 = {
  ...baseBudget,
  training_seeds: [2255401, 2255402, 2255403, 2255404, 2255405, 2255406, 2255407, 2255408, 2255409, 2255410, 2255411, 2255412, 2255413, 2255414, 2255415, 2255416],
  registered_training_seeds: [2255401, 2255402, 2255403, 2255404, 2255405, 2255406, 2255407, 2255408, 2255409, 2255410, 2255411, 2255412, 2255413, 2255414, 2255415, 2255416],
  cohort: 'r2_beta_sweep_20261120',
  reward_mix: {
    schema: 'bounded_native_r1.v1',
    reference_per_tick: 0.00977500000000008,
    scale_fraction: 0.1,
    alpha: 1,
    gamma_tick: 0.99985,
    source: 'same-sample calibrated R1 scale, beta supplied per run',
  },
  beta_sweep: [0, 0.02, 0.05, 0.10],
};

for (const beta of [0.02, 0.05, 0.10]) {
  const tag = `beta_${String(Math.round(beta * 100)).padStart(3, '0')}`;
  writeJson(path.join(output, 'configs', '22554', `budget-${tag}.json`), { ...combined22554, r2_beta: beta });
}

writeJson(path.join(output, 'configs', '22554', 'budget-beta-0.json'), { ...combined22554, r2_beta: 0 });
console.log(JSON.stringify({ output, seeds22021: combined22021.training_seeds.length, seeds22554: combined22554.training_seeds.length, betas: combined22554.beta_sweep }));
