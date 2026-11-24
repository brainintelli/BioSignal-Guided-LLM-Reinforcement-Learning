const fs = require('fs');
const path = require('path');

const workspace = path.resolve(__dirname, '..');
const source = path.join(workspace, 'output', 'mea_predictability_22021_20261118T093720Z', 'offline-mea-stim-ablation');
const nextRoot = path.join(workspace, 'output', 'mea_six_arm_r1_off_20261120', 'staging', '22021');
const seeds = Array.from({ length: 8 }, (_, i) => 102600001 + i);

function readJson(file) {
  return JSON.parse(fs.readFileSync(file, 'utf8'));
}

function writeJson(file, value) {
  fs.writeFileSync(file, `${JSON.stringify(value, null, 2)}\n`);
}

function episodesForSeeds() {
  return Object.fromEntries(seeds.map((seed) => [String(seed), [seed, seed + 100000, seed + 200000, seed + 300000]]));
}

function makeRound(name, sourceRoot, arms, files) {
  const destination = path.join(nextRoot, name);
  fs.mkdirSync(destination, { recursive: true });
  for (const file of files) fs.copyFileSync(path.join(sourceRoot, file), path.join(destination, file));
  const design = readJson(path.join(sourceRoot, 'design.json'));
  design.training_seeds = seeds;
  design.episodes = episodesForSeeds();
  design.arms = arms;
  design.contexts = Object.fromEntries(Object.entries(design.contexts).filter(([arm]) => arms.includes(arm)));
  design.round = '20261120_extra_seeds_r2_off';
  design.parent_training_seeds = [102122616, 102083199, 102411608, 102473463, 102385481, 102476656, 102481912, 102452542];
  writeJson(path.join(destination, 'design.json'), design);
}

makeRound('main', source, ['masked', 'sham'], [
  'train.py', 'run_study.py', 'verify_game.py', 'resources.py', 'model_config.json', 'masked.json', 'sham.json',
]);
makeRound('baseline', path.join(source, 'pre-stim-baseline'), ['baseline'], [
  'train.py', 'run_study.py', 'verify_game.py', 'resources.py', 'model_config.json', 'baseline.json',
]);

console.log(JSON.stringify({ nextRoot, seeds, arms: ['masked', 'sham', 'baseline'] }));
