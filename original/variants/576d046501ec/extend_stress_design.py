import json
import random
from pathlib import Path


source = Path(r"output/mea_predictability_22021_20261119T042416Z/offline-stress-three-arms/design.json")
target = Path(r"tmp/stress_design_32.json")
data = json.loads(source.read_text(encoding="utf-8"))

old_seeds = list(data["training_seeds"])
used = set(old_seeds) | set(data["worlds"])
used |= {world for worlds in data["episodes"].values() for world in worlds}
rng = random.Random(220211192032)
new_seeds = []
for candidate in rng.sample(range(354000000, 355000000), 200):
    if candidate not in used:
        new_seeds.append(candidate)
        used.add(candidate)
        if len(new_seeds) == 24:
            break
assert len(new_seeds) == 24
seeds = old_seeds + new_seeds

train_worlds = set(world for worlds in data["episodes"].values() for world in worlds)
new_train_worlds = []
for candidate in rng.sample(range(355000000, 356000000), 200):
    if candidate not in used and candidate not in train_worlds:
        new_train_worlds.append(candidate)
        used.add(candidate)
        if len(new_train_worlds) == 24 * 4:
            break
assert len(new_train_worlds) == 96

for index, seed in enumerate(new_seeds):
    data["episodes"][str(seed)] = new_train_worlds[index * 4:(index + 1) * 4]

def permute(windows, seed):
    rng_local = random.Random(seed + 835)
    result = []
    for trial_start in range(0, len(windows), 4):
        blocks = [list(range(trial_start, trial_start + 2)), list(range(trial_start + 2, trial_start + 4))]
        if len(blocks) > 1:
            blocks.reverse()
        result.extend(index for block in blocks for index in block)
    return result

data["training_seeds"] = seeds
data["training_order"] = {str(seed): permute(data["windows"][:48], seed) for seed in seeds}
data["study"] = "offline_confirmatory_stress_22021_v1"
data["session_id"] = "confirm22021-20261120T"
data["registration_sha256"] = "draft-registration-20261120"
target.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
print(target, len(seeds), len(data["worlds"]), sum(len(v) for v in data["episodes"].values()))
