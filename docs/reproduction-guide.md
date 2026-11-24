# Reproduce the paired analyses

Recompute the paper’s paired comparisons from archived game outcomes. The agent received microelectrode-array measurements by offline replay. Its actions changed the game state. The tissue received no game-action feedback.

From the repository root, run the offline smoke test with Python:

```sh
python -m src.reproduce smoke
```

The command uses the Python standard library, checks the distributed input hashes, recomputes the primary paired distribution and compares it to the archived reference. Its final JSON line names a fresh directory under `runs/`. Open `result.json` there for the result and `run_manifest.json` for inputs, output digests, source revision and environment. In the intact group versus the all-zero group, the expected counts are **101 wins, 615 ties and 52 losses** in 768 pairs.

## Install and run

The maintained analyses were tested on CPU with Python 3.12.14. Use an environment with Python 3.12 and Git. From that environment:

```sh
python -m pip install -r requirements.txt
python -m src.reproduce all
python -m unittest discover -s tests -v
```

The smoke test and `exploration` run offline before installation. The complete numerical analysis uses the pinned NumPy release. The HDF5 feature audit additionally uses `configs/raw-requirements.txt`. On systems with GNU Make, `make smoke` and `make reproduce` invoke the same entry point. Use `python -m src.reproduce --help` for options. The included Dockerfile provides the CPU analysis environment. The historical training environment is only partially documented. [Original training, evaluation and PPO source](../historical/README.md) · [Environment](environment.md)

## Results and commands

| Paper analysis | What the archived data support | Recompute from the repository root |
|---|---|---|
| Short trial, first recording on chip 22021 | After 6 decisions, the intact group revealed 1.99 more cells per game than the all-zero group on average. Median paired difference 0. The unstimulated group averaged 2.16 more cells and tied on 602/768 pairs. | `python -m src.reproduce exploration` |
| Fixed maps and new-map inference | On the same trajectories truncated at game tick 6, the fixed-map sensitivity intervals were above zero. Both intervals obtained by resampling seeds and maps crossed zero at the nominal 98.333% level. Equal-time exploration on new maps remains uncertain. | `python -m src.reproduce exploration-intervals` |
| Score decomposition in the first-day subtrial | Action duration accounted for 84.3% and 95.9% of the intact and unstimulated groups’ respective score differences from the all-zero group. Both residual intervals included zero. | `python -m src.reproduce score` |
| Adverse starts on the second sample | 0/6 primary contrasts met the available draft’s success rule. Corrected intervals for thirst and hunger survival and low-health exploration all crossed zero. | `python -m src.reproduce stress` |
| Stimulation conditions on each acquisition day | Corrected exploration and score intervals for regular and irregular stimulation versus the unstimulated group crossed zero on both days. A reproducible benefit under the measured conditions was undetected. | `python -m src.reproduce stimulation` |
| First-day cross-chip comparison, chip 22554 | Exploration did not replicate. The intact and shuffled groups had higher scores than the native game-reward-only comparator. The score interval for intact versus shuffled crossed zero. | `python -m src.reproduce cross-chip` |
| Next-day leave-site-file-out comparison | Exploration intervals crossed zero at both game-time budgets. The output contains the per-endpoint means and paired intervals for the shared budgets. | `python -m src.reproduce heldout` |
| Game-time boundary scoring | Recalculate each scene-by-group signed difference and the change in each paired score contrast. Endpoint agreement can coexist with a discrepancy in the score at the cutoff. | `python -m src.reproduce boundary` |
| Action composition | Count selected actions in every archived short game and calculate percentage-point differences from the all-zero group. These are descriptive frequencies. | `python -m src.reproduce actions` |
| Training-budget audit | Sum batch-level counts and inspect recorded decisions and policy and value updates per final model. The successful training processes’ original elapsed time and GPU-hours remain unverified. | `python -m src.reproduce budgets` |
| Pre-registration deviations | Retrieve the documented receipt counts, pilot overlap and reload summary. Independent verification of the original timing and unique pre-unblinding lock remains unresolved. | `python -m src.reproduce deviations` |

[Claim-level reproduction matrix](reproduction_matrix.csv) · [Signed corrections and available source tables](source-notes.md) · [Data dictionary](../data/data_dictionary.csv)

## Groups and scope

For the first short trial, the **intact group** (historical figure label *self-generated*) received the pre-stimulation spontaneous recording. The **unstimulated group** (*zero-output*) received measured input during a period with zero commanded stimulation and possible previous exposure. The **all-zero group** (*masked*) received a constant zero feature vector. This short trial had no shuffled group. In the adverse-start trial, the **shuffled group** received a within-trial block permutation. In the first-day cross-chip trial, the shuffle affected training-trial pairings and both recording-connected groups used ordered input during evaluation.

The first short trial used 16 training seeds and the same 48 maps per seed. Differences are paired within seed and map. A game pair is a software evaluation unit. The recording’s origin-specific contribution and the separate contributions of training and inference-time input remain unidentified.

The available pre-registration records contain deviations. After evaluation 18,720 receipts acquired new design identifiers. Independent earlier copies cover 99 receipts. The registered audit game source was unavailable at evaluation, a project snapshot was used and the scenario-verification digest differed from the registered value. All 64 adverse-start maps and 8 seeds had appeared in pilot work. Exact reproduction of a downstream table does not resolve the date or integrity of upstream records.

## Data, code and citation

The package contains paired game outcomes, compact feature and budget tables, figures and maintained analysis source. The [data card](../data/datasheet.md) describes acquisition, the live-training gate and cross-chip interpretation. See the [original-data download](data-access.md) for the author-supplied archive and the [verification commands](../data/README.md) for checking locally retrieved files.

The [MIT license](../LICENSE) covers study-authored software. The [data terms](../LICENSE-DATA) describe public access and redistribution of the derived game tables, public access to the raw recording share, and the distinct rights for model weights and third-party assets. Cite [Brainintelli Research Lab](../CITATION.cff).
