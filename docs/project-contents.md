# Project contents and reproduction boundaries

## Choose an entry point

| Goal | Files | Verification or requirement |
|---|---|---|
| See the environment in action | [Gameplay recording](../media/gameplay.mp4) and [viewing notes](video.md) | The recording shows the Meadowmind game, its resource meters and action history. |
| Open the game or inspect the simulation | [Godot project](../original/project/project.godot), [scenes](../original/project/scenes/), [game scripts](../original/project/scripts/), [assets](../original/project/assets/) | Open the project with Godot. The archived training-machine launchers require their original services; a local game scene can be run independently. |
| Trace the model and the input pipeline | [Agent and PPO implementation](../original/project/agent/), [experiment tools](../original/project/tools/), [configuration](../original/project/configs/), [paper-specific frozen programs](../historical/) | The [source guide](original-source.md) separates the current working tree from historical training and acquisition snapshots. |
| Recreate the original source layout for an experiment | [Source-path and SHA-256 mapping](../original/source-index.csv), [materialization utility](../original/materialize.py) | Run `python original/materialize.py list` to see the named collections and `python original/materialize.py verify` to check the stored bytes. The [omission register](../original/omissions.csv) records excluded items. |
| Check the paper’s reported behavior | [Original paired-game table](../data/upstream/22021_逐局配对实测.csv), [derived game outcomes](../data/), [reference outputs](../results/), [test suite](../tests/) | Run `python -m src.reproduce smoke` or select an endpoint in the [command guide](reproduction-guide.md). CPU only. |
| Inspect raw electrophysiology | [Download and verify the H5 archive](data-access.md) | The archive has 182 distinct H5 files, including zero-frame files preserved for the acquisition audit. Verifying raw bytes or re-extracting features and replaying archived game outcomes are separate checks. |
| Rebuild the paper and figures | [Manuscript build instructions](../manuscript/README.md), [Word file](../manuscript/paper.docx), [vector figures and source](../figures/) | Regenerate all figures with `python figures/draw.py --all --output runs/figures` after installing `configs/figure-requirements.txt`. |
| Identify the exact model or historical archive | [Model components and digests](../data/upstream/确认性实验_原始外部清单.sha256), [model and archive inventories](../data/upstream/), [environment](environment.md) | Match model identity by seed and group. The file inventory includes the base/warm-start, optimizer and checkpoint packages. |

## Run the game locally

Install Godot 4.7.2 and place its executable on your path as `godot`. From the repository root, import the assets once, then start the default human-controlled game:

```sh
godot --headless --path original/project --import
godot --path original/project
```

For a short rule-based demonstration with no remote model or hardware, run:

```sh
godot --headless --path original/project --fixed-fps 60 --quit-after 600 -- --baseline --difficulty=m9 --smoke-test --smoke-duration=6
godot --headless --path original/project --script res://godot_tests/test_m9_world.gd
```

On a clean copy with Godot 4.7.2, the baseline smoke test reached tick 9 alive and the M9 world test reported zero failures. The historical `test_survival_ui.gd` still reported three failed assertions about the old default scenario and labels shown for a simulated model response. The original assertions are preserved. The manual and baseline checks do not require a GPU policy service.

## Verify evidence at three levels

1. **Published estimate.** The included software outcomes suffice to recompute the paired distribution. The original Chinese per-game table regenerates the compact English paired table byte for byte. To check the seed/map estimators, timing decomposition, primary endpoints and stimulation contrasts, use the [reproduction guide](reproduction-guide.md).
2. **Inputs and recorded decisions.** The full local set of 182 H5 files passed the published SHA-256 checks. Using the 19 adverse-start recordings and the matching historical training/evaluation archive, we ran `audit-chain` over **18,432 games and 669,715 recorded decisions**. It links feature windows to the input and model identifiers in the archived receipts, then rebuilds the endpoint table. The paper also describes a separate check that actually reloaded policies on a subset of games.
3. **New training or new hardware runs.** These need the historical base model, warm-start weights, checkpoints, a compatible GPU/software stack and, for acquisition, the instrument and its vendor runtime. A successful CPU replay uses archived actions and does not validate fresh PPO training, hardware stimulation or biological feedback. The [environment notes](environment.md) identify which historical settings could be verified.

## Historical checkpoints, logs and other large assets

The **base model and tokenizer** are available at the [Qwen3.5-0.8B upstream snapshot](https://huggingface.co/Qwen/Qwen3.5-0.8B/tree/2fc06364715b967f1860aea9cf38778875588b17). All 11 base-model files in the historical base/warm-start archive matched this revision, including the 1,746,942,600-byte weights shard. The [per-file fingerprint report](../data/upstream/base-model.json) lists the matches. The project-specific PPO warm starts, trained adapters, critics and optimizer states have their own file identities in the study inventories.

The [public receipts collection](data-access.md#training-evidence-and-parameters) contains training trajectories, model registrations, update records, evaluation receipts, scene setup and archived programs. The verification command accepts either the original export or the public distribution and records the SHA-256 of the supplied bytes.

The [original archive inventory](../data/upstream/archive-inventory.csv) records 12 containers totaling 65,747,939,480 bytes. The [public release guide](archive-downloads.md) links the screened distributions and explains any substitutions or exclusions. Several collections overlap. Raw H5 recordings are distributed separately through Quark.

Model-training machine images, a complete accelerator environment lock and a permanent public DOI are not supplied. No new recordings or training runs were made to assemble this repository.
