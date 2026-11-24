# Game, training and acquisition source

[Read the paper](../paper.pdf) · [Browse the game project](../original/project/) · [Browse the archived confirmatory training entry point](../historical/train.py) · [Get the recordings and game receipts](data-access.md)

The Godot game, policy and acquisition programs evolved across several experiments. The current project is useful for exploring how the game and agent work. For a comparison in the paper, first identify the recording and study, then choose the corresponding archived program and configuration. On chip 22021 the readouts were replayed offline. The tree also contains live-acquisition and development software from other runs.

## Browse the project

| What to inspect | Starting point |
|---|---|
| Game and survival rules | [Godot project](../original/project/project.godot), [scene](../original/project/scenes/main.tscn), [game scripts](../original/project/scripts/), [graphics and sounds](../original/project/assets/) |
| Observation, available actions and policy | [M9 action contract](../original/project/agent/m9_contract.py), [policy](../original/project/agent/m9_policy.py), [environment](../original/project/agent/m9_env.py) |
| Offline replay and MEA features | [Recording readout](../original/project/standalone/mea_live_pair/livepair/readout.py), [offline runtime](../original/project/standalone/mea_online/mea_online/offline.py), [M9 interface](../original/project/standalone/mea_online/mea_online/m9_runtime.py) |
| Training, evaluation and the adverse starts | [Historical training and evaluation guide](../historical/README.md), [experiment configurations](../original/project/configs/), [whole-project tools](../original/project/tools/) |
| Hardware integration and other experiments | [Acquisition and study programs](../original/project/standalone/), [acquisition adapters](../original/project/tools/mea_next_recording/) |
| Tests | [Agent and software tests](../original/project/tests/), [Godot environment tests](../original/project/godot_tests/) |

To open the game with an installed Godot 4 executable, run these commands from the repository root. The first command imports the bundled art and audio. The second launches the game.

```sh
godot --headless --path original/project --editor --import
godot --path original/project
```

The following checks use the game’s collision rules and the agent’s action contract. The archived scene imported successfully with Godot 4.7.2 on Windows; the collision check and 15 agent tests passed. A separate check of the legacy survival UI suite reported three failing assertions concerning the default game mode and the source or timing of a simulated model response.

```sh
godot --headless --path original/project --script res://godot_tests/test_collision.gd
cd original/project
python -B -m unittest discover -s tests -p test_agent.py -q
```

## Recover the source layout used in a run

The [source index](../original/source-index.csv) maps each collection and original path to its stored file, the upstream SHA-256, the distributed SHA-256 and the preservation status. Distinct historical versions of a file can have the same name. Equal-content copies can point to a single stored file. The materializer verifies the distributed file hashes before copying any files to a new directory.

```sh
python original/materialize.py list
python original/materialize.py verify
python original/materialize.py extract confirmatory restored-confirmatory
```

| Collection | Contents |
|---|---|
| `workspace` | Current project layout, including source, assets, configuration and tests. Internal working notes and historical narrative reports remain in the source archive. |
| `project-archive` | Historical project source, including earlier versions of the environment, model and run programs. |
| `runtime-a`, `runtime-b` | The two preserved training-node source snapshots, with separate trees for different runs and scene configurations. |
| `confirmatory`, `reading-copy` | Confirmatory training, stress-scene setup, action replay, evaluation and analysis programs; the compact reading copy includes experiment design and context files. |
| `six-arm-a`, `six-arm-b` | Code entries indexed in the two six-group and reward-weight sweep archives. |
| `historical-supplement`, `analysis-and-figures` | Older source versions and scripts used for the original supplementary analyses and figures. |
| `manuscript-figure-source`, `supplementary-local`, `base-and-warm-start` | An additional original figure script, later local diagnostics and the indexed model chat template. |

The catalogue covers 7,795 source and supporting-file entries in 1,447 distributed files. The [older archive location index](../original/historical-locations.csv) records another 9,479 path references to these contents. An independent check against the original archive index located all 6,047 program, scene, build-script and template entries by source hash. Of these, 5,917 have a distributed source mapping and 130 have an explicit exclusion. The [omissions table](../original/omissions.csv) also identifies excluded supporting files. [Archive names, sizes and registered hashes](../original/archive-index.csv) refer to the original containers.

## Boundaries of a rerun

- Files marked `exact` match their recorded source bytes. Files marked `site-specific-values-replaced` retain the historical implementation with workstation paths, lab network addresses, account names or contact strings replaced. Supply your own authorized resource locations when using a deployment script. The unmodified containers can be checked separately against the registered archive hashes.
- The hardware adapters and study controllers are included. The MaxLab SDK sources and generated build or dependency files identified in the omissions table are separate dependencies. Hardware runs need the matching SDK and instrument. The paper's offline result commands are documented in the [reproduction guide](reproduction-guide.md).
- Download the [raw electrophysiology](data-access.md), [training and evaluation receipts](data-access.md#training-evidence-and-parameters), and [checkpoints](archive-downloads.md). Match files by recording, seed, group and study-specific layout. The [project contents](project-contents.md) link the pinned upstream base and runtime requirements.
- The two runtime snapshots preserve the game sources that were available for evaluation. The manuscript documents the unavailable registered audit source and the scenario-verification hash difference. A successful local replay cannot recover the missing source or attest to the timing of the registration.
