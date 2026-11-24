# Archived training and evaluation methods

| Stage | Inspect the source | Data dependency |
|---|---|---|
| Training | [Original training driver](train.py), including fixed-context observation preparation, action collection, actor/critic initialization, trajectory finalization and checkpoint writes | Registered `design.json`, `context.json`, configured model paths, original project modules and licensed base weights, a matching game and Godot engine |
| PPO updates | [Delegating wrapper](methods/training_update.py.txt), [optimizer update](methods/update.py.txt), and [trajectory finalization](methods/finish.py.txt) from matching runtime snapshots | Project tensor preparation, reference policy, PyTorch and the recorded environment |
| Frozen policy evaluation | [Original evaluation entry point](stress/evaluate.py), [action loop](methods/play.py.txt) and [cutoff scoring](methods/score_step.py.txt) | Matching final checkpoints, game/observer scenes, secondary runtime and the evaluated window plan |
| Archived adverse-start analysis | [Original statistical driver](stress/analyze.py) | Project design, saved episode/step files and pinned historical source identities |

The training, evaluation and statistical drivers are complete byte-for-byte copies. Files under `methods/` are source excerpts for reading. Each preserves the named function with normalized line endings and omits its enclosing runtime. [Source manifest](source_manifest.json) records archive and member digests, line ranges for excerpts and the exact bytes delivered. Matching snapshots show the source text available in the archives. Complete runtime fingerprints of the successful training processes remain unverified.

For a numerical reproduction using the distributed outcomes, run `python -m src.reproduce all` at the repository root. [Download and restore the public receipts](../docs/data-access.md#training-evidence-and-parameters), then extract the archived drivers with `python -m src.reproduce archived-code --archive receipts.zip`. The extracted archive retains historical host-layout assumptions in resource lookup. Supply the actual licensed dependencies and original run tree in a separate working directory before attempting an upstream replay.

The original training command takes `--root`, `--project`, `--godot`, `--model-config`, `--seed` and `--arm` (masked, ordered or shuffled). The evaluation command takes the trained seed, archived arm and starting scene. Inspect source and fingerprints locally with `python scripts/check_historical.py`. That check performs no training, downloads or inference. The [archive guide](../docs/archive-downloads.md) locates the saved project checkpoints. The base weights and third-party runtime have their own upstream distributions.

Study-authored software in this release is distributed under the [MIT License](../LICENSE). The [third-party notices](../THIRD_PARTY_NOTICES.md) and [data terms](../LICENSE-DATA) describe the separate terms for dependencies, recordings and model assets.
