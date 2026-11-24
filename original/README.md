# Original experiment source

Start in [project](project/) to browse the game, policy, recording interface, local configurations, assets and tests. The [source guide](../docs/original-source.md) connects each subsystem to the experiment and explains which files and dependencies to use for a historical rerun.

For earlier versions, find the study in [source-index.csv](source-index.csv). Its `stored_as` column points to the copy held here; its two SHA-256 columns distinguish the original bytes from versions with laboratory addresses or paths replaced. The [archive list](archive-index.csv) identifies the corresponding original containers and their registered checksums.

Run these commands from the repository root:

```sh
python original/materialize.py verify
python original/materialize.py list
python original/materialize.py extract confirmatory restored-confirmatory
```

The last command restores the study's program layout into a new directory. Raw recordings, checkpoints and bulk logs have [separate access and verification instructions](../data/README.md).
