# Obtain and verify the project data

## Raw microelectrode-array recordings

[Open the author's Quark Drive share](https://pan.quark.cn/s/3f2e5122ab68), enter **`mFFv`**, and download the folder **`原始H5`** (“raw H5”). On a Chinese-language client, **提取码** is the extraction code and **下载** is Download. The authors have made this entry and extraction code available to the public. The Quark service manages file delivery separately from GitHub.

The extraction code was accepted and the remote directory was enumerated on 24 November 2026. It contained **182 distinct H5 files totaling 73,876,256,484 bytes**, plus the original acquisition index. All 182 remote filenames and byte counts matched the local files listed for the study. The original index has 192 provenance rows because a recording may have more than one archived source. Preserve the original filenames when downloading.

Each of the 182 **local** recordings was also checked against the frozen SHA-256 manifest. The share listing exposes names and sizes. Verify your downloaded bytes to establish content identity on your own machine.

| Inventory | What it identifies |
|---|---|
| [Share inventory](../data/upstream/share-inventory.csv) | One row per H5 file, with its remote size and expected SHA-256 from the local study archive |
| [Original acquisition index](../data/upstream/acquisition-index.csv) | Original Chinese column names, chip, start time, recording type, channel count, frames and source paths |
| [Frozen analysis manifest](../data/h5_manifest.csv) | Distinct byte streams and the subset of 19 recordings used in the adverse-start reconstruction |
| [Datasheet](../data/datasheet.md) | Collection conditions, channel qualification, interpretation and material-use status |

From the **repository root**, point the verifier at the folder that contains your downloaded H5 files. Directory names in the example are local choices.

```sh
python -m pip install -r configs/raw-requirements.txt
python -m src.reproduce verify-h5 --data-dir "./原始H5"
```

For just the adverse-start inputs, use `--stress-only`. To re-extract the calibration and neural feature windows used by the analysis, run:

```sh
python -m src.reproduce raw-windows --data-dir "./原始H5"
```

## Paired game data and aggregate tables

The [original paired-game CSV](../data/upstream/22021_逐局配对实测.csv) and [original aggregate catalogue](../data/upstream/data_table.csv) retain their Chinese headers and historical labels. In the first short experiment, 自发 means the **intact** pre-stimulation recording, 屏蔽 means **all-zero**, and 零输出 means the **unstimulated recording**. The latter still contains measured activity. Other cohorts used different controls. For corrected cohort-specific interpretations, refer to the [paper](../paper.pdf), the [reproduction guide](reproduction-guide.md) and [source notes](source-notes.md).

To regenerate the English paired table from the original CSV and check its digest:

```sh
python scripts/pack_pairs.py "data/upstream/22021_逐局配对实测.csv" runs/rebuilt-pairs.csv
python -m src.reproduce exploration
```

For the other endpoints, the [data dictionary](../data/data_dictionary.csv) lists columns and units, and the [reproduction matrix](reproduction_matrix.csv) locates each result and command.

## Training evidence and parameters

Download the [original training and evaluation receipts ZIP](https://github.com/brainintelli/organoid-replay-survival/releases/download/study-evidence-v1/training-evaluation-receipts.zip) while signed into a GitHub account with repository access. This private release asset contains 20,497 files (157,790,368 compressed bytes). Its SHA-256 is `248907bc759199690d475dd8c6ccbac2920f30df21316cb944a6ad7a1fa523f2`. The archive preserves historical machine paths.

With the ZIP in the repository root and the recordings in the directory indicated below:

```sh
python -m src.reproduce archived-code --archive training-evaluation-receipts.zip
python -m src.reproduce audit-chain --data-dir "./原始H5" --archive training-evaluation-receipts.zip
```

The second command reconstructs 18,432 adverse-start games from the recorded decisions. The resulting `stress.csv` matched the bundled endpoint table byte for byte (SHA-256 `3960b473a5bbd43ed97476a0e7b81a5cfe94572d6f595f343f2b819bb9232e00`). Fresh policy inference requires the model assets separately.

The Quark directory above contains the **electrode recordings and their index**. For game receipts, source, model indexes, weights and runtime requirements, use the [project contents](project-contents.md). The [source archive inventory](../data/upstream/archive-inventory.csv) identifies local historical deliveries by original name, byte count and SHA-256. The [96-model index](../data/upstream/确认性实验_96个模型索引.csv) identifies the adverse-start policies, and the [six-group index](../data/upstream/六组探索性实验_模型索引.csv) identifies the exploratory models. These CSVs identify files; each download location is listed separately in the project contents.

The [data terms](../LICENSE-DATA) describe the scope of reuse. The public H5 download entry and its extraction code provide access to the recordings. Other reuse and redistribution rights for raw recordings have not been specified.
