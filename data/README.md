# Data access

The CSV files in this directory contain derived outcomes used by the paper. Files in `../results/` hold the reference outputs. For the author-supplied raw-data share and extraction code, see [Download the original data](../docs/data-access.md). The [dictionary](data_dictionary.csv), [data card](datasheet.md), [raw checksums](h5_manifest.csv) and [data terms](../LICENSE-DATA) describe how to interpret and verify the files.

The [original schema inventory](upstream_schema.json) lists every original column in the upstream paired-game CSV, the historical aggregate catalogue and the acquisition index, including units and missingness. Catalogue labels must be interpreted within their original cohort. The English data use study-local names and pair keys.

## For an existing authorized holder

From the repository root, make the following inputs available at paths you choose. The Python commands below use example relative names. A directory search accepts existing acquisition filenames. It matches size and SHA-256.

1. Put, link or mount a locally authorized recording export at `authorized-h5/`.
2. Install the pinned dependencies and verify the recording inventory. Choose the stress-only option when your authorization covers that subset.

```sh
python -m pip install -r configs/raw-requirements.txt
python -m src.reproduce verify-h5 --data-dir authorized-h5 --stress-only
python -m src.reproduce raw-windows --data-dir authorized-h5
```

The feature audit re-reads the event datasets, checks channel mapping and recorded event digests, validates calibration and the normalized inputs, then writes a feature CSV and its digest. The subsequent policy and game steps need separate historical assets. For a complete recording export, use `python -m src.reproduce verify-h5 --data-dir authorized-h5` to verify every distinct raw digest in the manifest.

3. An authorized holder of the original program archive can place it at `authorized-model-programs.zip` and run:

```sh
python -m src.reproduce archived-code --archive authorized-model-programs.zip
```

The command checks the archive and each allowlisted source file before extracting training, evaluation and scene-setup programs into the private output directory. The exact archive fingerprint and member digests are in `../configs/study.yaml`. Extraction runs no historical program.

4. With both authorized inputs, follow the archived chain from device events to game outcomes:

```sh
python -m src.reproduce audit-chain --data-dir authorized-h5 --archive authorized-model-programs.zip
```

This command reconstructs the feature windows, checks each recorded evaluation decision against its assigned window and frozen model identity, recovers resource waits from the step receipts, and rebuilds the adverse-start outcome table byte for byte. It then calculates the registered comparisons from those outcomes. The feature and outcome CSVs, digests and estimates appear together in the run directory. Historical training and fresh policy inference require separate runtime and checkpoint access.

A holder of the original short-trial paired CSV may save it locally as `authorized-pairs.csv` and run `python scripts/pack_pairs.py authorized-pairs.csv runs/rebuilt-pairs.csv`. The script validates the paired design, decodes the original group labels and compares the rebuilt SHA-256 with the reviewer table before saving a new file.

## Requesting controlled materials

The [download entry](../docs/data-access.md) gives the raw-recording URL and code. If you need the matching historical training receipts, model checkpoints or rights documentation, contact Brainintelli Research Lab at **research@brainintelli.com**. Include the requested archive or recording digest, intended use and contact details. The optional local command `python -m src.reproduce access-request` prepares a template in a fresh run directory. Send completed requests privately.
