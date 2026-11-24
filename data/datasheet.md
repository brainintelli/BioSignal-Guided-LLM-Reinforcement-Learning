# Datasheet for the replay analyses

## Collection and sampling units

These data connect archived microelectrode-array recordings to repeated game evaluations. Training seed and game map identify a matched software comparison. Both arms of a comparison use the same software conditions. Maps recur across seeds. Some seeds and maps also recur across acquisition days. Their multiplicity does not add biological samples.

On chip 22021, the first recording supplied a 6-decision trial with 16 training seeds evaluated on the same 48 maps. The intact group received pre-stimulation spontaneous activity, the unstimulated group received measured activity in a later zero-command period, and the all-zero group received a constant vector. Prior stimulation, time and biological state may differ between recording periods. The shuffled group belonged to the adverse-start experiment on the second sample. It used within-trial blocks. On chip 22554, the first-day shuffle reassigned training-trial pairings and both recording-connected groups received ordered input during evaluation. The native game-reward-only comparator there had a different input architecture.

## Measurement and live-training qualification

The observed channel pass rates across the first chip’s warm-up windows were 75.5%, 74.8% and 71.4%, below the 95% live-training gate. The actual training used offline replay. The contemporaneous reason for switching remains unverified. Game actions provided no feedback to the tissue.

The recorded biphasic command was 199.79 mV per phase. The manufacturer’s suggested starting range for ordinary MaxOne/6WP platinum-black electrodes was 300–600 mV. The numerical command is below that range. The conversion between the guide’s amplitude convention and each commanded phase remains unverified. Under-load electrode waveforms and peaks were not measured.

The first-day fixed readout panel on chip 22554 failed verification before exploratory acquisition. In post-acquisition leave-one-block-out classification, the post-stimulation result was 5/9 correct (exact permutation p ≈ 0.306), below the reference criterion. That acquisition did not invoke the qualification procedure. The retained recordings support engineering checks. Site-specific biological discriminability remains unresolved.

## Outcomes and limitations

In the first short trial, 615/768 intact versus all-zero pairs tied on explored cells. The paired median was 0. The mean was 1.99 cells per game. In a post hoc equal-time sensitivity analysis, the intervals crossed zero when seeds and maps were both resampled. The first-day cross-chip exploration result did not replicate. No stable advantage was detected across the 6 adverse-start primary comparisons. The intervals for regular or irregular stimulation versus the unstimulated group crossed zero across both measured days. Observed survival and resource use characterize game behavior. Subjective fear was not measured.

The adverse-start evaluation reused 64 pilot maps and 8 pilot seeds. The design identifiers on 18,720 receipts changed after evaluation. Independent earlier copies are available for 99. A substitute project snapshot was used when the registered audit game source was unavailable, and the scenario-verification digest did not match the registered value. Accounting for the full historical receipt set and independently establishing the unique pre-unblinding design remain unresolved.

## Contents, processing and uses

The [dictionary](data_dictionary.csv) gives fields, units, domains and missingness. The [machine-readable metadata](croissant.json) links data fields to their source columns. The outcome tables replace original simulation seeds and maps with study-local indices. Pair only within the relevant study, day, scene, seed and map. Feature-window rows include file and event-stream digests. Calibration rows lack standardized input by design. The HDF5 inventory includes files with zero waveform frames or structural anomalies. These entries preserve the archive’s history and are distinguishable in its `kind` column.

Use the software to reproduce paired outcomes, evaluate sensitivity to the resampling unit, and examine the relationship between action duration and score. Recording provenance, a short-horizon difference and offline replay do not identify a tissue-specific mechanism, learning independent of inference-time conditioning, or a live bidirectional system.

## Data access

For the recording export, use the [H5 download and verification procedure](../docs/data-access.md). Training trajectories, evaluation receipts and saved model states can be downloaded through the [public archive guide](../docs/archive-downloads.md). The [data terms](../LICENSE-DATA) cover the study tables and the [training artifact terms](../LICENSE-TRAINING) cover project-owned training and evaluation records.
