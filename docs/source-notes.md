# Source notes for the frozen manuscript

The signed source correction accompanying the Chinese manuscript specifies what could be checked after the review packets were assembled. The English article preserves the frozen text. The files below allow the local reader to inspect the relevant material.

| Manuscript locator | Evidence now included | Command and remaining limit |
|---|---|---|
| Appendix B, paragraph P179, next-day site-file holdout | [Paired outcomes](../data/heldout.csv) and [endpoint means and intervals](../results/heldout.csv). Training seeds are resampled with replacement. The held-out file follows the seed and fold sizes can vary. Map draws are shared across folds and arms. Both budgets use identical draw indices. | `python -m src.reproduce heldout` recomputes all 72 of 72 archived interval endpoints. The historical generator state was not separately saved. |
| Appendix B, paragraph P182, native cutoff scoring | [Native and recorded scores by episode](../data/boundary.csv), [nine scene-by-arm rows](../results/boundary_groups.csv) and [six paired contrast rows](../results/boundary_contrasts.csv). Signed differences are native cutoff recomputation minus the recorded score. | `python -m src.reproduce boundary` recomputes the detail tables using decimal arithmetic. Field-by-field causal attribution remains limited by the absence of paired complete observations at the action end and the cutoff. |
| Final training-budget review | [Recorded model and per-batch budgets](../data/model_budgets.csv). A reviewer counted 93 rows in an earlier statement. The material supplied to that reviewer contained 96, comprising 32 seeds in three groups. | `python -m src.reproduce budgets` checks the 384 batch totals and the recorded 48 decisions, eight policy updates and 64 value updates per model. Original elapsed training time, GPU-hours and the successful processes’ complete environments remain unverified. |

For the acquisition export, see the [original-data download](data-access.md). The [verification guide](../data/README.md) explains how to identify a matching recording or historical program archive by its checksum.
