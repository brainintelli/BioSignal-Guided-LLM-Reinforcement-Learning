"""Within-seed, within-map comparisons of the first 22021 acquisition."""

from __future__ import annotations

from collections import Counter
from statistics import mean, median, stdev
from typing import Any

from .files import Release


def paired_summary(control: list[int], values: list[int]) -> dict[str, Any]:
    if len(control) != len(values) or len(control) < 2:
        raise ValueError("Paired arms must include the same observations and at least two pairs.")
    differences = [a - c for a, c in zip(values, control, strict=True)]
    if min(control + values) < 0:
        raise ValueError("Explored cell counts must be nonnegative.")
    ordered = sorted(differences)
    drop = len(differences) // 20
    wins = sum(value > 0 for value in differences)
    losses = sum(value < 0 for value in differences)
    ranks: dict[int, float] = {}
    below = 0
    for magnitude, count in sorted(Counter(abs(d) for d in differences if d).items()):
        ranks[magnitude] = below + (count + 1) / 2
        below += count
    rplus = sum(ranks[d] for d in differences if d > 0)
    rminus = sum(ranks[-d] for d in differences if d < 0)
    sd = stdev(differences)
    effect = mean(differences)
    baseline = mean(control)
    return {
        "pairs": len(differences),
        "mean_control": baseline,
        "mean_arm": mean(values),
        "mean_difference": effect,
        "median_difference": median(differences),
        "percent_of_control": 100 * effect / baseline if baseline else None,
        "wins": wins,
        "ties": len(differences) - wins - losses,
        "losses": losses,
        "paired_dz": effect / sd if sd else None,
        "rank_biserial": (rplus - rminus) / (rplus + rminus) if rplus + rminus else None,
        "symmetric_trim_mean": mean(ordered[drop : len(ordered) - drop]),
        "upper_trim_mean": mean(ordered[: len(ordered) - drop]),
        "removed_from_upper_tail": drop,
    }


def load_pairs(release: Release) -> list[dict[str, str]]:
    rows = release.rows("exploration")
    spec = release.spec["short"]
    expected = {(s, m) for s in range(1, spec["seeds"] + 1) for m in range(1, spec["maps"] + 1)}
    observed = {(int(r["seed_index"]), int(r["map_index"])) for r in rows}
    if observed != expected or len(observed) != len(rows):
        raise ValueError("Paired table is incomplete or contains duplicate seed–map keys.")
    return sorted(rows, key=lambda r: (int(r["seed_index"]), int(r["map_index"])))


def exploration(release: Release) -> dict[str, Any]:
    rows = load_pairs(release)
    out: dict[str, Any] = {}
    for budget in ("d6", "t6"):
        control = [int(r[f"all_zero_{budget}"]) for r in rows]
        arms = {
            arm: [int(r[f"{arm}_{budget}"]) for r in rows] for arm in ("intact", "unstimulated")
        }
        out[budget] = {
            arm: paired_summary(control, observations) for arm, observations in arms.items()
        }
        out[budget]["all_three_equal"] = sum(
            c == a == b for c, a, b in zip(control, *arms.values(), strict=True)
        )
    return out
