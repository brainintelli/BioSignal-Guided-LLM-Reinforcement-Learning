"""Archive-matched resampling of paired seed by map arrays."""

from __future__ import annotations

from typing import Any

import numpy as np

from .exploration import load_pairs
from .files import Release


def weights(nseed: int, nmap: int, draws: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    seed_weights = rng.multinomial(nseed, np.full(nseed, 1 / nseed), size=draws)
    map_weights = rng.multinomial(nmap, np.full(nmap, 1 / nmap), size=draws)
    return seed_weights.astype(float), map_weights.astype(float)


def interval(draws: np.ndarray, alpha: float) -> list[float]:
    return np.quantile(draws, [alpha / 2, 1 - alpha / 2]).tolist()


def crossed(matrix: np.ndarray, indices: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
    sw, mw = indices
    return ((sw @ matrix) * mw).sum(axis=1) / matrix.size


def decision_seed_draws(seed_means: np.ndarray, draws: int, initial_seed: int) -> np.ndarray:
    """Historical JavaScript unsigned 32-bit LCG, in sorted seed order."""
    state = initial_seed
    indices = np.empty((draws, len(seed_means)), dtype=np.int32)
    for idx in range(indices.size):
        state = (1664525 * state + 1013904223) & 0xFFFFFFFF
        indices.flat[idx] = (state * len(seed_means)) >> 32
    return seed_means[indices].mean(axis=1)


def grouped(
    rows: list[dict[str, str]], group: str, metric: str, dimensions: tuple[int, int]
) -> np.ndarray:
    matrix = np.full(dimensions, np.nan)
    for row in rows:
        if row["arm"] != group:
            continue
        i, j = int(row["seed_index"]) - 1, int(row["map_index"]) - 1
        if not (0 <= i < dimensions[0] and 0 <= j < dimensions[1]) or not np.isnan(matrix[i, j]):
            raise ValueError(f"Duplicate or invalid pair index for {group}, {metric}.")
        matrix[i, j] = float(row[metric])
    if not np.isfinite(matrix).all():
        raise ValueError(f"Missing or non-finite observations for {group}, {metric}.")
    return matrix


def exploration_intervals(release: Release) -> dict[str, list[float]]:
    rows = load_pairs(release)
    s = release.spec["short"]
    shape = s["seeds"], s["maps"]
    new = weights(*shape, s["draws"], s["bootstrap_seed"])
    archive_tick = weights(*shape, s["draws_tick_archived"], s["seed_tick_archived"])
    order = np.ix_(s["ordering_seeds0"], s["ordering_maps0"])
    output: dict[str, list[float]] = {}
    for arm in ("intact", "unstimulated"):
        for budget in ("d6", "t6"):
            matrix = np.array(
                [int(r[f"{arm}_{budget}"]) - int(r[f"all_zero_{budget}"]) for r in rows]
            ).reshape(shape)
            fixed = new[0] @ matrix.mean(axis=1) / shape[0]
            both = crossed(matrix, new)
            archive = (
                decision_seed_draws(
                    matrix.mean(axis=1),
                    s["draws_decisions_archived"],
                    s["seed_decisions_archived"][arm],
                )
                if budget == "d6"
                else crossed(matrix[order], archive_tick)
            )
            for key, alpha in (
                (("95", 0.05),)
                if budget == "d6"
                else (("95", 0.05), ("98333", 0.05 / s["sensitivity_family"]))
            ):
                for method, sample in (
                    ("archive", archive),
                    ("seed_pct", fixed),
                    ("crossed", both),
                ):
                    output["/".join((budget, arm, key, method))] = interval(sample, alpha)
    return output


def stress(release: Release) -> dict[str, Any]:
    s = release.spec["stress"]
    shape = (s["seeds"], s["maps"])
    samples = weights(*shape, s["draws"], s["bootstrap_seed"])
    all_rows = release.rows("stress")
    output = {}
    for scene, metrics in (
        ("low_health", ["cells"]),
        ("thirst", ["survived", "drink_wait"]),
        ("hunger", ["survived", "eat_wait"]),
    ):
        rows = [r for r in all_rows if r["scene"] == scene]
        for metric in metrics:
            intact = grouped(rows, "intact", metric, shape)
            waiting = metric.endswith("_wait")
            for arm in ("all-zero", "shuffled"):
                delta = intact - grouped(rows, arm, metric, shape)
                if waiting:
                    delta = -delta
                if metric == "survived":
                    delta = delta * 100
                low, high = interval(
                    crossed(delta, samples),
                    0.05 / s["secondary_family" if waiting else "primary_family"],
                )
                nseeds = int((delta.mean(axis=1) > 0).sum())
                output["/".join((scene, metric, arm))] = {
                    "diff": float(delta.mean()),
                    "low": low,
                    "high": high,
                    "favorable_seeds": nseeds,
                    "stable": bool(nseeds >= s["favorable_seed_threshold"] and low > 0),
                }
    return output


def cross_chip(release: Release) -> list[dict[str, Any]]:
    rows = release.rows("cross-chip")
    shape = (len({r["seed_index"] for r in rows}), len({r["map_index"] for r in rows}))
    spec = release.spec["first_day_22554"]
    estimates = []
    for left, right in (("intact", "pure-r1"), ("shuffled", "pure-r1"), ("intact", "shuffled")):
        for metric, field in (("exploration", "cells"), ("r1", "r1")):
            d = grouped(rows, left, field, shape).mean(axis=1) - grouped(
                rows, right, field, shape
            ).mean(axis=1)
            seed = spec["seeds"]["/".join((left, right, metric))]
            low, high = interval(decision_seed_draws(d, spec["draws"], seed), 0.05)
            estimates.append(
                dict(
                    arm_a=left, arm_b=right, metric=metric, diff=float(d.mean()), low=low, high=high
                )
            )
    return estimates


def stimulation(release: Release) -> dict[str, Any]:
    """Each historical batch drew seeds then maps; both metrics share the same draw."""
    all_rows = release.rows("stimulation")
    out: dict[str, Any] = {}
    for day, spec in release.spec["stimulation"].items():
        rows = [r for r in all_rows if r["day"] == day]
        shape = (len({r["seed_index"] for r in rows}), len({r["map_index"] for r in rows}))
        deltas = {}
        for arm in ("regular", "irregular"):
            deltas[arm] = np.stack(
                [
                    grouped(rows, arm, m, shape) - grouped(rows, "unstimulated", m, shape)
                    for m in ("cells", "r1")
                ],
                axis=-1,
            )
        rng = np.random.default_rng(spec["seed"])
        samples: dict[str, list[np.ndarray]] = {arm: [] for arm in deltas}
        for offset in range(0, spec["draws"], 200):
            n = min(200, spec["draws"] - offset)
            i = rng.integers(shape[0], size=(n, shape[0], 1))
            j = rng.integers(shape[1], size=(n, 1, shape[1]))
            for arm, diff in deltas.items():
                samples[arm].append(diff[i, j].mean(axis=(1, 2)))
        out[day] = {}
        for arm, draws in samples.items():
            bounds = np.quantile(np.concatenate(draws), spec["quantiles"], axis=0)
            jointly = int(np.all(deltas[arm].mean(axis=1) > 1e-10, axis=1).sum())
            out[day][arm + "-sham"] = {
                "jointly_positive": jointly,
                "stable": bool(jointly >= np.ceil(0.75 * shape[0]) and np.all(bounds[0] > 0)),
                "intervals": {"new_cells": bounds[:, 0].tolist(), "r1_sum": bounds[:, 1].tolist()},
            }
    return out


def heldout(release: Release) -> list[dict[str, Any]]:
    """Both budgets and arms use shared seed and map draws; folds follow seeds."""
    s = release.spec["day19"]
    shape = s["training_seeds"], s["maps"]
    rng = np.random.default_rng(s["bootstrap_seed"])
    i = rng.integers(0, shape[0], (s["draws"], shape[0], 1))
    j = rng.integers(0, shape[1], (s["draws"], 1, shape[1]))
    rows = release.rows("heldout")
    out = []
    for tick in s["ticks"]:
        subset = [r for r in rows if int(r["tick"]) == tick]
        for metric in (
            "cells",
            "drink",
            "eat",
            "water",
            "food",
            "health",
            "drink_gain",
            "eat_gain",
            "death",
        ):
            matrices = {
                a: grouped(subset, a, metric, shape) for a in ("intact", "shuffled", "pure-r1")
            }
            record: dict[str, Any] = dict(
                tick=tick, endpoint=metric, pairs_per_comparison=shape[0] * shape[1]
            )
            for arm, label in (
                ("intact", "ordered"),
                ("shuffled", "shuffled"),
                ("pure-r1", "pure"),
            ):
                record[label] = float(matrices[arm].mean())
            for c, label in (("shuffled", "shuffled"), ("pure-r1", "pure")):
                d = matrices["intact"] - matrices[c]
                key = "ordered_minus_" + label
                record[key + "_delta"] = float(d.mean())
                record[key + "_ci_lo"], record[key + "_ci_hi"] = interval(
                    d[i, j].mean(axis=(1, 2)), 0.05
                )
            out.append(record)
    return out


def score(release: Release) -> dict[str, Any]:
    """Decompose the short-trial R1 using each observed action's tick duration."""
    rows = release.rows("score")
    spec = release.spec["score"]
    game: dict[tuple[str, int, int], list[float]] = {}
    for row in rows:
        key = (row["arm"], int(row["seed_index"]), int(row["map_index"]))
        start, end = int(row["start_tick"]), int(row["end_tick"])
        entry = game.setdefault(key, [0.0, 0.0, 0.0])
        if end <= start or entry[2] != start:
            raise ValueError("Action ticks must form a contiguous game trajectory.")
        entry[0] += float(row["r1"])
        entry[1] += sum(
            spec["reward_per_tick"] * spec["discount_per_tick"] ** k for k in range(end - start)
        )
        entry[2] = end
    nseed = len({int(r["seed_index"]) for r in rows})
    nmap = len({int(r["map_index"]) for r in rows})
    rng = np.random.default_rng(spec["bootstrap_seed"])
    i = rng.integers(nseed, size=(spec["draws"], nseed, 1))
    j = rng.integers(nmap, size=(spec["draws"], 1, nmap))

    def grid(arm: str) -> np.ndarray:
        return np.array(
            [[game[arm, s, m] for m in range(1, nmap + 1)] for s in range(1, nseed + 1)]
        )

    base = grid("all-zero")
    out = {}
    for arm in ["intact", "unstimulated"]:
        delta = grid(arm) - base
        total, living, ticks = delta.mean(axis=(0, 1))
        residual = delta[:, :, 0] - delta[:, :, 1]
        lo, hi = interval(residual[i, j].mean(axis=(1, 2)), 0.025)
        out[arm] = {
            "total": float(total),
            "living": float(living),
            "mean_ticks": float(ticks),
            "remaining": float(residual.mean()),
            "living_pct": float(100 * living / total),
            "residual_975_lo": lo,
            "residual_975_hi": hi,
        }
    return out
