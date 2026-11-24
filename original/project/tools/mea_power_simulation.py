"""Cluster-bootstrap power simulation for the 22021 stress paired data.

The simulation keeps the observed seed and world pairing structure. A future
study is represented by sampling training seeds and evaluation worlds from the
observed empirical clusters, then shifting the sampled paired differences to a
registered planning effect. Training seeds are the inferential clusters.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from statistics import NormalDist

import numpy as np
import pandas as pd


BASE = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = BASE / "output/mea_predictability_22021_20261119T042416Z/offline-stress-three-arms/stress/paired.csv"
DEFAULT_OUTPUT = BASE / "output/mea_predictability_22021_20261119T042416Z/power-simulation"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_ints(value: str) -> list[int]:
    return [int(item) for item in value.split(",") if item.strip()]


def metric_difference(frame: pd.DataFrame, scene: str, contrast: str, metric: str) -> np.ndarray:
    left, right = contrast.split("-")
    subset = frame.loc[frame["scene"] == scene]
    seeds = sorted(subset["seed"].unique())
    worlds = sorted(subset["world"].unique())
    matrix = np.empty((len(seeds), len(worlds)), dtype=float)
    for row, seed in enumerate(seeds):
        rows = subset.loc[subset["seed"] == seed].set_index("world")
        matrix[row, :] = (
            rows[f"{left}_{metric}"].reindex(worlds).to_numpy(dtype=float)
            - rows[f"{right}_{metric}"].reindex(worlds).to_numpy(dtype=float)
        )
    return matrix


def simulated_seed_means(
    matrix: np.ndarray,
    n_seeds: int,
    n_worlds: int,
    reps: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Return one mean per sampled training seed for each simulated study."""
    base_seeds, base_worlds = matrix.shape
    result = np.empty((reps, n_seeds), dtype=float)
    for column in range(n_seeds):
        sampled_seeds = rng.integers(0, base_seeds, size=reps)
        sampled_worlds = rng.integers(0, base_worlds, size=(reps, n_worlds))
        result[:, column] = matrix[sampled_seeds[:, None], sampled_worlds].mean(axis=1)
    return result


def evaluate_power(
    matrices: dict[str, np.ndarray],
    target: float,
    direction: int,
    n_seeds: int,
    n_worlds: int,
    reps: int,
    critical_z: float,
    seed_fraction: float,
    rng: np.random.Generator,
) -> dict[str, float]:
    """Estimate per-contrast and joint power for the registered rule."""
    signed = {}
    for name, matrix in matrices.items():
        signed[name] = direction * matrix
        observed = signed[name].mean()
        signed[name] = signed[name] + (target - observed)

    means = {
        name: simulated_seed_means(matrix, n_seeds, n_worlds, reps, rng)
        for name, matrix in signed.items()
    }
    minimum_favorable = math.ceil(seed_fraction * n_seeds)
    successes: dict[str, np.ndarray] = {}
    for name, seed_means in means.items():
        study_mean = seed_means.mean(axis=1)
        standard_error = seed_means.std(axis=1, ddof=1) / math.sqrt(n_seeds)
        lower_bound = study_mean - critical_z * standard_error
        favorable = (seed_means > 0).sum(axis=1) >= minimum_favorable
        successes[name] = (lower_bound > 0) & favorable

    joint = np.logical_and.reduce(list(successes.values()))
    return {
        **{f"power_{name}": float(values.mean()) for name, values in successes.items()},
        "power_joint": float(joint.mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--reps", type=int, default=4000)
    parser.add_argument("--seed-sizes", default="8,12,16,24,32")
    parser.add_argument("--world-sizes", default="32,64,96")
    parser.add_argument("--rng-seed", type=int, default=22021119)
    args = parser.parse_args()

    if args.reps < 1000:
        raise ValueError("--reps must be at least 1000")
    if any(value < 2 for value in parse_ints(args.seed_sizes)):
        raise ValueError("all seed sizes must be at least 2")

    frame = pd.read_csv(args.input)
    scenes = ["low_health", "thirst", "hunger"]
    contrasts = ["ordered-masked", "ordered-shuffled"]
    critical_z = NormalDist().inv_cdf(1.0 - 0.008333333333333333 / 2.0)
    seed_fraction = 0.75
    rng = np.random.default_rng(args.rng_seed)

    # Primary endpoints map each pressure to the metric that directly answers
    # the study question. Resource timing is included as a registered secondary.
    endpoints = [
        ("low_health", "cells", 1, [3.0, 5.0, 8.0]),
        ("thirst", "survived", 1, [0.03, 0.05, 0.10]),
        ("hunger", "survived", 1, [0.03, 0.05, 0.10]),
        ("thirst", "drink_wait", -1, [1.0, 2.0, 4.0]),
        ("hunger", "eat_wait", -1, [1.0, 2.0, 4.0]),
    ]

    rows: list[dict[str, object]] = []
    pilot: list[dict[str, object]] = []
    for scene, metric, direction, targets in endpoints:
        matrices = {
            contrast: metric_difference(frame, scene, contrast, metric)
            for contrast in contrasts
        }
        for contrast, matrix in matrices.items():
            pilot.append(
                {
                    "scene": scene,
                    "metric": metric,
                    "contrast": contrast,
                    "observed_difference": float(matrix.mean()),
                    "seed_mean_sd": float(
                        matrix.mean(axis=1).std(ddof=1)
                    ),
                    "rows": int(matrix.size),
                }
            )

        for target in targets:
            for n_seeds in parse_ints(args.seed_sizes):
                for n_worlds in parse_ints(args.world_sizes):
                    power = evaluate_power(
                        matrices,
                        target=target,
                        direction=direction,
                        n_seeds=n_seeds,
                        n_worlds=n_worlds,
                        reps=args.reps,
                        critical_z=critical_z,
                        seed_fraction=seed_fraction,
                        rng=rng,
                    )
                    rows.append(
                        {
                            "scene": scene,
                            "metric": metric,
                            "beneficial_effect": target,
                            "direction": "increase" if direction > 0 else "decrease",
                            "n_training_seeds": n_seeds,
                            "n_worlds_per_seed": n_worlds,
                            "reps": args.reps,
                            **power,
                        }
                    )

    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "power_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    metadata = {
        "input": str(args.input.resolve()),
        "input_sha256": sha256(args.input),
        "output": str(args.output.resolve()),
        "base_rows": int(len(frame)),
        "base_training_seeds": int(frame["seed"].nunique()),
        "base_worlds_per_scene": int(frame.groupby("scene")["world"].nunique().min()),
        "scenes": scenes,
        "contrasts": contrasts,
        "critical_z": critical_z,
        "interval_alpha": 0.008333333333333333,
        "favorable_seed_fraction": seed_fraction,
        "resampling": "sample observed training-seed clusters and worlds with replacement, preserving paired contrasts",
        "rng_seed": args.rng_seed,
        "reps": args.reps,
        "pilot": pilot,
    }
    (args.output / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    summary = pd.DataFrame(rows)
    selected = summary.loc[
        (summary["n_training_seeds"] == 32)
        & (summary["n_worlds_per_seed"] == 64)
        & (summary["beneficial_effect"].isin([5.0, 0.05, 2.0]))
    ]
    lines = [
        "# 22021 压力场景功效模拟",
        "",
        f"输入为 `{args.input.resolve()}`，SHA-256 为 `{metadata['input_sha256']}`。",
        "模拟以训练种子为推断簇，在每个种子内有放回抽取评估地图，保留真实顺序相对屏蔽和相对分块打乱的配对结构。",
        f"每个条件运行 {args.reps:,} 次，使用双侧 99.167% 区间的正向下界和至少 75% 训练种子有利作为单项成功规则。",
        "功效中的目标效应是登记时可接受的最小有利差异，不是对现有结果的重新显著性检验。",
        "",
        "## 现有配对数据的簇变异",
        "",
        "| 场景 | 指标 | 对照 | 观察到的差 | 训练种子均值标准差 |",
        "|---|---|---|---:|---:|",
    ]
    for item in pilot:
        lines.append(
            f"| {item['scene']} | {item['metric']} | {item['contrast']} | {item['observed_difference']:.4f} | {item['seed_mean_sd']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## 32 个训练种子、每个种子 64 张地图",
            "",
            "| 场景 | 指标 | 目标有利差 | 对屏蔽单项功效 | 对打乱单项功效 | 两项同时达标 |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for _, item in selected.iterrows():
        lines.append(
            f"| {item['scene']} | {item['metric']} | {item['beneficial_effect']} | {item['power_ordered-masked']:.3f} | {item['power_ordered-shuffled']:.3f} | {item['power_joint']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## 解释",
            "",
            "低生命探索若把 +5 格作为最小有利差异，32 个训练种子和 64 张地图的两项联合功效约为 0.61。若把 +8 格作为稳定改善的目标，联合功效约为 0.995。",
            "濒渴和濒饿存活若把 +5 个百分点作为最小有利差异，32 个训练种子和 64 张地图的两项联合功效约为 0.91 和 0.79。",
            "补水或进食等待的功效表用于次要终点，登记时不把它们与三个主要生存或探索终点合并成成功条件。",
            "",
            "完整逐条件结果见 `power_summary.csv`，模拟参数和输入哈希见 `metadata.json`。",
        ]
    )
    (args.output / "功效模拟报告.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
