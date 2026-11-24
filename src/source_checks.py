"""Rebuild the signed terminal-score tables and inspect recorded update budgets."""

from collections import Counter, defaultdict
from decimal import Decimal
from typing import Any

from .files import Release, csv_rows, read_json


def actions(release: Release) -> dict[str, Any]:
    keys = set()
    totals: dict[str, Counter[str]] = defaultdict(Counter)
    for row in release.rows("actions"):
        key = row["arm"], row["seed_index"], row["map_index"]
        if key in keys:
            raise ValueError("Duplicate game in the action table.")
        keys.add(key)
        totals[row["arm"]].update(row["verbs"].split("|"))
    pairs = release.spec["short"]["seeds"] * release.spec["short"]["maps"]
    expected = read_json(release.root / "results/actions.json")
    if set(totals) != set(expected) or len(keys) != len(totals) * pairs:
        raise ValueError("Incomplete action-table arms or pairings.")
    counts = {arm: counter.total() for arm, counter in totals.items()}
    if len(set(counts.values())) != 1:
        raise ValueError("Action totals differ across arms.")
    for arm, verblist in expected.items():
        if any(totals[arm][verb] != count for verb, count in verblist.items()):
            raise ValueError("A plotted action count differs from its archived reference.")
    baseline = totals["all-zero"]
    deltas = {
        arm: {
            verb: 100 * (counter[verb] / counts[arm] - baseline[verb] / counts["all-zero"])
            for verb in expected[arm]
        }
        for arm, counter in totals.items()
        if arm != "all-zero"
    }
    return {
        "scope": "observed action frequencies; descriptive",
        "total_per_arm": counts,
        "observed_counts": dict(totals),
        "difference_pp": deltas,
    }


def boundary(release: Release) -> dict[str, Any]:
    table = release.rows("boundary")
    threshold = Decimal(release.spec["boundary_difference_threshold"])
    sums: dict[tuple[str, str], list[tuple[Decimal, Decimal, int]]] = defaultdict(list)
    paired: dict[tuple[str, str, str], dict[str, Decimal]] = defaultdict(dict)
    for row in table:
        arm = row["arm"]
        key = (row["scene"], row["seed_index"], row["map_index"])
        if arm in paired[key]:
            raise ValueError("Duplicate arm at a boundary-score key.")
        recorded = Decimal(row["r1_recorded"])
        native = Decimal(row["r1_native"])
        paired[key][arm] = native - recorded
        sums[row["scene"], arm].append((recorded, native, int(row["cut"])))
    groups = {}
    for row in csv_rows(release.root / "results/boundary_groups.csv"):
        group = row["scene"], row["arm"]
        obs = sums.pop(group)
        ds = [b - a for a, b, _ in obs]
        above = [d for d in ds if abs(d) > threshold]
        n = len(ds)
        stat = dict(
            n=n,
            cut=sum(c for _, _, c in obs),
            above_threshold=len(above),
            positive=sum(d > 0 for d in above),
            negative=sum(d < 0 for d in above),
            mean_recorded=sum(a for a, _, _ in obs) / n,
            mean_native=sum(b for _, b, _ in obs) / n,
            mean_signed_delta=sum(ds) / n,
            min_signed=min(ds),
            max_signed=max(ds),
        )
        for field, value in stat.items():
            if Decimal(row[field]) != value:
                raise ValueError(f"Boundary-score reference mismatch at {group}, field {field}.")
        groups["/".join(group)] = {k: str(v) for k, v in stat.items()}
    if sums or any(set(v) != {"intact", "all-zero", "shuffled"} for v in paired.values()):
        raise ValueError("Boundary-score table contains an unexpected cohort or pairing.")
    contrasts = {}
    for row in csv_rows(release.root / "results/boundary_contrasts.csv"):
        control = row["contrast"].removeprefix("intact versus ")
        ds = [
            arms["intact"] - arms[control]
            for (scene, _, _), arms in paired.items()
            if scene == row["scene"]
        ]
        effect = sum(ds) / len(ds)
        if int(row["n"]) != len(ds) or Decimal(row["change_in_mean_contrast"]) != effect:
            raise ValueError("The signed mean-contrast change differs from its archived reference.")
        contrasts[row["scene"] + "/" + control] = {
            "n": len(ds),
            "change_in_mean_contrast": str(effect),
        }
    return {"direction": "native minus recorded", "grouped": groups, "paired_changes": contrasts}


def budgets(release: Release) -> dict[str, Any]:
    records = release.rows("budgets")
    spec = release.spec["stress"]
    key = {(int(r["seed_index"]), r["arm"]) for r in records}
    expected = {
        (i, a) for i in range(1, spec["seeds"] + 1) for a in ("all-zero", "intact", "shuffled")
    }
    if key != expected or len(key) != len(records):
        raise ValueError("Missing or repeated model budget.")
    counts = set()
    batches = 0
    for row in records:
        sizes = set()
        for detail, total in (
            ("batch_decisions", "training_decisions"),
            ("batch_ticks", "training_ticks"),
            ("batch_actor_steps", "actor_steps"),
            ("batch_critic_steps", "critic_steps"),
        ):
            x = [int(v) for v in row[detail].split(",")]
            sizes.add(len(x))
            if not x or sum(x) != int(row[total]):
                raise ValueError(f"Batch sum differs from model total in {total}.")
        if len(sizes) != 1:
            raise ValueError("Model has inconsistent numbers of batches.")
        batches += sizes.pop()
        counts.add(
            tuple(int(row[k]) for k in ("training_decisions", "actor_steps", "critic_steps"))
        )
    return {
        "models": len(key),
        "training_seeds": spec["seeds"],
        "batches": batches,
        "recorded_per_model": [
            dict(zip(("decisions", "actor_steps", "critic_steps"), item, strict=True))
            for item in sorted(counts)
        ],
    }
