"""Small fixed-state capability probes. These do NOT prove autonomous survival."""
import argparse
import copy
import hashlib
import json
import math
import random
from datetime import datetime, timezone
from pathlib import Path

from .broker import DecisionEngine, add_model_arguments, engine_from_args


def cases(seed=41001):
    rng = random.Random(seed)
    base = {
        "schema_version": 1, "tick": 0, "day": 1, "phase": "day",
        "player": {"health": 90, "hunger": 80, "thirst": 80, "energy": 80, "x": 0, "y": 0},
        "inventory": {"wood": 0, "stone": 0, "berries": 2}, "nearby": [], "threats": [],
        "camp": {"exists": False, "x": 0, "y": 0},
        "allowed_actions": [{"verb": "wait"}],
    }
    result = []
    def add(name, obs, expected):
        # Random action ordering discourages positional selection; expected stays evaluator-only.
        rng.shuffle(obs["allowed_actions"])
        result.append({"name": name, "observation": obs, "expected": expected})
    obs = copy.deepcopy(base)
    obs["player"]["hunger"] = 12
    obs["allowed_actions"] += [{"verb": "eat"}, {"verb": "rest"}]
    add("urgent_food", obs, {"verb": "eat"})
    obs = copy.deepcopy(base)
    obs["player"]["thirst"] = 10
    obs["allowed_actions"] += [{"verb": "drink"}, {"verb": "eat"}]
    add("urgent_water", obs, {"verb": "drink"})
    obs = copy.deepcopy(base)
    obs["player"]["energy"] = 8
    obs["camp"]["exists"] = True
    obs["allowed_actions"] += [{"verb": "rest"}, {"verb": "eat"}]
    add("safe_recovery", obs, {"verb": "rest"})
    obs = copy.deepcopy(base)
    obs["threats"] = [{"id": "boar", "distance": 1}]
    obs["allowed_actions"] += [{"verb": "flee"}, {"verb": "rest"}]
    add("immediate_threat", obs, {"verb": "flee"})
    obs = copy.deepcopy(base)
    obs["phase"] = "dusk"
    obs["inventory"].update(wood=6, stone=3)
    obs["allowed_actions"] += [{"verb": "build_camp"}, {"verb": "rest"}]
    add("shelter_before_night", obs, {"verb": "build_camp"})
    obs = copy.deepcopy(base)
    target = f"pond_{rng.randrange(1000, 9999)}"
    obs["player"]["thirst"] = 20
    obs["nearby"] = [{"id": target, "kind": "water", "distance": 3, "x": 3, "y": 0, "available": True},
                     {"id": "tree", "kind": "tree", "distance": 2, "x": 2, "y": 0, "available": True}]
    obs["allowed_actions"] += [{"verb": "move", "target_id": target}, {"verb": "move", "target_id": "tree"}]
    add("walk_to_water_not_unavailable_drink", obs, {"verb": "move", "target_id": target})
    obs = copy.deepcopy(base)
    target = f"berry_{rng.randrange(1000, 9999)}"
    obs["player"]["hunger"] = 20
    obs["inventory"]["berries"] = 0
    obs["nearby"] = [{"id": target, "kind": "berries", "distance": 1, "x": 1, "y": 0, "available": True}]
    obs["allowed_actions"] += [{"verb": "gather", "target_id": target}, {"verb": "rest"}]
    add("gather_before_eating_empty_inventory", obs, {"verb": "gather", "target_id": target})
    obs = copy.deepcopy(base)
    add("only_legal_wait", obs, {"verb": "wait"})
    return result


def evaluate(engine: DecisionEngine, repeats=3, seed=41001):
    if repeats < 1:
        raise ValueError("repeats must be positive")
    rows = []
    suite_hash = hashlib.sha256(json.dumps(cases(seed), sort_keys=True).encode()).hexdigest()
    for repeat in range(repeats):
        for case in cases(seed + repeat):
            result = engine.decide(copy.deepcopy(case["observation"]))
            correct = result["action"] == case["expected"]
            rows.append({"case": case["name"], "repeat": repeat, "expected": case["expected"],
                         "correct": correct, "model_correct": correct and result["source"] == "model",
                         "decision": result})
    n = len(rows)
    calls = n if engine.model else 0
    latencies = sorted(r["decision"]["latency_ms"] for r in rows)
    return {
        "created_utc": datetime.now(timezone.utc).isoformat(), "scope": "fixed_state_probes_only",
        "mode": "model" if engine.model else "baseline", "model": engine.model,
        "assisted_decoding": engine.constrained,
        "observation_format": engine.observation_format,
        "prompt_version": engine.prompt_version, "prompt_sha256": engine.prompt_sha256,
        "raw_valid_definition": "Unmodified API message.content after server reasoning separation, not all generated tokens",
        "quantization": engine.quantization, "seed": seed, "suite_first_repeat_sha256": suite_hash,
        "observations": n, "model_calls": calls, "long_horizon_tested": False,
        "raw_valid_rate": sum(r["decision"]["raw_valid"] is True for r in rows) / calls if calls else None,
        "model_correct_rate": sum(r["model_correct"] for r in rows) / calls if calls else None,
        "baseline_correct_rate": sum(r["correct"] for r in rows) / n if not calls else None,
        "fallback_rate": sum(r["decision"]["source"] == "fallback" for r in rows) / n,
        "latency_p50_ms": latencies[(n - 1) // 2],
        "latency_p95_ms": latencies[math.ceil(n * .95) - 1],
        "limitations": ["No world rollout, memory, tool-chain or multi-day survival is measured.",
                        "Legacy immediate_threat case uses reserved flee; it is not a live world action.",
                        "Public probes are calibration fixtures, not a held-out generalization benchmark.",
                        "Do not unlock difficulty from this report alone; see docs/CAPABILITY_GATES.md."],
        "rows": rows,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_model_arguments(parser)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=41001)
    parser.add_argument("--output", default="output/agent/probe_report.json")
    args = parser.parse_args()
    report = evaluate(engine_from_args(args), args.repeats, args.seed)
    report["inference_settings"] = {"temperature": 0, "max_tokens": 80, "timeout_s": args.timeout,
                                    "disable_thinking_requested": args.disable_thinking,
                                    "assisted_decoding": args.constrained}
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "rows"}, ensure_ascii=False, indent=2))
    print("Report: " + str(path))


if __name__ == "__main__":
    main()
