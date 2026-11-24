"""Run the registered 22554 paired reward comparison in the original GPU environment."""
import argparse
import fcntl
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path


def read(path):
    return json.loads(path.read_bytes())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    temporary = path.with_suffix(".part")
    temporary.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def run_seed(root, source, budget_path, arm, destination):
    import train_long_recorded as trainer

    registry = read(root / "protocol.json")
    budget = read(budget_path)
    beta = budget["r2_beta"]
    if (sha(Path(trainer.__file__)) != registry["trainer_sha256"]
            or trainer.SCALE_FRACTION != registry["scale_fraction"]
            or trainer.REFERENCE_PER_TICK != registry["reference_per_tick"]
            or trainer.GAMMA != registry["gamma_tick"] or beta not in registry["weights"]):
        raise ValueError("Trainer or scale differs from the registered comparison")
    original = trainer.decisions

    def checked_steps(*args, **kwargs):
        for row, trace, observation, prior in original(*args, **kwargs):
            native = row["r1"]
            ticks = row["duration"]
            phase = row["recorded_phase"]
            if phase not in ("pre", "post") or ticks < 1 or not math.isfinite(native):
                raise ValueError("Missing native reward, event phase or elapsed ticks")
            sign = 0 if row["terminal"] else (1 if phase == "pre" else -1)
            if row["r2"] != (sign if beta and sign else None):
                raise ValueError("Delivery label differs from the training input")
            # Calculate the audit reference independently of the trainer's expm1 formula.
            duration = math.fsum(trainer.GAMMA ** k for k in range(ticks))
            scaled = sign * trainer.SCALE_FRACTION * min(abs(native), trainer.REFERENCE_PER_TICK * duration)
            contribution = row["r2_contribution"]
            if (not math.isclose(contribution, beta * scaled, rel_tol=1e-12, abs_tol=1e-15)
                    or abs(contribution) > registry["max_fraction"] * abs(native) + 1e-15
                    or row["reward"] != native + contribution
                    or (native != 0 and (row["reward"] > 0) != (native > 0))):
                raise ValueError("Calibrated contribution or PPO reward differs")
            # The original row goes to PPO. Write unambiguous names in the evidence trace.
            trace.update(r2_event_sign=sign, r2=scaled, r2_scaled=scaled)
            yield row, trace, observation, prior

    trainer.decisions = checked_steps
    trainer.run(argparse.Namespace(data=source / "replay.json", prime=source / "preceding-quiet.json",
                                  settings=source / "settings.json", budget=budget_path, arm=arm,
                                  out=destination))


def batch(root, host):
    reg = read(root / "protocol.json")
    source = Path.home() / reg["source_directory"]
    template = read(source / "budget-beta_010.json")
    if (sha(source / "code/train_long_recorded.py") != reg["trainer_sha256"]
            or sha(source / "replay.json") != reg["replay_sha256"]
            or template["game_eval_seeds"] != reg["worlds"]
            or template["training_seeds"] != reg["seeds"]
            or (template["train_steps"], template["rollout_batch"], template["eval_horizon"]) != (48, 12, 6)):
        raise ValueError("Original data split, training code or decision budget differs")
    seeds = reg["hosts"][host]
    env = dict(os.environ)
    env.update(read(Path.home() / "mea-state2-r2-20261118/run02/environment.json"))
    env["PYTHONPATH"] = str(source / "code") + os.pathsep + env.get("PYTHONPATH", "")
    python = Path.home() / "mea-r2-22554/runtime_20261116_01/assets/python/bin/python"
    conditions = [(beta, arm) for beta in reg["weights"] for arm in reg["arms"]]
    completed = 0
    for seed in seeds:
        offset = reg["seeds"].index(seed) % len(conditions)
        for beta, arm in conditions[offset:] + conditions[:offset]:
            tag = ("off" if beta == 0 else "cap02") + "-" + arm
            run = root / "runs" / tag / str(seed)
            run.mkdir(parents=True, exist_ok=True)
            model = run / "training" / str(seed)
            budget = dict(template, training_seeds=[seed], r2_beta=beta, beta_sweep=reg["weights"],
                          primary=reg["primary"], secondary=reg["secondary"], tick_horizon=6,
                          eval_rule=reg["evaluation"], cohort=reg["study"], reward_mix=reg["reward_mix"])
            budget_path = run / "budget.json"
            if budget_path.exists() and read(budget_path) != budget:
                raise ValueError(f"Budget changed for {run}")
            save(budget_path, budget)
            if not (run / "done.json").exists():
                if (run / "training").exists():
                    # A partial attempt is retained as evidence. Only a completed attempt is reusable.
                    (run / "training").rename(run / f"partial-{time.time_ns()}")
                save(root / "status.json", dict(complete=completed, current=[tag, seed], started=time.time()))
                cmd = [str(python), "-u", __file__, "seed", str(root), str(source), str(budget_path),
                       arm, str(run / "training")]
                with (run / "process.log").open("ab") as log:
                    subprocess.run(cmd, env=env, cwd=root, stdout=log, stderr=log, check=True)
            result = read(model / "result.json")
            manifest = model / "checkpoint/manifest.json"
            if (result["training_seed"] != seed or result["arm"] != arm
                    or not result["checkpoint"]["reload_verified"] or result["update"]["world_size"] != 8
                    or result["training_steps"] != 48 or result["update_count"] != 4
                    or [g["seed"] for g in result["eval"]] != reg["worlds"]
                    or sha(manifest) != result["checkpoint"]["manifest_sha256"]
                    or read(model / "registration.json")["budget"] != budget):
                raise ValueError(f"Checkpoint or split validation failed for {run}")
            save(run / "done.json", dict(seed=seed, arm=arm, beta=beta, host=host,
                                        manifest=sha(manifest), registration=sha(model/"registration.json")))
            completed += 1
    save(root / "status.json", dict(complete=completed, finished=time.time()))


if __name__ == "__main__":
    task, *args = sys.argv[1:]
    if task == "seed":
        run_seed(Path(args[0]), Path(args[1]), Path(args[2]), args[3], Path(args[4]))
    elif task == "batch":
        folder = Path(args[0])
        with (folder / "batch.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            try:
                batch(folder, args[1])
            except Exception as err:
                save(folder / "error.json", {"type":type(err).__name__, "message":str(err)})
                raise
    else:
        raise SystemExit("Expected batch or seed")
