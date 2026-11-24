"""One-use held-out assessment, authorized by frozen validation evidence."""
from __future__ import annotations
import argparse
from datetime import datetime,timezone
import gc
import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


FROZEN_FILES=("project.godot","scripts/world_state.gd","scripts/rl_environment_session.gd",
              "godot_tests/rl_environment.gd","agent/rl_contract.py","agent/rl_env.py",
              "agent/m6_contract.py","agent/m6_policy.py","agent/m6_train.py")


def validate_report(report,config):
    if report["profile"]!=config["profile"] or report["horizon"]!=config["horizon"]:
        raise ValueError("Validation report conditions differ")
    if report.get("greedy")!="argmax verb then conditional target":
        raise ValueError("Validation decoding differs")
    episodes=report["episodes"]
    if len(episodes)!=4 or {e["seed"] for e in episodes}!=set(config["validation_seeds"]):
        raise ValueError("Validation report must contain all four distinct locked seeds")
    for episode in episodes:
        ticks=episode["survival_ticks"]
        if not isinstance(ticks,(int,float)) or not math.isfinite(ticks) or not 0<ticks<=config["horizon"]:
            raise ValueError("Invalid survival ticks")
        sources=episode["source_counts"]
        if sources.get("model")!=episode["decisions"] or any(v for k,v in sources.items() if k!="model"):
            raise ValueError("Non-model or unaccounted validation action")
    mean=sum(e["survival_ticks"] for e in episodes)/len(episodes)
    if not math.isclose(mean,report["mean_survival_ticks"],rel_tol=0,abs_tol=1.e-9):
        raise ValueError("Validation mean differs from episode evidence")
    if report["alive_at_horizon"]!=sum(bool(e["alive_at_horizon"]) for e in episodes):
        raise ValueError("Validation survivor count differs")
    if not isinstance(report.get("actor_sha256"),str) or len(report["actor_sha256"])!=64:
        raise ValueError("Missing validation actor identity")
    return mean


def authorize(request):
    """No observations or model inference occurs until all validation gates pass."""
    baseline=Path(request["baseline_run"]).resolve()
    selected=Path(request["selected_run"]).resolve()
    registration=read(baseline/"preregistration.json")
    config=registration["config"]
    current=read(selected/"preregistration.json")
    if config["resume"] is not None:
        raise ValueError("Baseline must be fresh base, never a resumed checkpoint")
    for key in ("model","profile","horizon","validation_seeds","test_seeds","max_tokens","seed"):
        if config[key]!=current["config"][key]:raise ValueError("Comparison conditions differ: "+key)
    for registration_item in (registration,current):
        c=registration_item["config"]
        train=set(range(c["train_seed_start"],c["train_seed_start"]+c["train_seed_count"]))
        validation=set(c["validation_seeds"]);test=set(c["test_seeds"])
        if len(c["validation_seeds"])!=4 or len(validation)!=4 or len(c["test_seeds"])!=20 or len(test)!=20:
            raise ValueError("Locked protocol requires 4 validation and 20 test seeds")
        if train&validation or train&test or validation&test:
            raise ValueError("Train/validation/test seed partitions overlap")
        if c["profile"]!="open1" or c["horizon"]!=900:raise ValueError("Locked world conditions changed")
        for file in FROZEN_FILES:
            sha=hashlib.sha256((Path(c["project"])/file).read_bytes()).hexdigest()
            if registration["source_sha256"][file]!=sha or current["source_sha256"][file]!=sha:
                raise ValueError("Frozen environment/policy implementation changed: "+file)
    before=read(baseline/"before.json")
    validate_report(before,config)
    validations=[json.loads(line) for line in (selected/"validation.jsonl").read_text().splitlines() if line]
    if not validations or len({v["update"] for v in validations})!=len(validations):
        raise ValueError("Missing or duplicate validation checkpoints")
    for row in validations:
        report=read(selected/f"validation_{row['update']:04d}.json")
        mean=validate_report(report,config)
        if (not math.isclose(mean,row["mean_survival_ticks"],rel_tol=0,abs_tol=1.e-9)
            or row["survival_ticks"]!=[e["survival_ticks"] for e in report["episodes"]]
            or row["alive_at_horizon"]!=report["alive_at_horizon"]):
            raise ValueError("Validation summary differs from complete report")
        if Path(row["checkpoint"]).resolve()!=selected/f"checkpoint_{row['update']:04d}":
            raise ValueError("Validation checkpoint outside selected run")
    winner=max(validations,key=lambda row:(row["mean_survival_ticks"],-row["update"]))
    checkpoint=Path(request["checkpoint"]).resolve()
    if checkpoint!=Path(winner["checkpoint"]).resolve():raise ValueError("Checkpoint is not the predeclared validation winner")
    summary=read(selected/"training_summary.json")
    if (Path(summary["selected_checkpoint"]).resolve()!=checkpoint
        or not math.isclose(summary["best_validation_survival"],winner["mean_survival_ticks"],rel_tol=0,abs_tol=1.e-9)):
        raise ValueError("Training summary and validation selection differ")
    if winner["mean_survival_ticks"]<1.15*before["mean_survival_ticks"]:
        raise ValueError("Validation improvement is below 15%; held-out seeds remain sealed")
    if len(config["test_seeds"])!=20 or len(set(config["test_seeds"]))!=20:
        raise ValueError("The locked protocol requires 20 distinct test seeds")
    return config,before,winner


def main():
    import torch
    from agent.m6_contract import VERSION
    from agent.m6_policy import Policy
    from agent.m6_train import evaluate,write
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--request",required=True);args=p.parse_args()
    request=read(args.request)
    config,before,winner=authorize(request)
    checkpoint=Path(request["checkpoint"])
    adapter_sha=hashlib.sha256((checkpoint/"adapter/adapter_model.safetensors").read_bytes()).hexdigest()
    selected_report=read(Path(request["selected_run"])/f"validation_{winner['update']:04d}.json")
    ledger=Path(request["test_ledger"])
    output=Path(request["output"])
    if output.exists():raise FileExistsError("Test output already exists")
    # Exclusive creation prevents accidentally turning the hold-out into another validation set.
    with ledger.open("x",encoding="utf-8") as stream:
        json.dump(dict(status="started",request=request,validation_baseline=before["mean_survival_ticks"],
            selection=winner,seeds=config["test_seeds"],adapter_sha256=adapter_sha,
            candidate_actor_sha256=selected_report["actor_sha256"],created_utc=datetime.now(timezone.utc).isoformat()),stream,indent=2)
    output.mkdir(parents=True)
    write(output/"authorization.json",dict(request=request,config=config,selection=winner))
    torch.set_num_threads(8)
    c=SimpleNamespace(**config)
    base=Policy(c.model,seed=c.seed,max_tokens=c.max_tokens,gradient_checkpointing=c.gradient_checkpointing)
    if base.digest()!=before["actor_sha256"]:raise ValueError("Fresh-base identity differs from validation baseline")
    base_report=evaluate(base,c,c.test_seeds,output/"base.json")
    base_report["contract"]=VERSION;write(output/"base.json",base_report)
    del base;gc.collect();torch.cuda.empty_cache()
    candidate=Policy(c.model,checkpoint=request["checkpoint"],seed=c.seed,max_tokens=c.max_tokens,
                     gradient_checkpointing=c.gradient_checkpointing)
    if (hashlib.sha256((checkpoint/"adapter/adapter_model.safetensors").read_bytes()).hexdigest()!=adapter_sha
        or candidate.digest()!=selected_report["actor_sha256"]):
        raise ValueError("Candidate changed after test authorization")
    candidate_report=evaluate(candidate,c,c.test_seeds,output/"candidate.json")
    candidate_report["contract"]=VERSION;write(output/"candidate.json",candidate_report)
    state=read(ledger);state.update(status="completed",finished_utc=datetime.now(timezone.utc).isoformat())
    write(ledger,state);write(output/"complete.json",state)


if __name__=="__main__":main()
