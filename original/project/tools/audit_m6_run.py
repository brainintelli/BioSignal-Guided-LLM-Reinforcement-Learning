"""Deterministically replay M6 logs; never substitute scripted model evaluation."""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from agent.m6_contract import encode, target_messages, SurvivalReward
from agent.rl_contract import transition_at_horizon
from agent.rl_env import GodotEnv
from agent.m6_stochastic_eval import DECODING,REPLICATES,episode_seed,validate_probabilities
from tools.rl_audit_io import iter_trace_lines,file_sha256


def assert_equivalent(actual, recorded, path="root"):
    if isinstance(actual,dict):
        if not isinstance(recorded,dict) or actual.keys()!=recorded.keys():
            raise AssertionError(path+": dictionary keys differ")
        for k in actual: assert_equivalent(actual[k],recorded[k],path+"."+k)
    elif isinstance(actual,list):
        if not isinstance(recorded,list) or len(actual)!=len(recorded):
            raise AssertionError(path+": list length differs")
        for i,(a,b) in enumerate(zip(actual,recorded)):assert_equivalent(a,b,f"{path}[{i}]")
    elif type(actual) in (float,int) and type(recorded) in (float,int):
        if not math.isfinite(actual) or not math.isfinite(recorded) or abs(actual-recorded)>1.e-9:
            raise AssertionError(path+": numeric mismatch")
    elif type(actual) is not type(recorded) or actual!=recorded:
        raise AssertionError(path+": value/type mismatch")


def validate_step(row, observation, actual_transition, tracker, horizon, stochastic=False):
    """Pure evidence checks used by replay and tampering tests."""
    assert_equivalent(observation,row["observation"],"observation")
    encoded=encode(observation)
    assert_equivalent(encoded["messages"],row["model_input"],"model_input")
    v,t=row["verb_index"],row["target_index"]
    if type(v) is not int or type(t) is not int or not 0<=v<len(encoded["branches"]):
        raise AssertionError("Invalid verb index")
    choices=encoded["branches"][v]
    if not 0<=t<len(choices):raise AssertionError("Invalid target index")
    assert_equivalent(choices[t],row["action"],"action")
    evaluation="scored_horizon" in row
    if evaluation:
        assert_equivalent(horizon,row["scored_horizon"],"scored_horizon")
        if stochastic:
            if row.get("greedy") is not False or row.get("decoding")!=DECODING:
                raise AssertionError("Stochastic decoding identity differs")
        elif row.get("greedy") is not True:raise AssertionError("Assessment must use greedy decoding")
    else:
        expected=target_messages(encoded,v) if len(choices)>1 else None
        assert_equivalent(expected,row["target_input"],"target_input")
    if stochastic:
        if not evaluation:raise AssertionError("Stochastic audit is assessment-only")
        expected=target_messages(encoded,v) if len(choices)>1 else None
        assert_equivalent(expected,row["target_input"],"target_input")
        if row["rng_seed"]!=episode_seed(row["seed"],row["replicate"]):raise AssertionError("RNG seed mismatch")
        validate_probabilities(row["verb_probabilities"],encoded["verb_mask"])
        from agent.m6_contract import LABELS
        validate_probabilities(row["target_probabilities"],[i<len(choices) for i in range(len(LABELS))])
        if row["verb_probabilities"][v]<=0 or row["target_probabilities"][t]<=0:raise AssertionError("Selected zero-probability action")
        for field in ("rng_before_sha256","rng_after_sha256"):
            value=row[field]
            if not isinstance(value,str) or len(value)!=64 or any(c not in "0123456789abcdef" for c in value):
                raise AssertionError("Invalid RNG state SHA")
    if row["source"]!="model":raise AssertionError("Non-model source")
    assert_equivalent(actual_transition,row["transition"],"transition")
    scored=transition_at_horizon(actual_transition,horizon) if evaluation else actual_transition
    reward=tracker.score(scored)
    assert_equivalent(reward,row["reward"],"reward")
    terminal=bool(scored["after"]["dead"])
    truncated=scored["after"]["tick"]>=horizon and not terminal
    assert_equivalent(terminal,row["terminal"],"terminal")
    assert_equivalent(truncated,row["truncated"],"truncated")
    return terminal or truncated


def audit(run, trace, godot, project, stochastic=False):
    registration=json.loads((run/("authorization.json" if stochastic else "preregistration.json")).read_text(encoding="utf-8"))
    c=registration["config"]
    expected_matrix=None
    if stochastic:
        from agent.m6_stochastic_eval import TEST_SEEDS,VALIDATION_SEEDS
        mode=registration["request"]["mode"]
        if mode not in ("validation","test"):raise AssertionError("Invalid assessment mode")
        seeds=TEST_SEEDS if mode=="test" else VALIDATION_SEEDS
        expected_matrix={(s,r) for s in seeds for r in REPLICATES}
    contexts={}; count=0; ticks=0
    try:
        for line in iter_trace_lines(trace):
            if not line.strip():continue
            row=json.loads(line)
            key=(row["seed"],row["replicate"]) if stochastic else (("env",row["env"]) if "env" in row else ("seed",row["seed"]))
            if stochastic and key not in expected_matrix:raise AssertionError("Unexpected world/replicate")
            if key not in contexts:
                contexts[key]={"env":GodotEnv(godot,project),"state":None,"ended":True}
            context=contexts[key]
            if context["ended"]:
                if stochastic and context["state"] is not None:raise AssertionError("Repeated episode after terminal")
                if row["observation"]["tick"]!=0:raise AssertionError("New episode must begin at tick zero")
                context["state"]=context["env"].reset(row["seed"],c["profile"])
                context["tracker"]=SurvivalReward(context["state"]["observation"])
                context["seed"]=row["seed"]
            elif row["seed"]!=context["seed"]:
                raise AssertionError("Seed switched before episode ended")
            observation=context["state"]["observation"]
            if stochastic and "rng_after" in context and context["rng_after"]!=row["rng_before_sha256"]:
                raise AssertionError("Episode RNG state chain broken")
            nxt=context["env"].step(row["action"])
            context["ended"]=validate_step(row,observation,nxt["transition"],context["tracker"],c["horizon"],stochastic)
            if stochastic:context["rng_after"]=row["rng_after_sha256"]
            context["state"]=nxt
            if stochastic and context["ended"]:
                context["env"].close()
            count+=1;ticks+=row["reward"]["elapsed_ticks"]
        if not count:raise AssertionError("Empty trace is not evidence")
        if stochastic:
            if set(contexts)!=expected_matrix or not all(v["ended"] for v in contexts.values()):
                raise AssertionError("Incomplete stochastic episode matrix")
        if trace.name=="transitions.jsonl" and (run/"training_summary.json").exists():
            summary=json.loads((run/"training_summary.json").read_text())
            assert_equivalent(summary["decisions"],count,"summary.decisions")
            assert_equivalent(summary["world_ticks"],ticks,"summary.world_ticks")
        return dict(passed=True,replayed_decisions=count,scored_world_ticks=ticks,
                    stochastic_mode=stochastic,rng_sampling_reproduced=False,
                    scope="Trace replay only; passed does not mean training completed or survival improved",
                    training_summary_present=(run/"training_summary.json").exists(),
                    completion_marker_present=(run/"complete.json").exists(),
                    trace_sha256=file_sha256(trace),
                    checks="observation, both inputs, full action, raw transition, full reward/config/discount/dt, terminal/truncated",
                    numerical_tolerance=1.e-9,is_new_model_rollout=False)
    finally:
        for context in contexts.values():context["env"].close()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run",type=Path,required=True);p.add_argument("--trace",type=Path)
    p.add_argument("--godot",required=True);p.add_argument("--project",default=str(Path(__file__).resolve().parents[1]))
    p.add_argument("--output",type=Path)
    p.add_argument("--stochastic",action="store_true",help="Explicit separately registered sampling audit; greedy default stays strict")
    a=p.parse_args();trace=a.trace or a.run/"transitions.jsonl"
    result=audit(a.run,trace,a.godot,a.project,a.stochastic)
    output=a.output or trace.with_name(trace.stem+"_replay_audit.json")
    output.write_text(json.dumps(result,indent=2),encoding="utf-8");print(json.dumps(result,indent=2))


if __name__=="__main__":main()
