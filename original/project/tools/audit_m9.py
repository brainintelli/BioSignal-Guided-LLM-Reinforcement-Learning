"""Independent native Godot transition/reward replay and CUDA categorical replay."""
import argparse
import gzip
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from agent.m9_contract import encode,Reward
from agent.m9_env import GodotEnv
from agent.m8s3_training import write,sha
from agent.rl_contract import transition_at_horizon
from tools.audit_m8s3_replay import Comparator
from tools.audit_m8s3_training import verify_generator_draws
from tools.m9_campaign import verify_registration,validate_task

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--campaign",required=True);ap.add_argument("--config",required=True)
    ap.add_argument("--evaluation-only",action="store_true")
    a=ap.parse_args();reg=json.loads(Path(a.campaign).read_text());verify_registration(reg)
    c=json.loads(Path(a.config).read_text());validate_task(reg,c)
    out=Path(c["output"]);comparison=Comparator(1e-12);counts={}
    groups=[] if a.evaluation_only else [sorted(out.glob("rollout_*.jsonl.gz"))]
    groups += [[p] for p in sorted(out.glob("test_*.jsonl.gz" if a.evaluation_only else "*.jsonl.gz")) if not p.name.startswith("rollout_")]
    if not groups or not any(groups):raise ValueError("No audit evidence")
    for paths in groups:
        envs={};states={};trackers={};seeds={};generators={}
        try:
            for path in paths:
                count=0;endpoints={};decisions={};ten={}
                with gzip.open(path,"rt",encoding="utf-8") as stream:
                    for line in stream:
                        r=json.loads(line);slot,seed=r["env"],r["seed"]
                        if slot not in envs:
                            envs[slot]=GodotEnv(c["godot"],c["project"],map_mode=c["map_mode"],navigation_steps=c["navigation_steps"])
                        if seeds.get(slot)!=seed:
                            states[slot]=envs[slot].reset(seed)["observation"]
                            trackers[slot]=Reward(states[slot],c["reward_variant"],c["gamma_tick"])
                            seeds[slot]=seed
                            if "rng_before_hex" in r:
                                import torch
                                from agent.m8_eval import episode_seed
                                generators[slot]=torch.Generator(device="cuda").manual_seed(episode_seed(c["rng_prefix"],"m9",seed,0))
                        comparison.same(states[slot],r["observation"],"observation")
                        prepared=encode(states[slot],c["state_variant"])
                        chosen=prepared["branches"][r["verb_index"]][r["target_index"]]
                        comparison.same(chosen,r["action"],"chosen")
                        nxt=envs[slot].step(chosen)
                        comparison.same(nxt["transition"],r["transition"],"transition")
                        comparison.same(nxt["observation"],r["next_observation"],"successor")
                        comparison.same(trackers[slot].score(nxt["transition"],nxt["observation"]),r["reward"],"reward")
                        states[slot]=nxt["observation"]
                        if not path.name.startswith("rollout_"):
                            decisions[seed]=decisions.get(seed,0)+1
                            for tick in nxt["transition"]["trace"]:
                                if tick["tick"]==1800:ten[seed]=not tick["dead"]
                            final=transition_at_horizon(nxt["transition"],c["horizon"])["after"]
                            if final["dead"] or final["tick"]==c["horizon"]:
                                if seed in endpoints:raise ValueError("Continued after assessment endpoint")
                                endpoints[seed]=dict(survival_ticks=final["tick"],terminal=final["dead"],
                                    alive_at_horizon=not final["dead"],alive_at_1800=ten.get(seed,False),
                                    decisions=decisions[seed],final=final)
                        if "rng_before_hex" in r:verify_generator_draws(generators[slot],r,"cuda")
                        count+=1
                if not path.name.startswith("rollout_"):
                    report=json.loads(path.with_suffix("").with_suffix(".json").read_text())
                    if len(report["episodes"])!=len(endpoints) or {e["seed"] for e in report["episodes"]}!=set(endpoints):raise ValueError("Incomplete endpoints")
                    for e in report["episodes"]:
                        for k,val in endpoints[e["seed"]].items():comparison.same(val,e[k],"episode."+k)
                    if sha(path)!=report["trace_sha256"]:raise ValueError("Evaluation trace hash differs")
                counts[path.name]=dict(transitions=count,sha256=sha(path))
                print(json.dumps(dict(audited=path.name,transitions=count)),flush=True)
        finally:
            for env in envs.values():env.close()
    verify_registration(reg)
    write(out/("test_audit.json" if a.evaluation_only else "audit.json"),
        dict(passed=True,files=counts,transitions=sum(v["transitions"] for v in counts.values()),
             max_float_error=comparison.max_float_error,auditor_sha256=sha(__file__),
             registration_sha256=reg["registration_sha256"]))
if __name__=="__main__":main()

