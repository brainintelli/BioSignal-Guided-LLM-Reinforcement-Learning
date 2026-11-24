"""Finite M9 PPO and paired 30-day evaluations; native game ticks are authoritative."""
import argparse
from collections import Counter
import gzip
import json
from pathlib import Path
import random
import time
from types import SimpleNamespace
from agent.m8s3_training import write,sha
from agent.m8s3_run import emit,record,preserve_rng
from agent.m9_contract import Reward,validate
from agent.m9_env import GodotEnv
from agent.m9_metrics import summarize,compare
from agent.rl_contract import gae,transition_at_horizon

def finish_trajectory(rows,bootstrap,gamma,lam):
    if not rows or not 0<gamma<1 or not 0<lam<=1:raise ValueError("Invalid trajectory")
    next_values=[0. if r["terminal"] else rows[i+1]["value"] if i+1<len(rows) else bootstrap for i,r in enumerate(rows)]
    rows[-1]["boundary"]=True
    adv,ret=gae([r["reward"] for r in rows],[r["value"] for r in rows],next_values,
        [r["duration"] for r in rows],[r["terminal"] for r in rows],[r["boundary"] for r in rows],gamma_tick=gamma,lambda_tick=lam)
    for r,a,v in zip(rows,adv,ret):
        r.update(advantage=a,**{"return":v});r.pop("next_obs",None)
    return rows

def environment(c):
    return GodotEnv(c.godot,c.project,map_mode=c.map_mode,navigation_steps=c.navigation_steps)

def evaluate(policy,c,seeds,path,reference=False):
    import torch
    from agent.m8_eval import episode_seed
    from agent.m8x_eval import SlotSchedule
    path=Path(path);trace=path.with_suffix(".jsonl.gz")
    if path.exists() or trace.exists():raise FileExistsError("Never reuse evaluation evidence")
    schedule=SlotSchedule([(s,0) for s in seeds])
    envs={};states={};trackers={};generators={};episodes={};reports=[]
    actor=policy.reference_digest() if reference else policy.digest()
    began=time.monotonic()
    def begin(slot):
        seed,rep=schedule.active[slot]
        states[slot]=envs[slot].reset(seed)
        validate(states[slot]["observation"])
        trackers[slot]=Reward(states[slot]["observation"],c.reward_variant,c.gamma_tick)
        rng=episode_seed(c.rng_prefix,"m9",seed,rep)
        generators[slot]=torch.Generator(device=policy.device_name).manual_seed(rng)
        episodes[slot]=dict(seed=seed,replicate=rep,rng_seed=rng,decisions=0,actions=Counter(),alive_at_1800=False)
    try:
        for slot in sorted(schedule.active):
            envs[slot]=environment(c);begin(slot)
        with torch.no_grad(),gzip.open(trace,"xt",encoding="utf-8",compresslevel=1) as stream:
            while schedule.active:
                active=sorted(schedule.active)
                prepared=policy.prepare([states[i]["observation"] for i in active])
                with policy._scope(prepared,reference):
                    vd=policy.verb_distribution(prepared,reference)
                    before=[generators[i].get_state().cpu().numpy().tobytes().hex() for i in active]
                    vs=[int(torch.multinomial(vd.probs[j],1,generator=generators[i])) for j,i in enumerate(active)]
                    td=policy.target_distribution(prepared,vs,reference)
                    ts=[int(torch.multinomial(td.probs[j],1,generator=generators[i])) for j,i in enumerate(active)]
                done=[]
                for j,i in enumerate(active):
                    action=prepared[j]["branches"][vs[j]][ts[j]]
                    nxt=envs[i].step(action)
                    scored=trackers[i].score(nxt["transition"],nxt["observation"])
                    final=transition_at_horizon(nxt["transition"],c.horizon)["after"]
                    for tick in nxt["transition"]["trace"]:
                        if tick["tick"]==1800:episodes[i]["alive_at_1800"]=not tick["dead"]
                    terminal=final["dead"];alive=final["tick"]==c.horizon and not terminal
                    record(stream,dict(seed=episodes[i]["seed"],env=i,observation=states[i]["observation"],
                        action=action,transition=nxt["transition"],next_observation=nxt["observation"],reward=scored,
                        source="model",actor_sha256=actor,verb_index=vs[j],target_index=ts[j],
                        verb_probabilities=vd.probs[j].tolist(),target_probabilities=td.probs[j].tolist(),
                        rng_before_hex=before[j],rng_after_hex=generators[i].get_state().cpu().numpy().tobytes().hex()))
                    episodes[i]["decisions"]+=1;episodes[i]["actions"][action["verb"]]+=1
                    states[i]=nxt
                    if terminal or alive:
                        episodes[i].update(survival_ticks=final["tick"],terminal=terminal,alive_at_horizon=alive,final=final)
                        reports.append(episodes[i]);done.append(i)
                        emit(event="evaluation_episode",evaluation=path.stem,seed=episodes[i]["seed"],ticks=final["tick"],alive=alive)
                for slot in done:
                    if schedule.complete(slot) is not None:begin(slot)
        if (policy.reference_digest() if reference else policy.digest())!=actor:raise ValueError("Evaluation changed weights")
    finally:
        for env in envs.values():env.close()
    reports.sort(key=lambda e:e["seed"])
    if [e["seed"] for e in reports]!=sorted(seeds):raise ValueError("Incomplete assessment")
    result=dict(episodes=reports,summary=summarize(reports,c.horizon),seconds=time.monotonic()-began,
        decisions=sum(e["decisions"] for e in reports),trace_sha256=sha(trace),actor_sha256=actor,
        map_mode=c.map_mode,navigation_steps=c.navigation_steps,state_variant=c.state_variant,horizon=c.horizon,
        temperature=1,reference=reference,registration_sha256=c.registration_sha256)
    write(path,result)
    return result

def train(policy,c,out):
    import torch
    from agent.m8_train import update
    from agent.m9_policy import save,load
    from tools.benchmark_m8s3 import selected_logps
    initial_actor=policy.digest()
    with preserve_rng():
        initial=evaluate(policy,c,c.evaluation_seeds,out/"initial.json")
    actor_opt=torch.optim.AdamW([p for p in policy.model.parameters() if p.requires_grad],lr=c.actor_lr,eps=1e-5,weight_decay=0)
    critic_opt=torch.optim.AdamW(policy.critic.parameters(),lr=c.critic_lr,eps=1e-5,weight_decay=0)
    envs=[];states=[];trackers=[];seeds=[];cursor=0
    def reset(i):
        nonlocal cursor
        seed=c.train_seed_start+cursor
        if cursor>=c.train_seed_count:raise ValueError("Training seed budget exhausted")
        cursor+=1;state=envs[i].reset(seed)
        return state,Reward(state["observation"],c.reward_variant,c.gamma_tick),seed
    random.seed(c.rng_seed);torch.manual_seed(c.rng_seed)
    began=time.monotonic();decisions=ticks=0
    try:
        for _ in range(c.envs):envs.append(environment(c))
        for i in range(c.envs):
            state,tracker,seed=reset(i);states.append(state);trackers.append(tracker);seeds.append(seed)
        for number in range(1,c.updates+1):
            start=time.monotonic();trajectories=[[] for _ in envs];actions=Counter();probe_obs=[]
            with gzip.open(out/f"rollout_{number:04d}.jsonl.gz","xt",encoding="utf-8",compresslevel=1) as rollout_stream:
                for step in range(c.rollout_steps):
                    prepared=policy.prepare([s["observation"] for s in states])
                    sampled=policy.act(prepared,with_reference=True)
                    for i,env in enumerate(envs):
                        action=sampled["actions"][i];nxt=env.step(action)
                        scored=trackers[i].score(nxt["transition"],nxt["observation"])
                        terminal=nxt["terminated"]
                        row=dict(prepared=prepared[i],verb=sampled["verbs"][i],target=sampled["targets"][i],
                            value=sampled["values"][i],logp=sampled["logp"][i],verb_logp=sampled["verb_logp"][i],
                            reference_verb=sampled["reference_verb"][i],reference_target=sampled["reference_target"][i],
                            reward=scored["reward"],duration=scored["elapsed_ticks"],terminal=terminal,boundary=terminal,
                            next_obs=nxt["observation"])
                        trajectories[i].append(row);decisions+=1;ticks+=row["duration"];actions[action["verb"]]+=1
                        record(rollout_stream,dict(update=number,env=i,seed=seeds[i],observation=states[i]["observation"],
                            action=action,transition=nxt["transition"],next_observation=nxt["observation"],
                            reward=scored,source="model",verb_index=row["verb"],target_index=row["target"],logp=row["logp"]))
                        if len(probe_obs)<16:probe_obs.append(dict(observation=states[i]["observation"],verb=row["verb"],target=row["target"]))
                        if terminal:
                            with (out/"episodes.jsonl").open("a",encoding="utf-8") as episode_stream:
                                record(episode_stream,dict(update=number,seed=seeds[i],survival_ticks=nxt["observation"]["tick"],final=nxt["transition"]["after"]))
                            states[i],trackers[i],seeds[i]=reset(i)
                        else:states[i]=nxt
            rows=[]
            for trajectory in trajectories:
                with torch.no_grad():
                    bootstrap=0. if trajectory[-1]["terminal"] else float(policy.values(policy.prepare([trajectory[-1]["next_obs"]]))[0])
                rows.extend(finish_trajectory(trajectory,bootstrap,c.gamma_tick,c.gae_lambda))
            lp=selected_logps(policy,[r["prepared"] for r in rows],[r["verb"] for r in rows],[r["target"] for r in rows])
            error=max(abs(x-r["logp"]) for x,r in zip(lp,rows))
            if error>1e-6:raise ValueError("Pre-update all-row probability drift: "+str(error))
            rollout_seconds=time.monotonic()-start
            stats=update(policy,actor_opt,critic_opt,rows,c)
            torch.cuda.synchronize()
            metric=dict(update=number,decisions=decisions,ticks=ticks,rollout_seconds=rollout_seconds,
                update_seconds=time.monotonic()-start-rollout_seconds,pre_update_all_logp_error=error,actions=dict(actions),**stats)
            with (out/"metrics.jsonl").open("a",encoding="utf-8") as metrics_stream:record(metrics_stream,metric)
            emit(event="training_update",**metric)
            write(out/"status.json",dict(status="training",latest=metric,milestone_achieved=False))
            if number==c.updates:
                checkpoint=out/f"checkpoint_{number:04d}"
                save(policy,checkpoint,actor_opt,critic_opt,number,c.registration_sha256)
                checks=policy.prepare([r["observation"] for r in probe_obs]);vs=[r["verb"] for r in probe_obs];ts=[r["target"] for r in probe_obs]
                before=selected_logps(policy,checks,vs,ts)
                load(policy,checkpoint)
                after=selected_logps(policy,checks,vs,ts)
                if max(abs(a-b) for a,b in zip(before,after))>1e-6:raise ValueError("Checkpoint roundtrip drift")
                write(checkpoint/"reload_probe.json",dict(rows=probe_obs,logp=before))
    finally:
        for env in envs:env.close()
    if initial_actor==policy.digest():raise ValueError("No actor update occurred")
    with preserve_rng():final=evaluate(policy,c,c.evaluation_seeds,out/"evaluation.json")
    result=dict(status="research_training_completed",updates=c.updates,decisions=decisions,ticks=ticks,
        seconds=time.monotonic()-began,initial_actor=initial_actor,final_actor=policy.digest(),
        initial=initial["summary"],final=final["summary"],checkpoint=str(checkpoint),milestone_achieved=False)
    write(out/"training_complete.json",result);write(out/"status.json",result)

def recover_evaluation(policy,c,out):
    """Reuse the final checkpoint, verify original probabilities, re-evaluate all seeds."""
    from agent.m9_policy import load
    from tools.benchmark_m8s3 import selected_logps
    checkpoint=out/f"checkpoint_{c.updates:04d}"
    load(policy,checkpoint)
    probe=json.loads((checkpoint/"reload_probe.json").read_text())
    rows=probe["rows"]
    actual=selected_logps(policy,policy.prepare([r["observation"] for r in rows]),
                          [r["verb"] for r in rows],[r["target"] for r in rows])
    error=max(abs(x-y) for x,y in zip(actual,probe["logp"]))
    if error>1e-6:raise ValueError("Recovered checkpoint differs from original probability probe")
    began=time.monotonic()
    with preserve_rng():final=evaluate(policy,c,c.evaluation_seeds,out/"evaluation.json")
    initial=json.loads((out/"initial.json").read_text())
    metrics=[json.loads(line) for line in (out/"metrics.jsonl").read_text().splitlines() if line.strip()]
    result=dict(status="research_training_completed",updates=c.updates,decisions=metrics[-1]["decisions"],
        ticks=metrics[-1]["ticks"],seconds=time.monotonic()-began,reused_training=True,
        recovered_probe_error=error,initial_actor=initial["actor_sha256"],final_actor=policy.digest(),
        initial=initial["summary"],final=final["summary"],checkpoint=str(checkpoint),
        source=c.resume_source,milestone_achieved=False)
    write(out/"training_complete.json",result);write(out/"status.json",result)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--campaign",required=True);ap.add_argument("--config",required=True)
    ap.add_argument("--action",choices=["train","reload","test","recover"],default="train")
    a=ap.parse_args()
    from tools.m9_campaign import verify_registration,validate_task
    reg=json.loads(Path(a.campaign).read_text());verify_registration(reg)
    spec=json.loads(Path(a.config).read_text());validate_task(reg,spec)
    c=SimpleNamespace(**spec)
    from tools.benchmark_m8s3 import setup
    setup()
    from agent.m9_policy import Policy,load
    out=Path(c.output)
    if a.action=="train":
        out.mkdir(parents=True,exist_ok=False);write(out/"config.json",spec)
    elif a.action=="recover":
        from tools.m9_resume import copy_evaluation_recovery
        copy_evaluation_recovery(spec)
    policy=Policy(c)
    if a.action=="train":
        train(policy,c,out)
    elif a.action=="recover":
        recover_evaluation(policy,c,out)
    elif a.action=="reload":
        from tools.benchmark_m8s3 import selected_logps
        checkpoint=out/f"checkpoint_{c.updates:04d}";load(policy,checkpoint)
        probe=json.loads((checkpoint/"reload_probe.json").read_text())
        rows=probe["rows"];prepared=policy.prepare([r["observation"] for r in rows])
        actual=selected_logps(policy,prepared,[r["verb"] for r in rows],[r["target"] for r in rows])
        error=max(abs(x-y) for x,y in zip(actual,probe["logp"]))
        if error>1e-6:raise ValueError("Independent reload failed")
        write(checkpoint/"independent_reload.json",dict(passed=True,logp_error=error,actor_sha256=policy.digest(),registration_sha256=c.registration_sha256))
    else:
        checkpoint=out/f"checkpoint_{c.updates:04d}";load(policy,checkpoint)
        if not json.loads((checkpoint/"independent_reload.json").read_text())["passed"]:raise ValueError("Reload evidence absent")
        audit=json.loads((out/"audit.json").read_text())
        if not audit["passed"]:raise ValueError("Independent audit absent")
        for name,item in audit["files"].items():
            if sha(out/name)!=item["sha256"]:raise ValueError("Audited evidence changed")
        with (out/"test_claim.json").open("x",encoding="utf-8") as f:json.dump(dict(actor_sha256=policy.digest(),registration_sha256=c.registration_sha256),f)
        candidate=evaluate(policy,c,reg["test_seeds"],out/"test_candidate.json")
        parent=evaluate(policy,c,reg["test_seeds"],out/"test_parent.json",reference=True)
        gate=compare(candidate["episodes"],parent["episodes"])
        write(out/"numerical_gate.json",gate)
        emit(event="final_gate",**gate)
    verify_registration(reg)

if __name__=="__main__":main()

