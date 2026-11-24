"""Audit bounded auxiliary rewards and the preregistered, host-paired 22554 comparison."""
import argparse
import csv
import hashlib
import json
import math
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

import audit_r2_beta_clock as clock

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output/mea_r2_ratio_22554_20261120"
OLD = ROOT / "output/mea_r2_beta_sweep_20261120"
LONG = ROOT / "output/22554_fresh_maps_20261119"
CAL = ROOT / "output/22554-20261118T090720Z/exploratory/extended-96x48/reward-mix-tick48"
REG = clock.read(OUT / "protocol.json")
GAME_FIELDS = list(clock.MET) + ["deaths"]
VERBS = list(clock.VERBS)


def check(value, message):
    if not value:
        raise ValueError(message)


def digest_file(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def bounds(native, sign, duration):
    check(type(duration) is int and duration > 0 and sign in (-1, 0, 1) and math.isfinite(native),
          "Invalid reward input")
    ticks = math.fsum(REG["gamma_tick"] ** k for k in range(duration))
    return sign * REG["scale_fraction"] * min(abs(native), REG["reference_per_tick"] * ticks)


def previous_rewards():
    rows = defaultdict(list)
    ppo_checks = defaultdict(list)
    for chip, arms in clock.ARMS.items():
        seeds, _ = clock.study(chip)
        for arm in arms:
            for old_beta in (2, 5, 10):
                for seed in seeds:
                    folder = clock.loc(chip, old_beta, arm, seed)
                    if chip == "22021":
                        train = clock.read(folder / "reward-audit.json")
                    else:
                        trace = clock.read(folder / "training-trace.json")
                        train = [(r["r1"], r["r2"], r["duration"], r["r2_contribution"], r["reward"]) for r in trace]
                        check(all(s == (None if r["terminal"] else 1 if r["recorded_phase"] == "pre" else -1)
                                  for r, (_, s, *_) in zip(trace,train)), "Physical label mismatch")
                        normalized=[]
                        for r in trace:
                            label=r["r2"] or 0
                            small=bounds(r["r1"],label,r["duration"])
                            normalized.append(dict(r,r2=small,r2_event_sign=label,r2_scaled=small))
                        _,batches=audit_train(normalized,clock.read(folder/"updates.json"),old_beta/100)
                        ppo_checks[(arm,old_beta)].extend(batches)
                    check(len(train) == 48, "Historical training budget mismatch")
                    for r1, sign, ticks, weighted, total in train:
                        scaled = bounds(r1, sign or 0, ticks)
                        check(math.isclose(scaled*old_beta/100, weighted, abs_tol=1e-14)
                              and math.isclose(r1+weighted,total,abs_tol=1e-14), "Historical mixer mismatch")
                        rows[(chip,arm)].append((r1, sign or 0, scaled, ticks))
    quantiles, ratios = [], []
    for (chip,arm), values in rows.items():
        native, signs, scaled, _ = np.asarray(values).T
        eligible = signs != 0
        abs_r1 = np.abs(native)
        q = np.quantile(abs_r1,[.1,.5,.9])
        quantiles.append(dict(chip=chip, arm=arm, steps=len(native), labelled=int(eligible.sum()),
                              p10_abs_r1=q[0], median_abs_r1=q[1], p90_abs_r1=q[2],
                              raw_dominant=int(np.sum(np.abs(signs)>abs_r1)),
                              raw_flip=int(np.sum(native*(native+signs)<0)),
                              raw_ratio=sum(np.abs(signs))/sum(abs_r1)))
        for beta in (0., .02, .05, .1, .2, .5):
            part = beta * scaled
            fractions = np.divide(np.abs(part), abs_r1, out=np.zeros_like(abs_r1), where=abs_r1!=0)
            check(np.all(fractions<=beta*REG["scale_fraction"]+1e-12) and
                  not np.any(native*(native+part)<0), "Counterfactual cap violation")
            ratios.append(dict(chip=chip,arm=arm,beta=beta,step_cap=beta*REG["scale_fraction"],
                               sum_abs_ratio=np.abs(part).sum()/abs_r1.sum(),max_step_ratio=fractions.max(),
                               sign_flips=int(np.sum(native*(native+part)<0))))
    clock.csvout(OUT/"historical_r1_scale.csv",quantiles)
    clock.csvout(OUT/"same_trace_ratio_grid.csv",ratios)
    clock.csvout(OUT/"historical_ppo_reward_check.csv",[
        dict(chip="22554",arm=arm,beta=b/100,batches=len(samples),
             max_mixed_residual=max(s["mixed_max_residual"] for s in samples),
             native_only_excluded=sum(s["native_only_excluded"] for s in samples))
        for (arm,b),samples in ppo_checks.items()])
    return quantiles


def bootstrap_weights(seeds, worlds, strata, rng):
    random = np.random.default_rng(rng if rng is not None else REG["analysis"]["bootstrap_seed"])
    a = np.zeros((20000, len(seeds)))
    for ids in strata:
        indices = [seeds.index(seed) for seed in ids]
        a[:,indices] = random.multinomial(len(indices),np.full(len(indices),1/len(indices)),size=len(a))
    b = random.multinomial(len(worlds),np.full(len(worlds),1/len(worlds)),size=len(a))
    return (a[:,:,None]*b[:,None,:]).reshape(len(a),-1) / (len(seeds)*len(worlds))


def estimates(matrix, metrics, weights, name, **meta):
    s,w,m = matrix.shape
    check(m == len(metrics) and np.isfinite(matrix).all(), "Incomplete matched outcomes")
    mean = matrix.mean(axis=(0,1))
    lo,hi = np.quantile(weights @ matrix.reshape(s*w,m),[.025,.975],axis=0)
    positive = (matrix.mean(axis=1)>1e-10).sum(axis=0)
    return [dict(**meta,contrast=name,metric=k,delta=float(mean[i]),ci95_low=float(lo[i]),
                 ci95_high=float(hi[i]),positive_seeds=int(positive[i]),seeds=s,worlds=w)
            for i,k in enumerate(metrics)]


def previous_long_horizon():
    study = clock.read(LONG/"protocol.json")
    seeds=sorted({m["training_seed"] for m in study["models"].values()})
    arms=["ordered","stateonly","pure","r2only","shuffled"]
    dates=study["ticks"]
    worlds=study["maps"]
    check(len(seeds)==6 and len(worlds)==24 and dates==[48,192] and
          math.isclose(study["reward_mix"]["r2_weight"]*study["reward_mix"]["amplitude_fraction"],.05),
          "Prior five-percent study mismatch")
    labels=["cells","health","hunger","thirst","deaths"]
    cubes={a:np.empty((len(seeds),len(worlds),len(dates),len(labels))) for a in arms}
    first={}
    for a in arms:
        for i,s in enumerate(seeds):
            key=f"{a}-{s}"
            meta=study["models"][key]
            folder=LONG/"runs"/key
            done=clock.read(folder/"done.json")
            check(done["final_state"]==meta["state"] and len(done["game_hashes"])==len(worlds),"Frozen checkpoint mismatch")
            for j,w in enumerate(worlds):
                p=folder/f"{w}.json"
                run=clock.read(p)
                check(digest_file(p)==done["game_hashes"][str(w)] and run["checkpoint"]==meta["manifest"]
                      and run["seed"]==w and run["protocol"]==digest_file(LONG/"protocol.json")
                      and first.setdefault(w,run["first_obs_hash"])==run["first_obs_hash"], "Unpaired archived game")
                for t,tick in enumerate(dates):
                    cubes[a][i,j,t]=[*run["ticks"][tick],int(run["terminal_tick"] is not None and run["terminal_tick"]<=tick)]
    weights=bootstrap_weights(seeds,worlds,[seeds],2026111902)
    results=[]
    for t,tick in enumerate(dates):
        wet = cubes["ordered"][:,:,t]-cubes["stateonly"][:,:,t]
        dry = cubes["r2only"][:,:,t]-cubes["pure"][:,:,t]
        for name,mat in [("ordered_cap05_minus_ordered_off",wet),("no_MEA_cap05_minus_no_MEA_off",dry),
                         ("R2_by_state2",wet-dry),("ordered_minus_shuffled_cap05",cubes["ordered"][:,:,t]-cubes["shuffled"][:,:,t])]:
            results.extend(estimates(mat,labels,weights,name,tick=tick))
    clock.csvout(OUT/"archived_5pct_effects.csv",results)
    return results


def audit_train(rows, updates, beta):
    from agent.rl_contract import gae as training_gae
    data, gae = [], []
    check(len(rows)==48 and len(updates)==4,"PPO budget")
    for i,r in enumerate(rows):
        expected = 0 if r["terminal"] else (1 if r["recorded_phase"]=="pre" else -1)
        native = r["r1"]
        small = bounds(native,expected,r["duration"])
        part = beta * small
        check(r["r2_event_sign"]==expected and math.isclose(r["r2"],small,abs_tol=1e-14)
              and r["r2"]==r["r2_scaled"] and math.isclose(r["r2_contribution"],part,abs_tol=1e-14)
              and r["reward"]==native+r["r2_contribution"] and r["decision"]==i+1
              and r["reward"]*native>=0 and
              ((native!=0 and expected!=0 and beta!=0) or r["r2_contribution"]==0),
              "Inconsistent per-step reward")
        data.append((native,small,part,expected))
    for i,update in enumerate(updates):
        check(update["unique_decisions"]==12 and update["world_size"]==8 and
              all(update["checks"].values()) and update["actor_changed"] and update["critic_changed"],
              "Eight-card optimizer update validation")
        batch=rows[i*12:(i+1)*12]
        # With the same trajectory, values and final bootstrap, the difference in GAE
        # cancels all value terms. Use the same duration, reset and batch boundaries.
        delta=0.
        deltas=[]
        for r in reversed(batch):
            carry = 0. if r["boundary"] or r["terminal"] else delta*(REG["gamma_tick"]*.999)**r["duration"]
            delta=r["r2_contribution"]+carry
            deltas.append(delta)
        ds=[r["duration"] for r in batch]
        term=[r["terminal"] for r in batch]
        edges=[r["boundary"] or j==len(batch)-1 for j,r in enumerate(batch)]
        values=[r["value"] for r in batch]

        def trajectories(field, bootstrap):
            adv,ret=training_gae([r[field] for r in batch],values,values[1:]+[bootstrap],ds,term,edges,
                                 REG["gamma_tick"],.999)
            return np.asarray(adv),np.asarray(ret)

        # Holding the same bootstrap for both reward functions cancels it in the subtraction.
        check(np.allclose(trajectories("reward",0)[0]-trajectories("r1",0)[0],deltas[::-1],
                          rtol=1e-10,atol=1e-13),"GAE sensitivity mismatch")
        std=update["advantage_std"]
        check(math.isfinite(std) and std>0,"Nonpositive PPO advantage variance")
        def residual(field):
            # Mean return fixes the only unlogged scalar (the final bootstrap). Both standard
            # deviations are independent checks against the actual GPU batch, for each hypothesis.
            _,r0=trajectories(field,0)
            _,r1=trajectories(field,1)
            slope=np.mean(r1-r0)
            boot=(update["return_mean"]-np.mean(r0))/slope if abs(slope)>1e-12 else 0.
            av,ret=trajectories(field,boot)
            return float(max(abs(np.mean(ret)-update["return_mean"]),
                             abs(np.std(av)-std),abs(np.std(ret)-update["return_std"])))
        error_mixed=residual("reward")
        error_native=residual("r1")
        check(error_mixed<=1e-5,"GPU return or advantage statistics disagree with the mixed reward")
        gae.append(dict(delta_adv_std_over_ppo_adv_std=float(np.std(deltas)/std),
                        applied_reward="R1_plus_bounded_R2" if beta else "native_R1",beta=beta,
                        legacy_engine_reward_tag=update["reward"],engine_automatic_r2=update["r2_enabled"],
                        mixed_max_residual=error_mixed,native_only_max_residual=error_native,
                        native_only_excluded=bool(error_native>1e-5)))
    return data,gae


def new_results():
    seeds,worlds=REG["seeds"],REG["worlds"]
    games,charge,runs=[],[],[]
    identities,phases,first={}, {}, {}
    for beta in REG["weights"]:
        for arm in REG["arms"]:
            tag=("off" if beta==0 else "cap02")+"-"+arm
            for seed in seeds:
                folder=OUT/"runs"/tag/str(seed)
                done=clock.read(folder/"done.json")
                reg=clock.read(folder/"registration.json")
                res=clock.read(folder/"result.json")
                cfg=reg["config"]
                host=next(h for h,ss in REG["hosts"].items() if seed in ss)
                check(done["host"]==host and done["beta"]==beta and done["arm"]==arm and
                      reg["budget"]["training_seeds"]==[seed] and reg["eval_seeds"]==worlds
                      and cfg["r2_beta"]==beta and cfg["seed"]==seed and not cfg["stimulation"]
                      and reg["budget"]["replay_sha256"]==REG["replay_sha256"]
                      and reg["physical_files"]==9 and done["registration"]==digest_file(folder/"registration.json")
                      and res["checkpoint"]["manifest_sha256"]==done["manifest"]
                      and res["checkpoint"]["reload_verified"] and res["checkpoint"]["optimizer_roundtrip_verified"]
                      and res["checkpoint"]["state"]==res["after"] and res["arm"]==arm
                      and res["training_steps"]==48 and res["update_count"]==4,"Unmatched model")
                check(identities.setdefault(seed,res["initial"])==res["initial"],"Unequal initial model")
                pairs,tests=reg["train_pairs"],reg["test_pairs"]
                check(len(pairs)==12 and len(tests)==6 and
                      set(p[0] for p in pairs)==set(p[1] for p in pairs)==set(range(6))
                      and {p[0] for p in tests}=={6,7,8} and all(p[0]==p[1] for p in tests)
                      and all((p[0]==p[1])==(arm=="ordered") for p in pairs),"Physical split")
                schedule=[(p[0],p[2]) for p in pairs]
                check(phases.setdefault(seed,schedule)==schedule,"Arms received different signs")
                trace=clock.read(folder/"training-trace.json")
                for i,r in enumerate(trace):
                    src,don,ph=pairs[i%len(pairs)]
                    check(r["recorded_phase"]==ph and r["next_state2_trial"]==don
                          and r["r2_trial"]==(src if beta else None),"Training donor or label phase")
                audited,gae=audit_train(trace,clock.read(folder/"updates.json"),beta)
                a=np.array(audited)
                charge.append(dict(host=host,arm=arm,beta=beta,seed=seed,steps=len(a),
                                   positive=int(sum(a[:,3]==1)),negative=int(sum(a[:,3]==-1)),
                                   sum_abs_r1=np.abs(a[:,0]).sum(),sum_abs_r2_scaled=np.abs(a[:,1]).sum(),
                                   sum_abs_contribution=np.abs(a[:,2]).sum(),
                                   positive_contribution=a[a[:,2]>0,2].sum(),negative_contribution=a[a[:,2]<0,2].sum(),
                                   net_contribution=a[:,2].sum(),
                                   max_fraction=np.divide(np.abs(a[:,2]),np.abs(a[:,0]),out=np.zeros(len(a)),where=a[:,0]!=0).max(),
                                   sign_flips=int(np.sum(a[:,0]*(a[:,0]+a[:,2])<0))))
                runs.extend(dict(host=host,arm=arm,seed=seed,batch=i+1,**v) for i,v in enumerate(gae))
                evaluation=clock.read(folder/"eval-trace.json")
                check([d["seed"] for d in res["eval"]]==worlds and len(evaluation)==len(worlds),"World split")
                for e in res["eval"]:
                    w=e["seed"]
                    rows=evaluation[str(w)]
                    check(len(rows)==e["decisions"]==6 and e["death"]==0,"Unexpected early game end")
                    base=rows[0]
                    entry=(base["obs_hash"],base["context_donor"],base["state2"])
                    check(first.setdefault((seed,w),entry)==entry,"Unmatched evaluation input")
                    for j,z in enumerate(rows):
                        p=z["verb_probabilities"]
                        _,don,phase=tests[j]
                        sg=1 if phase=="pre" else -1
                        small=bounds(z["r1"],sg,z["duration"])
                        check(z["recorded_phase"]==phase and z["next_state2_trial"]==don and z["r2_event_sign"]==sg
                              and math.isclose(z["r2_contribution"],beta*small,abs_tol=1e-14)
                              and math.isclose(z["reward"],z["r1"]+beta*small,abs_tol=1e-14)
                              and len(p)==8 and min(p)>=0 and abs(sum(p)-1)<2e-5 and
                              abs(p[VERBS.index(z["action"]["verb"])]-math.exp(z["verb_logp"]))<1e-5,
                              "Frozen-state policy or evaluation feedback")
                    key=clock.digest(["22554",w,[r["action"] for r in rows]])
                    end=(e["explored"],e["r1"],e["health"],e["hunger"],e["thirst"])
                    games.append(("22554",round(beta*100),arm,seed,w,key,rows,end,e["ticks"]))
    clock.csvout(OUT/"training_ratios.csv",charge)
    clock.csvout(OUT/"ppo_advantage_sensitivity.csv",runs)
    clock.CACHE=OUT/"clock_cache"
    clock.CACHE.mkdir(exist_ok=True)
    for old in (OLD/"audit_cache").glob("tick6-v2-*.json.gz"):
        if not (clock.CACHE/old.name).exists():
            shutil.copy2(old,clock.CACHE/old.name)
    fixed,actions=clock.audit(games,clock.runreplay(games),write_index=False)
    clock.csvout(OUT/"fixed_tick_worlds.csv",fixed)
    clock.csvout(OUT/"action_probabilities_by_world.csv",actions)
    return fixed,actions,charge,runs


def paired_comparisons(fixed,actions):
    seeds,worlds=REG["seeds"],REG["worlds"]
    axes={s:i for i,s in enumerate(seeds)},{w:j for j,w in enumerate(worlds)}
    dims=(len(seeds),len(worlds),len(GAME_FIELDS)+len(VERBS))
    cube={(b,a):np.full(dims,np.nan) for b in REG["weights"] for a in REG["arms"]}
    def item(r):
        return cube[(r["beta"],r["arm"])][axes[0][r["train_seed"]],axes[1][r["world_seed"]]]
    for r in fixed:
        item(r)[:len(GAME_FIELDS)]=[r[x] for x in clock.MET]+[0.]
    for r in actions:
        check(r["probability_decisions"]==r["decisions"],"Incomplete probabilities")
        item(r)[len(GAME_FIELDS)+VERBS.index(r["verb"])]=100*r["probability_sum"]/r["decisions"]
    for v in cube.values():
        check(np.isfinite(v).all() and np.allclose(v[:,:,len(GAME_FIELDS):].sum(axis=-1),100.),"Incomplete cube")
    weighted=bootstrap_weights(seeds,worlds,list(REG["hosts"].values()),None)
    wet=cube[(.2,"ordered")]-cube[(0.,"ordered")]
    control=cube[(.2,"shuffled")]-cube[(0.,"shuffled")]
    contrasts={"R2_by_order":wet-control,"ordered_cap02_minus_off":wet,"shuffled_cap02_minus_off":control,
               "ordered_minus_shuffled_cap02":cube[(.2,"ordered")]-cube[(.2,"shuffled")],
               "ordered_minus_shuffled_off":cube[(0.,"ordered")]-cube[(0.,"shuffled")]}
    labels=GAME_FIELDS+["p_"+v for v in VERBS]
    output,cohorts=[],[]
    for name,mat in contrasts.items():
        output.extend(estimates(mat,labels,weighted,name,tick=6))
        for h,ss in REG["hosts"].items():
            ids=[seeds.index(s) for s in ss]
            m=mat[ids].mean(axis=(0,1))
            cohorts.extend(dict(host=h,contrast=name,metric=k,delta=m[j],seeds=len(ids),worlds=len(worlds))
                           for j,k in enumerate(labels))
    base=list(csv.DictReader((OLD/"r1_reference_worlds.csv").open(encoding="utf-8-sig")))
    pure=np.full((len(seeds),len(worlds),len(clock.MET)),np.nan)
    for r in base:
        if r["chip"]=="22554":
            check(r["arm"]=="pure" and int(r["tick"])==6,"Historical pure model")
            pure[axes[0][int(r["train_seed"])],axes[1][int(r["world_seed"])]]=[float(r[m]) for m in clock.MET]
    for (b,arm),v in cube.items():
        output.extend(estimates(v[:,:,:len(clock.MET)]-pure,clock.MET,weighted,
                                f"{arm}_{'off' if b==0 else 'cap02'}_minus_historical_pure",tick=6))
    clock.csvout(OUT/"paired_effects.csv",output)
    clock.csvout(OUT/"host_differences.csv",cohorts)
    return output


def publish(effects, fixed, charge, gaes, history):
    sys.path.append(str(ROOT/".tools/mea_baseline_deps"))
    from report_r2_beta_sweep import plt,plot_save

    points={(v["contrast"],v["metric"]):v for v in effects}
    old={(v["tick"],v["contrast"],v["metric"]):v for v in history}
    precision=lambda k: 6 if k=="r1" else 3 if k.startswith("p_") else 2

    def interval(v):
        p=precision(v["metric"])
        return f"{v['delta']:+.{p}f}〔{v['ci95_low']:+.{p}f}，{v['ci95_high']:+.{p}f}〕"

    def charge_fraction(arm):
        sel=[c for c in charge if c["beta"] and c["arm"]==arm]
        return sum(c["sum_abs_contribution"] for c in sel)/sum(c["sum_abs_r1"] for c in sel)

    old_charges=list(csv.DictReader((OLD/"reward_scale.csv").open(encoding="utf-8-sig")))
    ordered=[100*float(v["ratio_abs"]) for v in old_charges if v["chip"]=="22554" and v["arm"]=="ordered"]
    shuffled=[100*float(v["ratio_abs"]) for v in old_charges if v["chip"]=="22554" and v["arm"]=="shuffled"]
    historical=clock.read(CAL/"reward-audit.json")
    for series,a in [(ordered,"ordered"),(shuffled,"shuffled")]:
        series.extend([100*charge_fraction(a),100*historical["by_arm"][a]["train"]["sum_abs_ratio"]])

    fig,axes=plt.subplots(2,3,figsize=(13.8,7.5))
    fig.subplots_adjust(left=.071,right=.982,bottom=.11,top=.83,wspace=.46,hspace=.59)
    fig.suptitle("R2 占多大比例，游戏表现改变了多少？",fontsize=20,fontweight="bold",y=.965)
    fig.text(.5,.89,"芯片 22554  ·  新训练 16 个配对种子 × 48 个留存世界  ·  相同游戏时钟第 6 刻",ha="center",fontsize=11,color="#40534d")
    ax=axes[0,0]
    pos=np.arange(5)
    ax.bar(pos-.18,ordered,.36,color="#19796d",label="有序")
    ax.bar(pos+.18,shuffled,.36,color="#d3844d",label="打乱")
    ax.set_xticks(pos,["0.2%","0.5%","1%","2%","5%*"])
    ax.set_xlabel("每步允许的上限");ax.set_ylabel("累计实际占比（%）")
    ax.set_title("训练中 R2 的实际分量",loc="left",fontweight="bold")
    ax.legend(frameon=False,fontsize=8)
    contrasts=["ordered_cap02_minus_off","shuffled_cap02_minus_off","R2_by_order"]
    captions=["有序组","打乱组","两组增量差"]
    for ax,m,title in zip([axes[0,1],axes[0,2],axes[1,0],axes[1,1]],
                          ["cells","r1","hunger","thirst"],["新探索格数","原生游戏回报 R1","饱食","水分"]):
        ax.axvline(0,lw=.8,color="#788781")
        for y,(c,col) in enumerate(zip(contrasts,["#19796d","#d3844d","#354f77"])):
            v=points[c,m]
            ax.hlines(y,v["ci95_low"],v["ci95_high"],color=col,linewidth=2)
            ax.plot(v["delta"],y,"o",color=col,ms=6)
        ax.set_yticks(range(3),captions);ax.set_ylim(2.65,-.65)
        ax.set_title(title,loc="left",fontweight="bold")
        ax.set_xlabel("2% 上限相对关闭 R2 的变化");ax.grid(axis="x",alpha=.17)
        if m=="r1":ax.ticklabel_format(axis="x",style="sci",scilimits=(-3,-3),useMathText=True)
    ax=axes[1,2]
    ax.axvline(0,lw=.8,color="#788781")
    for y,(vname,caption) in enumerate(zip(["move","drink","eat"],["移动","饮水","进食"])):
        v=points["R2_by_order","p_"+vname]
        ax.hlines(y,v["ci95_low"],v["ci95_high"],color="#354f77",lw=2);ax.plot(v["delta"],y,"o",color="#354f77")
    ax.set_yticks(range(3),["移动","饮水","进食"]);ax.set_ylim(2.65,-.65)
    ax.set_xlabel("两组增量差（百分点）");ax.set_title("动作概率",loc="left",fontweight="bold");ax.grid(axis="x",alpha=.17)
    fig.text(.5,.025,"线段为配对 95% 区间。训练占比按绝对值之和计算。*5% 来自较早的独立训练批次。",ha="center",fontsize=9,color="#43534e")
    plot_save(fig,OUT/"r2_ratio_effects.png")

    primary=points["R2_by_order","cells"]
    within=points["ordered_cap02_minus_off","cells"]
    cap=points["ordered_minus_shuffled_cap02","cells"]
    water=points["ordered_cap02_minus_historical_pure","thirst"]
    accepted=primary["ci95_low"]>0 and within["delta"]>0
    s=sum(v["steps"] for v in charge)
    applied=[v for v in charge if v["beta"]]
    active_gae=[v for v in gaes if v["beta"]]
    std=[v["delta_adv_std_over_ppo_adv_std"] for v in active_gae]
    q25,q50,q75=100*np.quantile(std,[.25,.5,.75])
    pmax=max(v["max_fraction"] for v in charge)
    oldrisk=clock.read(CAL/"calibration.json")["legacy_training_audit"]["ordered"]
    hs=list(csv.DictReader((OUT/"historical_r1_scale.csv").open(encoding="utf-8-sig")))
    old225=[r for r in hs if r["chip"]=="22554"]
    nd=sum(int(r["labelled"]) for r in old225); flips=sum(int(r["raw_flip"]) for r in old225)
    affected=sum(int(r["raw_dominant"]) for r in old225)
    dates=clock.read(LONG/"protocol.json")["ticks"]
    lines=[
        "# 芯片 22554 的 R2 比例复验",
        "",
        f"新训练的 2% 上限{'达到' if accepted else '未达到'}预登记的探索改善判据。R2 给有序信号带来的额外探索差为 {interval(primary)} 格，{primary['positive_seeds']}/16 个训练种子方向为正。有序组自身对关闭 R2 的差为 {interval(within)} 格。",
        "",
        "## 奖励的单位和实际大小",
        "",
        "原生游戏奖励 R1 的权重固定为 1。已记录的刺激前无输出窗口取 z=+1，已记录的刺激后窗口取 z=−1，终止步取 0。无法确认时相的窗口不进入该对照。每步先以 R1 的单位计算 R2，再加权进入 PPO。",
        "",
        "`R2 = z × 0.10 × min(|R1|, 0.009775 × D)`，其中 D 是动作实际占用的折扣游戏刻数之和，折扣因子为 0.99985。尺度来自冻结的纯 R1 训练数据。",
        "",
        "`训练奖励 = R1 + β × R2`。当 R1 为零或到达终止步时，辅助贡献为零。单刻 R1 为 +0.009775 时，R2 为 ±0.0009775，β 为 0.20 时实际加减 0.0001955。正负号、标定后幅值和加权贡献分别记入每步的 `r2_event_sign`、`r2_scaled`、`r2_contribution`。",
        "",
        f"在上一轮 22554 的 {nd:,} 个有标签训练步上回算，直接加入 ±1 会在 {affected:,} 步压过 |R1|，有 {flips:,} 步会改变总奖励正负号。更早的逐刻固定 0.001 公式在有序组 {oldrisk['n']} 步中有 {oldrisk['dominant_steps']} 步超出 |R1|，最大为 {100*oldrisk['max_step_ratio']:.1f}%。",
        "",
        "| 单步加权上限 | 有序组训练累计占比 | 打乱组训练累计占比 | 数据来源 |",
        "|---:|---:|---:|---|"]
    for i,(cap_name,source) in enumerate(zip(["0.2%","0.5%","1%","2%","5%"],
                                            ["上一轮","上一轮","上一轮","本轮新训练","早期单独训练"])):
        lines.append(f"| {cap_name} | {ordered[i]:.3f}% | {shuffled[i]:.3f}% | {source} |")
    lines += ["",
        f"比例的分母为 Σ|R1|。本轮四组共 {s:,} 个训练决策，启用 R2 的 {sum(c['steps'] for c in applied):,} 步中有 {sum(c['positive'] for c in applied):,} 步正标签、{sum(c['negative'] for c in applied):,} 步负标签。实测最大单步占比 {100*pmax:.2f}%，奖励翻号共 {sum(c['sign_flips'] for c in charge)} 步。",
        "",
        f"在非零权重的 {len(std)} 个 PPO 批次中，固定轨迹、价值预测和末端自举值，单独计算 R2 对 GAE 优势的改变量。其批内标准差相当于实训优势标准差的中位数 {q50:.2f}%（四分位 {q25:.2f}% 至 {q75:.2f}%）。此比值描述优势信号的扰动大小，策略优劣由下表的留存游戏检验。",
        "",
        f"底层更新摘要沿用自动奖励开关，写作 `native_R1`。离线训练器在送入 PPO 前已经合入校准项。用 GPU 返回的批次平均回报推回唯一未记录的末端自举值，再独立核对回报和优势的两项标准差，全部 {len(active_gae)} 个非零权重批次与混合奖励相符，最大绝对误差 {max(v['mixed_max_residual'] for v in active_gae):.2g}。只送入 R1 的解释在 {sum(v['native_only_excluded'] for v in active_gae)} 个批次超过 0.00001 的核对容差。[每批 PPO 实际奖励核验](ppo_advantage_sensitivity.csv)",
        "",
        "## 同时刻的游戏结果",
        "",
        "每个种子的四组都在同一台八卡服务器从相同模型起步，48 个训练决策，4 次 PPO 更新。两台服务器各承担 8 个种子。共 64 个新模型、3,072 局冻结权重评估。每个模型评估相同的 48 个留存世界。第 6 个游戏刻从引擎实际状态读取，并核对动作、历时、原生 R1、资源和动作概率。",
        "",
        "| 指标 | 有序组，2% 减关闭 | 打乱组，2% 减关闭 | 两组增量之差 |",
        "|---|---:|---:|---:|"]
    for m,label in [("cells","探索新格"),("r1","原生 R1"),("health","生命"),("hunger","饱食"),("thirst","水分"),
                    ("p_move","移动概率，百分点"),("p_drink","饮水概率，百分点"),("p_eat","进食概率，百分点")]:
        lines.append("| "+label+" | "+" | ".join(interval(points[c,m]) for c in contrasts)+" |")
    lines += ["",
        f"直接比较 2% 档的有序与打乱，探索差为 {interval(cap)} 格。单独用已训练的历史纯 R1 模型作参照，2% 有序组的水分差为 {interval(water)} 点。水分是否由 R2 引起，应看上表同输入下的开关差。",
        "",
        f"第 6 刻生命范围为 {min(v['health'] for v in fixed):g}–{max(v['health'] for v in fixed):g}，留存评估共零次死亡。这一时长没有产生可比较的避死事件。所有双侧 95% 区间按 16 个训练种子和 48 个世界交叉配对重采样 20,000 次，种子在各自主机内抽取。探索增量之差是登记的主比较，其余为次要或描述性比较。",
        "",
        "![R2 比例和游戏指标](r2_ratio_effects.png)",
        "",
        "## 已有 5% 模型的较长游戏",
        "",
        "另行复核较早的 6 个种子、24 张新地图、5 组冻结模型的 720 局原始记录。此处固定有序 MEA 输入，比较训练时启用与关闭 R2。这次配对用于探索，评估时钟和训练预算与本轮不同。",
        "",
        "| 时刻 | 探索新格差 | 水分差 | 饱食差 |",
        "|---:|---:|---:|---:|"]
    for t in dates:
        lines.append("| "+str(t)+" | "+" | ".join(interval(old[t,"ordered_cap05_minus_ordered_off",m]) for m in ("cells","thirst","hunger"))+" |")
    lines += ["",
        "数据均来自历史回放，游戏动作无法改变已经发生的刺激。22554 的奖励检查使用物理试次的前后时相，6 次试次训练、另外 3 次测试。22021 的既有 R2 来自游戏动作结果，每三个合格决策才标一次，在历史尺度复核中单列。两枚芯片的奖励来源不同，本轮直接检验刺激事件比例使用 22554。此前已反复评估这 48 个世界，独立新芯片和新世界仍是复现所需的证据。",
        "",
        "后续验证可把单步比例固定在已验证的 2% 上限。固定有序 state2 和相同游戏时长，比较关闭 R2、按实测时相定号，以及在各训练试次内随机互换前后标签的 R2。后两组配平正负次数并核对实际累计占比，用于区分刺激时相的信息和一般奖惩扰动。提前登记探索新格和水分差，在独立世界及独立记录上重复；低水饮水、低饱食进食同时报告出现合法动作机会的分母。",
        "",
        "[登记方案](protocol.json) · [配对效应与全部八类动作](paired_effects.csv) · [每个种子的训练贡献](training_ratios.csv) · [逐世界游戏指标](fixed_tick_worlds.csv) · [历史 R1 与裸标签核对](historical_r1_scale.csv) · [同轨迹改权重复算](same_trace_ratio_grid.csv) · [早期 5% 模型的配对结果](archived_5pct_effects.csv)"]
    (OUT/"report.md").write_text("\n".join(lines)+"\n",encoding="utf-8")


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--history-only",action="store_true")
    opts=parser.parse_args()
    previous_rewards()
    history=previous_long_horizon()
    if not opts.history_only:
        data,actions,charge,gaes=new_results()
        result=paired_comparisons(data,actions)
        publish(result,data,charge,gaes,history)
        print(f"Verified {len(charge)} freshly trained policies, {sum(e['steps'] for e in charge)} training decisions and {len(data)} frozen evaluations.")
