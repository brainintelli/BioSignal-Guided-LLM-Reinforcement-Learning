"""30-day statistics; neither screening nor numerical success authorizes release."""
from tools.m8b_metrics import lower_tail_mean,binomial_interval,_paired_interval

def summarize(episodes,horizon=5400):
    if not episodes:raise ValueError("No episodes")
    keys=set();ticks=[];success=early=ten=0
    for e in episodes:
        key=(e["seed"],e["replicate"]);t=e["survival_ticks"]
        if key in keys or type(t) is not int or not 0<t<=horizon or e["replicate"]!=0:raise ValueError("Invalid world matrix")
        if type(e["terminal"]) is not bool or type(e["alive_at_horizon"]) is not bool or e["terminal"]==e["alive_at_horizon"]:raise ValueError("Invalid censoring")
        if e["alive_at_horizon"] and t!=horizon:raise ValueError("Incomplete survivor")
        if type(e["alive_at_1800"]) is not bool:raise ValueError("Missing exact10-day endpoint")
        if e["alive_at_1800"]!=(t>1800 or (t==1800 and e["alive_at_horizon"])):
            raise ValueError("10-day endpoint disagrees with scored lifetime")
        keys.add(key);ticks.append(t);success+=e["alive_at_horizon"];early+=e["terminal"] and t<360;ten+=e["alive_at_1800"]
    return dict(episodes=len(episodes),successes=success,success_rate=success/len(episodes),
        successes_10day=ten,rmst=sum(ticks)/len(ticks),rmst_days=sum(ticks)/len(ticks)/180,
        cvar20=lower_tail_mean(ticks),early_deaths=early,horizon=horizon)

def select_arm(rows,arms):
    groups={a:[] for a in arms}
    for r in rows:
        if r["arm"] not in groups:raise ValueError("Unregistered arm")
        groups[r["arm"]].append(r)
    if any(len(rs)!=2 or {r["replicate"] for r in rs}!={0,1} for rs in groups.values()):
        raise ValueError("Both independent optimization replicates required")
    def rank(a):
        rs=[r["summary"] for r in groups[a]]
        return (sum(r["successes"] for r in rs),min(r["cvar20"] for r in rs),sum(r["rmst"] for r in rs),-arms.index(a))
    return sorted(arms,key=rank,reverse=True)

def compare(candidate,parent):
    a,b=summarize(candidate),summarize(parent)
    ca={(e["seed"],e["replicate"]):e for e in candidate};pa={(e["seed"],e["replicate"]):e for e in parent}
    if len(ca)!=100 or set(ca)!=set(pa):raise ValueError("100 paired independent worlds required")
    diffs=[ca[k]["survival_ticks"]-pa[k]["survival_ticks"] for k in sorted(ca)]
    success=[int(ca[k]["alive_at_horizon"])-int(pa[k]["alive_at_horizon"]) for k in sorted(ca)]
    ci=_paired_interval(diffs,912800);sci=_paired_interval(success,912800)
    ni=ci[0]>-270 and sci[0]>-.05 and a["cvar20"]>=.9*b["cvar20"]
    return dict(numerical_gate_passed=a["successes"]>=95 and a["successes_10day"]>=90 and a["early_deaths"]<=5 and ni,
        candidate=a,parent=b,rmst_difference_ci=ci,success_difference_ci=sci,noninferiority=ni,
        interval=binomial_interval(a["successes"],100),milestone_achieved=False,release_authorized=False)

