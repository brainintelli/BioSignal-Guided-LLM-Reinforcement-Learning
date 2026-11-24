"""Draft M8 mathematical contracts. Pure statistics never authorize release.

Frozen protocol must supply seeds and evidence checks externally. Observations
are censored at 1800; this does not estimate unlimited natural lifetime.
"""
import math
import random

HORIZON=1800
CHECKPOINTS=(8192,16384,24576,32768)


def _quantile(values,p):
    """Inverse empirical CDF (nearest rank), including boundary probabilities."""
    xs=sorted(values)
    if not xs or not 0<=p<=1:raise ValueError('Invalid quantile')
    return xs[max(0,math.ceil(len(xs)*p)-1)]


def lower_tail_mean(values,alpha=.2):
    xs=sorted(values)
    if not xs or not 0<alpha<=1:raise ValueError('Invalid lower-tail mass')
    mass=alpha*len(xs);whole=math.floor(mass);fraction=mass-whole
    return (sum(xs[:whole])+(fraction*xs[whole] if whole<len(xs) else 0))/mass


def _binomial_tail(k,n,p):
    return sum(math.comb(n,j)*p**j*(1-p)**(n-j) for j in range(k,n+1))


def _lower(k,n,alpha):
    if k==0:return 0.
    lo,hi=0.,1.
    for _ in range(80):
        mid=(lo+hi)/2
        if _binomial_tail(k,n,mid)<alpha:lo=mid
        else:hi=mid
    return (lo+hi)/2


def binomial_interval(k,n):
    if type(k) is not int or type(n) is not int or n<1 or not 0<=k<=n:raise ValueError('Invalid binomial counts')
    return dict(method='Clopper-Pearson exact binomial',one_sided_95_lower=_lower(k,n,.05),
                two_sided_95=[_lower(k,n,.025),1-_lower(n-k,n,.025)],
                scope='Independent world episodes only; per-comparison intervals, not simultaneous or cross-attempt guarantees')


def summarize(episodes):
    if not episodes:raise ValueError('Empty episodes')
    keys=[];ticks=[];successes=early=0
    for e in episodes:
        seed=e['seed'];rep=e['replicate'];t=e['survival_ticks'];dead=e['terminal'];alive=e['alive_at_horizon']
        if type(seed) is not int or type(rep) is not int or rep<0:raise ValueError('Invalid episode identity')
        if type(t) is not int or not 0<=t<=HORIZON:raise ValueError('Invalid scored lifetime')
        if type(dead) is not bool or type(alive) is not bool or dead==alive:raise ValueError('Inconsistent terminal flags')
        if alive and t!=HORIZON:raise ValueError('Alive episode not observed to horizon')
        keys.append((seed,rep));ticks.append(t);successes+=alive;early+=dead and t<360
    if len(set(keys))!=len(keys):raise ValueError('Duplicate world/replicate')
    n=len(ticks)
    return dict(episodes=n,successes=successes,success_rate=successes/n,early_deaths=early,early_death_rate=early/n,
                rmst=sum(ticks)/n,median=_quantile(ticks,.5),p10=_quantile(ticks,.1),cvar20=lower_tail_mean(ticks),
                quantile_definition='Inverse empirical CDF / nearest rank',
                cvar_definition='Lower 20 percent of empirical mass, fractional boundary episode',
                horizon=HORIZON,right_censored=successes)


def select_checkpoint(candidates):
    if set(candidates)!=set(CHECKPOINTS):raise ValueError('All four budget checkpoints required')
    summaries={};matrix=None
    for node,es in candidates.items():
        summaries[node]=summarize(es)
        current={(e['seed'],e['replicate']) for e in es};seeds={s for s,r in current}
        if len(seeds)!=8 or current!={(s,r) for s in seeds for r in (0,1)}:raise ValueError('Selection requires 8 worlds x 2 replicates')
        if matrix is not None and current!=matrix:raise ValueError('Selection worlds changed across checkpoints')
        matrix=current
    winner=max(CHECKPOINTS,key=lambda n:(summaries[n]['successes'],summaries[n]['cvar20'],summaries[n]['rmst'],-n))
    return dict(selected_decisions=winner,summaries=summaries,release_authorized=False,
                scope='Selection only; no independent test or completion evidence supplied')


def _paired_interval(differences,seed):
    rng=random.Random(seed);n=len(differences)
    means=sorted(sum(differences[rng.randrange(n)] for _ in range(n))/n for _ in range(10000))
    def percentile(p):
        index=(len(means)-1)*p;left=math.floor(index);fraction=index-left
        return means[left]*(1-fraction)+means[min(left+1,len(means)-1)]*fraction
    return [percentile(.025),percentile(.975)]


def qualify_selection(candidates,semantic_parent,old_parent):
    selection=select_checkpoint(candidates)
    chosen=candidates[selection['selected_decisions']]
    matrix={(e['seed'],e['replicate']) for e in chosen}
    parents=[summarize(es) for es in (semantic_parent,old_parent)]
    if any({(e['seed'],e['replicate']) for e in es}!=matrix for es in (semantic_parent,old_parent)):
        raise ValueError('Selection parent matrix differs')
    selected=selection['summaries'][selection['selected_decisions']]
    total=sum(e['survival_ticks'] for e in chosen)
    ratios=[total*10>=sum(e['survival_ticks'] for e in es)*factor for es,factor in ((old_parent,12),(semantic_parent,11))]
    return dict(**selection,eligible_for_test=selected['successes']>=15 and selected['early_deaths']==0 and all(ratios),
                selected=selected,original_v2=parents[1],semantic_parent=parents[0],original_v2_ratio_passed=ratios[0],semantic_ratio_passed=ratios[1])


def compare_test(candidate,semantic_parent,old_parent,*,bootstrap_seed=812800):
    if bootstrap_seed!=812800:raise ValueError('Frozen M8 bootstrap seed is 812800')
    summaries=[summarize(es) for es in (candidate,semantic_parent,old_parent)]
    matrices=[{(e['seed'],e['replicate']):e for e in es} for es in (candidate,semantic_parent,old_parent)]
    keys=set(matrices[0])
    if len(keys)!=100 or any(rep!=0 for _,rep in keys) or any(set(m)!=keys for m in matrices):raise ValueError('Test requires same 100 worlds x replicate 0')
    comparisons={}
    for index,name,min_ratio in ((2,'original_v2',1.2),(1,'semantic_parent',1.1)):
        parent=summaries[index];diff=[matrices[0][k]['survival_ticks']-matrices[index][k]['survival_ticks'] for k in sorted(keys)]
        # Reinitializing the same RNG reuses exactly the same paired indices.
        ci=_paired_interval(diff,bootstrap_seed)
        ratio_pass=sum(e['survival_ticks'] for e in candidate)*10>=sum(e['survival_ticks'] for e in (semantic_parent if index==1 else old_parent))*(11 if index==1 else 12)
        comparisons[name]=dict(mean_difference=sum(diff)/100,paired_95ci=ci,minimum_ratio=min_ratio,
            relative_gain=(summaries[0]['rmst']/parent['rmst']-1) if parent['rmst'] else None,
            ratio_passed=ratio_pass,gate_passed=ratio_pass and ci[0]>0)
    passed=summaries[0]['successes']>=90 and summaries[0]['early_deaths']<=5 and all(x['gate_passed'] for x in comparisons.values())
    return dict(numerical_gate_passed=passed,release_authorized=False,candidate=summaries[0],semantic_parent=summaries[1],old_parent=summaries[2],
                comparisons=comparisons,success_interval=binomial_interval(summaries[0]['successes'],100),
                sample_90_percent_is_not_population_90_proof=True,bootstrap_resamples=10000,bootstrap_seed=bootstrap_seed,
                paired_indices_shared_across_comparisons=True,two_attempt_familywise_correction=False,
                scope='Draft statistics only; identity, source, RNG, budget, ledger and real replay audits required separately')
