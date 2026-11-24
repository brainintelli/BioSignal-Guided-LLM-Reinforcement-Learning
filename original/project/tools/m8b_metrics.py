"""M8B mathematical contracts. Pure statistics never authorize release.

Frozen protocol must supply seeds and evidence checks externally. Observations
are censored at 1800; this does not estimate unlimited natural lifetime.
"""
import math
import random

HORIZON=1800
CHECKPOINTS=(16384,32768,49152)


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


def matrix(episodes,n):
    summarize(episodes)
    result={(e['seed'],e['replicate']):e for e in episodes}
    if len(result)!=n or any(rep!=0 for _,rep in result):raise ValueError('Independent worlds with replicate zero required')
    return result


def qualify_selection(candidates,parent_episodes):
    if set(candidates)!=set(CHECKPOINTS):raise ValueError('All three complete checkpoints required')
    parent=matrix(parent_episodes,16);stats={}
    for node,es in candidates.items():
        if set(matrix(es,16))!=set(parent):raise ValueError('Selection world matrix differs')
        stats[node]=summarize(es)
    winner=max(CHECKPOINTS,key=lambda n:(stats[n]['successes'],stats[n]['cvar20'],stats[n]['rmst'],-n))
    child=stats[winner];baseline=summarize(parent_episodes)
    rmst=child['rmst']-baseline['rmst']>=-90
    success=(child['successes']-baseline['successes'])/16>=-.05
    cvar=child['cvar20']>=.9*baseline['cvar20']
    return dict(selected_decisions=winner,summaries=stats,selected=child,balanced_parent=baseline,
        eligible_for_test=child['successes']>=15 and child['early_deaths']==0 and rmst and success and cvar,
        rmst_noninferiority_passed=rmst,success_noninferiority_passed=success,cvar_passed=cvar,
        release_authorized=False,scope='Selection point estimates only; no test or source evidence')


def _paired_interval(differences,seed):
    rng=random.Random(seed);n=len(differences)
    means=sorted(sum(differences[rng.randrange(n)] for _ in range(n))/n for _ in range(10000))
    def p(q):
        index=(len(means)-1)*q;left=math.floor(index);fraction=index-left
        return means[left]*(1-fraction)+means[min(left+1,len(means)-1)]*fraction
    return [p(.025),p(.975)]


def compare_test(candidate,balanced_parent,old_parent,*,bootstrap_seed=912800):
    if bootstrap_seed!=912800:raise ValueError('Frozen bootstrap seed differs')
    arms=[matrix(es,100) for es in (candidate,balanced_parent,old_parent)]
    if any(set(m)!=set(arms[0]) for m in arms):raise ValueError('Paired worlds differ')
    summaries=[summarize(es) for es in (candidate,balanced_parent,old_parent)]
    comparisons={}
    for index,name in ((1,'balanced_parent'),(2,'original_v2')):
        diffs=[arms[0][k]['survival_ticks']-arms[index][k]['survival_ticks'] for k in sorted(arms[0])]
        success=[int(arms[0][k]['alive_at_horizon'])-int(arms[index][k]['alive_at_horizon']) for k in sorted(arms[0])]
        comparisons[name]=dict(mean_difference=sum(diffs)/100,paired_95ci=_paired_interval(diffs,bootstrap_seed),
            success_difference=sum(success)/100,success_paired_95ci=_paired_interval(success,bootstrap_seed),
            scope='Same environment RL comparison' if index==1 else 'Different environment descriptive comparison; not causal RL gain')
    world_diff=[arms[1][k]['survival_ticks']-arms[2][k]['survival_ticks'] for k in sorted(arms[0])]
    world_success=[int(arms[1][k]['alive_at_horizon'])-int(arms[2][k]['alive_at_horizon']) for k in sorted(arms[0])]
    world_comparison=dict(mean_difference=sum(world_diff)/100,paired_95ci=_paired_interval(world_diff,bootstrap_seed),
        success_difference=sum(world_success)/100,success_paired_95ci=_paired_interval(world_success,bootstrap_seed),
        scope='Same released parent weights; world balance plus factual rule text amendment, NOT new RL benefit; descriptive only')
    direct=comparisons['balanced_parent'];child,parent=summaries[:2]
    ni=dict(rmst_passed=direct['paired_95ci'][0]>-90,success_passed=direct['success_paired_95ci'][0]>-.05,
        cvar_passed=child['cvar20']>=.9*parent['cvar20'])
    target=child['successes']>=90 and child['early_deaths']<=5
    gain=direct['paired_95ci'][0]>0
    return dict(numerical_gate_passed=target and all(ni.values()),candidate=child,balanced_parent=parent,old_parent=summaries[2],
        noninferiority=ni,comparisons=comparisons,balanced_parent_vs_original_v2=world_comparison,rl_improvement_demonstrated=gain,
        prefer_parent=parent['successes']>=90 and parent['early_deaths']<=5 and not gain,
        success_interval=binomial_interval(child['successes'],100),sample_90_percent_is_not_population_90_proof=True,
        bootstrap_resamples=10000,bootstrap_seed=bootstrap_seed,paired_indices_shared_across_comparisons=True,
        release_authorized=False,scope='Statistics only; full independent evidence required')
