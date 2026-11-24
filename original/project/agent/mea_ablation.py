"""Frozen short-recording ablation protocol, representations and paired analysis."""
import itertools
import math
import random
from statistics import mean

from agent import m9_contract
from agent.m8s3_supply import readiness_features
from agent.mea_contract import encode
from agent.mea_replay import RECORDING_CONTRACT,MANIFEST_SHA256

ARMS=('none','real','shuffled')


def protocol():
    return dict(schema='mea.ablation.protocol.v1',source_manifest_sha256=MANIFEST_SHA256,
        arms=list(ARMS),blocks=8,train_indices=list(range(33)),guard_indices=[33,34,35],
        eval_indices=list(range(36,57)),train_decisions=32,eval_decisions=20,eval_worlds=2,
        shuffle_rule='derange_action_windows_keep_boundary_fixed',
        alpha=1.,r2_enabled=False,stimulation=False,realtime_training=False,
        primary='post_mean_discounted_r1_at_20_decisions',bootstrap_resamples=20000,
        uncertainty_unit='digital_seed_block_conditional_on_one_recording',
        initial_seed=4400100,train_world_seed=4200100,eval_world_seed=4300100,
        train_action_seed=4500100,eval_action_seed=4600100,ppo_seed=4700100,
        train_permutation_seed=4800100,eval_permutation_seed=4900100,
        analysis_seed=5000100,biological_replications=1)


def assignment(arm,split,block):
    if arm not in ARMS or split not in ('train','eval') or type(block) is not int or not 0<=block<8:
        raise ValueError('unregistered arm/split/block')
    p=protocol();original=p[split+'_indices'];result=original[:-1].copy()
    if arm=='shuffled':
        rng=random.Random(p[split+'_permutation_seed']+block)
        while True:
            rng.shuffle(result)
            if all(a!=b for a,b in zip(original[:-1],result)):break
    return result+[original[-1]]


def encode_arm(obs,snapshot,arm):
    if arm not in ARMS:raise ValueError('unregistered arm')
    if arm=='none':
        if snapshot is not None:raise ValueError('none policy must not receive a snapshot')
        row=m9_contract.encode(obs,'reserves')
        row.update(contract=RECORDING_CONTRACT,base_contract=m9_contract.VERSION,
            wetware_snapshot=None,wetware_features=[0.]*4,
            critic_input=row['features']+readiness_features(row)+[0.]*4)
    else:row=encode(obs,snapshot,'reserves')
    row['ablation_condition']=arm
    return row


def episode_metrics(rows,gamma=.99985):
    if not rows:raise ValueError('empty episode')
    elapsed=0;discounted=0.
    for r in rows:
        discounted+=gamma**elapsed*r['reward'];elapsed+=r['duration']
    after=rows[-1]['transition']['after']
    return dict(decisions=len(rows),discounted_r1=discounted,undiscounted_r1=sum(r['reward'] for r in rows),
        elapsed_ticks=elapsed,terminated=rows[-1]['terminal'],
        final_health=after['health'],final_hunger=after['hunger'],final_thirst=after['thirst'],
        final_inventory=after['inventory'],visited_count=after['visited_count'])


def paired_stats(differences):
    if len(differences)!=8 or any(not math.isfinite(x) for x in differences):raise ValueError('eight finite blocks required')
    observed=mean(differences)
    tolerance=1e-12*max(1.,abs(observed))
    p=sum(abs(mean(s*x for s,x in zip(signs,differences)))>=abs(observed)-tolerance
          for signs in itertools.product((-1,1),repeat=8))/256
    rng=random.Random(protocol()['analysis_seed'])
    draws=sorted(mean(rng.choices(differences,k=8)) for _ in range(20000))
    def quantile(q):
        at=q*(len(draws)-1);i=int(at)
        return draws[i]+(draws[min(i+1,len(draws)-1)]-draws[i])*(at-i)
    return dict(differences=differences,mean_difference=observed,bootstrap_95=[quantile(.025),quantile(.975)],
        p_two_sided=p,p_bonferroni_two_contrasts=min(1.,2*p))


def analyze(cells):
    indexed={}
    for c in cells:
        key=(c['block'],c['arm'])
        if key in indexed or key not in itertools.product(range(8),ARMS):raise ValueError('duplicate/unregistered cell')
        indexed[key]=c
    missing=[list(k) for k in itertools.product(range(8),ARMS) if k not in indexed]
    if missing:return dict(status='incomplete',missing=missing,contrasts=None)
    means={}
    for key,c in indexed.items():
        means[key]={}
        for phase in ('pre','post'):
            values=[x['discounted_r1'] for x in c[phase]]
            if len(values)!=2 or any(not math.isfinite(x) for x in values):raise ValueError('two finite paired worlds required')
            means[key][phase]=mean(values)
    contrasts={}
    for control in ('none','shuffled'):
        post=[means[b,'real']['post']-means[b,control]['post'] for b in range(8)]
        gain=[post[b]-(means[b,'real']['pre']-means[b,control]['pre']) for b in range(8)]
        contrasts['real_minus_'+control]=dict(post=paired_stats(post),gain=paired_stats(gain))
    return dict(status='complete',replication_units=8,biological_replications=1,
        inference_scope='digital_seeds_conditional_on_this_recording',
        arms={a:{s:mean(means[b,a][s] for b in range(8)) for s in ('pre','post')} for a in ARMS},
        contrasts=contrasts,biological_effect_verified=False)
