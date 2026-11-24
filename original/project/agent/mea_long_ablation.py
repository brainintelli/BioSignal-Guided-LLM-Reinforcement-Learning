"""Predeclared 28080 offline comparison; pure protocol, no model/device imports."""
import itertools
import math
import random
from statistics import mean
from agent.mea_ablation import ARMS

PIN='49302e7794da71b5c6038827c182eb7cdf8edf9eca4f2d1c5faf49d2af7fe507'
# Independent accepted 28080 run initial_parameters.json; also identical to
# the earlier 22021 experiment's frozen_registration common_initial_tensors.
# This is the ORIGINAL checkpoint, not the trained 28080 final checkpoint.
ORIGINAL_TENSORS=dict(
    actor='94d7dd9adaa86ecaffacb8dbec584054ba96091c816da3bb1bf59db8249df5e0',
    critic='52a7c71c972fe83ea149a14d91472f0c4e8a397d3b4a656ffab8a9a58ab2b63e',
    base='0ddb7bb4f250d5ec4ed7b0a59180234fe8df0aed660868ea7da6856d3e29a7f0',
    reference='c4e0cac8056d8537efad0ff82725665da41c5f0a723aae5c4dbe6d78f1852d1f')
PROTOCOL=dict(schema='mea.long.ablation.v1',chip_id='28080',source_manifest_sha256=PIN,
    blocks=8,arms=list(ARMS),train_decisions=1024,rounds=8,decisions_per_round=128,
    actor_steps_per_round=2,critic_steps_per_round=16,eval_decisions=600,
    train=[0,1025],gap=[1025,1085],heldout=[1085,1686],
    shuffle='within_split_action_derangement_fixed_boundary',initial_seed=6100000,
    primary='post_discounted_R1_at_600_decisions_or_terminal',
    secondary='paired_prepost_difference_in_differences',bootstrap_resamples=20000,
    analysis_seed=6800100,alpha=1.,r2_enabled=False,stimulation=False,realtime_training=False,
    biological_replications=1,inference_scope='digital_seed_blocks_conditional_on_one_recording',
    hardware_mapping_independently_verified=False,original_tensors=ORIGINAL_TENSORS)


def cell_config(block,arm):
    if type(block) is not int or not 0<=block<8 or arm not in ARMS:raise ValueError('unregistered cell')
    return dict(block=block,arm=arm,initial_seed=6100000,world_seed=6100100+1000*block,
        action_seed=6200100+10000*block,ppo_seed=6300100+100*block)


def assignment(arm,split,block):
    cell_config(block,arm)
    if split not in ('train','eval'):raise ValueError('unregistered split')
    lo,hi=(0,1024) if split=='train' else (1085,1685)
    original=list(range(lo,hi));result=original.copy()
    if arm=='shuffled':
        rng=random.Random((6600100 if split=='train' else 6700100)+block)
        while True:
            rng.shuffle(result)
            if all(x!=y for x,y in zip(original,result)):break
    return result+[hi]


def validate_budget(applied):
    if set(applied)!={'actor','critic'} or any(type(v) is not int for v in applied.values()) or applied!={'actor':2,'critic':16}:
        raise ValueError('unequal registered optimizer budget')


def columns_expected(arm,nonzero):
    if arm not in ARMS or type(nonzero) is not bool:raise ValueError('column condition')
    return nonzero==(arm!='none')


class AssignedRecording:
    """Finite training index indirection; original snapshots retain source indices."""
    def __init__(self,data,config):
        if config!=cell_config(config['block'],config['arm']):raise ValueError('cell config differs')
        self.data=data;self.order=assignment(config['arm'],'train',config['block'])
        self.window_count=data.window_count
    def snapshot(self,index,context_id):
        if type(index) is not int or not 0<=index<len(self.order):raise ValueError('outside registered train split')
        return self.data.snapshot(self.order[index],context_id)
    def summary(self):return self.data.summary()


def paired_stats(differences):
    if len(differences)!=8 or any(not math.isfinite(v) for v in differences):raise ValueError('eight finite blocks required')
    observed=mean(differences);tol=1e-12*max(1.,abs(observed))
    p=sum(abs(mean(s*x for s,x in zip(signs,differences)))>=abs(observed)-tol
        for signs in itertools.product((-1,1),repeat=8))/256
    rng=random.Random(PROTOCOL['analysis_seed'])
    draws=sorted(mean(rng.choices(differences,k=8)) for _ in range(20000))
    def q(f):
        at=f*(len(draws)-1);i=int(at)
        return draws[i]+(draws[min(i+1,len(draws)-1)]-draws[i])*(at-i)
    return dict(differences=differences,mean_difference=observed,bootstrap_95=[q(.025),q(.975)],
        p_two_sided=p,p_bonferroni_two_contrasts=min(1.,2*p))


def analyze(cells):
    indexed={}
    for c in cells:
        cell_config(c['block'],c['arm']);key=(c['block'],c['arm'])
        if key in indexed:raise ValueError('duplicate cell')
        if any(not math.isfinite(c[k]) for k in ('pre','post')):raise ValueError('nonfinite outcome')
        indexed[key]=c
    missing=[list(k) for k in itertools.product(range(8),ARMS) if k not in indexed]
    if missing:return dict(status='incomplete',missing=missing,positive_evidence=False,contrasts=None)
    contrasts={}
    for control in ('none','shuffled'):
        post=[indexed[b,'real']['post']-indexed[b,control]['post'] for b in range(8)]
        gain=[post[b]-(indexed[b,'real']['pre']-indexed[b,control]['pre']) for b in range(8)]
        contrasts['real_minus_'+control]=dict(post=paired_stats(post),gain=paired_stats(gain))
    positive=all(c['post']['mean_difference']>0 and c['post']['bootstrap_95'][0]>0 and
        c['post']['p_bonferroni_two_contrasts']<.05 for c in contrasts.values())
    return dict(status='complete',replication_units=8,biological_replications=1,
        inference_scope=PROTOCOL['inference_scope'],positive_evidence=positive,contrasts=contrasts,
        arms={a:{phase:mean(indexed[b,a][phase] for b in range(8)) for phase in ('pre','post')} for a in ARMS},
        biological_effect_verified=False)
