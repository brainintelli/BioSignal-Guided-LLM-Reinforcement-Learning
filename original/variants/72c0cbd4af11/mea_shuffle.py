"""Frozen 28080 follow-up: finite temporal shuffle, new paired digital seeds, R1 only."""
import itertools,math,random
from functools import lru_cache
from statistics import mean
from agent.mea_protocol import digest
from agent.mea_stim_offline import StimRecording,PROTOCOL as PARENT_PROTOCOL,assignment,validate_budget
from agent.mea_stim_offline import cell_config as parent_config

PIN='fe86f1e84f4b3239f2f02a7e5fb140503cfac02e8b4bc0587f76e0fbf753ff61'
ENDPOINTS=('visited_count','discounted_r1','final_health','final_hunger','final_thirst','terminated')
SURVIVAL=ENDPOINTS[1:]
CONTROLS=('none','before','shuffled_after')
PROTOCOL=PARENT_PROTOCOL|dict(schema='mea.stim.shuffle.followup.v1',arms=['none','before','after','shuffled_after'],
    source_pin=PIN,seed_offset=1000000,shuffle_seed=8100100,analysis_seed=8800100,
    shuffle='train_whole_window_derangement_actions_only_fixed_bootstrap',
    primary='final_visited_count_mean_across_common_before_after_heldout',
    primary_comparisons=3,survival_comparisons=15,bootstrap_resamples=20000,
    replication_min_positive_pairs=6,replication_context_direction_required=True,
    biological_replications=1,discovery_scores_pooled=False)

def cell_config(block,arm):
    if type(block) is not int or not 0<=block<8 or arm not in PROTOCOL['arms']:raise ValueError('Unregistered follow-up cell')
    cfg=parent_config(block,'after' if arm=='shuffled_after' else arm)
    for k in ('world_seed','action_seed','ppo_seed','eval_world_seed','eval_action_seed'):cfg[k]+=1000000
    return cfg|dict(arm=arm)

def train_order(block,arm):
    cell_config(block,arm);order=list(range(384))
    if arm=='shuffled_after':
        rng=random.Random(PROTOCOL['shuffle_seed']+block)
        while True:
            rng.shuffle(order)
            if all(i!=v for i,v in enumerate(order)):break
    return order+[384]

class AssignedRecording:
    def __init__(self,data,config):
        if config!=cell_config(config['block'],config['arm']):raise ValueError('Cell substitution')
        self.data=data;self.order=train_order(config['block'],config['arm']);self.window_count=data.window_count
    def snapshot(self,index,context_id):
        if type(index) is not int or not 0<=index<385:raise ValueError('Outside finite training assignment')
        return self.data.snapshot(self.order[index],context_id)
    def summary(self):
        return self.data.summary()|dict(association='offline_assigned_windows',assignment=self.order,assignment_sha256=digest(self.order))

def training_recording(source,pin,config):
    if pin!=PIN or config!=cell_config(config['block'],config['arm']):raise ValueError('Frozen binding')
    phase='after' if config['arm'] in ('after','shuffled_after') else 'before'
    return AssignedRecording(StimRecording(source,pin=pin,phase=phase,split='train'),config)

@lru_cache(maxsize=1)
def bootstrap_indices():
    rng=random.Random(PROTOCOL['analysis_seed'])
    return tuple(tuple(rng.choices(range(8),k=8)) for _ in range(20000))

def paired_stats(differences,family_size):
    if len(differences)!=8 or any(not math.isfinite(v) for v in differences):raise ValueError('Eight finite paired values required')
    if type(family_size) is not int or family_size<1:raise ValueError('Multiplicity')
    observed=mean(differences);tol=1e-12*max(1.,abs(observed))
    p=sum(abs(sum(s*x for s,x in zip(signs,differences))/8)>=abs(observed)-tol
          for signs in itertools.product((-1,1),repeat=8))/256
    draws=sorted(sum(differences[i] for i in ix)/8 for ix in bootstrap_indices())
    def q(frac):
        at=frac*(len(draws)-1);i=int(at)
        return draws[i]+(draws[min(i+1,len(draws)-1)]-draws[i])*(at-i)
    return dict(differences=list(differences),mean_difference=observed,bootstrap_95=[q(.025),q(.975)],
                p_two_sided=p,p_adjusted=min(1.,family_size*p),family_size=family_size,
                positive_pairs=sum(v>0 for v in differences),negative_pairs=sum(v<0 for v in differences))

def analyze(cells):
    indexed={}
    for cell in cells:
        cell_config(cell['block'],cell['arm']);key=cell['block'],cell['arm']
        if key in indexed:raise ValueError('Duplicate cell')
        for phase in ('pre','post'):
            if set(cell[phase])!={'before','after'}:raise ValueError('Common contexts required')
            for values in cell[phase].values():
                if any(k not in values or not isinstance(values[k],(int,float)) or not math.isfinite(values[k]) for k in ENDPOINTS):raise ValueError('Invalid endpoint')
                if values['terminated'] not in (0,1):raise ValueError('Terminal endpoint')
        indexed[key]=cell
    missing=[[b,a] for b in range(8) for a in PROTOCOL['arms'] if (b,a) not in indexed]
    if missing:return dict(status='incomplete',missing=missing,positive_evidence=False,exploration_replicated=False,temporal_order_evidence=False)
    avg=lambda b,a,p,k:mean(indexed[b,a][p][c][k] for c in ('before','after'))
    groups={a:{p:{k:mean(avg(b,a,p,k) for b in range(8)) for k in ENDPOINTS} for p in ('pre','post')} for a in PROTOCOL['arms']}
    contrasts={}
    for control in CONTROLS:
        comparisons={}
        for k in ENDPOINTS:
            d=[avg(b,'after','post',k)-avg(b,control,'post',k) for b in range(8)]
            context_means={c:mean(indexed[b,'after']['post'][c][k]-indexed[b,control]['post'][c][k] for b in range(8)) for c in ('before','after')}
            comparisons[k]=paired_stats(d,3 if k=='visited_count' else 15)|dict(context_mean_differences=context_means)
        contrasts['after_minus_'+control]=comparisons
    def positive(stat,stability=False):
        basic=stat['mean_difference']>0 and stat['bootstrap_95'][0]>0 and stat['p_adjusted']<.05
        return basic and (not stability or (stat['positive_pairs']>=6 and all(x>0 for x in stat['context_mean_differences'].values())))
    replicated=all(positive(contrasts['after_minus_'+c]['visited_count'],True) for c in ('none','before'))
    temporal=replicated and positive(contrasts['after_minus_shuffled_after']['visited_count'],True)
    r1=all(positive(contrasts['after_minus_'+c]['discounted_r1']) for c in CONTROLS)
    return dict(status='complete',positive_evidence=replicated,exploration_replicated=replicated,
        temporal_order_evidence=temporal,overall_r1_benefit=r1,arms=groups,contrasts=contrasts,
        exploration_family_size=3,survival_family_size=15,biological_replications=1,
        inference_scope='new_digital_seeds_conditional_on_same_single_recording',biological_causality_verified=False,
        survival_noninferiority_tested=False,survival_minimum_adjusted_p=15*2/256,
        survival_significance_reachable=False,
        survival_interpretation='descriptive_effects_only_n8_cannot_reach_0.05_with_two_sided_exact_test_and_bonferroni15')
