"""Frozen exploratory comparison; pure functions, no device or ML imports."""
import itertools,math,random,statistics

ARMS=('baseline','state','reward','both','shuffled')
STATE_ARMS={'state','both','shuffled'};REWARD_ARMS={'reward','both','shuffled'}
STATE_FEATURES=('spike_count','rate_hz_per_channel','active_channel_fraction','mean_abs_amplitude_uv')
PROTOCOL=dict(schema='mea.22554.state_reward.v1',chip='22554',source_sha256='45a341c80d7ac60eb2b9ba846dce926f9e7fd4b6fb27f0b7b8c963798640f84c',
    window_seconds=1,calibration=[0,60],train=[60,188],train_bootstrap=188,embargo=[189,199],evaluation=[199,295],eval_bootstrap=295,
    windows=298,blocks=8,arms=list(ARMS),train_decisions=128,eval_decisions=96,eval_worlds=2,aux_weight=.05,gamma=.99985,gae_lambda=.999,
    initial_seed=5290000,train_world_seed=5250000,eval_world_seed=5260000,train_action_seed=5270000,eval_action_seed=5280000,ppo_seed=5300000,shuffle_seed=5310000,
    analysis_seed=5320000,bootstrap_resamples=20000,primary_comparisons=['both-baseline','both-shuffled'],
    stimulation=False,realtime_training=False,source_full_qc_passed=False,biological_validated=False,source_association='prerecorded_action_independent',
    primary='post_mean_discounted_original_r1_at_96_decisions',uncertainty_unit='paired_digital_seed_block_conditional_on_one_recording')

def finite(x):
    if isinstance(x,bool) or not isinstance(x,(int,float)) or not math.isfinite(x):raise ValueError('finite numeric value required')
    return float(x)

def assignment(arm,split,block):
    if arm not in ARMS or split not in ('train','eval') or type(block) is not int or not 0<=block<8:raise ValueError('unregistered assignment')
    base=list(range(60,188)) if split=='train' else list(range(199,295));boundary=188 if split=='train' else 295
    if arm=='shuffled':
        rng=random.Random(PROTOCOL['shuffle_seed']+2*block+int(split=='eval'));order=base.copy()
        while True:
            rng.shuffle(order)
            if all(a!=b for a,b in zip(order,base)):break
        base=order
    return base+[boundary]

def robust(rows):
    center=[statistics.median(c) for c in zip(*rows)];scale=[1.4826*statistics.median(abs(r[j]-center[j]) for r in rows) for j in range(len(center))]
    return center,scale

def proximity_features(w):return [math.log1p(finite(w['rate_hz_per_channel'])),finite(w['active_channel_fraction']),finite(w['coincident_event_fraction'])]

def raw_proximity(w,profile):
    x=proximity_features(w);z=[(x[i]-c)/s for i,(c,s) in enumerate(zip(profile['aux_center'],profile['aux_scale'])) if s>1e-12]
    if not z:raise ValueError('all auxiliary features constant')
    return math.exp(-.5*sum(v*v for v in z)/len(z))

def fit_profile(windows):
    if len(windows)<296:raise ValueError('finite disjoint data budget unavailable')
    fit=windows[:60];state=[[finite(w[n]) for n in STATE_FEATURES] for w in fit];sc,ss=robust(state)
    ac,aas=robust([proximity_features(w) for w in fit])
    p=dict(fit_indices=list(range(60)),state_names=list(STATE_FEATURES),state_center=sc,state_scale=[s if s>1e-12 else 1. for s in ss],
        state_zero_scale=[i for i,s in enumerate(ss) if s<=1e-12],aux_center=ac,aux_scale=aas,aux_inactive=[i for i,s in enumerate(aas) if s<=1e-12],
        normalization_clip=5.,experimental=True,biological_validated=False)
    mu=statistics.mean(raw_proximity(w,p) for w in fit);p.update(aux_probability_mean=mu,aux_denominator=max(mu,1-mu));return p

def score(w,profile):
    p=raw_proximity(w,profile);u=(p-profile['aux_probability_mean'])/profile['aux_denominator']
    if not -1-1e-12<=u<=1+1e-12:raise ValueError('auxiliary score outside bounds')
    return dict(p=p,u=max(-1.,min(1.,u)))

def features(w,profile):return [max(-5.,min(5.,(finite(w[n])-c)/s)) for n,c,s in zip(STATE_FEATURES,profile['state_center'],profile['state_scale'])]

def reward_record(r1,s,arm,*,training):
    if arm not in ARMS or type(training) is not bool:raise ValueError('arm/phase required')
    r1=finite(r1);u=finite(s['u']);p=finite(s['p'])
    if not -1<=u<=1 or not 0<=p<=1:raise ValueError('score bounds')
    weight=PROTOCOL['aux_weight'] if training and arm in REWARD_ARMS else 0.
    return dict(r1=r1,r1_weight=1.,aux_probability=p,aux_raw=u,aux_weight=weight,total_reward=r1+weight*u,
        auxiliary_enabled=weight!=0,experimental=True,biological_validated=False,action_contingent=False,
        reward_contract='additive_experimental_v1',association='prerecorded_current_window',phase='train' if training else 'evaluation')

def episode_metrics(rows):
    if not rows:raise ValueError('empty trajectory')
    ticks=0;discounted=0.;raw=0.;terminals=0
    for r in rows:
        if type(r['duration']) is not int or r['duration']<=0:raise ValueError('native positive duration required')
        reward=finite(r['r1']);discounted+=PROTOCOL['gamma']**ticks*reward;raw+=reward;ticks+=r['duration'];terminals+=int(r['terminal'])
    return dict(discounted_r1=discounted,r1_sum=raw,ticks=ticks,decisions=len(rows),natural_terminals=terminals)

def paired_stats(differences):
    if len(differences)!=8:raise ValueError('exactly eight paired seed blocks')
    d=[finite(x) for x in differences];observed=statistics.mean(d);eps=1e-12*max(1.,abs(observed))
    p=sum(abs(sum(x*s for x,s in zip(d,signs))/8)>=abs(observed)-eps for signs in itertools.product((-1,1),repeat=8))/256
    rng=random.Random(PROTOCOL['analysis_seed']);draws=sorted(sum(rng.choices(d,k=8))/8 for _ in range(PROTOCOL['bootstrap_resamples']))
    def q(value):
        at=value*(len(draws)-1);i=int(at);return draws[i]+(draws[min(i+1,len(draws)-1)]-draws[i])*(at-i)
    return dict(differences=d,mean_difference=observed,bootstrap_95=[q(.025),q(.975)],p_two_sided=p,p_adjusted_primary=min(1.,2*p))

def analyze(cells):
    indexed={};expected=set(itertools.product(range(8),ARMS))
    for c in cells:
        key=(c['block'],c['arm'])
        if key not in expected or key in indexed:raise ValueError('duplicate/unregistered cell')
        if c['verified'] is not True:raise ValueError('unverified cell')
        means={}
        for phase in ('pre','post'):
            if len(c[phase])!=2:raise ValueError('two paired evaluation worlds required')
            means[phase]=statistics.mean(finite(x['discounted_r1']) for x in c[phase])
        indexed[key]=means
    missing=sorted(expected-set(indexed))
    if missing:return dict(status='incomplete',missing=[list(k) for k in missing],primary=None)
    def contrast(a,b):
        d=[indexed[i,a]['post']-indexed[i,b]['post'] for i in range(8)]
        gain=[d[i]-(indexed[i,a]['pre']-indexed[i,b]['pre']) for i in range(8)]
        return dict(post=paired_stats(d),gain=paired_stats(gain))
    primary={f'both-{b}':contrast('both',b) for b in ('baseline','shuffled')}
    secondary={f'{a}-{b}':contrast(a,b) for a,b in [('state','baseline'),('reward','baseline'),('both','state')]}
    for v in secondary.values():
        for p in v.values():p.pop('p_adjusted_primary')
    for v in primary.values():v['gain'].pop('p_adjusted_primary')
    return dict(status='complete',replication_units=8,biological_recordings=1,uncertainty_unit=PROTOCOL['uncertainty_unit'],
        arms={a:{phase:statistics.mean(indexed[b,a][phase] for b in range(8)) for phase in ('pre','post')} for a in ARMS},
        primary=primary,secondary=secondary,biological_effect_verified=False)
