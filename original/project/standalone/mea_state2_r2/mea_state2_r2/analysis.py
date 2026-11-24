"""Preregistered paired digital-seed outcomes, never a biological efficacy claim."""
import itertools
import math
import random
from functools import lru_cache
from statistics import mean
from .protocol import ARMS, cell

METRICS=('r1_sum','visited_count','final_health','final_hunger','final_thirst','elapsed_ticks','terminated')
CONTRASTS=(('C','B'),('C','A'),('C','A0'),('C','D'),('B','A'))


@lru_cache(maxsize=1)
def indices():
    rng=random.Random(18000100)
    return tuple(tuple(rng.choices(range(8),k=8)) for _ in range(20000))


def paired(values):
    if len(values)!=8 or any(not math.isfinite(v) for v in values):
        raise ValueError('Eight finite paired seed differences required')
    observed=mean(values)
    tol=1e-12*max(1.,abs(observed))
    p=sum(abs(sum(s*v for s,v in zip(signs,values))/8)>=abs(observed)-tol
          for signs in itertools.product((-1,1),repeat=8))/256
    draws=sorted(sum(values[i] for i in sample)/8 for sample in indices())
    def quantile(q):
        index=q*(len(draws)-1);lo=int(index)
        return draws[lo]+(draws[min(lo+1,len(draws)-1)]-draws[lo])*(index-lo)
    return dict(differences=values,mean_difference=observed,bootstrap_95=[quantile(.025),quantile(.975)],
                p_two_sided=p,positive_pairs=sum(v>0 for v in values),negative_pairs=sum(v<0 for v in values))


def analyze(rows,expected_registration=None):
    if (not isinstance(expected_registration,str) or len(expected_registration)!=64
            or any(c not in '0123456789abcdef' for c in expected_registration)):
        raise ValueError('Explicit frozen registration digest required')
    indexed={}
    for row in rows:
        if row.get('registration_sha256')!=expected_registration or row.get('smoke') is not False:
            raise ValueError('Mixed/missing registration or technical smoke outcome')
        cell(row['block'],row['arm'])
        key=(row['block'],row['arm'])
        if key in indexed:raise ValueError('Duplicate experiment cell')
        for phase in ('after','before'):
            for horizon in ('full','first16'):
                metric=row['metrics'][phase][horizon]
                if any(type(metric.get(k)) not in (int,float) or not math.isfinite(metric[k]) for k in METRICS):
                    raise ValueError('Nonfinite or missing outcome')
        indexed[key]=row
    missing=[[b,a] for b in range(8) for a in ('A0',*ARMS)
             if (b,a) not in indexed or indexed[b,a].get('passed') is not True]
    if missing:return dict(status='incomplete',missing_or_failed=missing,digital_state2_benefit_detected=False,
                            wetware_closed_loop_effect_verified=False)
    summaries={phase:{arm:{k:mean(indexed[b,arm]['metrics'][phase]['full'][k] for b in range(8)) for k in METRICS}
                      for arm in ('A0',*ARMS)} for phase in ('after','before')}
    contrasts={}
    secondary={}
    for phase,target in [('after',contrasts),('before',secondary)]:
        for treatment,control in CONTRASTS:
            target[treatment+'_minus_'+control]={k:paired([indexed[b,treatment]['metrics'][phase]['full'][k]-
                indexed[b,control]['metrics'][phase]['full'][k] for b in range(8)]) for k in METRICS}
    def supported(stat):
        return stat['mean_difference']>0 and stat['bootstrap_95'][0]>0 and stat['p_two_sided']<.05 and stat['positive_pairs']>=6
    first16={arm:{k:mean(indexed[b,arm]['metrics']['after']['first16'][k] for b in range(8)) for k in METRICS}
             for arm in ('A0',*ARMS)}
    return dict(status='complete',cells=len(rows),registration_sha256=expected_registration,primary='after/r1_sum/C_minus_B',primary_tests=1,
        arms=summaries,first16_after=first16,contrasts=contrasts,secondary_before_contrasts=secondary,
        digital_state2_benefit_detected=supported(contrasts['C_minus_B']['r1_sum']),
        temporal_order_supported=supported(contrasts['C_minus_D']['r1_sum']),
        secondary_tests_interpretation='exploratory_unadjusted_no_confirmatory_claim',
        wetware_closed_loop_effect_verified=False,r2_source='simulation_from_game_state',
        state2_source='historical_28080',biological_replications=1,
        inference_scope='eight_digital_seed_pairs_conditional_on_one_archival_recording',
        rows=rows)
