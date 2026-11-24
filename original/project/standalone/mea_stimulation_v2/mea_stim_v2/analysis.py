"""Prespecified delayed response and balanced-order randomization; no hardware I/O."""
from itertools import combinations
import math
import numpy as np


def trial_response(spike_times, spike_channels, roi, event_s, duration):
    if len(roi)==0 or not math.isfinite(event_s) or event_s-5 < 0 or event_s+5 > duration:
        raise ValueError('Complete pre -5..-2 and post +2..+5 windows required')
    times=np.asarray(spike_times)[np.isin(spike_channels,roi)]
    pre=int(np.count_nonzero((times>=event_s-5)&(times<event_s-2)))
    post=int(np.count_nonzero((times>=event_s+2)&(times<event_s+5)))
    return {'pre_count':pre,'post_count':post,'pre_rate_hz':pre/3.,'post_rate_hz':post/3.,
            'delta_hz':(post-pre)/3.,'pre_window_s':[event_s-5,event_s-2],
            'post_window_s':[event_s+2,event_s+5],'early_0_2s_excluded':True}


def summarize_pairs(pair_results):
    """Retain all pairs. Only 10 qualified pairs with balanced first arms earn a p value.

    Under the sharp no-effect/no-carryover null, hold the first/second position
    outcomes fixed and assign exactly five of the ten first positions to true.
    This is 252 assignments, not independent sign flips of observed effects.
    """
    issues=[]; rows=[]; position_differences=[]; true_first=0
    result={'status':'incomplete_or_failed','pairs':rows,'errors':issues,'p_two_sided':None,
            'randomization_assignments':0,'repeat_response_candidate':False,
            'r2_enabled':False,'training_enabled':False,
            'limitations':['Single organoid; trials are not biological replicates.',
                          '0-2 seconds excluded; early responses cannot be ruled out.',
                          'Randomization assumes no cross-trial carryover; inspect time trends.',
                          'Increased or decreased rate does not establish improvement or learning.']}
    if len(pair_results)!=10: issues.append('requires_complete_10_pairs')
    seen=[];event_ids=[]
    for q in pair_results:
        number=q.get('pair');seen.append(number)
        if q.get('status')!='pass':issues.append('pair_%s_technical_failure'%number)
        trials=q.get('trials',[])
        event_ids.extend(t.get('id') for t in trials)
        if any(t.get('pair')!=number for t in trials):issues.append('pair_%s_trial_pair_mismatch'%number)
        if len(trials)!=2 or sorted(t.get('arm','') for t in trials)!=['sham','true']:
            issues.append('pair_%s_invalid_trials'%number);continue
        trials=sorted(trials,key=lambda t:t.get('time_s',float('nan')))
        if not all(math.isfinite(t.get('delta_hz',float('nan'))) and math.isfinite(t.get('time_s',float('nan'))) for t in trials) or trials[0]['time_s']>=trials[1]['time_s']:
            issues.append('pair_%s_invalid_values'%number);continue
        by_arm={t['arm']:t for t in trials}
        true_first+=int(trials[0]['arm']=='true')
        position_differences.append(trials[0]['delta_hz']-trials[1]['delta_hz'])
        rows.append({'pair':number,'first_arm':trials[0]['arm'],
                     'true_pre_rate_hz':by_arm['true']['pre_rate_hz'],'true_post_rate_hz':by_arm['true']['post_rate_hz'],
                     'sham_pre_rate_hz':by_arm['sham']['pre_rate_hz'],'sham_post_rate_hz':by_arm['sham']['post_rate_hz'],
                     'true_delta_hz':by_arm['true']['delta_hz'],'sham_delta_hz':by_arm['sham']['delta_hz'],
                     'difference_hz':by_arm['true']['delta_hz']-by_arm['sham']['delta_hz']})
    if sorted(x for x in seen if isinstance(x,int))!=list(range(1,11)) or len(set(seen))!=len(seen):issues.append('pair_identity_incomplete_or_duplicate')
    if len(set(event_ids))!=len(event_ids) or any(type(i) is not int for i in event_ids):issues.append('event_identity_duplicate_or_invalid')
    if true_first!=5 or len(rows)!=10:issues.append('requires_5_true_first_5_sham_first')
    if rows:
        differences=np.array([x['difference_hz'] for x in rows])
        result['mean_difference_hz']=float(differences.mean())
    if issues:return result
    rows.sort(key=lambda r:r['pair'])
    differences=np.array([x['difference_hz'] for x in rows]);observed=float(differences.mean())
    distribution=[]
    for chosen in combinations(range(10),5):
        signs=np.full(10,-1.);signs[list(chosen)]=1.
        distribution.append(float(np.mean(signs*np.array(position_differences))))
    p=sum(abs(x)>=abs(observed)-1e-12 for x in distribution)/252.
    positive=int(np.sum(differences>0));negative=int(np.sum(differences<0))
    first=float(differences[:5].mean());last=float(differences[5:].mean())
    same_halves=first*last>0
    result.update(status='complete_qualified',p_two_sided=p,randomization_assignments=252,
                  mean_difference_hz=observed,direction='increase' if observed>0 else 'decrease' if observed<0 else 'zero',
                  positive_pairs=positive,negative_pairs=negative,first_half_mean_hz=first,last_half_mean_hz=last,
                  halves_same_direction=same_halves,randomization_distribution_hz=distribution,
                  repeat_response_candidate=bool(p<=.05 and max(positive,negative)>=8 and same_halves))
    return result
