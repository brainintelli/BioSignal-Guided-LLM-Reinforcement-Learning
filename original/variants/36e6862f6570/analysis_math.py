import itertools
import numpy as np
from protocol import SESSION_ID

def completion_issues(state,baseline,stage):
 issues=[]
 if not any(x.get('stage')==stage and x.get('recording_closed_utc') for x in state.get('stage_history',[])):
  issues.append('stage_not_recorded_as_fully_closed')
 roi=baseline.get('roi_channels',[])
 if baseline.get('status')!='pass' or baseline.get('stage')!='baseline' or baseline.get('session_id')!=SESSION_ID:
  issues.append('baseline_not_qualified')
 if len(roi)<50 or len(set(roi))!=len(roi) or any(type(c) is not int or not 0<=c<1024 for c in roi):issues.append('baseline_ROI_invalid')
 if state.get('cleanup_errors') or state.get('audit_write_failed'):issues.append('cleanup_or_audit_errors')
 if state.get('r2_enabled') is not False or state.get('training_enabled') is not False:issues.append('R2_training_not_confirmed_off')
 return issues

def covered_rate(times,start,end,duration):
 a=max(0,float(start));b=min(float(duration),float(end))
 if b<=a:raise ValueError('No observation exposure')
 count=int(np.count_nonzero((np.asarray(times)>=a)&(np.asarray(times)<b)))
 return {'start':a,'end':b,'seconds':b-a,'count':count,'hz':count/(b-a)}

def trial_response(times,channels,roi,event):
 times=np.asarray(times);keep=np.isin(channels,roi)
 before=int(np.count_nonzero(keep&(times>=event-5)&(times<event-2)))
 after=int(np.count_nonzero(keep&(times>=event+2)&(times<event+5)))
 return {'before_count':before,'after_count':after,'before_hz':before/3,'after_hz':after/3,'delta_hz':(after-before)/3}

def paired_summary(rows):
 pairs=[];position_differences=[];true_first=0
 for p in sorted(set(x['pair'] for x in rows)):
  items=[x for x in rows if x['pair']==p]
  if len(items)!=2 or sorted(x['arm'] for x in items)!=['sham','true']:raise ValueError('Incomplete or duplicate pair')
  by_arm={x['arm']:x['delta_hz'] for x in items}
  position_differences.append(items[0]['delta_hz']-items[1]['delta_hz'])
  true_first+=items[0]['arm']=='true'
  pairs.append({'pair':p,'true_delta_hz':by_arm['true'],'sham_delta_hz':by_arm['sham'],'difference_hz':by_arm['true']-by_arm['sham']})
 if len(pairs)!=6:raise ValueError('Expected six pairs')
 d=np.array([x['difference_hz'] for x in pairs]);observed=float(d.mean())
 null=np.array([np.mean(d*np.array(signs)) for signs in itertools.product([-1,1],repeat=len(d))])
 p=float(np.mean(np.abs(null)>=abs(observed)-1e-10))
 balanced_p=None;assignments=0
 if true_first==3:
  pos=np.array(position_differences);balanced=[]
  for choice in itertools.combinations(range(6),3):
   signs=np.full(6,-1);signs[list(choice)]=1;balanced.append(np.mean(pos*signs))
  assignments=len(balanced);balanced_p=float(np.mean(np.abs(balanced)>=abs(observed)-1e-10))
 # This is an exchangeability-based descriptive test, not the exact distribution
 # of the constrained 3/3 order randomization; within-session dependence remains.
 return {'pairs':pairs,'mean_true_minus_sham_hz':observed,'median_true_minus_sham_hz':float(np.median(d)),
  'exact_sign_flip_two_sided_p':p,'sign_flip_assumption':'paired contrasts exchangeable under independent sign flips; exploratory within-session only',
  'balanced_order_randomization_assignments':assignments,'balanced_order_randomization_two_sided_p':balanced_p,
  'balanced_randomization_assumption':'sharp no-effect null, fixed observed outcomes, no carryover; conditional on three true-first pairs',
  'positive_pairs':int((d>0).sum()),'negative_pairs':int((d<0).sum())}

def rate_bins(times,duration,width):
 edges=np.arange(0,duration,width)
 edges=np.append(edges,duration)
 return (edges[:-1]+edges[1:])/2,np.histogram(times,bins=edges)[0]/np.diff(edges)
