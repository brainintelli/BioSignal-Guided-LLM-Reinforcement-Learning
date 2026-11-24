import numpy as np

def baseline_gate(*,channels,counts_before,counts_after,seconds_before,seconds_after,
                  sd_before,sd_after,median_before,median_after,rails_before,rails_after,excluded):
 channels=np.asarray(channels);before=np.asarray(counts_before);after=np.asarray(counts_after)
 ratio=np.asarray(sd_after)/np.maximum(np.asarray(sd_before),1e-9)
 shift=np.abs(np.asarray(median_after)-np.asarray(median_before))
 candidates=(before>=6)&~np.isin(channels,excluded)
 stable=(ratio>=.67)&(ratio<=1.5)&(shift<=5)&(np.asarray(rails_before)==0)&(np.asarray(rails_after)==0)
 roi=candidates&stable
 stable_fraction=float(roi.sum()/max(candidates.sum(),1))
 pop_before=float(before[roi].sum()/seconds_before);pop_after=float(after[roi].sum()/seconds_after)
 rate_ratio=pop_after/pop_before if pop_before else 0
 errors=[]
 if roi.sum()<50 or np.count_nonzero(roi&(after>=6))<50:errors.append('fewer_than_50_active_stable_channels')
 if stable_fraction<.8:errors.append('less_than_80pct_active_candidates_stable')
 if not .5<=rate_ratio<=2:errors.append('connected_population_rate_ratio_outside_0p5_to_2')
 return {'status':'pass' if not errors else 'failed','errors':errors,
  'candidate_channels':int(candidates.sum()),'stable_fraction':stable_fraction,
  'roi_channels':channels[roi].astype(int).tolist(),'connected_active_roi_channels':int(np.count_nonzero(roi&(after>=6))),
  'population_rate_before_hz':pop_before,'population_rate_connected_hz':pop_after,'population_rate_ratio':rate_ratio,
  'sd_ratios':ratio.tolist(),'median_shifts_adc':shift.tolist()}

def recovery_gate(before_sd,after_sd,after_rails):
 ratios=np.asarray(after_sd)/np.maximum(np.asarray(before_sd),1e-9)
 good=(ratios>=.67)&(ratios<=1.5)&(np.asarray(after_rails)==0)
 fractions=np.mean(good,axis=1)
 return {'passed':bool(np.all(fractions>=.95)),'fraction_channels_recovered_each_window':fractions.tolist(),
         'sd_ratio_quantiles_each_window':np.percentile(ratios,[0,5,50,95,100],axis=1).T.tolist()}
