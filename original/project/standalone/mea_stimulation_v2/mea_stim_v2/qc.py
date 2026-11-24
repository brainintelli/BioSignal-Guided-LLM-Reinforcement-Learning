"""Closed-file v2 admission QC. Failures never authorize further stimulation.

All timestamps are relative to actual first frames. Fixed candidate selection,
independent holdout, and pilot-only technical exclusions follow the registered
protocol. These are offline technical gates, not online physiological monitoring.
"""
import math
import numpy as np

from .analysis import trial_response
from .config import digest
from .recording import RecordingError


def _stats(data):
    a=np.asarray(data,dtype=float)
    if a.ndim!=2 or not a.size or not np.all(np.isfinite(a)) or np.any((a<0)|(a>1023)):
        raise RecordingError('Invalid raw ADC sample')
    median=np.median(a,axis=1)
    return {'median':median,'mad':1.4826*np.median(np.abs(a-median[:,None]),axis=1),
            'sd':np.std(a,axis=1),'rails':np.any((a==0)|(a==1023),axis=1)}


def stability(before, after):
    """95% must jointly pass MAD ratio, median shift, and no ADC rails."""
    if np.shape(before)!=np.shape(after):raise RecordingError('Unequal stability window dimensions')
    b=_stats(before);a=_stats(after)
    ratio=np.divide(a['mad'],b['mad'],out=np.full_like(a['mad'],np.nan),where=b['mad']>0)
    shift=np.abs(a['median']-b['median'])
    passed=(b['mad']>0)&(a['mad']>0)&(ratio>=.67)&(ratio<=1.5)&(shift<=5)&(~b['rails'])&(~a['rails'])
    fraction=float(np.mean(passed))
    return {'passed':bool(int(passed.sum())*100>=95*len(passed)),
            'fraction_channels_passed':fraction,'channel_pass':passed.tolist(),
            'mad_ratios':[float(v) if math.isfinite(v) else None for v in ratio],
            'median_shifts_adc':shift.tolist(),'before_mad_adc':b['mad'].tolist(),
            'after_mad_adc':a['mad'].tolist(),'before_sd_adc':b['sd'].tolist(),'after_sd_adc':a['sd'].tolist(),
            'before_saturated':b['rails'].tolist(),'after_saturated':a['rails'].tolist()}


def _sample(recording, roi, start, seconds=1):
    if start<0 or start+seconds>recording.duration+1e-9:raise RecordingError('Insufficient requested window coverage')
    data=recording.sample(roi,start,seconds)
    if np.shape(data)!=(len(roi),int(round(recording.fs*seconds))):
        raise RecordingError('Raw sample window truncated or sampling differs')
    return data


def _common(recording,config,stage):
    result={'stage':stage,'status':'failed','errors':[], 'source':dict(recording.source),
            'metadata':dict(recording.metadata),'session_id':config.session_id,
            'duration_seconds':float(recording.duration),'r2_enabled':config.r2_enabled,
            'training_enabled':config.training_enabled,'events':recording.events}
    errors=result['errors']
    try:
        expected=config.validate_binding()
        result['binding_sha256']=config.binding_sha256()
        if sorted(recording.mapping)!=expected:errors.append('route_mapping_mismatch')
    except (ValueError,TypeError,AttributeError,OSError) as e:errors.append('invalid_binding: '+str(e))
    m=recording.metadata
    for key,value in [('chip_id',config.chip_id),('fs',20000.),('gain',512.),('hpf',300.),
                      ('threshold_value',config.threshold_value),('threshold_h5_dataset',config.threshold_h5_dataset),('frame_gaps',0)]:
        if key not in m or m[key]!=value:errors.append('metadata_mismatch_'+key)
    if not math.isfinite(m.get('lsb_uv',float('nan'))) or m.get('lsb_uv',0)<=0:errors.append('missing_lsb_readback')
    if config.r2_enabled is not False or config.training_enabled is not False:errors.append('R2_or_training_enabled')
    if not math.isfinite(recording.duration) or recording.duration<=0:errors.append('invalid_duration')
    for e in recording.events:
        if e.get('eventtype')!=1 or e.get('session')!=config.session_id:errors.append('event_type_or_session_mismatch')
        if not math.isfinite(e.get('time_s',float('nan'))) or not 0<=e['time_s']<recording.duration:errors.append('event_outside_recording')
    if any(b.get('time_s',-1)<=a.get('time_s',-1) for a,b in zip(recording.events,recording.events[1:])):errors.append('event_order_invalid')
    return result


def _finish(result):
    result['errors']=list(dict.fromkeys(result['errors']))
    result['status']='pass' if not result['errors'] else 'failed'
    return result


def _roi(recording,config,channels):
    roi=[int(c) for c in channels]
    if len(roi)!=len(set(roi)) or not set(roi)<=set(recording.channels.tolist()):raise RecordingError('Invalid fixed ROI channels')
    stim=[c for c,e in recording.mapping if e==config.stim_electrode]
    if len(stim)!=1:raise RecordingError('Stimulation electrode missing/ambiguous')
    if stim[0] in roi:raise RecordingError('Stimulating electrode cannot be in readout ROI')
    if len(roi)<50:raise RecordingError('Fewer than 50 readout channels')
    return sorted(roi),stim[0]


def _rate(recording,roi,start,seconds):
    if start<0 or start+seconds>recording.duration+1e-9:raise RecordingError('Insufficient rate window coverage')
    t=recording.spike_times
    return float(np.count_nonzero(np.isin(recording.spike_channels,roi)&(t>=start)&(t<start+seconds))/seconds)


def _trend(recording,roi):
    selected=recording.spike_times[np.isin(recording.spike_channels,roi)]
    starts=np.arange(0,recording.duration,30.)
    return [{'start_s':float(a),'end_s':float(min(a+30,recording.duration)),
             'rate_hz':float(np.count_nonzero((selected>=a)&(selected<min(a+30,recording.duration)))/(min(a+30,recording.duration)-a))}
            for a in starts]


def _holdout(recording,roi,connected):
    checks=[]
    for offset in range(630,1171,30):
        t=connected+offset
        before=_sample(recording,roi,t-3)
        windows=[stability(before,_sample(recording,roi,t+post)) for post in (2,10)]
        checks.append({'time_s':t,'passed':all(w['passed'] for w in windows),'windows':windows})
    count=sum(x['passed'] for x in checks)
    return {'passed':count>=18,'passed_count':count,'total':19,'checks':checks}


def _reference(recording,roi,connected):
    rates=[_rate(recording,roi,connected+i,30.) for i in range(600,1200,30)]
    return float(np.median(rates)),rates


def baseline(recording,config):
    result=_common(recording,config,'baseline');errors=result['errors']
    result.update(candidate_roi=[],roi_channels=[],holdout19=None,reference_rate=None)
    try:
        ev=recording.events
        if len(ev)!=2 or [e.get('phase') for e in ev]!=['disconnected','connected']:
            errors.append('baseline_markers_incorrect');return _finish(result)
        if len({e['id'] for e in ev})!=2:errors.append('baseline_marker_ids_duplicate')
        disconnected,connected=[e['time_s'] for e in ev]
        if connected-disconnected<300-1e-9 or recording.duration-connected<1200-1e-9:
            errors.append('baseline_5min_disconnected_20min_connected_required');return _finish(result)
        if errors:return _finish(result)
        channels=recording.channels.tolist()
        t=recording.spike_times
        counts=np.array([np.count_nonzero((recording.spike_channels==c)&(t>=connected+120)&(t<connected+600)) for c in channels])
        valid=(counts/480.>=.05)
        calibration=[]
        for offset in range(150,571,30):
            stats=_stats(_sample(recording,channels,connected+offset))
            valid &= (~stats['rails'])&(stats['mad']>0)
            calibration.append({'time_s':connected+offset,'mad_adc':stats['mad'].tolist(),'sd_adc':stats['sd'].tolist(),
                                'saturated':stats['rails'].tolist()})
        stim=[c for c,e in recording.mapping if e==config.stim_electrode]
        if len(stim)!=1:raise RecordingError('Stimulation electrode missing/ambiguous')
        candidate=[c for c,passed in zip(channels,valid) if passed and c!=stim[0]]
        result.update(candidate_roi=candidate,roi_channels=candidate,candidate_roi_sha256=digest(candidate),
                      calibration_rates_hz=(counts/480.).tolist(),calibration_channels=channels,
                      calibration_samples=calibration,connected_time_s=connected)
        candidate,_=_roi(recording,config,candidate)
        holdout=_holdout(recording,candidate,connected)
        reference,rates=_reference(recording,candidate,connected)
        result.update(holdout19=holdout,reference_rate=reference,reference_30s_rates=rates,
                      activity_trend_30s=_trend(recording,candidate))
        if not holdout['passed']:errors.append('holdout_less_than_18_of_19')
        if reference<=0 or not math.isfinite(reference):errors.append('zero_or_invalid_reference_rate')
    except (RecordingError,ValueError,TypeError,KeyError,IndexError,OSError) as e:errors.append('baseline_qc_error: '+str(e))
    return _finish(result)


def _event_checks(recording,roi,stim,event):
    t=event['time_s'];before=_sample(recording,roi,t-3)
    recovery=[stability(before,_sample(recording,roi,t+post)) for post in (2,10)]
    selected=_sample(recording,[stim],t,.6)[0].astype(float)
    center=float(np.median(_sample(recording,[stim],t-3)))
    uv=recording.metadata['lsb_uv']
    rails=np.flatnonzero((selected==0)|(selected==1023))
    crossings=np.flatnonzero(np.abs(selected-center)*uv>=1000)
    onset=float(crossings[0]/recording.fs) if len(crossings) else None
    last=float(rails[-1]/recording.fs) if len(rails) else None
    near=_sample(recording,roi,t,.6)
    near_rails=np.any((near==0)|(near==1023),axis=1)
    artifact_ok=(onset is not None and onset<=.010 and (last is None or last<=.500)) if event['arm']=='true' else (onset is None and last is None)
    return {'id':event['id'],'arm':event['arm'],'time_s':t,'recovery':recovery,
            'passed':bool(artifact_ok and all(x['passed'] for x in recovery)),
            'artifact_passed':bool(artifact_ok),'selected_first_1000uv_crossing_s':onset,
            'selected_last_rail_s':last,'selected_peak_abs_uv_0_600ms':float(np.max(np.abs(selected-center))*uv),
            'roi_saturated_channels_0_600ms':[c for c,r in zip(roi,near_rails) if r]}


def _postchecks(recording,config,result,roi,stim,expected_interval,expected_post,reference):
    errors=result['errors'];events=recording.events
    if abs((events[1]['time_s']-events[0]['time_s'])-expected_interval)>.1000000001:errors.append('event_interval_outside_100ms')
    pre_required=60 if result['stage']=='pilot' else 30
    if events[0]['time_s']<pre_required-1e-9:errors.append('initial_pre_recording_short')
    if recording.duration-events[-1]['time_s']<expected_post-1e-9:errors.append('post_recording_short')
    result['pulses']=[];result['trials']=[]
    for event in events:
        check=_event_checks(recording,roi,stim,event);result['pulses'].append(check)
        if not check['artifact_passed']:errors.append(event['arm']+'_artifact_failed')
        if not all(x['passed'] for x in check['recovery']):errors.append(event['arm']+'_MAD_recovery_failed')
        result['trials'].append({k:event[k] for k in ('id','arm','pair','time_s')} | trial_response(
            recording.spike_times,recording.spike_channels,roi,event['time_s'],recording.duration))
    final=stability(_sample(recording,roi,events[0]['time_s']-10),_sample(recording,roi,recording.duration-3))
    result['final_recovery']=final
    if not final['passed']:errors.append('final_MAD_recovery_failed')
    if not math.isfinite(reference) or reference<=0:raise RecordingError('Positive frozen reference rate required')
    rates=[_rate(recording,roi,events[-1]['time_s']+start,30) for start in range(0,expected_post,30)]
    abnormal=[r<reference*.25 or r>reference*4 for r in rates]
    if any(a and b for a,b in zip(abnormal,abnormal[1:])):errors.append('consecutive_activity_outliers')
    result.update(reference_rate=reference,post_event_30s_rates=rates,activity_outliers=abnormal,
                  activity_trend_30s=_trend(recording,roi))


def pilot(recording,baseline_recording,config,candidate_roi):
    result=_common(recording,config,'pilot');errors=result['errors']
    result.update(roi=[],roi_channels=[],holdout19=None,reference_rate=None)
    try:
        ev=recording.events
        if len(ev)!=2 or [e.get('arm') for e in ev]!=['sham','true'] or any(e.get('pair')!=0 or e.get('phase')!='pilot' for e in ev):
            errors.append('pilot_events_incorrect');return _finish(result)
        if len({e['id'] for e in ev})!=2:errors.append('pilot_marker_ids_duplicate')
        baseline_result=baseline(baseline_recording,config)
        result['baseline_source']=baseline_result['source']
        if baseline_result['status']!='pass':errors.append('baseline_not_qualified')
        if sorted(candidate_roi)!=baseline_result['candidate_roi']:errors.append('candidate_set_not_frozen_baseline')
        if errors:return _finish(result)
        candidate,stim=_roi(recording,config,candidate_roi)
        near=_sample(recording,candidate,ev[1]['time_s'],.6)
        rail=np.any((near==0)|(near==1023),axis=1)
        roi=[c for c,bad in zip(candidate,rail) if not bad]
        result.update(roi=roi,roi_channels=roi,roi_sha256=digest(roi),
                      excluded_true_saturation_channels=[c for c,bad in zip(candidate,rail) if bad])
        roi,stim=_roi(recording,config,roi)
        connected=baseline_result['connected_time_s']
        holdout=_holdout(baseline_recording,roi,connected)
        reference,rates=_reference(baseline_recording,roi,connected)
        result.update(holdout19=holdout,reference_rate=reference,reference_30s_rates=rates)
        if not holdout['passed']:errors.append('final_roi_holdout_less_than_18_of_19')
        _postchecks(recording,config,result,roi,stim,60,600,reference)
    except (RecordingError,ValueError,TypeError,KeyError,IndexError,OSError) as e:errors.append('pilot_qc_error: '+str(e))
    return _finish(result)


def pair(recording,config,pair_spec,roi,reference_rate,expected_post=60):
    result=_common(recording,config,'pair');errors=result['errors']
    result.update(pair=pair_spec.get('pair'),roi=list(roi),roi_channels=list(roi),roi_sha256=digest(list(roi)),trials=[])
    try:
        if expected_post not in (60,600):raise RecordingError('Post duration must be 60 or 600 seconds')
        if type(pair_spec.get('pair')) is not int or not 1<=pair_spec['pair']<=10:raise RecordingError('Pair number outside frozen 1..10 design')
        if pair_spec['pair']==10 and expected_post!=600:raise RecordingError('Final pair requires complete 600-second recovery')
        if type(pair_spec.get('interval_s')) is not int or not 45<=pair_spec['interval_s']<=75:raise RecordingError('Invalid frozen interval')
        ev=recording.events;expected=pair_spec['events']
        if len(ev)!=2 or len(expected)!=2 or any(any(e.get(k)!=wanted.get(k) for k in ('id','arm','pair')) for e,wanted in zip(ev,expected)):
            errors.append('pair_events_mismatch');return _finish(result)
        if sorted(e.get('arm','') for e in ev)!=['sham','true'] or any(e.get('pair')!=pair_spec['pair'] for e in ev):
            errors.append('pair_arm_or_number_invalid')
        if errors:return _finish(result)
        roi,stim=_roi(recording,config,roi)
        _postchecks(recording,config,result,roi,stim,pair_spec['interval_s'],expected_post,float(reference_rate))
    except (RecordingError,ValueError,TypeError,KeyError,IndexError,OSError) as e:errors.append('pair_qc_error: '+str(e))
    return _finish(result)
