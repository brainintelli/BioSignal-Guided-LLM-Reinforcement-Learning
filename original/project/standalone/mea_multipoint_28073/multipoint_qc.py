"""Closed-file capability checks for the separately authorized 20-point train.

This is not v2 baseline qualification, a sham comparison, or an efficacy test.
No hardware or transport modules are imported. The caller freezes the readout
after excluding all 20 stimulating channels; no post-stimulation selection occurs.
"""
from pathlib import Path
import math
import re
import sys

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parent.parent/'mea_stimulation_v2'))
from mea_stim_v2.recording import H5Recording
from mea_stim_v2.qc import _common, _stats, stability
from mea_stim_v2.config import digest


def _sample(recording,channels,start,seconds=1.):
    if start<0 or start+seconds>recording.duration+1e-9:
        raise ValueError('Insufficient actual frame coverage for raw window')
    data=recording.sample(channels,start,seconds)
    if np.shape(data)!=(len(channels),round(recording.fs*seconds)):
        raise ValueError('Truncated raw sample')
    return data


def _rate(recording,roi,start,seconds):
    if start<0 or start+seconds>recording.duration+1e-9:raise ValueError('Insufficient rate window')
    t=recording.spike_times
    mask=np.isin(recording.spike_channels,roi)&(t>=start)&(t<start+seconds)
    return float(np.count_nonzero(mask)/seconds)


def analyze(path,config,roi,events15,reference_rate,expected_post=60):
    """Check one point's 15-event train and preserve the supplied fixed ROI.

    events15 is the frozen message/identity schedule. Integer group, pulse and
    electrode values are compared to the actual decimal-string H5 properties.
    Selected-channel artifact checks are descriptive output traces, not load
    voltage measurements. ``trials`` contains one train-level delayed diagnostic.
    """
    result={'stage':'multipoint','status':'failed','errors':[],
            'source':{'path':str(Path(path).resolve())},'metadata':{},'events':[],
            'pulses':[],'roi':list(roi),'roi_sha256':digest(list(roi)),'trials':[],
            'v2_qualified':False,'causal_or_improvement_claim':False,
            'analysis_scope':'capability_only_no_sham_no_primary_test',
            'r2_enabled':False,'training_enabled':False,
            'limits':['No sham/control comparison and no p value.',
                      'Overlapping pulses are not independent trials.',
                      'Activity changes do not establish improvement, learning, or health.']}
    try:
        with H5Recording(path,config) as recording:
            common=_common(recording,config,'multipoint')
            result.update(common)
            errors=result['errors']
            if expected_post not in (60,600):raise ValueError('Recovery duration must be 60 or 600')
            if not isinstance(reference_rate,(int,float)) or not math.isfinite(reference_rate) or reference_rate<=0:
                raise ValueError('Frozen positive calibration reference rate required')
            if len(roi)==0 or any(type(c) is not int for c in roi) or len(set(roi))!=len(roi) or not set(roi)<=set(recording.channels.tolist()):
                raise ValueError('Invalid fixed physical-channel ROI')
            selected=[c for c,e in recording.mapping if e==config.stim_electrode]
            if len(selected)!=1 or selected[0] in roi:raise ValueError('Stimulation channel missing or included in ROI')
            all_stim_channels=getattr(config,'stim_channels',None)
            if all_stim_channels is not None and set(roi)&set(all_stim_channels):
                raise ValueError('Frozen ROI contains a stimulating channel')
            if len(events15)!=15 or len(recording.events)!=15:raise ValueError('Exactly 15 frozen and recorded events required')
            if len({e['id'] for e in events15})!=15:raise ValueError('Duplicate expected event identities')
            point=events15[0].get('pair')
            if type(point) is not int or not 1<=point<=20:raise ValueError('Point index must be 1..20')
            if point==20 and expected_post!=600:raise ValueError('Twentieth point requires 600 seconds recovery')
            for i,(actual,expected) in enumerate(zip(recording.events,events15)):
                if any(expected.get(k)!=v for k,v in [('arm','true'),('pair',point),('phase','multipoint')]):
                    raise ValueError('Frozen train arm/point/phase invalid')
                electrode=expected.get('electrode')
                if type(electrode) not in (int,str) or not re.fullmatch(r'[0-9]+',str(electrode)) or int(electrode)!=config.stim_electrode:
                    raise ValueError('Frozen electrode must match physical binding as decimal string')
                if any(actual.get(k)!=expected.get(k) for k in ('id','arm','pair','phase')):
                    errors.append('event_%d_identity_mismatch'%(i+1))
                if any(actual.get('message',{}).get(k)!=str(expected.get(k)) for k in ('group','pulse','electrode')):
                    errors.append('event_%d_message_mismatch'%(i+1))
                if str(expected.get('group'))!=str(i//3+1) or str(expected.get('pulse'))!=str(i%3+1):
                    raise ValueError('Frozen schedule group/pulse order invalid')
            first=recording.events[0]['time_s'];first_frame=recording.events[0]['frame']
            for i,e in enumerate(recording.events):
                target=(i//3)*4000+(i%3)*200
                if abs((e['frame']-first_frame)-target)>1:errors.append('event_%d_offset_outside_one_sample'%(i+1))
            if first<10.-1e-9:errors.append('less_than_10_seconds_pre_recording')
            end=recording.events[-1]['time_s']+8/recording.fs
            result.update(point=point,first_event_s=first,last_pulse_end_s=end,
                          expected_post_s=expected_post,actual_post_end_seconds=recording.duration-end,
                          reference_rate=float(reference_rate))
            if recording.duration-end<expected_post-1e-9:errors.append('post_end_recording_short')
            if errors:return result
            # Read through 600 ms after the FINAL pulse ends, including the whole train.
            trace=_sample(recording,selected,first,end-first+.6)[0].astype(float)
            center=float(np.median(_sample(recording,selected,first-3)))
            magnitude=np.abs(trace-center)*recording.metadata['lsb_uv']
            rails=np.flatnonzero((trace==0)|(trace==1023))
            last_rail=float(first+rails[-1]/recording.fs-end) if len(rails) else None
            result['selected_last_rail_relative_final_pulse_end_s']=last_rail
            result['selected_peak_abs_uv_entire_train_and_600ms']=float(np.max(magnitude))
            if last_rail is not None and last_rail>.5+1e-10:errors.append('selected_saturation_over_500ms_after_final_pulse_end')
            for event in recording.events:
                start=round((event['time_s']-first)*recording.fs)
                crossing=np.flatnonzero(magnitude[start:start+201]>=1000)
                onset=float(crossing[0]/recording.fs) if len(crossing) else None
                passed=onset is not None and onset<=.01
                result['pulses'].append({'id':event['id'],'time_s':event['time_s'],
                                         'first_1000uv_crossing_s':onset,'artifact_passed':passed})
                if not passed:errors.append('event_%s_output_trace_not_confirmed_within_10ms'%event['id'])
            before=_sample(recording,roi,first-3)
            recovery=[stability(before,_sample(recording,roi,end+offset)) for offset in (2,10)]
            result['recovery']=recovery
            if not all(q['passed'] for q in recovery):errors.append('fixed_roi_MAD_recovery_failed')
            final=stability(_sample(recording,roi,first-10),_sample(recording,roi,recording.duration-3))
            result['final_recovery']=final
            if not final['passed']:errors.append('final_fixed_roi_MAD_recovery_failed')
            rates=[_rate(recording,roi,end+offset,30.) for offset in (0,30)]
            abnormal=[r<reference_rate*.25 or r>reference_rate*4 for r in rates]
            result.update(post_end_30s_rates=rates,activity_outliers=abnormal)
            if all(abnormal):errors.append('consecutive_activity_outliers')
            pre=_rate(recording,roi,first-5,3.);post=_rate(recording,roi,end+2,3.)
            result['trials']=[{'point':point,'unit':'whole_train_descriptive_only',
                               'pre_window_s':[first-5,first-2],'post_window_s':[end+2,end+5],
                               'pre_rate_hz':pre,'post_rate_hz':post,'delta_hz':post-pre,
                               'early_train_and_0_2s_after_end_excluded':True}]
            result['activity_trend_30s']=[{'start_s':float(t),'duration_s':float(min(30,recording.duration-t)),
                                            'rate_hz':_rate(recording,roi,float(t),float(min(30,recording.duration-t)))}
                                           for t in np.arange(0,recording.duration,30.)]
            result['status']='pass' if not errors else 'failed'
    except Exception as error:
        result['errors'].append(type(error).__name__+': '+str(error))
        result['status']='failed'
    return result
