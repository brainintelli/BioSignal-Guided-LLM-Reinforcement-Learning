"""Import VoltageStimulation metadata as parameters, never as qualified training data."""
import argparse
from datetime import datetime, timedelta, timezone
import hashlib
import json
import math
from pathlib import Path
import shutil

from .profile import PulseProfile


def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def scalar(group,name):
    dataset=group[name]
    if dataset.shape != (1,):raise ValueError('One scalar required: '+name)
    value=dataset[0]
    if isinstance(value,bytes):return value.decode('utf-8')
    return value.item() if hasattr(value,'item') else value


def number(value,name):
    if type(value) not in (str,int,float):raise ValueError('Numeric field required: '+name)
    result=float(value)
    if not math.isfinite(result):raise ValueError('Finite field required: '+name)
    return result


def integer(value,name):
    result=number(value,name)
    if result != int(result):raise ValueError('Integer required: '+name)
    return int(result)


def write(path,value):
    with path.open('x',encoding='utf-8') as stream:
        json.dump(value,stream,ensure_ascii=False,indent=2,allow_nan=False)
        stream.write('\n')


def inspect_recording(path):
    import h5py
    import numpy as np
    path=Path(path).resolve();digest=sha(path)
    with h5py.File(path,'r') as f:
        if scalar(f,'assay/script_id') != 'VoltageStimulation_v1.0':
            raise ValueError('Only inspected VoltageStimulation_v1.0 schema is supported')
        if set(f['data_store']) != {'data0000'}:raise ValueError('Exactly one recording required')
        d=f['data_store/data0000'];inputs=f['assay/inputs']
        chip=str(scalar(f,'wellplate/id'));run=str(scalar(f,'assay/run_id'))
        if not chip.isdigit() or len(chip)>16 or not run.isdigit() or len(run)>16:
            raise ValueError('Expected numeric chip and assay IDs')
        sampling=number(scalar(d,'settings/sampling'),'sampling')
        if sampling != 20000:raise ValueError('20kHz recording required for this profile')
        values=str(scalar(inputs,'amplitude_range')).split(',')
        if len(values)!=3:raise ValueError('Expected start,end,step amplitude range')
        amplitudes=[number(value,'amplitude range') for value in values]
        if amplitudes[0]!=amplitudes[1] or amplitudes[2]<=0:
            raise ValueError('Amplitude sweep cannot be imported as one fixed profile')
        phase=integer(scalar(inputs,'phase'),'phase')
        pulses=integer(scalar(inputs,'pulses'),'pulses')
        trains=integer(scalar(inputs,'pulse_trains'),'pulse trains')
        if pulses!=3 or not 1<=trains<=50000:raise ValueError('Bounded3pulse groups required')
        ipi_ms=number(scalar(inputs,'ipi'),'IPI')
        ipti_s=number(scalar(inputs,'ipti'),'IPTI')
        ipi_us=integer(ipi_ms*1000,'IPI microseconds')
        pulse_frames=integer(ipi_us/50,'pulse frames')
        group_frames=integer(ipti_s*sampling,'group frames')
        if group_frames<=(pulses-1)*pulse_frames+2*phase/50:
            raise ValueError('Groups overlap or interval is invalid')
        profile=PulseProfile('mea.pulse_profile.v1',f'h5-{chip}-{run}-{digest[:12]}',digest,
                             amplitudes[0],phase,ipi_us,3,'positive_first')
        electrodes=json.loads(str(scalar(inputs,'electrodes')))
        well=integer(scalar(d,'well_id'),'well ID')
        sites=electrodes['electrodes'][str(well)]['stimulation']
        if (type(sites) is not list or not 1<=len(sites)<=1020
                or any(type(x) is not int or x<0 for x in sites) or len(set(sites))!=len(sites)):
            raise ValueError('Unique configured stimulation electrodes required')
        event_ds=d['events']
        if event_ds.shape != (trains*pulses,):raise ValueError('Event count differs from declared pulse groups')
        if set(event_ds.dtype.names or ()) != {'frameno','eventtype','eventid','eventmessage'}:
            raise ValueError('Unsupported event schema')
        events=event_ds[()]
        frame_matrix=events['frameno'].reshape(trains,pulses)
        if (np.any(frame_matrix<0) or not np.all(events['eventtype']==1)
                or not np.all(events['eventid'].reshape(trains,pulses)==np.arange(1,trains+1)[:,None])):
            raise ValueError('Event IDs/types do not attest to the declared train structure')
        within=np.unique(np.diff(frame_matrix,axis=1)).tolist()
        between=np.unique(np.diff(frame_matrix[:,0])).tolist()
        if within!=[pulse_frames] or (trains>1 and between!=[group_frames]):
            raise ValueError('Observed event intervals differ from declared profile')
        for row in events:
            raw=row['eventmessage'];message=json.loads(raw.decode('utf-8') if isinstance(raw,bytes) else str(raw))
            if (type(message) is not dict or set(message)!={'amplitude','phase'}
                    or number(message['amplitude'],'event amplitude')!=amplitudes[0]
                    or number(message['phase'],'event phase')!=phase):
                raise ValueError('Event message differs from configured amplitude/phase')
        mapping=d['settings/mapping'][()]
        if not {'channel','electrode'}.issubset(mapping.dtype.names or ()):
            raise ValueError('Expected channel/electrode mapping')
        matched={str(site):[int(row['channel']) for row in mapping if int(row['electrode'])==site] for site in sites}
        spike_ds=d['spikes']
        if len(spike_ds.shape)!=1 or not {'frameno','channel','amplitude'}.issubset(spike_ds.dtype.names or ()):
            raise ValueError('Expected spike table')
        if len(spike_ds)>10000000:raise ValueError('Recording exceeds bounded parameter-intake size')
        spike_times=spike_ds.fields('frameno')[()]
        spike_min=int(spike_times.min()) if len(spike_times) else None
        spike_max=int(spike_times.max()) if len(spike_times) else None
        raw_present='groups/routed/raw' in d or 'groups/routed/sig' in d
        start=integer(scalar(d,'start_time'),'start time');stop=integer(scalar(d,'stop_time'),'stop time')
        if start<0 or stop<=start:raise ValueError('Invalid recording wall times')
        zone=timezone(timedelta(hours=8))
        h=dict(chip_id=chip,chip_type=str(scalar(f,'wellplate/version')),assay_run_id=run,
            assay_script_id='VoltageStimulation_v1.0',sampling_hz=sampling,
            gain=number(scalar(d,'settings/gain'),'gain'),hpf_hz=number(scalar(d,'settings/hpf'),'HPF'),
            recording_adc_lsb_volts=number(scalar(d,'settings/lsb'),'recording ADC LSB'),
            stimulation_dac_lsb_mv=None,load_voltage_measured=False,
            amplitude_range_as_stored=values,declared_target_mv=amplitudes[0],declared_phase_us=phase,
            pulses_per_train=pulses,train_count=trains,pulse_onset_interval_ms=ipi_ms,
            group_onset_interval_seconds=ipti_s,randomize_as_stored=str(scalar(inputs,'randomize')),
            configured_stimulation_electrodes=sites,matching_recording_channels=matched,
            actual_stimulus_routing_verified=False,mapped_channel_count=len(mapping),
            observed_event_count=len(events),observed_pulse_interval_frames=within,
            observed_group_start_interval_frames=between,first_event_frame=int(frame_matrix[0,0]),
            last_event_frame=int(frame_matrix[-1,-1]),spike_count=len(spike_times),
            spike_frame_min=spike_min,spike_frame_max=spike_max,
            observed_spike_seconds_after_last_marker=None if spike_max is None else (spike_max-int(frame_matrix[-1,-1]))/sampling,
            continuous_raw_present=raw_present,continuous_coverage_verified=False,
            recording_start_unix_ms=start,recording_stop_unix_ms=stop,
            recording_start_china=datetime.fromtimestamp(start/1000,zone).isoformat(),
            recording_stop_china=datetime.fromtimestamp(stop/1000,zone).isoformat(),
            wall_duration_seconds=(stop-start)/1000,
            pre_stim_period_as_stored=str(scalar(inputs,'pre_stim_period')),
            post_stim_period_as_stored=str(scalar(inputs,'post_stim_period')))
        groups=[dict(historical_train_id=i+1,pulse_frames=[int(v) for v in row]) for i,row in enumerate(frame_matrix)]
    reasons=['no_game_transition_or_r1_link','organoid_identity_unknown','route_and_dac_output_unverified',
             'recovery_qc_unverified','intentional_quiet_unverified','no_independent_baseline_admitted']
    if not raw_present:reasons.insert(0,'no_continuous_raw_waveform')
    return dict(schema='mea.stimulus_profile_import.v1',source=dict(path=str(path),bytes=path.stat().st_size,sha256=digest),
        execution_profile=profile.to_dict(),stimulus_profile_sha256=profile.digest,
        historical_recording=h,event_groups=groups,
        software_mapping=dict(scope='one3pulse feedback group when game encoder requests stimulation',
            healthy_feedback='intentional_quiet',game_threshold_enter_below=85,game_threshold_exit_at=90,
            candidate_minimum_cycle_seconds=20,candidate_response_window_seconds=[12,13],
            candidate_timing_validated_by_this_recording=False,
            polarity_source='inherited_software_candidate_not_verified_in_h5',
            historical_train_count_used_per_game_action=False,configured_electrodes_are_not_live_routes=True),
        reward_contract=dict(r1_source='native_game_required_not_in_h5',r2_quiet=1,r2_stimulation=-1,beta=.002,
            gamma_tick=.99985,total_formula='R1 + beta * r2_raw * sum(gamma_tick**k for k in range(game_duration))',
            simulation_r2_enabled=True),
        admission=dict(parameter_import_passed=True,historical_training_admitted=False,
            real_stimulation_enabled=False,real_r2_enabled=False,reasons=reasons))


def import_recording(source,output,expected_sha256=None):
    source=Path(source).resolve();out=Path(output).resolve()
    out.mkdir(parents=True,exist_ok=False)
    fixed=out/'source.h5'
    with source.open('rb') as src,fixed.open('xb') as dst:shutil.copyfileobj(src,dst)
    digest=sha(fixed)
    if expected_sha256 is not None and digest!=expected_sha256:
        raise ValueError('Fixed input copy differs from expected source hash')
    document=inspect_recording(fixed)
    if document['source']['sha256']!=digest:raise ValueError('Fixed source changed during inspection')
    document['source']['original_path']=str(source)
    groups=document.pop('event_groups')
    write(out/'stimulus_profile.json',document)
    with (out/'reward_event_candidates.jsonl').open('x',encoding='utf-8') as stream:
        for group in groups:
            row=group|dict(source='recording_marker_only',source_sha256=digest,
                stimulus_profile_sha256=document['stimulus_profile_sha256'],r1=None,r2_candidate=-1,
                r2_confirmed=False,total_reward=None,game_transition_id=None,training_admitted=False,
                reasons=document['admission']['reasons'])
            stream.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n')
    result=dict(parameter_import_passed=True,source_sha256=digest,
        historical_training_admitted=False,real_stimulation_enabled=False,model_training_performed=False,
        observed_event_count=document['historical_recording']['observed_event_count'],
        historical_group_candidates=len(groups),generated_quiet_rewards=0,
        stimulus_profile_sha256=document['stimulus_profile_sha256'])
    write(out/'result.json',result)
    write(out/'manifest.json',{p.name:sha(p) for p in sorted(out.iterdir()) if p.is_file()})
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--expected-sha256')
    args=parser.parse_args()
    print(json.dumps(import_recording(args.input,args.output,args.expected_sha256),ensure_ascii=False))


if __name__=='__main__':main()
