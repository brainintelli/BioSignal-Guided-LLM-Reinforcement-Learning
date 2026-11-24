"""Analyze a CLOSED real first-probe H5. Descriptive, single chip, no sham control."""
import os
os.environ['HDF5_PLUGIN_PATH'] = '/pub/soft/MaxLab/so'
from pathlib import Path
import datetime as dt
import hashlib
import json
import re
import sys
import h5py
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE = Path(__file__).resolve().parent
OUT = BASE / 'analysis'


def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    reports = list((BASE / 'results').glob('*/run.json'))
    if len(reports) != 1:
        raise RuntimeError('Exactly one first-probe run required')
    run = json.loads(reports[0].read_text())
    if run['status'] in ('starting','recording'):
        raise RuntimeError('File is not yet finalized; do not read an active H5')
    source = Path(run['h5_directory_on_server']) / 'stimulation.raw.h5'
    OUT.mkdir(exist_ok=True)
    issues = []
    with h5py.File(source,'r') as f:
        d = f['data_store/data0000']
        mapping = d['settings/mapping'][:]
        spikes = d['spikes'][:]
        events = d['events'][:]
        frames = d['groups/routed/frame_nos'][:]
        raw = d['groups/routed/raw']
        fs = float(d['settings/sampling'][0])
        uv = float(d['settings/lsb'][0])*1e6
        f0 = int(frames[0])
        duration = len(frames)/fs
        valid = np.isin(spikes['channel'],mapping['channel']) & (spikes['frameno']>=frames[0]) & (spikes['frameno']<=frames[-1])
        spikes_valid = spikes[valid]
        times = (spikes_valid['frameno'].astype(float)-f0)/fs
        event_seconds = (events['frameno'].astype(float)-f0)/fs
        chip = f['wellplate/id'][0].decode()
        gap_count = int(np.count_nonzero(np.diff(frames)!=1))
        expected_pairs = sorted((int(a),int(b)) for a,b in re.findall(r'(\d+)\((\d+)\)',(BASE/'qualified_route.cfg').read_text().split()[0]))
        pairs = sorted((int(x['channel']),int(x['electrode'])) for x in mapping)
        if chip != '22554': issues.append('chip_id_mismatch')
        if gap_count: issues.append('frame_gaps')
        if pairs != expected_pairs: issues.append('actual_H5_mapping_differs')
        if len(events)!=5: issues.append('event_count_not_five')
        if not np.array_equal(events['eventid'],np.arange(2,7)): issues.append('unexpected_event_ids')
        if len(events)>1 and not np.all(np.diff(events['frameno'])==200000): issues.append('event_spacing_not_10seconds')
        if run['status']!='completed' or run['cleanup_errors']: issues.append('runner_or_cleanup_failure')
        if not len(events):
            (OUT/'analysis.json').write_text(json.dumps({'issues':issues,'event_count':0},indent=2))
            return 2
        electrode = int(run['parameters']['stimulation_electrodes'][0])
        ch = int(mapping[mapping['electrode']==electrode]['channel'][0])
        selected_times = times[spikes_valid['channel']==ch]
        first,last = float(event_seconds[0]),float(event_seconds[-1])
        if first < 599 or duration-last < 599: issues.append('pre_or_post_duration_short')
        windows = {'baseline_early': [first-600,first-540],
                   'baseline_matched_540s':[first-540,first],
                   'early_recovery_1_to_60s':[last+1,last+60],
                   'late_recovery_60_to_600s':[last+60,min(last+600,duration)]}
        summaries = {}
        for name,(a,b) in windows.items():
            mask=(times>=a)&(times<b)
            summaries[name]={'start_s':a,'end_s':b,'duration_s':b-a,
                'spikes':int(mask.sum()),'population_rate_hz':float(mask.sum()/(b-a)),
                'active_channels':int(len(np.unique(spikes_valid['channel'][mask]))),
                'selected_electrode_rate_hz':float(np.count_nonzero((selected_times>=a)&(selected_times<b))/(b-a))}
        trials=[]
        psth_edges=np.arange(-1,2.0001,.02)
        trial_hist=[]
        raw_time=np.arange(-int(.05*fs),int(.5*fs))/fs
        raw_trials=[]
        sampled_channels=mapping['channel'].astype(int)
        for i,t in enumerate(event_seconds):
            rel=times-t
            pre=int(np.count_nonzero((rel>=-.5)&(rel<-.02)))
            post=int(np.count_nonzero((rel>=.02)&(rel<.5)))
            late=int(np.count_nonzero((rel>=.5)&(rel<2)))
            trial_hist.append(np.histogram(rel,bins=psth_edges)[0])
            index=int(events['frameno'][i])-f0
            # Read all channels together, then inspect mapped channels and the selected trace.
            chunk=raw[:,index-int(.05*fs):index+int(.5*fs)]
            selected=chunk[ch].astype(float)
            baseline=selected[raw_time<-.005]
            centered=(selected-np.median(baseline))*uv
            raw_trials.append(centered)
            near=(raw_time>=0)&(raw_time<.02)
            later=(raw_time>=.02)&(raw_time<.05)
            rails=(chunk[sampled_channels]==0)|(chunk[sampled_channels]==1023)
            saturated_at=np.any(rails,axis=0)&(raw_time>=0)
            trials.append({'pulse':i+1,'event_second':float(t),'event_frame':int(events['frameno'][i]),
                'pre_480ms_population_spikes':pre,'post_20_to_500ms_population_spikes':post,
                'post_500_to_2000ms_population_spikes':late,
                'post_minus_pre_spikes':post-pre,
                'selected_pre_raw_std_uv':float(np.std(baseline)*uv),
                'selected_post_20_to_50ms_raw_std_uv':float(np.std(selected[later])*uv),
                'selected_artifact_peak_abs_uv_0_to_20ms':float(np.max(np.abs(centered[near]))),
                'mapped_saturated_channels_first20ms':int(np.count_nonzero(np.any(rails[:,near],axis=1))),
                'last_any_mapped_rail_ms_within500ms':float(raw_time[saturated_at][-1]*1000) if np.any(saturated_at) else None})
        # Distributed raw windows characterize recovery independently of detector events.
        raw_qc=[]
        for label,t in [('baseline',max(1,first-30)),('recovery_60s',last+60),('recovery_300s',last+300),('recovery_end',duration-2)]:
            sample=raw[:,int(t*fs):int((t+1)*fs)]
            selected=sample[ch].astype(float)
            raw_qc.append({'label':label,'start_s':t,'selected_std_uv':float(selected.std()*uv),
                'selected_mean_adc':float(selected.mean()),
                'mapped_rail_fraction':float(np.mean((sample[sampled_channels]==0)|(sample[sampled_channels]==1023)))})
        base_rate=summaries['baseline_matched_540s']['population_rate_hz']
        late_rate=summaries['late_recovery_60_to_600s']['population_rate_hz']
        result={'schema':'mea.first_probe.observation.v1','source_h5':str(source),
            'analyzed_utc':dt.datetime.now(dt.timezone.utc).isoformat(),
            'technical_qc_passed':not issues,'issues':issues,'chip_id':chip,
            'organoid_biological_id':None,'organoid_session_label':'22554-20261117T032900Z',
            'sample_rate_hz':fs,'duration_seconds':duration,'frames':len(frames),'frame_gap_count':gap_count,
            'mapping_channels':len(mapping),'stim_electrode':electrode,'stim_channel':ch,
            'gain':float(d['settings/gain'][0]),'hpf_hz':float(d['settings/hpf'][0]),'lsb_uv':uv,
            'all_spikes':len(spikes),'invalid_spike_records_excluded':int(np.count_nonzero(~valid)),
            'valid_mapped_spikes':len(spikes_valid),'event_count':len(events),
            'event_seconds':event_seconds.tolist(),'event_intervals_seconds':np.diff(event_seconds).tolist(),
            'target_phase_mv':300,'nominal_quantized_phase_mv':run['actual_amplitude_schedule_mv'][0],
            'physical_output_voltage_measured':False,'runner_status':run['status'],
            'cleanup_errors':run['cleanup_errors'],'windows':summaries,'trials':trials,'raw_qc':raw_qc,
            'late_recovery_population_rate_change_percent':100*(late_rate/base_rate-1) if base_rate else None,
            'r2_enabled':False,'realtime_training':False,
            'limits':['One chip/session; five pulses, no randomized sham or independent biological replication.',
                      'Event markers plus raw artifacts verify recorded timing, not electrode voltage/charge measurement.',
                      '20ms post-stimulus exclusion is provisional; raw artifact recovery must be checked before biological interpretation.',
                      'Higher firing rate alone is not an improvement, learning, exploration, fear or thirst.']}
        np.savez_compressed(OUT/'comparison_data.npz',spike_seconds=times,spike_channels=spikes_valid['channel'],
            spike_amplitude_uv=spikes_valid['amplitude']*uv,mapping=mapping,
            event_seconds=event_seconds,psth_edges=psth_edges,trial_hist=np.asarray(trial_hist),
            raw_seconds=raw_time,selected_raw_trials_uv=np.asarray(raw_trials))
    result['source_h5_sha256']=sha256(source)
    (OUT/'analysis.json').write_text(json.dumps(result,indent=2))
    plt.rcParams.update({'figure.dpi':150,'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(2,2,figsize=(13,8),constrained_layout=True)
    edges=np.arange(0,duration+10,10)
    rate=np.histogram(times,bins=edges)[0]/np.diff(edges)
    axes[0,0].plot((edges[:-1]+edges[1:])/120,rate,color='#2563eb',lw=1.4)
    axes[0,0].axvspan(first/60,last/60,color='#dc2626',alpha=.18,label='5-pulse period')
    axes[0,0].set(xlabel='Minutes from recording start',ylabel='Population spikes / second',title='A. Full recording (10 s bins)')
    axes[0,0].legend(frameon=False)
    centers=(psth_edges[:-1]+psth_edges[1:])/2
    hist=np.asarray(trial_hist)
    displayed_rate=hist.mean(axis=0)/.02
    displayed_rate[(centers>=0)&(centers<.02)]=np.nan
    axes[0,1].step(centers,displayed_rate,where='mid',color='#7c3aed')
    axes[0,1].axvspan(0,.02,color='gray',alpha=.3,label='0–20 ms excluded')
    axes[0,1].set(xlabel='Seconds from pulse event',ylabel='Mean population spikes / second',title='B. Event-aligned activity (5 pulses)')
    axes[0,1].legend(frameon=False)
    for tr in trials:
        axes[1,0].plot([0,1],[tr['pre_480ms_population_spikes'],tr['post_20_to_500ms_population_spikes']],marker='o',alpha=.75,label=str(tr['pulse']))
    axes[1,0].set(xticks=[0,1],xticklabels=['Pre: -500 to -20 ms','Post: +20 to +500 ms'],ylabel='Population spikes / 480 ms',title='C. Equal windows, each pulse')
    axes[1,0].legend(title='Pulse',ncol=5,frameon=False)
    for i,y in enumerate(raw_trials):
        axes[1,1].plot(raw_time*1000,y,alpha=.65,lw=.65,label=str(i+1))
    axes[1,1].axvspan(0,20,color='gray',alpha=.2)
    axes[1,1].set(xlim=(-20,100),xlabel='Milliseconds from pulse event',ylabel='Recorded voltage (uV, centered)',title=f'D. Electrode {electrode}: raw traces')
    fig.suptitle('MaxOne 22554 | first physical probe | descriptive, no sham control',fontsize=14)
    fig.savefig(OUT/'first_probe_comparison.png')
    fig.savefig(OUT/'first_probe_comparison.svg')
    plt.close(fig)
    print(json.dumps({k:result[k] for k in ['technical_qc_passed','issues','duration_seconds','event_count','event_intervals_seconds','late_recovery_population_rate_change_percent','cleanup_errors']},indent=2))
    print('TRIALS',json.dumps(trials))
    print('RAW_QC',json.dumps(raw_qc))
    return int(bool(issues))


if __name__=='__main__':
    raise SystemExit(main())
