"""Recompute first-trial output, ADC saturation and response from the archived H5."""
import json
import os
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path
from types import SimpleNamespace

OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[1]
os.environ['HDF5_PLUGIN_PATH']=str(ROOT/'.tools/mea_maxwell_hdf5_plugin')
sys.path.insert(0,str(ROOT/'standalone/brain_device_session'))
from report_recorded_pulses import Figure,_plot_style,_save,np
from livepair.figures import draw_sites
from mea_stim_v2.recording import H5Recording
from mea_stim_v2.qc import stability

RUN=OUT/'cpp-crossover-v1'
def read(path):
    return json.loads(path.read_bytes())

def main():
    cfg=SimpleNamespace(threshold_h5_dataset='settings/spike_threshold')
    trial=read(RUN/'trial_01-evidence.json')
    route=read(RUN/'route.json')
    d=read(RUN/'design.json')
    b=read(RUN/'binding.json')
    end=read(RUN/'end.json')
    audit=[json.loads(x) for x in (RUN/'api_audit.jsonl').read_text().splitlines()]
    acks=[r for r in audit if r['kind']=='ack' and r['reply']=='OK']
    output=[r for r in acks if r['command'].startswith('sequence_send ')]
    assert len(output)==1
    disabled_units={int(r['command'].split()[1]) for r in acks if r['utc']>output[0]['utc'] and r['command'].startswith('mea_set_stimulation_unit ') and r['command'].split()[2:]==['0']*6}
    assert len(disabled_units)==32 and not end['cleanup'] and end['closed']
    assert read(OUT/'device-final.json')['servers'][0]['open_h5']==[]
    stream=[json.loads(l) for l in (RUN/'cpp-spikes.jsonl').read_text().splitlines()]
    assert len(stream)==end['cpp_rows'] and end['cpp_error'] is None
    stamp=datetime.fromisoformat(output[0]['utc']).timestamp()
    roi=set(d['roi'])
    cpp=[sum(stamp+a<=ms/1000<stamp+z and ch in roi for ms,ch,am,rx in stream) for a,z in d['windows']]
    config=read(OUT/'baseline-qc.json')
    draw_sites(OUT,b,read(OUT/'baseline/site-candidates.json'))
    channels=[route['channels'][s] for s in 'HWF']
    with H5Recording(OUT/'baseline/quiet.raw.h5',cfg) as rec:
        assert rec.metadata==config['meta'] and not rec.events
    with H5Recording(RUN/'warmup/warmup.raw.h5',cfg) as rec:
        warm=read(RUN/'warmup.json')
        assert rec.metadata==warm['metadata'] and not rec.events
        quiet_fractions=[stability(rec.sample(d['roi'],3),rec.sample(d['roi'],t))['fraction_channels_passed'] for t in (14,28,55)]
        assert np.allclose(quiet_fractions,warm['fractions'])
    with H5Recording(RUN/'trial_01/trial_01.raw.h5',cfg) as rec:
        assert rec.metadata['chip_id']==b['chip_id']=='27701' and rec.metadata['frame_gaps']==0
        assert all(dict(rec.mapping)[ch]==b['electrodes'][s] for ch,s in zip(channels,'HWF'))
        assert [e['frame'] for e in rec.events]==trial['frames'] and len(rec.events)==3
        assert all(e['message']['site']=='H' for e in rec.events)
        t=rec.events[0]['time_s'];fs=rec.fs
        offsets=[e['frame']-rec.events[0]['frame'] for e in rec.events]
        assert offsets==[0,4000,8000]
        raw=rec.sample(channels,t-.004,.7)
        zero=np.median(rec.sample(channels,t-3),axis=1)
        rails=(raw==0)|(raw==1023)
        onsets=[]
        for off in offsets:
            pos=80+off
            onsets.append([k for j,k in enumerate('HWF') if not rails[j,pos-8:pos].any() and rails[j,pos:pos+40].any()])
        assert trial['rails_at_onsets']==onsets==[['H']]*3
        late={s:int(np.count_nonzero(x[-1000:])) for s,x in zip('HWF',rails)}
        where={s:((np.flatnonzero(x[-1000:])+12920)/20).tolist() for s,x in zip('HWF',rails)}
        assert any(late.values()) and trial['late_site_rail']
        late_extent={k:([v[0],v[-1]] if v else []) for k,v in where.items()}
        span={}
        for site,voltage in zip('HWF',rec.sample(channels,t,12)):
            rail=(voltage==0)|(voltage==1023)
            center=int(.65*fs)
            if rail[center]:
                left=np.flatnonzero(~rail[:center])
                right=np.flatnonzero(~rail[center:])
                assert left.size and right.size
                span[site]=[float(1000*(left[-1]+1)/fs),float(1000*(center+right[0])/fs)]
        later=rec.sample(channels,t+12,1)
        later_rails={s:int(np.count_nonzero((x==0)|(x==1023))) for s,x in zip('HWF',later)}
        response=stability(rec.sample(d['roi'],t-3),rec.sample(d['roi'],t+12))
        counts=[int(np.sum(np.isin(rec.spike_channels,d['roi'])&(rec.spike_times>=t+a)&(rec.spike_times<t+z))) for a,z in d['windows']]
        assert counts==[w['spikes'] for w in trial['windows']]
        assert abs(response['fraction_channels_passed']-trial['recovery_fraction'])<1e-12
        passed=sum(response['channel_pass'])
        assert sum(k.endswith('.h5') for k in read(OUT/'transfer.json'))==3
        with _plot_style():
            fig=Figure(figsize=(10,6.2),layout='constrained',dpi=145)
            a,z=fig.subplots(2,1,gridspec_kw={'height_ratios':[1.6,1]})
            x=np.arange(raw.shape[1])/20-4
            for j,s in enumerate('HWF'):
                a.plot(x,(raw[j].astype(float)-zero[j])*rec.metadata['lsb_uv']/1000+8*j,lw=.75,label=f'{s} · {b["electrodes"][s]}')
            for off in offsets:a.axvline(off/20,ls=':',lw=.8,color='#303c43')
            a.axvspan(646,696,color='#e8a249',alpha=.24,label='饱和检查窗')
            a.set(xlabel='相对首个设备事件的时间（毫秒）',ylabel='记录电压（mV，曲线已错开）',yticks=[0,8,16],yticklabels=['H','W','F'])
            a.legend(ncol=4,loc='upper center',fontsize=8)
            a.set_title('27701 · 首次规律刺激的三个波形起点与饱和检查窗')
            labs=['刺激前 −6 至 −1 秒','刺激后 2 至 7 秒','刺激后 8 至 13 秒']
            z.bar(labs,counts,color=['#657880','#187d81','#629c9b'])
            for i,n in enumerate(counts):z.text(i,n+5,str(n),ha='center')
            z.set(ylabel='H5 尖峰数／5 秒',ylim=(0,max(counts)*1.22),title=f'{len(roi)} 个非刺激通道 · 仅一次试次')
            _save(fig,OUT/'首个试次实测回读.png')
    detail=dict(chip=b['chip_id'],event_frames=trial['frames'],onset_hits=onsets,late_window_ms=[646,696],
                late_rail_samples_out_of_1000=late,late_rail_first_last_ms=late_extent,continuous_rail_span_ms=span,rail_samples_12_to_13s=later_rails,
                stability_pass=passed,roi_channels=len(roi),cpp_counts=cpp,h5_counts=counts,all_units_off=len(disabled_units),
                raw_verified=[k for k in read(OUT/'transfer.json') if k.endswith('.h5')])
    (OUT/'raw-confirmation.json').write_text(json.dumps(detail,ensure_ascii=False,indent=2),encoding='utf8')
    print(json.dumps(detail,ensure_ascii=False))

if __name__=='__main__':main()
