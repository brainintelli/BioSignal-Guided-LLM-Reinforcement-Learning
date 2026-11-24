"""Read the closed H5 and C++ records of the 28080 single-pattern technical test."""
import json
from datetime import datetime, timezone, timedelta
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
from livepair.figures import _plot_style
from mea_stim_v2.recording import H5Recording
from mea_stim_v2.qc import stability


root = Path(__file__).resolve().parent
read = lambda f: json.loads((root / f).read_text(encoding='utf-8'))
plan = read('technical-plan.json')
roi = plan['roi']
cpp = np.asarray([json.loads(l)[:2] for l in (root / 'cpp-spikes.jsonl').read_text().splitlines()],dtype=np.int64)
audit = [json.loads(l) for l in (root/'api_audit.jsonl').read_text().splitlines()]
sends = [r for r in audit if r['kind']=='ack' and r['command'].startswith('sequence_send ')]
assert len(sends)==2 and all(s['reply']=='OK' for s in sends)
results, plots = {}, {}
edges = np.arange(-8,16,dtype=float)
for name, sending in zip(('sham','regular'),sends):
    path = root/name/(name+'.raw.h5')
    with H5Recording(path,SimpleNamespace(threshold_h5_dataset='settings/spike_threshold')) as rec:
        assert rec.metadata['chip_id']=='28080' and dict(rec.mapping)[plan['channel']]==plan['electrode']
        e = rec.events
        relative = [x['frame']-e[0]['frame'] for x in e]
        assert relative == ([0,4000,8000] if name=='regular' else [0])
        assert all(r['message']['phase']==name and r['message']['site']=='H' and
                   int(r['message']['idx'])==i and r['id']==(104 if name=='regular' else 100)+i for i,r in enumerate(e))
        t0=e[0]['time_s']
        assert t0 >= 8 and rec.duration-t0 > 15
        cpp_zero=datetime.fromisoformat(sending['utc']).timestamp()
        cpp_t=cpp[:,0]/1000.-cpp_zero
        h5_t=rec.spike_times-t0
        cpp_roi=np.isin(cpp[:,1],roi)
        h5_roi=np.isin(rec.spike_channels,roi)
        binned={}
        for label,(t,ch,ok) in dict(cpp=(cpp_t,cpp[:,1],cpp_roi),h5=(h5_t,rec.spike_channels,h5_roi)).items():
            rows=[]
            for lo,hi in (plan['before_seconds'],plan['after_seconds'],plan['late_seconds']):
                ix=ok&(t>=lo)&(t<hi)
                rows.append(dict(seconds=[lo,hi],spikes=int(ix.sum()),active_channels=int(len(np.unique(ch[ix])))))
            binned[label]=rows
        before=rec.sample(roi,t0-3)
        late=rec.sample(roi,t0+12)
        target_pre=rec.sample([plan['channel']],t0-3)[0]
        median=float(np.median(target_pre))
        wave=rec.sample([plan['channel']],t0-.004,.43)[0]
        peaks=[]
        for off in relative:
            i=80+off
            raw=wave[i:i+40]
            # Separate new local edges from the long recovery tail on the driven channel.
            peaks.append(dict(frame=e[0]['frame']+off,first_rail_frames=np.flatnonzero((raw==0)|(raw==1023)).tolist()[:1],
                              prior_8_frames_rail=bool(((wave[i-8:i]==0)|(wave[i-8:i]==1023)).any()),
                              min_code=int(raw.min()),max_code=int(raw.max())))
        stats=stability(before,late)
        with h5py.File(path) as f:
            sp=f['data_store/data0000/spikes'][:]
            bad=sp[(sp['frameno']<rec.first_frame)|(sp['frameno']>rec.last_frame)]
            outside=[dict(channel=int(s['channel']),relative_to_first_ms=(int(s['frameno'])-rec.first_frame)/20,
                          relative_to_anchor_s=(int(s['frameno'])-e[0]['frame'])/20000) for s in bad]
        results[name]=dict(host_ack=datetime.fromtimestamp(cpp_zero,timezone(timedelta(hours=8))).isoformat(),
            roi_channels=len(roi),raw_frames=rec.frame_count,frame_gaps=rec.metadata['frame_gaps'],
            invalid_spikes_removed=rec.metadata['invalid_spikes_excluded'],out_of_frame=outside,
            markers=[x['frame'] for x in e],pulse_onsets=peaks,windows=binned,
            recovery_fraction=stats['fraction_channels_passed'],network_rail_free_late=float((~np.any((late==0)|(late==1023),axis=1)).mean()))
        plots[name]=dict(bins=np.histogram(cpp_t[cpp_roi],edges)[0],wave=(wave.astype(float)-median)*rec.metadata['lsb_uv'])

res=dict(chip='28080',site=plan['electrode'],channel=plan['channel'],unit=plan['unit'],
    command_phase_peak_mv=abs(512-445)*read('regular-send.json')['lsb_mv'],phase_us=plan['phase_us'],
    wall_alignment='C++ host milliseconds relative to sequence_send OK; H5 relative to first on-device event',
    ppo_updates=0,formal_training_admission=False,tests=results)
(root/'response.json').write_text(json.dumps(res,indent=2),encoding='utf-8')
with _plot_style():
    fig=Figure(figsize=(9,5.8),dpi=150,layout='constrained')
    a,b=fig.subplots(2,1,gridspec_kw={'height_ratios':[1,1.15]})
    regular=plots['regular']['wave']
    a.plot(np.arange(len(regular))/20-4,regular/1000,color='#215f8f',lw=.8)
    for t in (0,200,400):a.axvline(t,color='#a84b36',ls=':',lw=.8)
    a.set(xlabel='相对首个 H5 事件的时间（毫秒）',ylabel='记录通道电压（mV）',
          title='电极 13032 的原始读回波形 · 三个脉冲起始处均达到记录量程边界')
    for name,title,color in [('sham','零输出对照','#777777'),('regular','规律刺激','#176993')]:
        b.stairs(plots[name]['bins'],edges,color=color,lw=1.6,label=title)
    b.axvspan(0,1,color='#f1c69c',alpha=.4,label='含刺激伪迹，未计入前后比较')
    b.set_yscale('symlog',linthresh=10)
    b.set(xlabel='相对 Python 发送回执的时间（秒）',ylabel='每秒尖峰数（分段对数刻度）',
          title='C++ 返回流 · 固定的 99 个独立通道')
    b.legend(fontsize=8,loc='upper right')
    fig.suptitle('芯片 28080 · 一次零输出对照与一次固定电极刺激')
    FigureCanvasAgg(fig).print_png(str(root/'response.png'))
print(json.dumps(res))
