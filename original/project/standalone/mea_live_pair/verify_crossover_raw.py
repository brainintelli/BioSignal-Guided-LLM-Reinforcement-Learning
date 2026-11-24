"""Recheck the delivered raw H5 files and render an example of each measured output."""
import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from livepair.figures import _plot_style
from mea_stim_v2.recording import H5Recording

def read(path):
    return json.loads(path.read_text(encoding='utf8'))

def verify(folder):
    checksum=read(folder/'raw-hashes.json')
    plan=read(folder/'design.json')
    binding=read(folder/'binding.json')
    actual=read(folder/'route.json')
    rois=plan['roi'];sites=['H','W','F'];chs=[actual['channels'][s] for s in sites]
    config=SimpleNamespace(threshold_h5_dataset=binding['threshold_h5_dataset'])
    signals={}
    stats=[]
    if len(checksum)!=19:
        raise ValueError('Expected warmup and 18 trials')
    last=None
    for idx in range(19):
        stem='warmup' if not idx else f'trial_{idx:02d}'
        path=folder/stem/(stem+'.raw.h5')
        with path.open('rb') as f:
            if hashlib.file_digest(f,'sha256').hexdigest()!=checksum[path.relative_to(folder).as_posix()]['sha256']:
                raise ValueError('Local raw recording checksum mismatch')
        with H5Recording(path,config) as rec:
            ch_map=dict(rec.mapping)
            if rec.metadata['chip_id']!=binding['chip_id'] or rec.metadata['frame_gaps'] or any(ch_map[ch]!=binding['electrodes'][s] for ch,s in zip(chs,sites)) or (last and rec.first_frame<=last):
                raise ValueError('Wrong chip, channel routing or device-clock coverage')
            last=rec.last_frame
            if not idx:
                if rec.events:raise ValueError('Unexpected event in quiet recording')
                continue
            trial=plan['trials'][idx-1]
            receipt=read(folder/(stem+'-evidence.json'))
            expected=trial['events'] or [('quiet',0)]
            if len(rec.events)!=len(expected) or not receipt['pattern_verified'] or not receipt['intended_edges_present']:
                raise ValueError('Wrong H5 event count or waveform receipt')
            anchor=rec.events[0]['frame'];t=rec.events[0]['time_s']
            for j,((s,offset),ev) in enumerate(zip(expected,rec.events)):
                if (ev['id']!=4*idx+j or ev['frame']!=anchor+offset or ev['message']['site']!=s or
                    int(ev['message']['trial'])!=idx or int(ev['message']['idx'])!=j):
                    raise ValueError('Device-marker mismatch')
            raw=rec.sample(chs,t-.004,.7)
            rails=(raw==0)|(raw==1023)
            baseline=np.median(rec.sample(chs,t-3),axis=1)
            delta=(raw.astype(float)-baseline[:,None])*rec.metadata['lsb_uv']
            if trial['events']:
                for site,offset in expected:
                    pos=80+offset
                    hits=[s for j,s in enumerate(sites) if not rails[j,pos-8:pos].any() and rails[j,pos:pos+40].any()]
                    if hits!=[site]:raise ValueError('Raw onsets differ from scheduled electrodes')
            elif np.abs(delta).max()>=1000:raise ValueError('Unexpected sham artifact')
            if rails[:,-1000:].any():raise ValueError('Late rail saturation')
            counts=[int((np.isin(rec.spike_channels,rois)&(rec.spike_times>=t+lo)&(rec.spike_times<t+hi)).sum())
                    for lo,hi in plan['windows']]
            if counts!=[w['spikes'] for w in receipt['windows']]:raise ValueError('H5 spike counts differ')
            stats.append(dict(index=idx,condition=trial['condition'],spikes=counts,chip=rec.metadata['chip_id'],
                      outside=rec.metadata['outside_frame_coverage'],pulses=len(trial['events']),duration=rec.duration))
            if trial['condition'] not in signals:
                signals[trial['condition']]=(idx,delta[:,:9200]/1000,trial['events'])
    with _plot_style():
        fig=Figure(figsize=(9.4,6),dpi=145,layout='constrained')
        axs=fig.subplots(3,1,sharex=True,sharey=True)
        for ax,(kind,title) in zip(axs,[('regular','固定电极规律刺激'),('irregular','多点随机刺激'),('sham','零输出')]):
            idx,curves,events=signals[kind]
            x=np.arange(curves.shape[1])/20-4
            for j,(s,color) in enumerate(zip(sites,['#126e90','#b65f33','#758243'])):
                ax.plot(x,curves[j]+7*j,lw=.6,color=color,label=str(binding['electrodes'][s]))
            for _,offset in events:ax.axvline(offset/20,color='#818181',linestyle=':',linewidth=.8)
            ax.set_title(f'{title} · 第 {idx} 次记录',fontsize=10)
            ax.set_yticks([0,7,14],[str(binding['electrodes'][s]) for s in sites]);ax.set_ylabel('电极编号')
        axs[-1].set_xlabel('相对第一个设备标记的时间（毫秒）')
        fig.suptitle('芯片 '+binding['chip_id']+' · 原始记录电压轨迹（按电极错开显示）')
        FigureCanvasAgg(fig).print_png(str(folder/'measured-waveforms.png'))
    (folder/'raw-readback-check.json').write_text(json.dumps(stats,indent=2))
    return stats

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('folder',type=Path)
    r=verify(p.parse_args().folder)
    print('Verified measurements',len(r),'pulses',sum(x['pulses'] for x in r))
