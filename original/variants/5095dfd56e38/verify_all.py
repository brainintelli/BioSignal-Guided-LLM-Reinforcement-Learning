"""Recheck closed H5 checksums and same-amplifier input-switch data on the analysis host."""
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from signal_check import OUT,H5Recording,np,Figure,_plot_style,_save,load
from mea_stim_v2.config import route_mapping

def main():
    b=load(OUT.parent/'baseline-binding.json');man=load(OUT/'verified.json')
    m=route_mapping(OUT.parent.parent/'sample-route.cfg');g=load(OUT/'switch-plan.json')['groups']
    qcs={};switch={}
    for k,entry in man.items():
        f=OUT.parent.parent/k
        with f.open('rb') as fh:assert hashlib.file_digest(fh,'sha256').hexdigest()==entry['sha256'] and f.stat().st_size==entry['bytes']
        name=f.parent.name;qc=load(f.parent/'qc.json');qcs[name]=qc['stable_counts']
        with H5Recording(f,SimpleNamespace(**b)) as r:
            assert not r.events and r.metadata['chip_id']==b['chip_id'] and r.metadata['frame_gaps']==0 and r.duration>=65
            label=name.removeprefix('independent-')
            if label in ['A-connected','B-connected']:
                keep=label[0];detach='B' if keep=='A' else 'A'
                assert sorted(r.mapping)==sorted((c,e) for c,e in m if c not in g[detach])
                if 'switch' not in qc:continue
                blocks={}
                for group,cs in g.items():
                    rows=sorted(cs);fractions=[];counts=[]
                    for t,v in zip((3,23,43,57),qc['switch'][group],strict=True):
                        start=int(r.fs*t);a=r._raw[rows,start:start+int(r.fs)];clip=(a==0)|(a==1023)
                        frac=float(clip.mean());n=int(clip.any(axis=1).sum())
                        assert abs(frac-v['fraction'])<1e-12 and n==v['channels'] and int(clip.sum(axis=0).max())==v['simultaneous']
                        fractions.append(frac);counts.append(n)
                    picks=rows[::4];traces=[]
                    for ts in range(0,60,5):
                        n=int(ts*r.fs);a=r._raw[picks,n:n+int(5*r.fs)]
                        traces.append(np.median(a.reshape(len(picks),-1,200),axis=2))
                    bandtrace=np.concatenate(traces,axis=1)
                    center=np.median(bandtrace,axis=0)
                    segments=[]
                    for v in np.array_split(center,6):
                        x=np.arange(len(v));res=v-np.polyval(np.polyfit(x,v,1),x)
                        sp=abs(np.fft.rfft(res*np.hanning(len(res))))**2;hz=np.fft.rfftfreq(len(v),.01)
                        band=(hz>=.2)&(hz<=5);component=(hz>=1.0)&(hz<=1.6)
                        segments.append(dict(range_90_adc=float(np.percentile(v,95)-np.percentile(v,5)),
                                    peak_hz=float(hz[band][sp[band].argmax()]) if sp[band].sum()>0 else None,
                                    oscillation_adc_rms=float(np.std(np.fft.irfft(np.fft.rfft(res)*component,n=len(v))))))
                    blocks[group]=dict(connected=keep==group,fractions=fractions,counts=counts,segments=segments)
                switch[name]=blocks
            else:assert sorted(r.mapping)==m
    assert len(man)==len(list(OUT.glob('*/*.h5'))) and len(switch)==2
    raw=load(OUT/'原始信号复核.json')
    with _plot_style():
        fig=Figure(figsize=(10.5,5.0),dpi=150,layout='constrained');wave,sw=fig.subplots(1,2,gridspec_kw={'width_ratios':[1.2,1]})
        for name,label,c in [('as-found-512','原增益 512','#285797'),('diagnostic-112','诊断增益 112','#bf7338')]:
            trace=raw[name]['slow_trace_adc_10ms'];wave.plot(np.arange(500)/100,trace[:500],label=label,color=c,lw=1.5)
        wave.set(xlabel='各段记录起始时间（秒）',ylabel='公共波形（ADC 码）',title='无刺激时已存在周期起伏',ylim=(-35,1058))
        wave.legend(loc='upper right',frameon=False)
        colors={'A':'#356d83','B':'#b65c36'}
        for group,delta in [('A',-.16),('B',.16)]:
            for n,name in enumerate(['A-connected','B-connected']):
                row=switch[name][group];x=n+delta;vals=np.array([v['oscillation_adc_rms'] for v in row['segments']])
                sw.bar(x,np.median(vals),width=.27,label='放大器组 '+group if n==0 else None,color=colors[group],alpha=.65)
                sw.scatter(np.full(6,x)+np.linspace(-.06,.06,6),vals,s=19,edgecolors='white',linewidths=.4,color=colors[group],zorder=3)
        sw.set(xticks=[0,1],xticklabels=['A 接入，B 断开','B 接入，A 断开'],ylabel='1.0–1.6 Hz 波动幅度（ADC 码，RMS）',title='交叉切换 · 六段测量及其中位数')
        sw.legend(loc='upper right',frameon=False)
        for ax in [wave,sw]:ax.spines[['top','right']].set_visible(False)
        _save(fig,OUT/'共同波形与线路切换.png')
    summary=dict(raw_files=len(man),bytes=sum(r['bytes'] for r in man.values()),all_zero_events=True,
                 roi_check_counts=qcs,switch=switch,scope='four one-second switch windows per group, no stimulation')
    (OUT/'诊断核验.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf8')
    print(json.dumps(summary,ensure_ascii=False))
if __name__=='__main__':main()
