"""Independent analysis of raw no-output recordings, preserving the original ROI and clock."""
from pathlib import Path
import json
import sys
from types import SimpleNamespace
OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(OUT.parent))
from survey_report import H5Recording, load, np, Figure, _plot_style, _save
from mea_stim_v2.qc import stability

TIMES=[3,9,16,23,30,37,43,50,57]
NAMES={'previous':'上轮接线静默','as-found-512':'本次初始','reinitialized-512':'初始化后','diagnostic-112':'低增益诊断','restored-512':'恢复原增益','settled-full-512':'切回完整线路后'}

def measure(file,b,roi):
    with H5Recording(file,SimpleNamespace(**b)) as r:
        assert r.metadata['chip_id']==b['chip_id'] and not r.metadata['frame_gaps'] and not r.events
        assert set(roi)<=set(r.channels.tolist())
        roi=sorted(roi,key=lambda c:dict(r.mapping)[c])
        probe=roi[::3]
        pos=np.array([r.channel_xy_um[c] for c in probe])
        ii,jj=np.triu_indices(len(probe),1)
        dist=np.sqrt(((pos[ii]-pos[jj])**2).sum(axis=1))
        # H5Recording verifies that the raw dataset uses the physical amplifier row as its first axis.
        # Unmapped rows are an electronics comparison only. Their electrical termination is unknown.
        unmapped=sorted(set(range(1024))-set(r.channels.tolist()))
        spare=[unmapped[k] for k in np.linspace(0,len(unmapped)-1,min(48,len(unmapped)),dtype=int)]
        chrono=[];checked=[0,0,0];baseline=r.sample(roi,3)
        for j,t in enumerate((23,43,57)):
            checked[j]=int(sum(stability(baseline,r.sample(roi,t))['channel_pass']))
        for t in TIMES:
            raw=r.sample(roi,t)
            vals=np.median(raw,axis=1);drift=vals-np.median(baseline,axis=1);rails=(raw==0)|(raw==1023)
            # The correlation of simultaneous ADC fluctuations on widely spaced electrodes.
            wave=raw[::3].astype(float);wave-=np.mean(wave,axis=1,keepdims=True)
            scale=np.std(wave,axis=1);valid=(scale[ii]>0)&(scale[jj]>0)
            w=wave/np.where(scale>0,scale,1)[:,None]
            corr=(w@w.T/w.shape[1])[ii,jj]
            common=np.median(w,axis=0);taper=np.hanning(len(common));ps=abs(np.fft.rfft(common*taper))**2
            freq=np.fft.rfftfreq(len(common),1/r.fs);bins=(freq>=1)&(freq<=1000)
            line=((abs(freq-50)<=1)|(abs(freq-100)<=1));den=float(ps[bins].sum())
            def med(x):return float(np.median(x)) if len(x) else None
            reference=r._raw[spare,int(t*r.fs):int((t+1)*r.fs)]
            spare_rails=(reference==0)|(reference==1023)
            chrono.append(dict(t=t,rail_channels=int(rails.any(axis=1).sum()),
                unassigned_rail_rows=int(spare_rails.any(axis=1).sum()),
                unassigned_median_adc=float(np.median(np.median(reference,axis=1))),
                low_rail_samples_fraction=float((raw==0).mean()),high_rail_samples_fraction=float((raw==1023).mean()),
                fraction_frames_half_roi_clipped=float((rails.sum(axis=0)>=len(roi)/2).mean()),
                median_voltage_adc=float(np.median(vals)),
                rail_any_frame_max=int(rails.sum(axis=0).max()),median_drift_adc=float(np.median(drift)),
                drifted_channels=int((abs(drift)>5).sum()),median_pair_corr=med(corr[valid]),
                far_pair_corr=med(corr[valid&(dist>=400)]),far_pairs=int((valid&(dist>=400)).sum()),
                common_peak_hz=float(freq[bins][np.argmax(ps[bins])]) if den else None,
                common_line_fraction=float(ps[line].sum()/den) if den else None))
        common=[]
        subset=roi[::10]
        for start in range(0,60,5):
            raw=r.sample(subset,start,5)
            # Retain 10-ms medians for slow common-mode signal; raw rails are assessed above at 20 kHz.
            common.extend(np.median(np.median(raw.reshape(len(subset),-1,200),axis=2),axis=0).tolist())
        v=np.asarray(common);pieces=[]
        for half in np.array_split(v,2):
            x=np.arange(len(half));detrended=half-np.polyval(np.polyfit(x,half,1),x)
            power=abs(np.fft.rfft(detrended*np.hanning(len(half))))**2
            frequency=np.fft.rfftfreq(len(half),.01);band=(frequency>=.1)&(frequency<=10)
            peak=int(np.argmax(power[band]));pieces.append(float(frequency[band][peak]) if power[band].sum()>0 else None)
        return dict(metadata=r.metadata,roi=len(roi),correlation_channels=len(probe),unassigned_rows=spare,checked=checked,
                    slow_trace_adc_10ms=common,slow_peak_hz_per_30s=pieces,
                    spikes_on_roi=int(np.isin(r.spike_channels,roi).sum()),windows=chrono)

def analyze():
    b=load(OUT.parent/'baseline-binding.json');old=OUT.parent.parent/'cpp-crossover-v1'
    roi=load(old/'design.json')['roi']
    files=[('previous',old/'warmup/warmup.raw.h5')]
    names=list(NAMES)[1:]+[p.parent.name for p in OUT.glob('independent-*/quiet.raw.h5') if 'connected' not in p.parent.name]
    files.extend((n,OUT/n/'quiet.raw.h5') for n in names)
    results={}
    for name,file in files:
        if not file.exists():continue
        m=measure(file,b,roi);results[name]=m
        qc=file.parent/'qc.json'
        if name!='previous':
            assert m['checked']==load(qc)['stable_counts']
        print(name,'QC',m['checked'],'clipped',[(v['rail_channels'],v['far_pair_corr']) for v in m['windows']],flush=True)
    (OUT/'原始信号复核.json').write_text(json.dumps(results,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf8')
    if len(results)<2:return
    with _plot_style():
        fig=Figure(figsize=(10.5,5.6),dpi=150,layout='constrained');clip,qc_ax=fig.subplots(1,2)
        colors=['#9299a3','#285797','#aa551f','#22816e','#743e84','#52502b','#227799']
        for (name,v),color in zip(results.items(),colors*(1+len(results)//len(colors))):
            label=NAMES.get(name,name)
            clip.plot(TIMES,[100*w['rail_channels']/len(roi) for w in v['windows']],marker='o',ms=3,color=color,label=label)
            qc_ax.plot([23,43,57],[100*n/len(roi) for n in v['checked']],marker='o',color=color)
        clip.set(title='原始电压触及量程边界',xlabel='静默记录时间（秒）',ylabel='固定读出通道占比（%）',ylim=(-3,104))
        qc_ax.set(title='相对第 3 秒基线的稳定通道',xlabel='静默记录时间（秒）',xticks=[23,43,57],ylim=(-3,104))
        qc_ax.axhline(95,color='#333333',ls=':',lw=1);qc_ax.text(23,93,'原方案阈值 95%',va='top')
        for ax in (qc_ax,clip):ax.spines[['top','right']].set_visible(False)
        fig.suptitle('27701 静默诊断 · 同一路由的 158 个读出通道')
        fig.legend(*clip.get_legend_handles_labels(),loc='outside lower center',ncol=3,frameon=False)
        _save(fig,OUT/'静默诊断.png')
if __name__=='__main__':analyze()
