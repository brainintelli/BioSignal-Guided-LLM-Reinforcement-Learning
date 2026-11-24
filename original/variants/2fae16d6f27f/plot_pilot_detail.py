"""Post-QC presentation only; uses frozen analysis outputs, performs no new tests."""
from pathlib import Path
import json,numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from analysis_math import rate_bins
B=Path(__file__).resolve().parent;out=B/'analysis'
q=json.loads((B/'pilot_qc.json').read_text());r=json.loads((out/'pilot_summary.json').read_text());z=np.load(out/'pilot_data.npz')
plt.rcParams.update({'figure.dpi':160,'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
fig,axs=plt.subplots(2,2,figsize=(13,8),constrained_layout=True)
x=np.arange(2);names=['Sham (0 mV)','True (298.19 mV nominal)']
for i,(key,label,color) in enumerate([('before_hz','Before: -5..-2 s','#94a3b8'),('after_hz','After: +2..+5 s','#2563eb')]):
 vals=[t[key] for t in r['trials']];bars=axs[0,0].bar(x+(i-.5)*.32,vals,.32,label=label,color=color)
 for b,v in zip(bars,vals):axs[0,0].text(b.get_x()+b.get_width()/2,v+2,f'{v:.2f}',ha='center',fontsize=9)
axs[0,0].set(xticks=x,xticklabels=names,ylim=(0,195),ylabel='Fixed ROI spikes / second',title='A. Both arms rose; post-window rates were nearly equal')
axs[0,0].legend(frameon=False,loc='upper left',fontsize=9)
for i,arm in enumerate(['Sham','True']):axs[0,1].plot(z['trace_seconds']*1000,z['stim_channel_traces_uv'][i],label=arm,color=['#2563eb','#dc2626'][i],lw=.7)
axs[0,1].set(xlim=(-10,400),xlabel='ms from actual recorded event',ylabel='Stim-electrode signal (uV)',title='B. True artifact: onset 0.15 ms, last rail 170.45 ms');axs[0,1].legend(frameon=False)
for i,(label,color) in enumerate([('2-3 s','#0891b2'),('10-11 s','#8b5cf6')]):
 vals=[p['recovery']['fraction_channels_recovered_each_window'][i]*100 for p in q['pulses']]
 bars=axs[1,0].bar(x+(i-.5)*.32,vals,.32,label=label,color=color)
 for b,v in zip(bars,vals):axs[1,0].text(b.get_x()+b.get_width()/2,v-5,f'{v:.2f}%',ha='center',color='white',fontsize=9)
axs[1,0].axhline(95,color='#dc2626',ls='--',label='95% prespecified threshold')
axs[1,0].set(xticks=x,xticklabels=names,ylim=(0,108),ylabel='ROI channels within raw-SD stability limits (%)',title='C. Neither arm passed; late recovery check: 93.37%');axs[1,0].legend(frameon=False,fontsize=8,loc='lower left')
rt=z['spike_seconds'][np.isin(z['spike_channels'],z['roi_channels'])];c,rate=rate_bins(rt,r['duration_seconds'],10)
axs[1,1].plot(c/60,rate,lw=1.2,color='#334155')
for t,color,label in zip(z['event_seconds'],['#2563eb','#dc2626'],['Sham','True']):axs[1,1].axvline(t/60,color=color,ls='--',label=label)
axs[1,1].set(xlabel='Minutes from pilot recording start',ylabel='Fixed ROI spikes / second',title='D. Complete pilot with 10-minute recovery recording');axs[1,1].legend(frameon=False)
fig.suptitle('22554 | QC stopped further stimulation | 1 trial per arm, no efficacy conclusion | R2 OFF',fontsize=14,color='#9a3412')
fig.savefig(out/'pilot_detail.png');fig.savefig(out/'pilot_detail.svg');plt.close(fig)
print('Saved pilot_detail.png and .svg from existing closed-file outputs')
