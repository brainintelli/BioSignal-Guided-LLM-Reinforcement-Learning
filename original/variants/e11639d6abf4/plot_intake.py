from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

root=Path(__file__).resolve().parent
d=json.loads((root/'intake_descriptive.json').read_text())
plt.rcParams.update({'font.family':'DejaVu Sans','axes.spines.top':False,'axes.spines.right':False,'font.size':10})
fig,axes=plt.subplots(2,2,figsize=(12,8),layout='constrained')
color='#147d92';series=np.array(d['network_rate_1s_hz'])
ax=axes[0,0];ax.plot(np.arange(len(series)),series,color=color,lw=1.1)
ax.axhline(d['valid_spikes']/d['duration_s'],color='#db8344',ls='--',label='Whole-record mean')
ax.set(xlabel='Time from first recorded frame (s)',ylabel='Mapped spikes / s',title='Activity over 5 minutes');ax.legend(frameon=False)
mapping=d['mapping'];rates=np.array([d['channel_rates_hz'][str(x['channel'])] for x in mapping])
ax=axes[0,1];im=ax.scatter([x['x_um'] for x in mapping],[x['y_um'] for x in mapping],c=rates,s=10,cmap='viridis')
ax.set(xlabel='Electrode x (µm)',ylabel='Electrode y (µm)',title='Actual mapped electrodes');ax.set_aspect('equal');fig.colorbar(im,ax=ax,label='Spikes / s / channel',shrink=.8)
ax=axes[1,0];ax.hist(rates,bins=35,color=color,alpha=.85);ax.set(xlabel='Spikes / s / channel',ylabel='Channels',title='Channel rate distribution (all 770 mapped)')
ax=axes[1,1];ax.axis('off')
text=(f"Chip: {d['chip_id']}\nRecorded duration: {d['duration_s']:.2f} s\n"
      f"Mapped, in-range spikes: {d['valid_spikes']:,}\nExcluded unmapped / out-of-range: {d['unmapped_or_outside_spikes_excluded']:,}\n"
      f"Mean mapped rate: {d['valid_spikes']/d['duration_s']:.2f} spikes/s\n\n"
      "No stimulation delivered; R2 OFF.\nDescriptive intake, not formal baseline QC.\nNo stimulation effect or learning claim.\nThis record is not admitted to training.")
ax.text(.02,.97,text,va='top',linespacing=1.65,transform=ax.transAxes)
fig.suptitle('28073 | Unstimulated intake',fontsize=18,weight='bold')
fig.savefig(root/'intake_overview.png',dpi=160);fig.savefig(root/'intake_overview.svg')
