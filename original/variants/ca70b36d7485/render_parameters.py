"""Plot the verified software waveform; no device access or image editing."""
from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

root=Path(__file__).resolve().parent
p=json.loads((root/'sequence_preview.json').read_text())
a=p['nominal_phase_amplitude_mv']
t=[0.0];v=[0.0]
for onset in p['pulse_onset_ms']:
    t.extend([onset,onset,onset+.2,onset+.2,onset+.4,onset+.4])
    v.extend([0,a,a,-a,-a,0])
t.append(850);v.append(0)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
fig,axes=plt.subplots(2,1,figsize=(12,6.5),gridspec_kw={'height_ratios':[1.1,1]},layout='constrained')
fig.suptitle('Photo-parameter trial | software waveform preview',fontsize=17,weight='bold')
for ax in axes:
    ax.plot(t,v,color='#167a91',lw=1.5)
    ax.axhline(0,color='#7b8794',lw=.7)
    ax.set_ylabel('Nominal phase voltage (mV)');ax.set_ylim(-700,700)
    ax.set_yticks([-600,0,600]);ax.grid(axis='y',alpha=.18)
axes[0].set_xlim(0,850);axes[0].set_xlabel('Time from sequence start (ms)')
for i in range(5):
    axes[0].axvspan(5+i*200,25.4+i*200,color='#167a91',alpha=.12)
    axes[0].text(15+i*200,635,f'Burst {i+1}',ha='center',fontsize=9)
axes[0].set_title('5 bursts × 3 pulses = 15 pulses | burst onset spacing 200 ms',loc='left',fontsize=11)
axes[1].set_xlim(4.7,5.7);axes[1].set_xlabel('Time from sequence start (ms), first pulse enlarged')
axes[1].set_title('Each phase 200 µs | zero interphase gap | return to zero',loc='left',fontsize=11)
axes[1].annotate('Target 600 mV\n201 DAC steps → 599.362101 mV nominal',xy=(5.1,a),xytext=(5.28,330),arrowprops={'arrowstyle':'->','color':'#425466'},fontsize=10)
fig.supxlabel('Software preview only. Actual electrode voltage, injected charge and neural response have not been measured here.',fontsize=9,color='#52606d')
fig.savefig(root/'parameter_preview.png',dpi=180)
fig.savefig(root/'parameter_preview.svg')
plt.close(fig)
print(root/'parameter_preview.png')
