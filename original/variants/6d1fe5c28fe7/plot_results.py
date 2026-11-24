"""Plot observed finalized receipts only; never supplies missing outcomes."""
from pathlib import Path
import argparse,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def setup():
 plt.rcParams.update({'font.family':'DejaVu Sans','axes.spines.top':False,'axes.spines.right':False,'font.size':10})

def biological(state,out):
 if state['status'] not in ('completed','stopped','failed'):raise ValueError('Final acquisition state required')
 qc=state.get('qc',{});base=qc.get('baseline',{});pilot=qc.get('pilot',{});pairs=state.get('accepted_pairs',[])
 fig,axs=plt.subplots(2,2,figsize=(12,8),layout='constrained')
 ax=axs[0,0];trend=base.get('activity_trend_30s',[])
 if trend:
  ax.plot([r['start_s']/60 for r in trend],[r['rate_hz'] for r in trend],color='#147d92')
  ax.axvline(base['connected_time_s']/60,color='grey',ls='--',label='Zero-output connection')
  ax.legend(frameon=False)
 else:ax.text(.5,.5,'No qualified baseline rate series',ha='center',transform=ax.transAxes)
 ax.set(title='Baseline: recorded ROI activity',xlabel='Minutes from raw first frame',ylabel='ROI spikes / s')
 ax=axs[0,1];pulses=pilot.get('pulses',[])
 if pulses:
  for i,p in enumerate(pulses):
   y=[r['fraction_channels_passed']*100 for r in p['recovery']]
   ax.plot([2,10],y,'o-',label=p['arm'],color='#db8344' if p['arm']=='true' else '#147d92')
  ax.axhline(95,color='#a84646',ls='--',label='Required 95%');ax.set_ylim(0,102);ax.legend(frameon=False)
 else:ax.text(.5,.5,'Pilot not reached / no diagnostic receipt',ha='center',transform=ax.transAxes)
 ax.set(title='Pilot: waveform recovery QC',xlabel='Post-event window start (s)',ylabel='Channels passing MAD/shift/rail checks (%)')
 ax=axs[1,0]
 if pairs:
  for q in pairs:
   tr={t['arm']:t for t in q['trials']};ax.scatter(q['pair'],tr['true']['delta_hz']-tr['sham']['delta_hz'],color='#147d92')
  ax.axhline(0,color='grey',ls='--');ax.set(xlabel='Qualified pair',ylabel='True minus sham delta (spikes / s)')
 else:
  tr=pilot.get('trials',[])
  if tr:
   ax.bar([t['arm'] for t in tr],[t['delta_hz'] for t in tr],color=['#147d92' if t['arm']=='sham' else '#db8344' for t in tr])
   ax.axhline(0,color='grey',ls='--');ax.set_ylabel('Post +2..5 minus pre -5..-2 (spikes / s)')
  else:ax.text(.5,.5,'No paired stimulation result',ha='center',transform=ax.transAxes)
 ax.set_title('Qualified paired differences' if pairs else 'Pilot response: diagnostic only')
 ax=axs[1,1];ax.axis('off')
 message=(f"Final status: {state['status']}\nReserved attempts: {state['dose_totals']}\n"
          f"Baseline candidate channels: {len(base.get('candidate_roi',[]))}\nFrozen pilot ROI: {len(pilot.get('roi_channels',[]))}\n"
          f"Qualified formal pairs: {len(pairs)} / 10\nR2 OFF; one temporary biological session.\n\n"
          'Stop/failure reason:\n'+'\n'.join(state.get('errors',[]))[:650])
 ax.text(.01,.97,message,va='top',wrap=True,linespacing=1.5,transform=ax.transAxes)
 fig.suptitle('28073 | Actual acquisition and stimulation QC',fontsize=17,weight='bold')
 for ext in ('png','svg'):fig.savefig(out/('biological_comparison.'+ext),dpi=160)
 plt.close(fig)

def training(result,out):
 if result['status']!='complete':raise ValueError('Complete verified training analysis required')
 names=['baseline','state','shuffled'];labels=['R1 only','R1 + recorded state','R1 + shuffled state']
 fig,axs=plt.subplots(1,2,figsize=(12,4.7),layout='constrained')
 ax=axs[0];x=np.arange(3);before=[result['arms'][a]['pre'] for a in names];after=[result['arms'][a]['post'] for a in names]
 ax.plot(x,before,'o--',label='Before training',color='#829298');ax.plot(x,after,'o-',label='After training',color='#147d92')
 ax.set_xticks(x,labels,rotation=12);ax.set_ylabel('Mean discounted original R1');ax.set_title('Eight matched digital seed blocks');ax.legend(frameon=False)
 ax=axs[1]
 for i,name in enumerate(('state-baseline','state-shuffled')):
  p=result['primary'][name]['post'];d=np.array(p['differences']);mean=p['mean_difference'];lo,hi=p['bootstrap_95']
  ax.scatter(d,np.full(8,i)+np.linspace(-.09,.09,8),s=24,color='#80989e')
  ax.errorbar(mean,i,xerr=[[mean-lo],[hi-mean]],fmt='D',capsize=4,color='#147d92')
  ax.text(.98,.9-i*.15,f"Adjusted p = {p['p_adjusted_primary']:.3g}",ha='right',transform=ax.transAxes)
 ax.axvline(0,color='grey',ls='--');ax.set_yticks([0,1],['State − R1 only','State − shuffled']);ax.set_xlabel('Post-training R1 difference; bootstrap 95% interval');ax.set_ylim(-.5,1.5)
 fig.suptitle('Offline prerecorded state comparison | R2 disabled',fontsize=16,weight='bold')
 for ext in ('png','svg'):fig.savefig(out/('model_comparison.'+ext),dpi=160)
 plt.close(fig)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--state',type=Path);p.add_argument('--analysis',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True);setup()
 if a.state:biological(json.loads(a.state.read_text()),a.output)
 if a.analysis:training(json.loads(a.analysis.read_text()),a.output)
