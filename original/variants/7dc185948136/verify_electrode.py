from pathlib import Path
import json,os,math
root=Path.cwd();out=Path(__file__).resolve().parent
os.environ['HDF5_PLUGIN_PATH']=str(root/'.tools/mea_maxwell_hdf5_plugin')
import h5py,numpy as np
from matplotlib import pyplot as plt,font_manager
data=root/'output/mea_full_28080_20261117_1406'
r=json.loads((out/'forward-result.json').read_text(encoding='utf8'));eid=r['stim_electrode'];channel=r['excluded_stim_channel']
with h5py.File(data/'backup/recording_resume2/stimulus_post.raw.h5') as f:
 g=f['data_store/data0000'];buf=g['groups/routed'];ids=buf['channels'][:];frames=buf['frame_nos'];raw=buf['raw'];start=int(frames[0]);lsb=float(g['settings/lsb'][0])*1e3;rate=float(g['settings/sampling'][0])
 phys=g['settings/mapping'][:];site,=phys[phys['channel']==channel]; assert site['electrode']==eid
 idx,=np.flatnonzero(ids==channel);events=list(map(int,g['events'][:]['frameno']))
 assert events==r['stim_event_frames'] and len(events)==15 and rate==20000
 t0=events[0]-start;med=float(np.median(raw[idx,t0-20000:t0-2000]))
 waveform=[];d=[]
 for group in range(5):
  group_events=events[group*3:group*3+3];t=group_events[0]-start
  assert group_events==[events[0]+4000*group+200*j for j in range(3)]
  assert np.array_equal(frames[t-60:t+580],np.arange(t+start-60,t+start+580))
  v=(raw[idx,t-60:t+580].astype(float)-med)*lsb
  waveform.append(v)
  for j,frame in enumerate(group_events):
   at=60+200*j;peak=float(np.max(np.abs(v[at:at+40])))
   prior=float(np.max(np.abs(v[at-8:at])))
   segment=v[at:at+40]
   d.append({'frame':frame,'group':group+1,'pulse':j+1,'max_abs_adc_mv_next_2ms':peak,'max_abs_adc_mv_previous_0p4ms':prior,'swing_gt_2mV':bool(np.ptp(segment)>2),'separate_crossing_1mV':bool(prior<1 and peak>=1)})
 tail=events[-1]-start+8
 first_settled=None
 for start_s in range(60):
  vals=raw[idx,tail+start_s*20000:tail+(start_s+1)*20000].astype(float).reshape(100,200)
  good=(np.max(np.abs(vals-med),axis=1)*lsb<1)&(np.abs(np.median(vals,axis=1)-med)<=5)&(np.min(vals,axis=1)>0)&(np.max(vals,axis=1)<1023)
  for tick,passed in enumerate(good):
   if start_s==0 and tick==0:streak=0
   streak=streak+1 if passed else 0
   if streak==5:
    first_settled=round(start_s+(tick-4)*.01,3)
    break
  if first_settled is not None:break
pos=sum(x['separate_crossing_1mV'] for x in d)
res=dict(chip_id='28080',electrode=int(eid),channel=int(channel),xy_um=[float(site['x']),float(site['y'])],markers=len(events),onset_hits=pos,criterion='ADC relative to pretrain median; |signal| below 1 mV in previous 0.4 ms and crosses 1 mV within 2 ms. Engineering check of recorded channel only.',pulse_checks=d)
res.update(fast_swing_gt_2mV_count=sum(v['swing_gt_2mV'] for v in d),first_50ms_settled_since_last_tail_s=first_settled,settled_definition='Five consecutive 10ms bins with no ADC rails, median within 5 ADC of pretrain reference and |signal| <1mV throughout; search first 60s after last pulse tail, on stimulation readback only.')
(out/'electrode-readback.json').write_text(json.dumps(res,indent=2),encoding='utf8')
font_manager.fontManager.addfont('C:/Windows/Fonts/msyh.ttc')
plt.rcParams.update({'font.family':font_manager.FontProperties(fname='C:/Windows/Fonts/msyh.ttc').get_name(),'axes.unicode_minus':False})
fig,ax=plt.subplots(figsize=(9,4.2),dpi=145);fig.subplots_adjust(top=.74,bottom=.20,left=.12,right=.95)
fig.text(.08,.89,'28080  |  电极 8858 的刺激事件与原始 MEA 波形',fontsize=16,weight='bold',color='#173f4a')
fig.text(.08,.795,'5 组实测波形按每组首个事件对齐；虚线为设备事件。',size=10,color='#48636d')
x=(np.arange(640)-60)/20
for v in waveform:ax.plot(x,v,lw=.85,alpha=.55,color='#206e78')
for k in (0,10,20):ax.axvline(k,ls=':',color='#ad5c40',lw=1.1)
ax.set(xlabel='相对每组首个事件（ms）',ylabel='记录通道相对基线（mV）',xlim=(-3,29))
ax.spines[['right','top']].set_visible(False);fig.savefig(out/'electrode-readback.png')
print(json.dumps({'onsets':pos,'count':len(d),'actual_xy_um':res['xy_um'],'fast_swings':res['fast_swing_gt_2mV_count'],'settled_s':first_settled}))
