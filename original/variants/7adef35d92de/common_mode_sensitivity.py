"""Sensitivity of native spike counts to globally synchronous voltage excursions."""
import json
import os
import runpy
import sys
from pathlib import Path
from register import out,rd,store

code="""import sys,os,json
from pathlib import Path
os.environ['HDF5_PLUGIN_PATH']='/pub/soft/MaxLab/so'
sys.path.insert(0,'/home/researcher/brainDesktop/mea-standby-20261118')
import session as station
import numpy as np
from mea_state2_r2.features import extract
p,panels=map(json.loads,sys.argv[1:3]);out={}
for k,ch in panels.items():
 with station.H5Recording(Path(p['source'][k]['path']),station.CFG) as rec:
  ev=rec.events[0];out[k]={}
  frames=np.rint(rec.spike_times*20000).astype(np.int64)+rec.first_frame
  for off in [-3,12]:
   start=ev['frame']+off*20000
   y=rec.sample(ch,ev['time_s']+off).astype('float32')
   cm=np.median(y-np.median(y,axis=1)[:,None],axis=0)
   hit=(np.abs(cm)>=4)
   blank=np.convolve(hit.astype('int16'),np.ones(81,dtype='int16'),'same')>0
   idx=(frames>=start)&(frames<start+20000)&np.isin(rec.spike_channels,ch)
   t=frames[idx];cs=rec.spike_channels[idx];ok=~blank[t-start]
   evs=sorted((int(f),int(c),0.) for f,c in zip(t[ok],cs[ok]))
   out[k][off]=dict(masked_frames=int(blank.sum()),excluded_spikes=int((~ok).sum()),retained_spikes=len(evs),features=extract(evs,start,start+20000,ch,20000))
print(json.dumps(out))"""
panels={k:v['channels'] for k,v in rd(out/'paired-windows.json').items()}
shared='--shared' in sys.argv
if shared:
    common=sorted(set.intersection(*(set(p) for p in panels.values())))
    panels={k:common for k in panels}
v=runpy.run_path(str(Path(os.environ['TEMP'])/'codex_live22554_1119.py'))
with v['connect']() as c:
    data=json.loads(v['execute'](c,code,[json.dumps(rd(out/'paired-protocol.json')),
                        json.dumps(panels)],timeout=120))
record=dict(threshold_ADC=4,half_width_samples=40,results=data)
if shared:record['common_channels']=common
store(out/('shared-panel-check.json' if shared else 'common-mode-sensitivity.json'),record)
print({k:[(a['masked_frames'],a['excluded_spikes'],a['retained_spikes']) for a in x.values()] for k,x in data.items()})
