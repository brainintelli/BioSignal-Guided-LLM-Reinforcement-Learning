"""Independently audit coincident events and common-mode voltage in retained channels."""
import json
import os
import runpy
from pathlib import Path
from register import out,rd,store

code="""import sys,os,json
from pathlib import Path
os.environ['HDF5_PLUGIN_PATH']='/pub/soft/MaxLab/so'
sys.path.insert(0,'/home/researcher/brainDesktop/mea-standby-20261118')
import session as station
import numpy as np
p,panels=map(json.loads,sys.argv[1:3]);out={}
for k,ch in panels.items():
 with station.H5Recording(Path(p['source'][k]['path']),station.CFG) as rec:
  ev=rec.events[0];out[k]={}
  frames=np.rint(rec.spike_times*20000).astype(np.int64)+rec.first_frame
  for off in [-3,12]:
   start=ev['frame']+off*20000
   y=rec.sample(ch,ev['time_s']+off).astype('float32')
   centered=(y-np.median(y,axis=1)[:,None])*rec.metadata['lsb_uv']
   cm=np.median(centered,axis=0); hit=np.abs(centered)>250
   idx=(frames>=start)&(frames<start+20000)&np.isin(rec.spike_channels,ch)
   t=frames[idx]-start;cs=rec.spike_channels[idx]
   pairs=np.unique(np.column_stack((t//20,cs)),axis=0)
   out[k][off]=dict(spikes=int(sum(idx)),active=int(len(np.unique(cs))),common_median_peak_uv=float(np.max(np.abs(cm))),
       max_simultaneous_channels_over_250uv=int(hit.sum(axis=0).max()),max_spiking_channels_in_1ms=int(np.bincount(pairs[:,0],minlength=1000).max()),
       samples_at_rail=int(np.count_nonzero((y==0)|(y==1023))))
print(json.dumps(out))"""
v=runpy.run_path(str(Path(os.environ['TEMP'])/'codex_live22554_1119.py'))
with v['connect']() as c:
    r=json.loads(v['execute'](c,code,[json.dumps(rd(out/'paired-protocol.json')),
                             json.dumps({k:v['channels'] for k,v in rd(out/'paired-windows.json').items()})],timeout=120))
store(out/'transients.json',r)
print(r)
