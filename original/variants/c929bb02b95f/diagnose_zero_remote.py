import json,sys
from pathlib import Path
import h5py,numpy as np
base=Path('/home/researcher/mea_full_28080_20261117_1406');folder=base/'multi_bounded_resume1'
routes=json.loads((folder/'connected_routes.json').read_text());selected={r['channel'] for r in routes}
with h5py.File(folder/'identity.raw.h5','r') as f:
 d=f['data_store/data0000'];g=d['groups/routed'];raw=g['raw'];m=d['settings/mapping'][:];channels=np.array([int(x['channel']) for x in m]);n=raw.shape[1]
 bysecond=[];tot=np.zeros(len(channels),dtype='i8');total=0
 for lo in range(0,n,20000):
  b=raw[:,lo:min(n,lo+20000)][channels,:];counts=((b==0)|(b==1023)).sum(axis=1);tot+=counts;total+=b.shape[1]
  if np.any(counts):bysecond.append({'second':lo/20000,'rails':[{ 'channel':int(channels[i]),'selected':int(channels[i]) in selected,'fraction':float(counts[i]/b.shape[1])} for i in np.where(counts)[0]]})
 frames=g['frame_nos'];events=[{'id':int(x['eventid']),'frame':int(x['frameno']),'t':(int(x['frameno'])-int(frames[0]))/20000} for x in d['events'][:]]
 result={'chip':f['wellplate/id'][0].decode(),'duration':n/20000,'mapping_count':len(m),'events':events,'rail_seconds':bysecond,
  'total_rail_samples':int(tot.sum()),'max_channel_rail_fraction':float(tot.max()/total)}
 (folder/'zero_failure_diagnosis.json').write_text(json.dumps(result,indent=2))
 print(json.dumps({'chip':result['chip'],'duration':result['duration'],'event_count':len(events),
  'event_first':events[:3],'event_last':events[-3:],'rail_seconds_count':len(bysecond),
  'rail_examples':bysecond[:4],'rail_last':bysecond[-2:],'max_channel_rail_fraction':result['max_channel_rail_fraction']},indent=2))
