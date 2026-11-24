"""Independent H5 recomputation, without importing sr_data or sr_protocol."""
import hashlib,json,math,statistics
from pathlib import Path
from collections import defaultdict
import h5py,numpy as np
R=Path(__file__).resolve().parent;d=json.loads((R/'dataset.json').read_text(encoding='utf-8'));source=R.parent/'mea_22554_baseline_20261116/source/22554_baseline.raw.h5'
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def digest(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()).hexdigest()
assert sha(source)==d['source']['sha256']=='45a341c80d7ac60eb2b9ba846dce926f9e7fd4b6fb27f0b7b8c963798640f84c'
with h5py.File(source,'r') as f:
    s=f['data_store/data0000'];frames=s['groups/routed/frame_nos'][:];assert np.all(np.diff(frames)==1) and len(frames)==6001600
    assert f['wellplate/id'][0]==b'22554';mapping=[[int(x['channel']),int(x['electrode'])] for x in s['settings/mapping']]
    events=[[i,int(x['frameno']),int(x['channel']),float(x['amplitude'])] for i,x in enumerate(s['spikes'])];lsb=float(s['settings/lsb'][0]);lo,hi=int(frames[0]),int(frames[-1])+1
assert mapping==d['source']['mapping'] and events==d['events'] and digest(mapping)==d['source']['channel_map_sha256']
chs={c for c,e in mapping};excluded=[e[0] for e in events if e[2] not in chs or not lo<=e[1]<hi]
assert excluded==d['source']['excluded_event_indices'] and len(excluded)==135 and d['source']['source_full_structural_passed'] is False
good=[e for e in events if e[0] not in set(excluded)];grouped=defaultdict(list)
for e in good:
    i=(e[1]-lo-20000)//20000
    if 0<=i<298:grouped[i].append(e)
state=[];aux=[];max_error=0.
for i,w in enumerate(d['windows']):
    e=grouped[i];start=lo+20000*(i+1);bins=defaultdict(set)
    for _,frame,ch,a in e:bins[(frame-start)//200].add(ch)
    co=sum(len(bins[(x[1]-start)//200])>1 for x in e)
    values=[len(e),len(e)/850,len({x[2] for x in e})/850,statistics.mean(abs(x[3]) for x in e)*lsb*1e6 if e else 0.]
    coincidence=co/len(e) if e else 0.;assert w['index']==i and w['start_frame']==start and w['end_frame']==start+20000 and w['events_sha256']==digest(e)
    for key,v in zip(('spike_count','rate_hz_per_channel','active_channel_fraction','mean_abs_amplitude_uv'),values):max_error=max(max_error,abs(w[key]-v));assert abs(w[key]-v)<1e-10
    assert abs(w['coincident_event_fraction']-coincidence)<1e-14;state.append(values);aux.append([math.log1p(values[1]),values[2],coincidence])
def fit(x):
    a=np.asarray(x[:60]);c=np.median(a,axis=0);s=1.4826*np.median(np.abs(a-c),axis=0);return c,s
sc,ss=fit(state);ac,aas=fit(aux);profile=d['profile'];assert profile['fit_indices']==list(range(60))
assert np.allclose(sc,profile['state_center'],atol=1e-12,rtol=0) and np.allclose(ac,profile['aux_center'],atol=1e-12,rtol=0)
active=aas>1e-12;prob=np.exp(-.5*np.mean(((np.asarray(aux)[:,active]-ac[active])/aas[active])**2,axis=1));mu=float(prob[:60].mean());u=(prob-mu)/max(mu,1-mu)
assert abs(mu-profile['aux_probability_mean'])<1e-12
z=np.clip((np.asarray(state)-sc)/np.where(ss>1e-12,ss,1),-5,5)
assert np.allclose(z,d['normalization'],atol=1e-10,rtol=0)
for i,s in enumerate(d['scores']):assert abs(s['p']-prob[i])<1e-12 and abs(s['u']-u[i])<1e-12
train=set(range(60,189));ev=set(range(199,296));assert not train&ev and not set(range(60))&(train|ev)
out=dict(passed=True,dataset_sha256=sha(R/'dataset.json'),source_sha256=sha(source),windows=298,excluded_events=135,valid_events=57517,
    source_full_qc_failed_preserved=True,calibration_only_fit=True,max_feature_error=max_error,production_modules_imported=False)
with (R/'data_independent_audit.json').open('x',encoding='utf-8') as f:json.dump(out,f,indent=2)
print(json.dumps(out))
