import json,os,sys,hashlib,math
from pathlib import Path
root=Path.cwd()
pkg=root/'standalone/mea_state2_r2'; sys.path[:0]=[str(root/'.tools/mea_baseline_deps'),str(pkg),str(root/'standalone/mea_binary_r2'),str(root)]
os.environ['HDF5_PLUGIN_PATH']=str(root/'.tools/mea_maxwell_hdf5_plugin')
import numpy as np,h5py
from agent.mea_contract import observation_key
from mea_state2_r2.features import extract,transform,digest
from mea_state2_r2.data import Dataset
rawdir=root/'output/mea_full_28080_20261117_1406'; campaign=root/'output/mea_state2_r2_20261118'
source=root/'output/mea_28080_shuffle_20261117_v2/data'
out=root/'output/28080_offline_response_probe_20261118_1546'
out.mkdir(exist_ok=True)
def read(p):return json.loads(p.read_text(encoding='utf8'))
def write(name,d):
 with (out/name).open('x',encoding='utf8') as f:json.dump(d,f,ensure_ascii=False,indent=2,allow_nan=False)
qc=read(rawdir/'recording_resume2/stimulus_post_qc.json')
route=read(rawdir/'recording_resume2/selected_route.json')
assert qc['status']=='passed' and route['electrode']==8858 and route['channel']==640
m=read(rawdir/'recording_resume2/raw_manifest.json'); expected_sha={r['path']:r['sha256'] for r in m}
data=Dataset(source);norm=data.normalization
assert data.normalization_id==read(campaign/'dataset.json')['normalization_id']
evaldir=campaign/'accepted_evidence/experiment01/b00_C/eval'
def firstrow(phase):
 with (evaldir/phase/'rollout.jsonl').open(encoding='utf8') as f:return json.loads(next(f))
original={p:firstrow(p) for p in ('before','after')}; obs=original['before']['observation']
assert obs==original['after']['observation']
ctx=observation_key(obs);samples=[];markers=[];sha_used={}
for phase,fn in [('before','baseline.raw.h5'),('after','stimulus_post.raw.h5')]:
 path=rawdir/'backup/recording_resume2'/fn
 h=hashlib.sha256()
 with path.open('rb') as f:
  for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
 sha_used[fn]=h.hexdigest()
 assert sha_used[fn]==expected_sha[fn]
 with h5py.File(path,'r') as file:
  g=file['data_store/data0000'];sp=g['spikes'][:];ch=sp['channel'];fr=sp['frameno']
  e=g['events'][:];mapping={int(r['channel']):int(r['electrode']) for r in g['settings/mapping'][:]}
  chans=tuple(c for c,_ in data.recordings[phase].mapping);assert len(chans)==804 and 640 not in chans and all(mapping[c]==e for c,e in data.recordings[phase].mapping)
  if phase=='after':
   markers=[int(v) for v in e['frameno']];assert markers==[r['frame'] for r in qc['events']] and len(markers)==15
  mask=np.isin(ch,chans);filt=fr[mask]; channels=ch[mask]
  for i in range(16):
   sn=data.snapshot(i,phase,ctx,'C',0,'eval');lo=sn['window_start'];hi=sn['window_end'];keep=(filt>=lo)&(filt<hi)
   pairs=sorted((int(f),int(c)) for f,c in zip(filt[keep],channels[keep])); orig=data.recordings[phase].rows[sn['window_index']]['events']
   assert pairs==[(f,c) for f,c,_ in orig]
   values=extract([(f,c,0.) for f,c in pairs],lo,hi,chans,20000)
   assert list(values)==sn['features'] and list(transform(values,norm))==sn['normalized']
   if i==0:
    archived=original[phase]['snapshot']
    assert all(sn[k]==archived[k] for k in ('window_index','window_start','window_end','events_sha256','normalized'))
    assert all(math.isclose(x,y,abs_tol=1e-12) for x,y in zip(sn['features'],archived['features']))
   samples.append(dict(phase=phase,i=i,spikes=len(pairs),frame_channel_pairs=pairs,snapshot=sn))
result=dict(chip='28080',source_raw_sha256=sha_used,stim_electrode=route['electrode'],excluded_readout=route['channel'],independent_channels=804,stim_event_frames=markers,last_pulse_end_frame=markers[-1]+8,sample_rate_hz=20000,obs=obs,normalization_id=data.normalization_id,normalization=norm,samples=samples,archived_first={p:{k:original[p][k] for k in ('verb','verb_logp','value','action')} for p in original},checkpoint='run01/experiment01/b00_C/train/round_03/checkpoint',rule='first 16 registered held-out windows (445..460) of each phase, b00_C final frozen checkpoint; same archived initial game observation; no training')
write('inference-input.json',result)
print(json.dumps({'windows':len(samples),'events':len(markers),'post_delay_s':(samples[16]['snapshot']['window_start']-result['last_pulse_end_frame'])/20000,'hashes':sha_used}))
