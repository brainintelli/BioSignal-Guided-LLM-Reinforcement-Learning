"""Independent raw-event reconstruction and final provisional actor input."""
import hashlib,json,math
from pathlib import Path
from collections import Counter
import h5py,numpy as np
r=Path(__file__).resolve().parent
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def read(n):return json.loads((r/n).read_text(encoding='utf-8'))
original=read('state2_provisional.json');manifest=read('source_manifest.json')
for name,q in manifest['files'].items():assert sha(r/name)==q['sha256']
with h5py.File(r/'source.raw.h5','r') as f:
    d=f['data_store/data0000'];raw_map=[(int(x['channel']),int(x['electrode'])) for x in d['settings/mapping'][:]]
    configured=set(json.loads(f['assay/inputs/electrodes'][0].decode())['electrodes']['0']['stimulation'])
    excluded={c for c,e in raw_map if e in configured};valid=sorted((c,e) for c,e in raw_map if c not in excluded)
    assert len(set(c for c,e in valid))==len(valid)==len(set(e for c,e in valid))
    assert [list(x) for x in valid]==original['mapping']
    ids={c for c,e in valid};w=original['window'];lo=w['start_frame'];hi=w['end_frame']
    markers=d['events']['frameno'];assert lo==int(markers.max())+61*20000 and hi==lo+20000
    records=d['spikes'][:];lsb=float(d['settings/lsb'][0]);events=[]
    left=int(np.searchsorted(records['frameno'],lo));right=int(np.searchsorted(records['frameno'],hi))
    for x in records[left:right]:
        if int(x['channel']) in ids:events.append([int(x['frameno']),int(x['channel']),float(x['amplitude'])*lsb*1e6])
    events.sort();assert events==w['events']
    counts=Counter(e[1] for e in events);n=len(events)
    values=[float(n),sum(counts.values())/len(valid),sum(v>0 for v in counts.values())/len(valid),math.fsum(abs(e[2]) for e in events)/n if n else 0.]
    error=max(abs(a-b) for a,b in zip(values,w['features']));assert error<1e-10
    assert hi<=int(records['frameno'].max())+1
original['window']['feature_names']=['spike_count','rate_hz_per_channel','active_channel_fraction','mean_abs_amplitude_uv']
original['window']['features']=values
with (r/'probe_input.json').open('x',encoding='utf-8') as f:json.dump(original,f,ensure_ascii=False,indent=2,allow_nan=False)
receipt=dict(numerical_reconstruction_passed=True,source_sha256=sha(r/'source.raw.h5'),source_manifest_sha256=sha(r/'source_manifest.json'),
    probe_input_sha256=sha(r/'probe_input.json'),events=n,channels=len(valid),max_feature_difference=error,
    raw_quality_verified=False,formal_training_admitted=False,normalization='identity_engineering_only',
    note='Corrected displayname population_rate_hz to rate_hz_per_channel; calculationunchanged. Initialfullmapuniquenesscheckrejectedduplicate807/823; allstimchannelsremovedbeforeunique-subsetcheck.')
with (r/'input_verification.json').open('x',encoding='utf-8') as f:json.dump(receipt,f,indent=2)
print(json.dumps(receipt,indent=2))
