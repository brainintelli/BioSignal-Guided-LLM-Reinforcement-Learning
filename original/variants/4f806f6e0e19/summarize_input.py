"""Descriptive split features from hash-verified finalized exported events."""
from pathlib import Path
import hashlib
import json
from statistics import mean
from agent.mea_protocol import ResearchStream
from agent.mea_long_ablation import PIN
from agent.mea_snapshot import FEATURE_NAMES

root=Path(__file__).resolve().parent
package=root.parent/'mea_long_20261116/dataset28080_metadata'
assert hashlib.sha256((package/'manifest.json').read_bytes()).hexdigest()==PIN
manifest=json.loads((package/'manifest.json').read_text())
name=manifest['accepted_run']+'/recording.jsonl';path=package/name
assert hashlib.sha256(path.read_bytes()).hexdigest()==manifest['files'][name]['sha256']
stream=ResearchStream(max_spikes=1000000)
with path.open('rb') as f:
    for chunk in iter(lambda:f.read(65536),b''):stream.feed(chunk)
stream.finish();session=stream.session
size=int(session.sample_rate_hz);count=(stream.watermark-session.start_frame)//size
bins=[[] for _ in range(count)];tail=0
for event in stream.spikes:
    i=(event[0]-session.start_frame)//size
    if i<count:bins[i].append(event)
    else:tail+=1
n=len(session.channels)
features=[(float(len(b)),len(b)/n,len({e[1] for e in b})/n,mean(abs(e[2]) for e in b) if b else 0.) for b in bins]
splits={}
for label,lo,hi in [('train_current',0,1024),('train_boundary',1024,1025),('gap',1025,1085),('eval_current',1085,1685),('eval_boundary',1685,1686),('unused',1686,count)]:
    rows=features[lo:hi]
    splits[label]=dict(range_half_open=[lo,hi],windows=len(rows),events=int(sum(r[0] for r in rows)),zero_spike_windows=sum(r[0]==0 for r in rows),
        features={name:dict(mean=mean(r[i] for r in rows),minimum=min(r[i] for r in rows),maximum=max(r[i] for r in rows)) for i,name in enumerate(FEATURE_NAMES)})
result=dict(source_manifest_sha256=PIN,chip_id='28080',channels=n,full_windows=count,tail_events=tail,
    splits=splits,interpretation='descriptive only; no stationarity or independent biological replication claim')
(root/'input_summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps({k:{a:v[a] for a in ('windows','events','zero_spike_windows')} for k,v in splits.items()}))
