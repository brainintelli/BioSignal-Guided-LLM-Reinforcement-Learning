"""Package closed-session evidence without including multi-GB source recordings."""
from pathlib import Path
import json,hashlib,datetime,os,tarfile
B=Path(__file__).resolve().parent
state=json.loads((B/'state.json').read_text())
assert state['status'] in ('completed','stopped') and state.get('finished_utc')
assert not state['cleanup_errors'] and not state.get('audit_write_failed')
assert state['r2_enabled'] is False and state['training_enabled'] is False
pid=int((B/'pipeline.pid').read_text());assert not Path('/proc/'+str(pid)).exists(),'Controller still present'
opened=[]
for p in Path('/proc').iterdir():
 if not p.name.isdigit():continue
 try:
  if (p/'comm').read_text().strip()!='mxwserver':continue
  for fd in (p/'fd').iterdir():
   target=os.readlink(fd)
   if '.h5' in target:opened.append(target)
 except FileNotFoundError:pass
assert not opened,opened
rows=[json.loads(x) for x in (B/'api_audit.jsonl').read_text().splitlines()]
last_units={};last_dac=None
for row in rows:
 if row['kind']!='ack':continue
 words=row['command'].split()
 if words[0]=='mea_set_stimulation_unit':last_units[int(words[1])]=[int(words[2]),int(words[3])]
 if words[0]=='mea_set_dac':last_dac=row['command']
assert len(last_units)==32 and all(x==[0,0] for x in last_units.values())
assert last_dac=='mea_set_dac 0 512 512'
start=datetime.datetime.fromisoformat('2026-11-17T03:29:00+00:00');end=datetime.datetime.fromisoformat(state['finished_utc'])
elapsed=(end-start).total_seconds();assert 0<=elapsed<=7200
def sha(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for chunk in iter(lambda:f.read(16*1024*1024),b''):h.update(chunk)
 return h.hexdigest()
sources=[]
for stage,path in state['stage_files'].items():
 p=Path(path);sources.append({'stage':stage,'path':str(p),'bytes':p.stat().st_size,'sha256':sha(p)})
checks={'controller_exited':True,'no_maxlab_h5_open':True,'all_32_units_off_disconnected_ack':True,'last_direct_dac_zero_ack':True,
 'cleanup_errors':state['cleanup_errors'],'finished_utc':state['finished_utc'],'cumulative_biological_seconds':elapsed,
 'deadline_utc':'2026-11-17T05:29:00+00:00','cumulative_reserved_true_sham':state['cumulative_reserved'],
 'r2_enabled':False,'training_enabled':False,'source_h5':sources,
 'limits':['Off/disconnect/zero are software acknowledgements, not independent physical metering.','Source H5 files retained on acquisition server, excluded from evidence archive.']}
(B/'delivery_checks.json').write_text(json.dumps(checks,indent=2))
paths=sorted(set([p for p in B.iterdir() if p.is_file() and p.suffix in ('.json','.jsonl','.log','.cfg','.py')]+list((B/'dose_claims').glob('*.json'))+[p for p in (B/'analysis').glob('*') if p.is_file()]))
paths=[p for p in paths if p.name!='evidence_manifest.json']
manifest=[{'path':p.relative_to(B).as_posix(),'bytes':p.stat().st_size,'sha256':sha(p)} for p in paths]
(B/'evidence_manifest.json').write_text(json.dumps(manifest,indent=2));paths.append(B/'evidence_manifest.json')
archive=B/'closed_session_evidence.tar.gz'
with tarfile.open(archive,'w:gz') as tar:
 for p in paths:tar.add(p,arcname=p.relative_to(B).as_posix())
print(json.dumps({'checks':checks,'archive':str(archive),'archive_bytes':archive.stat().st_size,'archive_sha256':sha(archive)},indent=2))
