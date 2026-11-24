from pathlib import Path
from datetime import datetime,timezone,timedelta
from collections import Counter
import json,hashlib,subprocess,sys,gzip,io
base=Path('/home/researcher/meadowmind-gpu')
project=base/'m9-research-v5-project'
sys.path.insert(0,str(project))
from agent.m9_metrics import summarize,select_arm,compare
from agent.rl_contract import transition_at_horizon
job=base/'m9-training-jobs/research5'
root=base/'m9-runs/research_attempt5'
regpath=base/'m9-runs/research_attempt5_registration/registration.json'
def read(p):return json.loads(p.read_text()) if p.exists() else None
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
 return h.hexdigest()
def command(args):
 p=subprocess.run(args,capture_output=True,text=True,timeout=15)
 return dict(exit_code=p.returncode,stdout=p.stdout.strip(),stderr=p.stderr.strip())
reg=read(regpath);launch=read(job/'launch.json')
assert sha(regpath)==launch['registration_file_sha256']
sources={str(p):sha(project/p)==digest for p,digest in reg['source_hashes'].items()}
assert all(sources.values())
snapshot=dict(registration_sha256=reg['registration_sha256'],registration_file_sha256=sha(regpath),
 source_files_verified=len(sources),input_files_rehashed_this_check=False,registration_config=reg['config'],
 launch=launch,manager=read(job/'manager.json'),state=read(job/'status.json'),stages=read(job/'stages.json'),
 finished=read(job/'finished.json'),ranking=read(root/'explore_ranking.json'),selection=read(root/'selection.json'),
 prior_snapshot_path=str(base/'m9-inspections/20261113_1052/snapshot.json'),
 prior_snapshot_sha256=sha(base/'m9-inspections/20261113_1052/snapshot.json'),runs=[])
rank_rows=[]
for phase in ['explore','confirm']:
 for folder in sorted((root/phase).glob('*')):
  if not folder.is_dir():continue
  metrics=[]
  reuse=read(folder/'reuse.json')
  source=Path(reuse['source']) if reuse and reuse.get('kind')=='complete' else folder
  p=source/'metrics.jsonl'
  if p.exists():
   for line in p.read_text().splitlines():
    try:metrics.append(json.loads(line))
    except ValueError:pass
  row=dict(name=folder.name,phase=phase,path=str(folder),source=str(source),reuse=reuse,metrics=metrics,
   updates=len(metrics),decisions=metrics[-1]['decisions'] if metrics else 0,ticks=metrics[-1]['ticks'] if metrics else 0,
   status=read(folder/'status.json'),training_complete=read(folder/'training_complete.json'),evaluations={})
  for name in ['initial','evaluation','test_candidate','test_parent']:
   report=read(folder/(name+'.json'))
   if report:
    assert summarize(report['episodes'])==report['summary']
    trace=folder/(name+'.jsonl.gz')
    if not trace.exists():
     reuse=read(folder/'reuse.json');trace=Path(reuse['source'])/(name+'.jsonl.gz')
    assert sha(trace)==report['trace_sha256']
    row['evaluations'][name]=dict(report=report,summary_verified=True,trace_hash_verified=True,trace_path=str(trace))
  if phase=='explore':
   rank_rows.append(dict(arm=folder.name.rsplit('_r',1)[0],replicate=int(folder.name.rsplit('_r',1)[1]),summary=row['evaluations']['evaluation']['report']['summary']))
  if phase=='confirm':
   row['checkpoints']=[]
   for cp in sorted(folder.glob('checkpoint_*')):
    manifest=read(cp/'manifest.json')
    if not manifest:continue
    checks={name:sha(cp/name)==digest for name,digest in manifest['files'].items()}
    assert all(checks.values())
    row['checkpoints'].append(dict(path=str(cp),manifest=manifest,files_verified=True,
     reload=read(cp/'independent_reload.json'),reload_probe=read(cp/'reload_probe.json')))
   row['audit']=read(folder/'audit.json')
   row['test_audit']=read(folder/'test_audit.json')
   row['numerical_gate']=read(folder/'numerical_gate.json')
   if 'test_candidate' in row['evaluations'] and 'test_parent' in row['evaluations']:
    recomputed=compare(row['evaluations']['test_candidate']['report']['episodes'],row['evaluations']['test_parent']['report']['episodes'])
    assert recomputed==row['numerical_gate']
    row['numerical_gate_recomputed']=True
   log=job/('confirm_'+folder.name+'.log')
   events=[]
   if log.exists():
    for line in log.read_text().splitlines():
     try:e=json.loads(line)
     except ValueError:continue
     if e.get('event')=='evaluation_episode' and e.get('evaluation')=='evaluation':events.append(e)
   row['evaluation_log_events']=events
   trace=folder/'evaluation.jsonl.gz'
   if trace.exists() and 'evaluation' not in row['evaluations']:
    captured=datetime.now(timezone.utc)
    raw=trace.read_bytes()
    endpoints={};last={};ten={};decisions=Counter();count=0;truncated=False
    actor_hashes=set()
    try:
     with gzip.GzipFile(fileobj=io.BytesIO(raw)) as f:
      for line in f:
       if not line.endswith(b'\n'):truncated=True;break
       x=json.loads(line);seed=x['seed'];count+=1;decisions[seed]+=1
       actor_hashes.add(x['actor_sha256'])
       final=transition_at_horizon(x['transition'],5400)['after']
       last[seed]=final
       for t in x['transition']['trace']:
        if t['tick']==1800:ten[seed]=not t['dead']
       if final['dead'] or final['tick']==5400:
        endpoints[seed]=dict(seed=seed,replicate=0,survival_ticks=final['tick'],terminal=final['dead'],
         alive_at_horizon=not final['dead'],alive_at_1800=ten.get(seed,False),decisions=decisions[seed],final=final)
    except EOFError:truncated=True
    for e in events:
     if e['seed'] in endpoints:
      assert e['ticks']==endpoints[e['seed']]['survival_ticks'] and e['alive']==endpoints[e['seed']]['alive_at_horizon']
    if row['checkpoints']:assert actor_hashes=={row['checkpoints'][-1]['manifest']['actor_sha256']}
    eps=sorted(endpoints.values(),key=lambda e:e['seed'])
    row['partial_evaluation']=dict(captured_utc=captured.isoformat(),compressed_prefix_bytes=len(raw),
     compressed_prefix_sha256=hashlib.sha256(raw).hexdigest(),complete_records=count,
     stream_incomplete=truncated,episodes=eps,completed_subset_summary=summarize(eps) if eps else None,
     started_worlds=len(last),active={s:v for s,v in last.items() if s not in endpoints},
     counts_by_seed=dict(decisions),actor_hashes=sorted(actor_hashes),
     evaluation_started_utc=datetime.fromtimestamp((folder/'checkpoint_0064').stat().st_mtime,timezone.utc).isoformat(),
     max_possible_30day_successes=24-sum(e['terminal'] for e in eps),final_report_complete=False)
  snapshot['runs'].append(row)
assert select_arm(rank_rows,list(reg['config']['arms']))==snapshot['ranking']['ranking']
snapshot['ranking_verified']=True
snapshot['final_test_files']=[str(p) for folder in (root/'confirm').glob('*') for p in folder.glob('*test*')]
snapshot['ps']=command(['ps','-eo','pid,ppid,pgid,etimes,pcpu,stat,args'])
snapshot['ps']['stdout']='\n'.join(line for line in snapshot['ps']['stdout'].splitlines() if 'm9-research-v5-project' in line or 'research_attempt5' in line or line.lstrip().startswith('PID'))
snapshot['gpu']=command(['nvidia-smi','--query-gpu=name,utilization.gpu,memory.used,memory.total','--format=csv,noheader'])
snapshot['compute']=command(['nvidia-smi','--query-compute-apps=pid,process_name,used_gpu_memory','--format=csv,noheader'])
snapshot['checked_utc']=datetime.now(timezone.utc).isoformat()
now=datetime.fromisoformat(snapshot['checked_utc'])
snapshot['elapsed_seconds']=(now-datetime.fromisoformat(launch['created_utc'])).total_seconds()
snapshot['hard_wall_remaining_seconds']=max(0,reg['config']['hard_wall_seconds']-snapshot['elapsed_seconds'])
stamp=now.astimezone(timezone(timedelta(hours=8))).strftime('%Y%m%d_%H%M%S')
dest=base/'m9-inspections'/stamp
dest.mkdir(parents=True,exist_ok=False)
target=dest/'snapshot.json';target.write_text(json.dumps(snapshot,ensure_ascii=False,indent=2))
print(json.dumps(dict(snapshot=str(target),sha256=sha(target),checked_utc=snapshot['checked_utc'],
 total_updates=sum(r['updates'] for r in snapshot['runs']),total_decisions=sum(r['decisions'] for r in snapshot['runs']),
 source_files_verified=len(sources),confirmations=[dict(name=r['name'],updates=r['updates'],
 complete='evaluation' in r['evaluations'],partial_summary=r.get('partial_evaluation',{}).get('completed_subset_summary'),
 max_possible=r.get('partial_evaluation',{}).get('max_possible_30day_successes')) for r in snapshot['runs'] if r['phase']=='confirm'],
 gpu=snapshot['gpu'],compute=snapshot['compute'],finished=snapshot['finished'])))

