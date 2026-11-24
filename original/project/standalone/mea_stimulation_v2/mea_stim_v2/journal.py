"""Exclusive biological launch and append-only attempts, durably before output."""
from pathlib import Path
import hashlib
import json
import os

def atomic_json(path,value):
 path=Path(path);tmp=path.with_name(path.name+'.tmp')
 with tmp.open('w',encoding='utf-8') as f:
  json.dump(value,f,indent=2,allow_nan=False);f.flush();os.fsync(f.fileno())
 os.replace(tmp,path)

def exclusive_json(path,value):
 with Path(path).open('x',encoding='utf-8') as f:
  json.dump(value,f,indent=2,allow_nan=False);f.flush();os.fsync(f.fileno())

class Ledger:
 def __init__(self,registry,organoid_id,registration,run_dir,resume_zero_preflight=None):
  self.root=Path(registry);self.root.mkdir(parents=True,exist_ok=True)
  key=hashlib.sha256(organoid_id.encode()).hexdigest()
  self.path=self.root/key
  retry=None
  if resume_zero_preflight is None:
   self.path.mkdir(exist_ok=False) # same organoid cannot restart by changing output/start
  else:
   previous=Path(resume_zero_preflight).resolve()
   launch=json.loads((self.path/'launch.json').read_text(encoding='utf-8'))
   state=json.loads((previous/'state.json').read_text(encoding='utf-8'))
   audit=(previous/'api_audit.jsonl').read_bytes()
   rows=[json.loads(line) for line in audit.decode().splitlines()]
   if (Path(launch['run_dir']).resolve()!=previous or launch['organoid_id']!=organoid_id or
       launch.get('binding_sha256')!=registration.get('binding_sha256') or launch.get('config')!=registration.get('config') or
       state.get('status')!='failed' or state.get('stage')!='preflight' or state.get('cleanup_errors')!=[] or
       state.get('dose_totals')!={'true':0,'sham':0} or list((self.path/'claims').glob('*.json')) or not rows):
    raise ValueError('Only a bound, cleaned, zero-dose preflight failure may continue without resetting its registry')
   for row in rows:
    command=row['command']
    if command.startswith('sequence_send ') or any(line.startswith('mea_set_dac ') and line!='mea_set_dac 0 512 512' for line in command.splitlines()):
     raise ValueError('Preflight retry cannot follow a send or nonzero queued DAC')
   retry={'previous_run':str(previous),'audit_sha256':hashlib.sha256(audit).hexdigest(),'new_run':str(Path(run_dir).resolve()),
          'same_binding_sha256':registration['binding_sha256'],'start_and_doses_not_reset':True}
   exclusive_json(self.path/'zero_preflight_continuation.json',retry)
  self.run=Path(run_dir);self.run.mkdir(parents=True,exist_ok=False)
  if retry is None:exclusive_json(self.path/'launch.json',{'organoid_id':organoid_id,'run_dir':str(self.run.resolve()),**registration})
  else:exclusive_json(self.run/'continuation.json',retry)
  exclusive_json(self.run/'registration.json',registration)
  self.claims=self.path/'claims'
  if retry is None:self.claims.mkdir()
  self.audit=self.run/'audit.jsonl'
 def totals(self):
  result={'true':0,'sham':0}
  for p in self.claims.glob('*.json'):
   row=json.loads(p.read_text(encoding='utf-8'));result[row['arm']]+=1
  return result
 def reserve(self,event_id,arm,wall):
  if type(event_id) is not int or not 1<=event_id<2**20 or arm not in ('true','sham'):raise ValueError('Invalid attempt')
  if self.totals()[arm]>=11:raise RuntimeError('Cumulative 11/11 dose cap')
  exclusive_json(self.claims/(str(event_id)+'.json'),{'event_id':event_id,'arm':arm,'attempt_wall':wall})
  self.log('dose_reserved',event_id=event_id,arm=arm,wall=wall)
 def log(self,kind,**values):
  with self.audit.open('a',encoding='utf-8') as f:
   f.write(json.dumps({'kind':kind,**values},allow_nan=False)+'\n');f.flush();os.fsync(f.fileno())
 def state(self,value):atomic_json(self.run/'state.json',value)
