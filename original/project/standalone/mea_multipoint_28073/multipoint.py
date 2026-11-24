"""User-directed 20-point finite-train capability test, never an efficacy trial."""
from pathlib import Path
import argparse,hashlib,json,math,os,signal,subprocess,sys,time
V2=Path(os.environ.get('MEA_V2_ROOT',Path(__file__).resolve().parent.parent/'mea_stimulation_v2'))
sys.path.insert(0,str(V2))
from mea_stim_v2.config import Config,digest
from mea_stim_v2.protocol import Budget,RealClock,BudgetError
from mea_stim_v2.journal import atomic_json,exclusive_json

class PhysicalBinding(Config):
 def binding_sha256(self):
  d=self.to_dict();d.pop('authorization_sha256');d.pop('live_authorized')
  return digest({'protocol':'capability-20point-15pulse-10ms-200ms-300max-R2off',**d})

def binding(plan,point):
 d=dict(plan['base_config'],stim_electrode=point['electrode'],authorization_sha256=None)
 c=PhysicalBinding.from_dict(d);d['authorization_sha256']=c.binding_sha256()
 return PhysicalBinding.from_dict(d)

def events_for(point,electrode,session):
 return [dict(id=500000+(point-1)*100+g*3+j+1,arm='true',pair=point,phase='multipoint',point=point,
   electrode=electrode,group=g+1,pulse=j+1,session=session) for g in range(5) for j in range(3)]

class MultiLedger:
 def __init__(self,path):self.path=Path(path);self.claims=self.path/'claims'
 def total(self):return sum(json.loads(p.read_text())['arm']=='true' for p in self.claims.glob('*.json'))
 def reserve(self,events):
  if len(events)!=15 or self.total()+15>300 or not 1<=events[0]['point']<=20:raise ValueError('Cumulative 300-pulse / 20-point cap')
  if len({e['id'] for e in events})!=15 or any(e['arm']!='true' for e in events):raise ValueError('Unique true train required')
  for e in events:exclusive_json(self.claims/(str(e['id'])+'.json'),{'arm':'true','event_id':e['id'],'point':e['point'],'event':e,'reserved_wall':time.time()})

class Stop(RuntimeError):pass

def qc_child(request):
 from mea_stim_v2.recording import H5Recording
 from mea_stim_v2 import qc
 c=PhysicalBinding.from_dict(request['config']);path=request['path'];kind=request['kind']
 if kind=='train':
  from multipoint_qc import analyze
  result=analyze(path,c,request['roi'],request['events'],request['reference_rate'],request['post'])
 else:
  with H5Recording(path,c) as r:
   result=qc._common(r,c,kind)
   if kind=='preflight':
    if r.duration<5 or len(r.events)!=1 or r.events[0]['id']!=request['marker_id'] or r.events[0]['phase']!='routing_check':result['errors'].append('preflight_event_or_coverage')
    if not result['errors']:
     st=qc._stats(r.sample(request['roi'],2,1))
     if int(((~st['rails'])&(st['mad']>0)).sum())*100<95*len(request['roi']):result['errors'].append('preflight_raw_failed')
   elif kind=='recovery':
    if r.duration<600 or r.events:result['errors'].append('recovery_coverage_or_events')
   else:raise ValueError('Unknown QC kind')
   result=qc._finish(result)
 result.update(request_sha256=digest(request),capability_only=True,efficacy_test=False,r2_enabled=False)
 return result

class Runner:
 def __init__(self,plan,output):
  self.plan=plan;self.out=Path(output).resolve();self.clock=RealClock();self.dev=None;self.active=False;self.last_output=None;self.covered=False;self.recovering=False
  self.base=PhysicalBinding.from_dict(plan['base_config']);self.budget=Budget(self.base.started_wall(),self.clock)
  self.api_spent=0.
  self.state=dict(status='initializing',capability_only=True,efficacy_test=False,session_id=self.base.session_id,r2_enabled=False,training_enabled=False,
    records=[],points=[],errors=[],cleanup_errors=[],selected=plan['points'],started_wall=self.clock.wall(),hard_deadline_wall=self.budget.wall_end)
 def guard(self):
  self.budget.require(120)
  if not self.recovering and (self.out/'STOP').exists():raise Stop('User STOP')
 def wait(self,seconds):
  end=self.clock.mono()+seconds
  while self.clock.mono()<end:self.guard();self.clock.sleep(min(.25,end-self.clock.mono()))
  self.guard()
 def save(self):
  self.state.update(reserved_true_pulses=self.ledger.total(),updated_wall=self.clock.wall());atomic_json(self.out/'state.json',self.state)
 def operate(self,method,*args):
  self.guard();value=getattr(self.dev,method)(*args);self.api_spent+=float(getattr(self.dev,'last_api_seconds',0))
  if self.api_spent>120:raise BudgetError('Per-point API reserve exhausted')
  return value
 def close(self):
  r=self.operate('close_record');self.active=False;self.state['records'].append(r);self.save();return r['path']
 def analyze(self,kind,path,**kw):
  self.guard();self.budget.require(30+120)
  request=dict(kind=kind,path=str(Path(path).resolve()),config=self.cfg.to_dict(),roi=self.plan['roi'],**kw)
  tag=self.state['stage']+'_'+kind;req=self.out/(tag+'.request.json');dest=self.out/(tag+'.qc.json');exclusive_json(req,request)
  command=[sys.executable,str(Path(__file__).resolve()),'qc','--request',str(req),'--response',str(dest)]
  with (self.out/(tag+'.qc.log')).open('x') as log:
   proc=subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=log,stderr=log,close_fds=True,start_new_session=True)
   end=self.clock.mono()+30
   try:
    while proc.poll() is None:
     self.guard()
     if self.clock.mono()>=end:raise Stop('Closed-file QC exceeded 30 seconds')
     self.clock.sleep(.05)
    if proc.returncode:raise Stop('Closed-file QC child failed')
   finally:
    if proc.poll() is None:proc.kill();proc.wait(timeout=5)
  result=json.loads(dest.read_text())
  if result.get('request_sha256')!=digest(request) or result.get('source',{}).get('path')!=request['path']:raise Stop('QC response binding mismatch')
  self.state.setdefault('qc',{})[tag]=result;self.save()
  return result
 def require_pass(self,result):
  if result.get('status')!='pass' or result.get('errors'):raise Stop('QC failed: '+str(result.get('errors')))
 def pre_send(self,event):
  self.guard()
  if self.api_spent+float(getattr(self.dev,'last_api_seconds',0))>120:raise BudgetError('API reserve exhausted before train send')
  self.budget.before_output(self.required)
 def cleanup(self,final=False):
  pending=[];signals=(signal.SIGALRM,signal.SIGTERM,signal.SIGINT);old={s:signal.getsignal(s) for s in signals}
  for sig in signals:signal.signal(sig,signal.SIG_IGN if final else lambda sig,frame:pending.append(sig))
  if final:signal.setitimer(signal.ITIMER_REAL,0)
  try:
   if self.dev is not None:
    errors=self.dev.cleanup();self.state['cleanup_errors'].extend(errors);self.dev=None;self.active=False
    if errors and not final:raise RuntimeError('Point cleanup errors: '+repr(errors))
  finally:
   if not final:
    for sig,handler in old.items():signal.signal(sig,handler)
  if pending:raise KeyboardInterrupt('Signal deferred until point cleanup completed: '+str(pending))
 def recovery(self):
  if self.last_output is None or self.covered or self.dev is None:return
  self.budget.require(871);self.recovering=True
  if self.active:self.close()
  self.state['stage']='recovery';self.save();self.operate('open_record','recovery');self.active=True;self.wait(601)
  q=self.analyze('recovery',self.close());self.require_pass(q);self.covered=True
 def register(self):
  p=self.plan
  if p['schema']!='mea.28073.multipoint.capability.v1' or p['r2_enabled'] is not False or p['training_enabled'] is not False:raise ValueError('Unregistered plan')
  if len(p['points'])!=20 or len({r['electrode'] for r in p['points']})!=20 or len({r['channel'] for r in p['points']})!=20:raise ValueError('20 unique points required')
  if p['waveform']!={'target_mv':300,'phase_samples':4,'pulses_per_group':3,'groups':5,'pulse_period_samples':200,'group_period_samples':4000,'sham':False}:raise ValueError('Waveform changed')
  expected=dict(self.base.validate_binding());roi=p['roi'];stim={r['channel'] for r in p['points']}
  if not math.isfinite(p['reference_rate']) or p['reference_rate']<=0:raise ValueError('Positive finite observed reference required')
  if len(roi)<50 or roi!=sorted(set(roi)) or not set(roi)<=set(expected) or set(roi)&stim:raise ValueError('Invalid fixed readout ROI')
  if any(expected.get(r['channel'])!=r['electrode'] for r in p['points']):raise ValueError('Point mapping differs')
  old=Path(p['previous_run']);s=json.loads((old/'state.json').read_text())
  if s['session_id']!=self.base.session_id or s['dose_totals']!={'true':0,'sham':0} or s['cleanup_errors'] or s['status']!='failed':raise ValueError('Prior run was not stopped/cleaned with zero dose')
  for line in (old/'api_audit.jsonl').read_text().splitlines():
   for cmd in json.loads(line)['command'].splitlines():
    if cmd.startswith('mea_set_dac ') and cmd!='mea_set_dac 0 512 512':raise ValueError('Prior nonzero DAC attempt')
  registry=Path('/home/researcher/.local/share/mea_stim_v2/launch_registry')/hashlib.sha256(self.base.organoid_id.encode()).hexdigest()
  launch=json.loads((registry/'launch.json').read_text());original=launch['config']
  if any(original[k]!=self.base.to_dict()[k] for k in ('session_id','organoid_id','chip_id','started_utc','route_file','route_sha256')):raise ValueError('Biological clock/identity cannot change')
  self.ledger=MultiLedger(registry)
  if self.ledger.total()!=0:raise ValueError('New 300-pulse plan requires zero prior pulses in this session')
  self.out.mkdir(exist_ok=False)
  registration=dict(plan=p,plan_sha256=digest(p),prior_state_sha256=hashlib.sha256((old/'state.json').read_bytes()).hexdigest(),run=str(self.out),hard_deadline_wall=self.budget.wall_end)
  exclusive_json(registry/'multipoint_authorization.json',registration);exclusive_json(self.out/'registration.json',registration)
 def run(self):
  self.register()
  def interrupted(sig,frame):raise KeyboardInterrupt('User/deadline signal '+str(sig))
  for sig in (signal.SIGALRM,signal.SIGTERM,signal.SIGINT):signal.signal(sig,interrupted)
  signal.setitimer(signal.ITIMER_REAL,max(.001,self.budget.remaining()-120))
  try:
   from multipoint_backend import MultipointBackend
   self.state['status']='running'
   for index,point in enumerate(self.plan['points'],1):
    post=600 if index==20 else 60
    try:self.budget.require(40+1+post+1+30+631+240)
    except BudgetError as error:raise Stop('No reserve for next complete point and recovery') from error
    self.cleanup();self.guard();self.api_spent=0.;self.cfg=binding(self.plan,point);self.state['stage']='point_%02d'%index;self.save()
    folder=self.out/self.state['stage'];folder.mkdir();self.dev=MultipointBackend(self.cfg,folder,self.clock)
    pre=self.operate('preflight');self.operate('prepare');self.operate('connect',True)
    self.operate('open_record','routing');self.active=True;self.operate('marker',30000+index,dict(phase='routing_check',point=index,electrode=point['electrode']));self.wait(5.1)
    self.require_pass(self.analyze('preflight',self.close(),marker_id=30000+index))
    self.operate('open_record','train');self.active=True;self.wait(11)
    events=events_for(index,point['electrode'],self.base.session_id)
    self.required=1+post+1+30+631+240;self.pre_send(events[0]);self.ledger.reserve(events);self.save()
    self.dev.before_sequence_send=self.pre_send;receipt=self.operate('train',events,pre['lsb_mv']);self.last_output=receipt['send_mono'];self.covered=False
    self.state['points'].append({'point':index,**point,'train_receipt':receipt,'status':'sent_pending_qc'});self.save();self.wait(post+1)
    q=self.analyze('train',self.close(),events=events,reference_rate=self.plan['reference_rate'],post=post)
    if post==600 and q.get('metadata',{}).get('frame_gaps')==0 and len(q.get('events',[]))==15:
     self.covered=q['metadata']['duration_seconds']-q['events'][-1]['time_s']-.0004>=600
    self.state['points'][-1].update(status=q.get('status'),qc_path=self.state['stage']+'_train.qc.json');self.save();self.require_pass(q)
   self.state['status']='completed'
  except Stop as error:
   self.state['status']='stopped';self.state['errors'].append(str(error))
   try:self.recovery()
   except BaseException as e:self.state['errors'].append('Recovery incomplete: '+repr(e))
  except BaseException as error:self.state['status']='failed';self.state['errors'].append(repr(error))
  finally:
   try:self.cleanup(final=True)
   except BaseException as error:self.state['cleanup_errors'].append(repr(error))
   if self.state['cleanup_errors']:self.state['status']='failed'
   self.state['finished_wall']=self.clock.wall();self.save()
  return self.state

def main():
 p=argparse.ArgumentParser();sub=p.add_subparsers(dest='command',required=True)
 q=sub.add_parser('qc');q.add_argument('--request',required=True);q.add_argument('--response',required=True)
 r=sub.add_parser('run');r.add_argument('--plan',required=True);r.add_argument('--output',required=True)
 a=p.parse_args()
 if a.command=='qc':
  request=json.loads(Path(a.request).read_text());result=qc_child(request);exclusive_json(a.response,result);return
 result=Runner(json.loads(Path(a.plan).read_text()),a.output).run();print(json.dumps({k:result[k] for k in ('status','reserved_true_pulses','errors','cleanup_errors')}),flush=True)
 if result['status']!='completed':raise SystemExit(2)

if __name__=='__main__':main()
