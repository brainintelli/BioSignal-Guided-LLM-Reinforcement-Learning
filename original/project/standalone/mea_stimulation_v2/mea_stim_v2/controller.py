"""Single-owner protocol controller. All admission decisions precede output."""
from pathlib import Path
import math
from .config import digest
from .journal import Ledger
from .protocol import Budget, BudgetError, schedule
from .worker import WorkerError

class SoftStop(RuntimeError): pass

class Controller:
 def __init__(self,config,run_dir,registry,backend,analyzer,clock,stop_requested=None,resume_zero_preflight=None):
  self.c=config;self.out=Path(run_dir);self.registry=Path(registry);self.dev=backend
  self.analyze=analyzer;self.clock=clock;self.stop_requested=stop_requested or (lambda: (self.out/'STOP').exists())
  self.active=False;self.last_true=None;self.last_true_id=None;self.post_complete=False;self.api_spent=0.;self.target=None;self.required=0
  self.before_cleanup=lambda:None
  self.in_recovery=False
  self.resume_zero_preflight=resume_zero_preflight
  self.state={'status':'initializing','session_id':config.session_id,'biological_specimen_id':None,
   'r2_enabled':False,'training_enabled':False,'accepted_pairs':[],'records':[],'qc':{},'errors':[],'cleanup_errors':[]}
 def guard(self,manual=True):
  self.budget.require(120)
  if manual and not self.in_recovery and self.stop_requested():raise SoftStop('Manual STOP requested')
 def wait(self,seconds,manual=True):
  end=self.clock.mono()+seconds
  while self.clock.mono()<end:
   self.guard(manual);self.clock.sleep(min(.25,end-self.clock.mono()))
 def call(self,method,*args):
  self.guard();value=getattr(self.dev,method)(*args)
  self.api_spent+=float(getattr(self.dev,'last_api_seconds',0))
  if self.api_spent>120:raise BudgetError('API reserve exhausted')
  return value
 def save(self):
  self.state.update(dose_totals=self.log.totals(),remaining_seconds=self.budget.remaining(),updated_wall=self.clock.wall())
  self.log.state(self.state)
 def open(self,name):
  self.state['stage']=name;self.save();self.call('open_record',name);self.active=True
 def close(self):
  value=self.call('close_record');self.active=False;self.state['records'].append(value);self.save();return value['path']
 def qc(self,stage,path,**kw):
  self.guard()
  try:r=self.analyze(stage,path,**kw)
  except WorkerError as error:raise SoftStop('QC worker failed: '+stage+': '+str(error)) from error
  self.state['qc'][stage]=r;self.save()
  if r.get('session_id')!=self.c.session_id or Path(r.get('source',{}).get('path','')).resolve()!=Path(path).resolve():
   raise SoftStop('QC receipt binding mismatch: '+stage)
  if stage=='pilot' or stage=='pair_10':
   m=r.get('metadata',{});ev=r.get('events',[])
   if (m.get('chip_id')==self.c.chip_id and m.get('frame_gaps')==0 and len(ev)==2 and
       any(e.get('id')==self.last_true_id and e.get('arm')=='true' for e in ev) and
       all(e.get('session')==self.c.session_id for e in ev) and
       m.get('duration_seconds',0)-max(e.get('time_s',float('inf')) for e in ev)>=600):self.post_complete=True
  if r.get('status')!='pass' or r.get('errors'):raise SoftStop('QC failed: '+stage+': '+str(r.get('errors')))
  return r
 def before_send(self,event):
  self.guard()
  if self.api_spent+float(getattr(self.dev,'last_api_seconds',0))>120:raise BudgetError('API reserve exhausted before send')
  if self.target is not None:
   if self.clock.mono()>self.target+.09:raise SoftStop('Second event missed its interval; no delayed pulse')
   self.wait(max(0,self.target-self.clock.mono()))
  self.guard()
  self.budget.before_output(self.required)
 def pulse(self,event,remaining_work,target=None):
  self.required=remaining_work;self.target=target
  self.before_send(event) if target is None else self.guard()
  self.log.reserve(event['id'],event['arm'],self.clock.wall())
  # The live adapter invokes this after building, immediately before sequence_send.
  self.dev.before_sequence_send=self.before_send
  if target is not None and not hasattr(self.dev,'sent_monotonic_actual'):
   self.before_send(event) # deterministic non-device test adapter
  sent=self.clock.mono();reply=self.call('pulse',event,self.lsb)
  if isinstance(reply,dict):sent=reply['send_mono']
  if event['arm']=='true':self.last_true=sent;self.last_true_id=event['id'];self.post_complete=False
  self.log.log('pulse_acknowledged',event=event,send_monotonic=sent,wall=self.clock.wall());self.save()
  self.target=None;return sent
 def disk(self):
  self.guard()
  records=self.state['records']
  if records:
   path=Path(records[-1]['path']);duration=records[-1].get('recorded_seconds')
   qc=self.state['qc'].get(self.state['stage'],{})
   duration=duration or qc.get('metadata',{}).get('duration_seconds')
   if duration and self.dev.free_bytes()<path.stat().st_size/float(duration)*self.budget.remaining()*1.3:
    raise SoftStop('Insufficient disk for remaining raw data and reserve')
 def recovery(self):
  if self.last_true is None or self.post_complete:return
  self.budget.require(601+30+120+120);self.in_recovery=True
  if self.active:
   # Closing is required before a distinct recovery file; report the real gap.
   value=self.dev.close_record();self.active=False;self.state['records'].append(value)
  gap=self.clock.mono()-self.last_true
  self.dev.open_record('recovery');self.active=True
  self.state['recovery']={'fresh_recording':True,'seconds_since_last_true_at_open':gap,'continuous_from_stimulus':False}
  self.wait(601,manual=False)
  value=self.dev.close_record();self.active=False;self.state['records'].append(value)
  self.qc('recovery',value['path']);self.post_complete=True;self.in_recovery=False
 def run(self):
  self.c.validate_binding();self.budget=Budget(self.c.started_wall(),self.clock)
  specs=schedule(self.c.seed)
  self.log=Ledger(self.registry,self.c.organoid_id,{'config':self.c.to_dict(),'binding_sha256':self.c.binding_sha256(),
    'schedule':specs,'schedule_sha256':digest(specs),'hard_deadline_wall':self.budget.wall_end},self.out,resume_zero_preflight=self.resume_zero_preflight)
  try:
   self.save();p=self.call('preflight');self.lsb=p['lsb_mv']
   self.budget.require(1500+120);self.call('prepare')
   self.open('preflight');self.call('marker',7001,{'phase':'preflight'});self.wait(11)
   self.qc('preflight',self.close());self.api_spent=0
   self.open('baseline');self.call('marker',8001,{'phase':'disconnected'});self.wait(300.01)
   self.call('connect',True);self.call('marker',8002,{'phase':'connected'});self.wait(1201)
   baseline_path=self.close();base=self.qc('baseline',baseline_path)
   candidate=base['candidate_roi']
   if len(candidate)<50 or len(set(candidate))!=len(candidate):raise SoftStop('Invalid candidate ROI')
   self.disk();self.api_spent=0;self.budget.require(120+601+300+240)
   self.open('pilot');self.wait(61)
   first=self.pulse({'id':8201,'arm':'sham','pair':0,'phase':'pilot'},60+601+300+631+240)
   self.wait(max(0,first+59-self.clock.mono()))
   self.pulse({'id':8202,'arm':'true','pair':0,'phase':'pilot'},601+300+631+240,target=first+60)
   self.wait(601)
   pil=self.qc('pilot',self.close(),baseline_path=baseline_path,candidate_roi=candidate)
   roi=pil['roi'];reference=float(pil['reference_rate'])
   if len(roi)<50 or not set(roi)<=set(candidate) or len(set(roi))!=len(roi) or not math.isfinite(reference) or reference<=0:
    raise SoftStop('Invalid frozen pilot ROI/reference')
   self.state['frozen_roi']=roi;self.state['frozen_roi_sha256']=digest(roi);self.state['reference_rate']=reference
   for spec in specs:
    self.api_spent=0;self.budget.admit_pair();self.disk();name='pair_%02d'%spec['pair'];post=600 if spec['pair']==10 else 60
    self.open(name);self.wait(31)
    extra=631
    required=spec['interval_s']+post+1+30+extra+240
    first=self.pulse({**spec['events'][0],'phase':'formal'},required)
    self.wait(max(0,first+spec['interval_s']-1-self.clock.mono()))
    self.pulse({**spec['events'][1],'phase':'formal'},post+1+30+extra+240,target=first+spec['interval_s'])
    self.wait(post+1)
    result=self.qc(name,self.close(),pair_spec=spec,roi=roi,reference_rate=reference,expected_post=post)
    self.state['accepted_pairs'].append(result);self.save()
   self.state['status']='completed'
  except SoftStop as e:
   self.state['status']='stopped';self.state['errors'].append(str(e))
   try:self.recovery()
   except BaseException as e:self.state['errors'].append('Recovery incomplete: '+repr(e))
  except BaseException as e:
   self.state['status']='failed';self.state['errors'].append(repr(e))
  finally:
   # Never let reporting failures skip independent device cleanup.
   try:self.before_cleanup()
   except BaseException as e:self.state['errors'].append('Signal barrier: '+repr(e))
   try:self.state['cleanup_errors']=self.dev.cleanup()
   except BaseException as e:self.state['cleanup_errors']=[repr(e)]
   if self.state['cleanup_errors']:self.state['status']='failed'
   self.state['finished_wall']=self.clock.wall();self.save()
  return self.state
