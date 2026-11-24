"""One bounded same-configuration baseline -> gated pilot -> gated sham control."""
from pathlib import Path
from contextlib import contextmanager
import datetime as dt
import hashlib,json,os,re,signal,socket,sys,time,uuid
from protocol import Budget,SESSION_ID,make_control_schedule,sequence_commands,validate_gate

BASE=Path(__file__).resolve().parent
OLD=Path('/home/researcher/mea_first_probe_20261117')
DEADLINE=dt.datetime.fromisoformat('2026-11-17T05:29:00+00:00').timestamp()


class Command:
 def __init__(self,text):self.text=text
 def set(self):return self.text


def run():
 if socket.gethostname()!='researcher-biolab-1':raise RuntimeError('Wrong acquisition host')
 sys.path.insert(0,str(BASE/'BrainStimulation'))
 from brain_stimulation import maxlab as mx
 from brain_stimulation.runner import ManagedSequence
 from brain_stimulation.transport import Api,Endpoint,bind_sdk
 prior=json.loads((OLD/'first_probe_send_claim.json').read_text())
 if prior['reserved_pulses']!=5 or prior['deadline_wall']!=DEADLINE:raise RuntimeError('Prior session ledger mismatch')
 budget=Budget(BASE/'dose_claims',deadline_wall=DEADLINE,wall_now=time.time(),mono_now=time.monotonic(),prior_true=5)
 with (BASE/'launch.json').open('x') as f:
  json.dump({'session_id':SESSION_ID,'started_utc':dt.datetime.now(dt.timezone.utc).isoformat(),'deadline_wall':DEADLINE},f)
  f.flush();os.fsync(f.fileno())
 report={'session_id':SESSION_ID,'status':'initializing','r2_enabled':False,'training_enabled':False,
         'cleanup_errors':[],'stage_files':{},'stimulation_electrode':10900,'stage_history':[]}
 def state(status,**fields):
  report.update(status=status,updated_utc=dt.datetime.now(dt.timezone.utc).isoformat(),**fields)
  tmp=BASE/'state.json.tmp';tmp.write_text(json.dumps(report,indent=2));tmp.replace(BASE/'state.json')
  print(status,report['updated_utc'],flush=True)
 audit=(BASE/'api_audit.jsonl').open('a',buffering=1)
 original_send=Api.send
 def log(kind,message,**extra):
  try:audit.write(json.dumps({'utc':dt.datetime.now(dt.timezone.utc).isoformat(),'kind':kind,'command':message[:180],
          'command_sha256':hashlib.sha256(message.encode()).hexdigest(),**extra})+'\n');audit.flush()
  except Exception:report['audit_write_failed']=True
 def checked(api,message,timeout=None):
  # Direct DAC writes are zero only. Nonzero levels exist only in gated sequences.
  if message.startswith('mea_set_dac') and message!='mea_set_dac 0 512 512':raise RuntimeError('Direct nonzero DAC forbidden')
  log('request',message)
  try:reply=original_send(api,message,timeout)
  except BaseException as exc:log('error',message,error=repr(exc));raise
  log('ack',message,reply=reply[:150]);return reply
 Api.send=checked
 endpoint=Endpoint(timeout=20)
 sequences=[];array_token=None;touched=False;file_open=recording=False;saving=mx.Saving()
 def stop(signum,frame):raise KeyboardInterrupt('stop/deadline '+str(signum))
 for sig in (signal.SIGINT,signal.SIGTERM,signal.SIGALRM):signal.signal(sig,stop)
 signal.setitimer(signal.ITIMER_REAL,max(.01,budget.remaining(time.time(),time.monotonic())-120))
 def wait(seconds):
  if budget.remaining(time.time(),time.monotonic())<seconds+120:raise RuntimeError('Time reserve insufficient')
  end=time.monotonic()+seconds
  while True:
   left=end-time.monotonic()
   if left<=0:break
   time.sleep(min(1,left))
 def make_sequence(commands,label):
  token='qc_'+label+'_'+uuid.uuid4().hex;sequences.append(token)
  seq=ManagedSequence(token=token,initial_delay=0,persistent=True)
  if seq.token!=token:raise RuntimeError('Sequence token mismatch')
  for c in commands:seq.append(Command(c))
  return seq
 def open_file(stage):
  nonlocal file_open,recording
  folder=BASE/stage;folder.mkdir(exist_ok=False)
  saving.open_directory(str(folder));saving.set_legacy_format(False)
  file_open=True;saving.start_file(stage);saving.group_delete_all();saving.group_define(0,'routed',list(range(1024)))
  recording=True;saving.start_recording([0]);report['stage_files'][stage]=str(folder/(stage+'.raw.h5'))
  state('recording_'+stage)
 def close_file():
  nonlocal file_open,recording
  errors=[]
  for label,flag,fn in [('record',recording,saving.stop_recording),('file',file_open,saving.stop_file),('groups',file_open,saving.group_delete_all)]:
   if flag:
    try:fn()
    except Exception as exc:errors.append(label+':'+repr(exc))
  if errors:raise RuntimeError('File close failed '+repr(errors))
  recording=file_open=False
 def gate(stage,required_seconds,expected_roi=None):
  state('awaiting_'+stage+'_qc')
  limit=time.monotonic()+600
  while True:
   if budget.remaining(time.time(),time.monotonic())<required_seconds+140:raise RuntimeError('Not enough time for next stage')
   if time.monotonic()>limit:raise RuntimeError('QC gate timed out')
   receipt=BASE/('gate_'+stage+'.json')
   if receipt.exists():
    rec=json.loads(receipt.read_text());qpath=BASE/(stage+'_qc.json');data=qpath.read_bytes()
    if rec.get('qc_sha256')!=hashlib.sha256(data).hexdigest():raise RuntimeError('QC gate hash mismatch')
    q=json.loads(data);validate_gate(q,stage)
    if expected_roi is not None and sorted(q['roi_channels'])!=sorted(expected_roi):raise RuntimeError('ROI changed after pilot')
    return q
   fail=BASE/('stop_'+stage+'.json')
   if fail.exists():raise RuntimeError('QC requested stop: '+fail.read_text()[:500])
   time.sleep(1)
 def stim_stage(stage,schedule,pre,post):
  # Clear registry BEFORE any Event gets appended. No clearing until file closes.
  mx.clear_events();seq=make_sequence(sequence_commands(schedule,report['dac_lsb_mv']),stage)
  (BASE/(stage+'_schedule.json')).write_text(json.dumps(schedule,indent=2))
  open_file(stage);wait(pre)
  duration=(len(schedule)-1)*45+.0054
  budget.reserve(stage,sum(t['arm']=='true' for t in schedule),sum(t['arm']=='sham' for t in schedule),duration,time.time(),time.monotonic())
  state('sending_'+stage,cumulative_reserved=budget.totals())
  seq.send();wait(duration);state('recovering_'+stage);wait(post);close_file()
  report['stage_history'].append({'stage':stage,'recording_closed_utc':dt.datetime.now(dt.timezone.utc).isoformat()})
 try:
  with bind_sdk(mx,endpoint):
   opened=[];servers=[]
   for p in Path('/proc').iterdir():
    if not p.name.isdigit():continue
    try:
     if (p/'comm').read_text().strip()!='mxwserver':continue
     servers.append(int(p.name))
     for fd in (p/'fd').iterdir():
      target=os.readlink(fd)
      if '.h5' in target:opened.append(target)
    except FileNotFoundError:continue
   if len(servers)!=1 or opened:raise RuntimeError('Device server busy/ambiguous: '+repr(opened))
   report['server_pid']=servers[0];report['dac_lsb_mv']=float(mx.query_DAC_lsb_mV())
   sequence_commands(make_control_schedule(),report['dac_lsb_mv'])
   touched=True;mx.initialize();mx.send(mx.Core().enable_stimulation_power(True))
   for unit in range(32):mx.send(mx.StimulationUnit(unit).power_up(False).connect(False))
   array_token='qc_array_'+uuid.uuid4().hex;array=mx.Array(array_token,persistent=True)
   cfg=(BASE/'qualified_route.cfg').read_text();array.load_config(str(BASE/'qualified_route.cfg'))
   array.connect_electrode_to_stimulation(10900);unit_reply=array.query_stimulation_at_electrode(10900).strip()
   if not unit_reply.isdigit() or not 0<=int(unit_reply)<32:raise RuntimeError('No valid stimulus route')
   unit=int(unit_reply);report['stimulation_unit']=unit
   mx.activate([0]);array.download([0]);wait(5);mx.offset();mx.send(mx.DAC(0,512))
   routed=mx.send_raw('mea_array_command '+array_token+'\nmea_array_get_config')
   if sorted(re.findall(r'(\d+)\((\d+)\)',cfg.split()[0]))!=sorted(re.findall(r'(\d+)\((\d+)\)',routed.split()[0])):raise RuntimeError('Recording route changed')
   (BASE/'downloaded_route.cfg').write_text(routed)
   mx.clear_events()
   markers=[make_sequence(['system_delay_samples 100',f'system_trigger_event 0 1 {8001+i} phase {phase}'],'base_'+phase) for i,phase in enumerate(['disconnected','connected'])]
   open_file('baseline');markers[0].send();wait(120)
   mx.send(mx.StimulationUnit(unit).power_up(True).connect(True).set_voltage_mode().dac_source(0))
   markers[1].send();state('recording_baseline_connected');wait(120);close_file()
   q=gate('baseline',60+45.0054+600)
   roi=q['roi_channels'];report['roi_channels']=roi
   pilot=[{'pair':0,'arm':'sham','event_id':8201,'amplitude_mv':0},{'pair':0,'arm':'true','event_id':8202,'amplitude_mv':300}]
   stim_stage('pilot',pilot,60,600)
   gate('pilot',120+495.0054+600,roi)
   stim_stage('control',make_control_schedule(),120,600)
   state('completed',cumulative_reserved=budget.totals())
 except BaseException as exc:
  report['error']=repr(exc);state('stopping',error=repr(exc))
 finally:
  signal.setitimer(signal.ITIMER_REAL,0)
  for sig in (signal.SIGTERM,signal.SIGINT,signal.SIGALRM):signal.signal(sig,signal.SIG_IGN)
  def attempt(label,fn):
   try:fn()
   except Exception as exc:report['cleanup_errors'].append(label+':'+repr(exc))
  # Each call can spend timeout in connection + timeout in reply; <=1 s
  # gives at most ~82 s for all 41 cleanup requests, inside the 120 s reserve.
  with bind_sdk(mx,Endpoint(timeout=1)):
   if touched:
    for u in range(32):attempt('off '+str(u),lambda u=u:mx.send(mx.StimulationUnit(u).power_up(False).connect(False)))
    attempt('zero DAC',lambda:mx.send(mx.DAC(0,512)))
   if recording:attempt('stop recording',saving.stop_recording)
   if file_open:attempt('stop file',saving.stop_file);attempt('clear groups',saving.group_delete_all)
   for token in sequences:attempt('delete sequence',lambda token=token:mx.send_raw('sequence_delete '+token))
   if array_token:attempt('delete array',lambda:mx.send_raw('mea_array_delete '+array_token))
  Api.send=original_send
  try:audit.close()
  except Exception:report['audit_write_failed']=True
  final='completed' if report['status']=='completed' and not report['cleanup_errors'] and not report.get('audit_write_failed') else 'stopped'
  state(final,cumulative_reserved=budget.totals(),finished_utc=dt.datetime.now(dt.timezone.utc).isoformat())
 return 0 if report['status']=='completed' else 1


if __name__=='__main__':
 if sys.argv[1:]!=['--run']:raise SystemExit('Explicit --run required; biological deadline remains 13:29 CST')
 raise SystemExit(run())
