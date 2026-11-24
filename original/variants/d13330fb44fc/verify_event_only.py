"""10 s event-recording check. No DAC, Core, Array, initialize or stimulus-unit commands."""
from pathlib import Path
import json,os,signal,sys,time,datetime
BASE=Path(__file__).resolve().parent
sys.path.insert(0,str(BASE/'BrainStimulation'))
from brain_stimulation import maxlab as mx
from brain_stimulation.transport import Api,Endpoint,bind_sdk
from brain_stimulation.runner import ManagedSequence

ROOT=BASE/'event_only_verification_v2'
TOKEN='event_only_20261117_22554_v2'
allowed={'system_clear_events','sequence_new '+TOKEN,
 'sequence_append '+TOKEN+'\nsystem_delay_samples 100',
 'sequence_append '+TOKEN+'\nsystem_trigger_event 0 1 7001 mode event_only_no_DAC_output',
 'sequence_send '+TOKEN,'sequence_delete '+TOKEN,
 'saving_open_dir '+str(ROOT),'saving_use_old_file_format 0','saving_start_file event_only',
 'saving_group_clear','saving_group_define 0 routed '+','.join(map(str,range(1024))),
 'saving_set_recording_wells 0','saving_start_recording','saving_stop_recording','saving_stop_file'}
assert not any('mea_set_dac' in c or 'stimulation_unit' in c or 'system_initialize' in c for c in allowed)
ROOT.mkdir(exist_ok=False)
report={'mode':'event_markers_only_no_electrical_output_commands','status':'starting','commands':[],'cleanup_errors':[]}
def save(): (ROOT/'verification.json').write_text(json.dumps(report,indent=2))
original=Api.send
def guarded(api,message,timeout=None):
 if message not in allowed:raise RuntimeError('Event-only allowlist rejected '+message[:100])
 report['commands'].append({'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'message':message});save()
 return original(api,message,timeout)
Api.send=guarded
def stop(*args):raise KeyboardInterrupt('bounded event-only capture timeout')
for sig in (signal.SIGALRM,signal.SIGTERM,signal.SIGINT):signal.signal(sig,stop)
signal.alarm(60)
file_attempted=recording_attempted=sequence_attempted=False
s=mx.Saving()
try:
 if time.time()>datetime.datetime.fromisoformat('2026-11-17T05:27:00+00:00').timestamp():raise RuntimeError('Session deadline')
 opened=[os.readlink(fd) for fd in Path('/proc/2066401/fd').iterdir() if '.h5' in os.readlink(fd)]
 if opened:raise RuntimeError('Existing recording '+repr(opened))
 with bind_sdk(mx,Endpoint(timeout=5)):
  mx.clear_events()
  sequence_attempted=True
  seq=ManagedSequence(token=TOKEN,initial_delay=0,persistent=True)
  if seq.token!=TOKEN:raise RuntimeError('Unexpected sequence token')
  seq.append(mx.DelaySamples(100));seq.append(mx.Event(0,1,7001,'mode event_only_no_DAC_output'))
  s.open_directory(str(ROOT));s.set_legacy_format(False)
  file_attempted=True;s.start_file('event_only');s.group_delete_all();s.group_define(0,'routed',list(range(1024)))
  recording_attempted=True;s.start_recording([0]);time.sleep(3)
  seq.send();time.sleep(7)
  report['status']='captured'
except BaseException as exc:report.update(status='failed',error=repr(exc))
finally:
 signal.alarm(0)
 for sig in (signal.SIGTERM,signal.SIGINT):signal.signal(sig,signal.SIG_IGN)
 with bind_sdk(mx,Endpoint(timeout=2)):
  actions=[(recording_attempted,s.stop_recording),(file_attempted,s.stop_file),(file_attempted,s.group_delete_all),(sequence_attempted,lambda:mx.send_raw('sequence_delete '+TOKEN))]
  for attempted,fn in actions:
   if attempted:
    try:fn()
    except Exception as exc:report['cleanup_errors'].append(repr(exc))
 Api.send=original;save()
if report['status']=='captured' and not report['cleanup_errors']:
 import h5py
 with h5py.File(ROOT/'event_only.raw.h5','r') as f:
  ev=f['data_store/data0000/events'][:]
  report['events']=[{'frame':int(x['frameno']),'eventtype':int(x['eventtype']),'eventid':int(x['eventid']),'message':x['eventmessage'].decode()} for x in ev]
  report['chip_id']=f['wellplate/id'][0].decode()
  report['status']='passed' if len(ev)==1 and int(ev[0]['eventid'])==7001 and report['chip_id']=='22554' else 'failed'
 save()
print(json.dumps({k:v for k,v in report.items() if k!='commands'},indent=2))
raise SystemExit(0 if report['status']=='passed' else 1)
