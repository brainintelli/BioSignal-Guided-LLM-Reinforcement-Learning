"""One saving-only intake. Only all-unit OFF and zero DAC are permitted."""
from pathlib import Path
import datetime as dt
import json,os,signal,socket,sys,time

def allowed_commands(mx,root):
 return {'mea_set_dac 0 512 512', 'saving_open_dir '+str(root),
  'saving_use_old_file_format 0','saving_start_file intake','saving_group_clear',
  'saving_group_define 0 routed '+','.join(map(str,range(1024))),
  'saving_set_recording_wells 0','saving_start_recording','saving_stop_recording','saving_stop_file',
  *(mx.StimulationUnit(u).power_up(False).connect(False).set() for u in range(32))}

def main():
 base=Path(__file__).resolve().parent;root=base/'intake'
 deadline=dt.datetime.fromisoformat('2026-11-17T08:13:00+00:00').timestamp()
 if socket.gethostname()!='researcher-biolab-1' or deadline-time.time()<420:raise RuntimeError('Wrong host or insufficient biological time')
 opened=[]
 for p in Path('/proc').iterdir():
  if not p.name.isdigit():continue
  try:
   if (p/'comm').read_text().strip()!='mxwserver':continue
   for fd in (p/'fd').iterdir():
    try:
     target=os.readlink(fd)
     if '.h5' in target:opened.append(target)
    except FileNotFoundError:pass
  except (FileNotFoundError,PermissionError):pass
 if opened:raise RuntimeError('Device is recording '+repr(opened))
 root.mkdir(exist_ok=False)
 sys.path.insert(0,'/home/researcher/mea_22554_sham_20261117/BrainStimulation')
 from brain_stimulation import maxlab as mx
 from brain_stimulation.transport import Api,Endpoint,bind_sdk
 allowed=allowed_commands(mx,root);original=Api.send;cleanup=False
 report={'session_id':'28073-20261117T061300Z','expected_chip_id':'28073','organoid_biological_id':None,
  'session_started_utc':'2026-11-17T06:13:00+00:00','hard_deadline_utc':'2026-11-17T08:13:00+00:00',
  'status':'starting','duration_requested_s':300,'r2_enabled':False,'training_enabled':False,
  'nonzero_output_sent':False,'commands':[],'cleanup_errors':[]}
 def save():
  p=root/'capture.tmp';p.write_text(json.dumps(report,indent=2));p.replace(root/'capture.json')
 def guarded(api,cmd,timeout=None):
  if cmd not in allowed:raise RuntimeError('Saving/zero-only allowlist rejected '+cmd[:80])
  report['commands'].append({'utc':dt.datetime.now(dt.timezone.utc).isoformat(),'cmd':cmd})
  try:save()
  except Exception:
   report['audit_write_failed']=True
   if not cleanup:raise
  return original(api,cmd,timeout)
 def stop(sig,frame):raise KeyboardInterrupt('Stop '+str(sig))
 Api.send=guarded;file_open=recording=False;saving=mx.Saving()
 for sig in (signal.SIGINT,signal.SIGTERM,signal.SIGALRM):signal.signal(sig,stop)
 signal.alarm(max(1,int(min(420,deadline-time.time()-120))))
 try:
  with bind_sdk(mx,Endpoint(timeout=2)):
   for u in range(32):mx.send(mx.StimulationUnit(u).power_up(False).connect(False))
   mx.send(mx.DAC(0,512))
   saving.open_directory(str(root));saving.set_legacy_format(False)
   file_open=True;saving.start_file('intake');saving.group_delete_all();saving.group_define(0,'routed',list(range(1024)))
   recording=True;saving.start_recording([0]);report.update(status='recording',recording_start_utc=dt.datetime.now(dt.timezone.utc).isoformat());save()
   start=time.monotonic();time.sleep(300);report.update(status='completed',recording_elapsed=time.monotonic()-start)
 except BaseException as e:report.update(status='failed',error=repr(e))
 finally:
  cleanup=True;signal.alarm(0)
  for sig in (signal.SIGINT,signal.SIGTERM):signal.signal(sig,signal.SIG_IGN)
  with bind_sdk(mx,Endpoint(timeout=1)):
   for label,needed,fn in [('stop_recording',recording,saving.stop_recording),('stop_file',file_open,saving.stop_file),('groups',file_open,saving.group_delete_all)]:
    if needed:
     try:fn()
     except Exception as e:report['cleanup_errors'].append(label+':'+repr(e))
  Api.send=original
  if report['cleanup_errors'] or report.get('audit_write_failed'):report['status']='failed'
  report['finished_utc']=dt.datetime.now(dt.timezone.utc).isoformat();save()
 print(json.dumps({k:v for k,v in report.items() if k!='commands'},indent=2),flush=True)
 return int(report['status']!='completed')

if __name__=='__main__':
 if sys.argv[1:]!=['--run']:raise SystemExit('Explicit --run required for zero-output intake')
 raise SystemExit(main())
