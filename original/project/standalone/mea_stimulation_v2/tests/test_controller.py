from pathlib import Path
import hashlib,json,tempfile,unittest
from mea_stim_v2.config import Config
from mea_stim_v2.controller import Controller,SoftStop
from mea_stim_v2.protocol import Budget
from mea_stim_v2.worker import WorkerError

class Clock:
 def __init__(self):self.w=1794895980.;self.m=0.
 def wall(self):return self.w
 def mono(self):return self.m
 def sleep(self,s):self.w+=s;self.m+=s

def config(root):
 p=root/'route.cfg';p.write_text(' '.join([','.join(f'{i}({100+i})' for i in range(64))]))
 return Config.from_dict(dict(mode='simulate',session_id='SIM-1',chip_id='SIM-28073',organoid_id='SIM-ORG',
  started_utc='2026-11-17T06:13:00+00:00',device_hostname='SIM',route_file=str(p),route_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),
  stim_electrode=100,threshold_value=5,threshold_h5_dataset='settings/spike_threshold'))

class Device:
 def __init__(self,clock,out):self.clock=clock;self.out=out;self.pulses=[];self.closes=[];self.active=None;self.cleaned=0;self.fail_send=False;self.stop_after=None;self.stop=False;self.last_api_seconds=0
 def preflight(self):return {'lsb_mv':2.981901,'free_bytes':10**15}
 def prepare(self):pass
 def open_record(self,name):
  self.active=name;self.start=self.clock.mono();p=self.out/name;p.mkdir(parents=True);(p/(name+'.raw.h5')).write_bytes(b'fixture')
 def marker(self,i,properties):pass
 def connect(self,on):pass
 def assert_ready(self):pass
 def pulse(self,event,lsb):
  self.pulses.append((event.copy(),self.clock.mono()))
  if self.stop_after==len(self.pulses):self.stop=True
  if self.fail_send and event['arm']=='true':raise TimeoutError('lost ack')
 def close_record(self):
  name=self.active;self.active=None
  r={'path':str(self.out/name/(name+'.raw.h5')),'recorded_seconds':self.clock.mono()-self.start};self.closes.append((name,r['recorded_seconds']));return r
 def free_bytes(self):return 10**15
 def cleanup(self):self.cleaned+=1;self.active=None;return []

class Analyzer:
 def __init__(self,c,fail=None):self.c=c;self.fail=fail;self.calls=[]
 def __call__(self,stage,path,**kwargs):
  self.calls.append(stage)
  return {'status':'failed' if stage==self.fail else 'pass','errors':['injected'] if stage==self.fail else [],
   'session_id':self.c.session_id,'source':{'path':str(path)},'candidate_roi':list(range(1,64)),
   'roi':list(range(2,64)),'roi_channels':list(range(2,64)),'reference_rate':62.,'trials':[]}

class ControllerTests(unittest.TestCase):
 def run_case(self,fail=None,fail_send=False,stop_after=None):
  tmp=tempfile.TemporaryDirectory();self.addCleanup(tmp.cleanup);b=Path(tmp.name);c=config(b);clock=Clock();out=b/'run'
  dev=Device(clock,out);dev.fail_send=fail_send;dev.stop_after=stop_after
  ctl=Controller(c,out,b/'registry',dev,Analyzer(c,fail),clock,stop_requested=lambda:dev.stop)
  return ctl.run(),dev,clock,out
 def test_full_run_has_exact_dose_and_final_recovery(self):
  r,d,c,p=self.run_case();self.assertEqual(r['status'],'completed');self.assertEqual(r['dose_totals'],{'true':11,'sham':11})
  self.assertEqual(d.cleaned,1);self.assertEqual(len(d.pulses),22);self.assertGreaterEqual(d.closes[-1][1],675)
  self.assertLess(c.mono(),7200);self.assertFalse(r['r2_enabled']);self.assertEqual(len(r['accepted_pairs']),10)
 def test_baseline_failure_cannot_send_any_pulse(self):
  r,d,c,p=self.run_case(fail='baseline');self.assertEqual(d.pulses,[]);self.assertEqual(r['status'],'stopped');self.assertEqual(d.cleaned,1)
 def test_failed_pair_stops_next_pair_and_records_fresh_recovery(self):
  r,d,c,p=self.run_case(fail='pair_01');self.assertEqual(len(d.pulses),4);self.assertEqual(d.closes[-1][0],'recovery')
  self.assertGreaterEqual(d.closes[-1][1],600);self.assertEqual(r['status'],'stopped');self.assertEqual(r['accepted_pairs'],[])
 def test_lost_true_reply_is_reserved_once_and_no_retry(self):
  r,d,c,p=self.run_case(fail_send=True);self.assertEqual(r['dose_totals'],{'true':1,'sham':1});self.assertEqual(len(d.pulses),2)
  self.assertEqual(d.cleaned,1);self.assertFalse(any(x[0]=='recovery' for x in d.closes))
 def test_manual_stop_between_pair_events_blocks_second(self):
  r,d,c,p=self.run_case(stop_after=3);self.assertEqual(len(d.pulses),3);self.assertEqual(d.cleaned,1);self.assertEqual(r['status'],'stopped')
 def test_worker_timeout_after_true_runs_fresh_recovery(self):
  tmp=tempfile.TemporaryDirectory();self.addCleanup(tmp.cleanup);b=Path(tmp.name);c=config(b);clock=Clock();out=b/'run';dev=Device(clock,out)
  normal=Analyzer(c)
  def analyzer(stage,path,**kwargs):
   if stage=='pair_01':raise WorkerError('worker deadline')
   return normal(stage,path,**kwargs)
  r=Controller(c,out,b/'registry',dev,analyzer,clock).run()
  self.assertEqual(r['status'],'stopped');self.assertEqual(dev.closes[-1][0],'recovery');self.assertEqual(len(dev.pulses),4)
 def test_signal_barrier_runs_before_any_cleanup(self):
  tmp=tempfile.TemporaryDirectory();self.addCleanup(tmp.cleanup);b=Path(tmp.name);c=config(b);clock=Clock();out=b/'run';dev=Device(clock,out);events=[]
  cleanup=dev.cleanup
  def checked_cleanup():
   self.assertEqual(events,['signals_deferred']);events.append('cleanup');return cleanup()
  dev.cleanup=checked_cleanup
  ctl=Controller(c,out,b/'registry',dev,Analyzer(c,'baseline'),clock)
  ctl.before_cleanup=lambda:events.append('signals_deferred')
  r=ctl.run();self.assertEqual(events,['signals_deferred','cleanup']);self.assertEqual(r['cleanup_errors'],[])
 def test_stop_arriving_during_final_target_sleep_blocks_send(self):
  tmp=tempfile.TemporaryDirectory();self.addCleanup(tmp.cleanup);b=Path(tmp.name);c=config(b);clock=Clock();dev=Device(clock,b/'run')
  ctl=Controller(c,b/'run',b/'registry',dev,Analyzer(c),clock,stop_requested=lambda:dev.stop)
  ctl.budget=Budget(c.started_wall(),clock);ctl.target=.25;ctl.required=1000
  original=clock.sleep
  def sleep(seconds):original(seconds);dev.stop=True
  clock.sleep=sleep
  with self.assertRaises(SoftStop):ctl.before_send({'arm':'true'})
 def test_pilot_short_file_cannot_claim_host_wait_as_recovery(self):
  r,d,c,p=self.run_case(fail='pilot')
  self.assertEqual(r['status'],'stopped');self.assertEqual(d.closes[-1][0],'recovery')

if __name__=='__main__':unittest.main()
