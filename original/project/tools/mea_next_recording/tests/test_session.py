import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
try:
    from protocol import validate_binding, Deadline
    from session import Session
except ImportError:
    Session = None


class Clock:
    def __init__(self): self.t = 1000.; self.m = 0.
    def time(self): return self.t
    def monotonic(self): return self.m
    def sleep(self, seconds): self.t += seconds; self.m += seconds


class Device:
    def __init__(self, clock, bad=False, lost=False):
        self.clock=clock; self.calls=[]; self.bad=bad; self.lost=lost
    def call(self, name, *args): self.calls.append((name, self.clock.m, args))
    def preflight(self): self.call('preflight')
    def setup_identity(self): self.call('setup_identity')
    def open(self, name): self.call('open',name)
    def close(self): self.call('close')
    def identity(self): self.call('identity'); return {'electrode':42,'channel':2}
    def configure(self, candidate): self.call('configure',candidate)
    def prepare(self): self.call('prepare')
    def marker(self): self.call('marker')
    def qualify(self): self.call('qualify'); return {'status':'failed' if self.bad else 'passed'}
    def send(self):
        self.call('send')
        if self.lost: raise TimeoutError('lost reply')
    def zero(self): self.call('zero')
    def progress(self): pass
    def cleanup(self): self.call('cleanup'); return []
    def audit(self): self.call('audit'); return {'status':'passed','training_admitted':False}


def binding():
    return dict(chip_id='NEW42', session_started_at='1970-01-01T00:15:00+00:00',
                acquisition_host='192.0.2.101', route_cfg='fresh.cfg',
                alpha=1, r2_mode='record_only', realtime_training=False,
                offline_training=False, protocol='photo_full_v1')


class SessionTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(Session, 'next-round stage controller is not implemented')
        self.tmp=tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name); self.clock=Clock()
    def run_session(self, dev=None, config=None, name='run'):
        dev=dev or Device(self.clock)
        result=Session(config or binding(),dev,self.clock,self.root/name,self.root/'journal').run()
        return result,dev
    def test_unbound_and_invalid_boundaries_reject_before_device(self):
        for key,value in [('chip_id',None),('session_started_at',None),('alpha',0.5),
                          ('r2_mode','reward'),('realtime_training',True),('offline_training',True)]:
            b=binding(); b[key]=value; d=Device(self.clock)
            with self.assertRaises((ValueError,RuntimeError)): self.run_session(d,b,key)
            self.assertEqual(d.calls,[])
    def test_expired_future_and_naive_times_reject(self):
        for stamp in ['1970-01-01T00:00:00','1970-01-01T01:00:00+00:00','1969-12-30T00:00:00+00:00']:
            b=binding();b['session_started_at']=stamp
            with self.assertRaises((ValueError,RuntimeError)): self.run_session(config=b)
    def test_success_complete_stages_one_send_and_full_recovery(self):
        result,d=self.run_session()
        self.assertEqual(result['status'],'completed')
        names=[x[0] for x in d.calls]; self.assertEqual(names.count('send'),1)
        opens={x[2][0]:x[1] for x in d.calls if x[0]=='open'}
        sent=next(x[1] for x in d.calls if x[0]=='send')
        closes=[x[1] for x in d.calls if x[0]=='close']
        self.assertAlmostEqual(closes[0]-opens['identity'],60)
        self.assertAlmostEqual(closes[1]-opens['baseline'],600)
        self.assertAlmostEqual(sent-opens['stimulus_post'],60)
        self.assertAlmostEqual(closes[2]-sent,1800.8254)
        self.assertFalse(result['training_admitted'])
        self.assertLess(names.index('qualify'),names.index('send'))
        self.assertEqual(names[-2:],['cleanup','audit'])
    def test_failed_gate_never_sends(self):
        result,d=self.run_session(Device(self.clock,bad=True))
        self.assertEqual(result['status'],'failed')
        self.assertNotIn('send',[x[0] for x in d.calls])
        self.assertIn('cleanup',[x[0] for x in d.calls])
    def test_unknown_send_retains_full_post_no_retry(self):
        result,d=self.run_session(Device(self.clock,lost=True))
        self.assertEqual(result['status'],'output_unknown')
        sends=[x for x in d.calls if x[0]=='send']; self.assertEqual(len(sends),1)
        closed=[x[1] for x in d.calls if x[0]=='close'][-1]
        self.assertGreaterEqual(closed-sends[0][1]+1e-8,1800.8254)
    def test_different_output_cannot_repeat_same_biological_session(self):
        self.run_session()
        d=Device(self.clock)
        with self.assertRaises(FileExistsError): self.run_session(d,name='other_output')
        self.assertEqual(d.calls,[])
    def test_alias_ip_cannot_reset_local_session_journal(self):
        self.run_session()
        b=binding();b['acquisition_host']='127.0.0.1';d=Device(self.clock)
        with self.assertRaises(FileExistsError):self.run_session(d,b,'alias')
        self.assertEqual(d.calls,[])
    def test_remaining_budget_before_send_preserves_30_min(self):
        d=Device(self.clock)
        def slow_qc(): self.clock.sleep(5000); return {'status':'passed'}
        d.qualify=slow_qc
        result,_=self.run_session(d)
        self.assertEqual(result['status'],'failed')
        self.assertNotIn('send',[x[0] for x in d.calls])
    def test_clock_rollback_does_not_extend_deadline(self):
        g=Deadline(900,self.clock); self.clock.sleep(6000); self.clock.t=1000
        with self.assertRaises(RuntimeError): g.require(1940.8254)
    def test_cleanup_errors_prevent_completed(self):
        d=Device(self.clock); d.cleanup=lambda:['DAC reset unconfirmed']
        result,_=self.run_session(d)
        self.assertEqual(result['status'],'failed'); self.assertTrue(result['cleanup_errors'])

    def test_metadata_write_failure_cannot_skip_hardware_cleanup(self):
        d=Device(self.clock)
        s=Session(binding(),d,self.clock,self.root/'run',self.root/'journal')
        original=s.save
        def fail_save():
            if any(x[0]=='setup_identity' for x in d.calls):raise OSError('disk full')
            original()
        s.save=fail_save
        result=s.run()
        self.assertIn('cleanup',[x[0] for x in d.calls]);self.assertEqual(result['status'],'failed')

    def test_zero_shutdown_time_is_inside_full_post_window(self):
        d=Device(self.clock)
        d.zero=lambda:self.clock.sleep(70)
        result,d=self.run_session(d)
        self.assertEqual(result['status'],'completed')
        sent=next(x[1] for x in d.calls if x[0]=='send')
        closed=[x[1] for x in d.calls if x[0]=='close'][-1]
        self.assertAlmostEqual(closed-sent,1800.8254)


if __name__=='__main__': unittest.main()
