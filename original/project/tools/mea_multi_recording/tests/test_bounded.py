import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from bounded import batch_commands,drive_batches,available_batches,network_gate

class Clock:
    def __init__(self):self.t=0
    def monotonic(self):return self.t
    def sleep(self,s):self.t+=s

class BoundedTests(unittest.TestCase):
    def test_stable_connection_batches_have_no_unit_switches_and_end_at_zero(self):
        for zero in (False,True):
            p=batch_commands(2.981901,[0,1,3],zero=zero,switch_units=False)
            self.assertFalse(any(x.startswith('mea_set_stimulation_unit ') for x in p['commands']))
            self.assertEqual(p['commands'][0],'mea_set_dac 0 512 512')
            self.assertEqual(p['commands'][-2],'mea_set_dac 0 512 512')
            self.assertEqual(p['duration_s'],.8254)
            self.assertIn('system_delay_samples 100',p['commands'])
            self.assertEqual(sum('system_trigger_event ' in x for x in p['commands']),16)
            if zero:self.assertTrue(all(x=='mea_set_dac 0 512 512' for x in p['commands'] if x.startswith('mea_set_dac ')))

    def test_shortened_duration_retains_full_post_reserve(self):
        self.assertEqual(available_batches(2607),600)
        self.assertEqual(available_batches(2307.9),300)
        self.assertEqual(available_batches(1407,post_seconds=600),600)
        self.assertEqual(available_batches(1000,post_seconds=600),193)
        with self.assertRaises(ValueError):available_batches(3000,post_seconds=0)
        for seconds in [1945,1200,float('nan')]:
            with self.assertRaises(RuntimeError):available_batches(seconds)

    def test_post_audit_fsync_lateness_gate(self):
        network_gate(10.04,10,0)
        with self.assertRaises(RuntimeError):network_gate(10.051,10,0)
    def test_bounded_tail_and_zero_profile(self):
        for zero in [False,True]:
            p=batch_commands(2.981901,[0,1,3],zero=zero)
            self.assertEqual(p['duration_s'],.8254)
            self.assertEqual(p['pulse_count'],15)
            self.assertEqual(p['commands'][-34:-2],[f'mea_set_stimulation_unit {u} 0 0 0 0 0 0' for u in range(32)])
            self.assertEqual(p['commands'][-2],'mea_set_dac 0 512 512')
            self.assertEqual(sum('system_trigger_event ' in x for x in p['commands']),16)
            self.assertTrue(p['commands'][-1].startswith('system_trigger_event '))
            if zero:self.assertTrue(all(x=='mea_set_dac 0 512 512' for x in p['commands'] if x.startswith('mea_set_dac ')))

    def test_full_600_batches_and_no_catchup(self):
        c=Clock();calls=[];reservations=[]
        result=drive_batches(c,lambda i:reservations.append(i),lambda i:calls.append(i),lambda:None)
        self.assertEqual(calls,list(range(600)));self.assertEqual(reservations,calls)
        self.assertEqual(result['attempted_batches'],600);self.assertGreaterEqual(c.t,600)
        c=Clock();calls=[]
        result=drive_batches(c,lambda i:None,lambda i:calls.append(i),lambda:None,batch_count=5)
        self.assertEqual(len(calls),5);self.assertEqual(c.t,5)

    def test_lost_reply_never_retries(self):
        c=Clock();calls=[];reservations=[]
        def send(i):
            calls.append(i)
            if i==2:raise TimeoutError('unknown')
        result=drive_batches(c,lambda i:reservations.append(i),send,lambda:None)
        self.assertEqual(calls,[0,1,2]);self.assertEqual(reservations,calls)
        self.assertEqual(result['status'],'output_unknown')

    def test_lateness_stops_before_extra_send(self):
        c=Clock();calls=[]
        def send(i):calls.append(i);c.t+=1.2
        result=drive_batches(c,lambda i:None,send,lambda:None)
        self.assertEqual(calls,[0]);self.assertEqual(result['status'],'stopped_late')

    def test_slow_reservation_consumes_slot_without_send(self):
        c=Clock();calls=[]
        def reserve(i):c.t+=.06
        result=drive_batches(c,reserve,lambda i:calls.append(i),lambda:None)
        self.assertEqual(calls,[]);self.assertEqual(result['status'],'stopped_late')

    def test_delayed_ack_prevents_overlap(self):
        c=Clock();calls=[]
        def send(i):calls.append(i);c.t+=.3
        result=drive_batches(c,lambda i:None,send,lambda:None)
        self.assertEqual(calls,[0]);self.assertEqual(result['status'],'stopped_late')

    def test_health_and_reservation_fail_before_send(self):
        for where in ['reserve','health']:
            c=Clock();calls=[]
            def fail(*args):raise OSError('failed')
            result=drive_batches(c,fail if where=='reserve' else lambda i:None,
                                 lambda i:calls.append(i),fail if where=='health' else lambda:None)
            self.assertEqual(calls,[]);self.assertEqual(result['status'],'stopped_before_send')

    def test_invalid_site_set_rejected(self):
        for units in [[],[1],[1,1],list(range(33))]:
            with self.assertRaises(ValueError):batch_commands(2.981901,units,zero=False)

if __name__=='__main__':unittest.main()
