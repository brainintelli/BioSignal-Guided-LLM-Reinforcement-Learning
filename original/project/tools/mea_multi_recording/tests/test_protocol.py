import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from multi_protocol import compile_sequence,check_pulse_events,validate_routes,required_seconds

class ProtocolTests(unittest.TestCase):
    def test_true_sequence_exact_dose_and_timing(self):
        p=compile_sequence(2.981901,zero=False)
        self.assertEqual(p['pulse_count'],9000)
        self.assertEqual(p['onset_ticks'][:6],[100,300,500,4100,4300,4500])
        self.assertEqual(p['onset_ticks'][-1],11996500)
        self.assertEqual(p['duration_ticks'],12000100)
        self.assertEqual(p['phase_ticks'],4)
        self.assertEqual(p['codes'],[311,713,512])
        self.assertLessEqual(p['nominal_phase_mv'],600)
        self.assertEqual(sum(x=='mea_set_dac 0 311 512' for x in p['commands']),9000)
        self.assertEqual(sum(x=='mea_set_dac 0 713 512' for x in p['commands']),9000)
        self.assertEqual(p['commands'][-2],'mea_set_dac 0 512 512')
        self.assertEqual(p['commands'][-34:-2],[f'mea_set_stimulation_unit {u} 0 0 0 0 0 0' for u in range(32)])
        self.assertTrue(p['commands'][-1].startswith('system_trigger_event 0 1 209000 '))
        self.assertFalse(any(x.startswith('mea_set_stimulation_unit') and x.split()[2]!='0' for x in p['commands']))

    def test_zero_sequence_same_timing_never_nonzero(self):
        a=compile_sequence(2.981901,zero=False);b=compile_sequence(2.981901,zero=True)
        self.assertEqual(a['onset_ticks'],b['onset_ticks'])
        self.assertEqual(a['duration_ticks'],b['duration_ticks'])
        self.assertEqual(len(a['commands']),len(b['commands']))
        self.assertTrue(all(x=='mea_set_dac 0 512 512' for x in b['commands'] if x.startswith('mea_set_dac')))

    def test_lsb_rejected(self):
        for lsb in [0,-1,float('nan'),float('inf'),601,.1]:
            with self.subTest(lsb=lsb),self.assertRaises(ValueError):compile_sequence(lsb,zero=False)

    def test_events_complete_and_exact(self):
        p=compile_sequence(2.981901,zero=True)
        events=[dict(id=100000+i,type=1,frame=5000+t) for i,t in enumerate(p['onset_ticks'])]
        self.assertEqual(check_pulse_events(events,zero=True),[])
        self.assertTrue(check_pulse_events(events[:-1],zero=True))
        events[20]['frame']+=1
        self.assertTrue(check_pulse_events(events,zero=True))

    def test_distinct_units_required(self):
        good=[dict(electrode=100+u,channel=u,unit=u) for u in range(32)]
        validate_routes(good,{u:u for u in range(32)})
        for bad in [good+[dict(electrode=500,channel=500,unit=32)],good[:1],good+[good[0]]]:
            with self.assertRaises(ValueError):validate_routes(bad,{u:u for u in range(32)})
        with self.assertRaises(ValueError):validate_routes(good,{u:u for u in range(31)})

    def test_full_post_and_cleanup_reserved(self):
        self.assertEqual(required_seconds(),2540.005)

if __name__=='__main__':unittest.main()
