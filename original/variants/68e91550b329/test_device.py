from contextlib import contextmanager
from pathlib import Path
import socket
import sys
import tempfile
import unittest
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from test_session import Clock,binding
try:
    from device import MaxLabBackend, compile_preview
except ImportError:MaxLabBackend=None


class Wire:
    def __init__(self,clock):self.clock=clock;self.calls=[];self.lsb='2.981901';self.bad_route=False;self.links={9:7}
    def __call__(self,message,timeout=5):
        self.calls.append((self.clock.m,message))
        if message=='system_query_dac_lsb_mv':return self.lsb
        if message=='wellplate_query_version':return '0'
        if message.startswith('sequence_new '):return message.split()[1]
        if 'mea_array_disconnect_amplifier_from_stimulation ' in message:
            self.links.pop(int(message.split()[-1]),None);return 'Ok'
        if 'mea_array_connect_electrode_to_stimulation ' in message:self.links[2]=7;return 'Ok'
        if 'mea_array_query_stimulation_at_amplifier ' in message:return str(self.links.get(int(message.split()[-1]),-1))
        if 'mea_array_get_config' in message:return '0(40)1(41)2(42)3(43)'
        if 'mea_array_query_stimulation_at_electrode' in message:return '7'
        if 'mea_array_query_amplifier_at_stimulation ' in message:
            unit=int(message.split()[-1]);matches=[ch for ch,u in self.links.items() if u==unit]
            return str(3 if self.bad_route else matches[0]) if matches else ''
        if 'mea_array_query_amplifier_at_' in message:return '3' if self.bad_route else '2'
        return 'Ok'


class DeviceTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(MaxLabBackend,'device adapter is not implemented')
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.path=Path(self.tmp.name); (self.path/'fresh.cfg').write_text('0(40)1(41)2(42)3(43)')
        b=binding();b['route_cfg']=str(self.path/'fresh.cfg')
        self.clock=Clock();self.wire=Wire(self.clock)
        self.d=MaxLabBackend(b,self.path,request=self.wire,clock=self.clock)
        self.d.wait=self.clock.sleep
        self.net=patch.object(socket,'create_connection',side_effect=AssertionError('NO NETWORK'))
        self.net.start();self.addCleanup(self.net.stop)
    def test_sdk_timing_voltage_quantization_no_socket(self):
        r=compile_preview(2.981901)
        self.assertEqual(r['codes'],[311,713,512]);self.assertAlmostEqual(r['nominal_phase_mv'],599.362101)
        self.assertEqual(r['pulse_onset_ms'],[5,15,25,205,215,225,405,415,425,605,615,625,805,815,825])
        self.assertAlmostEqual(r['duration_s'],.8254)
        self.assertEqual(compile_preview(3.1)['codes'],[319,705,512])
        for v in [0,float('nan'),.01,601]:
            with self.assertRaises(ValueError):compile_preview(v)
    def test_calibration_wait_then_connection_wait_before_file(self):
        with self.d.sdk_context():
            self.d.configure({'channel':2,'electrode':42})
            self.clock.sleep(5);self.d.prepare();self.d.open('baseline');self.d.cleanup()
        calls=self.wire.calls
        offset=next(t for t,c in calls if c=='system_offset')
        connected=next(t for t,c in calls if c.startswith('mea_set_stimulation_unit 7 1 1 '))
        started=next(t for t,c in calls if c=='saving_start_recording')
        self.assertGreaterEqual(connected-offset,10)
        self.assertGreaterEqual(started-connected,5)
        cmds=[c for t,c in calls]
        self.assertLess(cmds.index('system_clear_events'),next(i for i,c in enumerate(cmds) if c.startswith('sequence_append')))
        self.assertFalse(any(c.startswith('sequence_send ') for c in cmds))
        self.assertTrue(all(c=='mea_set_dac 0 512 512' for c in cmds if c.startswith('mea_set_dac')))
    def test_mismatched_reverse_route_rejects_before_enable(self):
        self.wire.bad_route=True
        with self.d.sdk_context():
            with self.assertRaises(RuntimeError):self.d.configure({'channel':2,'electrode':42})
            self.d.cleanup()
        self.assertFalse(any(c.startswith('mea_set_stimulation_unit 7 1 1 ') for t,c in self.wire.calls))
    def test_inherited_shared_stimulus_route_removed_before_download(self):
        with self.d.sdk_context():self.d.configure({'channel':2,'electrode':42});self.d.cleanup()
        self.assertEqual(self.wire.links,{2:7})
    def test_no_disconnect_request_for_unconnected_unsupported_amplifier(self):
        original=self.d.request
        def real_semantics(message,timeout):
            if message.endswith('mea_array_disconnect_amplifier_from_stimulation 360'):
                raise RuntimeError('Error: unsupported amplifier')
            if 'mea_array_query_stimulation_at_amplifier ' in message:return '7'
            return original(message,timeout)
        self.d.request=real_semantics
        with self.d.sdk_context():
            self.d.configure({'channel':2,'electrode':42});self.d.cleanup()
        self.assertEqual(self.wire.links,{2:7})
    def test_unqualified_send_is_blocked(self):
        with self.d.sdk_context():
            self.d.configure({'channel':2,'electrode':42});self.d.prepare()
            with self.assertRaises(RuntimeError):self.d.send()
            self.d.cleanup()
        self.assertFalse(any(c.startswith('sequence_send ') for t,c in self.wire.calls))
    def test_missing_post_file_cannot_send_even_after_good_baseline(self):
        from types import SimpleNamespace
        self.d.controller=SimpleNamespace(report={'stage':'single_stimulus_attempt',
              'stimulus_attempted':True,'baseline_qc':{'status':'passed'}},
              deadline=SimpleNamespace(require=lambda value:None))
        self.d.pulse_token='full_pulse_test';self.d.current_file=self.path/'stimulus_post.raw.h5'
        self.d.record_attempted=True
        with self.d.sdk_context():
            with self.assertRaises(RuntimeError):self.d.send()
        self.assertFalse(any(c.startswith('sequence_send ') for t,c in self.wire.calls))
    def test_static_nonempty_header_does_not_count_as_recording_growth(self):
        self.d.current_file=self.path/'stimulus_post.raw.h5'
        self.d.current_file.write_bytes(b'header'*20000)
        self.d.record_attempted=True;self.d.last_growth=self.clock.monotonic()
        self.d.progress();self.clock.sleep(60)
        with self.assertRaises(RuntimeError):self.d.verify_recording_ready()
    def test_cleanup_continues_after_one_output_shutdown_fails(self):
        original=self.d.request
        def fail(message,timeout):
            if message.startswith('mea_set_stimulation_unit 0 0 0'):raise TimeoutError('injected')
            return original(message,timeout)
        self.d.request=fail;self.d.touched=True
        with self.d.sdk_context():errors=self.d.cleanup()
        self.assertTrue(errors)
        self.assertIn('mea_set_dac 0 512 512',[c for t,c in self.wire.calls])
        self.assertTrue(any(c.startswith('mea_set_stimulation_unit 31 0 0') for t,c in self.wire.calls))


if __name__=='__main__':unittest.main()
