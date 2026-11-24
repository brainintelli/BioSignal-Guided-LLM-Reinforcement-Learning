import importlib
from pathlib import Path
import sys
import unittest
ROOT=Path(__file__).resolve().parents[3]
sys.path[:0]=[str(ROOT/'standalone/mea_state2_r2'),str(ROOT/'standalone/mea_binary_r2'),str(ROOT)]


class PreparationTests(unittest.TestCase):
    def module(self):
        try:return importlib.import_module('mea_state2_r2.preparation')
        except ModuleNotFoundError:self.fail('Prospective preparation implementation missing')

    def test_waveform_offsets_symmetric_voltage_time_and_zero_end(self):
        m=self.module()
        waves=m.waveforms()
        self.assertEqual(set(waves),{'H','W','F','quiet'})
        for site in ('H','W','F'):
            w=waves[site]
            self.assertEqual(w['pulse_start_frames'],[0,200,400])
            self.assertEqual(w['end_frame'],408)
            self.assertEqual(sum(x['voltage_mv']*x['duration_frames'] for x in w['segments']),0.)
            self.assertEqual(w['segments'][-1]['voltage_mv'],0.)
            self.assertIsNone(w['physical_electrode'])
        self.assertFalse(waves['quiet']['stimulation'])

    def test_unknown_binding_cannot_be_compiled_for_hardware(self):
        m=self.module()
        b=m.binding_template()
        with self.assertRaises(ValueError):m.validate_binding(b)
        self.assertFalse(b['armed'])
        self.assertIsNone(b['organoid_id'])
        self.assertIsNone(b['actual_dac_lsb_mv'])

    def test_capture_includes_continuous_raw_spikes_events_and_budget(self):
        m=self.module()
        c=m.capture_template(746)
        self.assertEqual(c['sampling_hz'],20000)
        self.assertTrue(c['continuous_raw'])
        self.assertTrue(c['spikes'])
        self.assertTrue(c['events'])
        self.assertEqual(c['uncompressed_bytes_two_hours'],746*20000*2*7200)
        self.assertGreater(c['minimum_free_bytes'],c['uncompressed_bytes_two_hours'])
        self.assertEqual(c['feature_window_relative_frames'],[240000,260000])
        self.assertEqual(c['minimum_cycle_frames'],400000)
        with self.assertRaises(ValueError):m.capture_template(0)


if __name__=='__main__':unittest.main()
