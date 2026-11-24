import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from multi_prepare import software_allowed,append_ack_valid

class PrepareTests(unittest.TestCase):
    def test_event_empty_reply_is_command_specific(self):
        self.assertTrue(append_ack_valid('system_trigger_event 0 1 100000 mode zero pulse 0',''))
        self.assertTrue(append_ack_valid('system_delay_samples 100','OK'))
        for command in ['mea_set_dac 0 512 512','system_delay_samples 4','mea_set_stimulation_unit 1 0 0 0 0 0 0']:
            self.assertFalse(append_ack_valid(command,''))
        self.assertFalse(append_ack_valid('system_trigger_event 0 1 100000 mode zero','Error: failed'))
    def test_no_hardware_side_effect_commands(self):
        for command in ['sequence_send zero_x','system_clear_events','mea_set_dac 0 311 512',
                        'system_initialize','mea_array_command route_x\nmea_array_download 0',
                        'mea_array_delete online','sequence_append other\nsystem_delay_samples 100']:
            self.assertFalse(software_allowed(command,'route_x',{'zero_x'}),command)
    def test_owned_software_objects_only(self):
        for command in ['system_query_dac_lsb_mv','mea_array_new route_x','mea_array_delete route_x',
                        'mea_array_command route_x\nmea_array_query_amplifier_at_stimulation 1',
                        'mea_array_command route_x\nmea_array_query_stimulation_at_amplifier 640',
                        'sequence_new zero_x','sequence_append zero_x\nmea_set_dac 0 512 512',
                        'sequence_delete zero_x']:
            self.assertTrue(software_allowed(command,'route_x',{'zero_x'}),command)

if __name__=='__main__':unittest.main()
