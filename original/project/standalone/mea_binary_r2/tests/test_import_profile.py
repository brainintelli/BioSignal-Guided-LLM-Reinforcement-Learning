import hashlib
import json
from pathlib import Path
import tempfile
import unittest
import h5py
import numpy as np
from mea_binary_r2.profile import load_profile
try:
    from mea_binary_r2.import_profile import inspect_recording, import_recording
except ImportError:
    inspect_recording = import_recording = None


def fixture(path, change=None):
    with h5py.File(path,'w') as f:
        for name,value in dict(amplitude_range='200,200,200',phase='200',ipi='10',ipti='1',
                               pulses='3',pulse_trains='2',pre_stim_period='0',post_stim_period='1',
                               randomize='true',electrodes=json.dumps({'electrodes':{'0':{'stimulation':[7143,2733]}}})).items():
            f.create_dataset('assay/inputs/'+name,data=[value.encode()])
        for name,value in [('assay/script_id','VoltageStimulation_v1.0'),('assay/run_id','000014'),
                           ('wellplate/id','22021'),('wellplate/version','MaxOne Single Well MEA')]:
            f.create_dataset(name,data=[value.encode()])
        d=f.create_group('data_store/data0000')
        for name,value in [('sampling',20000.),('gain',512.),('hpf',300.),('lsb',6.29e-6)]:
            d.create_dataset('settings/'+name,data=[value])
        d.create_dataset('settings/mapping',data=np.array([(0,7143,0.,0.),(1,2733,1.,1.)],
            dtype=[('channel','i4'),('electrode','i4'),('x','f8'),('y','f8')]))
        message=json.dumps({'amplitude':'200','phase':'200'})
        rows=[(100000+train*20000+pulse*200,1,train+1,message) for train in range(2) for pulse in range(3)]
        if change=='timing':rows[1]=(100201,1,1,message)
        if change=='count':rows.pop()
        if change=='message':rows[0]=(100000,1,1,json.dumps({'amplitude':'300','phase':'200'}))
        if change=='id':rows[0]=(100000,1,99,message)
        d.create_dataset('events',data=np.array(rows,dtype=[('frameno','i8'),('eventtype','u4'),
            ('eventid','u4'),('eventmessage',h5py.string_dtype())]))
        d.create_dataset('spikes',data=np.array([(99000,0,-20.),(141000,1,-30.)],
            dtype=[('frameno','i8'),('channel','i4'),('amplitude','f4')]))
        d.create_dataset('well_id',data=[0]);d.create_dataset('start_time',data=[1794912889126])
        d.create_dataset('stop_time',data=[1794912891230])


class ImportTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(inspect_recording,'H5 profile importer not implemented')

    def test_metadata_and_frames_agree_without_inventing_dac_or_training(self):
        with tempfile.TemporaryDirectory() as td:
            source=Path(td)/'record.h5';fixture(source)
            result=inspect_recording(source)
            h=result['historical_recording'];p=result['execution_profile']
            self.assertEqual((h['observed_event_count'],h['train_count'],h['pulses_per_train']),(6,2,3))
            self.assertEqual(h['observed_pulse_interval_frames'],[200])
            self.assertEqual(h['observed_group_start_interval_frames'],[20000])
            self.assertEqual(h['configured_stimulation_electrodes'],[7143,2733])
            self.assertEqual((p['target_mv'],p['phase_us'],p['pulse_interval_us']),(200.,200,10000))
            self.assertIsNone(h['stimulation_dac_lsb_mv'])
            self.assertEqual(h['recording_adc_lsb_volts'],6.29e-6)
            self.assertFalse(result['admission']['historical_training_admitted'])
            self.assertFalse(result['admission']['real_stimulation_enabled'])
            self.assertIn('no_continuous_raw_waveform',result['admission']['reasons'])
            self.assertEqual(result['reward_contract']['r1_source'],'native_game_required_not_in_h5')

    def test_corrupt_count_timing_message_or_group_number_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            for change in ('count','timing','message','id'):
                source=Path(td)/(change+'.h5');fixture(source,change)
                with self.subTest(change=change),self.assertRaises(ValueError):inspect_recording(source)

    def test_fixed_copy_event_candidates_and_loaded_profile(self):
        with tempfile.TemporaryDirectory() as td:
            source=Path(td)/'record.h5';fixture(source)
            original=source.read_bytes();digest=hashlib.sha256(original).hexdigest()
            out=Path(td)/'import'
            result=import_recording(source,out,expected_sha256=digest)
            self.assertEqual(source.read_bytes(),original)
            self.assertEqual((out/'source.h5').read_bytes(),original)
            self.assertTrue(result['parameter_import_passed'])
            profile=load_profile(out/'stimulus_profile.json')
            self.assertEqual(profile.source_sha256,digest)
            rows=[json.loads(x) for x in (out/'reward_event_candidates.jsonl').read_text().splitlines()]
            self.assertEqual(len(rows),2)
            self.assertTrue(all(r['r2_candidate']==-1 and r['r1'] is None and r['total_reward'] is None
                                and r['game_transition_id'] is None and not r['training_admitted'] for r in rows))
            self.assertEqual(rows[0]['pulse_frames'],[100000,100200,100400])
            with self.assertRaises(FileExistsError):import_recording(source,out,expected_sha256=digest)

    def test_expected_source_hash_is_enforced(self):
        with tempfile.TemporaryDirectory() as td:
            source=Path(td)/'record.h5';fixture(source)
            with self.assertRaises(ValueError):import_recording(source,Path(td)/'bad',expected_sha256='0'*64)


if __name__=='__main__':unittest.main()
