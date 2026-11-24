import copy
import dataclasses
import json
from pathlib import Path
import tempfile
import unittest

from mea_binary_r2.encoding import Encoder, compile_sequence
from mea_binary_r2.contract import reward_record, TransitionGate
from mea_binary_r2.offline import run_offline, simulated_evidence
try:
    from mea_binary_r2.profile import PulseProfile, load_profile
except ImportError:
    PulseProfile = load_profile = None

PROFILE = dict(schema='mea.pulse_profile.v1', profile_id='test-200mv-10ms',
               source_sha256='a'*64, target_mv=200., phase_us=200,
               pulse_interval_us=10000, pulses_per_stimulus=3, polarity='positive_first')
ID = dict(session_id='sim', chip_id='sim-chip', organoid_id='sim-organoid',
          route_id='sim-route', episode_id='1', transition_id='1', context_id='after-1')


class ProfileTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(PulseProfile, 'Profile interface has not been implemented')
        self.profile = PulseProfile.from_dict(PROFILE)

    def args(self, quiet=False):
        intent=Encoder(pulses_per_stimulus=3).decide({'H':100,'W':100 if quiet else 80,'F':100})
        receipt=simulated_evidence(ID,intent,100000,'event','window',profile=self.profile)
        return dict(r1=.25,duration=2,gamma=.5,beta=.002,intent=intent,
                    evidence=receipt,expected_identity=ID,profile=self.profile)

    def test_new_profile_changes_real_compiler_and_encoder(self):
        for reserve in (80,40):
            intent=Encoder(pulses_per_stimulus=3).decide({'H':100,'W':reserve,'F':100})
            self.assertEqual(intent.pulses,3)
            sequence=compile_sequence(intent,2.9,profile=self.profile)
            elapsed=0;starts=[]
            for command,value in sequence:
                if command=='delay':elapsed+=value
                elif value==444:starts.append(elapsed)
            self.assertEqual(starts,[0,200,400])
            self.assertEqual(elapsed,408)
            self.assertEqual(sequence[:5],(('dac',444),('delay',4),('dac',580),('delay',4),('dac',512)))
            self.assertEqual(sequence[-1],('dac',512))

    def test_profile_is_frozen_strict_and_fingerprinted(self):
        self.assertEqual(self.profile.to_dict(),PROFILE)
        self.assertEqual(len(self.profile.digest),64)
        with self.assertRaises(dataclasses.FrozenInstanceError):self.profile.target_mv=300
        changed=PulseProfile.from_dict(PROFILE|{'source_sha256':'b'*64})
        self.assertNotEqual(changed.digest,self.profile.digest)
        for key,value in [('target_mv',True),('target_mv',float('nan')),('target_mv',301),
                          ('phase_us',201),('pulse_interval_us',399),('pulse_interval_us',True),
                          ('pulses_per_stimulus',202),('polarity','unknown'),('source_sha256','unverified')]:
            with self.subTest(key=key),self.assertRaises(ValueError):PulseProfile.from_dict(PROFILE|{key:value})
        with self.assertRaises(ValueError):PulseProfile.from_dict(PROFILE|{'run_this':'instruction'})

    def test_latest_profile_reward_preserves_r1_and_binary_signs(self):
        for quiet,total,sign in [(True,.253,1),(False,.247,-1)]:
            args=self.args(quiet);result=reward_record(**args)
            self.assertTrue(result['valid'])
            self.assertEqual(result['r1'],.25)
            self.assertAlmostEqual(result['total_reward'],total)
            self.assertEqual(result['r2_raw'],sign)
            self.assertEqual(result['stimulus_profile_sha256'],self.profile.digest)
            gate=TransitionGate({k:ID[k] for k in ('session_id','chip_id','organoid_id','route_id')})
            self.assertTrue(gate.accept(**args)['valid'])

    def test_wrong_profile_old_timing_and_excess_amplitude_rejected(self):
        args=self.args();args['evidence']['stimulus_profile_sha256']='b'*64
        with self.assertRaises(ValueError):reward_record(**args)
        args=self.args();del args['evidence']['stimulus_profile_sha256']
        with self.assertRaises(ValueError):reward_record(**args)
        for key,value in [('pulse_frames',[100000,106000,112000]),('max_abs_output_mv',200.01)]:
            args=self.args();args['evidence'][key]=value
            result=reward_record(**args)
            self.assertFalse(result['valid']);self.assertIsNone(result['total_reward'])
        intent=Encoder().decide({'H':100,'W':80,'F':100})
        with self.assertRaises(ValueError):compile_sequence(intent,2.9,profile=self.profile)

    def test_profile_does_not_admit_recording_or_live_receipts(self):
        for source in ('recording','live'):
            args=self.args();args['evidence']['source']=source
            with self.subTest(source=source),self.assertRaises(ValueError):reward_record(**args)
        args=self.args();args['evidence']['qc_passed']=False
        result=reward_record(**args)
        self.assertEqual(result['r2_raw'],-1);self.assertFalse(result['valid'])

    def test_profile_loading_and_offline_pipeline(self):
        with tempfile.TemporaryDirectory() as td:
            profile_path=Path(td)/'profile.json'
            profile_path.write_text(json.dumps(dict(schema='mea.stimulus_profile_import.v1',execution_profile=PROFILE)))
            self.assertEqual(load_profile(profile_path),self.profile)
            out=Path(td)/'run';result=run_offline(out,profile=self.profile)
            self.assertTrue(result['passed'])
            self.assertEqual(result['simulated_pulses'],12)
            self.assertEqual(result['stimulus_profile_sha256'],self.profile.digest)
            rows=[json.loads(line) for line in (out/'transitions.jsonl').read_text().splitlines()]
            self.assertEqual([r['reward']['r2_raw'] for r in rows],[1,-1,-1,1,-1,-1,1,None])
            self.assertEqual(rows[1]['receipt']['pulse_frames'],[500000,500200,500400])
            self.assertTrue(all(r['source']=='simulation' for r in rows))


if __name__=='__main__':unittest.main()
