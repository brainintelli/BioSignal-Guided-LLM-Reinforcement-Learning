import copy
import dataclasses
import importlib
import json
from pathlib import Path
import tempfile
import unittest

from agent.mea_protocol import canonical,digest
from agent.mea_recording import sha256
from agent.mea_snapshot import Normalization


def fixture(root, source='recording'):
    run=root/'accepted';run.mkdir();mapping=[[0,10],[1,11]]
    rows=[dict(type='session',protocol='mea.spikes.v1',session_id='accepted',seq=0,source=source,
               sample_rate_hz=1000,well_id=0,start_frame=100,channel_map_id=digest(mapping),channels=mapping),
          dict(type='spikes',session_id='accepted',seq=1,events=[[100,0,-10.],[1099,1,-20.],[2100,0,-30.]]),
          dict(type='watermark',session_id='accepted',seq=2,end_frame=3500,complete=True,dropped=0)]
    (run/'recording.jsonl').write_text(''.join(canonical(x)+'\n' for x in rows),encoding='utf-8')
    (run/'baseline.raw.h5').write_bytes(b'audit fixture only')
    (run/'events.jsonl').write_bytes(b'audit fixture only')
    acquisition=dict(success=True,stimulation=False,alpha=1,r2_reward_enabled=False,expected_chip_id='22021')
    (run/'acquisition.json').write_text(canonical(acquisition))
    audit=dict(offline_integrity_passed=True,errors=[],audit_interval=[100,3500],matched_events=3,
               channel_map_id=digest(mapping),metadata=dict(chip_id='22021',well_id=0,sample_rate_hz=1000,mapping=mapping),
               inputs={n:sha256(run/n) for n in ['baseline.raw.h5','events.jsonl','acquisition.json']})
    (run/'audit.json').write_text(canonical(audit))
    manifest=dict(accepted_run='accepted',files={'accepted/'+p.name:dict(sha256=sha256(p),size=p.stat().st_size) for p in run.iterdir()})
    (root/'manifest.json').write_text(canonical(manifest))
    return sha256(root/'manifest.json')


class RecordingReplayTests(unittest.TestCase):
    def setUp(self):
        self.module=importlib.import_module('agent.mea_replay')
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name);self.sha=fixture(self.root)

    def load(self):return self.module.AuditedRecording(self.root,manifest_sha256=self.sha)

    def test_true_windows_and_quiet_window_no_invented_stimulus(self):
        data=self.load();self.assertEqual(data.window_count,3)
        s=data.snapshot(0,'a');quiet=data.snapshot(1,'b');last=data.snapshot(2,'c')
        self.assertEqual(s.feature_values,(2.,1.,1.,15.))
        self.assertEqual(quiet.feature_values,(0.,0.,0.,0.))
        self.assertEqual(last.feature_values,(1.,.5,.5,30.))
        self.assertIsNone(s.to_dict()['stimulus_event_id'])
        self.assertEqual(s.to_dict()['association'],'sequential_offline')
        self.assertNotIn('stimulus_frame',s.to_dict())
        with self.assertRaises(ValueError):data.snapshot(3,'d')
        with self.assertRaises(ValueError):data.snapshot(-1,'d')
        self.assertEqual(data.summary()['tail_frames'],400)

    def test_source_hash_and_wrong_biological_hardware_fail(self):
        (self.root/'accepted/recording.jsonl').write_bytes(b'changed')
        with self.assertRaises(ValueError):self.load()

    def test_simulation_source_rejected_even_with_consistent_outer_hashes(self):
        other=self.root/'other';other.mkdir();sha=fixture(other,'simulation')
        with self.assertRaises(ValueError):self.module.AuditedRecording(other,manifest_sha256=sha)

    def test_wrong_chip_and_escaping_manifest_paths_fail(self):
        for mutation in ['chip','path']:
            with tempfile.TemporaryDirectory() as tmp:
                p=Path(tmp);fixture(p);m=json.loads((p/'manifest.json').read_text())
                if mutation=='path':m['accepted_run']='../escape'
                else:
                    a=p/'accepted/audit.json';d=json.loads(a.read_text());d['metadata']['chip_id']='22554';a.write_text(canonical(d))
                    m['files']['accepted/audit.json']=dict(size=a.stat().st_size,sha256=sha256(a))
                (p/'manifest.json').write_text(canonical(m))
                with self.assertRaises(ValueError):self.module.AuditedRecording(p,manifest_sha256=sha256(p/'manifest.json'))

    def test_immutable_roundtrip_and_future_isolation(self):
        d=self.load();a=d.snapshot(0,'x');d.snapshot(2,'y')
        self.assertEqual(d.snapshot(0,'x'),a)
        with self.assertRaises(dataclasses.FrozenInstanceError):a.context_id='y'
        payload=a.to_dict();self.assertEqual(self.module.RecordingSnapshot.from_dict(payload),a)
        payload['feature_values'][0]+=1
        with self.assertRaises(ValueError):self.module.RecordingSnapshot.from_dict(payload)

    def test_actor_and_critic_share_snapshot_and_binding(self):
        from tests.test_mea_p0 import observation
        from agent.mea_contract import encode,observation_key
        obs=observation();snapshot=self.load().snapshot(0,observation_key(obs))
        prepared=encode(obs,snapshot)
        self.assertEqual(prepared['contract'],'mea_rl_recording_v1')
        self.assertIn('离线记录',prepared['messages'][1]['content'])
        self.assertEqual(prepared['critic_input'][-4:],list(snapshot.normalized_values))
        self.assertEqual(len(prepared['critic_input']),112)
        with self.assertRaises(ValueError):encode(obs,self.load().snapshot(0,'wrong'))
