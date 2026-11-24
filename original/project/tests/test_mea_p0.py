"""P0 signal integrity, deterministic observation and native replay acceptance."""
import copy
import dataclasses
import importlib
import json
import math
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GODOT = ROOT / '.tools/godot/Godot_v4.7.2-stable_win64_console.exe'


def packets(context='context-a', source='simulation'):
    return [
        dict(type='session', protocol='mea.spikes.v1', session_id='session-a', seq=0,
             source=source, sample_rate_hz=1000, well_id=0, start_frame=0,
             channel_map_id='map-a', channels=[[0, 100], [1, 101]]),
        dict(type='stimulus', session_id='session-a', seq=1, event_id='event-a',
             frame=100, context_id=context, pattern_id='synthetic-a'),
        dict(type='spikes', session_id='session-a', seq=2,
             events=[[105, 0, -99.], [110, 0, -10.], [150, 1, -20.],
                     [200, 0, -30.], [210, 1, -88.]]),
        dict(type='watermark', session_id='session-a', seq=3, end_frame=300,
             complete=True, dropped=0),
    ]


def wire(rows):
    return b''.join((json.dumps(r, allow_nan=False) + '\n').encode() for r in rows)


def observation():
    from tests.test_m8s3_contract import observation as base
    obs = base()
    obs.update(profile='m9', world_version=7, environment_version='m9_regions_v1',
               map_mode='infinite_freeze', navigation_steps=4)
    return obs


class ProtocolTests(unittest.TestCase):
    def setUp(self):
        self.p = importlib.import_module('agent.mea_protocol')

    def test_legacy_csv_split_coalesced_units_and_no_device_clock(self):
        dec = self.p.LegacyCsvDecoder(time_unit='ms')
        self.assertEqual(dec.feed(b'1794800000123,42,-12'), [])
        result = dec.feed(b'5.89\r\n1794800000124,87,0\n')
        dec.finish()
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0].host_time_ms, 1794800000123)
        self.assertEqual(result[0].amplitude_uv, -125.89)
        self.assertIsNone(result[0].device_frame)
        seconds = self.p.LegacyCsvDecoder(time_unit='s')
        self.assertEqual(seconds.feed(b'1.25,0,-1\n')[0].host_time_ms, 1250)

    def test_bad_csv_and_truncated_eof_rejected(self):
        for row in (b'1,0,nan\n', b'1,-1,2\n', b'1,2\n', b'1,2,inf\n', b'1,x,0\n'):
            with self.subTest(row=row), self.assertRaises(ValueError):
                self.p.LegacyCsvDecoder().feed(row)
        dec = self.p.LegacyCsvDecoder()
        dec.feed(b'1,0,2')
        with self.assertRaises(ValueError): dec.finish()
        with self.assertRaises(ValueError):
            self.p.LegacyCsvDecoder(max_line_bytes=8).feed(b'123456789')

    def test_research_byte_splits_and_coalesced_packets(self):
        stream = self.p.ResearchStream()
        for byte in wire(packets()): stream.feed(bytes([byte]))
        stream.finish()
        self.assertEqual(stream.watermark, 300)
        self.assertEqual(len(stream.spikes), 5)
        self.assertEqual(stream.session_id, 'session-a')

    def test_missing_watermark_is_not_silence(self):
        stream = self.p.ResearchStream()
        stream.feed(wire(packets()[:-1]))
        self.assertEqual(stream.watermark, 0)

    def test_integrity_failures_poison_stream(self):
        mutations = [
            lambda r: r[2].update(seq=9),
            lambda r: r[3].update(dropped=1),
            lambda r: r[3].update(complete=False),
            lambda r: r[2].update(session_id='different-session'),
            lambda r: r[2].update(events=[[120, 55, -1]]),
            lambda r: r[2].update(events=[[120, 0, -1], [119, 1, -2]]),
            lambda r: r[2].update(events=[[120, 0, -1], [120, 0, -2]]),
        ]
        for mutate in mutations:
            rows = packets(); mutate(rows)
            stream = self.p.ResearchStream()
            with self.subTest(rows=rows), self.assertRaises(ValueError): stream.feed(wire(rows))
            self.assertTrue(stream.failed)
            with self.assertRaises(ValueError): stream.feed(b'')

    def test_late_spike_watermark_regression_and_duplicate_json_rejected(self):
        for extra in (
            dict(type='spikes', session_id='session-a', seq=4, events=[[299, 0, -1]]),
            dict(type='watermark', session_id='session-a', seq=4, end_frame=299, complete=True, dropped=0),
        ):
            stream = self.p.ResearchStream(); stream.feed(wire(packets()))
            with self.assertRaises(ValueError): stream.feed(wire([extra]))
        with self.assertRaises(ValueError): self.p.ResearchStream().feed(b'{"type":"x","type":"y"}\n')
        rows = packets(); rows[0]['sample_rate_hz'] = True
        with self.assertRaises(ValueError): self.p.ResearchStream().feed(wire(rows))

    def test_reconnect_and_limits_require_new_stream(self):
        stream = self.p.ResearchStream(); stream.feed(wire(packets()))
        with self.assertRaises(ValueError): stream.feed(wire(packets()))
        with self.assertRaises(ValueError): self.p.ResearchStream(max_spikes=2).feed(wire(packets()))
        stream = self.p.ResearchStream(); stream.feed(b'{"type":')
        with self.assertRaises(ValueError): stream.finish()


class SnapshotTests(unittest.TestCase):
    def setUp(self):
        self.p = importlib.import_module('agent.mea_protocol')
        self.s = importlib.import_module('agent.mea_snapshot')

    def snapshot(self, rows=None):
        stream = self.p.ResearchStream(); stream.feed(wire(rows or packets())); stream.finish()
        return self.s.build_snapshot(stream, 'event-a', blank_frames=10, window_frames=100,
                                     normalization=self.s.Normalization.identity())

    def test_half_open_window_features_and_frozen_nested_data(self):
        snap = self.snapshot()
        self.assertEqual(snap.feature_values, (3., 15., 1., 20.))
        self.assertEqual((snap.window_start_frame, snap.window_end_frame), (110, 210))
        self.assertEqual(snap.normalized_values, snap.feature_values)
        with self.assertRaises(dataclasses.FrozenInstanceError): snap.context_id = 'wrong'
        with self.assertRaises(TypeError): snap.feature_values[0] = 9
        data = snap.to_dict(); data['feature_values'][0] = 8
        self.assertEqual(snap.feature_values[0], 3.)

    def test_covered_quiet_window_zero_but_uncovered_raises(self):
        rows = packets(); rows[2]['events'] = []
        self.assertEqual(self.snapshot(rows).feature_values, (0., 0., 0., 0.))
        stream = self.p.ResearchStream(); stream.feed(wire(rows[:-1]))
        with self.assertRaises(self.s.IncompleteWindow):
            self.s.build_snapshot(stream, 'event-a', blank_frames=10, window_frames=100,
                                  normalization=self.s.Normalization.identity())

    def test_snapshot_roundtrip_and_tampering(self):
        snap = self.snapshot()
        self.assertEqual(snap, self.s.WetwareSnapshot.from_dict(json.loads(json.dumps(snap.to_dict()))))
        for field, value in [('feature_values', [99., 15., 1., 20.]), ('context_id', 'wrong'),
                             ('normalization_id', 'wrong'), ('source', 'live')]:
            data = snap.to_dict(); data[field] = value
            with self.subTest(field=field), self.assertRaises(ValueError): self.s.WetwareSnapshot.from_dict(data)

    def test_normalization_is_frozen_and_rejects_invalid_scales(self):
        for scale in ((0., 1., 1., 1.), (math.nan, 1., 1., 1.)):
            with self.assertRaises(ValueError): self.s.Normalization((0.,)*4, scale)
        n = self.s.Normalization((1., 5., 0., 10.), (2., 2., 1., 2.))
        stream = self.p.ResearchStream(); stream.feed(wire(packets()))
        snap = self.s.build_snapshot(stream, 'event-a', blank_frames=10, window_frames=100, normalization=n)
        self.assertEqual(snap.normalized_values, (1., 5., 1., 5.))


class ContractRewardTests(unittest.TestCase):
    def setUp(self):
        self.c = importlib.import_module('agent.mea_contract')
        self.p = importlib.import_module('agent.mea_protocol')
        self.s = importlib.import_module('agent.mea_snapshot')
        self.r = importlib.import_module('agent.mea_reward')

    def snapshot(self, obs):
        stream = self.p.ResearchStream(); stream.feed(wire(packets(self.c.observation_key(obs))))
        return self.s.build_snapshot(stream, 'event-a', blank_frames=10, window_frames=100,
                                     normalization=self.s.Normalization.identity())

    def test_state2_preserves_actions_and_extends_critic_without_mutation(self):
        from agent.m9_contract import encode as original
        obs = observation(); before = copy.deepcopy(obs); snap = self.snapshot(obs)
        row = self.c.encode(obs, snap)
        self.assertEqual(obs, before)
        self.assertEqual(row['branches'], original(obs, 'reserves')['branches'])
        self.assertEqual(row['verb_mask'], original(obs, 'reserves')['verb_mask'])
        self.assertEqual(row['contract'], 'mea_rl_p0_v1')
        self.assertEqual(len(row['critic_input']), 112)
        self.assertEqual(row['critic_input'][-4:], list(snap.normalized_values))
        self.assertIn('湿件观测', row['messages'][1]['content'])
        self.assertEqual(row, self.c.encode(json.loads(json.dumps(obs)), self.s.WetwareSnapshot.from_dict(snap.to_dict())))

    def test_wrong_observation_and_privileged_field_exclusion(self):
        obs = observation(); snap = self.snapshot(obs)
        dirty = copy.deepcopy(obs)
        dirty.update(reward='SECRET_R2', future='SECRET_R2', recommended_action='SECRET_R2')
        self.assertEqual(self.c.encode(dirty, snap), self.c.encode(obs, snap))
        self.assertNotIn('SECRET_R2', json.dumps(self.c.encode(dirty, snap)))
        changed = copy.deepcopy(obs); changed['player']['thirst'] = 13.
        with self.assertRaises(ValueError): self.c.encode(changed, snap)

    def test_r2_is_logged_but_reward_is_exactly_r1(self):
        for r1 in (-0.0, 0., -2.5, 1.e-200, 3.14):
            result = self.r.record_reward(r1, r2_raw=999., neural_metrics={'rate': 2.}, candidate_scores={'novelty': 7.})
            self.assertEqual(result['total_reward'], r1)
            self.assertEqual(math.copysign(1., result['total_reward']), math.copysign(1., r1))
            self.assertEqual(result['r2_raw'], 999.)
            self.assertFalse(result['r2_enabled']); self.assertFalse(result['r2_valid'])
            self.assertEqual(result['alpha'], 1.)
        self.assertIsNone(self.r.record_reward(1.)['r2_raw'])

    def test_alpha_and_nonfinite_cannot_bypass_guard(self):
        for alpha in (0., .9, 1.0001, math.nan, True, '1'):
            with self.subTest(alpha=alpha), self.assertRaises(ValueError): self.r.record_reward(1., alpha=alpha)
        for kwargs in ({'r1': math.inf}, {'r1': 1., 'r2_raw': math.nan},
                       {'r1': 1., 'neural_metrics': {'x': math.nan}}):
            with self.assertRaises(ValueError): self.r.record_reward(**kwargs)


class NativeSimulationTests(unittest.TestCase):
    @unittest.skipUnless(GODOT.exists(), 'native Godot executable unavailable')
    def test_scripted_native_simulation_replay_and_tamper(self):
        cli = importlib.import_module('tools.mea_p0')
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp)/'run'
            report = cli.simulate(out, godot=GODOT, project=ROOT, steps=8, seed=4100001)
            self.assertTrue(report['simulation_only'])
            self.assertEqual(report['decisions'], 8)
            self.assertEqual(report['alpha'], 1.)
            self.assertFalse(report['model_training_performed'])
            verified = cli.replay(out, godot=GODOT, project=ROOT)
            self.assertEqual(verified['verified_decisions'], 8)
            self.assertTrue(verified['passed'])
            with self.assertRaises(FileExistsError): cli.simulate(out, godot=GODOT, project=ROOT, steps=8, seed=4100001)
            path = out/'rollout.jsonl'; path.write_bytes(path.read_bytes()+b'{}\n')
            with self.assertRaises(ValueError): cli.replay(out, godot=GODOT, project=ROOT)

    @unittest.skipUnless(GODOT.exists(), 'native Godot executable unavailable')
    def test_replay_checks_content_even_with_updated_file_hashes(self):
        cli = importlib.import_module('tools.mea_p0')
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp)/'run'
            cli.simulate(out, godot=GODOT, project=ROOT, steps=3, seed=4100001)
            path = out/'rollout.jsonl'
            original = path.read_bytes()
            for mutation in ('actor', 'reward', 'reuse'):
                rows = [json.loads(line) for line in original.splitlines()]
                if mutation == 'actor': rows[0]['prepared']['messages'][1]['content'] += 'tampered'
                if mutation == 'reward': rows[0]['reward']['total_reward'] += 1
                if mutation == 'reuse': rows[1]['snapshot'] = rows[0]['snapshot']
                path.write_bytes(wire(rows))
                manifest = json.loads((out/'manifest.json').read_text())
                manifest['files']['rollout.jsonl'] = cli.sha(path)
                (out/'manifest.json').write_text(json.dumps(manifest), encoding='utf-8')
                with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                    cli.replay(out, godot=GODOT, project=ROOT)

    @unittest.skipUnless(GODOT.exists(), 'native Godot executable unavailable')
    def test_loss_retains_invalid_evidence_and_never_becomes_game_death(self):
        cli = importlib.import_module('tools.mea_p0')
        original = cli.SyntheticProducer.emit

        def loss(producer, row):
            if row['type'] == 'watermark': row = dict(row, dropped=1)
            return original(producer, row)

        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp)/'run'
            with patch.object(cli.SyntheticProducer, 'emit', loss), self.assertRaises(ValueError):
                cli.simulate(out, godot=GODOT, project=ROOT, steps=2, seed=4100001)
            failure = json.loads((out/'failure.json').read_text())
            self.assertEqual(failure['status'], 'invalid')
            self.assertEqual(failure['committed_decisions'], 0)
            self.assertFalse(failure['retry_performed'])
            self.assertFalse((out/'manifest.json').exists())
            self.assertEqual((out/'rollout.jsonl').read_text(), '')
            self.assertIn('"dropped":1', (out/'stream.jsonl').read_text())
            with self.assertRaises(ValueError): cli.replay(out, godot=GODOT, project=ROOT)

    def test_bad_config_rejected_before_creating_output(self):
        cli = importlib.import_module('tools.mea_p0')
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp)/'run'
            with self.assertRaises(ValueError): cli.simulate(out, godot=GODOT, project=ROOT, steps=2, seed=1, alpha=.9)
            self.assertFalse(out.exists())


if __name__ == '__main__': unittest.main()
