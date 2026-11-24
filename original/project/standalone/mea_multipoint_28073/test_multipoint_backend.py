"""Offline only: real SDK commands over a socket fake that checks key/value events."""
from pathlib import Path
import sys
import unittest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'mea_stimulation_v2'))
sys.path.insert(0, str(HERE.parent / 'mea_stimulation_v2' / 'tests'))
import test_live as fixtures


class MultipointTests(unittest.TestCase):
    def setUp(self):
        fixtures.LiveTests.setUp(self)
        from multipoint_backend import MultipointBackend
        self.backend = MultipointBackend(self.config, self.root, self.clock)
        self.wire.lsb = '2.981901'

    def recording(self):
        self.backend.preflight()
        self.backend.prepare()
        self.backend.open_record('point_01')
        self.backend.connect(True)

    def events(self):
        return [dict(id=10001+g*3+j, arm='true', pair=1, phase='multipoint', point=1,
                     electrode=1059, group=g+1, pulse=j+1, session='test-session')
                for g in range(5) for j in range(3)]

    def test_exact_fifteen_onsets_phase_width_and_finite_end(self):
        self.recording()
        result = self.backend.train(self.events(), 2.981901)
        commands = [c.split('\n', 1)[1] for c in self.wire.commands if c.startswith('sequence_append ')]
        cursor = 0
        onsets, dac = [], []
        for command in commands:
            if command.startswith('system_delay_samples '): cursor += int(command.split()[1])
            elif command.startswith('system_trigger_event '): onsets.append(cursor)
            elif command.startswith('mea_set_dac '): dac.append((cursor, int(command.split()[2])))
            else: self.fail('Unexpected sequence command ' + command)
        self.assertEqual(onsets, [100, 300, 500, 4100, 4300, 4500, 8100, 8300, 8500,
                                  12100, 12300, 12500, 16100, 16300, 16500])
        self.assertEqual(cursor, 16508)
        self.assertEqual(len(dac), 45)
        for onset, start in zip(onsets, range(0, 45, 3)):
            self.assertEqual(dac[start:start+3], [(onset, 412), (onset+4, 612), (onset+8, 512)])
        self.assertEqual(len(self.wire.event_messages), 15)
        self.assertEqual(self.wire.event_messages[-1], dict(id='10015', arm='true', pair='1',
            phase='multipoint', point='1', electrode='1059', group='5', pulse='3', session='test-session'))
        self.assertEqual(sum(c.startswith('sequence_send ') for c in self.wire.commands), 1)
        self.assertEqual(result['duration_samples'], 16508)
        self.assertAlmostEqual(result['duration_s'], .8254)
        self.assertEqual(len(result['events']), 15)
        self.assertGreaterEqual(self.clock.mono() - result['send_mono'], .826)
        self.assertEqual(self.backend.cleanup(), [])

    def test_invalid_count_order_identity_sham_or_duplicate_cannot_build(self):
        self.recording()
        cases = [self.events()[:-1]]
        for key, value in [('arm', 'sham'), ('group', 0), ('pulse', 0), ('pair', 2),
                           ('point', 21), ('electrode', 1058), ('session', 'wrong'), ('phase', 'other')]:
            events = self.events(); events[1][key] = value; cases.append(events)
        events = self.events(); events[1]['id'] = events[0]['id']; cases.append(events)
        for events in cases:
            with self.subTest(events=events[:2]):
                with self.assertRaises(ValueError): self.backend.train(events, 2.981901)
        self.assertFalse(any(c.startswith('sequence_new ') for c in self.wire.commands))

    def test_hook_is_once_immediately_before_send_and_can_block_whole_train(self):
        self.recording()
        called = []
        def deny(event):
            called.append(event)
            self.assertTrue(self.wire.commands[-1].endswith('\nmea_set_dac 0 512 512'))
            self.assertGreater(self.backend.last_api_seconds, 0)
            raise RuntimeError('budget exhausted')
        self.backend.before_sequence_send = deny
        with self.assertRaises(RuntimeError): self.backend.train(self.events(), 2.981901)
        self.assertEqual(len(called), 1)
        self.assertEqual(called[0]['id'], 10001)
        self.assertFalse(any(c.startswith('sequence_send ') for c in self.wire.commands))
        self.assertEqual(self.backend.cleanup(), [])

    def test_lost_constructor_ack_keeps_owned_token_and_never_retries(self):
        self.recording()
        self.wire.fail = lambda c: c.startswith('sequence_new ')
        with self.assertRaises(TimeoutError): self.backend.train(self.events(), 2.981901)
        token = next(c.split(' ', 1)[1] for c in self.wire.commands if c.startswith('sequence_new '))
        self.wire.fail = None
        with self.assertRaises(RuntimeError): self.backend.train(self.events(), 2.981901)
        self.backend.cleanup()
        self.assertEqual(sum(c.startswith('sequence_new ') for c in self.wire.commands), 1)
        self.assertIn('sequence_delete ' + token, self.wire.commands)

    def test_changed_dac_cannot_construct_train(self):
        self.recording()
        self.wire.lsb = '3.1'
        with self.assertRaises(RuntimeError): self.backend.train(self.events(), 2.981901)
        self.assertFalse(any(c.startswith('sequence_new ') for c in self.wire.commands))

    def test_lost_send_ack_preserves_actual_attempt_and_blocks_retry(self):
        self.recording()
        self.wire.fail = lambda c: c.startswith('sequence_send ')
        with self.assertRaises(TimeoutError): self.backend.train(self.events(), 2.981901)
        self.assertIsNotNone(self.backend.sent_monotonic_actual)
        self.assertEqual(self.backend.sent_event['id'], 10001)
        self.wire.fail = None
        with self.assertRaises(RuntimeError): self.backend.train(self.events(), 2.981901)
        self.assertEqual(sum(c.startswith('sequence_send ') for c in self.wire.commands), 1)
        self.assertEqual(self.backend.cleanup(), [])

    def test_completed_train_cannot_be_sent_again_with_fresh_ids(self):
        self.recording()
        self.backend.train(self.events(), 2.981901)
        events = self.events()
        for event in events: event['id'] += 100
        with self.assertRaises(RuntimeError): self.backend.train(events, 2.981901)
        self.assertEqual(sum(c.startswith('sequence_send ') for c in self.wire.commands), 1)


if __name__ == '__main__': unittest.main()
