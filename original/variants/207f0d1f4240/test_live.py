"""Real vendored SDK, with every network boundary replaced by a fake socket."""
from contextlib import ExitStack
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from mea_stim_v2.config import Config


class Clock:
    def __init__(self): self.now = 1000.
    def wall(self): return 1794896000. + self.now
    def mono(self): return self.now
    def sleep(self, value): self.now += value


class Wire:
    def __init__(self, route, clock):
        self.route, self.clock = route, clock
        self.commands, self.timeouts = [], []
        self.fail = None
        self.lsb = '2.9'
        self.error_reply = None
        self.connections = 0
    def connect(self, address, timeout):
        assert address == ('127.0.0.1', 7215)
        self.connections += 1
        self.timeouts.append(timeout)
        outer = self
        class Connection:
            def __enter__(self): return self
            def __exit__(self, *args): pass
            def settimeout(self, value): pass
            def sendall(self, payload):
                self.command = payload.removesuffix(b'\r\n###').decode('latin1')
                outer.commands.append(self.command)
            def recv(self, size):
                outer.clock.now += .01
                command = self.command
                if outer.fail and outer.fail(command): raise TimeoutError('lost acknowledgement')
                if command.startswith('sequence_new '): result = command.split(' ', 1)[1]
                elif command == 'system_query_dac_lsb_mv': result = outer.lsb
                elif command.endswith('mea_array_get_config'): result = outer.route
                elif '\nmea_array_query_stimulation_at_electrode ' in command: result = '7'
                else: result = 'OK'
                if outer.error_reply and outer.error_reply(command): result = 'Error: rejected command'
                return result.encode('latin1') + b'\r\n###'
        return Connection()


class LiveTests(unittest.TestCase):
    def setUp(self):
        from mea_stim_v2 import live
        self.live = live
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        route = self.root / 'route.cfg'
        route.write_text(''.join(f'{i}({1000+i})' for i in range(60)))
        self.config = Config(mode='live', session_id='test-session', chip_id='28073',
            organoid_id='test-org', started_utc='2026-11-17T06:13:00Z',
            device_hostname='test-acquisition', route_file=str(route),
            route_sha256=hashlib.sha256(route.read_bytes()).hexdigest(), stim_electrode=1059,
            threshold_value=5., threshold_h5_dataset='settings/spike_threshold', live_authorized=True)
        self.config = replace(self.config, authorization_sha256=self.config.binding_sha256())
        self.clock = Clock()
        self.wire = Wire(route.read_text(), self.clock)
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(patch('socket.socket', side_effect=AssertionError('REAL SOCKET FORBIDDEN')))
        self.stack.enter_context(patch.object(live.transport.socket, 'create_connection', self.wire.connect))
        self.stack.enter_context(patch.object(live.platform, 'system', return_value='Linux'))
        self.stack.enter_context(patch.object(live.socket, 'gethostname', return_value='test-acquisition'))
        self.snapshot = self.stack.enter_context(patch.object(live, '_server_snapshot', return_value=(42, [])))
        self.backend = live.MaxLabBackend(self.config, self.root, self.clock)

    def prepared(self):
        self.backend.preflight()
        self.backend.prepare()

    def recording(self):
        self.prepared()
        self.backend.open_record('pair_01')
        self.backend.connect(True)

    def test_unauthorized_or_wrong_host_never_opens_network_even_during_cleanup(self):
        self.assertEqual(self.wire.connections, 0)
        for config in (replace(self.config, live_authorized=False), replace(self.config, device_hostname='wrong')):
            backend = self.live.MaxLabBackend(config, self.root, self.clock)
            with self.assertRaises(Exception): backend.preflight()
            self.assertEqual(backend.cleanup(), [])
        with patch.object(self.live.platform, 'system', return_value='Windows'):
            with self.assertRaises(Exception): self.backend.preflight()
        self.assertEqual(self.wire.connections, 0)

    def test_preflight_is_read_only_and_rejects_an_open_h5_before_query(self):
        self.snapshot.return_value = (42, ['/other/session.raw.h5'])
        with self.assertRaises(RuntimeError): self.backend.preflight()
        self.assertEqual(self.wire.commands, [])
        self.snapshot.return_value = (42, [])
        result = self.backend.preflight()
        self.assertEqual(result['server_pid'], 42)
        self.assertEqual(result['lsb_mv'], 2.9)
        self.assertGreater(result['free_bytes'], 0)
        self.assertEqual(self.wire.commands, ['system_query_dac_lsb_mv'])

    def test_prepare_only_once_and_keeps_all_units_off(self):
        self.prepared()
        self.assertEqual(self.backend.stim_channel, 59)
        self.assertEqual(self.backend.stim_unit, 7)
        units = [c for c in self.wire.commands if c.startswith('mea_set_stimulation_unit ')]
        self.assertEqual(len(units), 32)
        self.assertTrue(all(c.split()[2:4] == ['0', '0'] for c in units))
        with self.assertRaises(RuntimeError): self.backend.prepare()
        self.assertEqual(self.wire.commands.count('system_initialize'), 1)

    def test_failed_initialization_still_turns_off_all_units_and_zeros_dac(self):
        self.backend.preflight()
        self.wire.fail = lambda c: c == 'system_initialize'
        with self.assertRaises(TimeoutError): self.backend.prepare()
        start = len(self.wire.commands)
        self.wire.fail = None
        self.assertEqual(self.backend.cleanup(), [])
        commands = self.wire.commands[start:]
        self.assertEqual(len([c for c in commands if c.startswith('mea_set_stimulation_unit ')]), 32)
        self.assertEqual(commands[32], 'mea_set_dac 0 512 512')

    def test_single_pulse_and_event_share_frame_without_future_pulses(self):
        self.recording()
        self.backend.pulse({'id': 9001, 'arm': 'true', 'pair': 1}, 2.9)
        append = [c.split('\n', 1)[1] for c in self.wire.commands if c.startswith('sequence_append ')]
        self.assertEqual(append[0], 'system_delay_samples 100')
        self.assertTrue(append[1].startswith('system_trigger_event 0 1 9001 '))
        self.assertEqual(json.loads(append[1].split(' ', 4)[4]), {'id': 9001, 'arm': 'true', 'pair': 1, 'session': 'test-session'})
        self.assertEqual(append[2:], ['mea_set_dac 0 409 512', 'system_delay_samples 4',
            'mea_set_dac 0 615 512', 'system_delay_samples 4', 'mea_set_dac 0 512 512'])
        self.assertEqual(sum(c.startswith('sequence_send ') for c in self.wire.commands), 1)
        self.assertGreater(self.backend.last_api_seconds, 0)

    def test_sham_is_zero_and_each_new_file_clears_events_before_append(self):
        self.recording()
        self.backend.marker(8001, {'phase': 'connected'})
        self.backend.pulse({'id': 9002, 'arm': 'sham', 'pair': 1}, 2.9)
        for command in self.wire.commands:
            if command.startswith('sequence_append ') and '\nmea_set_dac' in command:
                self.assertTrue(command.endswith('mea_set_dac 0 512 512'))
        result = self.backend.close_record()
        self.assertEqual(result['path'], str(self.root / 'pair_01' / 'pair_01.raw.h5'))
        self.assertIn('closed_utc', result)
        with self.assertRaises(FileExistsError): self.backend.open_record('pair_01')
        self.backend.open_record('pair_02')
        self.backend.marker(8002, {'phase': 'connected'})
        clears = [i for i, c in enumerate(self.wire.commands) if c == 'system_clear_events']
        appends = [i for i, c in enumerate(self.wire.commands) if c.startswith('sequence_append ')]
        self.assertEqual(len(clears), 2)
        self.assertLess(clears[0], appends[0])
        self.assertLess(clears[1], appends[-1])

    def test_route_or_server_change_prevents_output(self):
        self.recording()
        for change in ('route', 'server'):
            if change == 'route': self.wire.route = self.wire.route.replace('59(1059)', '59(2059)')
            else:
                self.wire.route = Path(self.config.route_file).read_text()
                self.snapshot.return_value = (99, [])
            with self.assertRaises(RuntimeError): self.backend.pulse({'id': 9001, 'arm': 'true', 'pair': 1}, 2.9)
        self.assertFalse(any(c.startswith('sequence_send ') for c in self.wire.commands))

    def test_lost_sequence_ack_is_not_retried_and_cleanup_owns_token(self):
        self.recording()
        self.wire.fail = lambda c: c.startswith('sequence_new ')
        with self.assertRaises(TimeoutError): self.backend.marker(8001, {'phase': 'connected'})
        token = next(c.split(' ', 1)[1] for c in self.wire.commands if c.startswith('sequence_new '))
        self.wire.fail = None
        self.backend.cleanup()
        self.assertEqual(sum(c.startswith('sequence_new ') for c in self.wire.commands), 1)
        self.assertIn('sequence_delete ' + token, self.wire.commands)

    def test_lost_start_file_ack_is_closed_during_cleanup(self):
        self.prepared()
        self.wire.fail = lambda c: c.startswith('saving_start_file ')
        with self.assertRaises(TimeoutError): self.backend.open_record('ambiguous')
        self.wire.fail = None
        self.backend.cleanup()
        self.assertIn('saving_stop_file', self.wire.commands)

    def test_audit_failure_blocks_normal_output_but_not_independent_cleanup(self):
        self.recording()
        start = len(self.wire.commands)
        with patch.object(self.backend, '_audit', side_effect=OSError('disk full')):
            with self.assertRaises(OSError): self.backend.marker(8001, {'phase': 'connected'})
            self.assertEqual(len(self.wire.commands), start)
            self.wire.fail = lambda c: c.startswith('mea_set_stimulation_unit 0 ')
            errors = self.backend.cleanup()
        self.assertTrue(errors)
        tail = self.wire.commands[start:]
        self.assertEqual(len([c for c in tail if c.startswith('mea_set_stimulation_unit ')]), 32)
        self.assertIn('mea_set_dac 0 512 512', tail)
        self.assertIn('saving_stop_file', tail)
        self.assertTrue(any(c.startswith('mea_array_delete ') for c in tail))
        self.assertTrue(all(t <= 1 for t in self.wire.timeouts[-len(tail):]))
        self.assertLessEqual(len(tail) * 2, 120)

    def test_api_audit_has_durable_request_and_ack_or_error(self):
        self.backend.preflight()
        rows = [json.loads(s) for s in (self.root / 'api_audit.jsonl').read_text().splitlines()]
        self.assertEqual([r['kind'] for r in rows], ['request', 'ack'])
        self.assertEqual(rows[0]['command'], 'system_query_dac_lsb_mv')
        self.wire.fail = lambda c: c == 'system_initialize'
        with self.assertRaises(TimeoutError): self.backend.prepare()
        rows = [json.loads(s) for s in (self.root / 'api_audit.jsonl').read_text().splitlines()]
        self.assertEqual(rows[-1]['kind'], 'error')

    def test_changed_dac_step_is_read_before_every_pulse_and_prevents_send(self):
        self.recording()
        self.wire.lsb = '3.1'
        with self.assertRaises(RuntimeError): self.backend.pulse({'id': 9001, 'arm': 'true', 'pair': 1}, 2.9)
        self.assertEqual(self.wire.commands.count('system_query_dac_lsb_mv'), 2)
        self.assertFalse(any(c.startswith('sequence_send ') for c in self.wire.commands))

    def test_api_rejection_propagates_and_latches_no_more_normal_requests(self):
        self.recording()
        self.wire.error_reply = lambda c: c.startswith('sequence_send ')
        with self.assertRaises(RuntimeError): self.backend.pulse({'id': 9001, 'arm': 'true', 'pair': 1}, 2.9)
        before = len(self.wire.commands)
        self.wire.error_reply = None
        with self.assertRaises(RuntimeError): self.backend.pulse({'id': 9002, 'arm': 'true', 'pair': 1}, 2.9)
        self.assertEqual(len(self.wire.commands), before)
        self.assertEqual(self.backend.cleanup(), [])

    def test_lost_array_ack_does_not_lose_array_cleanup_ownership(self):
        self.backend.preflight()
        self.wire.fail = lambda c: c.startswith('mea_array_new ')
        with self.assertRaises(TimeoutError): self.backend.prepare()
        token = next(c.split(' ', 1)[1] for c in self.wire.commands if c.startswith('mea_array_new '))
        self.wire.fail = None
        self.backend.cleanup()
        self.assertIn('mea_array_delete ' + token, self.wire.commands)

    def test_only_selected_unit_is_enabled_and_zero_comes_first(self):
        self.prepared()
        before = len(self.wire.commands)
        self.backend.connect(True)
        tail = self.wire.commands[before:]
        self.assertEqual(tail[-2:], ['mea_set_dac 0 512 512', 'mea_set_stimulation_unit 7 1 1 0 0 0 0'])

    def test_old_sequence_is_retired_before_next_creation_and_after_close(self):
        self.recording()
        for i in range(4): self.backend.marker(8001 + i, {'phase': 'marker'})
        self.assertLessEqual(len(self.backend.sequences), 2)
        tokens = [c.split(' ', 1)[1] for c in self.wire.commands if c.startswith('sequence_new ')]
        for first, second in zip(tokens, tokens[1:]):
            self.assertLess(self.wire.commands.index('sequence_delete ' + first), self.wire.commands.index('sequence_new ' + second))
        self.backend.close_record()
        self.assertLess(self.wire.commands.index('saving_stop_file'), self.wire.commands.index('sequence_delete ' + tokens[-1]))
        self.assertEqual(self.backend.sequences, [])

    def test_send_gate_can_abort_after_build_without_sending(self):
        self.recording()
        def deny(event):
            self.assertEqual(event['id'], 9001)
            self.assertTrue(self.wire.commands[-1].endswith('\nmea_set_dac 0 512 512'))
            self.assertGreater(self.backend.last_api_seconds, 0)
            raise RuntimeError('deadline reserve exhausted')
        self.backend.before_sequence_send = deny
        with self.assertRaises(RuntimeError): self.backend.pulse({'id': 9001, 'arm': 'true', 'pair': 1}, 2.9)
        self.assertFalse(any(c.startswith('sequence_send ') for c in self.wire.commands))
        # A local budget/manual gate is not a broken hardware connection;
        # zero-output disconnect and recovery recording must remain possible.
        self.backend.connect(False)
        self.assertEqual(self.backend.cleanup(), [])

    def test_send_gate_precedes_send_and_receipt_exposes_actual_host_send_time(self):
        self.recording()
        witnessed = []
        self.backend.before_sequence_send = lambda event: witnessed.append((len(self.wire.commands), self.clock.mono()))
        receipt = self.backend.pulse({'id': 9001, 'arm': 'true', 'pair': 1}, 2.9)
        index, mono = witnessed[0]
        self.assertTrue(self.wire.commands[index].startswith('sequence_send '))
        self.assertEqual(receipt['send_mono'], mono)
        self.assertEqual(self.backend.sent_monotonic_actual, mono)
        self.assertEqual(self.backend.sent_event['id'], 9001)


if __name__ == '__main__': unittest.main()
