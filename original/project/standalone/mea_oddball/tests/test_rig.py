"""The device rig with the real vendored SDK and existing controller; only the socket is fake."""
import contextlib
from contextlib import ExitStack
from datetime import datetime, timedelta, timezone
import hashlib
import io
import json
from pathlib import Path
import re
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from oddball import design
from oddball.session import Stop

ELECTRODES = {'H': 1010, 'W': 1020, 'F': 1030}
UNITS = {1010: 5, 1020: 9, 1030: 21}


class Wire:
    """Answers the MaxLab text protocol and keeps the stimulation attachments it is told about."""

    def __init__(self, route):
        self.route = route
        self.commands = []
        self.attached = {}
        self.lsb = '2.981901'
        self.fail = None

    def connect(self, address, timeout):
        assert address == ('127.0.0.1', 7215)
        outer = self

        class Connection:
            def __enter__(self): return self
            def __exit__(self, *args): pass
            def settimeout(self, value): pass
            def sendall(self, payload):
                self.command = payload.removesuffix(b'\r\n###').decode('latin1')
                outer.commands.append(self.command)
            def recv(self, size):
                if outer.fail and outer.fail(self.command):
                    raise TimeoutError('lost acknowledgement')
                return outer.reply(self.command).encode('latin1') + b'\r\n###'
        return Connection()

    def reply(self, command):
        if command.startswith('sequence_new '):
            return command.split(' ', 1)[1]
        if command == 'system_query_dac_lsb_mv':
            return self.lsb
        if command.startswith('mea_array_command '):
            verb, _, argument = command.split('\n', 1)[1].partition(' ')
            if verb == 'mea_array_get_config':
                return self.route
            if verb == 'mea_array_connect_electrode_to_stimulation':
                self.attached[UNITS[int(argument)]] = int(argument) - 1000
            if verb == 'mea_array_disconnect_amplifier_from_stimulation':
                self.attached = {u: c for u, c in self.attached.items() if c != int(argument)}
            if verb == 'mea_array_query_stimulation_at_electrode':
                return str(UNITS[int(argument)])
            if verb == 'mea_array_query_amplifier_at_stimulation':
                return str(self.attached.get(int(argument), ''))
        return 'OK'


class WiredCase(unittest.TestCase):
    def setUp(self):
        import run_session
        from livepair import device
        from mea_stim_v2 import live
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        route = root / 'route.cfg'
        route.write_text(''.join(f'{i}({1000 + i})' for i in range(60)))
        self.wire = Wire(route.read_text())
        self.binding = dict(
            chip_id='chip-1', organoid_id='org-1', session_id='sess-1', device_hostname='test-host',
            device_model='MaxOne', acquisition_host='192.0.2.101', replicate=0, record_root=str(root),
            started_at=(datetime.now(timezone.utc) - timedelta(minutes=10)).isoformat(),
            threshold_h5_dataset='settings/spike_threshold', threshold_value=5, bad_channels=[],
            electrodes=ELECTRODES, route_file=str(route), route_sha256=hashlib.sha256(route.read_bytes()).hexdigest(),
            material='Pt-Black', prior_nonzero_attempts=0)
        stack = ExitStack()
        self.addCleanup(stack.close)
        stack.enter_context(patch('socket.socket', side_effect=AssertionError('REAL SOCKET FORBIDDEN')))
        stack.enter_context(patch.object(live.transport.socket, 'create_connection', self.wire.connect))
        stack.enter_context(patch.object(live.platform, 'system', return_value='Linux'))
        stack.enter_context(patch.object(live.socket, 'gethostname', return_value='test-host'))
        stack.enter_context(patch.object(live, '_server_snapshot', return_value=(42, [])))
        stack.enter_context(patch.object(device.subprocess, 'check_output', return_value='192.0.2.101 192.0.2.2\n'))
        stack.enter_context(patch.object(device.time, 'sleep', lambda seconds: None))
        self.run_session, self.root = run_session, root

    def since(self, start, prefix):
        return [c for c in self.wire.commands[start:] if c.startswith(prefix)]


class RigTests(WiredCase):
    def setUp(self):
        super().setUp()
        out = self.root / 'oddball-rig'
        out.mkdir()
        self.rig = self.run_session.MaxOneRig(self.binding, dict(coordinates={}), out)
        self.plan = design.plan(5)
        self.block = next(b for b in self.plan['blocks'] if b['stage'] == 'main')
        self.rehearsal = self.plan['blocks'][0]

    def appended(self, start, token):
        return [c.split('\n', 1)[1] for c in self.wire.commands[start:] if c.startswith('sequence_append ' + token + '\n')]

    def test_nothing_is_sent_to_the_device_before_prepare(self):
        self.assertEqual(self.wire.commands, [])

    def test_prepare_routes_three_units_and_reports_the_actual_amplitudes(self):
        info = self.rig.prepare()
        self.assertEqual(info['units'], dict(H=5, W=9, F=21))
        self.assertEqual(info['channels'], dict(H=10, W=20, F=30))
        self.assertAlmostEqual(info['actual_mv']['300'], 298.1901)
        self.assertAlmostEqual(info['actual_mv']['400'], 399.574734)
        self.assertFalse(self.since(0, 'sequence_'))
        units = self.since(0, 'mea_set_stimulation_unit ')
        self.assertTrue(all(c.split()[2:4] == ['0', '0'] for c in units))

    def test_unrepresentable_dac_step_stops_before_any_configuration(self):
        self.wire.lsb = '900'
        with self.assertRaises(ValueError):
            self.rig.prepare()
        self.assertFalse(self.since(0, 'system_initialize'))

    def test_block_is_built_as_ten_chunks_and_nothing_runs_until_sent(self):
        self.rig.prepare()
        self.rig.open(self.block['name'])
        start = len(self.wire.commands)
        handles = self.rig.build(self.block, 100)
        tokens = [c.split(' ', 1)[1] for c in self.since(start, 'sequence_new ')]
        self.assertEqual(len(handles), 10)
        self.assertEqual(len(set(tokens)), 10)
        self.assertEqual(self.since(start, 'sequence_send '), [])
        self.assertEqual(self.since(start, 'mea_set_dac '), [])       # no direct DAC write, only queued ones
        for token, trials in zip(tokens, design.chunks(self.block)):
            lines = self.appended(start, token)
            self.assertEqual(lines[0], 'system_delay_samples 100')
            expected = []
            for position, i in enumerate(trials):
                site = self.block['trials'][i]
                number = self.block['index'] * 1000 + i
                expected.append(f'system_trigger_event 0 1 {number} blk {self.block["index"]} i {i} session sess-1 site {site or "Z"}')
                if site:
                    dac = 'HWF'.index(site)
                    expected += [f'mea_set_dac {dac} 412 512', 'system_delay_samples 4', f'mea_set_dac {dac} 612 512',
                                 'system_delay_samples 4', f'mea_set_dac {dac} 512 512']
                if position < 9:
                    expected.append('system_delay_samples %d' % (39992 if site else 40000))
            self.assertEqual(lines[1:], expected)

    def test_send_runs_the_gate_first_and_a_refusal_sends_nothing(self):
        self.rig.prepare()
        self.rig.open(self.block['name'])
        handles = self.rig.build(self.block, 100)
        start = len(self.wire.commands)
        seen = []
        event = dict(id=6000, block=self.block['name'], real=9)
        self.rig.send(handles[0], event, lambda e: seen.append((dict(e), len(self.since(start, 'sequence_send ')))))
        self.assertEqual(seen, [(event, 0)])
        self.assertEqual(self.since(start, 'sequence_send '), ['sequence_send ' + handles[0].token])

        def refuse(e):
            raise Stop('budget')
        with self.assertRaises(Stop):
            self.rig.send(handles[1], event, refuse)
        self.assertEqual(len(self.since(start, 'sequence_send ')), 1)
        self.assertFalse(self.rig.faulted)
        self.assertIsNone(self.rig.dev.before_sequence_send)
        self.rig.units(False)                       # still controllable after a refusal
        self.rig.close()
        for handle in handles:
            self.assertIn('sequence_delete ' + handle.token, self.wire.commands)
        self.assertEqual(self.rig.cleanup(), [])

    def test_rehearsal_queues_only_neutral_dac_codes(self):
        self.rig.prepare()
        self.rig.open('rehearsal')
        start = len(self.wire.commands)
        handles = self.rig.build(self.rehearsal, 0)
        self.assertEqual(len(handles), 4)
        queued = [c.split('\n', 1)[1] for c in self.since(start, 'sequence_append ') if '\nmea_set_dac ' in c]
        self.assertTrue(queued)
        self.assertTrue(all(re.fullmatch(r'mea_set_dac [012] 512 512', c) for c in queued))

    def test_units_bind_each_site_to_its_own_dac_at_neutral(self):
        self.rig.prepare()
        start = len(self.wire.commands)
        self.rig.units(True)
        self.assertEqual(self.wire.commands[start:], [
            'mea_set_dac 0 512 512', 'mea_set_dac 1 512 512', 'mea_set_dac 2 512 512',
            'mea_set_stimulation_unit 5 1 1 0 0 0 0', 'mea_set_stimulation_unit 9 1 1 0 0 1 0',
            'mea_set_stimulation_unit 21 1 1 0 0 2 0'])
        start = len(self.wire.commands)
        self.rig.units(False)
        self.assertEqual(self.wire.commands[start + 3:], [
            'mea_set_stimulation_unit 5 0 0 0 0 0 0', 'mea_set_stimulation_unit 9 0 0 0 0 1 0',
            'mea_set_stimulation_unit 21 0 0 0 0 2 0'])

    def test_changed_dac_step_blocks_building(self):
        self.rig.prepare()
        self.rig.open(self.block['name'])
        self.wire.lsb = '3.1'
        start = len(self.wire.commands)
        with self.assertRaises(RuntimeError):
            self.rig.build(self.block, 100)
        self.assertEqual(self.since(start, 'sequence_new '), [])

    def test_lost_send_acknowledgement_latches_a_fault_and_cleanup_turns_everything_off(self):
        self.rig.prepare()
        self.rig.open(self.block['name'])
        self.rig.units(True)
        handles = self.rig.build(self.block, 100)
        self.wire.fail = lambda c: c.startswith('sequence_send ')
        booked = []
        with self.assertRaises(TimeoutError):
            self.rig.send(handles[0], dict(id=6000, real=9), booked.append)
        self.assertEqual(len(booked), 1)              # the dose is booked although the outcome is unknown
        self.assertTrue(self.rig.faulted)
        with self.assertRaises(RuntimeError):
            self.rig.send(handles[1], dict(id=6010, real=9), booked.append)
        self.wire.fail = None
        start = len(self.wire.commands)
        self.assertEqual(self.rig.cleanup(), [])
        tail = self.wire.commands[start:]
        self.assertEqual(len([c for c in tail if re.fullmatch(r'mea_set_stimulation_unit \d+ 0 0 0 0 0 0', c)]), 32)
        self.assertIn('saving_stop_file', tail)
        self.assertEqual(len([c for c in tail if c.startswith('sequence_delete ')]), 10)


class RunTests(WiredCase):
    """`run` from the stage files to end.json. The fake wire records no data, so the
    session must end at the first closed-file read, with the device cleaned up."""

    def setUp(self):
        super().setUp()
        run_session, root = self.run_session, self.root
        self.stage = root / 'stage'
        self.stage.mkdir()
        original = {k: v for k, v in self.binding.items()
                    if k not in ('electrodes', 'route_file', 'route_sha256', 'material', 'prior_nonzero_attempts')}
        self.route = dict(chip_id='chip-1', organoid_id='org-1', session_id='sess-1', baseline=dict(path='quiet'),
                          electrodes=ELECTRODES, coordinates=dict(H=[0, 0], W=[500, 0], F=[0, 500]),
                          route_file=self.binding['route_file'], route_sha256=self.binding['route_sha256'])
        self.write(self.stage / 'baseline-binding.json', original)
        self.write(self.stage / 'session-route.json', self.route)
        selection = root / 'sess-1-site-selection'
        selection.mkdir()
        self.write(selection / 'site-candidates.json', dict(
            chip='chip-1', source=dict(path='quiet'), suggested={s: dict(electrode=e) for s, e in ELECTRODES.items()}))
        home = root / 'home'
        (home / '.local/state/mea_photo_full').mkdir(parents=True)
        self.alarms = []
        fake_signal = SimpleNamespace(alarm=self.alarms.append)
        fake_fcntl = SimpleNamespace(flock=lambda *args: None, LOCK_EX=1, LOCK_NB=2)
        stack = ExitStack()
        self.addCleanup(stack.close)
        stack.enter_context(patch.dict(sys.modules, fcntl=fake_fcntl))
        stack.enter_context(patch.object(run_session, 'signal', fake_signal))
        stack.enter_context(patch.object(run_session.Path, 'home', return_value=home))
        self.disk = stack.enter_context(patch.object(run_session.shutil, 'disk_usage',
                                                     return_value=SimpleNamespace(free=10 ** 12)))

    def write(self, path, data):
        path.write_text(json.dumps(data), encoding='utf-8')

    def go(self, output='oddball-v1'):
        with contextlib.redirect_stdout(io.StringIO()):
            self.run_session.run(self.stage, output, 7)
        return json.loads((self.root / output / 'end.json').read_text(encoding='utf-8'))

    def test_run_wires_stage_files_to_the_session_and_always_cleans_up(self):
        end = self.go()
        out = self.root / 'oddball-v1'
        self.assertEqual(end['outcome'], 'error')            # no acquisition server behind the wire, so no H5
        self.assertIn('warmup', end['error'])
        self.assertEqual(end['dose'], 0)
        self.assertEqual(end['cleanup'], [])
        self.assertEqual(json.loads((out / 'plan.json').read_text()), design.plan(7))
        made = json.loads((out / 'binding.json').read_text())
        self.assertEqual((made['evidence_mode'], made['material'], made['electrodes']),
                         ('oddball_habituation', 'Pt-Black', ELECTRODES))
        self.assertEqual(json.loads((out / 'route.json').read_text())['units'], dict(H=5, W=9, F=21))
        self.assertEqual(json.loads((out / 'manifest.json').read_text()), {})
        self.assertFalse((out / 'dose.jsonl').exists())
        self.assertFalse(self.since(0, 'sequence_'))
        self.assertEqual(len(self.alarms), 2)
        self.assertTrue(6470 <= self.alarms[0] <= 6480)        # sample loaded 10 min ago, alarm at minute 118
        self.assertEqual(self.alarms[1], 0)
        last_units = self.since(0, 'mea_set_stimulation_unit ')[-32:]
        self.assertTrue(all(c.split()[2:4] == ['0', '0'] for c in last_units))
        self.assertTrue(any(c.startswith('mea_array_delete ') for c in self.wire.commands))
        self.assertIn('saving_stop_file', self.wire.commands)

    def test_preconditions_refuse_before_touching_the_device(self):
        def other_sample(): self.write(self.stage / 'session-route.json', dict(self.route, organoid_id='org-2'))
        def other_electrodes(): self.write(self.stage / 'session-route.json', dict(self.route, electrodes=dict(ELECTRODES, H=1011)))
        def too_late():
            b = json.loads((self.stage / 'baseline-binding.json').read_text())
            b['started_at'] = (datetime.now(timezone.utc) - timedelta(minutes=36)).isoformat()
            self.write(self.stage / 'baseline-binding.json', b)
        def earlier_dose(): (self.root / 'older-run').mkdir(); (self.root / 'older-run' / 'dose.jsonl').write_text('{}')
        def earlier_attempt(): (self.root / 'attempt-01.json').write_text('{}')
        def full_disk(): self.disk.return_value = SimpleNamespace(free=79_000_000_000)
        cases = [other_sample, other_electrodes, too_late, earlier_dose, earlier_attempt, full_disk]
        for number, case in enumerate(cases):
            with self.subTest(case=case.__name__):
                self.write(self.stage / 'session-route.json', self.route)
                self.disk.return_value = SimpleNamespace(free=10 ** 12)
                case()
                with self.assertRaises(ValueError):
                    self.go('oddball-case%d' % number)
                self.assertEqual(self.wire.commands, [])
                self.assertFalse((self.root / ('oddball-case%d' % number)).exists())
                for leftover in (self.root / 'older-run' / 'dose.jsonl', self.root / 'attempt-01.json'):
                    leftover.unlink(missing_ok=True)
                if case is too_late:
                    b = json.loads((self.stage / 'baseline-binding.json').read_text())
                    b['started_at'] = self.binding['started_at']
                    self.write(self.stage / 'baseline-binding.json', b)
        with self.assertRaises(ValueError):
            self.go('crossover-v9')
        self.assertEqual(self.wire.commands, [])


if __name__ == '__main__':
    unittest.main()
