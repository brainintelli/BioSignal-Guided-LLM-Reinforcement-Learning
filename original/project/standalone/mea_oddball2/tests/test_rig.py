"""Routing and the device rig with the real vendored SDK; only the socket is fake."""
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

import numpy as np

from oddball2 import design, sites as geometry
from oddball2.session import Stop
from support import CHANNELS, ELECTRODES, LAYOUT, SITE_CHANNELS, write_h5

ROUTE_TEXT = ';'.join('%d(%d)' % (c, e) for c, e in enumerate(LAYOUT))


class Wire:
    """Answers the MaxLab text protocol and keeps a software array: selection, route and stimulation attachments.

    trouble maps an electrode to how connecting it goes wrong: 'error' (the server refuses),
    'none' (no unit), 'shared' (the unit of an earlier electrode) or 'amplifier' (unit on another channel).
    """

    def __init__(self, route=''):
        self.commands = []
        self.lsb = '2.981901'
        self.fail = None
        self.trouble = {}
        self.first_unit = 5
        self.selected, self.attached, self.unit_of = {}, {}, {}
        self.load(route)

    def load(self, text):
        self.text = text
        self.mapping = {int(e): int(c) for c, e in re.findall(r'(\d+)\((\d+)\)', text)}
        self.attached, self.unit_of = {}, {}

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
        if command == 'wellplate_query_version':
            return '0'
        if command.startswith('mea_array_command '):
            verb, _, argument = command.split('\n', 1)[1].partition(' ')
            answer = self.array(verb.removeprefix('mea_array_'), argument)
            return 'OK' if answer is None else answer
        return 'OK'

    def array(self, verb, argument):
        if verb == 'reset':
            self.selected, self.attached, self.unit_of = {}, {}, {}
        elif verb == 'clear_selected_electrodes':
            self.selected = {}
        elif verb == 'select_electrodes':
            for token in argument.split():
                electrode, weight = map(int, token.split('/'))
                self.selected[electrode] = max(weight, self.selected.get(electrode, 0))
        elif verb == 'route':
            routed = sorted(sorted(self.selected, key=lambda e: (-self.selected[e], e))[:1024])
            self.load(';'.join('%d(%d)' % (c, e) for c, e in enumerate(routed)))
        elif verb == 'set_config':
            self.load(argument)
        elif verb == 'get_config':
            return self.text
        elif verb == 'connect_electrode_to_stimulation':
            electrode, mode = int(argument), self.trouble.get(int(argument))
            if mode == 'error' or electrode not in self.mapping:
                return 'ERROR electrode cannot be connected'
            if mode == 'none':
                return 'OK'
            free = [u for u in range(self.first_unit, 32) if u not in self.attached]
            unit = min(self.attached) if mode == 'shared' else free[0]
            self.unit_of[electrode] = unit
            if mode != 'shared':
                self.attached[unit] = self.mapping[electrode] + (500 if mode == 'amplifier' else 0)
        elif verb == 'disconnect_amplifier_from_stimulation':
            self.attached = {u: c for u, c in self.attached.items() if c != int(argument)}
            self.unit_of = {e: u for e, u in self.unit_of.items() if self.mapping[e] != int(argument)}
        elif verb == 'query_stimulation_at_electrode':
            return str(self.unit_of.get(int(argument), -1))
        elif verb == 'query_amplifier_at_stimulation':
            return str(self.attached.get(int(argument), ''))


class WiredCase(unittest.TestCase):
    def setUp(self):
        import route_sites
        import run_session
        from oddball2 import rig
        from mea_stim_v2 import live
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        route = root / 'route.cfg'
        route.write_text(ROUTE_TEXT)
        self.wire = Wire(ROUTE_TEXT)
        self.binding = dict(
            chip_id='chip-test', organoid_id='org-1', session_id='sess-1', device_hostname='test-host',
            device_model='MaxOne', acquisition_host='192.0.2.101', replicate=0, record_root=str(root),
            started_at=(datetime.now(timezone.utc) - timedelta(minutes=10)).isoformat(),
            threshold_h5_dataset='settings/spike_threshold', threshold_value=5, bad_channels=[],
            electrodes=ELECTRODES, route_file=str(route), route_sha256=hashlib.sha256(route.read_bytes()).hexdigest(),
            material='Pt-Black', ceiling_mv=600)
        home = root / 'home'
        (home / '.local/state/mea_photo_full').mkdir(parents=True)
        self.alarms = []
        fake_fcntl = SimpleNamespace(flock=lambda *args: None, LOCK_EX=1, LOCK_NB=2)
        stack = ExitStack()
        self.addCleanup(stack.close)
        stack.enter_context(patch('socket.socket', side_effect=AssertionError('REAL SOCKET FORBIDDEN')))
        stack.enter_context(patch.object(live.transport.socket, 'create_connection', self.wire.connect))
        stack.enter_context(patch.object(live.platform, 'system', return_value='Linux'))
        stack.enter_context(patch.object(live.socket, 'gethostname', return_value='test-host'))
        stack.enter_context(patch.object(live, '_server_snapshot', return_value=(42, [])))
        stack.enter_context(patch.object(rig.subprocess, 'check_output', return_value='192.0.2.101 192.0.2.2\n'))
        stack.enter_context(patch.object(rig.time, 'sleep', lambda seconds: None))
        stack.enter_context(patch.dict(sys.modules, fcntl=fake_fcntl))
        stack.enter_context(patch.object(run_session, 'signal', SimpleNamespace(alarm=self.alarms.append)))
        stack.enter_context(patch.object(Path, 'home', return_value=home))
        self.disk = stack.enter_context(patch.object(run_session.shutil, 'disk_usage',
                                                     return_value=SimpleNamespace(free=10 ** 12)))
        self.run_session, self.route_sites, self.rig_module, self.root = run_session, route_sites, rig, root

    def since(self, start, prefix):
        return [c for c in self.wire.commands[start:] if c.startswith(prefix)]

    def write(self, path, data):
        path.write_text(json.dumps(data), encoding='utf-8')


class RigTests(WiredCase):
    def setUp(self):
        super().setUp()
        out = self.root / 'oddball2-rig'
        out.mkdir()
        self.out = out
        self.rig = self.rig_module.MaxOneRig(self.binding, dict(software_units=None), out)
        self.plan = design.plan(5, 600)
        self.block = next(b for b in self.plan['blocks'] if b['stage'] == 'main')
        self.rehearsal = self.plan['blocks'][0]

    def appended(self, start, token):
        return [c.split('\n', 1)[1] for c in self.wire.commands[start:] if c.startswith('sequence_append ' + token + '\n')]

    def test_nothing_is_sent_to_the_device_before_prepare(self):
        self.assertEqual(self.wire.commands, [])

    def test_prepare_gives_every_electrode_its_own_unit_and_reports_the_actual_amplitudes(self):
        info = self.rig.prepare()
        self.assertEqual(info['units'], dict(H=[5, 6, 7], W=[8, 9, 10], F=[11, 12, 13]))
        self.assertEqual(info['channels'], SITE_CHANNELS)
        self.assertEqual(self.rig.site_channels, SITE_CHANNELS)
        self.assertAlmostEqual(info['actual_mv']['300'], 298.1901)
        self.assertAlmostEqual(info['actual_mv']['450'], 447.28515)
        self.assertAlmostEqual(info['actual_mv']['600'], 599.362101)
        self.assertFalse(self.since(0, 'sequence_'))
        units = self.since(0, 'mea_set_stimulation_unit ')
        self.assertTrue(all(c.split()[2:4] == ['0', '0'] for c in units))
        self.assertEqual(self.since(0, 'mea_set_dac '), ['mea_set_dac %d 512 512' % d for d in range(3)])
        self.assertIn(ROUTE_TEXT, (self.out / 'downloaded.cfg').read_text())

    def test_only_the_amplitudes_up_to_the_ceiling_are_reported(self):
        self.binding['ceiling_mv'] = 450
        self.assertEqual(sorted(self.rig.prepare()['actual_mv']), ['300', '450'])

    def test_unrepresentable_dac_step_stops_before_any_configuration(self):
        self.wire.lsb = '900'
        with self.assertRaises(ValueError):
            self.rig.prepare()
        self.assertFalse(self.since(0, 'system_initialize'))

    def test_electrodes_that_cannot_each_have_a_unit_stop_before_the_route_is_downloaded(self):
        for mode in ('shared', 'none', 'error', 'amplifier'):
            with self.subTest(mode=mode):
                self.wire.commands.clear()
                self.wire.trouble = {ELECTRODES['W'][1]: mode}
                rig = self.rig_module.MaxOneRig(self.binding, {}, self.out)
                with self.assertRaises((ValueError, RuntimeError)):
                    rig.prepare()
                self.assertFalse([c for c in self.wire.commands if 'mea_array_download' in c])
                rig.cleanup()

    def test_a_failed_preparation_blocks_every_recording(self):
        self.wire.trouble = {ELECTRODES['F'][2]: 'amplifier'}
        with self.assertRaises(ValueError):
            self.rig.prepare()
        with self.assertRaises(RuntimeError):
            self.rig.open('warmup')
        self.assertFalse(self.since(0, 'saving_'))
        self.assertEqual(self.rig.cleanup(), [])

    def test_a_changed_route_file_or_site_definition_is_refused(self):
        Device = self.rig_module.Device
        for change in (dict(electrodes=dict(ELECTRODES, H=ELECTRODES['H'] + ELECTRODES['W'][:1] * 3)),   # six, one repeated
                       dict(electrodes=dict(ELECTRODES, H=[])),
                       dict(electrodes=dict(ELECTRODES, W=ELECTRODES['H'])),                            # shared between sites
                       dict(electrodes=dict(ELECTRODES, F=[25000])),                                    # not in the route
                       dict(electrodes={k: v for k, v in ELECTRODES.items() if k != 'F'}),
                       dict(route_sha256='0' * 64), dict(material='Au'), dict(device_model='MaxTwo')):
            with self.subTest(change=sorted(change)):
                made = None
                with self.assertRaises(ValueError):
                    made = Device(dict(self.binding, **change), self.out)
                    made.preflight()
        self.assertEqual(self.wire.commands, [])

    def test_block_is_built_as_ten_chunks_and_nothing_runs_until_sent(self):
        self.rig.prepare()
        self.rig.open(self.block['name'])
        start = len(self.wire.commands)
        handles = self.rig.build(self.block, 201)
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
                    expected += [f'mea_set_dac {dac} 311 512', 'system_delay_samples 4', f'mea_set_dac {dac} 713 512',
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
        event = dict(id=8000, block=self.block['name'], real=9)
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

    def test_units_bind_every_electrode_of_a_site_to_that_sites_dac_at_neutral(self):
        self.rig.prepare()
        start = len(self.wire.commands)
        self.rig.units(True)
        self.assertEqual(self.wire.commands[start:], [
            'mea_set_dac 0 512 512', 'mea_set_dac 1 512 512', 'mea_set_dac 2 512 512',
            *['mea_set_stimulation_unit %d 1 1 0 0 0 0' % u for u in (5, 6, 7)],
            *['mea_set_stimulation_unit %d 1 1 0 0 1 0' % u for u in (8, 9, 10)],
            *['mea_set_stimulation_unit %d 1 1 0 0 2 0' % u for u in (11, 12, 13)]])
        start = len(self.wire.commands)
        self.rig.units(False)
        self.assertEqual(self.wire.commands[start + 3:], [
            *['mea_set_stimulation_unit %d 0 0 0 0 0 0' % u for u in (5, 6, 7)],
            *['mea_set_stimulation_unit %d 0 0 0 0 1 0' % u for u in (8, 9, 10)],
            *['mea_set_stimulation_unit %d 0 0 0 0 2 0' % u for u in (11, 12, 13)]])

    def test_changed_dac_step_blocks_building(self):
        self.rig.prepare()
        self.rig.open(self.block['name'])
        self.wire.lsb = '3.1'
        start = len(self.wire.commands)
        with self.assertRaises(RuntimeError):
            self.rig.build(self.block, 100)
        self.assertEqual(self.since(start, 'sequence_new '), [])

    def test_an_attachment_that_changes_later_blocks_the_next_recording(self):
        self.rig.prepare()
        self.wire.attached[20] = 40                 # a unit nobody asked for
        with self.assertRaises(ValueError):
            self.rig.open('warmup')
        self.assertFalse(self.since(0, 'saving_'))

    def test_lost_send_acknowledgement_latches_a_fault_and_cleanup_turns_everything_off(self):
        self.rig.prepare()
        self.rig.open(self.block['name'])
        self.rig.units(True)
        handles = self.rig.build(self.block, 100)
        self.wire.fail = lambda c: c.startswith('sequence_send ')
        booked = []
        with self.assertRaises(TimeoutError):
            self.rig.send(handles[0], dict(id=8000, real=9), booked.append)
        self.assertEqual(len(booked), 1)              # the dose is booked although the outcome is unknown
        self.assertTrue(self.rig.faulted)
        with self.assertRaises(RuntimeError):
            self.rig.send(handles[1], dict(id=8010, real=9), booked.append)
        self.wire.fail = None
        start = len(self.wire.commands)
        self.assertEqual(self.rig.cleanup(), [])
        tail = self.wire.commands[start:]
        self.assertEqual(len([c for c in tail if re.fullmatch(r'mea_set_stimulation_unit \d+ 0 0 0 0 0 0', c)]), 32)
        self.assertIn('saving_stop_file', tail)
        self.assertEqual(len([c for c in tail if c.startswith('sequence_delete ')]), 10)

    def test_recordings_are_checked_against_the_binding_and_the_site_positions(self):
        self.rig.prepare()
        empty = (np.empty(0, dtype='u8'), np.empty(0, dtype='i4'))
        good, other = self.root / 'good.raw.h5', self.root / 'other.raw.h5'
        write_h5(good, 100, 40000, [], empty, [])
        write_h5(other, 100, 40000, [], empty, [], chip='chip-other')
        with self.rig.read(good) as rec:
            self.rig.verify(rec)
            moved = dict(rec.channel_xy_um)
            moved[SITE_CHANNELS['W'][1]] = (moved[SITE_CHANNELS['W'][1]][0] + 17.5, moved[SITE_CHANNELS['W'][1]][1])
            rec.channel_xy_um = moved
            with self.assertRaises(ValueError):
                self.rig.verify(rec)
        with self.rig.read(other) as rec:
            with self.assertRaises(ValueError):
                self.rig.verify(rec)


class StageCase(WiredCase):
    """A stage directory and the quiet baseline of the simulated chip, as quiet_baseline.py leaves them."""

    def setUp(self):
        super().setUp()
        from mea_stim_v2.recording import H5Recording
        from select_sites import candidates
        self.stage = self.root / 'stage'
        self.stage.mkdir()
        self.original = {k: v for k, v in self.binding.items()
                         if k not in ('electrodes', 'route_file', 'route_sha256', 'material', 'ceiling_mv')}
        self.write(self.stage / 'baseline-binding.json', self.original)
        folder = self.root / 'sess-1-site-selection'
        folder.mkdir()
        rng = np.random.default_rng(3)
        frames = 62 * 20000
        n = 2 * 62 * CHANNELS
        spikes = (1000 + rng.integers(0, frames, n).astype('u8'), rng.integers(0, CHANNELS, n).astype('i4'))
        noise = [(c, 1000 + t * 20000, (512 + np.rint(rng.normal(0, 2, 20000))).astype('u2'))
                 for c in range(CHANNELS) for t in (3, 23, 43, 57)]
        write_h5(folder / 'quiet.raw.h5', 1000, frames, [], spikes, [], noise=noise)
        with H5Recording(folder / 'quiet.raw.h5', SimpleNamespace(threshold_h5_dataset='settings/spike_threshold')) as rec:
            self.selection = candidates(rec, 'chip-test', [])
        self.write(folder / 'site-candidates.json', self.selection)
        self.folder = folder
        self.wire.load('')                          # nothing is routed until route_sites asks for it

    def plan_route(self, per_site=3):
        with contextlib.redirect_stdout(io.StringIO()) as printed:
            self.route_sites.plan(self.stage, per_site)
        self.printed = printed.getvalue()
        return json.loads((self.stage / 'session-route.json').read_text())


class RouteTests(StageCase):
    def test_sites_get_their_centre_and_nearest_neighbours_each_with_its_own_unit_in_software_only(self):
        route = self.plan_route(5)
        stable = {r['electrode'] for r in self.selection['candidates']}
        units = [u for s in design.SITES for u in route['software_units'][s]]
        self.assertEqual(len(set(units)), 15)
        for site in design.SITES:
            centre, *others = route['electrodes'][site]
            self.assertIn(centre, stable)
            self.assertEqual(others, geometry.neighbours(centre)[:4])
            self.assertEqual(route['sites'][site]['coordinates'][0], list(geometry.electrode_xy(centre)))
            self.assertGreaterEqual(route['sites'][site]['routed_near'], geometry.MIN_NEAR)
        self.assertGreaterEqual(min(route['selection']['distances_um'].values()), geometry.MIN_SEPARATION_UM)
        self.assertEqual((route['hardware_downloaded'], route['strategy'], route['attempts'], route['unroutable']),
                         (False, 0, 1, []))
        cfg = Path(route['route_file'])
        self.assertEqual(hashlib.sha256(cfg.read_bytes()).hexdigest(), route['route_sha256'])
        routed = {int(e) for _, e in re.findall(r'(\d+)\((\d+)\)', cfg.read_text())}
        self.assertTrue({e for s in design.SITES for e in route['electrodes'][s]} <= routed)
        # Software only: nothing is downloaded, no unit is powered, no sequence exists, no attachment is left.
        for forbidden in ('mea_array_download', 'mea_set_stimulation_unit', 'sequence_', 'system_initialize',
                          'mea_set_dac', 'saving_'):
            self.assertFalse([c for c in self.wire.commands if forbidden in c], forbidden)
        self.assertEqual(self.wire.attached, {})
        self.assertTrue(any(c.startswith('mea_array_delete ') for c in self.wire.commands))
        self.assertIn('site H: centre', self.printed)

    def test_an_unroutable_neighbour_is_replaced_by_the_next_nearest(self):
        first = self.plan_route(3)
        for path in (self.stage / 'session-route.json', self.stage / 'sample-route.cfg'):
            path.unlink()
        centre = first['electrodes']['W'][0]
        nearest = geometry.neighbours(centre)
        for mode in ('error', 'none', 'shared', 'amplifier'):
            with self.subTest(mode=mode):
                self.wire.trouble = {nearest[0]: mode}
                route = self.plan_route(3)
                self.assertEqual(route['electrodes']['W'], [centre, nearest[1], nearest[2]])
                self.assertEqual(route['electrodes']['H'], first['electrodes']['H'])
                self.assertEqual((route['attempts'], route['unroutable']), (2, [nearest[0]]))
                self.assertEqual(len({u for s in design.SITES for u in route['software_units'][s]}), 9)
                for path in (self.stage / 'session-route.json', self.stage / 'sample-route.cfg'):
                    path.unlink()

    def test_a_site_falls_back_to_fewer_electrodes_down_to_its_centre(self):
        first = self.plan_route(1)
        self.assertEqual([len(first['electrodes'][s]) for s in design.SITES], [1, 1, 1])
        for path in (self.stage / 'session-route.json', self.stage / 'sample-route.cfg'):
            path.unlink()
        centre = first['electrodes']['F'][0]
        self.wire.trouble = {e: 'none' for e in geometry.neighbours(centre)}
        route = self.plan_route(5)
        self.assertEqual(route['electrodes']['F'], [centre])
        self.assertEqual(len(route['electrodes']['H']), 5)
        self.assertEqual(len(route['electrodes']['W']), 5)
        self.assertLessEqual(route['attempts'], self.route_sites.MAX_ATTEMPTS)

    def test_an_unroutable_centre_or_an_existing_route_writes_nothing(self):
        centre = self.plan_route(1)['electrodes']['H'][0]
        with self.assertRaises(ValueError):          # a second route for the same stage
            self.plan_route(1)
        for path in (self.stage / 'session-route.json', self.stage / 'sample-route.cfg'):
            path.unlink()
        self.wire.trouble = {centre: 'error'}
        with self.assertRaises(ValueError):
            self.plan_route(3)
        self.assertFalse((self.stage / 'session-route.json').exists())
        self.assertFalse((self.stage / 'sample-route.cfg').exists())
        self.assertEqual(len([c for c in self.wire.commands if c.startswith('mea_array_delete ')]), 1 + 2)
        for bad in (0, 6, 2.5):
            with self.assertRaises(ValueError):
                self.route_sites.plan(self.stage, bad)

    def test_a_baseline_that_does_not_reproduce_is_refused_before_the_instrument_is_asked(self):
        changed = json.loads(json.dumps(self.selection))
        changed['candidates'][0]['min_spikes_per_20s'] += 1
        self.write(self.folder / 'site-candidates.json', changed)
        with self.assertRaises(ValueError):
            self.plan_route(3)
        self.assertEqual(self.wire.commands, [])


class RunTests(StageCase):
    """`run` from the stage files to end.json. The fake wire records no data, so the
    session must end at the first closed-file read, with the device cleaned up."""

    def setUp(self):
        super().setUp()
        self.route = self.plan_route(3)
        self.wire.commands.clear()

    def go(self, output='oddball2-v1', ceiling=600):
        with contextlib.redirect_stdout(io.StringIO()):
            self.run_session.run(self.stage, output, 7, ceiling)
        return json.loads((self.root / output / 'end.json').read_text(encoding='utf-8'))

    def test_run_wires_the_routed_stage_to_the_session_and_always_cleans_up(self):
        end = self.go()
        out = self.root / 'oddball2-v1'
        self.assertEqual(end['outcome'], 'error')            # no acquisition server behind the wire, so no H5
        self.assertIn('warmup', end['error'])
        self.assertEqual(end['dose'], 0)
        self.assertEqual(end['cleanup'], [])
        self.assertEqual(json.loads((out / 'plan.json').read_text()), design.plan(7, 600))
        made = json.loads((out / 'binding.json').read_text())
        self.assertEqual((made['evidence_mode'], made['material'], made['electrodes'], made['ceiling_mv']),
                         ('oddball2_habituation', 'Pt-Black', self.route['electrodes'], 600))
        self.assertEqual(made['electrodes_per_site'], dict(H=3, W=3, F=3))
        prepared = json.loads((out / 'route.json').read_text())
        self.assertEqual(prepared['units'], self.route['software_units'])     # the session reproduces the planned units
        self.assertEqual(prepared['electrodes'], self.route['electrodes'])
        self.assertEqual(json.loads((out / 'manifest.json').read_text()), {})
        self.assertFalse((out / 'dose.jsonl').exists())
        self.assertFalse(self.since(0, 'sequence_'))
        self.assertEqual(len(self.alarms), 2)
        self.assertTrue(6470 <= self.alarms[0] <= 6480)        # sample loaded 10 min ago, alarm at minute 118
        self.assertEqual(self.alarms[1], 0)
        # During the warmup all nine units were connected at neutral, each on its site's DAC.
        on = [c for c in self.wire.commands if re.fullmatch(r'mea_set_stimulation_unit \d+ 1 1 0 0 [012] 0', c)]
        self.assertEqual(len(on), 9)
        self.assertEqual([int(c.split()[6]) for c in on], [0, 0, 0, 1, 1, 1, 2, 2, 2])
        self.assertFalse([c for c in self.wire.commands if c.startswith('mea_set_dac ') and not c.endswith(' 512 512')])
        last_units = self.since(0, 'mea_set_stimulation_unit ')[-32:]
        self.assertTrue(all(c.split()[2:4] == ['0', '0'] for c in last_units))
        self.assertTrue(any(c.startswith('mea_array_delete ') for c in self.wire.commands))
        self.assertIn('saving_stop_file', self.wire.commands)

    def test_a_run_that_ended_clean_before_any_pulse_may_be_followed_by_another(self):
        self.go('oddball2-v1')
        start = len(self.wire.commands)
        end = self.go('oddball2-v2')
        self.assertEqual((end['dose'], end['cleanup']), (0, []))
        self.assertTrue(self.since(start, 'system_initialize'))
        older = self.root / 'oddball2-v1'
        # A rehearsal books chunks without pulses; those do not count either.
        (older / 'dose.jsonl').write_text('{"real": 0, "total": 0}\n{"real": 0, "total": 0}\n')
        self.go('oddball2-v3')
        cases = dict(
            pulses=lambda: (older / 'dose.jsonl').write_text('{"real": 0, "total": 0}\n{"real": 9, "total": 9}\n'),
            torn_ledger=lambda: (older / 'dose.jsonl').write_text('{"real": 0, "total": 0}\n{"real"'),
            unknown_end=lambda: (older / 'end.json').unlink(),
            dirty_device=lambda: self.write(older / 'end.json', dict(outcome='error', cleanup=['off 3: timeout'])))
        kept = (older / 'end.json').read_bytes()
        for number, (name, case) in enumerate(cases.items()):
            with self.subTest(case=name):
                (older / 'dose.jsonl').write_text('')
                (older / 'end.json').write_bytes(kept)
                case()
                start = len(self.wire.commands)
                with self.assertRaises(ValueError):
                    self.go('oddball2-again%d' % number)
                self.assertEqual(self.wire.commands[start:], [])

    def test_the_plan_follows_the_ceiling_given_on_the_command_line(self):
        self.go('oddball2-low', ceiling=300)
        self.assertEqual(json.loads((self.root / 'oddball2-low' / 'plan.json').read_text())['ladder_mv'], [300])
        self.assertEqual(json.loads((self.root / 'oddball2-low' / 'binding.json').read_text())['ceiling_mv'], 300)

    def test_preconditions_refuse_before_touching_the_device(self):
        centre = self.route['electrodes']['H'][0]
        far = centre + 5 if (centre % geometry.COLUMNS) < 200 else centre - 5
        def other_sample(): self.write(self.stage / 'session-route.json', dict(self.route, organoid_id='org-2'))
        def other_baseline(): self.write(self.stage / 'session-route.json', dict(self.route, baseline=dict(path='elsewhere')))
        def stray_electrode():
            self.write(self.stage / 'session-route.json',
                       dict(self.route, electrodes=dict(self.route['electrodes'], H=[centre, far])))
        def six_electrodes():
            self.write(self.stage / 'session-route.json',
                       dict(self.route, electrodes=dict(self.route['electrodes'], H=[centre] + geometry.neighbours(centre)[:5])))
        def unstable_centre():
            thin = dict(self.selection, candidates=[r for r in self.selection['candidates'] if r['electrode'] != centre])
            self.write(self.folder / 'site-candidates.json', thin)
        def close_sites():
            self.write(self.stage / 'session-route.json', dict(self.route, electrodes=dict(
                self.route['electrodes'], W=[next(e for e in sorted(r['electrode'] for r in self.selection['candidates'])
                                                  if e != centre and 100 < _gap(e, centre) < 450)])))
        def too_late():
            b = dict(self.original, started_at=(datetime.now(timezone.utc) - timedelta(minutes=36)).isoformat())
            self.write(self.stage / 'baseline-binding.json', b)
        def earlier_dose(): (self.root / 'older-run').mkdir(); (self.root / 'older-run' / 'dose.jsonl').write_text('{}')
        def earlier_attempt(): (self.root / 'attempt-01.json').write_text('{}')
        def full_disk(): self.disk.return_value = SimpleNamespace(free=79_000_000_000)
        cases = [other_sample, other_baseline, stray_electrode, six_electrodes, unstable_centre, close_sites, too_late,
                 earlier_dose, earlier_attempt, full_disk]
        for number, case in enumerate(cases):
            with self.subTest(case=case.__name__):
                self.write(self.stage / 'session-route.json', self.route)
                self.write(self.stage / 'baseline-binding.json', self.original)
                self.write(self.folder / 'site-candidates.json', self.selection)
                self.disk.return_value = SimpleNamespace(free=10 ** 12)
                case()
                with self.assertRaises(ValueError):
                    self.go('oddball2-case%d' % number)
                self.assertEqual(self.wire.commands, [])
                self.assertFalse((self.root / ('oddball2-case%d' % number)).exists())
                for leftover in (self.root / 'older-run' / 'dose.jsonl', self.root / 'attempt-01.json'):
                    leftover.unlink(missing_ok=True)
                if (self.root / 'older-run').is_dir():
                    (self.root / 'older-run').rmdir()
        self.write(self.stage / 'session-route.json', self.route)
        self.write(self.stage / 'baseline-binding.json', self.original)
        self.write(self.folder / 'site-candidates.json', self.selection)
        for name in ('oddball-v1', 'crossover-v9', 'sub/oddball2-v1'):
            with self.assertRaises(ValueError):
                self.go(name)
        with self.assertRaises(ValueError):
            self.go('oddball2-v2', ceiling=500)
        self.assertEqual(self.wire.commands, [])


def _gap(a, b):
    (ax, ay), (bx, by) = geometry.electrode_xy(a), geometry.electrode_xy(b)
    return ((ax - bx) ** 2 + (ay - by) ** 2) ** .5


if __name__ == '__main__':
    unittest.main()
