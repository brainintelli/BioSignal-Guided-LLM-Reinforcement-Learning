"""Exercise the actual Runner.run branches without subprocesses, signals or sockets."""
from contextlib import ExitStack
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import multipoint
import multipoint_backend


class Clock:
    def __init__(self):
        self.wall_start = datetime(2026, 11, 17, 6, 58, tzinfo=timezone.utc).timestamp()
        self.elapsed = 0.
    def wall(self): return self.wall_start + self.elapsed
    def mono(self): return self.elapsed
    def sleep(self, seconds): self.elapsed += seconds


class Signals:
    SIGALRM, SIGTERM, SIGINT, ITIMER_REAL = 14, 15, 2, 0
    SIG_IGN = object()
    def __init__(self): self.handlers, self.timers = {}, []
    def getsignal(self, sig): return self.handlers.get(sig, self.SIG_IGN)
    def signal(self, sig, handler):
        old = self.getsignal(sig)
        self.handlers[sig] = handler
        return old
    def setitimer(self, which, duration): self.timers.append((which, duration))
    def deliver(self, sig):
        handler = self.getsignal(sig)
        if callable(handler): handler(sig, None)


class Backend:
    def __init__(self, scenario, config, run_dir, clock):
        self.scenario, self.config, self.folder, self.clock = scenario, config, Path(run_dir), clock
        self.point = config.stim_electrode - 999
        self.before_sequence_send = None
        self.active = None
        self.cleanups = []
        self.sent_events = []
        self.send_mono = None
        scenario.devices.append(self)
    def preflight(self):
        self.scenario.operations.append((self.point, 'preflight'))
        return {'lsb_mv': 2.981901, 'free_bytes': 10**15, 'server_pid': 42}
    def prepare(self):
        self.scenario.operations.append((self.point, 'prepare'))
        self.clock.sleep(10)
    def connect(self, enabled): self.scenario.operations.append((self.point, 'connect', enabled))
    def open_record(self, name):
        if self.active is not None: raise AssertionError('Concurrent recording')
        folder = self.folder / name
        folder.mkdir()
        self.active = {'name': name, 'point': self.point, 'start_mono': self.clock.mono(),
                       'path': str(folder / (name + '.raw.h5'))}
        Path(self.active['path']).write_bytes(b'fake closed H5; analyzer is the boundary double')
        self.scenario.operations.append((self.point, 'open', name))
    def marker(self, event_id, properties):
        self.scenario.markers.append((self.point, self.active['name'], event_id, dict(properties)))
    def train(self, events, lsb_mv):
        if self.before_sequence_send: self.before_sequence_send(events[0])
        self.sent_events = [dict(event) for event in events]
        self.send_mono = self.clock.mono()
        self.scenario.trains.append((self.point, self.sent_events, self.send_mono))
        if self.scenario.lost_ack_point == self.point:
            raise TimeoutError('lost train acknowledgement')
        self.clock.sleep(.826)
        return {'send_mono': self.send_mono, 'events': self.sent_events, 'duration_s': .8254}
    def close_record(self):
        if self.active is None: raise AssertionError('No recording')
        record = {**self.active, 'closed_mono': self.clock.mono(),
                  'recorded_seconds': self.clock.mono() - self.active['start_mono']}
        self.scenario.records.append(record)
        self.active = None
        return record
    def cleanup(self):
        self.cleanups.append('all32off')
        if self.scenario.signal_cleanup_point == self.point and not self.scenario.signal_delivered:
            self.scenario.signal_delivered = True
            self.scenario.signals.deliver(self.scenario.signals.SIGTERM)
        self.cleanups.extend(['zero', 'close', 'delete_sequence', 'delete_array'])
        self.active = None
        return []


class Scenario:
    def __init__(self, *, fail_qc=None, lost_ack_point=None, signal_cleanup_point=None):
        self.fail_qc = fail_qc
        self.lost_ack_point = lost_ack_point
        self.signal_cleanup_point = signal_cleanup_point
        self.signal_delivered = False
        self.clock, self.signals = Clock(), Signals()
        self.devices, self.trains, self.records, self.markers, self.operations, self.analyses = [], [], [], [], [], []
    def analyze(self, runner, kind, path, **kwargs):
        point = runner.dev.point
        self.analyses.append((point, kind))
        record = next(r for r in reversed(self.records) if r['path'] == path)
        failure = self.fail_qc == (point, kind)
        events = []
        if kind == 'train':
            events = [{**event, 'time_s': runner.dev.send_mono - record['start_mono'] + .005 +
                       (index // 3) * .2 + (index % 3) * .01}
                      for index, event in enumerate(runner.dev.sent_events)]
        return {'status': 'failed' if failure else 'pass', 'errors': ['injected QC failure'] if failure else [],
                'metadata': {'frame_gaps': 0, 'duration_seconds': record['recorded_seconds']},
                'events': events, 'source': {'path': path}}


class RunnerTests(unittest.TestCase):
    def run_scenario(self, **kwargs):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        route = root / 'route.cfg'
        route.write_text(''.join(f'{i}({1000+i})' for i in range(100)))
        config = multipoint.Config(mode='simulate', session_id='same-biological-session', chip_id='28073',
            organoid_id='same-organoid', started_utc='2026-11-17T06:13:00Z', device_hostname='offline',
            route_file=str(route), route_sha256=hashlib.sha256(route.read_bytes()).hexdigest(),
            stim_electrode=1000, threshold_value=5., threshold_h5_dataset='settings/spike_threshold')
        plan = {'base_config': config.to_dict(), 'points': [{'channel': i, 'electrode': 1000+i} for i in range(20)],
                'roi': list(range(20, 80)), 'reference_rate': 10., 'r2_enabled': False, 'training_enabled': False}
        scenario = Scenario(**kwargs)
        def register(runner):
            # Keep real durable MultiLedger reserve/total; replace only machine-specific registration.
            registry = root / 'registry'
            (registry / 'claims').mkdir(parents=True)
            runner.out.mkdir()
            runner.ledger = multipoint.MultiLedger(registry)
        with ExitStack() as stack:
            stack.enter_context(patch('socket.socket', side_effect=AssertionError('NETWORK FORBIDDEN')))
            stack.enter_context(patch.object(multipoint, 'RealClock', return_value=scenario.clock))
            stack.enter_context(patch.object(multipoint, 'signal', scenario.signals))
            stack.enter_context(patch.object(multipoint.Runner, 'register', register))
            stack.enter_context(patch.object(multipoint.Runner, 'analyze',
                lambda runner, kind, path, **kw: scenario.analyze(runner, kind, path, **kw)))
            stack.enter_context(patch.object(multipoint_backend, 'MultipointBackend',
                lambda cfg, folder, clock: Backend(scenario, cfg, folder, clock)))
            runner = multipoint.Runner(plan, root / 'run')
            result = runner.run()
        return result, scenario, runner

    def test_full_twenty_points_reserve_300_true_pulses_and_final_601_seconds(self):
        result, scenario, runner = self.run_scenario()
        self.assertEqual(result['status'], 'completed')
        self.assertEqual(result['reserved_true_pulses'], 300)
        self.assertEqual(len(scenario.trains), 20)
        events = [event for _, train, _ in scenario.trains for event in train]
        self.assertEqual(len(events), 300)
        self.assertEqual(len({event['id'] for event in events}), 300)
        self.assertEqual({event['arm'] for event in events}, {'true'})
        claims = [json.loads(path.read_text()) for path in runner.ledger.claims.glob('*.json')]
        self.assertEqual(len(claims), 300)
        self.assertEqual({claim['arm'] for claim in claims}, {'true'})
        final_record = next(r for r in scenario.records if r['point'] == 20 and r['name'] == 'train')
        self.assertGreaterEqual(final_record['closed_mono'] - scenario.trains[-1][2] - .8254, 601.)
        self.assertFalse(any(r['name'] == 'recovery' for r in scenario.records))
        self.assertTrue(all(marker[1] == 'routing' for marker in scenario.markers))
        self.assertTrue(all(d.cleanups == ['all32off', 'zero', 'close', 'delete_sequence', 'delete_array'] for d in scenario.devices))
        self.assertEqual(result['hard_deadline_wall'], datetime(2026, 11, 17, 8, 13, tzinfo=timezone.utc).timestamp())
        self.assertLess(result['finished_wall'], result['hard_deadline_wall'])
        self.assertFalse(result['r2_enabled'])
        self.assertFalse(result['training_enabled'])

    def test_first_train_qc_failure_blocks_next_point_and_gets_fresh_recovery(self):
        result, scenario, _ = self.run_scenario(fail_qc=(1, 'train'))
        self.assertEqual(result['status'], 'stopped')
        self.assertEqual(result['reserved_true_pulses'], 15)
        self.assertEqual([point for point, _, _ in scenario.trains], [1])
        self.assertEqual([d.point for d in scenario.devices], [1])
        recovery = [r for r in scenario.records if r['name'] == 'recovery']
        self.assertEqual(len(recovery), 1)
        self.assertGreaterEqual(recovery[0]['recorded_seconds'], 601)
        self.assertEqual(scenario.analyses[-1], (1, 'recovery'))
        self.assertEqual(result['cleanup_errors'], [])

    def test_second_point_routing_qc_failure_blocks_its_train_and_recovers(self):
        result, scenario, _ = self.run_scenario(fail_qc=(2, 'preflight'))
        self.assertEqual(result['status'], 'stopped')
        self.assertEqual(result['reserved_true_pulses'], 15)
        self.assertEqual([point for point, _, _ in scenario.trains], [1])
        self.assertEqual([d.point for d in scenario.devices], [1, 2])
        recovery = [r for r in scenario.records if r['name'] == 'recovery']
        self.assertEqual(len(recovery), 1)
        self.assertGreaterEqual(recovery[0]['recorded_seconds'], 601)
        self.assertTrue(all(len(d.cleanups) == 5 for d in scenario.devices))

    def test_lost_first_train_ack_consumes_15_without_retry_then_cleans(self):
        result, scenario, runner = self.run_scenario(lost_ack_point=1)
        self.assertEqual(result['status'], 'failed')
        self.assertEqual(result['reserved_true_pulses'], 15)
        self.assertEqual(runner.ledger.total(), 15)
        self.assertEqual(len(scenario.trains), 1)
        self.assertEqual(len(scenario.devices), 1)
        self.assertEqual(scenario.devices[0].cleanups, ['all32off', 'zero', 'close', 'delete_sequence', 'delete_array'])
        self.assertIsNone(scenario.devices[0].active)
        self.assertFalse(any(r['name'] == 'recovery' for r in scenario.records))

    def test_signal_during_point_cleanup_finishes_cleanup_and_blocks_next_point(self):
        result, scenario, _ = self.run_scenario(signal_cleanup_point=1)
        self.assertTrue(scenario.signal_delivered)
        self.assertEqual(result['status'], 'failed')
        self.assertEqual(result['reserved_true_pulses'], 15)
        self.assertEqual(len(scenario.trains), 1)
        self.assertEqual(len(scenario.devices), 1)
        self.assertEqual(scenario.devices[0].cleanups, ['all32off', 'zero', 'close', 'delete_sequence', 'delete_array'])
        self.assertIn('Signal deferred until point cleanup completed', ' '.join(result['errors']))
        self.assertEqual(result['cleanup_errors'], [])
        self.assertEqual(scenario.signals.timers[-1], (scenario.signals.ITIMER_REAL, 0))


if __name__ == '__main__': unittest.main()
