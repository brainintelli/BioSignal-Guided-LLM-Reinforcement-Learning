"""Software-only closed-loop contracts; synthetic signals are never biology."""
import copy
from dataclasses import replace
import importlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
import uuid
from unittest.mock import patch

from tests.test_mea_p0 import observation

ROOT = Path(__file__).resolve().parents[1]
GODOT = ROOT / '.tools/godot/Godot_v4.7.2-stable_win64_console.exe'


class ModuleTest(unittest.TestCase):
    def module(self, name):
        self.assertIsNotNone(importlib.util.find_spec(name), f'{name} not implemented')
        return importlib.import_module(name)

    def baseline(self, **changes):
        mod = self.module('agent.mea_stimulation')
        values = dict(session_id='sim-session-' + uuid.uuid4().hex, chip_id='sim-chip', well_id=0,
                      sample_rate_hz=20000, channels=((0, 100), (1, 101), (2, 102), (3, 103)),
                      electrode_id=100, start_frame=0, end_frame=18_000_000,
                      events=tuple((i + 1, 0, -50.) for i in range(100)))
        return mod.Baseline(**(values | changes))

    def path(self, **changes):
        mod = self.module('agent.mea_stimulation')
        return mod.OutputPath(**(dict(path_id='virtual-uv', dac_lsb_mv=.001,
                                     uncertainty_mv=.001, calibration_id='synthetic-calibration') | changes))


class PlanningTests(ModuleTest):
    def test_parameter_profile_is_required_and_rejects_unit_mode_and_cap_changes(self):
        runner = self.module('tools.mea_game_loop')
        config = runner.default_config()
        self.assertIn('stimulation_parameters', config)
        for key, value in [('mode', 'current'), ('phase_us', 4), ('time_step_us', 100),
                           ('p90_multiplier', 1.21), ('dac_channel', 3),
                           ('polarity', 'negative_first'), ('inter_phase_us', 50),
                           ('onset_interval_min_ms', 300), ('onset_interval_max_ms', 14001),
                           ('initial_delay_samples', 0), ('interval_seed', True)]:
            changed = copy.deepcopy(config)
            changed['stimulation_parameters'][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError): runner.validate_config(changed)
        missing = copy.deepcopy(config); missing.pop('stimulation_parameters')
        with self.assertRaises(ValueError): runner.validate_config(missing)

    def test_maxlab_sequence_units_polarity_and_initial_delay_are_explicit(self):
        mod = self.module('agent.mea_stimulation')
        self.assertTrue(hasattr(mod, 'compile_sequence'), 'MaxLab sequence preview not implemented')
        obs = observation(); obs['player']['thirst'] = 10
        plan = mod.make_plan(obs, self.baseline(sample_rate_hz=10000), self.path(), 'event-0', 18_000_000)
        sequence = mod.compile_sequence(plan)
        self.assertEqual(sequence['initial_delay_samples'], 100)
        self.assertEqual(sequence['time_step_us'], 50)
        self.assertEqual(sequence['commands'], [
            {'kind': 'delay_samples', 'samples': 1900},
            {'kind': 'event', 'event_id': 'event-0'},
            {'kind': 'dac', 'channel': 0, 'code': 453},
            {'kind': 'delay_samples', 'samples': 4},
            {'kind': 'dac', 'channel': 0, 'code': 571},
            {'kind': 'delay_samples', 'samples': 4},
            {'kind': 'dac', 'channel': 0, 'code': 512}])
        self.assertEqual(plan.stimulus_frame-plan.reference_frame, 1000)
        self.assertEqual(sequence['total_duration_us'], 100400)
        self.assertEqual(mod.delay_commands(280000), [
            {'kind': 'delay_samples', 'samples': 65535},
            {'kind': 'delay_samples', 'samples': 65535},
            {'kind': 'delay_samples', 'samples': 65535},
            {'kind': 'delay_samples', 'samples': 65535},
            {'kind': 'delay_samples', 'samples': 17860}])
        with self.assertRaises(ValueError): mod.compile_sequence(replace(plan, phase_us=201))

    def test_p90_uses_own_absolute_spikes_and_freezes_input(self):
        rows = [(i, 0, -float(i)) for i in range(1, 101)] + [(101, 1, -99999.)]
        baseline = self.baseline(events=rows)
        self.assertAlmostEqual(baseline.p90_uv, 90.1)
        self.assertAlmostEqual(baseline.cap_mv, .10812)
        before = baseline.baseline_id
        rows.clear()
        self.assertEqual(len(baseline.events), 101)
        self.assertEqual(baseline.baseline_id, before)
        with self.assertRaises(Exception): baseline.p90_uv = 999

    def test_cap_includes_uncertainty_and_quantizes_down(self):
        mod = self.module('agent.mea_stimulation')
        plan = mod.make_plan(observation(), self.baseline(), self.path(), 'event-0', 18_000_000)
        self.assertEqual(plan.dac_offset, 59)
        self.assertEqual(plan.nominal_peak_mv, .059)
        self.assertEqual(plan.peak_upper_mv, .06)
        self.assertEqual(plan.cap_mv, .06)
        self.assertEqual(plan.phase_us, 200)
        self.assertEqual(plan.dac_codes, (453, 571, 512))
        self.assertEqual(plan.stimulus_frame - plan.reference_frame, plan.code_delay_ms * 20)

    def test_historical_lsb_cannot_be_promoted_to_one_step(self):
        mod = self.module('agent.mea_stimulation')
        with self.assertRaisesRegex(ValueError, 'nonzero'):
            mod.make_plan(observation(), self.baseline(), self.path(dac_lsb_mv=2.981901), 'event-0', 18_000_000)
        sham = mod.make_plan(observation(), self.baseline(), self.path(dac_lsb_mv=2.981901),
                             'event-0', 18_000_000, condition='sham')
        self.assertEqual(sham.nominal_peak_mv, 0)
        self.assertEqual(sham.dac_codes, (512, 512, 512))
        self.assertEqual(sham.condition, 'sham')

    def test_float_display_rounding_cannot_raise_the_hard_cap(self):
        mod = self.module('agent.mea_stimulation')
        baseline = self.baseline(events=tuple((i+1, 0, -.8333333333333333) for i in range(100)))
        # Exact decimal cap is 0.00099999999999999996 mV, below one 0.001 mV step.
        with self.assertRaisesRegex(ValueError, 'nonzero'):
            mod.make_plan(observation(), baseline, self.path(uncertainty_mv=0), 'event-0', 18_000_000)

    def test_inert_sequence_rejects_forged_codes(self):
        mod = self.module('agent.mea_stimulation')
        plan = mod.make_plan(observation(), self.baseline(), self.path(), 'event-0', 18_000_000)
        with self.assertRaises(ValueError): mod.compile_sequence(replace(plan, dac_codes=(0, 1023, 0)))

    def test_task_encoding_uses_current_facts_and_not_reward_extras(self):
        mod = self.module('agent.mea_stimulation')
        obs = observation()
        obs['player']['health'] = 100
        obs['player']['thirst'] = 80
        obs['threats'] = []
        baseline = self.baseline()
        plan = lambda x: mod.make_plan(x, baseline, self.path(), 'event-0', 18_000_000)
        a = plan(obs)
        self.assertEqual(a.code_delay_ms, 300)
        injected = copy.deepcopy(obs)
        injected.update(reward=999, future={'danger': True}, recommended_action='drink')
        self.assertEqual(a.plan_id, plan(injected).plan_id)
        obs['player']['health'] = 10
        self.assertEqual(plan(obs).code_delay_ms, 200)
        obs['player']['thirst'] = 10
        self.assertEqual(plan(obs).code_delay_ms, 100)

    def test_invalid_baselines_and_output_numbers_fail_closed(self):
        for changes in (dict(events=((1, 0, -50.),)), dict(electrode_id=777),
                        dict(events=tuple((1, 0, -50.) for _ in range(100))),
                        dict(end_frame=50), dict(sample_rate_hz=True),
                        dict(channels=((0, 100), (0, 101)))):
            with self.subTest(changes=changes), self.assertRaises(ValueError): self.baseline(**changes)
        for changes in (dict(dac_lsb_mv=0), dict(dac_lsb_mv=float('nan')),
                        dict(uncertainty_mv=-1), dict(dac_lsb_mv=True),
                        dict(simulation_only=False)):
            with self.subTest(changes=changes), self.assertRaises(ValueError): self.path(**changes)
        mod = self.module('agent.mea_stimulation')
        with self.assertRaises(ValueError):
            mod.make_plan(observation(), self.baseline(), self.path(uncertainty_mv=.06), 'event-0', 18_000_000)


class InferenceTests(ModuleTest):
    def test_existing_policy_tokenization_uses_only_projected_state2_input(self):
        mod = self.module('agent.mea_inference')
        sim = self.module('agent.mea_simulator')
        loop_mod = self.module('agent.mea_closed_loop')
        baseline = self.baseline()
        trial = loop_mod.ClosedLoop(baseline, self.path()).transact(observation(), sim.Simulator(baseline), sim.State2Probe())
        class Tokenizer:
            eos_token_id = 0
            def encode(self, text, add_special_tokens=False): return [ord(x) + 1 for x in text]
        class TokenizationPort:
            tokenizer = Tokenizer()
            def _tokens(self, messages): return self.tokenizer.encode(json.dumps(messages, ensure_ascii=False))
        prepared = mod.prepare_inference_row(TokenizationPort(), trial['actor_input'])
        decoded = ''.join(chr(i - 1) for i in prepared['input_ids'])
        self.assertIn('湿件观测', decoded)
        self.assertIn('spike_count', decoded)
        self.assertNotIn(trial['plan']['plan_id'], decoded)
        self.assertEqual(prepared['input_ids'], prepared['kv_prefix_ids'])
        self.assertTrue(prepared['semantic_trees'])
        with self.assertRaises(ValueError):
            mod.prepare_inference_row(TokenizationPort(), trial['actor_input'] | {'reward': 99})


class LoopTests(ModuleTest):
    def test_onset_grid_at_both_limits_is_independent_of_recording_rate(self):
        mod = self.module('agent.mea_stimulation')
        loops = self.module('agent.mea_closed_loop')
        sim = self.module('agent.mea_simulator')
        for interval, expected_wait in ((10000, 199992), (14000, 279992)):
            baseline = self.baseline(sample_rate_hz=10000, end_frame=9_000_000)
            params = mod.StimulationParameters(onset_interval_min_ms=interval, onset_interval_max_ms=interval)
            loop = loops.ClosedLoop(baseline, self.path(), parameters=params)
            backend, actor = sim.Simulator(baseline), sim.State2Probe()
            first = loop.transact(observation(), backend, actor)
            second = loop.transact(observation(), backend, actor, condition='sham')
            self.assertEqual(second['plan']['stimulus_frame'] - first['plan']['stimulus_frame'], interval * 10)
            self.assertEqual(second['timing']['post_pulse_delay_samples'], expected_wait)
            self.assertEqual(second['plan']['parameters'], params.to_dict())

    def test_time_code_change_does_not_exceed_fourteen_second_onset_limit(self):
        loop, backend, actor = self.setup_loop()
        obs = observation(); obs['threats'] = []; obs['player']['health'] = 100
        obs['player']['thirst'] = 10
        first = loop.transact(obs, backend, actor)
        obs['player']['thirst'] = 80
        second = loop.transact(obs, backend, actor)
        self.assertLessEqual(second['plan']['stimulus_frame'] - first['plan']['stimulus_frame'], 280000)

    def test_onset_intervals_are_fixed_before_game_state_and_include_sham(self):
        first_intervals = None
        for reverse in (False, True):
            loop, backend, actor = self.setup_loop()
            previous = None
            intervals = []
            for index in range(8):
                obs = observation(); obs['threats'] = []; obs['player']['health'] = 100
                obs['player']['thirst'] = 10 if bool(index % 2) != reverse else 80
                row = loop.transact(obs, backend, actor, 'sham' if index % 3 == 2 else 'stimulus')
                plan = row['plan']
                if previous is not None:
                    interval = (plan['stimulus_frame'] - previous['stimulus_frame']) / 20
                    self.assertGreaterEqual(interval, 10000)
                    self.assertLessEqual(interval, 14000)
                    self.assertEqual(interval, row['timing']['onset_interval_ms'])
                    self.assertEqual(row['timing']['post_pulse_delay_samples'] * 50 + 400, interval * 1000)
                    intervals.append(interval)
                previous = plan
            if first_intervals is None: first_intervals = intervals
            else: self.assertEqual(first_intervals, intervals)
            self.assertGreater(len(set(intervals)), 1)

    def test_recreating_a_session_cannot_reset_budget_or_reuse_old_receipt(self):
        module = self.module('agent.mea_closed_loop')
        simulator = self.module('agent.mea_simulator')
        baseline = self.baseline()
        first = module.ClosedLoop(baseline, self.path(), max_pulses=1)
        old = first.transact(observation(), simulator.Simulator(baseline), simulator.State2Probe())
        with self.assertRaisesRegex(ValueError, 'session'):
            module.ClosedLoop(replace(baseline), self.path(), max_pulses=1)
        fresh = replace(baseline, session_id='new-session-' + uuid.uuid4().hex)
        second = module.ClosedLoop(fresh, self.path())
        class OldResponse:
            def capture(self, plan): return {k: old[k] for k in ('receipt', 'packets', 'sequence')}
        with self.assertRaises(ValueError): second.transact(observation(), OldResponse(), simulator.State2Probe())
        self.assertEqual(second.state, 'INVALID')

    def test_saturation_attestation_is_required_and_strict(self):
        for value in (True, None, 0, 'false', 'missing'):
            loop, backend, actor = self.setup_loop()
            class Fault:
                def capture(self, plan):
                    data = backend.capture(plan)
                    if value == 'missing': data['receipt'].pop('saturated', None)
                    else: data['receipt']['saturated'] = value
                    return data
            class ForbiddenActor:
                def choose(self, projection): raise AssertionError('Saturated/unknown response reached actor')
            with self.subTest(value=value), self.assertRaises(ValueError):
                loop.transact(observation(), Fault(), ForbiddenActor())
            self.assertEqual(loop.state, 'INVALID'); self.assertEqual(loop.pulse_attempts, 1)

    def setup_loop(self, **kwargs):
        module = self.module('agent.mea_closed_loop')
        simulator = self.module('agent.mea_simulator')
        baseline = self.baseline()
        loop = module.ClosedLoop(baseline, self.path(), **kwargs)
        return loop, simulator.Simulator(baseline), simulator.State2Probe()

    def test_complete_loop_excludes_artifact_stim_channel_and_metadata(self):
        loop, backend, actor = self.setup_loop()
        obs = observation(); obs['threats'] = []
        obs['player']['health'] = 100; obs['player']['thirst'] = 80
        row = loop.transact(obs, backend, actor)
        self.assertEqual(row['snapshot']['feature_values'][0], 6.)
        self.assertEqual(row['snapshot']['mapped_channels'], 3)
        self.assertAlmostEqual(row['snapshot']['feature_values'][1], 6 / (.18 * 3))
        self.assertEqual(row['snapshot']['feature_values'][2], 1.)
        self.assertLess(row['snapshot']['feature_values'][3], 100)
        self.assertIn(row['action'], obs['allowed_actions'])
        self.assertEqual(loop.state, 'READY')
        self.assertEqual(loop.pulse_attempts, 1)
        self.assertEqual(set(row['actor_input']), {'messages', 'branches', 'target_texts', 'verb_mask', 'wetware_features'})
        self.assertNotIn(row['plan']['plan_id'], json.dumps(row['actor_input']))
        next_row = loop.transact(obs, backend, actor)
        self.assertNotEqual(row['plan']['event_id'], next_row['plan']['event_id'])
        self.assertEqual(next_row['plan']['stimulus_frame'] - row['plan']['stimulus_frame'],
                         next_row['timing']['onset_interval_ms'] * 20)

    def test_quiet_complete_window_is_valid_but_missing_watermark_is_not(self):
        loop, backend, actor = self.setup_loop()
        quiet = loop.transact(observation(), backend, actor, condition='sham')
        self.assertEqual(quiet['snapshot']['feature_values'], [0., 0., 0., 0.])
        self.assertEqual(loop.sham_attempts, 1)
        self.assertEqual(loop.pulse_attempts, 0)
        class Missing:
            def capture(self, plan):
                data = backend.capture(plan); data['packets'].pop(); return data
        with self.assertRaises(ValueError): loop.transact(observation(), Missing(), actor)
        self.assertEqual(loop.state, 'INVALID')
        self.assertEqual(loop.pulse_attempts, 1)
        self.assertIn('packets', loop.audit[-1])
        with self.assertRaises(ValueError): loop.transact(observation(), backend, actor)
        self.assertEqual(loop.pulse_attempts, 1)

    def test_receipt_failures_stop_before_actor_and_consume_attempt(self):
        mutations = [dict(status='unknown'), dict(session_id='old'), dict(chip_id='other'),
                     dict(plan_id='old'), dict(event_id='old'), dict(channel_map_id='other'),
                     dict(peak_upper_mv=.060000001), dict(peak_upper_mv=float('nan')),
                     dict(actual_frame=-1), dict(artifact_recovery_ms=16), dict(recovered=False)]
        for mutation in mutations:
            loop, backend, actor = self.setup_loop()
            class Fault:
                def capture(self, plan):
                    result = backend.capture(plan); result['receipt'].update(mutation); return result
            class ForbiddenActor:
                def choose(self, row): raise AssertionError('Invalid response reached actor')
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                loop.transact(observation(), Fault(), ForbiddenActor())
            self.assertEqual(loop.state, 'INVALID')
            self.assertEqual(loop.pulse_attempts, 1)

    def test_raw_stream_mismatch_drop_late_event_and_live_source_are_rejected(self):
        for kind in ('mapping', 'source', 'drop', 'marker', 'late', 'duplicate'):
            loop, backend, actor = self.setup_loop()
            class Fault:
                def capture(self, plan):
                    data = backend.capture(plan); packets = data['packets']
                    if kind == 'mapping': packets[0]['channels'][1][1] = 999
                    if kind == 'source': packets[0]['source'] = 'live'
                    if kind == 'drop': packets[-1]['dropped'] = 1
                    if kind == 'marker': packets[1]['frame'] += 1
                    if kind == 'late':
                        packets.append(dict(type='spikes', session_id=plan.session_id, seq=4,
                                            events=[[plan.stimulus_frame + 400, 1, -10.]]))
                    if kind == 'duplicate': packets[2]['events'].append(packets[2]['events'][-1])
                    return data
            with self.subTest(kind=kind), self.assertRaises(ValueError): loop.transact(observation(), Fault(), actor)
            self.assertEqual(loop.state, 'INVALID')

    def test_context_cannot_change_while_response_is_pending(self):
        loop, backend, actor = self.setup_loop()
        obs = observation()
        class ChangeObservation:
            def capture(self, plan):
                data = backend.capture(plan); obs['player']['thirst'] = 1; return data
        with self.assertRaisesRegex(ValueError, 'observation'): loop.transact(obs, ChangeObservation(), actor)
        self.assertEqual(loop.state, 'INVALID')

    def test_budget_exhaustion_and_deadline_never_issue_extra_attempt(self):
        loop, backend, actor = self.setup_loop(max_pulses=1, max_shams=1)
        loop.transact(observation(), backend, actor)
        with self.assertRaises(ValueError): loop.transact(observation(), backend, actor)
        self.assertEqual(loop.state, 'STOPPED'); self.assertEqual(loop.pulse_attempts, 1)
        now = [0.]
        loop, backend, actor = self.setup_loop(clock=lambda: now[0])
        now[0] = 6300
        with self.assertRaises(ValueError): loop.transact(observation(), backend, actor)
        self.assertEqual(loop.pulse_attempts, 0)
        loop, backend, actor = self.setup_loop(clock=lambda: now[0])
        class Slow:
            def choose(self, prepared):
                now[0] += 6301; return actor.choose(prepared)
        with self.assertRaises(ValueError): loop.transact(observation(), backend, Slow())
        self.assertEqual(loop.state, 'STOPPED')
        self.assertNotIn('action', loop.audit[-1])

    def test_virtual_frame_deadline_and_clock_regression_stop(self):
        mod = self.module('agent.mea_closed_loop')
        sim = self.module('agent.mea_simulator')
        baseline = self.baseline(end_frame=125_999_000)
        loop = mod.ClosedLoop(baseline, self.path())
        with self.assertRaises(ValueError): loop.transact(observation(), sim.Simulator(baseline), sim.State2Probe())
        self.assertEqual(loop.pulse_attempts, 0)
        now = [10.]
        loop, backend, actor = self.setup_loop(clock=lambda: now[0])
        now[0] = 9.
        with self.assertRaises(ValueError): loop.transact(observation(), backend, actor)
        self.assertEqual(loop.state, 'STOPPED')

    def test_probe_consumes_state2_but_never_gets_reward_or_stimulus_labels(self):
        loop, backend, actor = self.setup_loop()
        row = loop.transact(observation(), backend, actor)
        prepared = row['actor_input']
        a = copy.deepcopy(prepared); b = copy.deepcopy(prepared)
        a['wetware_features'] = [0., 0., 0., 0.]
        b['wetware_features'] = [1., 0., 0., 0.]
        self.assertNotEqual(actor.choose(a), actor.choose(b))
        self.assertNotIn('reward', prepared)
        self.assertNotIn('condition', prepared)
        self.assertNotIn('context_id', prepared)


class NativeLoopTests(ModuleTest):
    @unittest.skipUnless(GODOT.exists(), 'native Godot unavailable')
    def test_configured_intervals_reach_native_rollout_and_replay(self):
        mod = self.module('tools.mea_game_loop')
        config = mod.default_config()
        config['stimulation_parameters'].update(onset_interval_min_ms=10000, onset_interval_max_ms=10000)
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp) / 'run'
            mod.simulate(out, config=config, steps=3)
            rows = [json.loads(s) for s in (out / 'rollout.jsonl').read_text(encoding='utf-8').splitlines()]
            for previous, row in zip(rows, rows[1:]):
                self.assertEqual(row['response']['plan']['parameters'], config['stimulation_parameters'])
                self.assertEqual(row['response']['plan']['stimulus_frame'] - previous['response']['plan']['stimulus_frame'], 200000)
            self.assertTrue(mod.replay(out)['passed'])

    @unittest.skipUnless(GODOT.exists(), 'native Godot unavailable')
    def test_independent_runs_have_fresh_session_identity_but_replay_is_explicit(self):
        mod = self.module('tools.mea_game_loop')
        with tempfile.TemporaryDirectory() as temp:
            a, b = Path(temp) / 'a', Path(temp) / 'b'
            mod.simulate(a, steps=1); mod.simulate(b, steps=1)
            first = json.loads((a / 'baseline.json').read_text(encoding='utf-8'))
            second = json.loads((b / 'baseline.json').read_text(encoding='utf-8'))
            self.assertNotEqual(first['session_id'], second['session_id'])
            self.assertTrue(mod.replay(a)['passed'])

    def test_unsafe_flags_and_unknown_config_fields_rejected_before_output(self):
        mod = self.module('tools.mea_game_loop')
        for changes in (dict(alpha=.9), dict(r2_reward_enabled=True), dict(realtime_training=True),
                        dict(actual_stimulation_enabled=True), dict(acquisition_enabled=True),
                        dict(source='live'), dict(host='192.0.2.1'), dict(max_pulses=26),
                        dict(action_source='qwen')):
            with tempfile.TemporaryDirectory() as temp:
                out = Path(temp) / 'run'
                with self.subTest(changes=changes), self.assertRaises(ValueError):
                    mod.simulate(out, config=mod.default_config() | changes, steps=3)
                self.assertFalse(out.exists())

    @unittest.skipUnless(GODOT.exists(), 'native Godot unavailable')
    def test_native_loop_and_replay_without_device_connections(self):
        mod = self.module('tools.mea_game_loop')
        with tempfile.TemporaryDirectory() as temp, patch('socket.socket.connect', side_effect=AssertionError('Network forbidden')):
            out = Path(temp) / 'run'
            report = mod.simulate(out, steps=6, godot=GODOT, project=ROOT)
            self.assertEqual(report['decisions'], 6)
            self.assertEqual(report['simulated_nonzero_attempts'], 4)
            self.assertEqual(report['simulated_sham_attempts'], 2)
            self.assertEqual(report['device_commands_sent'], 0)
            self.assertEqual(report['optimizer_steps'], 0)
            self.assertFalse(report['r2_reward_enabled'])
            self.assertEqual(report['alpha'], 1)
            self.assertGreater(report['elapsed_ticks'], 0)
            self.assertGreater(report['behavior']['unique_walked_positions'], 0)
            replay = mod.replay(out, godot=GODOT, project=ROOT)
            self.assertTrue(replay['passed']); self.assertEqual(replay['verified_decisions'], 6)
            rows = [json.loads(s) for s in (out / 'rollout.jsonl').read_text(encoding='utf-8').splitlines()]
            self.assertTrue(all(r['reward']['total_reward'] == r['r1_score']['reward'] for r in rows))
            self.assertTrue(all(r['response']['snapshot']['source'] == 'simulation' for r in rows))
            with self.assertRaises(FileExistsError): mod.simulate(out, steps=6)

    @unittest.skipUnless(GODOT.exists(), 'native Godot unavailable')
    def test_replay_checks_inner_content_after_hashes_are_rewritten(self):
        mod = self.module('tools.mea_game_loop')
        from agent.mea_protocol import canonical
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp) / 'run'; mod.simulate(out, steps=3)
            path = out / 'rollout.jsonl'; original = path.read_text(encoding='utf-8')
            manifest = json.loads((out / 'manifest.json').read_text(encoding='utf-8'))
            for kind in ('reward', 'snapshot', 'action', 'receipt'):
                rows = [json.loads(s) for s in original.splitlines()]
                if kind == 'reward': rows[0]['reward']['total_reward'] += 1
                if kind == 'snapshot': rows[0]['response']['snapshot']['feature_values'][0] += 1
                if kind == 'receipt': rows[0]['response']['receipt']['actual_frame'] += 1
                if kind == 'action':
                    actions = rows[0]['observation']['allowed_actions']
                    rows[0]['response']['action'] = next(a for a in actions if a != rows[0]['response']['action'])
                path.write_text(''.join(canonical(r)+'\n' for r in rows), encoding='utf-8')
                manifest['files']['rollout.jsonl'] = mod.sha(path)
                (out / 'manifest.json').write_text(canonical(manifest), encoding='utf-8')
                with self.subTest(kind=kind), self.assertRaises(ValueError): mod.replay(out)

    @unittest.skipUnless(GODOT.exists(), 'native Godot unavailable')
    def test_failed_response_retains_evidence_without_a_completion_marker(self):
        mod = self.module('tools.mea_game_loop')
        original = mod.Simulator.capture
        def lost(backend, plan):
            result = original(backend, plan); result['packets'][-1]['dropped'] = 1; return result
        with tempfile.TemporaryDirectory() as temp, patch.object(mod.Simulator, 'capture', lost):
            out = Path(temp) / 'run'
            with self.assertRaises(ValueError): mod.simulate(out, steps=3)
            failure = json.loads((out / 'failure.json').read_text(encoding='utf-8'))
            self.assertEqual(failure['committed_decisions'], 0)
            self.assertEqual(failure['pulse_attempts'], 1)
            self.assertFalse(failure['retry_performed'])
            self.assertFalse((out / 'manifest.json').exists())
            self.assertIn('dropped', (out / 'failure_audit.txt').read_text(encoding='utf-8'))
            self.assertEqual((out / 'rollout.jsonl').read_text(), '')
            with self.assertRaises(ValueError): mod.replay(out)


if __name__ == '__main__': unittest.main()
