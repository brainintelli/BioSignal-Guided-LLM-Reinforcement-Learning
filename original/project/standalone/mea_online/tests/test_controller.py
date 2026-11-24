import copy
import importlib
import json
import tempfile
import unittest
from pathlib import Path


def module(name):
    try:return importlib.import_module('mea_online.'+name)
    except ModuleNotFoundError as e:raise AssertionError('Required online module not implemented: '+name) from e


class Clock:
    def __init__(self):self.t=1000.;self.m=0.
    def time(self):return self.t
    def monotonic(self):return self.m
    def advance(self,n):self.t+=n;self.m+=n


class ConfigTests(unittest.TestCase):
    def test_unbound_template_cannot_start_live(self):
        c=module('config');v=c.template()
        self.assertEqual(c.validate_config(v,live=False)['gpu_host'],'192.0.2.16')
        with self.assertRaisesRegex(ValueError,'binding'):c.validate_config(v,live=True)

    def test_no_flag_can_enable_r2_stimulation_or_change_reward(self):
        c=module('config')
        for field,value in [('r2_enabled',True),('stimulation',True),('reward','r1_plus_r2'),('max_session_seconds',7201)]:
            v=c.template();v[field]=value
            with self.subTest(field=field),self.assertRaises(ValueError):c.validate_config(v,live=False)

    def test_identity_explicit_and_not_old_chip_default(self):
        c=module('config');v=c.template()
        self.assertIsNone(v['binding']);self.assertIsNone(v['calibration'])
        self.assertEqual(v['devices'],list(range(8)))

    def test_budget_cannot_extend_after_wall_clock_rollback(self):
        c=module('config');clock=Clock();b=c.Budget(1000.,clock)
        clock.advance(7000);clock.t-=2000
        self.assertAlmostEqual(b.remaining(),200)
        with self.assertRaises(TimeoutError):b.require(81)
        self.assertAlmostEqual(b.remaining(),200)

    def test_forward_clock_correction_cannot_be_reclaimed_by_rollback(self):
        c=module('config');clock=Clock();b=c.Budget(1000.,clock)
        clock.t=7900.;clock.m=1.
        self.assertEqual(b.remaining(),300.)
        clock.t=1002.;clock.m=2.
        self.assertEqual(b.remaining(),299.)
        with self.assertRaises(TimeoutError):b.require(180)

    def test_registry_same_organoid_cannot_restart_with_new_session_name(self):
        c=module('config')
        with tempfile.TemporaryDirectory() as d:
            b={'organoid_id':'org-01','started_utc':'2026-11-18T01:00:00+00:00','session_id':'first'}
            c.claim_session(Path(d),b)
            b['session_id']='second'
            with self.assertRaises(FileExistsError):c.claim_session(Path(d),b)


class ControllerTests(unittest.TestCase):
    def run_case(self,**kw):return module('offline').exercise(**kw)

    def test_gae_matches_existing_m9_discount_not_new_default(self):
        rows=[dict(reward=1.,duration=2,value=.1,terminal=False),dict(reward=2.,duration=1,value=.2,terminal=True)]
        module('trainer').finish_rows(rows,0.)
        expected=1.+(.99985**2)*.2-.1+(.99985**2)*(.999**2)*(2.-.2)
        self.assertAlmostEqual(rows[0]['advantage'],expected,12)

    def test_real_small_backend_update_and_checkpoint_reload(self):
        r=self.run_case()
        self.assertEqual(r['status'],'completed');self.assertEqual(r['updates'],2)
        self.assertEqual(r['decisions'],8);self.assertEqual(r['windows'],10)
        self.assertTrue(r['reload_verified']);self.assertTrue(r['actor_changed']);self.assertTrue(r['critic_changed'])
        self.assertFalse(r['r2_enabled']);self.assertFalse(r['stimulation']);self.assertEqual(r['backend'],'offline_tiny_numeric_not_m9')
        self.assertEqual(len(set(r['decision_window_ids'])),8)
        self.assertTrue(set(r['decision_window_ids']).isdisjoint(r['bootstrap_window_ids']))

    def test_exact_terminal_batch_needs_no_bootstrap(self):
        r=self.run_case(terminal_every=4)
        self.assertEqual(r['updates'],2);self.assertEqual(r['windows'],8)
        self.assertEqual(r['bootstrap_values'],[0.,0.]);self.assertEqual(r['reset_count'],3)

    def test_midbatch_terminal_isolates_returns(self):
        r=self.run_case(terminal_every=2)
        self.assertEqual(r['terminal_indices'],[1,3,5,7]);self.assertEqual(r['updates'],2)
        self.assertTrue(r['gae_verified'])

    def test_stream_fault_preserves_partial_and_never_updates(self):
        r=self.run_case(fault_at=3)
        self.assertEqual(r['status'],'failed');self.assertEqual(r['updates'],0)
        self.assertEqual(r['decisions'],2);self.assertTrue(r['source_closed']);self.assertTrue(r['runtime_closed'])

    def test_stale_duplicate_and_simulation_source_not_live(self):
        for fault in ['duplicate','stale','wrong_source','wrong_binding','mutated_hash','missing_frame','nan']:
            with self.subTest(fault=fault):
                r=self.run_case(fault_at=3,fault=fault)
                self.assertEqual(r['status'],'failed');self.assertEqual(r['updates'],0)

    def test_deadline_after_input_prevents_action(self):
        r=self.run_case(expire_at=3)
        self.assertEqual(r['status'],'failed');self.assertEqual(r['decisions'],2);self.assertEqual(r['updates'],0)

    def test_input_expiring_during_model_inference_prevents_action(self):
        r=self.run_case(inference_delay=11.)
        self.assertEqual(r['status'],'failed');self.assertEqual(r['decisions'],0);self.assertEqual(r['updates'],0)

    def test_deadline_checked_after_slow_model_state_audit(self):
        r=self.run_case(initial_remaining=154.,inference_delay=6.,state_delay=3.5)
        self.assertTrue(all(v>=121. for v in r['actions_remaining']),r['actions_remaining'])

    def test_r2_attempt_from_runtime_rejected(self):
        r=self.run_case(runtime_reward_violation=True)
        self.assertEqual(r['status'],'failed');self.assertEqual(r['updates'],0)

    def test_offline_does_not_open_network(self):
        from unittest.mock import patch
        with patch('socket.create_connection',side_effect=AssertionError('offline network')),patch('socket.socket.connect',side_effect=AssertionError('offline network')):
            self.assertEqual(self.run_case()['status'],'completed')


if __name__=='__main__':unittest.main()
