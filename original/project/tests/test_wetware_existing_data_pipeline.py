"""Unit & real-Godot integration tests for the existing-data wetware training pipeline."""
from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
for extra in (ROOT, ROOT / 'standalone' / 'mea_oddball'):
    s = str(extra)
    if s not in sys.path:
        sys.path.insert(0, s)

from agent import mea_dynamics_explorer as mde
from agent.m9_env import GodotEnv
from tools import eval_2x2_attribution as attr2x2
from tools import eval_survival_drives as ev
from tools import run_wetware_spontaneous_study as study

GODOT_BIN = ROOT / '.tools' / 'godot' / 'Godot_v4.7.2-stable_win64_console.exe'
PHASE0_DIR = ROOT / 'output' / 'wetware_existing_data_phase0_20261123'


class RealGodotCurriculumTests(unittest.TestCase):
    @unittest.skipUnless(GODOT_BIN.is_file(), 'Local headless Godot binary required')
    def test_e4_night_threat_breaks_100pct_survival_ceiling_and_rewards_evasion(self):
        """Standing still under E4 night threat MUST die; stepping away MUST survive."""
        with GodotEnv(
            str(GODOT_BIN), str(ROOT),
            map_mode='infinite_freeze', navigation_steps=4,
            env_script='godot_tests/m9_curriculum_environment.gd',
        ) as env:
            # 1) Passive waiting under E4 -> dies before horizon (smashes old 100% ceiling!)
            suite = ev.EVAL_SUITES['E4_death_night_threat']
            st = env.reset(960001, scenario=suite['scenario'])
            obs = st['observation']
            self.assertEqual(int(obs['tick']), 140)
            self.assertAlmostEqual(float(obs['player']['health']), 18.0, places=2)
            self.assertTrue(len(obs.get('threats', [])) >= 1, 'Night threats must be present at tick 140')
            self.assertLess(min(float(t['distance']) for t in obs['threats']), 1.6)

            wait_rows = []
            for _ in range(30):
                nxt = env.step({'verb': 'wait'})
                wait_rows.append(dict(observation=obs, next_observation=nxt['observation'],
                                      action={'verb': 'wait'}, transition=nxt['transition']))
                obs = nxt['observation']
                if nxt['terminated']:
                    break
            m_wait = ev.evaluate_episode(wait_rows, horizon_tick=suite['horizon_ticks'], relative_horizon=True)
            self.assertFalse(m_wait['alive'], 'Waiting next to slime at low HP must die (no 100% ceiling)')
            self.assertGreater(m_wait['threat_damage_taken'], 10.0)

            # 2) Actively moving away from nearest slime -> survives & achieves high threat_evasion_rate!
            st2 = env.reset(960001, scenario=suite['scenario'])
            obs2 = st2['observation']
            evade_rows = []
            for _ in range(70):
                threats = obs2.get('threats') or []
                act = {'verb': 'wait'}
                if threats:
                    tx, ty = float(threats[0]['x']), float(threats[0]['y'])
                    best_d = -1.0
                    for cand in obs2['allowed_actions']:
                        if cand.get('verb') == 'move':
                            _, sx, sy = cand['target_id'].split('_')
                            d = (float(sx) - tx) ** 2 + (float(sy) - ty) ** 2
                            if d > best_d:
                                best_d = d
                                act = cand
                nxt2 = env.step(act)
                evade_rows.append(dict(observation=obs2, next_observation=nxt2['observation'],
                                       action=act, transition=nxt2['transition']))
                obs2 = nxt2['observation']
                if nxt2['terminated'] or int(obs2['tick']) >= 140 + suite['horizon_ticks']:
                    break
            m_evade = ev.evaluate_episode(evade_rows, horizon_tick=suite['horizon_ticks'], relative_horizon=True)
            self.assertTrue(m_evade['alive'], 'Actively stepping away from night threat must survive')
            self.assertIsNotNone(m_evade['threat_evasion_rate'])
            self.assertGreater(m_evade['threat_evasion_rate'], 0.75)

    @unittest.skipUnless(GODOT_BIN.is_file(), 'Local headless Godot binary required')
    def test_curriculum_scenarios_trigger_thirst_and_hunger_metrics_non_nan(self):
        with GodotEnv(
            str(GODOT_BIN), str(ROOT),
            map_mode='infinite_freeze', navigation_steps=4,
            env_script='godot_tests/m9_curriculum_environment.gd',
        ) as env:
            # E2 thirst acute triggers non-None fear_of_thirst_rate immediately
            st_w = env.reset(960002, scenario=ev.EVAL_SUITES['E2_thirst_acute']['scenario'])
            obs = st_w['observation']
            rows = []
            for _ in range(5):
                nxt = env.step({'verb': 'wait'})
                rows.append(dict(observation=obs, next_observation=nxt['observation'],
                                 action={'verb': 'wait'}, transition=nxt['transition']))
                obs = nxt['observation']
            mw = ev.evaluate_episode(rows, horizon_tick=60, relative_horizon=True)
            self.assertIsNotNone(mw['fear_of_thirst_rate'])
            self.assertEqual(mw['thirst_low_decisions'], 5)

            # E3 hunger acute triggers non-None fear_of_hunger_rate immediately
            st_f = env.reset(960002, scenario=ev.EVAL_SUITES['E3_hunger_acute']['scenario'])
            obs = st_f['observation']
            rows_f = []
            for _ in range(5):
                act = {'verb': 'eat'} if {'verb': 'eat'} in obs['allowed_actions'] else {'verb': 'wait'}
                nxt = env.step(act)
                rows_f.append(dict(observation=obs, next_observation=nxt['observation'],
                                   action=act, transition=nxt['transition']))
                obs = nxt['observation']
            mf = ev.evaluate_episode(rows_f, horizon_tick=60, relative_horizon=True)
            self.assertIsNotNone(mf['fear_of_hunger_rate'])
            self.assertTrue(mf['ate_after_low'])


class WetwareDynamicsExplorerTests(unittest.TestCase):
    @unittest.skipUnless((PHASE0_DIR / 'libraries' / '22554' / 'manifest.json').is_file(), 'Phase 0 output required')
    def test_real_phase0_libraries_gate_s_and_epsilon_parity(self):
        # 22554 passed Gate-S -> loads cleanly with SHA-256 verification
        lib_22554 = mde.load_dynamics_library(PHASE0_DIR / 'libraries' / '22554' / 'manifest.json', require_gate_s=True)
        self.assertEqual(lib_22554['manifest']['chip'], '22554')

        # 22021 failed Gate-S (anti-persistent H=0.360) -> blocked when require_gate_s=True
        with self.assertRaises(ValueError):
            mde.load_dynamics_library(PHASE0_DIR / 'libraries' / '22021' / 'manifest.json', require_gate_s=True)

        # Mean epsilon MUST strictly equal eps_base across ALL 4 stochastic arms
        for arm in ('real', 'shuffled', 'white', 'iaaft'):
            ex = mde.WetwareDynamicsExplorer(arm, lib_22554, seed=42, eps_base=0.10)
            self.assertAlmostEqual(float(ex.eps_stream.mean()), 0.10, delta=1e-4)
            self.assertGreaterEqual(float(ex.eps_stream.min()), 0.02 - 1e-9)
            self.assertLessEqual(float(ex.eps_stream.max()), 0.25 + 1e-9)

    @unittest.skipUnless((PHASE0_DIR / 'libraries' / '22554' / 'manifest.json').is_file(), 'Phase 0 output required')
    def test_anti_leakage_and_homeostatic_safety_gate_and_m2_budget(self):
        lib = mde.load_dynamics_library(PHASE0_DIR / 'libraries' / '22554' / 'manifest.json')
        ex1 = mde.WetwareDynamicsExplorer('real', lib, seed=7, eps_base=0.10, homeostatic_gate=True)
        ex2 = mde.WetwareDynamicsExplorer('real', lib, seed=7, eps_base=0.10, homeostatic_gate=True)

        obs_clean = dict(player=dict(x=10, y=10, health=95, thirst=85, hunger=85), threats=[], visited_count=5, r1=0.01)
        obs_tampered = dict(player=dict(x=10, y=10, health=95, thirst=85, hunger=85), threats=[], visited_count=999, r1=-9.0, terminated=True)
        s1 = ex1.step_signal(obs_clean, step_index=10)
        s2 = ex2.step_signal(obs_tampered, step_index=10)
        self.assertEqual(s1['eps'], s2['eps'])
        self.assertEqual(s1['burst_z'], s2['burst_z'])
        np.testing.assert_allclose(s1['dir_xy'], s2['dir_xy'])

        # Homeostatic safety gate shuts off exploratory perturbation in stress
        obs_thirsty = dict(player=dict(x=10, y=10, health=95, thirst=20, hunger=85), threats=[])
        st = ex1.step_signal(obs_thirsty, step_index=10)
        self.assertTrue(st['in_stress'])
        self.assertFalse(st['gated_safe'])
        self.assertEqual(st['eps'], 0.0)
        np.testing.assert_allclose(st['dir_xy'], [0.0, 0.0])

        # M2 entropy weights strictly conserve mean == 1.0
        mod = mde.WetwareEntropyModulator(eta=0.35, enabled=True)
        rows = [dict(mea_burst_z=float(z)) for z in lib['arrays']['real_burst_z'][:128]]
        w = mod.batch_weights(rows)
        self.assertAlmostEqual(float(np.mean(w)), 1.0, places=9)
        self.assertTrue(all(x > 0.0 for x in w))

    @unittest.skipUnless((PHASE0_DIR / 'libraries' / '22554' / 'manifest.json').is_file(), 'Phase 0 output required')
    def test_spontaneous_study_dry_run_and_2x2_attribution(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = study.build_parser().parse_args([
                '--library-manifest', str(PHASE0_DIR / 'libraries' / '22554' / 'manifest.json'),
                '--out', str(Path(tmp) / 'study_22554'),
                '--substudy', '2B_dual_drive', '--enable-m2', '--dry-run',
                '--seeds', '16', '--eval-worlds', '16', '--n-boot', '400',
            ])
            rep = study.run_study(args)
            self.assertEqual(rep['chip'], '22554')
            self.assertIn(rep['outcome']['verdict'], ('weak_positive', 'dual_weak_positive', 'strong_bio_specific_positive'))
            for arm_k in ('B', 'C', 'G1', 'G2'):
                self.assertAlmostEqual(rep['eps_parity_audit'][arm_k]['mean_eps'], 0.10, delta=1e-4)

            # Zero biological effect strength -> negative (D - A passes, B - C crosses 0)
            args_null = study.build_parser().parse_args([
                '--library-manifest', str(PHASE0_DIR / 'libraries' / '22554' / 'manifest.json'),
                '--out', str(Path(tmp) / 'study_null'),
                '--dry-run', '--dry-run-strength', '0.0',
                '--seeds', '16', '--eval-worlds', '16', '--n-boot', '400',
            ])
            rep_null = study.run_study(args_null)
            self.assertEqual(rep_null['outcome']['verdict'], 'negative')

            # 2x2 historical attribution decomposition dry-run
            a2 = attr2x2.build_parser().parse_args([
                '--out', str(Path(tmp) / 'attr2x2'), '--dry-run',
                '--seeds', '16', '--worlds', '24', '--n-boot', '400',
                '--dry-train-eff', '1.5', '--dry-infer-eff', '0.7',
            ])
            r2 = attr2x2.run_2x2(a2)
            self.assertAlmostEqual(r2['decomposition']['pure_training_effect_under_masked_eval']['mean'], 1.5, delta=0.3)
            self.assertAlmostEqual(r2['decomposition']['inference_prompt_effect_on_masked_model']['mean'], 0.7, delta=0.3)


if __name__ == '__main__':
    unittest.main()
