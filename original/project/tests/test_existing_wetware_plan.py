"""Comprehensive unit & integration tests for the existing-data wetware training adjustments."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
ODDBALL_ROOT = ROOT / 'standalone' / 'mea_oddball'
for extra in (ROOT, ODDBALL_ROOT):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

from oddball import endogenous_library as endo
from oddball import signals
from tools import audit_existing_clock_r1 as clock_audit
from tools import build_existing_libraries
from tools import launch_existing_wetware_gpu as campaign
from tools import r2_guard as guard
from tools import run_wetware_stage0 as stage0
from tools import stress_curriculum


def _obs(x=0, y=0, health=90.0, thirst=85.0, hunger=85.0, berries=1,
         water_xy=None, food_xy=None, threat_dist=None,
         visited_count=10, r1=0.01, terminated=False):
    nearby = []
    if water_xy is not None:
        nearby.append(dict(id='w1', kind='water', x=water_xy[0], y=water_xy[1], available=True))
    if food_xy is not None:
        nearby.append(dict(id='f1', kind='berry', x=food_xy[0], y=food_xy[1], available=True))
    threats = []
    if threat_dist is not None:
        threats.append(dict(id='t1', distance=threat_dist, alive=True))
    return dict(
        player=dict(x=x, y=y, health=health, thirst=thirst, hunger=hunger),
        inventory=dict(berries=berries),
        nearby=nearby,
        threats=threats,
        visited_count=visited_count,
        r1=r1,
        reward=r1,
        terminated=terminated,
        next_action='move_east',
    )


class EndogenousCausalLibraryTests(unittest.TestCase):
    def test_strict_past_only_causality_never_peeks_at_window_t(self):
        rng = np.random.default_rng(2026)
        counts_a = rng.poisson(3.0, (120, 18)).astype(float)
        counts_b = counts_a.copy()
        # Mutate window t=95 massively in counts_b; its own assigned bin at t=95 MUST NOT change!
        counts_b[95, :] = 5000.0
        ext_a = endo.extract_causal_std_rows(counts_a, fit_end=80, history_k=4)
        ext_b = endo.extract_causal_std_rows(counts_b, fit_end=80, history_k=4)

        rows_a_95 = [r for r in ext_a['habituation_rows'] if r['t'] == 95]
        rows_b_95 = [r for r in ext_b['habituation_rows'] if r['t'] == 95]
        self.assertEqual(len(rows_a_95), 3)
        for ra, rb in zip(rows_a_95, rows_b_95):
            self.assertEqual(ra['site'], rb['site'])
            self.assertEqual(ra['bin'], rb['bin'])
            self.assertEqual(ra['past_load'], rb['past_load'])

    def test_existing_archived_organoids_pass_gates_while_ctime_and_22554_fail(self):
        out_root = ROOT / 'output'
        ds_22021 = build_existing_libraries.load_22021_1123(out_root)
        real_22021 = endo.evaluate_and_build_bundle(
            ds_22021['counts'],
            dataset_id=ds_22021['dataset_id'],
            chip_id=ds_22021['chip_id'],
            block_ids=ds_22021['block_ids'],
            fit_end=ds_22021['fit_end'],
            channel_ids=ds_22021['channel_ids'],
            channel_xy_um=ds_22021['channel_xy_um'],
            history_k=ds_22021['history_k'],
            gate_seed=11,
        )
        self.assertTrue(real_22021['summary']['dual_drive_unlocked'])
        self.assertEqual(real_22021['summary']['n_channels'], 233)
        self.assertTrue({96, 737, 804}.isdisjoint(set(ds_22021['channel_ids'])))
        self.assertGreater(real_22021['summary']['rho_lib'], 0.44)
        self.assertEqual(len(real_22021['summary']['usable_habituation_sites']), 3)

        # Pre-stimulus-only [-300, -1) ms sensitivity check confirms zero stimulation-artifact reliance
        ds_22021_pre = build_existing_libraries.load_22021_1123(out_root, pre_stim_only=True)
        real_22021_pre = endo.evaluate_and_build_bundle(
            ds_22021_pre['counts'],
            dataset_id=ds_22021_pre['dataset_id'],
            chip_id='22021',
            block_ids=ds_22021_pre['block_ids'],
            fit_end=ds_22021_pre['fit_end'],
            channel_ids=ds_22021_pre['channel_ids'],
            channel_xy_um=ds_22021_pre['channel_xy_um'],
            history_k=ds_22021_pre['history_k'],
            gate_seed=13,
        )
        self.assertTrue(real_22021_pre['summary']['dual_drive_unlocked'])
        self.assertGreater(real_22021_pre['summary']['rho_lib'], 0.38)

        # Afternoon chip 25449 QC audit must mark G1 rejected
        qc_25449 = build_existing_libraries.audit_25449_afternoon_qc(out_root)
        self.assertIsNotNone(qc_25449)
        self.assertFalse(qc_25449['g1_passed'])
        self.assertEqual(qc_25449['dose_real_pulses'], 0)

        ctime_22021 = endo.evaluate_and_build_bundle(
            ds_22021['counts'],
            dataset_id='22021_ctime',
            chip_id='22021',
            block_ids=ds_22021['block_ids'],
            fit_end=ds_22021['fit_end'],
            channel_ids=ds_22021['channel_ids'],
            channel_xy_um=ds_22021['channel_xy_um'],
            history_k=ds_22021['history_k'],
            gate_seed=12,
            permute_timeline_seed=999,
            force_export_for_control=True,
        )
        self.assertFalse(ctime_22021['summary']['dual_drive_unlocked'])
        self.assertLess(abs(ctime_22021['summary']['rho_all_sites']), 0.08)

        ds_28080 = build_existing_libraries.load_28080_1116_long(out_root)
        real_28080 = endo.evaluate_and_build_bundle(
            ds_28080['counts'],
            dataset_id=ds_28080['dataset_id'],
            chip_id=ds_28080['chip_id'],
            block_ids=ds_28080['block_ids'],
            fit_end=ds_28080['fit_end'],
            channel_ids=ds_28080['channel_ids'],
            channel_xy_um=ds_28080['channel_xy_um'],
            history_k=ds_28080['history_k'],
            gate_seed=21,
        )
        self.assertTrue(real_28080['summary']['dual_drive_unlocked'])
        self.assertGreater(real_28080['summary']['rho_lib'], 0.30)

        ds_22554 = build_existing_libraries.load_cpp_spikes_session(
            out_root,
            dataset_id='22554_1119_qc_negative',
            chip_id='22554',
            spikes_rel='22554_full_20261119_1224/records/session/cpp-spikes.jsonl',
            history_k=3,
        )
        real_22554 = endo.evaluate_and_build_bundle(
            ds_22554['counts'],
            dataset_id=ds_22554['dataset_id'],
            chip_id='22554',
            block_ids=ds_22554['block_ids'],
            fit_end=ds_22554['fit_end'],
            channel_ids=ds_22554['channel_ids'],
            channel_xy_um=ds_22554['channel_xy_um'],
            history_k=3,
            gate_seed=31,
        )
        self.assertFalse(real_22554['summary']['dual_drive_unlocked'])


class ObservableDeficitReliefAndGuardTests(unittest.TestCase):
    def test_transition_relief_switches_on_resource_approach_and_ignores_outcomes(self):
        # Thirsty at (0,0) with water at (10,0): moving closer to (6,0) triggers 'sham' relief!
        prev_obs_a = _obs(x=0, y=0, thirst=25.0, water_xy=(10, 0), visited_count=5, r1=0.01)
        next_obs_a = _obs(x=4, y=0, thirst=24.6, water_xy=(10, 0), visited_count=5, r1=0.01)
        # Tamper hidden outcome fields drastically
        prev_obs_b = _obs(x=0, y=0, thirst=25.0, water_xy=(10, 0), visited_count=900, r1=-9.0, terminated=True)
        next_obs_b = _obs(x=4, y=0, thirst=24.6, water_xy=(10, 0), visited_count=999, r1=+9.0, terminated=True)

        c1, c2 = signals.EpisodeContext(), signals.EpisodeContext()
        e1 = c1.encode_step(signals.SensoryState.from_observation(next_obs_a), 'dual',
                            prev_state=signals.SensoryState.from_observation(prev_obs_a))
        e2 = c2.encode_step(signals.SensoryState.from_observation(next_obs_b), 'dual',
                            prev_state=signals.SensoryState.from_observation(prev_obs_b))
        self.assertEqual(e1, e2)
        self.assertEqual(e1['channel'], 'evoked')
        self.assertEqual(e1['bin'], 'sham')
        self.assertTrue(e1['deficit_relieved'])

        # Moving away from water while thirsty triggers 'W' real aversion
        away_obs = _obs(x=-4, y=0, thirst=24.6, water_xy=(10, 0))
        e_away = c1.encode_step(signals.SensoryState.from_observation(away_obs), 'dual',
                                prev_state=signals.SensoryState.from_observation(prev_obs_a))
        self.assertEqual(e_away['site'], 'W')
        self.assertEqual(e_away['bin'], 'real')
        self.assertFalse(e_away['deficit_relieved'])

    def test_decoupled_gae_preserves_landmark_departure_steps_and_pure_r1_critic_returns(self):
        rng = np.random.default_rng(19)
        r1 = np.full(128, 0.01)
        # Step 10 leaves a landmark (tiny |R1| = 0.0004), Step 20 drinks water (+0.45 jump)
        r1[10] = 0.0004
        r1[20] = 0.45
        raw_sig = rng.choice([-1.0, 1.0], 128).astype(float)
        raw_sig[10] = 1.0

        stats = guard.track_r1_stats(None, r1)
        old_r2 = guard.shape_calibrated(r1, raw_sig, stats, target_share=0.08)
        new_r2 = guard.shape_for_decoupled_gae(r1, raw_sig, stats, target_share=0.08)

        # Old per-step 0.85*|r1| crushed step 10 to <= 0.00034; new floor preserves full unclipped ~0.0010 signal
        self.assertLessEqual(abs(old_r2[10]), 0.0004)
        self.assertGreater(abs(new_r2[10]), 2.5 * abs(old_r2[10]))
        self.assertGreater(abs(new_r2[10]), 0.0009)
        sh = guard.shares(r1, new_r2)
        level, _ = guard.verdict(sh)
        self.assertNotEqual(level, 'redline')
        self.assertLess(sh['magnitude'], guard.SHARE_REDLINE)
        self.assertLess(sh['variance'], guard.SHARE_REDLINE)

        # Verify Critic `return` is 100% untouched pure R1 while Actor `advantage` gets bounded SNR
        rows = [
            dict(r1_reward=float(r1[i]), reward=float(r1[i]), r2_shaped=float(new_r2[i]),
                 value=0.05, duration=1, terminal=False, boundary=(i == 127))
            for i in range(128)
        ]
        pure_rows = [dict(r) for r in rows]
        from agent.m9_run import finish_trajectory
        finish_trajectory(pure_rows, 0.05, 0.99985, 0.999)
        diag = stage0.inject_decoupled_advantage(rows, 0.05, 0.99985, 0.999, beta_adv=0.18)
        np.testing.assert_allclose([r['return'] for r in rows], [r['return'] for r in pure_rows], rtol=0, atol=1e-12)
        self.assertGreater(diag['adv_snr'], 0.05)
        self.assertLess(diag['adv_snr'], 0.30)


class StressStagingAndCampaignTests(unittest.TestCase):
    def test_pinned_stress_scenes_stage_with_verified_sha256(self):
        log_root = ROOT / 'output/runtime_tests'
        log_root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=log_root) as tmp:
            staged = stress_curriculum.stage_pinned_stress_projects(Path(tmp))
            self.assertEqual(set(staged.keys()), {'low_health', 'thirst', 'hunger'})
            for root_p in staged.values():
                self.assertTrue((root_p / 'project.godot').is_file())

    def test_historical_clock_audit_and_seven_arm_campaign_dry_run(self):
        log_root = ROOT / 'output/runtime_tests'
        log_root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=log_root) as tmp:
            audit_res = clock_audit.run_audit(Path(tmp) / 'clock_audit', n_boot=300)
            self.assertIn('22021_baseline_minus_masked', audit_res['short_horizon_clock_audit'])
            self.assertIn('thirst', audit_res['tick60_confirmatory_stress_audit'])

            camp_args = campaign.build_parser().parse_args([
                '--mode', 'dual',
                '--libraries-dir', str(ROOT / 'output/existing_wetware_libraries'),
                '--out', str(Path(tmp) / 'camp'),
                '--dry-run', '--seeds', '16', '--eval-worlds', '24', '--n-boot', '300',
            ])
            res = campaign.run_campaign(camp_args)
            s3r = res['stage3_stage4']
            self.assertEqual(s3r['arms'], ['A', 'B', 'C', 'C_time', 'D', 'E', 'F'])
            self.assertTrue(s3r['channel_bound_audit']['passed'])
            self.assertFalse(s3r['channel_bound_audit']['ctime_leaks'])
            self.assertEqual(s3r['outcome']['verdict'], 'stable_positive')
            self.assertGreater(s3r['comparisons']['B_minus_C']['ci_low'], 0.0)
            self.assertGreater(s3r['comparisons']['B_minus_Ctime']['ci_low'], 0.0)


class MultiGpuRunnerAndComparisonReportTests(unittest.TestCase):
    def test_encode_rollout_contexts_passes_prev_state_for_crisis_relief(self):
        from tools import gpu_existing_data_runner as runner

        rows = [
            dict(
                episode_start=True,
                observation=_obs(x=0, y=0, thirst=22.0, water_xy=(10, 0), visited_count=1, r1=0.01),
                next_observation=_obs(x=4, y=0, thirst=21.6, water_xy=(10, 0), visited_count=5, r1=0.01),
                r1_reward=0.01,
            ),
            dict(
                episode_start=False,
                observation=_obs(x=4, y=0, thirst=21.6, water_xy=(10, 0), visited_count=5, r1=0.01),
                next_observation=_obs(x=0, y=0, thirst=21.2, water_xy=(10, 0), visited_count=5, r1=0.01),
                r1_reward=0.01,
            ),
        ]
        ctxs, r1_arr = runner.encode_rollout_contexts(rows, r2_mode='dual')
        self.assertEqual(len(ctxs), 2)
        self.assertEqual(len(r1_arr), 2)
        # Step 0 approached water while thirsty -> sham relief!
        self.assertEqual(ctxs[0]['channel'], 'evoked')
        self.assertEqual(ctxs[0]['bin'], 'sham')
        self.assertTrue(ctxs[0]['deficit_relieved'])
        # Step 1 moved away from water while thirsty -> real aversion on site W!
        self.assertEqual(ctxs[1]['channel'], 'evoked')
        self.assertEqual(ctxs[1]['site'], 'W')
        self.assertEqual(ctxs[1]['bin'], 'real')
        self.assertFalse(ctxs[1]['deficit_relieved'])

    def test_library_stage0_stage3_dry_run_report_and_deploy_packaging(self):
        from tools import deploy_wetware_gpu as deploy
        from tools import gpu_existing_data_runner as runner
        from tools import report_wetware_comparison as rwc

        lib_dir = ROOT / 'output/existing_wetware_libraries'
        log_root = ROOT / 'output/runtime_tests'
        log_root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=log_root) as tmp:
            tmp_p = Path(tmp)
            # 1) Stage 0 dry-run
            s0_dir = tmp_p / 'stage0'
            st0 = runner.orchestrate_library_stage0(
                out_dir=s0_dir, gpus=[0], common_cli=[], r2_mode='dual',
                tune_seeds=list(range(100, 112)), confirm_seeds=list(range(200, 212)),
                eval_worlds=list(range(960000, 960016)), ladder_tiers=(1, 2),
                n_boot=300, dry_run=True,
            )
            self.assertTrue(st0['confirmed'])
            self.assertIsNotNone(st0['rho_min'])
            self.assertLessEqual(st0['rho_min'], 0.46)

            # 2) Stage 3 dry-run (positive alignment on clean 22021 233-ROI library)
            bundle = runner.resolve_library_bundle(libraries_dir=lib_dir, role='primary')
            self.assertEqual(bundle['dataset_id'], '22021_1123_oddball_clean')
            self.assertGreater(bundle['rho_lib'], 0.44)

            s3_dir = tmp_p / 'stage3'
            seeds = list(range(300, 316))
            worlds = list(range(970000, 970020))
            jobs = []
            for s in seeds:
                for aname, r2_src in runner.LIBRARY_SEVEN_ARMS:
                    j = dict(arm_name=aname, r2_mode='dual', r2_source=r2_src, explorer_arm='none',
                             seed=s, tier=1, use_curriculum=True, rho_lib=bundle['rho_lib'])
                    if r2_src.startswith('wetware_'):
                        j.update(
                            habituation_library=bundle['habituation_library'],
                            evoked_library=bundle['evoked_library'],
                            ctime_habituation_library=bundle['ctime_habituation_library'],
                            ctime_evoked_library=bundle['ctime_evoked_library'],
                        )
                    jobs.append(j)
            res = runner.run_parallel_jobs(jobs, [0], s3_dir, [], dry_run=True, eval_worlds=worlds)
            st3 = runner.summarize_library_stage3(
                results=res, seeds=seeds, out_dir=s3_dir, n_boot=300,
                tier_cfg=runner.TIERS[1], r2_mode='dual', bundle=bundle,
                channel_bound_audit=dict(passed=True, real_auc=0.87, ideal_auc=1.0, shuffled_auc=0.50, ctime_auc=0.49),
                stage0_sha256=st0['sha256'], dry_run=True,
            )
            self.assertIn(st3['outcome']['verdict'], ('weak_positive', 'dual_weak_positive'))
            self.assertGreater(st3['comparisons']['B_minus_C']['E1_new_cells']['ci_low'], 0.0)
            self.assertGreater(st3['comparisons']['B_minus_Ctime']['E1_new_cells']['ci_low'], 0.0)

            # 3) Null-effect dry-run must classify as negative
            s3_null_dir = tmp_p / 'stage3_null'
            res_null = runner.run_parallel_jobs(jobs, [0], s3_null_dir, [], dry_run=True, eval_worlds=worlds, dry_run_strength=0.0)
            st3_null = runner.summarize_library_stage3(
                results=res_null, seeds=seeds, out_dir=s3_null_dir, n_boot=300,
                tier_cfg=runner.TIERS[1], r2_mode='dual', bundle=bundle,
                channel_bound_audit=dict(passed=True), stage0_sha256=st0['sha256'], dry_run=True,
            )
            self.assertEqual(st3_null['outcome']['verdict'], 'negative')

            # 4) Report + 3 PNG charts generation
            rep_dir = tmp_p / 'report'
            rep_out = rwc.generate_report(
                manifest_path=lib_dir / 'gate_manifest.json',
                stage0_path=s0_dir / 'stage0_summary.json',
                stage3_path=s3_dir / 'stage3_summary.json',
                stage4_path=None,
                out_dir=rep_dir,
            )
            self.assertTrue(Path(rep_out['report_md']).is_file())
            md_text = Path(rep_out['report_md']).read_text(encoding='utf-8')
            self.assertIn('DRY-RUN', md_text)
            self.assertIn('22021_1123_oddball_clean', md_text)
            self.assertEqual(len(rep_out['figures']), 3)
            for fig in rep_out['figures']:
                self.assertTrue(Path(fig).is_file())
                self.assertGreater(Path(fig).stat().st_size, 5000)

            # 5) Deploy tarball & remote script builder sanity check
            blob, sha_hex, n_files = deploy.build_stage_tarball(ROOT)
            self.assertGreater(n_files, 30)
            self.assertEqual(len(sha_hex), 64)
            script = deploy.build_remote_run_script(
                remote_dir='/tmp/test_wetware', remote_python='python3',
                remote_godot='/tmp/godot', phase='funnel', gpus='0,1',
                seeds=16, confirm_seeds=16, eval_worlds=48, n_boot=10000,
                tier=1, ladder_tiers='1,2,3', r2_mode='dual', include_stage4=True,
            )
            self.assertIn('library_smoke', script)
            self.assertIn('library_stage0', script)
            self.assertIn('library_stage3', script)
            self.assertIn('library_stage4', script)


if __name__ == '__main__':
    unittest.main()
