"""Comprehensive unit & integration tests for the v2 wetware training technical pipeline."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
ODDBALL_ROOT = ROOT / 'standalone' / 'mea_oddball'
if str(ODDBALL_ROOT) not in sys.path:
    sys.path.insert(0, str(ODDBALL_ROOT))

import build_library_v2
from oddball import analysis_v2, design_v2, library_v2, session_v2, signals
from tools import eval_survival_drives as ev
from tools import r2_guard as guard
from tools import run_wetware_stage0 as stage0
from tools import run_wetware_stage3 as stage3


def _make_obs(x=0, y=0, health=100.0, thirst=90.0, hunger=90.0, visited=10,
              r1=0.01, terminated=False, threats=None, nearby=None):
    return dict(
        player=dict(x=x, y=y, health=health, thirst=thirst, hunger=hunger),
        visited_count=visited,
        r1=r1,
        reward=r1,
        terminated=terminated,
        next_action='move_north',
        threats=threats or [],
        nearby=nearby or [],
    )


class R2GuardCalibratedTests(unittest.TestCase):
    def test_shape_calibrated_hits_target_band_and_never_exceeds_r1(self):
        rng = np.random.default_rng(42)
        # Mimic real M9 bottleneck R1: mostly ~0.01 survival ticks plus occasional 0.08 potential jumps
        r1 = np.full(512, 0.01) + rng.normal(0.0, 0.002, 512)
        jumps = rng.choice(512, 48, replace=False)
        r1[jumps] += rng.normal(0.0, 0.08, 48)
        sig = rng.choice([-1.0, 0.0, 1.0], 512)

        stats = guard.track_r1_stats(None, r1)
        r2 = guard.shape_calibrated(r1, sig, stats, target_share=0.08)
        sh = guard.shares(r1, r2)
        level, _ = guard.verdict(sh)
        self.assertEqual(level, 'ok')
        self.assertLessEqual(sh['magnitude'], 0.10)
        self.assertLessEqual(sh['variance'], 0.10)
        self.assertGreater(sh['magnitude'], 0.03)
        self.assertEqual(sh['larger_than_r1'], 0)

        with self.assertRaises(ValueError):
            guard.shape_calibrated(r1, sig, stats, target_share=0.25)

    def test_channel_bound_check_distinguishes_valid_vs_leaky(self):
        rng = np.random.default_rng(7)
        group = np.repeat(np.arange(8), 40)
        label = rng.random(320) < 0.5
        ideal = np.where(label, 1.0, -1.0)
        noisy_valid = ideal * 0.5 + rng.normal(0.0, 0.8, 320)
        shuf_clean = rng.permutation(noisy_valid)
        ok = guard.channel_bound_check(noisy_valid, ideal, shuf_clean, label, group, range(4, 8), seed=11, shuffles=100)
        self.assertTrue(ok['passed'])
        # If shuffled arm somehow leaks the label, gate must reject
        bad = guard.channel_bound_check(noisy_valid, ideal, ideal, label, group, range(4, 8), seed=11, shuffles=100)
        self.assertFalse(bad['passed'])
        self.assertTrue(bad['shuffled_leaks'])


class SignalsContractTests(unittest.TestCase):
    def test_strict_causal_isolation_ignores_outcome_fields(self):
        obs_a = _make_obs(x=12, y=8, health=85, thirst=70, hunger=75, visited=5, r1=0.01, terminated=False)
        obs_b = _make_obs(x=12, y=8, health=85, thirst=70, hunger=75, visited=999, r1=-5.0, terminated=True)
        sa = signals.SensoryState.from_observation(obs_a)
        sb = signals.SensoryState.from_observation(obs_b)
        self.assertEqual(sa, sb)
        c1, c2 = signals.EpisodeContext(), signals.EpisodeContext()
        for mode in signals.MODES:
            c1.reset(); c2.reset()
            self.assertEqual(c1.encode_step(sa, mode), c2.encode_step(sb, mode))

    def test_4x4_patch_cycling_vs_loitering(self):
        ctx = signals.EpisodeContext(patch_tiles=4, history=8)
        # Steadily moving +4 tiles along x each decision cycles H -> W -> F -> H
        moving_bins = []
        for step in range(9):
            st = signals.SensoryState(x=step * 4, y=0, health=100, thirst=90, hunger=90)
            moving_bins.append(ctx.encode_step(st, 'explore')['habituation_bin'])
        self.assertNotIn('high', moving_bins)

        ctx.reset()
        loiter_bins = []
        for _ in range(9):
            st = signals.SensoryState(x=1, y=1, health=100, thirst=90, hunger=90)
            loiter_bins.append(ctx.encode_step(st, 'explore')['habituation_bin'])
        self.assertEqual(loiter_bins[-1], 'high')

    def test_homeostatic_priority_and_dual_drive_switching(self):
        ctx = signals.EpisodeContext()
        # Health/threat takes priority over thirst & hunger
        st_hp = signals.SensoryState(x=0, y=0, health=45, thirst=10, hunger=10)
        self.assertEqual(ctx.encode_step(st_hp, 'dual')['site'], 'H')
        self.assertEqual(ctx.encode_step(st_hp, 'dual')['channel'], 'evoked')
        st_w = signals.SensoryState(x=0, y=0, health=90, thirst=25, hunger=20)
        self.assertEqual(ctx.encode_step(st_w, 'dual')['site'], 'W')
        st_f = signals.SensoryState(x=0, y=0, health=90, thirst=80, hunger=20)
        self.assertEqual(ctx.encode_step(st_f, 'dual')['site'], 'F')
        st_safe = signals.SensoryState(x=0, y=0, health=90, thirst=80, hunger=80)
        enc = ctx.encode_step(st_safe, 'dual')
        self.assertEqual(enc['channel'], 'habituation')

    def test_five_signal_sources_alignment_inversion_and_noise(self):
        hab_lib = dict(
            usable_sites=['H', 'W', 'F'],
            sites={s: {'low': [1.2, 1.0], 'mid': [0.0, 0.1], 'high': [-1.1, -1.0]} for s in ('H', 'W', 'F')},
        )
        evk_lib = dict(
            usable_sites=['H', 'W', 'F'],
            sites={s: {'real': [1.3, 1.1]} for s in ('H', 'W', 'F')},
            sham=[-1.2, -1.0],
        )
        ctx = signals.EpisodeContext()
        contexts = []
        for i in range(600):
            st = signals.SensoryState(
                x=(i * 4) if i % 3 != 0 else 0, y=0,
                health=90, thirst=20 if i % 5 == 0 else 85, hunger=85,
            )
            contexts.append(ctx.encode_step(st, 'dual'))

        src_none = signals.WetwareSignalSource('dual', 'none', 1)
        self.assertTrue(np.all(src_none.score_batch(contexts) == 0.0))

        ideal = signals.WetwareSignalSource('dual', 'ideal', 1).score_batch(contexts)
        noisy = signals.WetwareSignalSource('dual', 'noisy', 1, rho=0.4).score_batch(contexts)
        emp_rho = float(np.corrcoef(ideal, noisy)[0, 1])
        self.assertAlmostEqual(emp_rho, 0.4, delta=0.08)

        real = signals.WetwareSignalSource('dual', 'wetware_real', 7, habituation_library=hab_lib, evoked_library=evk_lib).score_batch(contexts)
        inv = signals.WetwareSignalSource('dual', 'wetware_inverted', 7, habituation_library=hab_lib, evoked_library=evk_lib).score_batch(contexts)
        shuf = signals.WetwareSignalSource('dual', 'wetware_shuffled', 7, habituation_library=hab_lib, evoked_library=evk_lib).score_batch(contexts)
        np.testing.assert_allclose(inv, -real)
        self.assertGreater(float(np.corrcoef(ideal, real)[0, 1]), 0.90)
        self.assertLess(abs(float(np.corrcoef(ideal, shuf)[0, 1])), 0.12)


class OddballV2Tests(unittest.TestCase):
    def test_v2_plan_ladder_confirm_and_budget(self):
        pl = design_v2.plan(20261123)
        self.assertEqual(pl['schema'], 'mea.oddball.v2')
        self.assertEqual(pl['amplitudes_mv'], [300, 450, 600])
        total_real = sum(design_v2.real_pulses(b) for b in pl['blocks'])
        self.assertEqual(total_real, 3 * 60 + 90 + 1080)
        self.assertLessEqual(total_real, design_v2.MAX_REAL_PULSES)
        self.assertEqual(design_v2.pulse_bits(2.981901, 600), 201)

    def test_causal_butterworth_has_zero_preringing_and_clean_passband(self):
        sos = analysis_v2.butter_bandpass_sos(300.0, 3000.0, fs=20000.0, order=4)
        # Strict causality: zero output strictly before step impulse at t = 400
        step = np.zeros(2000)
        step[400:] = 100.0
        y = analysis_v2.sosfilt_causal(sos, step)
        self.assertTrue(np.all(y[:400] == 0.0))

        # Frequency response: pass 1000 Hz, attenuate DC and 8000 Hz
        t = np.arange(4000) / 20000.0
        gain_1k = float(np.std(analysis_v2.sosfilt_causal(sos, np.sin(2 * np.pi * 1000.0 * t))[1000:])) / float(np.std(np.sin(2 * np.pi * 1000.0 * t)[1000:]))
        gain_8k = float(np.std(analysis_v2.sosfilt_causal(sos, np.sin(2 * np.pi * 8000.0 * t))[1000:])) / float(np.std(np.sin(2 * np.pi * 8000.0 * t)[1000:]))
        gain_dc = float(np.abs(analysis_v2.sosfilt_causal(sos, np.ones(4000))[2000:]).max())
        self.assertGreater(gain_1k, 0.95)
        self.assertLess(gain_8k, 0.05)
        self.assertLess(gain_dc, 1e-4)

    def test_blank_and_detect_rejects_artifact_and_catches_evoked_spikes(self):
        rng = np.random.default_rng(123)
        sos = analysis_v2.butter_bandpass_sos(300.0, 3000.0, fs=20000.0, order=4)
        # 3 channels: ch0 has 2 spikes in [5,50)ms, ch1 is quiet, ch2 is saturated past +5ms
        warmup = 512.0 + rng.normal(0.0, 1.5, (3, 4000))
        sigma = analysis_v2.estimate_channel_sigma(warmup, sos=sos)

        anchor = 600  # 30 ms pre-buffer
        raw = 512.0 + rng.normal(0.0, 1.5, (3, 2000))
        # Huge stimulation artifact in [+2, +10) frames (0.1 - 0.5 ms)
        raw[:, anchor + 2:anchor + 6] = 1023.0
        raw[:, anchor + 6:anchor + 10] = 0.0
        # Inject two realistic biphasic negative-first action potentials (~12 sigma) on ch0 at +15 ms and +30 ms
        ap = np.array([-5.0, -14.0, -18.0, -8.0, 6.0, 10.0, 5.0])
        for sp in (anchor + 300, anchor + 600):
            raw[0, sp - 2:sp + 5] += ap
        # Saturate ch2 during +10..+20 ms
        raw[2, anchor + 200:anchor + 400] = 1023.0

        counts, sat = analysis_v2.detect_trial_spikes_causal(raw, anchor, sigma, sos=sos)
        self.assertEqual(int(counts[0]), 2)
        self.assertEqual(int(counts[1]), 0)
        self.assertTrue(bool(sat[2]))
        self.assertEqual(int(counts[2]), 0)

    def test_site_cluster_selection_and_local_roi(self):
        # Build 3 dense clusters at (0,0), (1000,0), (500,900)
        xy = {}
        rates = {}
        ch = 0
        for cx, cy in ((0.0, 0.0), (1000.0, 0.0), (500.0, 900.0)):
            # 5 tight stimulation cluster electrodes within 30 um
            for dx, dy in ((0, 0), (17.5, 0), (-17.5, 0), (0, 17.5), (0, -17.5)):
                xy[ch] = (cx + dx, cy + dy)
                rates[ch] = 2.0
                ch += 1
            # 25 active readout neighbors in 150..400 um ring
            for k in range(25):
                ang = 2 * np.pi * k / 25.0
                r = 220.0 + (k % 3) * 40.0
                xy[ch] = (cx + r * np.cos(ang), cy + r * np.sin(ang))
                rates[ch] = 1.0
                ch += 1
        res = analysis_v2.select_site_clusters(xy, rates)
        self.assertGreaterEqual(res['min_spacing_um'], 500.0)
        for s in design_v2.SITES:
            self.assertEqual(len(res['sites'][s]['cluster_channels']), 5)
        rois = analysis_v2.select_site_rois(xy, rates, res['sites'])
        for s in design_v2.SITES:
            self.assertGreaterEqual(len(rois[s]), 20)

    def test_two_tier_gates_and_dual_library_export(self):
        rng = np.random.default_rng(99)
        pl = design_v2.plan(55)
        main_blocks = [b for b in pl['blocks'] if b['stage'] == 'main']

        def make_rows(with_habituation):
            rows = []
            for blk in main_blocks:
                counts = design_v2.recent_counts(blk['trials'])
                for i, (site, cnt) in enumerate(zip(blk['trials'], counts)):
                    is_real = site is not None
                    bin_name = None if cnt is None else design_v2.bin_of(cnt)
                    # 10 readout channels: ch 0..2 carry signal, ch 3..9 are bursty noise
                    ch_counts = {}
                    for s in design_v2.SITES:
                        vec = rng.poisson(0.4, 10).astype(float)
                        if is_real and s == site:
                            boost = 6.0 if not with_habituation else {'low': 12.0, 'mid': 6.0, 'high': 2.5}[bin_name]
                            vec[:3] += rng.poisson(boost, 3)
                        ch_counts[s] = vec.tolist()
                    rows.append(dict(
                        block=blk['name'], round=blk['round'], kind=blk['kind'], frequent=blk['frequent'],
                        trial=i, site=site or design_v2.SHAM, real=is_real, count=cnt, bin=bin_name,
                        valid=True, channel_counts=ch_counts,
                    ))
            return rows

        with tempfile.TemporaryDirectory() as tmp:
            # Case 1: Evoked ONLY (G2 passes, G3 fails) -> exports ONLY evoked_library.json
            rows_evk, w_evk = build_library_v2.score_rows_with_library_weights(make_rows(False))
            self.assertGreater(sum(w_evk['H'][:3]), 0.70)  # responsive channels dominate weights
            rep1 = build_library_v2.build_from_rows(rows_evk, Path(tmp) / 'evk_only', seed=101, rho_min=0.20, channel_weights=w_evk)
            self.assertTrue(rep1['gates']['evoked_training_allowed'])
            self.assertFalse(rep1['gates']['habituation_training_allowed'])
            self.assertTrue((Path(tmp) / 'evk_only' / 'evoked_library.json').is_file())
            self.assertFalse((Path(tmp) / 'evk_only' / 'habituation_library.json').is_file())

            # Case 2: Evoked + Habituation (G2 + G3 pass) -> exports BOTH libraries
            rows_both, w_both = build_library_v2.score_rows_with_library_weights(make_rows(True))
            rep2 = build_library_v2.build_from_rows(rows_both, Path(tmp) / 'both', seed=102, rho_min=0.20, channel_weights=w_both)
            self.assertTrue(rep2['gates']['evoked_training_allowed'])
            self.assertTrue(rep2['gates']['habituation_training_allowed'])
            self.assertGreater(rep2['gates']['rho_lib'], 0.50)
            self.assertTrue((Path(tmp) / 'both' / 'evoked_library.json').is_file())
            self.assertTrue((Path(tmp) / 'both' / 'habituation_library.json').is_file())

    def test_session_v2_ladder_confirm_and_stop_on_confirm_failure(self):
        class FakeClock:
            def __init__(self): self.t = 1000.0
            def wall(self): return self.t
            def mono(self): return self.t
            def sleep(self, s): self.t += s

        class FakeRig:
            def __init__(self, pass_amp=450, pass_confirm=True):
                self.lsb_mv = 2.981901
                self.faulted = False
                self.pass_amp = pass_amp
                self.pass_confirm = pass_confirm
                self.built_blocks = []
                self.units_state = False
            def prepare(self): return dict(units={'H': [0, 1, 2]}, lsb_mv=self.lsb_mv)
            def open(self, name): self._cur = name
            def units(self, on): self.units_state = on
            def build(self, block, bits):
                self.built_blocks.append((block['name'], bits))
                return [f'h_{i}' for i in range(len(design_v2.chunks(block)))]
            def send(self, handle, event, gate): gate(event)
            def close(self): return Path(self._cur + '.h5')
            def check_warmup(self, path): return dict(passed=True, active_channels=120)
            def evaluate_block(self, path, block, sid):
                rows = []
                counts = design_v2.recent_counts(block['trials'])
                for i, (s, c) in enumerate(zip(block['trials'], counts)):
                    is_real = s is not None and block['stage'] != 'rehearsal'
                    resp = 0.0
                    if is_real:
                        if block['stage'] == 'screen':
                            resp = 10.0 if block['amplitude_mv'] >= self.pass_amp else 0.0
                        elif block['stage'] == 'confirm':
                            resp = 10.0 if self.pass_confirm else 0.0
                        else:
                            resp = 10.0
                    rows.append(dict(trial=i, site=s or 'Z', real=is_real, valid=True,
                                     count=c, bin=None if c is None else design_v2.bin_of(c), response=resp))
                return dict(passed=True, errors=[], rows=rows)

        pl = design_v2.plan(77)
        with tempfile.TemporaryDirectory() as tmp:
            # Escalates 300 -> 450, passes confirm, completes all 12 main blocks
            out1 = Path(tmp) / 'ok'; out1.mkdir()
            rig1 = FakeRig(pass_amp=450, pass_confirm=True)
            st1 = session_v2.SessionV2(rig1, out1, pl, 900.0, 's1', [], FakeClock(), say=lambda *a, **k: None).run()
            self.assertEqual(st1['outcome'], 'complete')
            self.assertEqual(json.loads((out1 / 'decision.json').read_text())['amplitude_mv'], 450)
            self.assertFalse(rig1.units_state)

            # Confirm fails at 300 mV -> stops without escalating to 450/600 mV
            out2 = Path(tmp) / 'fail_confirm'; out2.mkdir()
            rig2 = FakeRig(pass_amp=300, pass_confirm=False)
            st2 = session_v2.SessionV2(rig2, out2, pl, 900.0, 's2', [], FakeClock(), say=lambda *a, **k: None).run()
            self.assertEqual(st2['outcome'], 'stopped')
            names = [n for n, _ in rig2.built_blocks]
            self.assertIn('confirm', names)
            self.assertNotIn('screen450', names)
            self.assertNotIn('screen600', names)


class SurvivalDrivesAndStagesTests(unittest.TestCase):
    def test_evaluate_episode_fear_of_thirst_hunger_death_and_tick_cutoff(self):
        rows = [
            # Step 0: safe -> steps onto new cell (visited 10->11)
            dict(
                observation=_make_obs(x=0, y=0, health=90, thirst=80, hunger=80, visited=10),
                next_observation=_make_obs(x=4, y=0, health=90, thirst=25, hunger=25, visited=11),
                action=dict(verb='navigate'),
                transition=dict(
                    before=dict(tick=0, health=90, thirst=80, hunger=80, visited_count=10, dead=False),
                    after=dict(tick=2, health=90, thirst=25, hunger=25, visited_count=11, dead=False),
                    elapsed_ticks=2,
                ),
            ),
            # Step 1: thirsty & hungry -> drinks water, does NOT step onto new cell (cautious under danger)
            dict(
                observation=_make_obs(x=4, y=0, health=90, thirst=18, hunger=25, visited=11),
                next_observation=_make_obs(x=4, y=0, health=90, thirst=60, hunger=25, visited=11),
                action=dict(verb='drink'),
                transition=dict(
                    before=dict(tick=2, health=90, thirst=18, hunger=25, visited_count=11, dead=False),
                    after=dict(tick=4, health=90, thirst=60, hunger=25, visited_count=11, dead=False),
                    elapsed_ticks=2,
                ),
            ),
            # Step 2: hungry -> eats food
            dict(
                observation=_make_obs(x=4, y=0, health=90, thirst=60, hunger=25, visited=11),
                next_observation=_make_obs(x=4, y=0, health=90, thirst=60, hunger=55, visited=11),
                action=dict(verb='eat'),
                transition=dict(
                    before=dict(tick=4, health=90, thirst=60, hunger=25, visited_count=11, dead=False),
                    after=dict(tick=6, health=90, thirst=60, hunger=55, visited_count=11, dead=False),
                    elapsed_ticks=2,
                ),
            ),
        ]
        m = ev.evaluate_episode(rows, horizon_tick=96)
        self.assertEqual(m['new_cells'], 1)
        self.assertEqual(m['fear_of_thirst_rate'], 1.0)
        self.assertEqual(m['thirst_drink_latency'], 2.0)
        self.assertEqual(m['fear_of_hunger_rate'], 0.5)  # 1 of 2 low-hunger steps ate
        self.assertEqual(m['hunger_eat_latency'], 4.0)
        self.assertEqual(m['fear_of_death_delta'], -1.0)  # explored 0% in danger vs 100% when safe
        self.assertEqual(m['fear_of_death_caution'], 1.0)

    def test_stage0_and_stage3_dry_run_end_to_end(self):
        with tempfile.TemporaryDirectory() as tmp:
            s0_dir = Path(tmp) / 'stage0'
            s0_args = stage0.build_parser().parse_args([
                '--mode', 'dual', '--out', str(s0_dir), '--dry-run',
                '--tune-seeds', '8', '--confirm-seeds', '16', '--eval-worlds', '16', '--n-boot', '400',
            ])
            s0_rep = stage0.run_stage0(s0_args)
            self.assertTrue(s0_rep['confirmed'])
            self.assertIsNotNone(s0_rep['rho_min'])
            self.assertGreaterEqual(s0_rep['recommended_stage3_seeds'], 32)

            # Stage 3 with genuine organoid signal -> dual_weak_positive or weak_positive
            s3_pos = Path(tmp) / 'stage3_pos'
            s3_args = stage3.build_parser().parse_args([
                '--mode', 'dual', '--stage0-report', str(s0_dir / 'stage0_report.json'),
                '--out', str(s3_pos), '--dry-run', '--dry-run-strength', '1.0',
                '--rho-lib', '0.50', '--seeds', '20', '--eval-worlds', '20', '--n-boot', '400',
            ])
            rep_pos = stage3.run_stage3(s3_args)
            self.assertIn(rep_pos['outcome']['verdict'], ('weak_positive', 'dual_weak_positive'))

            # Stage 3 with zero organoid signal -> negative (D-A passes, B-C crosses 0)
            s3_neg = Path(tmp) / 'stage3_neg'
            s3_args_neg = stage3.build_parser().parse_args([
                '--mode', 'dual', '--stage0-report', str(s0_dir / 'stage0_report.json'),
                '--out', str(s3_neg), '--dry-run', '--dry-run-strength', '0.0',
                '--rho-lib', '0.50', '--seeds', '20', '--eval-worlds', '20', '--n-boot', '400',
            ])
            rep_neg = stage3.run_stage3(s3_args_neg)
            self.assertEqual(rep_neg['outcome']['verdict'], 'negative')


if __name__ == '__main__':
    unittest.main()
