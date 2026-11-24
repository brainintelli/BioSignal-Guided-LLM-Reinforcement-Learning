from pathlib import Path
import tempfile
import unittest

import numpy as np

from oddball2 import analysis, design, library
from support import CHANNELS, CONFIG, READOUT, ROI, SESSION, SITE_CHANNELS, H5Recording, write_h5

FIRST = 3_000_000
ANCHOR = FIRST + 200000      # marker of the first trial


def block_file(path, block, change=None, spikes=None, patches=None, gap=40000):
    """One block with exact timing; `change(events, patches)` plants a defect."""
    events, pulse_patches = [], []
    frame = ANCHOR
    for i, site in enumerate(block['trials']):
        if i:
            frame += gap if i % design.CHUNK == 0 else design.TRIAL_FRAMES
        events.append((frame, design.event_id(block, i), dict(design.event_properties(block, i), session=SESSION)))
        if site and block['stage'] != 'rehearsal':
            pulse_patches += [(c, frame, frame + 800, 1023) for c in SITE_CHANNELS[site]]
    if change:
        change(events, pulse_patches)
    frames = events[-1][0] - FIRST + 200000
    if spikes is None:  # 2 spikes/s on every channel
        rng = np.random.default_rng(8)
        n = frames * CHANNELS // 10000
        spikes = (FIRST + rng.integers(0, frames, n).astype('u8'), rng.integers(0, CHANNELS, n).astype('i4'))
    write_h5(path, FIRST, frames, events, spikes, pulse_patches + (patches or []))


class StatisticsTests(unittest.TestCase):
    def test_auc_counts_ties_as_half(self):
        self.assertEqual(analysis.auc([3, 4], [1, 2]), 1.)
        self.assertEqual(analysis.auc([1, 2], [3, 4]), 0.)
        self.assertEqual(analysis.auc([1, 1], [1, 1]), .5)
        self.assertAlmostEqual(analysis.auc([2, 3], [1, 2, 4]), (1.5 + 2) / 6)
        self.assertIsNone(analysis.auc([], [1]))

    def test_permutation_p_separates_signal_from_noise(self):
        rng = np.random.default_rng(0)
        a, b = rng.normal(2, 1, 20), rng.normal(0, 1, 20)
        self.assertLess(analysis.permutation_p(a, b, 1), .001)
        self.assertGreater(analysis.permutation_p(b, a, 1), .99)
        self.assertEqual(analysis.permutation_p(a, b, 7), analysis.permutation_p(a, b, 7))
        null = [analysis.permutation_p(rng.normal(size=20), rng.normal(size=20), i, 500) for i in range(200)]
        self.assertLess(abs(np.mean(np.asarray(null) <= .05) - .05), .04)

    def test_spearman(self):
        self.assertAlmostEqual(analysis.spearman([1, 2, 3, 4], [10, 9, 5, 1]), -1.)
        self.assertIsNone(analysis.spearman([1, 1, 1], [1, 2, 3]))


class ClosedFileTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / 'block.raw.h5'
        plan = design.plan(3, 600)
        self.block = next(b for b in plan['blocks'] if b['name'] == 'screen300')
        self.rehearsal = plan['blocks'][0]

    def qc(self, block=None, **kwargs):
        block = block or self.block
        block_file(self.path, block, **kwargs)
        with H5Recording(self.path, CONFIG) as rec:
            return analysis.block_qc(rec, block, SESSION, SITE_CHANNELS, ROI, 144.)

    def one(self, offsets, channel):
        offsets = np.asarray(offsets)
        return ((ANCHOR + offsets).astype('u8'), np.full(len(offsets), channel, dtype='i4'))

    def test_clean_block_passes_and_reports_every_trial(self):
        qc = self.qc()
        self.assertEqual(qc['errors'], [])
        self.assertTrue(qc['passed'])
        self.assertEqual(len(qc['trials']), 80)
        self.assertEqual(sum(t['real'] for t in qc['trials']), 60)
        self.assertTrue(all(t['valid'] for t in qc['trials']))
        self.assertEqual([t['site'] for t in qc['trials']], [s or 'Z' for s in self.block['trials']])
        self.assertEqual([t['count'] for t in qc['trials']], design.recent_counts(self.block['trials']))
        self.assertTrue(all(set(t['r']) == set(design.SITES) for t in qc['trials']))
        real = next(t for t in qc['trials'] if t['real'])
        self.assertEqual(real['onset']['onset_channels'][real['site']], 3)      # all three electrodes of the site

    def test_response_window_is_5_to_50_ms_after_the_marker(self):
        qc = self.qc(spikes=self.one([-2000, 0, 50, 99, 100, 500, 999, 1000, 20000, 35999, 36000], READOUT['H'][0]))
        first = qc['trials'][0]
        self.assertEqual(first['r'], dict(H=3, W=0, F=0))
        self.assertAlmostEqual(first['rate_hz'], 3 / 1.75)     # 1000 .. 35999 frames

    def test_each_site_counts_only_its_own_readout_channels(self):
        frames = np.full(4, ANCHOR + 400, dtype='u8')
        channels = np.array([READOUT['W'][3], READOUT['W'][4], READOUT['F'][0], SITE_CHANNELS['H'][0]], dtype='i4')
        qc = self.qc(spikes=(frames, channels))
        self.assertEqual(qc['trials'][0]['r'], dict(H=0, W=2, F=1))

    def test_event_defects_fail_the_block(self):
        def drop(events, patches): del events[5]
        def shift(events, patches): events[7] = (events[7][0] + 1,) + events[7][1:]
        def relabel(events, patches): events[3][2]['site'] = 'Q'
        def session(events, patches): events[0][2]['session'] = 'other'
        def number(events, patches): events[2] = (events[2][0], 99, events[2][2])
        for change in (drop, shift, relabel, session, number):
            with self.subTest(change=change.__name__):
                qc = self.qc(change=change)
                self.assertFalse(qc['passed'])
                self.assertEqual(qc['trials'], [])
        for gap, ok in ((36000, True), (60000, True), (35999, False), (60001, False)):
            with self.subTest(gap=gap):
                self.assertEqual(self.qc(gap=gap)['passed'], ok)

    def test_a_few_missing_onsets_are_tolerated_and_marked_but_more_are_not(self):
        real = [i for i, s in enumerate(self.block['trials']) if s]
        def lose(pulses):
            def change(events, patches): del patches[:3 * pulses]
            return change
        qc = self.qc(change=lose(3))                        # 3 of 60 is the registered 5%
        self.assertTrue(qc['passed'])
        self.assertEqual(qc['missed_onsets'], real[:3])
        self.assertFalse(qc['trials'][real[0]]['valid'])
        qc = self.qc(change=lose(4))
        self.assertFalse(qc['passed'])
        self.assertEqual(qc['missed_onsets'], real[:4])

    def test_a_site_counts_as_driven_when_any_of_its_electrodes_shows_the_pulse(self):
        real = next(i for i, s in enumerate(self.block['trials']) if s)
        def partial(events, patches): del patches[:2]
        qc = self.qc(change=partial)
        self.assertTrue(qc['passed'])
        self.assertTrue(qc['trials'][real]['valid'])
        self.assertEqual(qc['trials'][real]['onset']['onset_channels'][self.block['trials'][real]], 1)

    def test_output_where_none_was_registered_fails(self):
        sham = self.block['trials'].index(None)
        real = next(i for i, s in enumerate(self.block['trials']) if s == 'H')
        def at(index, channel, value, start=0, stop=20):
            def change(events, patches):
                patches.append((channel, events[index][0] + start, events[index][0] + stop, value))
            return change
        h, w, f = (SITE_CHANNELS[s] for s in design.SITES)
        cases = [(at(sham, w[0], 1023), False),              # rail on a sham trial
                 (at(sham, w[2], 1023), False),              # also on a neighbour electrode of the site
                 (at(sham, f[1], 512 + 200), False),         # 1259 uV excursion on a sham trial
                 (at(sham, f[1], 512 + 100), True),          # 629 uV stays below the limit
                 (at(real, w[1], 0), False),                 # second site driven with the real one
                 (at(real, w[1], 512 + 100), True),          # small crosstalk is not an output
                 (at(real, h[2], 1023, 20000, 20010), False),  # still railed after one second
                 (at(real, h[0], 1023, -4, 0), False)]         # railed before its own marker
        for number, (change, ok) in enumerate(cases):
            with self.subTest(case=number):
                self.assertEqual(self.qc(change=change)['passed'], ok)

    def test_rehearsal_must_show_no_output_at_all(self):
        qc = self.qc(block=self.rehearsal)
        self.assertTrue(qc['passed'])
        self.assertFalse(any(t['real'] for t in qc['trials']))
        def leak(events, patches): patches.append((SITE_CHANNELS['H'][1], events[4][0], events[4][0] + 800, 1023))
        self.assertFalse(self.qc(block=self.rehearsal, change=leak)['passed'])

    def test_background_rate_rule_needs_two_consecutive_segments(self):
        rows = lambda rates: [dict(rate_hz=r) for r in rates for _ in range(15)]
        self.assertFalse(analysis.activity(rows([100, 10, 100, 500, 100]), 100.)['stop'])
        self.assertTrue(analysis.activity(rows([100, 10, 20, 100]), 100.)['stop'])
        self.assertTrue(analysis.activity(rows([100, 450, 10]), 100.)['stop'])
        self.assertEqual(analysis.activity(rows([25, 400]), 100.)['outlier'], [False, False])
        with self.assertRaises(ValueError):
            analysis.activity(rows([1]), 0.)

    def test_saturated_readout_channels_are_dropped_from_the_site_that_saturates_them(self):
        a, b, c = READOUT['H'][:3]
        def saturate(events, patches):
            for i, (frame, _, _) in enumerate(events):
                if self.block['trials'][i] == 'H':
                    patches.append((a, frame + 300, frame + 500, 0))
            patches.append((b, events[0][0] + 300, events[0][0] + 500, 1023))
            patches.append((c, events[0][0] + 1300, events[0][0] + 1500, 1023))  # after the window
        block_file(self.path, self.block, change=saturate)
        channels = analysis.union(ROI)
        with H5Recording(self.path, CONFIG) as rec:
            frames, errors = analysis.check_events(rec, self.block, SESSION)
            rails = analysis.roi_rails(rec, frames, channels)
        self.assertEqual(errors, [])
        kept, dropped = analysis.prune_roi(ROI, channels, rails, self.block['trials'])
        self.assertEqual(sorted(set(ROI['sites']['H']) - set(kept['sites']['H'])), [a])   # b was hit at most once in 20
        self.assertEqual(dropped['H'], {str(a): 1.})
        self.assertEqual(kept['sites']['W'], ROI['sites']['W'])
        self.assertEqual(dropped['W'], {})
        self.assertFalse(rails[:, channels.index(c)].any())

    def quiet(self, silent=()):
        rng = np.random.default_rng(2)
        active = [c for c in range(CHANNELS) if c not in silent]
        frames = FIRST + rng.integers(0, 64 * 20000, 40000).astype('u8')
        write_h5(self.path, FIRST, 64 * 20000, [], (frames, rng.choice(active, 40000).astype('i4')), [])
        return H5Recording(self.path, CONFIG)

    def test_readout_selection_is_per_site_and_excludes_driven_bad_and_silent_channels(self):
        silent, bad = READOUT['W'][5], READOUT['F'][7]
        with self.quiet(silent=[silent]) as rec:
            chosen = analysis.select_roi(rec, SITE_CHANNELS, bad_channels=[bad])
            centres = analysis.select_roi(rec, {s: v[:1] for s, v in SITE_CHANNELS.items()}, bad_channels=[])
        self.assertEqual(chosen['sites']['H'], sorted(READOUT['H']))
        self.assertEqual(chosen['sites']['W'], sorted(set(READOUT['W']) - {silent}))
        self.assertEqual(chosen['sites']['F'], sorted(set(READOUT['F']) - {bad}))
        self.assertEqual(chosen['readable'], list(design.SITES))
        self.assertEqual(chosen['excluded_near_sites'], [])
        self.assertAlmostEqual(chosen['reference_hz'], sum(chosen['channel_hz'].values()))
        self.assertGreater(chosen['reference_hz'], 0)
        # Undriven electrodes right next to a centre are not readout channels.
        self.assertEqual(centres['excluded_near_sites'], sorted(c for v in SITE_CHANNELS.values() for c in v[1:]))

    def test_a_site_without_enough_readout_channels_is_unreadable_and_two_are_required(self):
        with self.quiet() as rec:
            chosen = analysis.select_roi(rec, SITE_CHANNELS, bad_channels=READOUT['H'][:5])
            self.assertEqual(chosen['readable'], ['W', 'F'])
            self.assertEqual(len(chosen['sites']['H']), 19)
            with self.assertRaises(ValueError):
                analysis.select_roi(rec, SITE_CHANNELS, bad_channels=READOUT['H'][:5] + READOUT['W'][:5])


def rows_for(effect, habituation, seed=0, rounds=(1, 2, 3), dead=(), fade=1.):
    """Trial rows of complete rounds with a known evoked size and recent-use slope.

    dead: sites that never respond; fade: evoked size in round 3 relative to the earlier rounds.
    """
    rng = np.random.default_rng(seed)
    plan = design.plan(21, 600)
    rows = []
    for block in (b for b in plan['blocks'] if b['stage'] == 'main' and b['round'] in rounds):
        for i, (site, count) in enumerate(zip(block['trials'], design.recent_counts(block['trials']))):
            r = {s: float(rng.normal(5, 8)) for s in design.SITES}
            if site is not None and site not in dead:
                r[site] += effect * np.exp(-habituation * count) * (fade if block['round'] == 3 else 1.)
            rows.append(dict(block=block['name'], round=block['round'], kind=block['kind'], frequent=block['frequent'],
                             trial=i, site=site or 'Z', real=site is not None, count=count, valid=True,
                             bin=None if count is None else design.bin_of(count), r=r))
    return rows


class GateTests(unittest.TestCase):
    def test_evoked_and_habituating_responses_pass_every_gate(self):
        gates = analysis.offline_gates(rows_for(30, .25), 1, rho_min=.2)
        self.assertEqual(gates['held_out_round'], 3)
        self.assertEqual(gates['library_rounds'], [1, 2])
        self.assertTrue(gates['evoked']['passed'])
        self.assertTrue(gates['habituation']['passed'])
        self.assertEqual(gates['usable_sites'], list(design.SITES))
        self.assertTrue(gates['library_allowed'])
        self.assertGreater(gates['reliability']['rho'], .2)
        self.assertTrue(gates['training_allowed'])
        for site in design.SITES:
            self.assertLess(gates['habituation']['sites'][site]['uniform_spearman'], 0)

    def test_training_needs_the_stage_0_threshold_and_the_library_must_reach_it(self):
        rows = rows_for(30, .25)
        waiting = analysis.offline_gates(rows, 1)
        self.assertTrue(waiting['library_allowed'])
        self.assertFalse(waiting['training_allowed'])
        self.assertIsNone(waiting['reliability']['passed'])
        self.assertIn('not supplied', waiting['reason'])
        short = analysis.offline_gates(rows, 1, rho_min=.99)
        self.assertTrue(short['library_allowed'])
        self.assertFalse(short['training_allowed'])
        self.assertIn('below', short['reason'])

    def test_reliability_grows_with_the_size_of_the_habituation_effect(self):
        rho = [analysis.offline_gates(rows_for(effect, .25), 1)['reliability']['rho'] for effect in (15, 30, 90)]
        self.assertEqual(rho, sorted(rho))
        self.assertGreater(rho[2], .7)
        self.assertLess(rho[2], 1.)

    def test_evoked_response_without_habituation_fails_the_second_gate(self):
        gates = analysis.offline_gates(rows_for(30, 0.), 1, rho_min=.01)
        self.assertTrue(gates['evoked']['passed'])
        self.assertFalse(gates['habituation']['passed'])
        self.assertFalse(gates['library_allowed'])
        self.assertFalse(gates['training_allowed'])

    def test_no_response_fails_and_false_positives_stay_rare(self):
        allowed = [analysis.offline_gates(rows_for(0, 0., seed), seed, rho_min=0.)['library_allowed'] for seed in range(20)]
        self.assertEqual(sum(allowed), 0)

    def test_sensitisation_is_not_accepted_as_habituation(self):
        gates = analysis.offline_gates(rows_for(10, -.2), 1)
        self.assertFalse(gates['habituation']['passed'])
        self.assertFalse(gates['library_allowed'])

    def test_two_responding_sites_are_enough_and_the_third_is_left_out(self):
        gates = analysis.offline_gates(rows_for(30, .25, dead=('W',)), 1, rho_min=.2)
        self.assertEqual(gates['usable_sites'], ['H', 'F'])
        self.assertTrue(gates['training_allowed'])
        gates = analysis.offline_gates(rows_for(30, .25, dead=('W', 'F')), 1, rho_min=.2)
        self.assertFalse(gates['library_allowed'])

    def test_a_site_that_was_not_readable_on_site_cannot_become_usable(self):
        gates = analysis.offline_gates(rows_for(30, .25), 1, sites=['H', 'W'])
        self.assertEqual(gates['usable_sites'], ['H', 'W'])
        self.assertFalse(gates['evoked']['sites']['F']['passed'])
        self.assertGreater(gates['evoked']['sites']['F']['auc'], .7)        # reported, not used

    def test_a_response_that_ran_down_by_the_held_out_round_is_refused(self):
        gates = analysis.offline_gates(rows_for(60, .25, fade=.35), 1, rho_min=.01)
        self.assertEqual(gates['stability']['passed_sites'], [])
        self.assertFalse(gates['library_allowed'])
        steady = analysis.offline_gates(rows_for(60, .25, fade=.8), 1, rho_min=.01)
        self.assertEqual(steady['stability']['passed_sites'], list(design.SITES))

    def test_fewer_than_two_complete_rounds_cannot_pass(self):
        rows = rows_for(30, .25, rounds=(1, 2))
        gates = analysis.offline_gates(rows, 1)
        self.assertEqual((gates['library_rounds'], gates['held_out_round']), ([1], 2))
        partial = [r for r in rows if r['round'] == 1 or r['block'] != rows[-1]['block']]
        gates = analysis.offline_gates(partial, 1, rho_min=0.)
        self.assertFalse(gates['library_allowed'])
        self.assertEqual(gates['complete_rounds'], [1])

    def test_invalid_trials_never_enter_a_gate(self):
        rows = rows_for(30, .25)
        for r in rows:
            if r['real'] and r['site'] == 'H':
                r['valid'] = False
        gates = analysis.offline_gates(rows, 1)
        self.assertEqual(gates['evoked']['sites']['H']['n_positive'], 0)
        self.assertNotIn('H', gates['usable_sites'])

    def test_screen_and_confirmation_thresholds_differ_as_registered(self):
        rng = np.random.default_rng(4)
        rows = [dict(real=s is not None, valid=True, site=s or 'Z',
                     r={k: float(rng.normal(8 if k == s else 0, 10)) for k in design.SITES})
                for s in ['H', 'W', 'F', None] * 30]
        screen = analysis.site_gate(rows, 3, **analysis.SCREEN)
        confirm = analysis.site_gate(rows, 3, **analysis.EVOKED)
        self.assertEqual((screen['auc_min'], confirm['auc_min'], confirm['p_max']), (.65, .70, .01))
        self.assertGreaterEqual(len(screen['passed_sites']), len(confirm['passed_sites']))
        self.assertTrue(set(confirm['passed_sites']) <= set(screen['passed_sites']))
        self.assertFalse(analysis.site_gate(rows, 3, sites=['H'], **analysis.SCREEN)['qualified'])


class LibraryTests(unittest.TestCase):
    def setUp(self):
        self.rows, _, _ = analysis.split_rounds(rows_for(30, .25))
        self.library = library.build(self.rows, ['H', 'F'])

    def test_library_orders_bins_and_centres_each_stored_site(self):
        self.assertEqual(sorted(self.library['sites']), ['F', 'H'])
        for site in ('H', 'F'):
            means = self.library['stats'][site]['bin_means']
            self.assertGreater(means['low'], means['mid'])
            self.assertGreater(means['mid'], means['high'])
            self.assertAlmostEqual(sum(means.values()), 0., places=9)
        self.assertEqual(sum(len(v) for bins in self.library['sites'].values() for v in bins.values()),
                         sum(r['real'] and r['site'] in 'HF' for r in self.rows))

    def test_only_library_rounds_are_used(self):
        self.assertEqual({r['round'] for r in self.rows}, {1, 2})

    def test_changed_library_is_rejected(self):
        library.check(self.library)
        changed = dict(self.library, sites=dict(self.library['sites'], H=self.library['sites']['F']))
        with self.assertRaises(ValueError):
            library.check(changed)
        with self.assertRaises(ValueError):
            library.Sampler(changed, 1)

    def test_thin_bin_or_single_site_refuses_to_build(self):
        thin = [r for r in self.rows if not (r['site'] == 'F' and r['bin'] == 'mid')]
        with self.assertRaises(ValueError):
            library.build(thin, ['H', 'F'])
        for sites in (['H'], [], ['H', 'Q']):
            with self.assertRaises(ValueError):
                library.build(self.rows, sites)

    def test_sampler_rewards_unvisited_sites_more_and_gives_nothing_for_an_unstored_site(self):
        sampler = library.Sampler(self.library, 4)
        novel = np.mean([sampler.draw('H', 0) for _ in range(400)])
        repeated = np.mean([sampler.draw('H', 8) for _ in range(400)])
        self.assertGreater(novel, repeated + .5)
        self.assertIsNone(sampler.draw('W', 0))
        with self.assertRaises(ValueError):
            sampler.draw('Z', 0)
        first, second = library.Sampler(self.library, 9), library.Sampler(self.library, 9)
        self.assertEqual([first.draw('F', 3) for _ in range(5)], [second.draw('F', 3) for _ in range(5)])

    def test_context_counts_the_last_eight_decisions(self):
        context = library.Context()
        self.assertEqual([context.push(s) for s in 'HHWHHHHHHHW'], [0, 1, 0, 2, 3, 4, 5, 6, 7, 7, 1])
        with self.assertRaises(ValueError):
            context.push('Z')

    def test_neighbouring_regions_use_different_sites(self):
        for x in range(-3, 4):
            for y in range(-3, 4):
                here = library.game_site(x, y)
                self.assertIn(here, design.SITES)
                self.assertNotEqual(here, library.game_site(x + 1, y))
                self.assertNotEqual(here, library.game_site(x, y + 1))


if __name__ == '__main__':
    unittest.main()
