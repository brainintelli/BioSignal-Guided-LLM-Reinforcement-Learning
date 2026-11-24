from pathlib import Path
import tempfile
import unittest

import numpy as np

from oddball import analysis, design, library
from support import CONFIG, SESSION, SITE_CHANNELS, H5Recording, write_h5

FIRST = 3_000_000


def block_file(path, block, change=None, spikes=None, patches=None, gap=40000):
    """One block with exact timing; `change(events, patches)` plants a defect."""
    events, pulse_patches = [], []
    frame = FIRST + 200000
    for i, site in enumerate(block['trials']):
        if i:
            frame += gap if i % design.CHUNK == 0 else design.TRIAL_FRAMES
        events.append((frame, design.event_id(block, i), dict(design.event_properties(block, i), session=SESSION)))
        if site and block['stage'] != 'rehearsal':
            pulse_patches.append((SITE_CHANNELS[site], frame, frame + 800, 1023))
    if change:
        change(events, pulse_patches)
    frames = events[-1][0] - FIRST + 200000
    if spikes is None:  # 80 spikes/s spread over the readout channels
        rng = np.random.default_rng(8)
        n = frames // 250
        spikes = (FIRST + rng.integers(0, frames, n).astype('u8'), rng.integers(40, 80, n).astype('i4'))
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
        plan = design.plan(3)
        self.block = next(b for b in plan['blocks'] if b['name'] == 'screen300_1')
        self.rehearsal = plan['blocks'][0]
        self.roi = [c for c in range(40, 80)]

    def qc(self, block=None, **kwargs):
        block = block or self.block
        block_file(self.path, block, **kwargs)
        with H5Recording(self.path, CONFIG) as rec:
            return analysis.block_qc(rec, block, SESSION, SITE_CHANNELS, self.roi, 80.)

    def test_clean_block_passes_and_reports_every_trial(self):
        qc = self.qc()
        self.assertEqual(qc['errors'], [])
        self.assertTrue(qc['passed'])
        self.assertEqual(len(qc['trials']), 40)
        self.assertEqual(sum(t['real'] for t in qc['trials']), 30)
        self.assertTrue(all(t['valid'] for t in qc['trials']))
        self.assertEqual([t['site'] for t in qc['trials']], [s or 'Z' for s in self.block['trials']])
        self.assertEqual([t['count'] for t in qc['trials']], design.recent_counts(self.block['trials']))

    def test_response_window_is_10_to_300_ms_after_the_marker(self):
        anchor = FIRST + 200000
        offsets = np.array([-6000, -201, -200, 199, 200, 399, 400, 5999, 6000, 20000, 35999, 36000])
        qc = self.qc(spikes=((anchor + offsets).astype('u8'), np.full(len(offsets), 50, dtype='i4')))
        first = qc['trials'][0]
        self.assertEqual((first['before'], first['after'], first['after_20ms'], first['response']), (2, 4, 2, 2))
        self.assertEqual(first['artifact'], 1)                 # only the spike at 199 frames
        self.assertAlmostEqual(first['rate_hz'], 7 / 1.79)     # 200 .. 35999 frames

    def test_spikes_on_site_or_unlisted_channels_do_not_count(self):
        anchor = FIRST + 200000 + 1000
        channels = np.array([SITE_CHANNELS['H'], 5, 45], dtype='i4')
        qc = self.qc(spikes=(np.full(3, anchor, dtype='u8'), channels))
        self.assertEqual(qc['trials'][0]['after'], 1)

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

    def test_one_missing_onset_is_tolerated_and_marked_but_more_are_not(self):
        real = [i for i, s in enumerate(self.block['trials']) if s]
        def lose(count):
            def change(events, patches): del patches[:count]
            return change
        qc = self.qc(change=lose(1))
        self.assertTrue(qc['passed'])
        self.assertEqual(qc['missed_onsets'], real[:1])
        self.assertFalse(qc['trials'][real[0]]['valid'])
        qc = self.qc(change=lose(2))
        self.assertFalse(qc['passed'])
        self.assertEqual(qc['missed_onsets'], real[:2])

    def test_output_where_none_was_registered_fails(self):
        sham = self.block['trials'].index(None)
        real = next(i for i, s in enumerate(self.block['trials']) if s == 'H')
        def at(index, channel, value, start=0, stop=20):
            def change(events, patches):
                patches.append((channel, events[index][0] + start, events[index][0] + stop, value))
            return change
        cases = [(at(sham, SITE_CHANNELS['W'], 1023), False),        # rail on a sham trial
                 (at(sham, SITE_CHANNELS['F'], 512 + 200), False),   # 1259 uV excursion on a sham trial
                 (at(sham, SITE_CHANNELS['F'], 512 + 100), True),    # 629 uV stays below the limit
                 (at(real, SITE_CHANNELS['W'], 0), False),           # second site driven with the real one
                 (at(real, SITE_CHANNELS['W'], 512 + 100), True),    # small crosstalk is not an output
                 (at(real, SITE_CHANNELS['H'], 1023, 20000, 20010), False),  # still railed after one second
                 (at(real, SITE_CHANNELS['H'], 1023, -4, 0), False)]         # railed before its own marker
        for change, ok in cases:
            with self.subTest(case=cases.index((change, ok))):
                self.assertEqual(self.qc(change=change)['passed'], ok)

    def test_rehearsal_must_show_no_output_at_all(self):
        self.assertTrue(self.qc(block=self.rehearsal)['passed'])
        qc = self.qc(block=self.rehearsal)
        self.assertFalse(any(t['real'] for t in qc['trials']))
        def leak(events, patches): patches.append((SITE_CHANNELS['H'], events[4][0], events[4][0] + 800, 1023))
        self.assertFalse(self.qc(block=self.rehearsal, change=leak)['passed'])

    def test_background_rate_rule_needs_two_consecutive_segments(self):
        rows = lambda rates: [dict(rate_hz=r) for r in rates for _ in range(15)]
        self.assertFalse(analysis.activity(rows([100, 10, 100, 500, 100]), 100.)['stop'])
        self.assertTrue(analysis.activity(rows([100, 10, 20, 100]), 100.)['stop'])
        self.assertTrue(analysis.activity(rows([100, 450, 10]), 100.)['stop'])
        self.assertEqual(analysis.activity(rows([25, 400]), 100.)['outlier'], [False, False])
        with self.assertRaises(ValueError):
            analysis.activity(rows([1]), 0.)

    def test_saturated_readout_channels_are_found_and_dropped(self):
        def saturate(events, patches):
            for i, (frame, _, _) in enumerate(events):
                if self.block['trials'][i]:
                    patches.append((41, frame + 300, frame + 500, 0))
            patches.append((42, events[0][0] + 300, events[0][0] + 500, 1023))
            patches.append((43, events[0][0] + 1300, events[0][0] + 1500, 1023))  # after the window
        block_file(self.path, self.block, change=saturate)
        with H5Recording(self.path, CONFIG) as rec:
            frames, errors = analysis.check_events(rec, self.block, SESSION)
            rails = analysis.roi_rails(rec, frames, self.roi)
        self.assertEqual(errors, [])
        real = [s is not None for s in self.block['trials']]
        kept, dropped = analysis.prune_roi(self.roi, rails, real)
        self.assertEqual(sorted(set(self.roi) - set(kept)), [41])   # 42 was hit once: 1 of 30 at most
        self.assertEqual(dropped[41], 1.)
        self.assertFalse(rails[:, self.roi.index(43)].any())

    def test_readout_selection_excludes_sites_neighbours_bad_and_silent_channels(self):
        rng = np.random.default_rng(2)
        active = [c for c in range(80) if c != 79]
        frames = FIRST + rng.integers(0, 64 * 20000, 40000).astype('u8')
        write_h5(self.path, FIRST, 64 * 20000, [], (frames, rng.choice(active, 40000).astype('i4')), [])
        with H5Recording(self.path, CONFIG) as rec:
            chosen = analysis.select_roi(rec, SITE_CHANNELS, bad_channels=[77])
            self.assertGreater(chosen['reference_hz'], 0)
            with self.assertRaises(ValueError):
                analysis.select_roi(rec, SITE_CHANNELS, bad_channels=list(range(40, 80)))
        roi = set(chosen['roi'])
        self.assertFalse(roi & {10, 20, 30, 77, 79})
        self.assertFalse(roi & {0, 11, 21, 31, 40})      # 150 um from a site
        self.assertIn(12, roi)                            # 300 um away
        self.assertIn(11, chosen['excluded_near_sites'])
        self.assertGreaterEqual(len(roi), 50)


def rows_for(effect, habituation, seed=0, rounds=(1, 2, 3)):
    """Trial rows of complete rounds with a known evoked size and recent-use slope."""
    rng = np.random.default_rng(seed)
    plan = design.plan(21)
    rows = []
    for block in (b for b in plan['blocks'] if b['stage'] == 'main' and b['round'] in rounds):
        for i, (site, count) in enumerate(zip(block['trials'], design.recent_counts(block['trials']))):
            mean = 0. if site is None else effect * np.exp(-habituation * count)
            rows.append(dict(block=block['name'], round=block['round'], kind=block['kind'], frequent=block['frequent'],
                             trial=i, site=site or 'Z', real=site is not None, count=count, valid=True,
                             bin=None if count is None else design.bin_of(count),
                             response=float(rng.normal(mean, 8))))
    return rows


class GateTests(unittest.TestCase):
    def test_evoked_and_habituating_responses_pass_both_gates(self):
        gates = analysis.offline_gates(rows_for(30, .25), 1)
        self.assertEqual(gates['held_out_round'], 3)
        self.assertEqual(gates['library_rounds'], [1, 2])
        self.assertTrue(gates['evoked']['passed'])
        self.assertTrue(gates['habituation']['passed'])
        self.assertEqual(gates['usable_sites'], list(design.SITES))
        self.assertTrue(gates['training_allowed'])
        for site in design.SITES:
            self.assertLess(gates['habituation']['sites'][site]['uniform_spearman'], 0)

    def test_evoked_response_without_habituation_fails_the_second_gate(self):
        gates = analysis.offline_gates(rows_for(30, 0.), 1)
        self.assertTrue(gates['evoked']['passed'])
        self.assertFalse(gates['habituation']['passed'])
        self.assertFalse(gates['training_allowed'])

    def test_no_response_fails_and_false_positives_stay_rare(self):
        allowed = [analysis.offline_gates(rows_for(0, 0., seed), seed)['training_allowed'] for seed in range(20)]
        self.assertEqual(sum(allowed), 0)

    def test_sensitisation_is_not_accepted_as_habituation(self):
        gates = analysis.offline_gates(rows_for(10, -.2), 1)
        self.assertFalse(gates['habituation']['passed'])

    def test_fewer_than_two_complete_rounds_cannot_pass(self):
        rows = rows_for(30, .25, rounds=(1, 2))
        gates = analysis.offline_gates(rows, 1)
        self.assertEqual((gates['library_rounds'], gates['held_out_round']), ([1], 2))
        partial = [r for r in rows if r['round'] == 1 or r['block'] != rows[-1]['block']]
        gates = analysis.offline_gates(partial, 1)
        self.assertFalse(gates['training_allowed'])
        self.assertEqual(gates['complete_rounds'], [1])

    def test_invalid_trials_never_enter_a_gate(self):
        rows = rows_for(30, .25)
        for r in rows:
            if r['real'] and r['site'] == 'H':
                r['valid'] = False
        gates = analysis.offline_gates(rows, 1)
        self.assertEqual(gates['evoked']['sites']['H']['n_positive'], 0)
        self.assertNotIn('H', gates['usable_sites'])


class LibraryTests(unittest.TestCase):
    def setUp(self):
        self.rows, _, _ = analysis.split_rounds(rows_for(30, .25))
        self.library = library.build(self.rows)

    def test_library_orders_bins_and_centres_each_site(self):
        for site in design.SITES:
            means = self.library['stats'][site]['bin_means']
            self.assertGreater(means['low'], means['mid'])
            self.assertGreater(means['mid'], means['high'])
            self.assertAlmostEqual(sum(means.values()), 0., places=9)
        self.assertNotIn('Z', self.library['sites'])
        self.assertEqual(sum(len(v) for bins in self.library['sites'].values() for v in bins.values()),
                         sum(r['real'] for r in self.rows))

    def test_only_library_rounds_are_used(self):
        self.assertEqual({r['round'] for r in self.rows}, {1, 2})

    def test_changed_library_is_rejected(self):
        library.check(self.library)
        changed = dict(self.library, sites=dict(self.library['sites'], H=self.library['sites']['W']))
        with self.assertRaises(ValueError):
            library.check(changed)
        with self.assertRaises(ValueError):
            library.Sampler(changed, 1)

    def test_thin_bin_refuses_to_build(self):
        thin = [r for r in self.rows if not (r['site'] == 'F' and r['bin'] == 'mid')]
        with self.assertRaises(ValueError):
            library.build(thin)

    def test_sampler_rewards_unvisited_sites_more_than_repeated_ones(self):
        sampler = library.Sampler(self.library, 4)
        novel = np.mean([sampler.draw('H', 0) for _ in range(400)])
        repeated = np.mean([sampler.draw('H', 8) for _ in range(400)])
        self.assertGreater(novel, repeated + .5)
        first, second = library.Sampler(self.library, 9), library.Sampler(self.library, 9)
        self.assertEqual([first.draw('W', 3) for _ in range(5)], [second.draw('W', 3) for _ in range(5)])

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
