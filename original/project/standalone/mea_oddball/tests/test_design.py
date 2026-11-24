import unittest

from oddball import design


class DesignTests(unittest.TestCase):
    def setUp(self):
        self.plan = design.plan(11)
        self.blocks = {b['name']: b for b in self.plan['blocks']}

    def test_plan_is_reproducible_and_tamper_evident(self):
        self.assertEqual(self.plan, design.plan(11))
        self.assertNotEqual(self.plan['sha256'], design.plan(12)['sha256'])
        changed = dict(self.plan, blocks=[dict(b) for b in self.plan['blocks']])
        changed['blocks'][6]['trials'] = changed['blocks'][6]['trials'][::-1]
        with self.assertRaises(ValueError):
            design.check_plan(changed)
        for seed in (-1, 1.5, True, '3'):
            with self.assertRaises(ValueError):
                design.plan(seed)

    def test_block_composition_matches_the_registered_design(self):
        main = [b for b in self.plan['blocks'] if b['stage'] == 'main']
        self.assertEqual(len(main), 12)
        for rnd in (1, 2, 3):
            kinds = sorted(b['frequent'] or 'uniform' for b in main if b['round'] == rnd)
            self.assertEqual(kinds, ['F', 'H', 'W', 'uniform'])
        for b in main:
            counts = [b['trials'].count(s) for s in design.SITES]
            self.assertEqual(len(b['trials']), 100)
            self.assertEqual(b['trials'].count(None), 10)
            self.assertEqual(sorted(counts), [30, 30, 30] if b['kind'] == 'uniform' else [10, 10, 70])
            if b['frequent']:
                self.assertEqual(b['trials'].count(b['frequent']), 70)
        screen = [b for b in self.plan['blocks'] if b['stage'] == 'screen']
        self.assertEqual([b['amplitude_mv'] for b in screen], [300, 300, 400, 400])
        for b in screen:
            self.assertEqual([b['trials'].count(s) for s in design.SITES], [10, 10, 10])
            self.assertEqual(b['trials'].count(None), 10)
        self.assertEqual([b['index'] for b in self.plan['blocks']], list(range(1, 18)))

    def test_worst_case_dose_fits_the_budget(self):
        real = sum(design.real_pulses(b) for b in self.plan['blocks'])
        self.assertEqual(real, 60 + 60 + 1080)
        self.assertLessEqual(real, design.MAX_REAL_PULSES)
        self.assertEqual(design.real_pulses(self.blocks['rehearsal']), 0)

    def test_rehearsal_repeats_the_first_main_block_without_output(self):
        rehearsal = self.blocks['rehearsal']
        first = next(b for b in self.plan['blocks'] if b['stage'] == 'main')
        self.assertEqual(rehearsal['trials'], first['trials'][:40])
        self.assertEqual(len(design.chunks(rehearsal)), 4)
        for trials in design.chunks(rehearsal):
            codes = design.chunk_codes(rehearsal, trials, 0)
            self.assertTrue(all(value[1] == 512 for kind, value in codes if kind == 'dac'))
            self.assertEqual([(k, v if k != 'dac' else v[0]) for k, v in codes],
                             [(k, v if k != 'dac' else v[0]) for k, v in design.chunk_codes(first, trials, 100)])
        with self.assertRaises(ValueError):
            design.chunk_codes(first, range(10), 0)
        with self.assertRaises(ValueError):
            design.chunk_codes(rehearsal, range(10), 100)

    def test_chunk_timing_and_pulse_shape(self):
        block = next(b for b in self.plan['blocks'] if b['stage'] == 'main')
        for trials in design.chunks(block):
            codes = design.chunk_codes(block, trials, 100)
            clock, markers, level = 0, [], {}
            for kind, value in codes:
                if kind == 'marker':
                    markers.append((clock, value))
                    site = block['trials'][value]
                    pulse = []
                elif kind == 'dac':
                    channel, code = value
                    self.assertEqual(design.SITES[channel], site)
                    pulse.append((clock - markers[-1][0], code))
                    level[channel] = code
                    if len(pulse) == 3:
                        # positive phase first (code below 512), 200 us per phase, back to neutral
                        self.assertEqual(pulse, [(0, 412), (4, 612), (8, 512)])
                else:
                    self.assertLessEqual(value, 65535)
                    clock += value
            self.assertEqual([t for t, _ in markers], [i * design.TRIAL_FRAMES for i in range(10)])
            self.assertEqual([i for _, i in markers], list(trials))
            self.assertTrue(all(code == 512 for code in level.values()))
            self.assertLessEqual(clock, 9 * design.TRIAL_FRAMES + 8)
            self.assertNotEqual(codes[-1][0], 'delay')

    def test_amplitude_rounds_down_and_only_registered_targets_exist(self):
        self.assertEqual(design.pulse_bits(2.981901, 300), 100)
        self.assertEqual(design.pulse_bits(2.981901, 400), 134)
        self.assertLessEqual(134 * 2.981901, 400)
        for bad in (200, 450, 600):
            with self.assertRaises(ValueError):
                design.pulse_bits(2.981901, bad)
        for lsb in (0, -1, float('nan'), '2.9', 1000.):
            with self.assertRaises(ValueError):
                design.pulse_bits(lsb, 300)

    def test_context_counts_and_bins(self):
        sites = ['H', 'H', None, 'W', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'W']
        self.assertEqual(design.recent_counts(sites), [0, 1, None, 0, 2, 3, 4, 5, 6, 6, 6, 1])
        self.assertEqual([design.bin_of(c) for c in range(9)],
                         ['low', 'low', 'mid', 'mid', 'mid', 'high', 'high', 'high', 'high'])
        with self.assertRaises(ValueError):
            design.bin_of(9)

    def test_every_block_fills_each_bin_it_is_meant_to(self):
        for b in (b for b in self.plan['blocks'] if b['stage'] == 'main' and b['frequent']):
            counts = design.recent_counts(b['trials'])
            frequent = [design.bin_of(c) for s, c in zip(b['trials'], counts) if s == b['frequent']]
            rare = [design.bin_of(c) for s, c in zip(b['trials'], counts) if s not in (None, b['frequent'])]
            self.assertGreater(frequent.count('high'), 35)
            self.assertGreater(rare.count('low'), 12)

    def test_event_ids_are_unique_across_the_session(self):
        ids = [design.event_id(b, i) for b in self.plan['blocks'] for i in range(len(b['trials']))]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertLess(max(ids), 2 ** 20)


if __name__ == '__main__':
    unittest.main()
