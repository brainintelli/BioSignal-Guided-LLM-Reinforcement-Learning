from pathlib import Path
import tempfile
import unittest

import numpy as np

from oddball2 import analysis, detect
from support import CONFIG, DIP, READOUT, H5Recording, write_h5

FS = 20000.
CENTRE = detect.LEAD


def gain_at(sos, hz):
    z = np.exp(2j * np.pi * hz / FS)
    return float(np.prod([abs((b0 + b1 / z + b2 / z ** 2) / (1 + a1 / z + a2 / z ** 2)) for b0, b1, b2, a1, a2 in sos]))


class FilterTests(unittest.TestCase):
    def setUp(self):
        self.sos = detect.band_pass(FS)

    def test_band_pass_has_the_registered_edges(self):
        self.assertEqual(self.sos.shape, (4, 5))
        self.assertAlmostEqual(gain_at(self.sos, np.sqrt(300 * 3000)), 1., places=2)
        self.assertAlmostEqual(gain_at(self.sos, 300), np.sqrt(.5), places=3)
        self.assertAlmostEqual(gain_at(self.sos, 3000), np.sqrt(.5), places=3)
        self.assertLess(gain_at(self.sos, 50), .001)
        self.assertLess(gain_at(self.sos, 9000), .001)
        self.assertGreater(gain_at(self.sos, 1000), .99)
        for b0, b1, b2, a1, a2 in self.sos:                 # every pole inside the unit circle
            self.assertLess(max(abs(np.roots([1, a1, a2]))), 1.)

    def test_fast_form_equals_the_difference_equations(self):
        rng = np.random.default_rng(1)
        x = rng.normal(0, 5, (3, 2220))
        self.assertLess(np.max(np.abs(detect.convolved(self.sos, x) - detect.recursive(self.sos, x))), 1e-9)

    def test_filter_is_causal(self):
        rng = np.random.default_rng(2)
        x = rng.normal(0, 5, (2, 1500))
        changed = x.copy()
        changed[:, 800:] += 1000.
        a, b = detect.causal_filter(self.sos, x), detect.causal_filter(self.sos, changed)
        self.assertLess(np.max(np.abs(a[:, :800] - b[:, :800])), 1e-6)
        self.assertGreater(np.max(np.abs(a[:, 800:] - b[:, 800:])), 100.)

    def test_matches_scipy_where_scipy_is_installed(self):
        try:
            from scipy import signal
        except ImportError:
            self.skipTest('scipy not installed here; the acquisition host has it')
        reference = signal.butter(4, [300, 3000], 'bandpass', fs=FS, output='sos')
        x = np.random.default_rng(3).normal(0, 5, (2, 3000))
        wanted = signal.sosfilt(reference, x, axis=-1)
        for made in (detect.causal_filter(self.sos, x), detect.convolved(self.sos, x), detect.recursive(self.sos, x)):
            self.assertLess(np.max(np.abs(made - wanted)), 1e-6)


def trial(channels=3, seed=0, length=CENTRE + 1020):
    return np.rint(512 + np.random.default_rng(seed).normal(0, 2, (channels, length))).astype('u2')


def dip(raw, channel, offset):
    raw[channel, CENTRE + offset:CENTRE + offset + 4] -= DIP


class SpikeTests(unittest.TestCase):
    def setUp(self):
        self.sos = detect.band_pass(FS)
        self.threshold = np.full(3, 5.2)

    def found(self, raw):
        spikes = detect.trial_spikes(raw, self.threshold, self.sos, CENTRE)
        return [(int(c), int(f) - CENTRE) for c, f in zip(*np.nonzero(spikes))]

    def test_planted_spikes_are_found_once_and_noise_alone_gives_none(self):
        raw = trial()
        self.assertEqual(self.found(raw), [])
        dip(raw, 0, 300)
        dip(raw, 2, 700)
        got = self.found(raw)
        self.assertEqual([c for c, _ in got], [0, 2])
        self.assertTrue(300 <= got[0][1] <= 306 and 700 <= got[1][1] <= 706)

    def test_two_millisecond_dead_time_per_channel(self):
        raw = trial()
        dip(raw, 1, 300)
        dip(raw, 1, 310)        # 0.5 ms later: the same spike
        dip(raw, 1, 400)
        dip(raw, 2, 310)        # another channel has its own dead time
        self.assertEqual([c for c, _ in self.found(raw)], [1, 1, 2])

    def test_the_filter_rebound_of_a_large_spike_is_not_a_second_spike(self):
        raw = trial(seed=8)
        raw[0, CENTRE + 300:CENTRE + 304] -= 4 * DIP
        self.assertEqual(len(self.found(raw)), 1)
        ringing = detect.causal_filter(self.sos, raw.astype(float) - 512)[0, CENTRE + 320:CENTRE + 340]
        self.assertLess(ringing.min(), -self.threshold[0])      # the rebound alone would cross the threshold

    def test_nothing_is_counted_from_1_ms_before_to_5_ms_after_the_marker(self):
        raw = trial()
        for offset in (-18, 20, 60, 90):
            dip(raw, 0, offset)
        self.assertEqual(self.found(raw), [])
        dip(raw, 0, 110)
        self.assertEqual(len(self.found(raw)), 1)

    def test_a_pulse_artifact_inside_the_blanked_samples_changes_nothing_anywhere(self):
        clean = trial(seed=4)
        dip(clean, 1, 150)
        dip(clean, 2, 800)
        hit = clean.copy()
        hit[:, CENTRE:CENTRE + 4] += 300        # both phases of the pulse coupling into the readout
        hit[:, CENTRE + 4:CENTRE + 8] -= 300
        hit[:, CENTRE + 8:CENTRE + 45] += 40    # slow recovery
        self.assertEqual(self.found(hit), self.found(clean))
        self.assertEqual(len(self.found(clean)), 2)
        # The same artifact left in place would be counted: this is what the blanking removes.
        unblanked = detect.causal_filter(self.sos, hit.astype(float) - 512)
        self.assertLess(unblanked[:, CENTRE:CENTRE + 200].min(), -50)

    def test_a_saturated_stretch_is_bridged_and_nothing_is_counted_until_it_has_settled(self):
        raw = trial(seed=5)
        raw[1, CENTRE + 300:CENTRE + 400] = 1023
        dip(raw, 1, 440)        # inside the padded stretch
        dip(raw, 1, 490)        # still settling
        dip(raw, 1, 700)
        dip(raw, 0, 440)        # another channel is unaffected
        got = self.found(raw)
        self.assertEqual([c for c, _ in got], [0, 1])
        self.assertTrue(700 <= got[1][1] <= 706)

    def test_a_channel_saturated_for_the_whole_window_counts_nothing(self):
        raw = trial(seed=6)
        raw[2] = 0
        self.assertEqual(self.found(raw), [])


class RecordingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.sos = detect.band_pass(FS)
        self.roi = dict(sites=dict(H=READOUT['H'][:3], W=READOUT['W'][:2], F=[]))
        self.channels = analysis.union(self.roi)
        self.rng = np.random.default_rng(7)

    def noise(self, first, start, frames):
        return [(c, first + start, (512 + np.rint(self.rng.normal(0, 2, frames))).astype('u2')) for c in self.channels]

    def test_thresholds_are_five_robust_sigmas_of_the_quiet_recording(self):
        path = self.root / 'quiet.raw.h5'
        empty = (np.empty(0, dtype='u8'), np.empty(0, dtype='i4'))
        write_h5(path, 1000, 60 * 20000, [], empty, [], noise=self.noise(1000, 0, 60 * 20000))
        with H5Recording(path, CONFIG) as rec:
            made = detect.thresholds(rec, self.channels, self.sos)
            self.assertEqual(sorted(made), self.channels)
            for value in made.values():                     # white noise of 2 steps keeps about half its power
                self.assertTrue(4.5 < value < 6.5, value)
            with self.assertRaises(ValueError):             # a flat channel has no noise to measure
                detect.thresholds(rec, self.channels + [0], self.sos)
        short = self.root / 'short.raw.h5'
        write_h5(short, 1000, 20 * 20000, [], empty, [])
        with H5Recording(short, CONFIG) as rec:
            with self.assertRaises(ValueError):
                detect.thresholds(rec, self.channels, self.sos)

    def test_counts_per_site_come_from_raw_samples_in_the_response_window(self):
        path = self.root / 'trials.raw.h5'
        first, frames = 5000, 8 * 20000
        markers = [first + 40000, first + 80000, first + 120000]
        h, w = self.roi['sites']['H'], self.roi['sites']['W']
        spikes = [(h[0], 0, 200), (h[1], 0, 600), (h[2], 0, 990),      # trial 0: three on H's channels
                  (h[0], 0, 60), (h[0], 0, 1010),                      # outside 5-50 ms
                  (w[0], 1, 300), (w[1], 1, 301), (h[1], 1, 500),      # trial 1: two on W, one on H
                  (READOUT['F'][0], 2, 400)]                           # trial 2: only on a channel outside the readout
        patches = [(c, markers[t] + o, markers[t] + o + 4, 512 - DIP) for c, t, o in spikes]
        patches += [(c, markers[0], markers[0] + 4, 900) for c in self.channels]    # pulse artifact on trial 0 only
        empty = (np.empty(0, dtype='u8'), np.empty(0, dtype='i4'))
        write_h5(path, first, frames, [], empty, patches, noise=self.noise(first, 0, frames))
        qc = dict(trials=[dict(frame=m - first) for m in markers])
        threshold = {c: 5.2 for c in self.channels}
        with H5Recording(path, CONFIG) as rec:
            counts = detect.recount(rec, qc, self.roi, threshold, self.sos)
        self.assertEqual(counts, [dict(H=3, W=0, F=0), dict(H=1, W=2, F=0), dict(H=0, W=0, F=0)])


if __name__ == '__main__':
    unittest.main()
