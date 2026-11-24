"""The closed-file checks on the real 22021 recordings of 2026-11-19 (200 mV, three sites).

Skipped when the recordings or the MaxWell HDF5 filter are not on this machine.
"""
import json
import os
from pathlib import Path
import unittest

import numpy as np

from oddball import analysis
from support import CONFIG, H5Recording

RUN = Path(__file__).resolve().parents[3] / 'output' / 'mea_predictability_22021_20261119T042416Z' / 'cpp-crossover-v1'


@unittest.skipUnless((RUN / 'plan.json').is_file() and os.environ.get('HDF5_PLUGIN_PATH'),
                     'real recordings or HDF5 filter unavailable')
class RealRecordingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sites = json.loads((RUN / 'route.json').read_text())['channels']
        cls.trials = json.loads((RUN / 'plan.json').read_text())['trials']
        with H5Recording(RUN / 'warmup' / 'warmup.raw.h5', CONFIG) as rec:
            cls.chosen = analysis.select_roi(rec, cls.sites, [])

    def each(self):
        for row in self.trials:
            name = 'trial_%02d' % row['index']
            with H5Recording(RUN / name / (name + '.raw.h5'), CONFIG) as rec:
                yield row, rec, [e['frame'] - rec.first_frame for e in rec.events]

    def test_readout_channels_stay_clear_of_the_three_sites(self):
        roi = self.chosen['roi']
        self.assertGreaterEqual(len(roi), 50)
        self.assertFalse(set(roi) & set(self.sites.values()))
        self.assertTrue(self.chosen['excluded_near_sites'])

    def test_every_recorded_pulse_is_found_at_its_site_and_sham_shows_none(self):
        checked = 0
        for row, rec, frames in self.each():
            block = dict(stage='screen', trials=[s for s, _ in row['events']] or [None])
            for onset in analysis.site_onsets(rec, frames, block, self.sites):
                checked += 1
                self.assertTrue(onset['intended'], (row['index'], onset))
                self.assertFalse(onset['unexpected'] or onset['rail_before'] or onset['rail_late'], (row['index'], onset))
        self.assertEqual(checked, 42)

    def test_artifact_ends_before_the_response_window_opens(self):
        roi = self.chosen['roi']
        real, sham, rails = [], [], []
        for row, rec, frames in self.each():
            counts = analysis.psth(rec, frames[:1], roi)[0]
            (real if row['events'] else sham).append(counts)
            if row['events']:
                rails.append(analysis.roi_rails(rec, frames[:1], roi))
        zero = -analysis.PSTH[0] // analysis.PSTH[2]            # bin starting at the marker
        real, sham = np.mean(real, axis=0), np.mean(sham, axis=0)
        self.assertGreater(real[zero], 100)                     # the pulse is detected as spikes on most channels
        self.assertLess(sham[zero], 5)
        self.assertLess(real[zero + 1:zero + 10].max(), 5)      # 10-100 ms is back at the spontaneous level
        self.assertFalse(np.concatenate(rails).any())           # no readout channel saturated 10-60 ms after a pulse


if __name__ == '__main__':
    unittest.main()
