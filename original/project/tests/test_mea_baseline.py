import copy
import unittest
import tempfile
from pathlib import Path
import numpy as np
from agent import mea_baseline as m


def fixture():
    meta = dict(chip_id='22554', sample_rate_hz=10., lsb_volts=1e-6,
                mapping=[[i, i + 100] for i in range(4)], raw_channels=list(range(4)),
                raw_shape=[4, 3000], start_time_unix_ms=0, stop_time_unix_ms=300000)
    frames = np.arange(3000, dtype=np.int64)
    spikes = [[10 + w * 50 + j, j % 4, -10. - j] for w in range(59) for j in range(w % 7 + 1)]
    return meta, frames, spikes


class BaselineTests(unittest.TestCase):
    def test_explicit_subset_keeps_source_failure_and_exclusion_identity(self):
        from tools.mea_baseline_calibrate import routed_scope_analysis
        meta, frames, spikes = fixture()
        spikes = [[0, 99, -2.]] + spikes + [[4000, 0, -2.]]
        source = m.analyze_arrays(meta, frames, spikes)
        result, retained = routed_scope_analysis(meta, frames, spikes, source)
        self.assertFalse(result['source_qc']['structural_passed'])
        self.assertTrue(result['qc']['structural_passed'])
        self.assertEqual(result['scope_exclusions']['event_indices'], [0, len(spikes)-1])
        self.assertEqual(len(retained), len(spikes)-2)
        self.assertEqual(result['candidates']['status'], 'exploratory_subset_only')
        self.assertFalse(result['candidates']['profile']['r2_training_enabled'])
        frames[40:] += 1
        source = m.analyze_arrays(meta, frames, spikes)
        with self.assertRaises(ValueError):
            routed_scope_analysis(meta, frames, spikes, source)

    def test_clean_recording_and_temporal_split(self):
        result = m.analyze_arrays(*fixture())
        self.assertTrue(result['qc']['structural_passed'])
        self.assertEqual(result['qc']['raw_frames'], 3000)
        self.assertEqual(result['qc']['raw_duration_seconds'], 300.)
        self.assertEqual(len(result['windows']), 59)
        p = result['candidates']['profile']
        self.assertEqual(p['split']['fit'], [0, 29])
        self.assertEqual(p['split']['embargo'], [29, 31])
        self.assertEqual(p['split']['validation'], [31, 59])
        self.assertFalse(p['r2_training_enabled'])
        self.assertTrue(p['exploratory_only'])
        self.assertEqual(p['alpha'], 1)
        self.assertFalse(result['qc']['online_coverage_proven'])

    def test_frame_gap_fails_even_without_spikes(self):
        meta, frames, _ = fixture(); frames[40:] += 1
        result = m.analyze_arrays(meta, frames, [])
        self.assertFalse(result['qc']['structural_passed'])
        self.assertIn('raw_frame_discontinuity', result['qc']['errors'])
        self.assertIsNone(result['candidates']['profile'])

    def test_duplicate_frame_fails(self):
        meta, frames, spikes = fixture(); frames[40] = frames[39]
        self.assertFalse(m.analyze_arrays(meta, frames, spikes)['qc']['structural_passed'])

    def test_true_quiet_is_valid_but_not_a_calibrated_reward(self):
        meta, frames, _ = fixture()
        result = m.analyze_arrays(meta, frames, [])
        self.assertTrue(result['qc']['structural_passed'])
        self.assertTrue(all(w['spike_count'] == 0 for w in result['windows']))
        self.assertEqual(result['candidates']['status'], 'unidentifiable_constant_features')
        self.assertIsNone(result['candidates']['profile'])

    def test_bad_events_and_mapping_are_not_silently_filtered(self):
        for bad, expected in [([20, 99, -2.], 'unmapped_events'),
                              ([4000, 0, -2.], 'events_outside_recording'),
                              ([20, 0, float('nan')], 'nonfinite_amplitude')]:
            meta, frames, spikes = fixture(); spikes.append(bad)
            result = m.analyze_arrays(meta, frames, spikes)
            self.assertFalse(result['qc']['structural_passed'])
            self.assertIn(expected, result['qc']['errors'])
            self.assertIsNone(result['candidates']['profile'])
        meta, frames, spikes = fixture(); spikes.append(spikes[0])
        self.assertIn('duplicate_spike_frame_channel', m.analyze_arrays(meta, frames, spikes)['qc']['errors'])
        meta, frames, spikes = fixture(); meta['mapping'][1] = [0, 101]
        self.assertIn('mapping_not_one_to_one', m.analyze_arrays(meta, frames, spikes)['qc']['errors'])

    def test_chip_raw_coverage_and_duration(self):
        for field, value, expected in [('chip_id', '22021', 'chip_id_mismatch'),
                                      ('raw_channels', [0, 1], 'raw_channel_coverage'),
                                      ('stop_time_unix_ms', 0, 'h5_time_duration_mismatch')]:
            meta, frames, spikes = fixture(); meta[field] = value
            self.assertIn(expected, m.analyze_arrays(meta, frames, spikes)['qc']['errors'])

    def test_non_integral_frames_and_channels_rejected(self):
        meta, frames, spikes = fixture(); spikes[0][0] = 10.5
        self.assertIn('invalid_event_identity', m.analyze_arrays(meta, frames, spikes)['qc']['errors'])

    def test_cooccurrence_requires_distinct_channels_in_same_bin(self):
        meta, frames, _ = fixture(); meta['sample_rate_hz'] = 1000.
        meta['raw_shape'] = [4, 300000]; meta['stop_time_unix_ms'] = 300000
        frames = np.arange(300000)
        r = m.analyze_arrays(meta, frames, [[1000, 0, -1.], [1001, 0, -1.], [1020, 0, -1.], [1021, 1, -1.]])
        self.assertAlmostEqual(r['windows'][0]['coincident_event_fraction'], .5)

    def test_validation_cannot_change_fit_or_earlier_scores(self):
        result = m.analyze_arrays(*fixture())
        windows = copy.deepcopy(result['windows'])
        original = result['candidates']
        for w in windows[-4:]:
            w['rate_hz_per_channel'] *= 100
            w['active_channel_fraction'] = .99
        altered = m.calibrate_candidates(windows)
        self.assertEqual(original['profile'], altered['profile'])
        self.assertEqual(original['scores'][:-4], altered['scores'][:-4])

    def test_exact_10ms_boundaries_do_not_merge_adjacent_bins(self):
        meta, _, _ = fixture(); meta['sample_rate_hz'] = 20000.
        meta['raw_shape'] = [4, 60000]; meta['stop_time_unix_ms'] = 3000
        frames = np.arange(60000)
        # Guard starts at frame 20000; offsets 5799/5800 are bins 28/29.
        result = m.analyze_arrays(meta, frames, [[25799, 0, -1.], [25800, 1, -1.]], window_seconds=1)
        self.assertEqual(result['windows'][0]['coincident_event_fraction'], 0.)

    def test_scores_use_only_past_history_and_have_bounds(self):
        result = m.analyze_arrays(*fixture())['candidates']
        start = result['profile']['split']['validation'][0]
        for s in result['scores']:
            self.assertEqual(s['history_last_window'], s['window_index'] - 1)
            self.assertGreaterEqual(s['baseline_proximity'], 0)
            self.assertLessEqual(s['baseline_proximity'], 1)
            if s['novelty'] is not None:
                self.assertGreaterEqual(s['novelty'], 0)
                self.assertLessEqual(s['novelty'], 1)
        self.assertEqual(result['scores'][0]['window_index'], start)
        self.assertEqual(result['response_repeatability']['status'], 'not_estimable_without_repeated_stimuli')

    def test_short_recording_does_not_create_profile(self):
        result = m.analyze_arrays(*fixture())
        candidate = m.calibrate_candidates(result['windows'][:8])
        self.assertEqual(candidate['status'], 'insufficient_temporal_windows')
        self.assertIsNone(candidate['profile'])

    def test_validation_drift_in_excluded_zero_scale_dimension_is_reported(self):
        windows = m.analyze_arrays(*fixture())['windows']
        for w in windows:
            w['coincident_event_fraction'] = 0.
        for w in windows[31:]:
            w['coincident_event_fraction'] = 1.
        result = m.calibrate_candidates(windows)
        self.assertFalse(result['profile']['active_features'][2])
        self.assertEqual(result['validation']['inactive_feature_changed_fraction']['coincident_event_fraction'], 1.)

    def test_cli_reads_closed_h5_and_refuses_overwrite_or_wrong_hash(self):
        import h5py
        from tools.mea_baseline_calibrate import analyze_file
        meta, frames, spikes = fixture()
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); source = root/'fixture.h5'
            with h5py.File(source, 'w') as f:
                f['wellplate/id'] = np.array([b'22554'])
                f['mxw_version'] = np.array([b'test'])
                store = f.create_group('data_store/data0000')
                for k, value in [('well_id', 0), ('start_time', 0), ('stop_time', 300000)]:
                    store[k] = [value]
                s = store.create_group('settings')
                s['sampling'] = [10.]; s['lsb'] = [1e-6]; s['spike_threshold'] = [5.]
                s['mapping'] = np.array([(i, i+100) for i in range(4)], dtype=[('channel','i4'),('electrode','i4')])
                raw = store.create_group('groups/routed')
                raw['channels'] = np.arange(4); raw['frame_nos'] = frames
                raw['raw'] = np.ones((4, 3000), dtype='u2') * 500
                store['spikes'] = np.array([tuple(r) for r in spikes], dtype=[('frameno','i8'),('channel','i4'),('amplitude','f4')])
                store['events'] = np.empty(0)
            from agent.mea_recording import sha256
            checksum = sha256(source)
            result = analyze_file(source, root/'out', expected_sha256=checksum, plots=False)
            self.assertTrue(result['qc']['structural_passed'])
            self.assertEqual(result['raw_sample_qc']['sampled_constant_channel_count'], 4)
            self.assertTrue((root/'out'/'candidate_profile.json').exists())
            with self.assertRaises(FileExistsError):
                analyze_file(source, root/'out', expected_sha256=checksum, plots=False)
            with self.assertRaises(ValueError):
                analyze_file(source, root/'wrong', expected_sha256='0'*64, plots=False)
            self.assertFalse((root/'wrong').exists())
            with h5py.File(source, 'r+') as f:
                dataset = f['data_store/data0000/spikes']
                broken = np.array([(float(r['frameno']), float(r['channel']), r['amplitude']) for r in dataset],
                                  dtype=[('frameno','f8'),('channel','f8'),('amplitude','f4')])
                broken[0]['frameno'] += .5
                del f['data_store/data0000/spikes']
                f['data_store/data0000/spikes'] = broken
            with self.assertRaisesRegex(ValueError, 'integer'):
                analyze_file(source, root/'fractional', expected_sha256=sha256(source), plots=False)
            self.assertFalse((root/'fractional').exists())


if __name__ == '__main__':
    unittest.main()
