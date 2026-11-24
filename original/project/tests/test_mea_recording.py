import copy
import tempfile
import unittest
from pathlib import Path

from agent import mea_recording as m


def fixture():
    meta = dict(chip_id='22021', well_id=0, sample_rate_hz=10., lsb_volts=1e-6,
                mapping=[[0, 100], [1, 101]], raw_channels=[0, 1], raw_shape=[2, 50],
                current_mapping_before=[[0, 100], [1, 101]],
                current_mapping_after=[[0, 100], [1, 101]])
    return meta, list(range(100, 150)), [[120, 0, -3.]], [[120, 0, 0, -3.]], dict(
        clean_end=True, sdk_open_status=0, sdk_close_status=0, sdk_errors=0,
        event_count=1, coordinator_ok=True)


class RecordingTests(unittest.TestCase):
    def test_truncated_duration_cannot_pass_requested_acquisition(self):
        args = list(fixture()); args[0]['requested_seconds'] = 60
        self.assertFalse(m.audit_recording(*args)['offline_integrity_passed'])
        args[0]['requested_seconds'] = 5
        self.assertTrue(m.audit_recording(*args)['offline_integrity_passed'])

    def test_recording_agreement_and_no_online_claim(self):
        result = m.audit_recording(*fixture())
        self.assertTrue(result['offline_integrity_passed'])
        self.assertFalse(result['online_coverage_proven'])
        self.assertEqual(result['audit_interval'], [110, 140])
        self.assertEqual(result['matched_events'], 1)

    def test_quiet_requires_contiguous_raw_coverage(self):
        args = list(fixture()); args[2] = []; args[3] = []; args[4]['event_count'] = 0
        self.assertTrue(m.audit_recording(*args)['offline_integrity_passed'])
        args[1][20] += 1
        self.assertFalse(m.audit_recording(*args)['offline_integrity_passed'])

    def test_mismatch_duplicates_nan_bad_mapping_and_failure_reject(self):
        mutations = [
            lambda a: a[3].clear(),
            lambda a: a[3].append(a[3][0]),
            lambda a: a[2].append(a[2][0]),
            lambda a: a[3][0].__setitem__(3, float('nan')),
            lambda a: a[3][0].__setitem__(3, -4.),
            lambda a: a[0].__setitem__('raw_channels', [0]),
            lambda a: a[0].__setitem__('mapping', [[0, 100], [1, 100]]),
            lambda a: a[0].__setitem__('current_mapping_after', [[0, 101], [1, 100]]),
            lambda a: a[4].__setitem__('sdk_close_status', 1),
            lambda a: a[4].__setitem__('coordinator_ok', False),
        ]
        for mutate in mutations:
            args = copy.deepcopy(list(fixture())); mutate(args)
            with self.subTest(mutation=mutate):
                self.assertFalse(m.audit_recording(*args)['offline_integrity_passed'])

    def test_exclusions_are_counted_and_not_used_as_agreement(self):
        args = list(fixture())
        args[2].extend([[105, 0, -2.], [151, 0, -2.], [121, 9, -2.]])
        result = m.audit_recording(*args)
        self.assertTrue(result['offline_integrity_passed'])
        self.assertEqual(result['h5_exclusions'], dict(unmapped=1, outside_raw=1, guard=1))

    def test_export_replays_as_recording_and_refuses_failure_or_overwrite(self):
        from agent.mea_protocol import ResearchStream
        args = fixture(); report = m.audit_recording(*args)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'recording.jsonl'
            m.export_recording(path, args[0], args[2], report, 'test')
            stream = ResearchStream(); stream.feed(path.read_bytes()); stream.finish()
            self.assertEqual(stream.session.source, 'recording')
            self.assertEqual(stream.session.start_frame, 110)
            self.assertEqual(stream.watermark, 140)
            self.assertEqual(stream.spikes, ((120, 0, -3.),))
            with self.assertRaises(FileExistsError):
                m.export_recording(path, args[0], args[2], report, 'test')
            report['offline_integrity_passed'] = False
            with self.assertRaises(ValueError):
                m.export_recording(Path(tmp) / 'bad', args[0], args[2], report, 'test')

    def test_live_log_eof_counts_and_sequence_are_checked(self):
        import json
        rows = [dict(type='session', protocol='mea.filtered.p1.v1', seq=0, sdk_open_status=0),
                dict(type='events', seq=1, events=[[120, 0, 0, -3.]]),
                dict(type='end', seq=2, clean_end=True, sdk_close_status=0, sdk_errors=0, event_count=1)]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'events.jsonl'
            def put(): path.write_text(''.join(json.dumps(r)+'\n' for r in rows))
            put(); events, status = m.load_live(path)
            self.assertEqual(len(events), 1)
            self.assertTrue(status['clean_end'])
            rows[-1]['event_count'] = 2; put()
            with self.assertRaises(ValueError): m.load_live(path)
            rows[-1]['event_count'] = 1; rows[-1]['seq'] = 4; put()
            with self.assertRaises(ValueError): m.load_live(path)
            path.write_text('{}')
            with self.assertRaises(ValueError): m.load_live(path)
