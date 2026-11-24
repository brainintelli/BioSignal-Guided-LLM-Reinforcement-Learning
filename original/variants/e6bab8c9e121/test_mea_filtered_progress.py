import copy
import json
import unittest
from agent.mea_timing import FilteredFrameStream, audit_filtered_progress


def packets():
    return [dict(type='session',seq=0,protocol='mea.filtered.frames.v1',session_id='test',sdk_open_status=0),
            dict(type='events',seq=1,frame_number=110,events=[[110,0,0,-3.]]),
            dict(type='frame_progress',seq=2,start_frame=100,end_frame=200,count=100,cumulative_frames=100,
                 event_count=1,complete=True,discontinuities=0,corrupted=0),
            dict(type='frame_progress',seq=3,start_frame=200,end_frame=300,count=100,cumulative_frames=200,
                 event_count=1,complete=True,discontinuities=0,corrupted=0),
            dict(type='end',seq=4,clean_end=True,sdk_close_status=0,sdk_errors=0,event_count=1,received_frames=200)]


def send(stream,row):stream.feed((json.dumps(row)+'\n').encode())


class FilteredProgressTests(unittest.TestCase):
    def test_audit_uses_actual_filtered_frames_and_same_stream_barrier(self):
        from tests.test_mea_timing import fixture
        rows,frames,rate,_=fixture();rows=copy.deepcopy(rows)
        rows[0].update(protocol='mea.filtered.frames.v1',session_id='test',sdk_open_status=0)
        for row in rows[1:-1]:
            row.update(type='frame_progress',event_count=0,complete=True,discontinuities=0,corrupted=0)
        rows[-1].update(sdk_close_status=0,sdk_errors=0,event_count=0,received_frames=80000)
        report=audit_filtered_progress(rows,frames,rate,True)
        self.assertTrue(report['observed_live_window_gate_passed'])
        self.assertTrue(report['filtered_frame_prefix_verified'])
        self.assertEqual(report['quiet_blocks'],80)
        self.assertEqual(report['covered_end_frame'],80000)
        contradicted=audit_filtered_progress(rows,frames,rate,False)
        self.assertFalse(contradicted['live_spike_window_complete'])
        rows[-1]['end_mono_ns']=rows[-2]['recv_end_ns']+2_000_000_000
        self.assertFalse(audit_filtered_progress(rows,frames,rate,True)['filtered_freshness_passed'])

    def test_non_object_packet_poison_hides_verified_watermark(self):
        for raw in (b'[]\n',b'null\n',b'"x"\n',b'42\n'):
            stream=FilteredFrameStream()
            for row in packets()[:3]:send(stream,row)
            with self.assertRaises(ValueError):stream.feed(raw)
            self.assertTrue(stream.failed);self.assertIsNone(stream.watermark)

    def test_progress_advances_through_quiet_window_only_after_event_barrier(self):
        stream=FilteredFrameStream();rows=packets()
        send(stream,rows[0]);send(stream,rows[1]);self.assertIsNone(stream.watermark)
        send(stream,rows[2]);self.assertEqual(stream.watermark,200)
        send(stream,rows[3]);self.assertEqual(stream.watermark,300)
        self.assertEqual(stream.quiet_blocks,1)
        send(stream,rows[4]);stream.finish()

    def test_bad_frame_event_counts_gaps_corruption_and_missing_footer_fail(self):
        mutations=[lambda r:r[1].update(frame_number=111),lambda r:r[2].update(start_frame=111),
                   lambda r:r[2].update(event_count=2),lambda r:r[3].update(start_frame=201),
                   lambda r:r[3].update(corrupted=1),lambda r:r[3].update(complete=False),
                   lambda r:r[-1].update(clean_end=False),lambda r:r[-1].update(received_frames=201)]
        for mutate in mutations:
            rows=copy.deepcopy(packets());mutate(rows);stream=FilteredFrameStream()
            with self.subTest(mutation=mutate),self.assertRaises(ValueError):
                for row in rows:send(stream,row)
                stream.finish()
            self.assertTrue(stream.failed)
            self.assertIsNone(stream.watermark)
        stream=FilteredFrameStream()
        for row in packets()[:-1]:send(stream,row)
        with self.assertRaises(ValueError):stream.finish()
