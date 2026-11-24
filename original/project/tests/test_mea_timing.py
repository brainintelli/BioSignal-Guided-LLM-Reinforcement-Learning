import copy
import unittest
from agent import mea_timing as m


def fixture():
    rows = [dict(type='session',seq=0,protocol='mea.raw_progress.v1',raw_open_status=0,filtered_open_status=0,mono_ns=990_000_000)]
    for i in range(80):
        end = (i+1)*1000; recv = 1_000_000_000 + (end-1)*50_000
        rows.append(dict(type='raw_block',seq=len(rows),start_frame=i*1000,end_frame=end,count=1000,
                         cumulative_frames=end,recv_begin_ns=recv-100,recv_end_ns=recv,
                         wall_before_ns=recv,wall_after_ns=recv+100,wall_ns=recv+1_700_000_000_000_000_000,
                         first_recv_ns=1_000_000_000,filtered_events=0))
    rows.append(dict(type='end',seq=len(rows),raw_frames=80000,raw_contiguous=True,clean_end=True,
                     raw_close_status=0,filtered_close_status=0,errors=0))
    return rows, range(2000,78000), 20000., []


class TimingTests(unittest.TestCase):
    def test_contiguous_progress_and_fit_do_not_claim_event_completeness(self):
        report = m.audit_progress(*fixture())
        self.assertTrue(report['raw_progress_passed'])
        self.assertTrue(report['receive_clock_fit_passed'])
        self.assertAlmostEqual(report['apparent_rate_hz'],20000.,places=5)
        self.assertFalse(report['live_spike_window_complete'])
        self.assertIsNone(report['absolute_hardware_clock_offset_ns'])

    def test_faults_clock_jump_h5_gap_sequence_and_footer_rejected(self):
        changes = [lambda a:a[0][10].update(start_frame=9001),
                   lambda a:a[0][10].update(seq=99),
                   lambda a:a[0][-1].update(clean_end=False),
                   lambda a:a[0][-1].update(raw_frames=79999),
                   lambda a:a[0].pop(),
                   lambda a:a.__setitem__(1,[2000,2002])]
        for change in changes:
            args=copy.deepcopy(list(fixture()));change(args)
            with self.subTest(change=change):self.assertFalse(m.audit_progress(*args)['raw_progress_passed'])
        args=copy.deepcopy(list(fixture()))
        for row in args[0][40:-1]:row['wall_ns']+=1_000_000_000
        self.assertFalse(m.audit_progress(*args)['wall_clock_stable'])

    def test_stale_heartbeat_and_receive_backlog_fail_freshness(self):
        args=copy.deepcopy(list(fixture()))
        args[0].insert(-1,dict(type='heartbeat',seq=81,since_raw_ns=500_000_000,mono_ns=5_500_000_000,raw_frames=80000))
        args[0][-1]['seq']=82
        self.assertFalse(m.audit_progress(*args)['raw_freshness_passed'])

    def test_startup_stall_and_forged_bracket_are_not_fresh(self):
        args=copy.deepcopy(list(fixture()));args[0][0]['mono_ns']=0
        args[0].insert(1,dict(type='heartbeat',seq=1,since_raw_ns=0,mono_ns=500_000_000,raw_frames=0))
        for i,row in enumerate(args[0]):row['seq']=i
        self.assertFalse(m.audit_progress(*args)['raw_freshness_passed'])
        args=copy.deepcopy(list(fixture()));args[0][20]['recv_begin_ns']=0
        self.assertFalse(m.audit_progress(*args)['raw_progress_passed'])

    def test_asymmetric_clock_exchange_gives_bounds_not_exact_offset(self):
        samples=[dict(c0=1_000_000_000,s1=1_012_000_000,s2=1_013_000_000,c3=1_009_000_000,
                      cm0=0,cm3=9_000_000,sm1=10_000_000_000,sm2=10_001_000_000,
                      client_elapsed_ns=9_000_000,server_elapsed_ns=1_000_000)]
        result=m.clock_bounds(samples)
        self.assertEqual(result['offset_bounds_ns'],[4_000_000,12_000_000])
        self.assertEqual(result['best_rtt_ns'],8_000_000)
        self.assertFalse(result['path_symmetry_proven'])
        samples[0]['c3']=999_000_000
        with self.assertRaises(ValueError):m.clock_bounds(samples)

    def test_clock_offset_reports_latest_sample_instead_of_stationary_intersection(self):
        a=dict(c0=1_000_000_000,s1=1_002_000_000,s2=1_003_000_000,c3=1_003_000_000,
               cm0=1_000_000_000,cm3=1_003_000_000,sm1=1_001_000_000,sm2=1_002_000_000,
               client_elapsed_ns=3_000_000,server_elapsed_ns=1_000_000)
        b=dict(c0=2_000_000_000,s1=2_015_000_000,s2=2_016_000_000,c3=2_021_000_000,
               cm0=2_000_000_000,cm3=2_021_000_000,sm1=2_010_000_000,sm2=2_011_000_000,
               client_elapsed_ns=21_000_000,server_elapsed_ns=1_000_000)
        self.assertEqual(m.clock_bounds([a,b])['offset_bounds_ns'],[-5_000_000,15_000_000])
        b['sm1']-=100_000_000;b['sm2']-=100_000_000
        with self.assertRaises(ValueError):m.clock_bounds([a,b])
