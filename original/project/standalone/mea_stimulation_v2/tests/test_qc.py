import dataclasses
from pathlib import Path
import tempfile
import unittest
import numpy as np

from test_recording import fixture


class SyntheticRecording:
    """Computed raw data avoids writing ~60 GB for the 25-minute baseline."""
    def __init__(self, config, stage='baseline', candidates=59):
        self.config = config
        self.mapping = [(i, 1000+i) for i in range(60)]
        self.channels = np.arange(60)
        self.fs = 20000.
        self.duration = 1500. if stage == 'baseline' else 720.
        self.metadata = dict(chip_id='28073', fs=20000., gain=512., hpf=300.,
                             lsb_uv=3., threshold_value=5., threshold_h5_dataset='settings/spike_threshold',
                             frame_gaps=0, first_frame=100, last_frame=int(self.duration*self.fs)+99)
        self.source = dict(path='synthetic', size_bytes=0, mtime_ns=0)
        if stage == 'baseline':
            self.events = [dict(id=8001, phase='disconnected', time_s=0., arm=None, pair=0, eventtype=1, session=config.session_id),
                           dict(id=8002, phase='connected', time_s=300., arm=None, pair=0, eventtype=1, session=config.session_id)]
        else:
            self.events = [dict(id=8201, phase='pilot', time_s=60., arm='sham', pair=0, eventtype=1, session=config.session_id),
                           dict(id=8202, phase='pilot', time_s=120., arm='true', pair=0, eventtype=1, session=config.session_id)]
        times = np.arange(1., self.duration, 10.)
        self.spike_times = np.repeat(times, candidates)
        self.spike_channels = np.tile(np.arange(candidates), len(times))
        self.bad_times = set()
        self.saturated_channels = []
        self.sham_artifact_at = None
        self.true_artifact_at = .001
        self.last_rail = None
        self.baseline_bad_channels = []

    def sample(self, channels, t, seconds=1):
        from mea_stim_v2.recording import RecordingError
        n = int(round(seconds*self.fs))
        if t < 0 or t+seconds > self.duration+1e-9: raise RecordingError('Insufficient window')
        channels = list(channels)
        data = np.tile(np.resize(np.array([502,502,522,522], dtype=float), n), (len(channels), 1))
        if round(t, 3) in self.bad_times: data[:4] = 512 + (data[:4]-512)*2
        if round(t, 3) in self.bad_times:
            for i,c in enumerate(channels):
                if c in self.baseline_bad_channels: data[i] = 512+(data[i]-512)*2
        for event in self.events:
            if abs(t-event['time_s']) < 1e-8 and abs(seconds-.6) < 1e-8:
                onset = self.true_artifact_at if event['arm']=='true' else self.sham_artifact_at
                if 59 in channels and onset is not None: data[channels.index(59), round(onset*self.fs)] = 1023
                if event['arm']=='true':
                    for c in self.saturated_channels:
                        if c in channels: data[channels.index(c),0] = 0
                    if self.last_rail is not None and 59 in channels: data[channels.index(59),round(self.last_rail*self.fs)] = 0
        return data


class QCTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        _, self.config = fixture(Path(self.temp.name))

    def test_baseline_requires_calibration_and_independent_18_of_19(self):
        from mea_stim_v2.qc import baseline
        rec = SyntheticRecording(self.config)
        rec.bad_times = {932.}
        result = baseline(rec, self.config)
        self.assertEqual(result['status'], 'pass', result)
        self.assertEqual(result['holdout19']['passed_count'], 18)
        self.assertEqual(len(result['candidate_roi']), 59)
        self.assertAlmostEqual(result['reference_rate'], 5.9)
        rec.bad_times.add(962.)
        self.assertEqual(baseline(rec,self.config)['status'], 'failed')

    def test_under_50_or_zero_reference_or_r2_fails(self):
        from mea_stim_v2.qc import baseline
        rec = SyntheticRecording(self.config, candidates=49)
        self.assertEqual(baseline(rec,self.config)['status'], 'failed')
        rec = SyntheticRecording(self.config)
        rec.spike_channels = rec.spike_channels[rec.spike_times < 900]
        rec.spike_times = rec.spike_times[rec.spike_times < 900]
        self.assertEqual(baseline(rec,self.config)['status'], 'failed')
        self.assertEqual(baseline(SyntheticRecording(self.config),dataclasses.replace(self.config,r2_enabled=True))['status'], 'failed')

    def test_mad_ratio_shift_and_95_percent_are_joint_per_channel(self):
        from mea_stim_v2.qc import stability
        base=np.tile([502,502,522,522], (100,250))
        post=base.copy();post[:3]+=6;post[3:6]=512+2*(post[3:6]-512)
        self.assertFalse(stability(base, post)['passed'])
        post=base.copy();post[:5]+=6
        self.assertTrue(stability(base,post)['passed'])
        post[5,0]=0
        self.assertFalse(stability(base,post)['passed'])

    def test_pilot_removes_only_true_near_saturation_and_sham_600ms_is_checked(self):
        from mea_stim_v2.qc import baseline, pilot
        base=SyntheticRecording(self.config); b=baseline(base,self.config)
        rec=SyntheticRecording(self.config,'pilot');rec.saturated_channels=[0]
        q=pilot(rec,base,self.config,b['candidate_roi'])
        self.assertEqual(q['status'],'pass',q)
        self.assertEqual(q['roi'],list(range(1,59)))
        self.assertEqual(q['holdout19']['passed_count'],19)
        rec.sham_artifact_at=.599
        self.assertEqual(pilot(rec,base,self.config,b['candidate_roi'])['status'],'failed')

    def test_formal_never_removes_channels_and_activity_two_bins_stops(self):
        from mea_stim_v2.qc import pair
        rec=SyntheticRecording(self.config,'pilot');rec.duration=150.
        rec.events[0].update(id=9001,pair=1,time_s=30.,phase='control')
        rec.events[1].update(id=9002,pair=1,time_s=90.,phase='control')
        spec={'pair':1,'interval_s':60,'events':[{'id':9001,'arm':'sham','pair':1},{'id':9002,'arm':'true','pair':1}]}
        rec.saturated_channels=[0]
        q=pair(rec,self.config,spec,list(range(59)),5.9)
        self.assertEqual(q['roi'],list(range(59)))
        self.assertEqual(q['status'],'pass',q)
        rec.spike_times=np.array([]);rec.spike_channels=np.array([],dtype=int)
        q=pair(rec,self.config,spec,list(range(59)),5.9)
        self.assertEqual(q['status'],'failed')
        self.assertIn('consecutive_activity_outliers',q['errors'])

    def test_short_window_interval_and_late_saturation_fail(self):
        from mea_stim_v2.qc import baseline,pilot
        base=SyntheticRecording(self.config); b=baseline(base,self.config)
        for kind in ('short','interval','saturation','onset'):
            rec=SyntheticRecording(self.config,'pilot')
            if kind=='short':rec.duration=719.
            if kind=='interval':rec.events[1]['time_s']=120.101
            if kind=='saturation':rec.last_rail=.501
            if kind=='onset':rec.true_artifact_at=.01005
            self.assertEqual(pilot(rec,base,self.config,b['candidate_roi'])['status'],'failed',kind)

    def test_last_pair_cannot_be_qualified_with_60_second_recovery(self):
        from mea_stim_v2.qc import pair
        rec=SyntheticRecording(self.config,'pilot');rec.duration=150.
        rec.events[0].update(id=9019,pair=10,time_s=30.,phase='control')
        rec.events[1].update(id=9020,pair=10,time_s=90.,phase='control')
        spec={'pair':10,'interval_s':60,'events':[{'id':9019,'arm':'sham','pair':10},{'id':9020,'arm':'true','pair':10}]}
        self.assertEqual(pair(rec,self.config,spec,list(range(59)),5.9)['status'],'failed')

    def test_baseline_rate_boundary_and_technical_sample_rejection(self):
        from mea_stim_v2.qc import baseline
        rec=SyntheticRecording(self.config)
        # Exactly 24 events / 480 seconds is eligible. 23 is not.
        times=np.arange(421.,899.,20.)
        rec.spike_times=np.repeat(times,59)
        rec.spike_channels=np.tile(np.arange(59),len(times))
        keep=np.ones(len(rec.spike_times),dtype=bool);keep[0]=False
        rec.spike_times=rec.spike_times[keep];rec.spike_channels=rec.spike_channels[keep]
        q=baseline(rec,self.config)
        self.assertNotIn(0,q['candidate_roi'])
        self.assertEqual(q['candidate_roi'],list(range(1,59)))

    def test_changed_measured_threshold_or_route_is_never_admitted(self):
        from mea_stim_v2.qc import baseline
        for kind in ('threshold','mapping'):
            rec=SyntheticRecording(self.config)
            if kind=='threshold':rec.metadata['threshold_value']=4.5
            if kind=='mapping':rec.mapping[0]=(0,100)
            self.assertEqual(baseline(rec,self.config)['status'],'failed')


if __name__=='__main__':unittest.main()
