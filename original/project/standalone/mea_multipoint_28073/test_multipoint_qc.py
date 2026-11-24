import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parent.parent/'mea_stimulation_v2'))
from mea_stim_v2.config import Config


class Recording:
    def __init__(self,config,expected):
        self.fs=20000.;self.duration=70.8204
        self.mapping=[(c,1000+c) for c in range(60)];self.channels=np.arange(60)
        self.source={'path':'synthetic.raw.h5','size_bytes':1,'mtime_ns':1}
        self.metadata=dict(chip_id='28073',fs=self.fs,gain=512.,hpf=300.,lsb_uv=6.2942504,
                           threshold_value=5.,threshold_h5_dataset='settings/spike_threshold',frame_gaps=0)
        self.events=[]
        for i,ex in enumerate(expected):
            t=10.+(i//3)*.2+(i%3)*.01
            self.events.append(dict(ex,time_s=t,frame=100+round(t*self.fs),eventtype=1,
                                    session=config.session_id,message=dict(ex)))
        times=np.arange(.5,self.duration,.5)
        self.spike_times=np.repeat(times,50);self.spike_channels=np.tile(np.arange(50),len(times))
        self.extra_rail=None;self.bad_recovery=False
    def __enter__(self):return self
    def __exit__(self,*args):pass
    def sample(self,channels,t,seconds=1):
        n=round(seconds*self.fs)
        if t<0 or t+seconds>self.duration+1e-9:raise ValueError('short')
        data=np.tile(np.resize([502.,502.,522.,522.],n),(len(channels),1))
        if 59 in channels and t==10. and seconds>1:
            for event in self.events:
                idx=round((event['time_s']-t+.001)*self.fs)
                if 0<=idx<n:data[channels.index(59),idx]=1023
            if self.extra_rail is not None:data[channels.index(59),round(self.extra_rail*self.fs)]=0
        if self.bad_recovery and abs(t-12.8204)<1e-7:data[:3]=512+2*(data[:3]-512)
        return data


class MultipointTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        route=Path(self.temp.name)/'route.cfg';route.write_text(''.join(f'{c}({1000+c})' for c in range(60)))
        self.config=Config(session_id='mp-test',chip_id='28073',organoid_id='temporary-28073',
                           started_utc='2026-11-17T06:13:00Z',device_hostname='192.0.2.21',
                           route_file=str(route),route_sha256=hashlib.sha256(route.read_bytes()).hexdigest(),
                           stim_electrode=1059,threshold_value=5.,threshold_h5_dataset='settings/spike_threshold')
        self.expected=[dict(id=10001+i,arm='true',pair=1,phase='multipoint',group=str(i//3+1),
                            pulse=str(i%3+1),electrode='1059') for i in range(15)]
        self.recording=Recording(self.config,self.expected)
    def analyze(self,**kw):
        import multipoint_qc
        with patch.object(multipoint_qc,'H5Recording',return_value=self.recording):
            return multipoint_qc.analyze('synthetic.raw.h5',self.config,list(range(50)),self.expected,100.,**kw)
    def test_complete_train_is_capability_only_with_fixed_roi(self):
        q=self.analyze()
        self.assertEqual(q['status'],'pass',q['errors'])
        self.assertFalse(q['v2_qualified']);self.assertFalse(q['causal_or_improvement_claim'])
        self.assertEqual(q['roi'],list(range(50)));self.assertEqual(len(q['pulses']),15)
        self.assertNotIn('p_value',q)
    def test_two_sample_timing_error_or_wrong_message_fails(self):
        self.recording.events[4]['frame']+=2
        self.assertEqual(self.analyze()['status'],'failed')
        self.recording.events[4]['frame']-=2
        self.recording.events[4]['message']['electrode']='1058'
        self.assertEqual(self.analyze()['status'],'failed')
    def test_backend_integer_schedule_matches_decimal_string_h5_properties(self):
        for e in self.expected:
            for key in ('group','pulse','electrode'):e[key]=int(e[key])
        self.assertEqual(self.analyze()['status'],'pass')
    def test_final_recovery_counts_from_pulse_end_and_20th_requires_600(self):
        self.recording.duration-=.00005
        self.assertEqual(self.analyze()['status'],'failed')
        self.recording.duration+=.00005
        for e in self.expected:e['pair']=20
        for e in self.recording.events:e['pair']=20;e['message']['pair']=20
        self.assertEqual(self.analyze()['status'],'failed')
    def test_last_saturation_is_relative_to_final_pulse_end(self):
        self.recording.extra_rail=.8204+.499
        self.assertEqual(self.analyze()['status'],'pass')
        self.recording.extra_rail=.8204+.501
        self.assertEqual(self.analyze()['status'],'failed')
    def test_joint_mad_failure_and_two_activity_outliers_fail_without_roi_deletion(self):
        self.recording.bad_recovery=True
        q=self.analyze();self.assertEqual(q['status'],'failed');self.assertEqual(len(q['roi']),50)
        self.recording.bad_recovery=False
        self.recording.spike_times=np.array([]);self.recording.spike_channels=np.array([],dtype=int)
        q=self.analyze();self.assertEqual(q['status'],'failed')
        self.assertIn('consecutive_activity_outliers',q['errors'])


if __name__=='__main__':unittest.main()
