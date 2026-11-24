from pathlib import Path
import sys
import tempfile
import unittest
import h5py
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
try:
    from quality import inspect_h5, check_events, drift_rejected, parse_mapping
except ImportError: inspect_h5=None


def fixture(path, chip='NEW42', gap=False, rail=False, marker=False,n=40000):
    with h5py.File(path,'w') as f:
        f['wellplate/id']=np.array([chip.encode()])
        f['wellplate/version']=np.array([b'MaxOne Single Well MEA'])
        d=f.create_group('data_store/data0000')
        d['settings/sampling']=[20000.]
        d['settings/mapping']=np.array([(0,40),(1,41),(2,42),(3,43)],dtype=[('channel','i4'),('electrode','i4')])
        g=d.create_group('groups/routed')
        frames=np.arange(n,dtype='u8')+9000
        if gap: frames[23456:]+=1
        g['frame_nos']=frames;g['channels']=np.arange(1024)
        raw=g.create_dataset('raw',(1024,n),dtype='u2',fillvalue=512,chunks=(1,2000))
        if rail:raw[1,15000:15800]=0 # Missed by old minute+1-second sampling.
        spikes=[(9000+t,c,-20.) for c in range(4) for t in range(100,39000,1500)]
        d['spikes']=np.array(spikes,dtype=[('frameno','u8'),('channel','i4'),('amplitude','f4')])
        events=[(9100,1,7001,b'baseline')] if marker else []
        d['events']=np.array(events,dtype=[('frameno','u8'),('eventtype','i4'),('eventid','i4'),('eventmessage','S64')])


class QualityTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(inspect_h5,'closed-H5 audit is not implemented')
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.path=Path(self.tmp.name)/'fresh.h5'
    def inspect(self, **kw):
        return inspect_h5(self.path,'NEW42',[(0,40),(1,41),(2,42),(3,43)],2,**kw)
    def test_fresh_data_selects_current_candidate(self):
        fixture(self.path);r=self.inspect()
        self.assertEqual(r['status'],'passed',r)
        self.assertEqual(r['candidate'],{'channel':0,'electrode':40})
        self.assertEqual(r['raw_scan'],'all_samples_of_mapped_channels')
    def test_resumed_identity_keeps_frozen_candidate_or_rejects(self):
        fixture(self.path)
        r=self.inspect(selected={'channel':2,'electrode':42})
        self.assertEqual(r['candidate'],{'channel':2,'electrode':42})
        with h5py.File(self.path,'r+') as f:
            s=f['data_store/data0000/spikes'][:];s['channel'][s['channel']==2]=-1
            f['data_store/data0000/spikes'][:]=s
        self.assertIn('frozen_candidate',self.inspect(selected={'channel':2,'electrode':42})['errors'])
    def test_wrong_chip_and_mapping_never_qualify(self):
        fixture(self.path,chip='22021');self.assertIn('identity',self.inspect()['errors'])
        fixture(self.path)
        r=inspect_h5(self.path,'NEW42',[(0,40),(1,41),(2,99),(3,43)],2)
        self.assertIn('mapping',r['errors'])
    def test_full_raw_scan_finds_unsampled_saturation(self):
        fixture(self.path,rail=True);self.assertIn('saturation',self.inspect()['errors'])
    def test_short_rail_excursion_not_diluted_by_total_duration(self):
        fixture(self.path,n=80000)
        with h5py.File(self.path,'r+') as f:f['data_store/data0000/groups/routed/raw'][1,100:350]=0
        # 1.25% of one second, diluted to .3125% overall and .078% across channels.
        result=self.inspect();self.assertNotIn('saturation',result['errors'])
        self.assertIn('rail_excursion',result['errors'])
    def test_frame_gap_rejects(self):
        fixture(self.path,gap=True);self.assertIn('frame_gaps',self.inspect()['errors'])
    def test_noninteger_raw_and_missing_channel_labels_reject(self):
        fixture(self.path)
        with h5py.File(self.path,'r+') as f:
            g=f['data_store/data0000/groups/routed'];del g['raw']
            g.create_dataset('raw',(1024,40000),dtype='f4',fillvalue=np.nan)
        with self.assertRaises(ValueError):self.inspect()
        fixture(self.path)
        with h5py.File(self.path,'r+') as f:del f['data_store/data0000/groups/routed/channels']
        with self.assertRaises(ValueError):self.inspect()
    def test_even_one_out_of_range_adc_value_rejects(self):
        fixture(self.path)
        with h5py.File(self.path,'r+') as f:f['data_store/data0000/groups/routed/raw'][1,30000]=65535
        with self.assertRaises(ValueError):self.inspect()
    def test_no_active_candidate_rejects(self):
        fixture(self.path)
        with h5py.File(self.path,'r+') as f: f['data_store/data0000/spikes']['channel']=-1
        self.assertIn('candidate',self.inspect()['errors'])
    def test_event_set_requires_exact_ids_and_frame_offsets(self):
        events=[{'frame':10000+b*4000+p*200,'type':1,'id':2+b*3+p} for b in range(5) for p in range(3)]
        self.assertEqual(check_events(events,'post'),[])
        events[4]['frame']+=1
        self.assertIn('pulse_frame_intervals',check_events(events,'post'))
        events[4]['frame']-=1;events.append(events[0])
        self.assertIn('pulse_event_ids',check_events(events,'post'))
        self.assertIn('baseline_event',check_events([],'baseline'))
    def test_drift_uses_fresh_first_five_minutes(self):
        self.assertFalse(drift_rejected([10]*10))
        self.assertTrue(drift_rejected([10]*5+[10,16,16,10,10]))
        self.assertFalse(drift_rejected([10]*5+[16,10,16,10,10]))
        self.assertTrue(drift_rejected([0]*10))
    def test_route_duplicates_and_empty_reject(self):
        self.assertEqual(parse_mapping('0(40)1(41) arbitrary tail'),[(0,40),(1,41)])
        for s in ['', '0(40)0(41)', '0(40)1(40)', '1024(4)']:
            with self.assertRaises(ValueError):parse_mapping(s)


if __name__=='__main__':unittest.main()
