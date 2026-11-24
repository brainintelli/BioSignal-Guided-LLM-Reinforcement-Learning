"""Pure planning/admission fixtures, never biological training data."""
import copy
import json
import tempfile
import unittest
from pathlib import Path
from prepare_v2_data import baseline_partitions,stimulated_partitions,normalize_events,validate_envelope,load_admission,v2_digest,event_pair
from sr_data import validate_feature_scope,write


def rows(sid,start,n):
    return [dict(source_id=sid,start_frame=start+i*20000,end_frame=start+(i+1)*20000) for i in range(n)]


class PreparationTests(unittest.TestCase):
    def test_baseline_fallback_exact_marker_relative_partitions(self):
        p=baseline_partitions('baseline',12345)
        self.assertEqual([(x['split'],(x['start_frame']-12345)//20000,(x['end_frame']-12345)//20000) for x in p],
            [('calibration',120,180),('train',610,739),('evaluation',760,857),('holdout',880,940)])
    def test_stimulated_sources_are_disjoint_and_last_source_retained(self):
        plan=stimulated_partitions('baseline',0,[('pilot',rows('pilot',40000000,700)),('pair_01',rows('pair_01',60000000,140)),('pair_02',rows('pair_02',80000000,140))])
        self.assertEqual([(p['split'],p['source_id']) for p in plan],[('calibration','baseline'),('train','pilot'),('evaluation','pair_01'),('holdout','pair_02')])
        self.assertEqual(plan[1]['end_frame']-plan[1]['start_frame'],129*20000)
        self.assertEqual(plan[2]['end_frame']-plan[2]['start_frame'],97*20000)
    def test_insufficient_stimulated_sources_never_auto_fallback(self):
        with self.assertRaisesRegex(ValueError,'independent'):
            stimulated_partitions('baseline',0,[('pilot',rows('pilot',40000000,700)),('pair_01',rows('pair_01',60000000,140))])
    def test_later_sources_cannot_replace_fixed_pair02_holdout(self):
        fixed=[('pilot',rows('pilot',40000000,700)),('pair_01',rows('pair_01',60000000,140)),('pair_02',rows('pair_02',80000000,140))]
        self.assertEqual(stimulated_partitions('baseline',0,fixed),stimulated_partitions('baseline',0,fixed+[('pair_10',rows('pair_10',90000000,600))]))
        with self.assertRaises(ValueError):stimulated_partitions('baseline',0,fixed[:2]+[('pair_10',rows('pair_10',90000000,600))])
    def test_masked_gaps_are_not_filled_or_budget_shrunk(self):
        train=rows('pilot',40000000,132);del train[2:5]
        plan=stimulated_partitions('baseline',0,[('pilot',train),('pair_01',rows('pair_01',60000000,97)),('pair_02',rows('pair_02',80000000,1))])
        self.assertEqual(plan[1]['end_frame']-plan[1]['start_frame'],132*20000)
    def test_true_and_sham_frames_mask_exact_registered_eight_sample_sequence(self):
        events=[dict(id=8201,frame=1200001,arm='sham',pair=0,phase='pilot'),dict(id=8202,frame=2400001,arm='true',pair=0,phase='pilot')]
        out=normalize_events('pilot',events)
        self.assertEqual([x['end_frame']-x['start_frame'] for x in out],[8,8])
        self.assertEqual([x['arm'] for x in out],['sham','true'])
        self.assertEqual(normalize_events('baseline',[dict(id=8001,phase='disconnected'),dict(id=8002,phase='connected')]),[])
    def test_qc_scope_must_exactly_match_manifest(self):
        validate_feature_scope({'feature_channels':[1,4]},{'approved_feature_channels':[1,4]})
        with self.assertRaisesRegex(ValueError,'QC scope'):validate_feature_scope({'feature_channels':[1,4,9]},{'approved_feature_channels':[1,4]})
    def test_failed_or_tampered_receipt_never_admitted(self):
        with self.assertRaises(ValueError):validate_envelope({'result':{'status':'failed','errors':[]}},'baseline',{},None)

    def test_pair_conversion_rejects_bool_null_float_sign_space_and_unicode(self):
        self.assertEqual(event_pair('0'),0);self.assertEqual(event_pair(1),1)
        for value in (True,None,0.,-1,' 0','+0','-0','０','0.0',''):
            with self.assertRaises(ValueError):event_pair(value)

    def test_load_admission_preserves_message_and_normalizes_actual_string_pair(self):
        import h5py
        import numpy as np
        with tempfile.TemporaryDirectory(prefix='unit_v2_receipt_only_',dir=Path(__file__).parent) as directory:
            run=Path(directory).resolve();(run/'qc').mkdir();raw=run/'ARTIFICIAL_FORMAT_ONLY.raw.h5'
            messages=[dict(arm=arm,pair='0',phase='pilot',session='UNIT_TEST_ONLY') for arm in ('sham','true')]
            with h5py.File(raw,'x') as f:
                f.create_dataset('wellplate/id',data=[b'28073']);d=f.require_group('data_store/data0000')
                d.create_dataset('settings/sampling',data=[20000.]);d.create_dataset('groups/routed/frame_nos',data=np.arange(1000,1100,dtype='uint64'))
                d.create_dataset('groups/routed/raw',shape=(1024,100),dtype='uint16',compression='gzip')
                d.create_dataset('events',data=np.array([(1010+i*30,8201+i,1,json.dumps(m).encode()) for i,m in enumerate(messages)],
                    dtype=[('frameno','i8'),('eventid','i4'),('eventtype','i4'),('eventmessage','S200')]))
            stat=raw.stat();source=dict(path=str(raw),size_bytes=stat.st_size,mtime_ns=stat.st_mtime_ns)
            config=dict(session_id='UNIT_TEST_ONLY',chip_id='28073',device_model='MaxOne',r2_enabled=False,training_enabled=False)
            registration=dict(config=config,binding_sha256='fixture-binding-only')
            events=[dict(id=8201+i,eventtype=1,frame=1010+i*30,time_s=(10+i*30)/20000,arm=m['arm'],pair=0,phase='pilot',session='UNIT_TEST_ONLY',message=m) for i,m in enumerate(messages)]
            common=dict(source=source,baseline_source=None,stage='pilot',session_id='UNIT_TEST_ONLY',config_sha256=v2_digest(config),binding_sha256='fixture-binding-only')
            request=common|dict(config=config);request['request_sha256']=v2_digest(request)
            req=run/'qc'/'pilot.fixture.request.json';write(req,request)
            result=dict(stage='pilot',status='pass',errors=[],source=source,session_id='UNIT_TEST_ONLY',binding_sha256='fixture-binding-only',r2_enabled=False,training_enabled=False,
                events=events,metadata=dict(chip_id='28073',fs=20000,frame_gaps=0,raw_rows=1024,raw_axis_layout='physical_channel_index',first_frame=1000,last_frame=1099,frames=100))
            receipt=common|dict(result=result,result_sha256=v2_digest(result),request_sha256=request['request_sha256'],request_path=str(req))
            write(run/'qc'/'pilot.json',receipt)
            state=dict(session_id='UNIT_TEST_ONLY',records=[dict(path=str(raw),closed_utc='unit-test-only')])
            admitted=load_admission(run,'pilot',registration,state)
            self.assertEqual(admitted['events'][0]['pair'],0)
            self.assertEqual(admitted['result']['events'][0]['message']['pair'],'0')

if __name__=='__main__':unittest.main()
