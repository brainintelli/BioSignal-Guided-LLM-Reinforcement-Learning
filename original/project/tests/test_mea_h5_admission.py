import copy
import importlib
from pathlib import Path
import tempfile
import unittest
import json

from agent import mea_recording as recording
from tests.test_mea_recording import fixture


def subset_fixture():
    args=list(fixture());m=args[0]
    m.update(chip_id='28080',mapping=[[0,100],[1,101],[3,103]],raw_channels=[0,1,2],raw_shape=[3,50])
    args[2].extend([[121,2,-5.],[122,3,-7.]])
    args[3].extend([[121,2,0,-5.],[122,3,0,-7.]])
    args[4]['event_count']=3
    return args


class H5AdmissionTests(unittest.TestCase):
    def test_reconstruction_and_training_resolve_same_manifest_run(self):
        from agent import mea_h5_replay as m
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);(root/'capture_A').mkdir();(root/'capture_B').mkdir()
            manifest={'accepted_run':'capture_B'}
            self.assertEqual(m.resolve_accepted_run(root,manifest),root/'capture_B')
            for name in ['../capture_A','/absolute','.',None]:
                with self.assertRaises(ValueError):m.resolve_accepted_run(root,{'accepted_run':name})

    def test_explicit_subset_preserves_raw_metadata_and_excludes_uncovered_channels(self):
        args=subset_fixture();before=copy.deepcopy(args[0])
        self.assertFalse(recording.audit_recording(*args)['offline_integrity_passed'])
        result=recording.audit_recording(*args,mapping_policy='finalized_h5_recorded_subset')
        self.assertTrue(result['offline_integrity_passed']);self.assertEqual(args[0],before)
        self.assertEqual(result['analysis_mapping'],[[0,100],[1,101]])
        self.assertEqual(result['excluded_unrouted_raw_channels'],[2])
        self.assertEqual(result['excluded_unrecorded_mapped_channels'],[3])
        self.assertFalse(result['hardware_mapping_independently_verified'])
        self.assertEqual(result['matched_events'],1);self.assertEqual(result['mapped_channels'],2)
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'recording.jsonl';recording.export_recording(p,args[0],args[2],result,'test')
            from agent.mea_protocol import ResearchStream
            stream=ResearchStream();stream.feed(p.read_bytes());stream.finish()
            self.assertEqual(len(stream.session.channels),2);self.assertEqual(len(stream.spikes),1)

    def test_subset_never_waives_data_or_sdk_faults(self):
        for mutate in [lambda a:a[1].__setitem__(10,999),lambda a:a[3][0].__setitem__(3,-99.),
                lambda a:a[3].pop(0),lambda a:a[4].__setitem__('sdk_close_status',1),
                lambda a:a[0].__setitem__('raw_channels',[8,9,10]),
                lambda a:a[0].__setitem__('raw_channels',[0,0,1])]:
            args=subset_fixture();mutate(args)
            self.assertFalse(recording.audit_recording(*args,mapping_policy='finalized_h5_recorded_subset')['offline_integrity_passed'])
        with self.assertRaises(ValueError):recording.audit_recording(*subset_fixture(),mapping_policy='ignore_all')

    def test_confirmation_cannot_bypass_wrong_chip_or_stimulation(self):
        m=importlib.import_module('agent.mea_h5_replay')
        acq=dict(success=True,errors=[],stimulation=False,alpha=1,r2_reward_enabled=False,expected_chip_id='22021')
        confirmation=dict(schema='mea.user_chip_confirmation.v1',chip_id='28080',source='explicit_user_reply',
            statement='实际是 28080',original_expected_chip_id='22021')
        m.validate_confirmation(acq,{'chip_id':'28080'},confirmation,'28080')
        for a,meta,c,expected in [(acq,{'chip_id':'22021'},confirmation,'28080'),
                (acq|{'stimulation':True},{'chip_id':'28080'},confirmation,'28080'),
                (acq,{'chip_id':'28080'},confirmation|{'source':'inferred'},'28080'),
                (acq,{'chip_id':'28080'},confirmation,'22021')]:
            with self.assertRaises(ValueError):m.validate_confirmation(a,meta,c,expected)

    def test_reader_requires_explicit_chip_schema_pin_and_consistent_subset(self):
        from agent.mea_h5_replay import H5MappedRecording,SCHEMA,PACKAGE_SCHEMA
        from agent.mea_protocol import canonical
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);run=root/'accepted';run.mkdir();args=subset_fixture()
            (run/'baseline.raw.h5').write_bytes(b'unit audit fixture only')
            (run/'events.jsonl').write_bytes(b'unit audit fixture only')
            acquisition=dict(success=True,errors=[],stimulation=False,alpha=1,r2_reward_enabled=False,expected_chip_id='22021')
            confirmation=dict(schema='mea.user_chip_confirmation.v1',chip_id='28080',source='explicit_user_reply',
                statement='实际是 28080',original_expected_chip_id='22021')
            for n,obj in [('acquisition.json',acquisition),('audit.json',{'offline_integrity_passed':False}),('chip_confirmation.json',confirmation)]:
                (run/n).write_text(canonical(obj),encoding='utf-8')
            admission=recording.audit_recording(*args,mapping_policy='finalized_h5_recorded_subset')
            admission.update(schema=SCHEMA,metadata=args[0],inputs={n:recording.sha256(run/n) for n in
                ['baseline.raw.h5','events.jsonl','acquisition.json','audit.json','chip_confirmation.json']})
            (run/'admission.json').write_text(canonical(admission),encoding='utf-8')
            recording.export_recording(run/'recording.jsonl',args[0],args[2],admission,'accepted')
            def seal(schema=PACKAGE_SCHEMA):
                manifest=dict(schema=schema,accepted_run='accepted',files={'accepted/'+p.name:
                    dict(size=p.stat().st_size,sha256=recording.sha256(p)) for p in run.iterdir()})
                (root/'manifest.json').write_text(canonical(manifest),encoding='utf-8')
                return recording.sha256(root/'manifest.json')
            pin=seal();data=H5MappedRecording(root,manifest_sha256=pin,expected_chip_id='28080')
            self.assertEqual(data.summary()['chip_id'],'28080');self.assertEqual(data.summary()['analysis_channels'],2)
            self.assertEqual(data.snapshot(0,'a').mapped_channels,2)
            with self.assertRaises(ValueError):H5MappedRecording(root,manifest_sha256='0'*64,expected_chip_id='28080')
            with self.assertRaises(ValueError):H5MappedRecording(root,manifest_sha256=pin,expected_chip_id='22021')
            wrong=seal('old_strict_schema')
            with self.assertRaises(ValueError):H5MappedRecording(root,manifest_sha256=wrong,expected_chip_id='28080')
            admission['analysis_mapping'][0][1]=999
            (run/'admission.json').write_text(canonical(admission),encoding='utf-8');pin=seal()
            with self.assertRaises(ValueError):H5MappedRecording(root,manifest_sha256=pin,expected_chip_id='28080')

if __name__=='__main__':unittest.main()
