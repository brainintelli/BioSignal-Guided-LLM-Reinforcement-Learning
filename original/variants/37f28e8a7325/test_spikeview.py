"""Small artificial format fixtures, explicitly not biological training inputs."""
import copy
import json
import tempfile
import unittest
from pathlib import Path
import h5py
import numpy as np
from export_spikeview import export
from sr_data import read_spikeview,read,write,sha,file_ref,derive,clean_windows


class SpikeviewTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(prefix='unit_fixture_only_',dir=Path(__file__).parent)
        self.base=Path(self.tmp.name);self.raw=self.base/'synthetic_format_fixture.h5'
        with h5py.File(self.raw,'x') as f:
            f.create_dataset('wellplate/id',data=[b'28073'])
            s=f.require_group('data_store/data0000');s.create_dataset('well_id',data=[0])
            s.create_dataset('settings/sampling',data=[20000.]);s.create_dataset('settings/lsb',data=[.000001])
            s.create_dataset('settings/mapping',data=np.array([(0,10),(1,11)],dtype=[('channel','i4'),('electrode','i4')]))
            s.create_dataset('spikes',data=np.array([(20001,0,-3.),(40001,1,-5.)],dtype=[('frameno','i8'),('channel','i4'),('amplitude','f8')]))
            s.create_dataset('events',data=np.array([(20000,99)],dtype=[('frameno','i8'),('event','i4')]))
            s.create_dataset('groups/routed/frame_nos',data=np.arange(100000,dtype='uint64'))
            s.create_dataset('groups/routed/channels',data=[0,1])
            s.create_dataset('groups/routed/raw',shape=(2,100000),dtype='uint16',compression='gzip')
        self.qc=dict(schema='mea.28073.source_qc.v1',source_id='unit_format_fixture_only',parent_raw=dict(sha256=sha(self.raw),bytes=self.raw.stat().st_size),
                     finalized=True,passed_for_declared_scope=True,excluded_intervals=[],declared_out_of_scope_events=0,
                     fixture_warning='ARTIFICIAL FORMAT TEST ONLY; NEVER TRAIN')
        self.qcp=self.base/'fixture.qc.json';write(self.qcp,self.qc)
        self.view=self.base/'fixture.spikeview.h5'
    def tearDown(self):self.tmp.cleanup()
    def source(self):
        return dict(source_id='unit_format_fixture_only',parent_raw=self.qc['parent_raw'],qc_receipt={'sha256':sha(self.qcp)},
                    well_id=0,raw_interval=[0,100000],declared_out_of_scope_events=0)

    def test_export_retains_real_event_fields_and_explicit_view_provenance(self):
        receipt=export(self.raw,self.view,self.qcp)
        self.assertFalse(receipt['contains_raw_waveforms'])
        with h5py.File(self.raw,'r') as raw,h5py.File(self.view,'r') as view:
            self.assertNotIn('raw',view['data_store/data0000/groups/routed'])
            self.assertTrue(np.array_equal(raw['data_store/data0000/events'][:],view['data_store/data0000/events'][:]))
        rows,identity=read_spikeview(self.view,self.source())
        self.assertEqual([r['spike_count'] for r in rows],[0,1,1,0,0])
        self.assertEqual(rows[1]['mean_abs_amplitude_uv'],3.)
        self.assertFalse(identity['contains_raw_waveforms'])

    def test_parent_and_qc_binding_tamper_rejected(self):
        export(self.raw,self.view,self.qcp)
        for key in ('parent_raw','qc_receipt'):
            source=copy.deepcopy(self.source());source[key]['sha256']='0'*64
            with self.assertRaises(ValueError):read_spikeview(self.view,source)

    def test_source_id_binding_tamper_rejected(self):
        export(self.raw,self.view,self.qcp)
        source=self.source();source['source_id']='another_recording'
        with self.assertRaisesRegex(ValueError,'source identity'):read_spikeview(self.view,source)

    def test_noncontiguous_frames_rejected(self):
        export(self.raw,self.view,self.qcp)
        with h5py.File(self.view,'r+') as f:f['data_store/data0000/groups/routed/frame_nos'][90000]=90002
        with self.assertRaisesRegex(ValueError,'noncontiguous'):read_spikeview(self.view,self.source())

    def test_file_integrity_and_traversal_rejected(self):
        ref=dict(path=self.qcp.name,sha256=sha(self.qcp))
        self.assertEqual(file_ref(self.base,ref),self.qcp)
        self.qcp.write_text('{}')
        with self.assertRaisesRegex(ValueError,'SHA'):file_ref(self.base,ref)
        ref['path']='../test_spikeview.py'
        with self.assertRaisesRegex(ValueError,'inside bundle'):file_ref(self.base,ref)

    def test_synthetic_manifest_can_never_be_loaded_as_training(self):
        p=self.base/'manifest.json'
        write(p,dict(schema='mea.28073.qc_manifest.v1',chip_id='28073',provenance='synthetic_unit_fixture_only',finalized=True))
        with self.assertRaisesRegex(ValueError,'real acquisition'):derive(p)

    def test_refuses_overwriting_source_or_existing_export(self):
        with self.assertRaises(ValueError):export(self.raw,self.raw,self.qcp)
        export(self.raw,self.view,self.qcp)
        with self.assertRaises(ValueError):export(self.raw,self.view,self.qcp)

if __name__=='__main__':unittest.main()
