import dataclasses
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

import h5py
import numpy as np
from test_recording import fixture
from mea_stim_v2.protocol import RealClock


class FastClock:
    def __init__(self):self.now=0.
    def mono(self):self.now+=10.;return self.now


class WorkerTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name);self.path,self.config=fixture(self.root)
        with h5py.File(self.path,'r+') as f:
            ds=f['data_store/data0000/events'];ev=ds[:]
            ev['eventid']=7001
            ev['eventmessage']=json.dumps({'session':'test-session','phase':'preflight'}).encode()
            ds[:]=ev
        self.resize_recording(220000)

    def resize_recording(self,count):
        with h5py.File(self.path,'r+') as f:
            group=f['data_store/data0000/groups/routed'];first=int(group['frame_nos'][0])
            del group['frame_nos'];del group['raw']
            group['frame_nos']=np.arange(first,first+count,dtype='u8')
            group.create_dataset('raw',shape=(1024,count),dtype='u2',chunks=(1,1000),fillvalue=512)

    def test_real_subprocess_preflight_checks_actual_h5_and_binds_receipt(self):
        from mea_stim_v2.worker import ClosedFileAnalyzer
        guards=[]
        analyzer=ClosedFileAnalyzer(self.config,self.root/'run',RealClock(),lambda:guards.append(1))
        q=analyzer('preflight',self.path)
        self.assertEqual(q['status'],'pass',q)
        self.assertEqual(q['metadata']['threshold_value'],5.)
        self.assertGreater(len(guards),0)
        receipt=json.loads((self.root/'run'/'qc'/'preflight.json').read_text())
        self.assertEqual(receipt['source']['path'],str(self.path.resolve()))
        self.assertEqual(receipt['session_id'],self.config.session_id)
        self.assertEqual(receipt['result']['status'],'pass')
        self.assertEqual(receipt['request_sha256'],q['request_sha256'])
        self.assertNotEqual(receipt['worker_pid'],__import__('os').getpid())

    def test_construction_does_not_create_ledger_run_directory(self):
        from mea_stim_v2.worker import ClosedFileAnalyzer
        run=self.root/'unclaimed-run'
        ClosedFileAnalyzer(self.config,run,RealClock(),lambda:None)
        self.assertFalse(run.exists())

    def test_wrong_preflight_marker_and_missing_threshold_fail_closed(self):
        from mea_stim_v2.worker import ClosedFileAnalyzer
        for kind in ('marker','threshold'):
            with h5py.File(self.path,'r+') as f:
                if kind=='marker':
                    ds=f['data_store/data0000/events'];ev=ds[:];ev['eventid']=7002;ds[:]=ev
                else:del f['data_store/data0000/settings/spike_threshold']
            q=ClosedFileAnalyzer(self.config,self.root/kind,RealClock(),lambda:None)('preflight',self.path)
            self.assertEqual(q['status'],'failed')

    def test_preflight_rejects_one_frame_short_of_10_seconds(self):
        from mea_stim_v2.worker import ClosedFileAnalyzer
        self.resize_recording(199999)
        q=ClosedFileAnalyzer(self.config,self.root/'short',RealClock(),lambda:None)('preflight',self.path)
        self.assertEqual(q['status'],'failed')
        self.assertIn('preflight_less_than_10_seconds',q['errors'])

    def test_fresh_recovery_requires_empty_events_and_600_actual_seconds(self):
        from mea_stim_v2.worker import ClosedFileAnalyzer
        with h5py.File(self.path,'r+') as f:
            group=f['data_store/data0000'];dtype=group['events'].dtype
            del group['events'];group.create_dataset('events',shape=(0,),dtype=dtype)
        self.resize_recording(11999999)
        analyze=ClosedFileAnalyzer(self.config,self.root/'recovery-run',RealClock(),lambda:None)
        short=analyze('recovery',self.path)
        self.assertEqual(short['status'],'failed')
        self.assertIn('recovery_less_than_600_seconds',short['errors'])
        self.resize_recording(12000000)
        full=analyze('recovery',self.path)
        self.assertEqual(full['status'],'pass',full)
        self.assertEqual(full['metadata']['duration_seconds'],600.)
        with h5py.File(self.path,'r+') as f:
            group=f['data_store/data0000'];del group['events']
            event=np.zeros(1,dtype=dtype);event['frameno']=group['groups/routed/frame_nos'][0]
            event['eventid']=7001;event['eventtype']=1
            event['eventmessage']=json.dumps({'session':'test-session','phase':'preflight'}).encode()
            group['events']=event
        contaminated=analyze('recovery',self.path)
        self.assertEqual(contaminated['status'],'failed')
        self.assertIn('recovery_requires_no_events',contaminated['errors'])

    def test_pilot_never_reuses_unconfirmed_baseline(self):
        from mea_stim_v2.worker import ClosedFileAnalyzer,WorkerError
        with self.assertRaises(WorkerError):
            ClosedFileAnalyzer(self.config,self.root/'run',RealClock(),lambda:None)('pilot',self.path,candidate_roi=list(range(59)))

    def test_timeout_kills_only_owned_child_and_writes_failure(self):
        from mea_stim_v2.worker import ClosedFileAnalyzer,WorkerError
        launched=[];popen=subprocess.Popen
        def slow_process(command,**kwargs):
            child=popen([sys.executable,'-c','import time; time.sleep(60)'],**kwargs)
            launched.append(child);return child
        with patch('mea_stim_v2.worker.subprocess.Popen',side_effect=slow_process):
            with self.assertRaises(WorkerError):
                ClosedFileAnalyzer(self.config,self.root/'run',FastClock(),lambda:None)('preflight',self.path)
        self.assertIsNotNone(launched[0].poll())
        receipt=json.loads((self.root/'run'/'qc'/'preflight.json').read_text())
        self.assertEqual(receipt['result']['status'],'failed')
        self.assertIn('timeout',' '.join(receipt['result']['errors']).lower())

    def test_guard_exception_reaps_child_and_propagates_stop(self):
        from mea_stim_v2.worker import ClosedFileAnalyzer
        launched=[];popen=subprocess.Popen;calls=[]
        def slow_process(command,**kwargs):
            child=popen([sys.executable,'-c','import time; time.sleep(60)'],**kwargs)
            launched.append(child);return child
        def guard():
            calls.append(1)
            if len(calls)>=3:raise RuntimeError('manual stop')
        with patch('mea_stim_v2.worker.subprocess.Popen',side_effect=slow_process):
            with self.assertRaisesRegex(RuntimeError,'manual stop'):
                ClosedFileAnalyzer(self.config,self.root/'run',RealClock(),guard)('preflight',self.path)
        self.assertEqual(len(launched),1)
        self.assertIsNotNone(launched[0].poll())

    def test_parent_rejects_worker_result_with_wrong_request_sha(self):
        from mea_stim_v2.worker import ClosedFileAnalyzer,WorkerError
        popen=subprocess.Popen
        # A real child writes a valid-looking but unbound result; no fake poll/wait.
        code="import json,sys; p=json.load(open(sys.argv[1])); json.dump({'request_sha256':'forged','result':{'status':'pass'}},open(sys.argv[2],'w'))"
        def forged_process(command,**kwargs):
            return popen([sys.executable,'-c',code,command[-3],command[-1]],**kwargs)
        with patch('mea_stim_v2.worker.subprocess.Popen',side_effect=forged_process):
            with self.assertRaises(WorkerError):
                ClosedFileAnalyzer(self.config,self.root/'run',RealClock(),lambda:None)('preflight',self.path)


if __name__=='__main__':unittest.main()
