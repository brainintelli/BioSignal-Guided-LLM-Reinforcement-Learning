import sys,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'mea_next_recording'))
from recording_backend import RecordingBackend

class Saving:
    def __init__(self):self.keys=set();self.started=False;self.commands=[]
    def open_directory(self,x):pass
    def set_legacy_format(self,x):pass
    def start_file(self,x):self.keys=set();self.started=False
    def write_assay_input(self,k,v):
        if k in self.keys:raise RuntimeError('HDF5 duplicate dataset')
        if self.started:raise RuntimeError('Metadata after recording start')
        self.keys.add(k);self.commands.append((k,v))
    def group_delete_all(self):pass
    def group_define(self,*args):pass
    def start_recording(self,x):self.started=True

class RecordingTests(unittest.TestCase):
    def test_transport_timeout_includes_connection_time(self):
        d=object.__new__(RecordingBackend)
        clock=type('Clock',(),{'now':0.,'monotonic':lambda self:self.now})();d.clock=clock
        session=type('Socket',(),{'__enter__':lambda self:self,'__exit__':lambda *args:None})()
        remaining=[]
        def connect(*args,**kwargs):clock.now+=.75;return session
        class Api:
            def __init__(self,s,timeout):remaining.append(timeout)
            def send(self,message):return 'OK'
        with patch('recording_backend.socket.create_connection',side_effect=connect),patch('recording_backend.Api',Api):
            self.assertEqual(d._network_request('query',2),'OK')
        self.assertEqual(remaining,[1.25])
    def test_unique_metadata_before_start_for_both_files(self):
        with tempfile.TemporaryDirectory() as td:
            d=object.__new__(RecordingBackend);d.out=Path(td);d.record_times={}
            d.clock=type('Clock',(),{'monotonic':lambda self:10.})()
            fake=Saving()
            with patch('recording_backend.mx.Saving',return_value=fake):
                for phase in ['identity','stimulus_post']:
                    d.open(phase)
                    self.assertEqual(len(fake.keys),5)
                    self.assertEqual(fake.commands[-5],('protocol','multi_bounded_short_v1'))
                    self.assertTrue(fake.started)

if __name__=='__main__':unittest.main()
