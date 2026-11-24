"""Write each H5 metadata key once, before starting acquisition."""
from device import MaxLabBackend
from brain_stimulation import maxlab as mx
from brain_stimulation.transport import Api
import socket

class RecordingBackend(MaxLabBackend):
    def _network_request(self,message,timeout):
        end=self.clock.monotonic()+timeout
        with socket.create_connection(('127.0.0.1',7215),timeout=min(2,timeout)) as connection:
            remaining=end-self.clock.monotonic()
            if remaining<=0:raise TimeoutError('Command budget exhausted during connection')
            return Api(connection,remaining).send(message)

    def open(self,name):
        if name not in ('identity','stimulus_post'):raise ValueError('Unexpected file stem')
        self.current_file=self.out/(name+'.raw.h5')
        if self.current_file.exists():raise FileExistsError(str(self.current_file))
        s=mx.Saving();s.open_directory(str(self.out));s.set_legacy_format(False)
        self.file_attempted=True;s.start_file(name)
        for key,value in {'protocol':'multi_bounded_short_v1','alpha':'1','r2':'record_only','training':'disabled','phase':name}.items():
            s.write_assay_input(key,value)
        s.group_delete_all();s.group_define(0,'routed',list(range(1024)))
        self.record_attempted=True;s.start_recording([0])
        self.record_times[name]={'start':self.clock.monotonic()}
        self.last_growth=self.clock.monotonic();self.last_size=0;self.growth_events=[]
