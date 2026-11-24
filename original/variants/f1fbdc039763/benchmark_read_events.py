"""CPU artificial-format microbenchmark; never creates a training source."""
import json
import tempfile
import time
from pathlib import Path
import h5py
import numpy as np
from sr_data import read_events,digest


def main():
    with tempfile.TemporaryDirectory(prefix='unit_benchmark_only_',dir=Path(__file__).parent) as directory:
        p=Path(directory)/'ARTIFICIAL_EVENT_FORMAT_ONLY.h5'
        values=np.zeros(100000,dtype=[('frameno','uint64'),('channel','int32'),('amplitude','float32')])
        values['frameno']=np.arange(100000,dtype='uint64')+510779078;values['channel']=np.arange(100000)%770;values['amplitude']=-3.25
        with h5py.File(p,'x') as f:f.create_dataset('spikes',data=values,compression='gzip')
        with h5py.File(p,'r') as f:
            start=time.perf_counter();scalar=[[i,int(e['frameno']),int(e['channel']),float(e['amplitude'])] for i,e in enumerate(f['spikes'])];old=time.perf_counter()-start
            start=time.perf_counter();chunked=read_events(f['spikes']);new=time.perf_counter()-start
        assert scalar==chunked and digest(scalar)==digest(chunked)
        result=dict(fixture='artificial_format_microbenchmark_not_biological_data',events=len(values),scalar_seconds=old,chunked_seconds=new,
                    identical_values=True,identical_sha256=True,events_sha256=digest(chunked),speedup_observed=old/new,
                    caveat='cached synthetic local HDF5 microbenchmark; not actual acquisition or GPU throughput')
        (Path(__file__).parent/'spike_read_benchmark.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
        print(json.dumps(result))

if __name__=='__main__':main()
