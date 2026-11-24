"""Check the restored full route without an additional routing or offset change."""
import fcntl
from pathlib import Path
import json
import signal
import socket
import sys
import time
from passive import assess,binding,capture,mx,off,rank,read,route_mapping,transport,write_once,_server_snapshot

def run(root):
    dest=root/'recovery';b=read(root/'baseline-binding.json');start=binding(b)
    assert b['chip_id']=='27701' and b['device_hostname']==socket.gethostname()
    assert 0<=time.time()-start<5700-180 and not (root/'selected-scan.json').exists()
    switch=read(dest/'switch-end.json');assert switch['closed'] and not switch.get('error')
    cfg=root.parents[1]/'sample-route.cfg';mapping=route_mapping(cfg);roi=read(root.parent/'cpp-crossover-v1/design.json')['roi']
    result=dict(started=time.time(),phases=[])
    with (Path.home()/'.local/state/mea_photo_full/device.lock').open('a') as lock,transport.bind_sdk(mx,transport.Endpoint(timeout=45)):
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);pid,opened=_server_snapshot();assert not opened
        try:
            assert mx.send_raw('wellplate_query_version').strip()=='0'
            conf=dest/'settled-full-route.cfg'
            # This named configuration belongs to the GUI. The current physical routing is
            # independently checked against the full expected 161 pairs in the new H5.
            result['gui_mapping_matches_bound_route']=route_mapping(conf)==mapping
            prev=None
            for name in ('settled-full-512','independent-settled-full-512'):
                f=capture(dest/name,pid);qc=assess(f,b,roi,mapping,512);write_once(f.parent/'qc.json',qc)
                result['phases'].append(name);print(name,qc['stable_counts'],qc['spiking_channels'],flush=True)
                if not qc['passed']:break
                r=rank(f,b)
                if prev is not None:r['new']=[x for x in r['new'] if x['electrode'] in prev]
                write_once(f.parent/'rank.json',r)
                if not r['new']:break
                if prev is not None:
                    write_once(root/'selected-scan.json',dict(scan='recovery/'+name,file=str(f),target=r['new'][0]))
                prev={x['electrode'] for x in r['new']}
        except BaseException:
            import traceback
            result['error']=traceback.format_exc()
        finally:
            result.update(off_acks=off(),closed=_server_snapshot()==(pid,[]),finished=time.time())
            write_once(dest/'settled-record-end.json',result);print(result.get('error','closed'),flush=True)
if __name__=='__main__':
    def stop(*_):raise TimeoutError('Original sample clock')
    signal.signal(signal.SIGALRM,stop);signal.alarm(250)
    run(Path(sys.argv[1]))
