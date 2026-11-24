"""Exchange the two passive electrode subsets on the same physical amplifier rows."""
from pathlib import Path
import fcntl
import hashlib
import json
import signal
import socket
import sys
import time
from types import SimpleNamespace
import uuid
import numpy as np
from passive import assess,binding,capture,H5Recording,mx,off,rank,read,route_mapping,transport,write_once,_server_snapshot

def verify_switch(f,b,mapping,groups):
    with H5Recording(f,SimpleNamespace(**b)) as r:
        assert sorted(r.mapping)==sorted(mapping) and not r.events and r.metadata['chip_id']==b['chip_id']
        output={}
        for label,rows in groups.items():
            rows=sorted(rows);result=[]
            for t in (3,23,43,57):
                n=int(t*r.fs);a=np.asarray(r._raw[rows,n:n+int(r.fs)])
                clipped=(a==0)|(a==1023);result.append(dict(t=t,channels=int(clipped.any(axis=1).sum()),
                    fraction=float(clipped.mean()),simultaneous=int(clipped.sum(axis=0).max())))
            output[label]=result
    return output

def run(root):
    b=read(root/'baseline-binding.json');base=binding(b);dest=root/'recovery';stage=root.parents[1]
    assert b['device_hostname']==socket.gethostname() and b['chip_id']=='27701'
    assert 0<=time.time()-base<5700-300 and read(dest/'end.json')['closed']
    assert not (root/'selected-scan.json').exists()
    cfg=stage/'sample-route.cfg';assert hashlib.sha256(cfg.read_bytes()).hexdigest()==read(stage/'session-route.json')['route_sha256']
    m=route_mapping(cfg);roi=read(root.parent/'cpp-crossover-v1/design.json')['roi']
    cs=sorted(roi,key=lambda c:dict(m)[c]);g={'A':cs[::2],'B':cs[1::2]}
    assert len(g['A'])==len(g['B'])==79
    write_once(dest/'switch-plan.json',dict(groups=g,original_roi=roi,phases=['A-connected','B-connected'],gain=512))
    state=dict(started=time.time(),phases=[])
    guard=Path.home()/'.local/state/mea_photo_full/device.lock'
    with guard.open('a') as lock,transport.bind_sdk(mx,transport.Endpoint(timeout=45)):
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        pid,busy=_server_snapshot();assert not busy and mx.send_raw('wellplate_query_version').strip()=='0'
        a=mx.Array('isol_'+uuid.uuid4().hex)
        def install(excluded):
            off();mx.activate([0]);mx.set_gain(512);mx.hpf('300Hz');a.load_config(str(cfg))
            for u in range(32):
                ch=a.query_amplifier_at_stimulation(u).strip()
                if ch not in ('','-1'):a.disconnect_amplifier_from_stimulation(int(ch))
            for c in excluded:assert a.disconnect_electrode(dict(m)[c]).strip().lower()=='ok'
            a.download([0]);time.sleep(5);mx.offset();time.sleep(30)
            assert all(a.query_amplifier_at_stimulation(u).strip() in ('','-1') for u in range(32))
        try:
            for keep,drop in [('A','B'),('B','A')]:
                if time.time()-base>5700-270:break
                name=keep+'-connected';install(g[drop]);newmap=[row for row in m if row[0] not in g[drop]]
                f=capture(dest/name,pid);qc=assess(f,b,g[keep],newmap,512)
                qc['switch']=verify_switch(f,b,newmap,g);write_once(f.parent/'qc.json',qc)
                state['phases'].append(name);print(name,qc['stable_counts'],qc['switch'],flush=True)
                if not qc['passed']:continue
                first=rank(f,b)
                if not first['new']:continue
                nxt=capture(dest/('independent-'+name),pid);q=assess(nxt,b,g[keep],newmap,512)
                write_once(nxt.parent/'qc.json',q)
                result=rank(nxt,b);fixed={v['electrode'] for v in first['new']}
                result['new']=[r for r in result['new'] if r['electrode'] in fixed]
                write_once(nxt.parent/'rank.json',result)
                if q['passed'] and result['new']:
                    write_once(root/'selected-scan.json',dict(scan='recovery/'+nxt.parent.name,file=str(nxt),target=result['new'][0]))
                    break
        except BaseException:
            import traceback
            state['error']=traceback.format_exc()
        finally:
            try:install([])
            finally:
                a.close();state.update(off_acks=off(),closed=_server_snapshot()==(pid,[]),finished=time.time())
                write_once(dest/'switch-end.json',state);print(state.get('error','closed'),flush=True)
if __name__=='__main__':
    def stop(*args):raise TimeoutError('Sample deadline')
    signal.signal(signal.SIGALRM,stop);signal.alarm(665)
    run(Path(sys.argv[1]))
