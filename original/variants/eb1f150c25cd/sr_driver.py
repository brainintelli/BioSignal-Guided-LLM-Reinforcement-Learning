"""Eight isolated seed-block pipelines, bounded owned processes and telemetry."""
import argparse,concurrent.futures,json,os,signal,subprocess,sys,threading,time
from pathlib import Path
from sr_data import read,write,sha,canonical
from sr_protocol import ARMS,PROTOCOL

def cell_plan(reg):
    if reg['pilot'] is not False or reg['protocol']!=PROTOCOL or set(reg['arm_orders'])!={str(b) for b in range(8)}:raise ValueError('formal eight-block registration required')
    result={}
    for b in range(8):
        expected=list(ARMS[b%len(ARMS):]+ARMS[:b%len(ARMS)])
        if reg['arm_orders'][str(b)]!=expected:raise ValueError('registered arm order changed')
        result[b]=[dict(block=b,arm=a,name=f'b{b:02d}_{a}') for a in expected]
    return result

def main(root):
    base=Path(__file__).resolve().parent;reg=read(root/'registration.json');plan=cell_plan(reg)
    if sha(root/'registration.json')!=read(root/'registration_pin.json')['sha256']:raise ValueError('registration pin')
    if subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():raise ValueError('GPU compute already occupied')
    code={n:sha(base/n) for n in ['sr_driver.py','sr_analyze.py']}
    write(root/'execution_plan.json',dict(code=code,registration_sha256=sha(root/'registration.json'),plan=plan,maximum_collection_decisions=12288,
        deadline_seconds=14400,per_stage_limit_seconds=3600,created_time=time.time(),gpu_assignment='physical GPU equals paired block'))
    logs=root/'logs';logs.mkdir();deadline=time.monotonic()+14400;failed=threading.Event();done=threading.Event();lock=threading.RLock();active={};started=time.time()
    def emit(event,**fields):
        x=dict(time=time.time(),event=event,**fields)
        with lock:
            with (root/'progress.jsonl').open('a',encoding='utf-8') as f:f.write(canonical(x)+'\n')
        print(canonical(x),flush=True)
    def cancel(signum,frame):
        failed.set()
        with lock:
            for p,_ in active.values():
                try:os.killpg(p.pid,signal.SIGTERM)
                except ProcessLookupError:pass
    signal.signal(signal.SIGTERM,cancel);signal.signal(signal.SIGINT,cancel)
    def stage(cell,which):
        out=root/cell['name'];command=[sys.executable,str(base/('sr_run.py' if which=='collect' else 'sr_verify.py'))]
        if which=='collect':command+=['collect','--root',str(root),'--output',str(out),'--arm',cell['arm'],'--block',str(cell['block'])]
        else:command+=['--root',str(root),'--input',str(out)]
        key=cell['name']+'_'+which;env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(cell['block']));rc=124
        with (logs/(key+'.log')).open('x') as log:
            remaining=min(3600,deadline-time.monotonic())
            if remaining>0:
                p=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,env=env,start_new_session=True)
                with lock:active[key]=(p,dict(cell=cell,stage=which,pid=p.pid))
                emit('stage_started',cell=cell['name'],stage=which,pid=p.pid,gpu=cell['block'])
                try:rc=p.wait(timeout=remaining)
                except subprocess.TimeoutExpired:
                    try:os.killpg(p.pid,signal.SIGTERM)
                    except ProcessLookupError:pass
                    try:p.wait(timeout=10)
                    except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
                if rc!=0:
                    try:os.killpg(p.pid,signal.SIGTERM)
                    except ProcessLookupError:pass
                with lock:active.pop(key,None)
        (logs/(key+'.exit')).write_text(str(rc)+'\n');emit('stage_finished',cell=cell['name'],stage=which,exit=rc)
        return rc==0
    def worker(cells):
        results=[]
        for cell in cells:
            if failed.is_set():break
            passed=stage(cell,'collect')
            if passed:passed=stage(cell,'verify')
            if passed:
                v=read(root/cell['name']/'independent_verification.json');passed=v['passed'] is True and v['block']==cell['block'] and v['arm']==cell['arm'] and v['registration_sha256']==sha(root/'registration.json')
            results.append(cell|dict(passed=passed))
            if not passed:failed.set();break
        return results
    def telemetry():
        with (root/'gpu_telemetry.jsonl').open('x') as f:
            while True:
                with lock:owners=[x for p,x in active.values()]
                try:
                    x=dict(time=time.time(),active=owners,gpus=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True,timeout=15),
                        compute=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid,used_memory','--format=csv,noheader,nounits'],text=True,timeout=15))
                except Exception as e:x=dict(time=time.time(),active=owners,error=type(e).__name__+': '+str(e))
                f.write(canonical(x)+'\n');f.flush()
                if done.is_set():break
                done.wait(5)
    monitor=threading.Thread(target=telemetry);monitor.start();results=[]
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
            for future in concurrent.futures.as_completed([pool.submit(worker,plan[b]) for b in range(8)]):results.extend(future.result())
    finally:done.set();monitor.join()
    passed=len(results)==24 and all(x['passed'] for x in results) and code=={n:sha(base/n) for n in code}
    write(root/'driver_status.json',dict(passed=passed,expected_cells=24,finished_cells=len(results),results=results,seconds=time.time()-started))
    emit('driver_finished',passed=passed,cells=len(results))
    if not passed:raise SystemExit(1)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True,type=Path);main(p.parse_args().root)
