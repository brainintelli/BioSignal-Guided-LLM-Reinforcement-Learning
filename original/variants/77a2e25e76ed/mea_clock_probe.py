"""Authenticated SSH stdio timestamps; no listening service or stored credentials."""
import argparse
import json
from pathlib import Path
import select
import shlex
import subprocess
import sys
import time
import uuid
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from agent.mea_timing import clock_bounds


def server():
    deadline=time.monotonic()+90
    for _ in range(100):
        ready,_,_=select.select([sys.stdin],[],[],max(0,min(10,deadline-time.monotonic())))
        if not ready: return
        line=sys.stdin.buffer.readline(4097)
        sm1=time.monotonic_ns();s1=time.time_ns()
        if not line: return
        if len(line)>4096 or not line.endswith(b'\n'): raise ValueError('invalid probe')
        row=json.loads(line)
        if set(row)!={'seq','nonce'} or type(row['seq']) is not int or not isinstance(row['nonce'],str) or len(row['nonce'])>64:
            raise ValueError('invalid request')
        sm2=time.monotonic_ns();s2=time.time_ns()
        print(json.dumps(dict(seq=row['seq'],nonce=row['nonce'],s1=s1,s2=s2,server_elapsed_ns=sm2-sm1)),flush=True)


def client(args):
    if not 3<=args.count<=100 or not .1<=args.interval<=1: raise ValueError('bounded count/interval required')
    args.output.mkdir(parents=True,exist_ok=False)
    command=['ssh','-T','-o','StrictHostKeyChecking=yes','-o','ConnectTimeout=8']
    if args.bind_address:command.extend(['-o','BindAddress='+args.bind_address])
    command.extend([args.host,'python3 -u '+shlex.quote(args.remote_script)+' --server'])
    child=None; samples=[];error=None
    try:
        child=subprocess.Popen(command,stdin=subprocess.PIPE,stdout=subprocess.PIPE,bufsize=0)
        for seq in range(args.count):
            nonce=uuid.uuid4().hex
            cm0=time.monotonic_ns();c0=time.time_ns()
            child.stdin.write((json.dumps(dict(seq=seq,nonce=nonce))+'\n').encode());child.stdin.flush()
            ready,_,_=select.select([child.stdout],[],[],60 if seq==0 else 5)
            if not ready:raise TimeoutError('clock response timeout')
            line=child.stdout.readline(4097);c3=time.time_ns();cm3=time.monotonic_ns()
            if not line.endswith(b'\n') or len(line)>4096:raise ValueError('invalid clock response')
            reply=json.loads(line)
            if reply['seq']!=seq or reply['nonce']!=nonce:raise ValueError('clock response identity')
            sample=dict(seq=seq,c0=c0,c3=c3,client_elapsed_ns=cm3-cm0,
                        s1=reply['s1'],s2=reply['s2'],server_elapsed_ns=reply['server_elapsed_ns'])
            samples.append(sample)
            with (args.output/'samples.jsonl').open('a') as f:f.write(json.dumps(sample)+'\n')
            time.sleep(args.interval)
    except Exception as exc:error=str(exc)
    finally:
        if child:
            if child.stdin:child.stdin.close()
            try:child.wait(timeout=3)
            except subprocess.TimeoutExpired:child.terminate();child.wait(timeout=3)
    report=dict(schema='mea.host_clock.v1',success=False,error=error,samples=len(samples))
    if error is None:
        try:report.update(clock_bounds(samples),success=True)
        except ValueError as exc:report['error']=str(exc)
    (args.output/'clock_audit.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report));return 0 if report['success'] else 2


def main():
    p=argparse.ArgumentParser();p.add_argument('--server',action='store_true')
    p.add_argument('--host');p.add_argument('--remote-script');p.add_argument('--bind-address')
    p.add_argument('--count',type=int,default=60);p.add_argument('--interval',type=float,default=.5)
    p.add_argument('--output',type=Path);args=p.parse_args()
    if args.server:server();return 0
    if not args.host or not args.remote_script or not args.output:p.error('host, remote-script, output required')
    return client(args)


if __name__=='__main__':sys.exit(main())
