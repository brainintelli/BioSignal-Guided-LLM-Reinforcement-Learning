"""Verify saved host keys, the actual acquisition RPC, GPU workers and their eight-card communication."""
import json
import os
from pathlib import Path
import shlex
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime,timezone
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'.tools'/'mea_ssh_deps'))
import paramiko
SERVERS=json.loads((ROOT/'output/mea_spatial_gate_20261118/servers.json').read_text(encoding='utf-8-sig'))
REFERENCE=json.loads((ROOT/'output/mea_live_pair_preparation_20261118/servers.json').read_text(encoding='utf-8-sig'))

def check(host):
    with paramiko.SSHClient() as cli:
        cli.load_system_host_keys()
        cli.connect(host,username='researcher',password=os.environ.get('MEA_SSH_'+host.replace('.','_')),timeout=7,auth_timeout=8)
        def command(parts,body=''):
            stdin,out,err=cli.exec_command(shlex.join(parts),timeout=180)
            if body:stdin.write(body);stdin.flush()
            stdin.channel.shutdown_write()
            result=out.read().decode()
            errors=err.read().decode(errors='replace')
            if out.channel.recv_exit_status():raise RuntimeError(errors[-1200:])
            return result
        if host not in SERVERS:
            output=command(['/usr/bin/python3','/home/researcher/brainDesktop/mea-standby-20261118/session.py','device'],
                           '{"op":"protocol"}\n{"op":"close"}\n')
            rows=[json.loads(i) for i in output.splitlines()]
            assert rows==[{'ok':True,'value':{'qualification':'mea.site_response.v1'}},{'ok':True,'value':{'closed':True,'errors':[]}}]
            return dict(rpc=rows)
        launch=SERVERS[host]['launch']
        devices=command(['nvidia-smi','--query-gpu=name,memory.used','--format=csv,noheader']).splitlines()
        jobs=command(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader']).strip()
        if len(devices)!=8:raise RuntimeError('Expected eight GPUs')
        command(['/usr/bin/python3',launch,'gpu'],'{"op":"close"}\n')
        report=dict(gpus=devices,worker_ready=True,active_compute_pids=jobs)
        if not jobs:
            rows=command(['/usr/bin/python3',launch,'check']).splitlines()
            result=json.loads(rows[-1])
            if result['nccl_all_reduce_sum']!=36 or result['actor']!=REFERENCE[host]['actor']:
                raise RuntimeError('Distributed sum or checkpoint differs')
            report.update(nccl_sum=36,actor=result['actor'],launcher=launch)
        return report

if __name__=='__main__':
    hosts=['192.0.2.21',*SERVERS]
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures={h:pool.submit(check,h) for h in hosts}
        report={}
        for h,future in futures.items():
            try:report[h]=future.result()
            except Exception as exc:report[h]={'error':str(exc)}
    print(json.dumps(dict(utc=datetime.now(timezone.utc).isoformat(),hosts=report)))
    sys.exit(int(any('error' in result for result in report.values())))
