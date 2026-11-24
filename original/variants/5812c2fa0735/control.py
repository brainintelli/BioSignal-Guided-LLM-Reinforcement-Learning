"""Operator-host orchestrator. Run after binding and local acquisition qualification."""
import argparse
import getpass
import json
from pathlib import Path
import shlex
import sys
from contextlib import ExitStack
from threading import Thread

from livepair.protocol import ARMS, PROTOCOL, binding, digest, host_map
from livepair.study import together,run_pair
from livepair.qualification import VERSION,require_pass


class Peer:
    def __init__(self,stack,host,launch,role,log):
        import paramiko
        cli=stack.enter_context(paramiko.SSHClient())
        cli.load_system_host_keys()
        try:
            cli.connect(host,username='researcher',timeout=10)
        except paramiko.SSHException as exc:
            if not isinstance(exc,paramiko.AuthenticationException) and 'No authentication methods available' not in str(exc):raise
            cli.connect(host,username='researcher',password=getpass.getpass(f'{host} SSH password: '),
                        look_for_keys=False,allow_agent=False,timeout=10)
        cmd=shlex.join(['/usr/bin/python3',launch,role])
        self.stdin,self.stdout,err=cli.exec_command(cmd,timeout=900)
        self.log=stack.enter_context(log.open('xb'))
        def drain():
            for line in err:self.log.write(line);self.log.flush()
        Thread(target=drain,daemon=True).start()

    def __call__(self,op,**args):
        self.stdin.write(json.dumps(dict(op=op,args=args),allow_nan=False)+'\n')
        self.stdin.flush()
        line=self.stdout.readline()
        if not line:raise RuntimeError('Peer disconnected')
        response=json.loads(line)
        if not response['ok']:raise RuntimeError(response.get('error',response))
        return response['value']

    def close(self,device=False):
        self.stdin.write('{"op":"close"}\n')
        self.stdin.flush()
        self.stdin.channel.shutdown_write()
        if device:
            status=json.loads(self.stdout.readline())
            if not status['ok'] or not status['value']['closed']:raise RuntimeError('Device close unverified')
        if self.stdout.channel.recv_exit_status()!=0:raise RuntimeError('Worker exit unverified')


def run(b,paths,out):
    binding(b)
    out.mkdir(parents=True,exist_ok=False)
    def save(name,row):
        with (out/name).open('x',encoding='utf-8') as f:json.dump(row,f,indent=2,allow_nan=False)
    save('binding.json',b)
    hosts=host_map(b['replicate'])
    with ExitStack() as stack:
        device=Peer(stack,b['acquisition_host'],b['device_launch'],'device',out/'device.log')
        if device('protocol')!={'qualification':VERSION}:
            raise ValueError('Acquisition package lacks the spatial three-site protocol')
        try:
            base=device('open',b=b)
            if base['binding']!=digest(b):raise ValueError('Acquisition registration mismatch')
            report=device('calibrate')
            save('qualification.json',report)
            require_pass(report,digest(b))
            gpus={a:Peer(stack,h,paths[h]['launch'],'gpu',out/(a+'.log')) for a,h in hosts.items()}
            def init(a):
                h=hosts[a]
                cfg=dict(experiment=PROTOCOL,arm=a,devices=list(range(8)),batch_size=16,source='live',
                         r2_enabled=True,stimulation=True,binding_sha256=digest(b),normalization=base['norm'],
                         normalization_id=digest(base['norm']),seed=740001+b['replicate']*1000,
                         **{k:paths[h][k] for k in ('project','godot','model_paths')})
                remote='/home/researcher/mea-live-pair-runs/'+b['session_id']+'-'+a
                return gpus[a]('initialize',config=cfg,out=remote)
            initial=together(init)
            if initial[ARMS[0]]!=initial[ARMS[1]]:raise ValueError('Training hosts differ at initialization')
            save('initial.json',initial)
            with (out/'audit.jsonl').open('x',encoding='utf-8') as f:
                def record(r):
                    f.write(json.dumps(r,allow_nan=False)+'\n');f.flush()
                result=run_pair(device,gpus,b['replicate'],record)
        finally:
            device.close(device=True)
        together(lambda a:gpus[a].close())
        save('paired_result.json',result)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--binding',type=Path,required=True)
    p.add_argument('--servers',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    run(json.loads(a.binding.read_text(encoding='utf-8-sig')),
        json.loads(a.servers.read_text(encoding='utf-8-sig')),a.output)
