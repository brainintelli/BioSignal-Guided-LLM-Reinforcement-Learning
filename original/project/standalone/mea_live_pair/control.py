"""Operator-host orchestrator. Run after binding and local acquisition qualification."""
import argparse
import getpass
import json
import os
from pathlib import Path,PurePosixPath
from queue import Empty,Queue
import shlex
import sys
import time
from contextlib import ExitStack
from threading import Thread

from livepair.protocol import ARMS, PROTOCOL, binding, digest, host_map
from livepair.study import together,run_pair
from livepair.qualification import VERSION,require_pass
from livepair.figures import draw_sites,draw_responses
from livepair.predictability import MODE


class Peer:
    def __init__(self,stack,host,launch,role,log):
        import paramiko
        cli=stack.enter_context(paramiko.SSHClient())
        self.client=cli
        cli.load_system_host_keys()
        try:
            cli.connect(host,username='researcher',timeout=10)
        except paramiko.SSHException as exc:
            if not isinstance(exc,paramiko.AuthenticationException) and 'No authentication methods available' not in str(exc):raise
            password=os.environ.get('MEA_SSH_'+host.replace('.','_')) or getpass.getpass(f'{host} SSH password: ')
            cli.connect(host,username='researcher',password=password,
                        look_for_keys=False,allow_agent=False,timeout=10)
        cmd=shlex.join(['/usr/bin/python3',launch,role])
        self.stdin,self.stdout,err=cli.exec_command(cmd,timeout=900)
        self.log=stack.enter_context(log.open('xb'))
        def drain():
            for line in err:self.log.write(line);self.log.flush()
        Thread(target=drain,daemon=True).start()

    def selection(self,remote_dir):
        with self.client.open_sftp() as sftp:
            with sftp.open(str(PurePosixPath(remote_dir)/'spatial-selection.json'),'rb') as stream:
                return json.load(stream)

    def fetch_localization(self,root,session,out):
        remote=PurePosixPath(root)/session
        with self.client.open_sftp() as sftp:
            try:entries=set(sftp.listdir(str(remote)))
            except FileNotFoundError:return
            for name in ('site-plan.json','HWF-sites.svg','session-end.json','H-result.json','W-result.json','F-result.json'):
                if name in entries:sftp.get(str(remote/name),str(out/name))

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


def announce(figures):
    print(json.dumps(figures,ensure_ascii=False),flush=True)


def present_in_chat(figures,started_at):
    """Let the controlling process display the PNGs before it advances this same live session."""
    announce(figures)
    if not figures['passed']:
        return
    replies=Queue()
    Thread(target=lambda:replies.put(sys.stdin.readline().strip()),daemon=True).start()
    try:
        response=replies.get(timeout=max(0.,started_at+7200-130-time.time()))
    except Empty as exc:
        raise TimeoutError('Presentation exceeded the biological deadline') from exc
    if response!='shown':
        raise RuntimeError('Presentation ended before the figures were displayed')


def run(b,paths,out,present=announce):
    binding(b)
    out.mkdir(parents=True,exist_ok=False)
    def save(name,row):
        with (out/name).open('x',encoding='utf-8') as f:json.dump(row,f,indent=2,allow_nan=False)
    save('binding.json',b)
    hosts=host_map(b['replicate'])
    with ExitStack() as stack:
        device=Peer(stack,b['acquisition_host'],b['device_launch'],'device',out/'device.log')
        mode = b.get('feedback_encoding')
        if mode not in (None, MODE):
            raise ValueError('Unknown feedback encoding')
        feedback = {'feedback_encoding':mode} if mode else {}
        if device('protocol',**feedback)!={'qualification':VERSION,**feedback}:
            raise ValueError('Acquisition package lacks the spatial three-site protocol')
        try:
            if b.get('device_flow')=='braindevice_5555_7215_v1':
                try:base=device('open',b=b)
                finally:device.fetch_localization(b['record_root'],b['session_id'],out)
            else:
                base=device('open',b=b)
            if b.get('device_flow')=='braindevice_5555_7215_v1':
                resolved=base['resolved_binding']
                if any(resolved.get(k)!=v for k,v in b.items()) or set(resolved)-set(b)!={
                        'material','electrodes','route_file','route_sha256','prior_nonzero_attempts'}:
                    raise ValueError('Prepared sample identity changed during localization')
                b=resolved
                save('resolved-binding.json',b)
            if base['binding']!=digest(b):raise ValueError('Acquisition registration mismatch')
            report=device('calibrate')
            save('qualification.json',report)
            if report.get('schema')!=VERSION or report.get('binding')!=digest(b):
                raise ValueError('Qualification report differs from current registration')
            selection=device.selection(base['out'])
            save('spatial-selection.json',selection)
            files=[draw_sites(out,b,selection),draw_responses(out,b,report)]
            present(dict(event='qualification_figures_ready',chip=b['chip_id'],session=b['session_id'],
                         passed=report['passed'],files=[str(p) for p in files]))
            require_pass(report,digest(b))
            gpus={a:Peer(stack,h,paths[h]['launch'],'gpu',out/(a+'.log')) for a,h in hosts.items()}
            def init(a):
                h=hosts[a]
                cfg=dict(experiment=PROTOCOL,arm=a,devices=list(range(8)),batch_size=16,source='live',
                         r2_enabled=True,stimulation=True,binding_sha256=digest(b),normalization=base['norm'],
                         normalization_id=digest(base['norm']),seed=740001+b['replicate']*1000,
                         **feedback, **{k:paths[h][k] for k in ('project','godot','model_paths')})
                remote='/home/researcher/mea-live-pair-runs/'+b['session_id']+'-'+a
                return gpus[a]('initialize',config=cfg,out=remote)
            initial=together(init)
            if initial[ARMS[0]]!=initial[ARMS[1]]:raise ValueError('Training hosts differ at initialization')
            if mode and initial[ARMS[0]].get('feedback_encoding')!=mode:
                raise ValueError('Workers lack the registered action-contingent feedback')
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
    p.add_argument('--present-before-training',action='store_true',
                   help='Publish chart paths and accept the driver cue "shown" before opening GPUs')
    a=p.parse_args()
    b=json.loads(a.binding.read_text(encoding='utf-8-sig'))
    present=(lambda figures:present_in_chat(figures,binding(b))) if a.present_before_training else announce
    run(b,json.loads(a.servers.read_text(encoding='utf-8-sig')),a.output,present)
