"""Recover tick-level game trajectories from audit-validated decisions on the original game engine."""
import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import shlex
import sys
import time
import zipfile
from replay_hosts import ROOT, SERVERS, command, paramiko, put

HOST='192.0.2.23'
GROUPS={'prior':('pure','ordered','shuffled'),'original':('stateonly','r2only'),'added':('stateonly','r2only')}
KEYS=('before','after','obs_hash','duration','terminal','action')


def make_pack(folder,group):
    protocol=json.loads((folder/'protocol.json').read_bytes())
    old=folder.parent/'three-arm-six-seeds'
    engines=json.loads((old/'game-runtime-hashes.json').read_bytes())
    data=dict(engine=engines[HOST],horizon=protocol['tick_horizon'],arms={})
    selected=protocol.get('rerun_arms',GROUPS[group]) if group!='prior' else GROUPS[group]
    for arm in selected:
        arm_data={}
        for cohort,seeds in protocol['cohorts'].items():
            if group!='prior' and group!=cohort:continue
            if group=='prior':
                root=folder.parent if cohort=='original' and arm!='pure' else old/cohort
            else:
                root=folder/cohort
            for seed in seeds:
                path=root/arm/str(seed)/'eval-trace.json'
                traces=json.loads(path.read_bytes())
                if set(map(int,traces))!=set(protocol['game_eval_seeds']):raise ValueError('Missing evaluation games')
                arm_data[seed]={g:[{k:s[k] for k in KEYS} for s in steps] for g,steps in traces.items()}
        data['arms'][arm]=arm_data
    return json.dumps(data,sort_keys=True,ensure_ascii=False,separators=(',',':')).encode()


def run(args):
    folder=args.folder
    files={'clock_replay.py':Path(__file__).with_name('clock_replay.py').read_bytes(),
           'agent/__init__.py':b'',**{'agent/'+n:(ROOT/'agent'/n).read_bytes() for n in ('m9_env.py','rl_env.py')}}
    packed=make_pack(folder,args.group)
    source_sha=hashlib.sha256(packed).hexdigest()
    build=hashlib.sha256(b''.join(files.values())).hexdigest()
    target='/home/researcher/mea-offline-single-pulse/bd-22554-fact48-cpu/'+args.group+'-'+source_sha[:10]+'-'+build[:10]
    output=folder/('clock-'+args.group+'.json')
    with paramiko.SSHClient() as c:
        c.load_system_host_keys()
        c.connect(HOST,username='researcher',password=os.environ['MEA_SSH_'+HOST.replace('.','_')],timeout=10)
        with c.open_sftp() as s:
            command(c,"import pathlib,sys;p=pathlib.Path(sys.argv[1]);(p/'agent').mkdir(parents=True,exist_ok=True)",[target])
            for n,contents in files.items():put(s,target+'/'+n,contents)
            try:s.stat(target+'/pack.json')
            except FileNotFoundError:
                s.putfo(io.BytesIO(packed),target+'/input.tmp')
                s.rename(target+'/input.tmp',target+'/pack.json')
            launch="""import pathlib,sys,subprocess,json,hashlib
p=pathlib.Path(sys.argv[1])
if hashlib.sha256((p/'pack.json').read_bytes()).hexdigest()!=sys.argv[2]:raise ValueError('Input changed')
if not (p/'pid').exists():
 log=(p/'replay.log').open('wb')
 job=subprocess.Popen([sys.executable,str(p/'clock_replay.py'),'--input',str(p/'pack.json'),'--output',str(p/'result.json'),
                       '--project',sys.argv[3],'--godot',sys.argv[4]],cwd=p,stdin=subprocess.DEVNULL,stdout=log,stderr=log,start_new_session=True)
 (p/'pid').write_text(str(job.pid))
print((p/'pid').read_text())
"""
            command(c,launch,[target,source_sha,SERVERS[HOST]['project'],SERVERS[HOST]['godot']])
            status="""import pathlib,sys
p=pathlib.Path(sys.argv[1]);proc=pathlib.Path('/proc')/(p/'pid').read_text()/'stat'
print('ready' if (p/'result.json').exists() else 'running' if proc.exists() and proc.read_text().split()[2]!='Z' else 'failed')
"""
            while True:
                state=command(c,status,[target])
                if state=='ready':break
                if state=='failed':
                    with s.open(target+'/replay.log','rb') as f:raise RuntimeError(f.read().decode(errors='replace')[-2400:])
                time.sleep(4)
            tmp=output.with_suffix('.partial')
            s.get(target+'/result.json',str(tmp))
            result=json.loads(tmp.read_bytes())
            if result['source_sha256']!=source_sha:raise ValueError('Result source mismatch')
            tmp.replace(output)
            facts=dict(source_sha=source_sha,build_sha=build,file_sha=hashlib.sha256(output.read_bytes()).hexdigest(),
                       games=sum(len(games) for records in result['arms'].values() for games in records.values()))
            (folder/('clock-'+args.group+'-audit.json')).write_text(json.dumps(facts,indent=2),encoding='utf8')
            print(json.dumps(facts))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('folder',type=Path)
    p.add_argument('group',choices=GROUPS)
    run(p.parse_args())
