"""Final stage of this one offline batch: transfer completed traces, audit, render, then exit."""
import hashlib,json,os,shlex,subprocess,sys,tarfile,time
from pathlib import Path
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R/'.tools/mea_ssh_deps'))
import paramiko
O=R/'output/mea_r2_beta_sweep_20261120';remote=O/'remote';cache=O/'audit_cache'
D='/home/researcher/mea-offline-single-pulse/bd-22554-r2beta-20261120'; E='/home/researcher/r2-evidence-20261120'
class Host:
    def __init__(self,host,secret):self.host,self.secret,self.c=host,secret,None
    def link(self):
        if self.c and self.c.get_transport() and self.c.get_transport().is_active():return self.c
        self.c=paramiko.SSHClient();self.c.load_host_keys(str(Path.home()/'.ssh/known_hosts'));self.c.set_missing_host_key_policy(paramiko.RejectPolicy())
        self.c.connect(self.host,username='researcher',password=self.secret,allow_agent=False,look_for_keys=False,timeout=12,banner_timeout=18,auth_timeout=20)
        self.c.get_transport().set_keepalive(25);return self.c
    def cmd(self,command):
        _,stdout,stderr=self.link().exec_command(command,timeout=120);data=stdout.read().decode();err=stderr.read().decode()
        if stdout.channel.recv_exit_status():raise RuntimeError(err[-900:])
        return data.strip()
    def get(self,src,dest):
        hash_expected=self.cmd('sha256sum '+shlex.quote(src)).split()[0];part=dest.with_suffix(dest.suffix+'.part')
        with self.link().open_sftp() as conn:conn.get(src,str(part))
        if hashlib.sha256(part.read_bytes()).hexdigest()!=hash_expected:raise ValueError('archive transfer checksum mismatch')
        part.replace(dest)
    def close(self):
        if self.c is not None:self.c.close();self.c=None
def atomic(p,obj):
    q=p.with_suffix('.tmp');q.write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf8');q.replace(p)
def extract(archive,arm):
    with tarfile.open(archive) as t:
        for f in t.getmembers():
            name=f.name
            if not f.isfile() or not (name.startswith('22554/beta_010/'+arm+'/') or name=='22554/beta_010/budget-beta_010.json') or not (remote/name).resolve().is_relative_to(remote.resolve()):raise ValueError('unexpected member in training evidence')
        t.extractall(remote,filter='data')
def batch():
    import msvcrt
    lock=(O/'batch-finalizer.lock').open('a+b');lock.seek(0)
    try:msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
    except OSError:raise SystemExit('This batch already has a local finalizer.')
    host=Host('192.0.2.16',os.environ.pop('MEA_SWEEP16_SECRET'));probhost=Host('192.0.2.23',os.environ.pop('MEA_SWEEP23_SECRET',''))
    start=time.time();status=O/'job_state.json'; first=True; failures=0
    probe="""from pathlib import Path as P
import json
r=P(%r);a={}
for name in ('ordered','shuffled'):
 d=r/'beta_010'/name;tr=d/'training';rc=json.loads((d/'exit.json').read_bytes())['exit_code'] if (d/'exit.json').exists() else None
 a[name]={'seeds':len(list(tr.glob('[0-9]*/result.json'))),'exit_code':rc,'aggregate':(tr/'result.json').is_file()}
print(json.dumps(a))"""%D
    while time.time()-start<6*3600:
        try:
            stat=json.loads(host.cmd('python3 -c '+shlex.quote(probe)))
            if any(v['exit_code'] not in (None,0) for v in stat.values()):raise ValueError('Remote training returned a nonzero exit: '+str(stat))
            changed=False
            for arm,v in stat.items():
                localroot=remote/'22554/beta_010'/arm;prev=len(list((localroot/'training').glob('[0-9]*/result.json')))
                localdone=(localroot/'exit.json').is_file() and (localroot/'training/result.json').is_file()
                if (v['seeds']//4>prev//4 or (v['exit_code']==0 and not localdone)):
                    name='evidence22554_010_'+arm+'.tgz';host.cmd('python3 /home/researcher/package_r2_evidence_20261120.py 22554 010 '+arm+' beta_010')
                    host.get(E+'/'+name,remote/name);extract(remote/name,arm);changed=True
                if v['exit_code']==0 and (localroot/'training/result.json').is_file():
                    ag=json.loads((localroot/'training/result.json').read_bytes());expected=list(range(2255401,2255417))
                    if ag['arm']!=arm or ag['r2_beta']!=.10 or [x['training_seed'] for x in ag['seeds']]!=expected:raise ValueError('final seed/beta registry mismatch')
            pp=cache/'22021-distributions.json.gz'
            if not pp.exists() and probhost.cmd('if test -f '+E+'/22021-distributions.json.gz; then echo ready; fi')=='ready':
                probhost.get(E+'/22021-distributions.json.gz',pp);changed=True
            if changed or first:
                for task in ['audit_r2_beta_clock.py','report_r2_beta_sweep.py']:
                    subprocess.run([sys.executable,'-X','utf8','-u',str(R/'tools'/task)],cwd=R,check=True)
                first=False
            ready=all(v['exit_code']==0 and v['aggregate'] and v['seeds']==16 for v in stat.values()) and pp.exists()
            if ready:
                import csv
                with (O/'coverage.csv').open(encoding='utf-8-sig') as f:rows=list(csv.DictReader(f))
                if len(rows)!=16 or any(int(r['complete_seeds'])!=16 for r in rows):raise ValueError('Incomplete analysis grid')
                with (O/'action_probabilities.csv').open(encoding='utf-8-sig') as f:results=list(csv.DictReader(f))
                if len([x for x in results if x['kind']=='策略概率'])!=128:raise ValueError('Frozen policy probabilities missing')
            record={'state':'complete' if ready else 'running','asof_unix':time.time(),'beta_010_22554':stat,'native_22021_probs_downloaded':pp.exists(),'report':str(O/'report.md')}
            atomic(status,record);failures=0
            if ready:print('Batch complete: 256 models, 12,288 games.',flush=True);return
        except (OSError,EOFError,paramiko.SSHException) as e:
            failures+=1;host.close();probhost.close();atomic(status,{'state':'retrying_transfer','asof_unix':time.time(),'error_type':type(e).__name__})
            if failures>=12:raise
        time.sleep(36)
    raise TimeoutError('Batch did not finish before the six-hour observation limit')
if __name__=='__main__':
    try:batch()
    except Exception as e:
        atomic(O/'job_state.json',{'state':'error','asof_unix':time.time(),'error':str(e)[:700]});raise
