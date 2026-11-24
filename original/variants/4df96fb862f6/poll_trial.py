"""Read-only progress retrieval. Credentials come from this process environment."""
import hashlib
import json
import os
from pathlib import Path
import sys
sys.stdout.reconfigure(encoding='utf-8')
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parents[1]/'.tools/mea_ssh_deps'))
import paramiko

class PinnedHost(paramiko.MissingHostKeyPolicy):
    def missing_host_key(self,client,hostname,key):
        if hashlib.sha256(key.asbytes()).hexdigest()!='286d75e321b511dff43cc6c35404d56cabc6fc6110b13816cc848ecfd6c5cddb':
            raise paramiko.SSHException('Unexpected host key')

def main():
    remote='/home/researcher/mea_photo_22021_20261117'
    c=paramiko.SSHClient();c.set_missing_host_key_policy(PinnedHost())
    try:
        c.connect('192.0.2.101',username='researcher',password=os.environ['MEA_TRIAL_SSH_PASSWORD'],timeout=8,look_for_keys=False,allow_agent=False)
        with c.open_sftp() as s:
            for name in ['progress.json','trial.log','connected_baseline_qc.json','stimulus_send_claim.json']:
                try:
                    s.stat(remote+'/'+name)
                    temp=ROOT/(name+'.download')
                    s.get(remote+'/'+name,str(temp));temp.replace(ROOT/name)
                except FileNotFoundError:pass
            p=json.loads((ROOT/'progress.json').read_text())
            for key in ['baseline_file','main_file']:
                if key in p:
                    try:p[key+'_bytes']=s.stat(p[key]).st_size
                    except FileNotFoundError:pass
            if (ROOT/'connected_baseline_qc.json').exists() and (ROOT/'connected_baseline_qc.json').stat().st_size:
                q=json.loads((ROOT/'connected_baseline_qc.json').read_text())
                p['qc_summary']={k:q[k] for k in ['status','errors','active_independent_channels','minute_rates_hz','sampled_independent_max_rail_fraction','sampled_selected_rail_fraction','events']}
            print(json.dumps(p,ensure_ascii=False,indent=2))
            print((ROOT/'trial.log').read_text(encoding='utf-8')[-1800:])
    finally:c.close()

if __name__=='__main__':main()
