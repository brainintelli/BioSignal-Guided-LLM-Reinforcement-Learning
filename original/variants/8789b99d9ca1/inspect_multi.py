import hashlib,json,sys
from pathlib import Path
import paramiko
root=Path(__file__).resolve().parent;remote='/home/researcher/mea_full_28080_20261117_1406'
c=paramiko.SSHClient();c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect('192.0.2.101',username='researcher',password=sys.stdin.readline().strip(),timeout=10,look_for_keys=False,allow_agent=False)
assert hashlib.sha256(c.get_transport().get_remote_server_key().asbytes()).hexdigest()=='286d75e321b511dff43cc6c35404d56cabc6fc6110b13816cc848ecfd6c5cddb'
s=c.open_sftp()
for folder in ['multi_software_prepare','multi_software_prepare_v2']:
    out=root/folder;out.mkdir(exist_ok=True)
    for name in ['api_audit.jsonl','failure_cleanup.json','preparation.json','progress.json']:
        try:s.get(remote+'/'+folder+'/'+name,str(out/name))
        except FileNotFoundError:continue
    records=[json.loads(x) for x in (out/'api_audit.jsonl').read_text().splitlines()]
    print(json.dumps({'folder':folder,'requests':sum('mono' in x for x in records),
        'sequence_tail':[x for x in records if x.get('command','').startswith(('sequence_new','sequence_append','sequence_delete','mea_array_delete'))][-10:]},indent=2))
s.close();c.close()
