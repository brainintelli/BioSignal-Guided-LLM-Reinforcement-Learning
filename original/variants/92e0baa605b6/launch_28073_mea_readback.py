import sys,os,json,getpass,re
from pathlib import Path
root=Path.cwd()
out=root/'output/pair28073-20261118T070238Z'
runtime=out/'mea-readback-runtime'
sys.path[:0]=[str(root/'.tools/mea_ssh_deps'),str(root/'.tools/mea_baseline_deps')]+[str(runtime/s) for s in ('','mea_online','mea_binary_r2','mea_state2_r2','mea_stimulation_v2')]
import control
def password(prompt):
    address=prompt.split()[0]
    if address not in ('192.0.2.21','192.0.2.16','192.0.2.23'):
        raise ValueError('Unexpected credential destination')
    print(json.dumps({'credential_request':address}),flush=True)
    value=json.loads(sys.stdin.readline())
    if value['address']!=address:raise ValueError('Mismatched credential response')
    return value['password']
getpass.getpass=password
b=json.loads((out/'device-readback-binding.json').read_text(encoding='utf-8-sig'))
servers=json.loads((root/'output/mea_spatial_gate_20261118/servers.json').read_text(encoding='utf-8-sig'))
control.run(b,servers,out/'readback-run',lambda f:control.present_in_chat(f,control.binding(b)))
