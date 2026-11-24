"""One-time transparent fork of the accepted long trainer; original unchanged."""
from pathlib import Path
import hashlib
r=Path(__file__).resolve().parents[2]
p=r/'tools/mea_long_train.py';s=p.read_text(encoding='utf-8')
s=s.replace('Fixed long-recording offline training and independent replay; no device API.',
    'Fixed stimulus before/after short offline training and independent replay; no device API.')
s=s.replace('from agent.mea_h5_replay import H5MappedRecording','from agent.mea_stim_offline import StimRecording,PROTOCOL,cell_config\nfrom agent.mea_long_ablation import ORIGINAL_TENSORS')
lo=s.index('PROTOCOL=dict(');hi=s.index('CHECK_NAMES=',lo);s=s[:lo]+s[hi:]
s=s.replace("'code/tools/mea_long_train.py'","'code/tools/mea_stim_train.py'")
s=s.replace('from agent.mea_long_ablation import cell_config','from agent.mea_stim_offline import cell_config')
s=s.replace("),ablation['arm'])","),ablation['policy_arm'])")
s=s.replace("return columns_expected(ablation['arm'],actual)","return columns_expected(ablation['policy_arm'],actual)")
lo=s.index('def recording(');hi=s.index('\n\ndef column_check',lo)
s=s[:lo]+'''def recording(source,pin,ablation=None):
    if ablation is None:raise ValueError('Registered stimulus comparison cell required')
    _same(ablation,cell_config(ablation['block'],ablation['arm']),'registered cell')
    return StimRecording(source,pin=pin,phase='after' if ablation['arm']=='after' else 'before',split='train')

def validate_common_initial(actual):
    _same(actual,ORIGINAL_TENSORS,'independently pinned original tensors')
'''+s[hi:]
s=s.replace("initial=state(policy);write_json", "initial=state(policy);validate_common_initial(initial);write_json")
s=s.replace("original=state(policy)\n", "original=state(policy);validate_common_initial(original)\n")
s=s.replace('range(1,9)',"range(1,PROTOCOL['rounds']+1)")
s=s.replace('decisions=1024','decisions=384')
s=s.replace("stimulation=False,device_commands_sent=0", "stimulation_during_training=False,device_commands_sent=0")
s=s.replace("source=data.summary(),protocol=PROTOCOL,", "source=data.summary(),protocol=PROTOCOL,")
# This dedicated controller is only driven by the registered campaign module.
s=s[:s.index('\ndef main():')]
(r/'tools/mea_stim_train.py').write_text(s,encoding='utf-8')
(Path(__file__).parent/'trainer_parent_sha256.txt').write_text(hashlib.sha256(p.read_bytes()).hexdigest())
