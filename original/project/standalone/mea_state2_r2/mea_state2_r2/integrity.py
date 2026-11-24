"""External-asset pins and independently reconstructed summary acceptance."""
import hashlib
import json
import math
from pathlib import Path
from agent.mea_long_ablation import ORIGINAL_TENSORS
from .features import digest


def file_sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(4*1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def verify_assets(root):
    path=Path(root)/'external_assets.json'
    spec=json.loads(path.read_text())
    if spec.get('schema')!='mea.external.assets.v1' or not spec.get('files') or not spec.get('roots'):
        raise ValueError('Pinned external asset manifest required')
    for name,expected in spec['files'].items():
        alias,relative=name.split('/',1)
        base=Path(spec['roots'][alias]).resolve()
        actual=(base/relative).resolve()
        if (not actual.is_relative_to(base) or not actual.is_file() or actual.stat().st_size!=expected['size']
                or file_sha(actual)!=expected['sha256']):
            raise ValueError('External asset differs: '+name)
    return dict(passed=True,external_assets_sha256=file_sha(path),files_verified=len(spec['files']))


def check_baseline_identity(actual,expanded_critic_hash):
    expected=dict(ORIGINAL_TENSORS)
    compared=dict(actual,critic=expanded_critic_hash)
    if compared!=expected:raise ValueError('Original pure-R1 baseline including critic differs')


def baseline_padded_critic_hash(policy):
    import torch
    from tools.mea_offline import _tensor_hash
    entries=[]
    for name,parameter in policy.critic.named_parameters():
        value=parameter.detach()
        if name=='0.weight':
            if tuple(value.shape)!=(128,108):raise ValueError('Original108D critic required')
            value=torch.cat((value,torch.zeros((128,4),dtype=value.dtype,device=value.device)),dim=1)
        entries.append((name,value))
    return _tensor_hash(entries)


def check_round_summary(report,rows,actual_steps):
    expected=dict(decisions=len(rows),positive=sum(r['reward_record']['r2_raw']==1 for r in rows),
        negative=sum(r['reward_record']['r2_raw']==-1 for r in rows),terminals=sum(r['terminal'] for r in rows),
        r1_sum=math.fsum(r['r1_score']['reward'] for r in rows),
        r2_contribution_sum=math.fsum(r['reward_record']['r2_contribution'] for r in rows),applied_steps=actual_steps)
    for key,value in expected.items():
        if report.get(key)!=value:raise ValueError('Round summary differs from replay: '+key)
    return expected


def check_mirrors(training_result,reports,pre_metrics,verified_pre_metrics):
    if training_result.get('passed') is not True or training_result.get('rounds')!=reports:
        raise ValueError('Training result mirror differs from verified rounds')
    if pre_metrics!=verified_pre_metrics:raise ValueError('Pre-training metrics mirror differs from replay')
