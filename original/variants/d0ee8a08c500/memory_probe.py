"""Technical backward/update probe on retained failed rollout; no efficacy evaluation."""
import json,sys,time
from pathlib import Path
from types import SimpleNamespace
import torch
r=Path(__file__).resolve().parent;sys.path.insert(0,str(r/'project'))
from tools import mea_shuffle_train as train
from agent.mea_shuffle import cell_config
from agent.mea_replay import RecordingSnapshot
from tools.mea_offline import evidence
from tools.mea_p0 import read_json,canonical
started=time.monotonic();policy=train.new_policy(r/'model_config.json',cell_config(3,'shuffled_after'))
initial=train.state(policy);train.validate_common_initial(initial)
source=Path('/home/researcher/mea-28080-shuffle-20261117/experiment01/b03_shuffled_after/train/round_01/rollout.jsonl')
rows=[json.loads(x) for x in source.read_text().splitlines()]
if len(rows)!=128:raise ValueError('Retained rollout budget')
prepared=policy.prepare([row['observation'] for row in rows],[RecordingSnapshot.from_dict(row['snapshot']) for row in rows])
for row,p in zip(rows,prepared):
    if evidence(p)!=row['prepared']:raise ValueError('Exact input replay differs')
    row['prepared']=p
error=train.validate_policy_statistics(train.stats(policy,rows),rows)
opts={name:torch.optim.AdamW(parameters,lr=lr,eps=1e-5,weight_decay=0) for name,parameters,lr in [
    ('actor',[p for p in policy.model.parameters() if p.requires_grad],train.CONFIG['actor_lr']),('critic',policy.critic.parameters(),train.CONFIG['critic_lr'])]}
torch.manual_seed(cell_config(3,'shuffled_after')['ppo_seed']+1)
metrics=train.training_update(policy,opts['actor'],opts['critic'],rows,SimpleNamespace(**train.CONFIG))
counts=train.steps(opts);after=train.state(policy)
if set(counts['actor'])!={2} or set(counts['critic'])!={16}:raise ValueError('Actual update budget')
if after['base']!=initial['base'] or after['reference']!=initial['reference'] or after['actor']==initial['actor'] or after['critic']==initial['critic']:raise ValueError('Tensor update identity')
receipt=dict(passed=True,technical_update_only=True,efficacy_evaluated=False,weights_reused_in_campaign=False,
    initial=initial,pre_update_max_error=error,actor_steps=2,critic_steps=16,r2_enabled=False,
    peak_cuda_allocated=torch.cuda.max_memory_allocated(),seconds=time.monotonic()-started)
(r/'memory_probe_receipt.json').write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt),flush=True)
