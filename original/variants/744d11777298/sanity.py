import json, pathlib, time, types
import torch
from agent.m8x_policy import Policy
from agent.m8x_fast_policy import FastPolicy
torch.set_num_threads(8)
root=pathlib.Path('/home/researcher/meadowmind-gpu')
c=json.loads((root/'m8b-project/configs/m8x_exploration_attempt1.json').read_text())
p=FastPolicy(c['model'],checkpoint=str(root/'m8b-runs/exploration_attempt1/checkpoint_0032'),parent_checkpoint=c['parent_checkpoint'],optimization='exact')
rows=[json.loads(x) for x in pathlib.Path('/tmp/meadowmind_rl_profile_20261107/fresh_transitions.jsonl').read_text().splitlines()][:4]
prepared=p.prepare([r['observation'] for r in rows]);verbs=[r['verb_index'] for r in rows];targets=torch.tensor([r['target_index'] for r in rows],device='cuda');v=torch.tensor(verbs,device='cuda')
fast=p._logits
result={'prefix_tokens':len(p.prefix_ids),'cases':[]}
base=None
for mode in ['baseline','exact','compact','prefix']:
 p._logits=types.MethodType(Policy._logits,p) if mode=='baseline' else fast
 p.optimization=mode
 elapsed=[]
 for k in range(3):
  torch.cuda.synchronize();begin=time.perf_counter()
  with torch.no_grad():vd,td=p.distributions(prepared,verbs);lp=vd.log_prob(v)+td.log_prob(targets)
  torch.cuda.synchronize();elapsed.append(time.perf_counter()-begin)
 if base is None:base=(vd.probs.clone(),td.probs.clone())
 with torch.enable_grad():
  pv,pt=p.distributions(prepared,verbs);newlp=pv.log_prob(v)+pt.log_prob(targets)
  drift=float((newlp-lp).abs().max());(-newlp.mean()).backward()
 grads=[x.grad for x in p.model.parameters() if x.requires_grad and x.grad is not None]
 case=dict(mode=mode,seconds=elapsed,zero_update_max=drift,verb_probability_difference=float((vd.probs-base[0]).abs().max()),target_probability_difference=float((td.probs-base[1]).abs().max()),finite_gradients=all(bool(torch.isfinite(g).all()) for g in grads),gradient_norm=float(torch.nn.utils.clip_grad_norm_([x for x in p.model.parameters() if x.requires_grad],1.)))
 p.model.zero_grad(set_to_none=True);result['cases'].append(case);print(json.dumps(case),flush=True)
pathlib.Path('/tmp/m8x_speed_sanity.json').write_text(json.dumps(result,indent=2))
