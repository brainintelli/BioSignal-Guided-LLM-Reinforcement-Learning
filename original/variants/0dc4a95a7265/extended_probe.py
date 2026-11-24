"""Supplemental GPU contract audit, no optimizer updates."""
import sys,json,copy,time
from pathlib import Path
from types import SimpleNamespace
sys.path.insert(0,'/home/researcher/meadowmind-gpu/m8s3-batch-project')
from tools.benchmark_m8s3 import setup,selected_logps,write
from tools.benchmark_m8s3_batch import make,verify
from agent.m8s3_batch_layout import canonical_batches
setup()
import torch
reg=json.loads(Path('/home/researcher/meadowmind-gpu/m8s3-runs/survival_batch_diagnostic1/registration.json').read_text());verify(reg)
c=SimpleNamespace(**reg['config']);p=make(c,'optimized');initial=p.digest()
data=json.loads(Path('/home/researcher/meadowmind-gpu/m8s3-batch-diagnostics/equal_update1/controlled_states.json').read_text())
rows=p.prepare(data[:4]);verbs=[0,1,0,1]
for i,v in enumerate(verbs):
 if len(rows[i]['branches'][v])<2:verbs[i]=0
params=[q for q in p.model.parameters() if q.requires_grad]
def grads(batch,vs,weight):
 v,t=p.distributions(batch,vs)
 loss=-(v.log_prob(torch.tensor(vs,device='cuda'))+t.logits[:,0]).mean()*weight
 loss.backward()
p.model.zero_grad(set_to_none=True);grads(rows,verbs,1.)
batched=[q.grad.detach().float().cpu().clone() if q.grad is not None else torch.zeros_like(q,device='cpu',dtype=torch.float32) for q in params]
p.model.zero_grad(set_to_none=True)
for row,v in zip(rows,verbs):grads([row],[v],.25)
serial=[q.grad.detach().float().cpu() if q.grad is not None else torch.zeros_like(q,device='cpu',dtype=torch.float32) for q in params]
a=torch.cat([v.flatten() for v in batched]);b=torch.cat([v.flatten() for v in serial])
gradient=dict(cosine=float(torch.nn.functional.cosine_similarity(a,b,dim=0)),relative_l2=float((a-b).norm()/b.norm()),finite=bool(torch.isfinite(a).all() and torch.isfinite(b).all()))
assert gradient['finite'] and gradient['cosine']>.999 and gradient['relative_l2']<.03,gradient
del a,b,batched,serial
p.model.zero_grad(set_to_none=True)
nav=next(row for row in p.prepare(data) if len(row['branches'][1])>1)
with torch.no_grad():
 v,t=p.distributions([nav],[1]);cached=t.probs[0,:len(nav['branches'][1])].clone()
 decisions={};nodes,paths=nav['semantic_trees'][1]
 for prefix,tokens in nodes.items():
  seq=nav['input_ids']+list(prefix)
  batch=next(canonical_batches([seq],p.bucket_boundaries,pad_id=p.tokenizer.pad_token_id))
  ids=torch.tensor(batch['input_ids'],device='cuda');mask=torch.tensor(batch['attention_mask'],device='cuda')
  pos=(mask.cumsum(-1)-1).clamp_min(0).unsqueeze(0).expand(3,-1,-1)
  out=p.model.get_base_model().model(input_ids=ids,attention_mask=mask,position_ids=pos,use_cache=False,return_dict=True)
  logits=torch.nn.functional.linear(out.last_hidden_state[0,-1].float(),p.model.get_output_embeddings().weight[tokens].float())
  decisions[prefix]=logits.log_softmax(-1)
 full=torch.stack([sum((decisions[k][i] for k,i in path),torch.zeros((),device='cuda')) for path in paths]).softmax(-1)
 nav_error=float((full-cached).abs().max())
# Synthetic token contexts explicitly test both long buckets; not sampled gameplay.
synthetic=[]
for length in (895,896,897,1279,1280,1281,1500):
 row=copy.deepcopy(rows[0]);row['input_ids']=(row['input_ids']+[row['input_ids'][-1]]*length)[:length];synthetic.append(row)
vs=[0]*len(synthetic);ts=[0]*len(synthetic)
old=selected_logps(p,synthetic,vs,ts);other=selected_logps(p,synthetic,vs,ts,size=1,order=list(reversed(range(len(vs)))))
long_error=max(abs(x-y) for x,y in zip(old,other));assert long_error<=1e-6
verify(reg)
result=dict(registration_sha256=reg['registration_sha256'],gradient=gradient,navigation_cached_full_probability_max=nav_error,synthetic_lengths=[len(r['input_ids']) for r in synthetic],synthetic_reorder_logp_max=long_error,actor_unchanged=initial==p.digest(),optimizer_steps=0,formal_training_started=False)
write('/home/researcher/meadowmind-gpu/m8s3-batch-diagnostics/extended_probe.json',result);print(json.dumps(result),flush=True)
