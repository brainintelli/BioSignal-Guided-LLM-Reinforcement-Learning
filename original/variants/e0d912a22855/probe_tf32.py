"""Small development probe; no optimizer steps or checkpoint writes."""
import gc
import json
from pathlib import Path
import sys
import time
sys.path.insert(0,'/home/researcher/meadowmind-gpu/m8b-project')
import torch
from agent.m8x_policy import Policy
from agent.m8x_fast_policy import FastPolicy
from agent.m8x_kv_policy import KVPolicy
from agent.m8x_causal_policy import CausalKVPolicy
from agent.m8x_merged_lora import enable_merged_lora
from agent.m8x_fused_norm import enable_fused_norm

torch.set_num_threads(8)
root=Path('/home/researcher/meadowmind-gpu')
c=json.loads((root/'m8b-project/configs/m8x_exploration_attempt1.json').read_text())
rows=[json.loads(x) for x in (root/'diagnostics/rl_speed_20261107_v2/baseline_fresh_transitions.jsonl').read_text().splitlines()][::12][:16]
cases=[]
for mode in ['ieee','tf32']:
    torch.backends.cuda.matmul.allow_tf32 = mode=='tf32'
    cls=CausalKVPolicy
    kw={'optimization':'compact'} if mode=='merged_compact' else {}
    p=cls(c['model'],checkpoint=str(root/'m8b-runs/exploration_attempt1/checkpoint_0032'),
          parent_checkpoint=c['parent_checkpoint'],with_reference=True,**kw)
    print(json.dumps(dict(event='merged_enabled',mode=mode,actor=enable_merged_lora(p.model),reference=enable_merged_lora(p.reference_model))),flush=True)
    print(json.dumps(dict(event='fused_norm',actor=enable_fused_norm(p.model),reference=enable_fused_norm(p.reference_model))),flush=True)
    prepared=p.prepare([r['observation'] for r in rows])
    v=torch.tensor([r['verb_index'] for r in rows],device='cuda')
    t=torch.tensor([r['target_index'] for r in rows],device='cuda')
    times=[];backward=[];errors=[];logps=[]
    for repeat in range(3):
        torch.cuda.synchronize();begin=time.perf_counter()
        with torch.no_grad():
            found=[]
            for start in range(0,len(rows),4):
                a,b=p.distributions(prepared[start:start+4],v[start:start+4].tolist())
                found.extend((a.log_prob(v[start:start+4])+b.log_prob(t[start:start+4])).tolist())
                p.distributions(prepared[start:start+4],v[start:start+4].tolist(),True)
        torch.cuda.synchronize();times.append(time.perf_counter()-begin)
        torch.cuda.synchronize();begin=time.perf_counter();largest=0
        for start in range(0,len(rows),4):
            p.model.zero_grad(set_to_none=True)
            a,b=p.distributions(prepared[start:start+4],v[start:start+4].tolist())
            logp=a.log_prob(v[start:start+4])+b.log_prob(t[start:start+4])
            largest=max(largest,max(abs(x-y) for x,y in zip(logp.detach().tolist(),found[start:start+4])))
            (-logp.mean()).backward()
            assert all(torch.isfinite(q.grad).all() for q in p.model.parameters() if q.grad is not None)
            del a,b,logp
        torch.cuda.synchronize();backward.append(time.perf_counter()-begin);errors.append(largest)
        logps=found
    case=dict(mode=mode,forward_actor_reference=times,forward_backward_actor=backward,
              gradient_mode_logp_error=errors,logp=logps,
              prefix_lengths=[len(r.get('kv_prefix_ids',[])) for r in prepared])
    cases.append(case)
    print(json.dumps(case),flush=True)
    Path('/home/researcher/meadowmind-gpu/diagnostics/rl_tf32_probe_20261107.json').write_text(json.dumps(cases,indent=2))
    del p,prepared;gc.collect();torch.cuda.empty_cache()
