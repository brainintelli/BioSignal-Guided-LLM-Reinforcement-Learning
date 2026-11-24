"""Bounded semantic probability/gradient probe. Never writes model weights."""
import json,math,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from agent.m8s_policy import SemanticPolicy

root=Path('/home/researcher/meadowmind-gpu')
output=Path(sys.argv[1]);output.mkdir(parents=True,exist_ok=False)
torch.set_num_threads(8)
records=[json.loads(x) for x in (root/'m8b-project/output/rl_kv2_20261107/optimized_transitions.jsonl').read_text().splitlines()]
p=SemanticPolicy(str(root/'models/Qwen3.5-0.8B'),checkpoint=str(root/'m8b-runs/exploration_attempt1/checkpoint_0032'),
                 parent_checkpoint=str(root/'m6-runs/small_v2_stable/checkpoint_0016'),with_reference=True)
initial=p.digest();prepared=p.prepare([r['observation'] for r in records[:4]])
print(json.dumps({'event':'loaded','digest':initial,'lengths':[len(r['input_ids']) for r in prepared]}),flush=True)
with torch.no_grad():
    sampled=p.act(prepared,with_reference=True)
    vd,td=p.distributions(prepared,sampled['verbs'])
    v=torch.tensor(sampled['verbs'],device='cuda');t=torch.tensor(sampled['targets'],device='cuda')
    recompute=(vd.log_prob(v)+td.log_prob(t)).tolist()
    initial_error=max(abs(a-b) for a,b in zip(recompute,sampled['logp']))
    masses={}
    for verb in range(8):
        active=[r for r in prepared if r['branches'][verb]]
        if active:
            q=p.target_distribution(active,[verb]*len(active))
            masses[verb]=q.probs.sum(-1).tolist()
    # Force navigate in the backward probe even if the policy sampled a singleton verb.
    q=p.target_distribution([prepared[0]],[1]);old=q.logits.detach().clone()
q=p.target_distribution([prepared[0]],[1])
grad_mode_error=float((q.logits-old).abs().max())
loss=-q.log_prob(torch.tensor([0],device='cuda')).mean()
loss.backward()
norm=float(torch.nn.utils.clip_grad_norm_([v for v in p.model.parameters() if v.requires_grad],1.))
p.model.zero_grad(set_to_none=True)
p.assert_reference_frozen()
result={'initial_error':initial_error,'grad_mode_error':grad_mode_error,'gradient_norm':norm,
        'target_mass':masses,'sampled':sampled,'weights_unchanged':p.digest()==initial,
        'kv_scope_released':p._kv_scope is None,'semantic_node_calls':p.semantic_node_calls,
        'checkpoint_writes':0,'formal_training_started':False,'metadata':p.metadata()}
(output/'result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
print(json.dumps({k:v for k,v in result.items() if k not in ('metadata','sampled')},ensure_ascii=False),flush=True)
assert initial_error<=1e-6 and grad_mode_error<=1e-6 and math.isfinite(norm) and norm>0
assert result['weights_unchanged'] and result['kv_scope_released']
