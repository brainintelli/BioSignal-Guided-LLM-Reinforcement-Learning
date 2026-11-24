import json,sys,torch
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from agent.m8x_grouped_lora import GroupedPolicy
r=json.loads(Path('/home/researcher/meadowmind-gpu/m8s-runs/semantic_diagnostic1/registration.json').read_text());c=r['config'];torch.set_num_threads(8)
rows=[json.loads(x) for x in Path('/home/researcher/meadowmind-gpu/m8s-diagnostics/semantic_compare1/old_optimized_transitions.jsonl').read_text().splitlines()][:4]
p=GroupedPolicy(c['model'],checkpoint=c['initial_checkpoint'],parent_checkpoint=c['parent_checkpoint'],with_reference=True,seed=c['rng_seed'],registration=r)
results=[]
def check(label):
 q=p.prepare([x['observation'] for x in rows]);v=torch.tensor([x['verb_index'] for x in rows],device='cuda');t=torch.tensor([x['target_index'] for x in rows],device='cuda')
 same=[]
 for a,b in zip(q,rows):
  same.append({'verb':a['input_ids']==b['executed_verb_token_ids'],'target':p._target_tokens(a,b['verb_index'])==b['target_prompt_token_ids'] if b['target_prompt_token_ids'] else True})
 with torch.no_grad():
  vd,td=p.distributions(q,v.tolist());logp=(vd.log_prob(v)+td.log_prob(t)).tolist()
 result={'label':label,'digest':p.digest(),'same_inputs':same,'actual':logp,'expected':[x['log_probability'] for x in rows],'tf32':torch.backends.cuda.matmul.allow_tf32,'precision':torch.get_float32_matmul_precision(),'model_training':p.model.training}
 results.append(result);print(json.dumps(result),flush=True)
check('just_loaded');p.metadata();check('after_metadata')
from agent.m8_train import update
actor=torch.optim.AdamW([v for v in p.model.parameters() if v.requires_grad],lr=c['actor_lr'],eps=1e-5,weight_decay=0.)
critic=torch.optim.AdamW(p.critic.parameters(),lr=c['critic_lr'],eps=1e-5,weight_decay=0.)
torch.manual_seed(c['rng_seed']);check('after_optimizers_and_seed')
q=p.prepare([x['observation'] for x in rows]);sampled=p.act(q,with_reference=True)
print(json.dumps({'sampled':sampled['logp'],'actions':sampled['actions']}),flush=True);check('after_act')
Path('/home/researcher/meadowmind-gpu/m8s-diagnostics/semantic_compare1/reload_control.json').write_text(json.dumps(results,indent=2))
