"""Pre-registration native baseline smoke: real graph, zero optimizer updates."""
import argparse,json,sys
from pathlib import Path
from types import SimpleNamespace
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from tools.benchmark_m8s3 import setup,make,controlled_states,selected_logps,write,emit,critic_digest
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--config',required=True);ap.add_argument('--output',required=True);a=ap.parse_args()
    c=SimpleNamespace(**json.loads(Path(a.config).read_text()));setup();import torch
    p=make(c,'baseline');prepared=p.prepare(controlled_states(c)[:4]);verbs=[0]*4;targets=[0]*4
    before=p.digest();lp=selected_logps(p,prepared,verbs,targets)
    small=selected_logps(p,prepared,verbs,targets,size=1)
    vd,td=p.distributions(prepared,verbs);v=torch.tensor(verbs,device='cuda');t=torch.tensor(targets,device='cuda')
    current=vd.log_prob(v)+td.log_prob(t);error=max(abs(x-y) for x,y in zip(lp,current.detach().tolist()))
    (-current.mean()).backward();norm=float(torch.nn.utils.clip_grad_norm_([q for q in p.model.parameters() if q.requires_grad],1.))
    del vd,td,current
    p.model.zero_grad(set_to_none=True)
    many=p.prepare(controlled_states(c));vs=[];ts=[]
    for i,row in enumerate(many):
        legal=[v for v,b in enumerate(row['branches']) if b];v=legal[i%len(legal)];vs.append(v);ts.append(i%len(row['branches'][v]))
    expected=selected_logps(p,many,vs,ts);actual=selected_logps(p,many,vs,ts,grad=True)
    mixed_error=max(abs(x-y) for x,y in zip(expected,actual))
    # Exercise real multi-token navigation autograd and independently score its
    # complete prefixes without any cache, to detect padding/KV position errors.
    nav=next(row for row in many if len(row['branches'][1])>1)
    vd,td=p.distributions([nav],[1]);(-vd.logits[0,1]-td.logits[0,0]).backward()
    nav_norm=float(torch.nn.utils.clip_grad_norm_([q for q in p.model.parameters() if q.requires_grad],1.))
    cached=td.probs[0,:len(nav['branches'][1])].detach();del vd,td
    p.model.zero_grad(set_to_none=True)
    from agent.m8s3_diag_contract import physical_batches
    decisions={};nodes,paths=nav['semantic_trees'][1]
    with torch.no_grad():
        p._prepare_execution(False)
        for prefix,token_ids in nodes.items():
            b=next(physical_batches([nav['input_ids']+list(prefix)],p.max_tokens,p.tokenizer.pad_token_id))
            ids=torch.tensor(b['input_ids'],device='cuda');mask=torch.tensor(b['attention_mask'],device='cuda')
            pos=(mask.cumsum(-1)-1).clamp_min(0).unsqueeze(0).expand(3,-1,-1)
            output=p.model.get_base_model().model(input_ids=ids,attention_mask=mask,position_ids=pos,use_cache=False,return_dict=True)
            logits=torch.nn.functional.linear(output.last_hidden_state[0,-1,:].float(),p.model.get_output_embeddings().weight[token_ids].float())
            decisions[prefix]=logits.log_softmax(-1)
        complete=torch.stack([sum((decisions[k][i] for k,i in path),torch.zeros((),device='cuda')) for path in paths]).softmax(-1)
        cache_full_error=float((cached-complete).abs().max())
    result=dict(navigation_grad_norm=nav_norm,cached_vs_complete_prefix_probability_max=cache_full_error,
                mixed_32_grad_error=mixed_error,reorder_max=max(abs(x-y) for x,y in zip(lp,small)),grad_error=error,grad_norm=norm,finite=bool(torch.isfinite(torch.tensor([norm,nav_norm])).all()),
                actor_unchanged=before==p.digest(),critic_sha256=critic_digest(p),optimizer_steps=0,formal_training_started=False,
                peak_allocated_bytes=torch.cuda.max_memory_allocated(),metadata=p.metadata())
    write(a.output,result);emit(**result)
    assert result['reorder_max']<=1e-6 and error<=1e-6 and mixed_error<=1e-6 and result['finite'] and result['actor_unchanged']
if __name__=='__main__':main()
