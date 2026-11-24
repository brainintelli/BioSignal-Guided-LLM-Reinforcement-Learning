"""Real-checkpoint smoke regression: cache modes, grouping and branch gradients."""
import importlib.util,json,sys,time,warnings
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
assert importlib.util.find_spec('agent.m8s2_policy') is not None,'Batched semantic policy missing'
import torch
from agent.m8s2_policy import BatchedSemanticPolicy

def main():
    warnings.filterwarnings('ignore',message='Input tensor shape suggests potential format mismatch:.*')
    torch.set_num_threads(8);root=Path('/home/researcher/meadowmind-gpu');reg=json.loads((root/'m8s-runs/semantic_diagnostic1/registration.json').read_text());c=reg['config']
    p=BatchedSemanticPolicy(c['model'],checkpoint=c['initial_checkpoint'],parent_checkpoint=c['parent_checkpoint'],with_reference=True,seed=c['rng_seed'])
    initial=p.digest();data=json.loads(Path(c['dataset']).read_text())['observations'];rows=p.prepare([x['observation'] for x in data[:4]])
    with torch.no_grad(),p._scope(rows,False):
        p.verb_distribution(rows);before=p.cache_stats['forward_calls']
        directions=p.target_distribution(rows,[0]*4)
        assert p.cache_stats['forward_calls']==before,'Movement target scoring must reuse state features without another backbone call'
        assert torch.allclose(directions.probs.sum(-1),torch.ones(4,device='cuda'))
    def evaluate(rows):
        v,t=p.distributions(rows,[1]*len(rows));return torch.cat([v.probs,t.probs],dim=-1)
    with torch.no_grad():
        warm=evaluate(rows);start=time.perf_counter()
        a=evaluate(rows);torch.cuda.synchronize();seconds=time.perf_counter()-start
        reverse=evaluate(rows[::-1]).flip(0)
        p.clear_prefix_cache();cold=evaluate(rows)
        single=torch.cat([evaluate([r]) for r in rows])
    b=evaluate(rows);loss=-b[:,8].clamp_min(1e-8).log().mean();loss.backward()
    grads=[x.grad for x in p.model.parameters() if x.requires_grad and x.grad is not None]
    result={'reorder_error':float((a-reverse).abs().max()),'prefix_cold_error':float((a-cold).abs().max()),'single_error':float((a-single).abs().max()),
        'grad_mode_error':float((a-b.detach()).abs().max()),'finite_gradients':all(bool(x.isfinite().all()) for x in grads),
        'gradient_norm':float(torch.nn.utils.clip_grad_norm_(p.model.parameters(),1.)), 'weights_unchanged':p.digest()==initial,
        'forward_four_states_seconds':seconds,'input_lengths':[len(x['input_ids']) for x in rows],'counters':p.cache_stats,
        'logps':a.tolist()}
    out=Path(sys.argv[1]);out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k!='logps'}),flush=True)
    assert max(result[k] for k in ('reorder_error','prefix_cold_error','single_error','grad_mode_error'))<=1e-6
    assert result['finite_gradients'] and result['gradient_norm']>0 and result['weights_unchanged']
if __name__=='__main__':main()
