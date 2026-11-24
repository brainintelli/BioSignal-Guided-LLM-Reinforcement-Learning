"""Capture layer-boundary evidence from independently loaded diagnostic models."""
import argparse,gc,hashlib,json,sys,warnings
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from agent.m8x_grouped_lora import GroupedPolicy
from agent.m8s_policy import SemanticPolicy

def main():
    a=argparse.ArgumentParser();a.add_argument('--output',required=True);a.add_argument('--mode',default='old_optimized');a.add_argument('--stable',action='store_true')
    a.add_argument('--warm-baseline',action='store_true')
    args=a.parse_args();torch.set_num_threads(8)
    warnings.filterwarnings('ignore',message='Input tensor shape suggests potential format mismatch:.*')
    root=Path('/home/researcher/meadowmind-gpu');reg=json.loads((root/'m8s-runs/semantic_diagnostic1/registration.json').read_text());c=reg['config']
    if args.stable:
        from agent.m8s2_kernels import pin_kernels
        pin_kernels()
    if args.warm_baseline:
        from agent.m8x_policy import Policy
        warm=Policy(c['model'],checkpoint=c['initial_checkpoint'],parent_checkpoint=c['parent_checkpoint'],with_reference=False,seed=c['rng_seed'])
        raw=json.loads(Path(c['dataset']).read_text())['observations'][:4]
        with torch.no_grad():warm.verb_distribution(warm.prepare([r['observation'] for r in raw]))
        del warm;gc.collect();torch.cuda.empty_cache()
    cls=SemanticPolicy if args.mode=='semantic' else GroupedPolicy
    p=cls(c['model'],checkpoint=c['initial_checkpoint'],parent_checkpoint=c['parent_checkpoint'],with_reference=False,seed=c['rng_seed'])
    records=[json.loads(x) for x in (root/'m8s-diagnostics/semantic_compare1'/(args.mode+'_transitions.jsonl')).read_text().splitlines()][:4]
    data={};handles=[];weights={}
    def digest(t):return hashlib.sha256(t.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest()
    def hook(name):
        def capture(module,inputs,output):
            if isinstance(output,tuple):output=output[0]
            if not isinstance(output,torch.Tensor):return
            key=name+'#'+str(sum(k.startswith(name+'#') for k in data))
            data[key]=output.detach().reshape(-1,output.shape[-1])[-2:].float().cpu()
        return capture
    for name,m in p.model.named_modules():
        if '.layers.0.' in name or ('.layers.' in name and name.rsplit('.',1)[-1].isdigit()):
            handles.append(m.register_forward_hook(hook(name)))
    with torch.no_grad():
        row=p.prepare([records[0]['observation']]);vd,td=p.distributions(row,[records[0]['verb_index']])
    for h in handles:h.remove()
    for name,m in p.model.named_modules():
        if hasattr(m,'_grouped_projection'):weights[name]=digest(m._grouped_projection.group.weight)
    errors=[];logps=[]
    with torch.no_grad():
        rows=p.prepare([r['observation'] for r in records]);vs=torch.tensor([r['verb_index'] for r in records],device='cuda');ts=torch.tensor([r['target_index'] for r in records],device='cuda')
        v,t=p.distributions(rows,vs.tolist());logps=(v.log_prob(vs)+t.log_prob(ts)).tolist();errors=[x-r['log_probability'] for x,r in zip(logps,records)]
    from triton.runtime.autotuner import Autotuner
    tuners=[]
    for obj in gc.get_objects():
        if type(obj) is Autotuner and obj.cache:
            tuners.append({'fn':str(obj.fn),'configs':{str(k):str(v) for k,v in obj.cache.items()}})
    result={'actor_digest':p.digest(),'mode':args.mode,'stable':args.stable,'logps':logps,'historical_errors':errors,'effective_weights':weights,
            'stages':[{'name':k,'sha256':digest(v),'shape':list(v.shape)} for k,v in data.items()],
            'autotuners':tuners,'tf32':torch.backends.cuda.matmul.allow_tf32,'matmul_precision':torch.get_float32_matmul_precision()}
    out=Path(args.output);out.parent.mkdir(parents=True,exist_ok=True)
    torch.save(data,str(out)+'.pt');out.write_text(json.dumps(result,indent=2));print(json.dumps({k:result[k] for k in ('mode','stable','logps','historical_errors')}),flush=True)
if __name__=='__main__':main()
