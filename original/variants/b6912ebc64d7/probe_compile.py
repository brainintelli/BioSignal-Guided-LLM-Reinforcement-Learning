"""Profile launch overhead, then compare regional compiled execution."""
import gc,json,time,sys
from pathlib import Path
sys.path.insert(0,'/home/researcher/meadowmind-gpu/m8b-project')
import torch
from agent.m8x_fast_policy import FastPolicy
from agent.m8x_causal_policy import CausalKVPolicy
torch.set_num_threads(8)
torch._inductor.config.compile_threads=4
torch._dynamo.config.cache_size_limit=64
# Installed Transformers calls the newer export-state query while tracing.
# This diagnostic uses torch.compile only, never torch.export.
if not hasattr(torch.compiler,'is_exporting'):
    torch.compiler.is_exporting=lambda:False
root=Path('/home/researcher/meadowmind-gpu')
c=json.loads((root/'m8b-project/configs/m8x_exploration_attempt1.json').read_text())
rows=[json.loads(x) for x in (root/'diagnostics/rl_speed_20261107_v2/baseline_fresh_transitions.jsonl').read_text().splitlines()][::12][:16]
results=[]
for mode in ['compact_eager','compact_compiled','kv_compiled']:
    cls=CausalKVPolicy if mode=='kv_compiled' else FastPolicy
    p=cls(c['model'],checkpoint=str(root/'m8b-runs/exploration_attempt1/checkpoint_0032'),
          parent_checkpoint=c['parent_checkpoint'],with_reference=True,**({'optimization':'compact'} if cls is FastPolicy else {}))
    prepared=p.prepare([r['observation'] for r in rows]);v=torch.tensor([r['verb_index'] for r in rows],device='cuda');t=torch.tensor([r['target_index'] for r in rows],device='cuda')
    if mode!='compact_eager':
        for model in (p.model,p.reference_model):
            text=model.get_base_model().model.language_model
            for layer in text.layers:
                layer.forward=torch.compile(layer.forward,dynamic=True,fullgraph=False)
        print(json.dumps(dict(event='compile_enabled',mode=mode)),flush=True)
    def inference():
        found=[]
        with torch.no_grad():
            for start in range(0,16,4):
                a,b=p.distributions(prepared[start:start+4],v[start:start+4].tolist())
                found.extend((a.log_prob(v[start:start+4])+b.log_prob(t[start:start+4])).tolist())
                p.distributions(prepared[start:start+4],v[start:start+4].tolist(),True)
        return found
    times=[];backward=[];errors=[]
    for repeat in range(3):
        torch.cuda.synchronize();begin=time.perf_counter();found=inference();torch.cuda.synchronize();times.append(time.perf_counter()-begin)
        print(json.dumps(dict(event='forward',mode=mode,repeat=repeat,seconds=times[-1])),flush=True)
        begin=time.perf_counter();largest=0
        for start in range(0,16,4):
            p.model.zero_grad(set_to_none=True)
            a,b=p.distributions(prepared[start:start+4],v[start:start+4].tolist())
            logp=a.log_prob(v[start:start+4])+b.log_prob(t[start:start+4])
            largest=max(largest,max(abs(x-y) for x,y in zip(logp.detach().tolist(),found[start:start+4])))
            (-logp.mean()).backward()
            assert all(torch.isfinite(q.grad).all() for q in p.model.parameters() if q.grad is not None)
            del a,b,logp
        torch.cuda.synchronize();backward.append(time.perf_counter()-begin);errors.append(largest)
        print(json.dumps(dict(event='backward',mode=mode,repeat=repeat,seconds=backward[-1],error=largest)),flush=True)
    if mode=='compact_eager':
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as profile:inference()
        (root/'diagnostics/rl_compile_profile_20261107.txt').write_text(profile.key_averages().table(sort_by='self_cpu_time_total',row_limit=25)+'\n'+profile.key_averages().table(sort_by='self_cuda_time_total',row_limit=25))
    results.append(dict(mode=mode,inference=times,backward=backward,error=errors,logp=found))
    (root/'diagnostics/rl_compile_probe_v2_20261107.json').write_text(json.dumps(results,indent=2))
    del p,prepared;gc.collect();torch.cuda.empty_cache()
