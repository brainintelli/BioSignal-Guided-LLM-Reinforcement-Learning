import sys,json,time
from pathlib import Path
sys.path.insert(0,'/home/researcher/meadowmind-gpu/m8s3-batch-project')
from types import SimpleNamespace
from tools.benchmark_m8s3 import setup,make
setup()
import torch
c=SimpleNamespace(**json.loads(Path('/home/researcher/meadowmind-gpu/m8s3-runs/survival_diagnostic3/registration.json').read_text())['config'])
p=make(c,'optimized')
data=json.loads(Path('/home/researcher/meadowmind-gpu/m8s3-diagnostics/equal_update3/controlled_states.json').read_text())
rows=p.prepare(data[:4])
def step():
 p.model.zero_grad(set_to_none=True)
 v,t=p.distributions(rows,[0]*4)
 (-v.logits[:,0].mean()-t.logits[:,0].mean()).backward()
step();torch.cuda.synchronize()
start=time.perf_counter()
for _ in range(5):step()
torch.cuda.synchronize()
print('microbatch_seconds', (time.perf_counter()-start)/5,flush=True)
with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA],record_shapes=True) as prof:step()
print(prof.key_averages(group_by_input_shape=True).table(sort_by='self_cuda_time_total',row_limit=35))
print(prof.key_averages().table(sort_by='self_cpu_time_total',row_limit=25))
print(json.dumps(dict(cache_stats=p.cache_stats,optimizer_steps=0,formal_training_started=False)))
