"""One diagnostic microbatch profile; no optimizer steps."""
import json,sys,warnings
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from agent.m8s2_policy import BatchedSemanticPolicy
warnings.filterwarnings('ignore',message='Input tensor shape suggests potential format mismatch:.*')
torch.set_num_threads(8);root=Path('/home/researcher/meadowmind-gpu');c=json.loads((root/'m8s-runs/semantic_diagnostic1/registration.json').read_text())['config']
p=BatchedSemanticPolicy(c['model'],checkpoint=c['initial_checkpoint'],parent_checkpoint=c['parent_checkpoint'],with_reference=False)
data=json.loads(Path(c['dataset']).read_text())['observations'];rows=p.prepare([x['observation'] for x in data[:4]])
def step():
    p.model.zero_grad(set_to_none=True);v,t=p.distributions(rows,[0]*4);loss=-v.logits[:,0].mean()-t.logits[:,0].mean();loss.backward()
step();torch.cuda.synchronize()
with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA],record_shapes=True) as prof:step()
print(prof.key_averages(group_by_input_shape=True).table(sort_by='self_cuda_time_total',row_limit=25))
