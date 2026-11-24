"""Independent serial-vs-batched semantic branch gradient check; no updates."""
import copy,json,sys,warnings
from pathlib import Path
from types import MethodType
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from agent.m8s2_policy import BatchedSemanticPolicy
warnings.filterwarnings('ignore',message='Input tensor shape suggests potential format mismatch:.*')
torch.set_num_threads(8)
root=Path('/home/researcher/meadowmind-gpu');c=json.loads((root/'m8s2-runs/semantic_diagnostic2/registration.json').read_text())['config']
p=BatchedSemanticPolicy(c['model'],checkpoint=c['initial_checkpoint'],parent_checkpoint=c['parent_checkpoint'],with_reference=False,seed=c['rng_seed'])
data=json.loads(Path(c['dataset']).read_text())['observations'];rows=p.prepare([x['observation'] for x in data[:4]])
initial=p.digest();original=p._forward_batch
def select_cache(cache,index,batch):
    def take(x):
        if isinstance(x,torch.Tensor):return x[index:index+1] if x.ndim and x.shape[0]==batch else x
        if isinstance(x,list):return [take(v) for v in x]
        if isinstance(x,dict):return {k:take(v) for k,v in x.items()}
        if isinstance(x,tuple):return tuple(take(v) for v in x)
        return x
    out=copy.copy(cache);out.layers=[]
    for layer in cache.layers:
        child=copy.copy(layer)
        for k,v in vars(layer).items():setattr(child,k,take(v))
        if hasattr(child,'batch_size'):child.batch_size=1
        out.layers.append(child)
    return out
def serial(self,model,sequences,cache):
    if len(sequences)==1:return original(model,sequences,cache)
    return torch.cat([original(model,[seq],select_cache(cache,i,len(sequences))) for i,seq in enumerate(sequences)])
def run():
    p.model.zero_grad(set_to_none=True);p.clear_prefix_cache()
    q=p.target_distribution(rows,[1]*4);loss=-q.logits[:,0].mean();loss.backward()
    gradient=torch.cat([v.grad.flatten().float() for v in p.model.parameters() if v.requires_grad and v.grad is not None])
    return q.probs.detach(),gradient,float(loss.detach())
a,ga,la=run();p._forward_batch=MethodType(serial,p);b,gb,lb=run()
cos=float(torch.nn.functional.cosine_similarity(ga,gb,dim=0));rel=float((ga-gb).norm()/ga.norm())
result={'max_probability_difference':float((a-b).abs().max()),'gradient_cosine':cos,'gradient_relative_l2':rel,
        'batch_loss':la,'serial_loss':lb,'gradient_elements':ga.numel(),'weights_unchanged':p.digest()==initial,'optimizer_steps':0,'formal_training_started':False}
Path(sys.argv[1]).write_text(json.dumps(result,indent=2));print(json.dumps(result))
assert result['weights_unchanged'] and torch.isfinite(ga).all() and torch.isfinite(gb).all() and cos>.99 and rel<.1
p._forward_batch=original;p.model.zero_grad(set_to_none=True);p.clear_prefix_cache()
from agent.m8x_env import GodotEnv
from agent.m8x_contract import SurvivalReward
navigation=[]
for seed in c['seeds']:
    with GodotEnv(c['godot'],c['project']) as env:
        state=env.reset(seed,'open3');obs=state['observation'];row=p.prepare([obs])[0]
        with torch.no_grad():q=p.target_distribution([row],[1]);target=int(q.probs[0].argmax())
        action=row['branches'][1][target];nxt=env.step(action)
        before=nxt['transition']['before']['player'];after=nxt['transition']['after']['player']
        assert abs(before['x']-after['x'])+abs(before['y']-after['y'])==1 and nxt['transition']['elapsed_ticks']==1
        navigation.append({'source':'forced_navigation_verb_probe_model_selects_target','seed':seed,
            'state':row['messages'],'semantic_target':row['target_texts'][1][target],
            'conditional_target_probability':float(q.probs[0,target]),'action':action,
            'transition':nxt['transition'],'reward':SurvivalReward(obs).score(nxt['transition'],nxt['observation'])})
Path(sys.argv[1]).with_name('one_step_navigation_probe.json').write_text(json.dumps(navigation,ensure_ascii=False,indent=2))
print(json.dumps({'navigation_cases':len(navigation),'all_moved_one_tile_one_tick':True,'verb_forced_not_natural_rollout':True}))
