"""One bounded 128-decision PPO pilot; no save, registration or formal launch."""
import json,sys,time,warnings
from pathlib import Path
from types import SimpleNamespace
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from agent.m8s2_policy import BatchedSemanticPolicy
from agent.m8x_contract import SurvivalReward
from agent.m8x_env import GodotEnv
from agent.m8x_train import finish_trajectory
from agent.m8_train import update

warnings.filterwarnings('ignore',message='Input tensor shape suggests potential format mismatch:.*')
torch.set_num_threads(8);root=Path('/home/researcher/meadowmind-gpu');reg=json.loads((root/'m8s-runs/semantic_diagnostic1/registration.json').read_text());c=SimpleNamespace(**reg['config']);c.actor_lr=5e-7
p=BatchedSemanticPolicy(c.model,checkpoint=c.initial_checkpoint,parent_checkpoint=c.parent_checkpoint,seed=c.rng_seed,with_reference=True)
actor=torch.optim.AdamW([x for x in p.model.parameters() if x.requires_grad],lr=c.actor_lr,eps=1e-5,weight_decay=0.)
critic=torch.optim.AdamW(p.critic.parameters(),lr=c.critic_lr,eps=1e-5,weight_decay=0.)
torch.manual_seed(c.rng_seed)
envs=[];states=[];track=[];trajectories=[[] for _ in range(4)]
try:
    for seed in c.seeds:
        e=GodotEnv(c.godot,c.project);envs.append(e);s=e.reset(seed,'open3');states.append(s);track.append(SurvivalReward(s['observation']))
    torch.cuda.synchronize();start=time.perf_counter()
    for _ in range(32):
        prepared=p.prepare([s['observation'] for s in states]);sampled=p.act(prepared,with_reference=True)
        for i,e in enumerate(envs):
            nxt=e.step(sampled['actions'][i]);r=track[i].score(nxt['transition'],nxt['observation']);assert not nxt['terminated']
            trajectories[i].append(dict(prepared=prepared[i],verb=sampled['verbs'][i],target=sampled['targets'][i],value=sampled['values'][i],logp=sampled['logp'][i],verb_logp=sampled['verb_logp'][i],reference_verb=sampled['reference_verb'][i],reference_target=sampled['reference_target'][i],reward=r['reward'],duration=r['elapsed_ticks'],terminal=False,boundary=False,next_obs=nxt['observation']))
            states[i]=nxt
    rows=[]
    for trajectory in trajectories:
        with torch.no_grad():value=float(p.values(p.prepare([trajectory[-1]['next_obs']]))[0])
        rows.extend(finish_trajectory(trajectory,value,c.gamma_tick,c.gae_lambda))
    torch.cuda.synchronize();middle=time.perf_counter();print(json.dumps({'event':'pilot_rollout','seconds':middle-start}),flush=True)
    metrics=update(p,actor,critic,rows,c)
    torch.cuda.synchronize();end=time.perf_counter()
    result=dict(actor_lr=c.actor_lr,rng_seed=c.rng_seed,rollout_seconds=middle-start,update_seconds=end-middle,total_seconds=end-start,metrics=metrics,counters=p.cache_stats,formal_training_started=False,checkpoint_writes=0)
    Path(sys.argv[1]).write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
finally:
    for e in envs:e.close()
