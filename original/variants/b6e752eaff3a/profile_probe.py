"""Bounded read-only-weight diagnostic. No optimizer step or checkpoint save."""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import statistics
import subprocess
import time


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--output', required=True)
    a = ap.parse_args()
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=False)
    occupied = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,process_name', '--format=csv,noheader'], text=True).strip()
    if occupied:
        raise RuntimeError('GPU occupied: ' + occupied)
    import torch
    from agent.m8x_policy import Policy
    from agent.m8x_env import GodotEnv
    from agent.m8x_contract import SurvivalReward, CODES, target_messages, potential, WATER_POTENTIAL
    from agent.m8b_water_potential import water_potential
    from agent.m8x_train import finish_trajectory
    root = Path('/home/researcher/meadowmind-gpu')
    project = root / 'm8b-project'
    config = json.loads((project / 'configs/m8x_exploration_attempt1.json').read_text())
    torch.set_num_threads(8)
    torch.manual_seed(20261107)
    started = time.perf_counter()
    p = Policy(config['model'], checkpoint=str(root/'m8b-runs/exploration_attempt1/checkpoint_0032'),
               parent_checkpoint=config['parent_checkpoint'], seed=20261107,
               max_tokens=1536, with_reference=True, gradient_checkpointing=False)
    torch.cuda.synchronize()
    load_seconds = time.perf_counter()-started
    initial = p.digest()
    ref_initial = p.reference_digest()
    metadata = p.metadata()
    # Fail immediately if this policy unexpectedly enters autoregressive generation.
    def forbidden_generate(*args, **kwargs):
        raise AssertionError('Unexpected generate() call in classification policy')
    p.model.generate = forbidden_generate
    p.reference_model.generate = forbidden_generate
    timers = defaultdict(list)
    forward_calls = []
    stage = 'warmup'
    def timed(label, fn):
        torch.cuda.synchronize()
        begin = time.perf_counter()
        result = fn()
        torch.cuda.synchronize()
        if stage == 'measured':
            timers[label].append(time.perf_counter()-begin)
        return result
    original_logits = p._logits
    def wrapped_logits(inputs, ids, reference=False):
        label = ('reference_' if reference else 'actor_') + ('verb' if ids == p.verb_ids else 'target')
        result = timed(label, lambda: original_logits(inputs, ids, reference))
        if stage == 'measured':
            forward_calls.append(dict(kind=label, real_rows=len(inputs), physical_rows=4*math.ceil(len(inputs)/4),
                                      lengths=[len(x) for x in inputs], sequence_length=1536))
        return result
    p._logits = wrapped_logits
    original_values = p.values
    p.values = lambda prepared: timed('critic', lambda: original_values(prepared))
    envs, states, trackers = [], [], []
    trajectories = [[] for _ in range(4)]
    records = []
    actions = Counter()
    branch_count = 0
    measured_start = None
    bench_prepared = None
    try:
        for i in range(4):
            env = GodotEnv(config['godot'], str(project)); envs.append(env)
            state = env.reset(882610+i, 'open3'); states.append(state)
            trackers.append(SurvivalReward(state['observation']))
        for step in range(72):
            if step == 8:
                stage = 'measured'; measured_start = time.perf_counter()
                torch.cuda.reset_peak_memory_stats()
            prepared = timed('prepare', lambda: p.prepare([s['observation'] for s in states]))
            if step == 8:
                bench_prepared = prepared
            sampled = timed('act_total', lambda: p.act(prepared, with_reference=True))
            for i, env in enumerate(envs):
                action = sampled['actions'][i]
                nxt = timed('env_step', lambda: env.step(action))
                scored = timed('reward', lambda: trackers[i].score(nxt['transition'], nxt['observation']))
                v, t = sampled['verbs'][i], sampled['targets'][i]
                target_ids = p._target_tokens(prepared[i], v) if len(prepared[i]['branches'][v]) > 1 else []
                record = dict(probe_step=step, env=i, seed=882610+i, measured=step >= 8,
                              source='model', actor_sha256=initial, observation=states[i]['observation'],
                              model_input=prepared[i]['messages'], facts=prepared[i]['facts'],
                              critic_features=prepared[i]['features'], verb_index=v, verb_code=CODES[v],
                              target_index=t, target_code=(prepared[i]['target_codes'][v][t] if target_ids else None),
                              target_input=(target_messages(prepared[i], v) if target_ids else None),
                              verb_input_tokens=len(prepared[i]['input_ids']), target_input_tokens=len(target_ids),
                              action=action, log_probability=sampled['logp'][i], value=sampled['values'][i],
                              transition=nxt['transition'], next_observation=nxt['observation'], reward=scored,
                              terminal=nxt['terminated'])
                records.append(record)
                if stage == 'measured':
                    actions[action['verb']] += 1
                    branch_count += bool(target_ids)
                    trajectories[i].append(dict(prepared=prepared[i], verb=v, target=t,
                        value=sampled['values'][i], logp=sampled['logp'][i], verb_logp=sampled['verb_logp'][i],
                        reference_verb=sampled['reference_verb'][i], reference_target=sampled['reference_target'][i],
                        reward=scored['reward'], duration=scored['elapsed_ticks'], terminal=nxt['terminated'],
                        boundary=nxt['terminated'], next_obs=nxt['observation']))
                states[i] = nxt
                if nxt['terminated']:
                    raise RuntimeError('Probe encountered death; shorten probe or explicitly record resets')
            if step % 16 == 15:
                print(json.dumps(dict(event='rollout_progress', steps=step+1, decisions=(step+1)*4)), flush=True)
        measured_seconds = time.perf_counter()-measured_start
        peak = torch.cuda.max_memory_allocated()
        stage = 'post_probe'
        p._logits = original_logits
        p.values = original_values
        rows = []
        for trajectory in trajectories:
            with torch.no_grad():
                bootstrap = float(p.values(p.prepare([trajectory[-1]['next_obs']]))[0])
            rows.extend(finish_trajectory(trajectory, bootstrap, .9995, .999))
        with (out/'fresh_transitions.jsonl').open('x', encoding='utf-8') as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False, allow_nan=False)+'\n')
        # Time the exact actor objective forward/backward, without ANY optimizer step.
        t = {k: torch.tensor([r[k] for r in rows], device='cuda') for k in
             ['advantage','logp','verb_logp','reference_verb','reference_target','verb','target']}
        adv = t['advantage'].float()
        adv = (adv-adv.mean())/adv.std(unbiased=False).clamp_min(1e-8)
        backward = []
        for repeat in range(3):
            p.model.zero_grad(set_to_none=True)
            torch.cuda.synchronize(); begin = time.perf_counter()
            for k in range(8):
                ix = list(range(k*4, (k+1)*4))
                vd, td = p.distributions([rows[j]['prepared'] for j in ix], t['verb'][ix].tolist())
                lp = vd.log_prob(t['verb'][ix])+td.log_prob(t['target'][ix])
                ratio = (lp-t['logp'][ix]).exp()
                pl = torch.maximum(-adv[ix]*ratio, -adv[ix]*ratio.clamp(.85,1.15)).mean()
                importance = (vd.log_prob(t['verb'][ix])-t['verb_logp'][ix]).exp()
                vkl = (vd.probs*(vd.logits-t['reference_verb'][ix].clamp_min(1e-12).log())).sum(-1)
                tkl = (td.probs*(td.logits-t['reference_target'][ix].clamp_min(1e-12).log())).sum(-1)
                ent = (vd.entropy()+importance*td.entropy()).mean()
                ((pl+.005*(vkl+importance*tkl).mean()-.002*ent)/8).backward()
            torch.nn.utils.clip_grad_norm_([x for x in p.model.parameters() if x.requires_grad], 1.)
            torch.cuda.synchronize(); backward.append(time.perf_counter()-begin)
        p.model.zero_grad(set_to_none=True)
        # Isolated shape experiment; these outputs never drive the environment.
        def raw_forward(inputs, length):
            b = p.tokenizer.pad({'input_ids':inputs},padding='max_length',max_length=length,return_tensors='pt')
            b = {k:v.to('cuda') for k,v in b.items()}
            z = p.model(**b,use_cache=False,output_hidden_states=True,return_dict=True,logits_to_keep=1)
            return torch.nn.functional.linear(z.hidden_states[-1][:,-1,:].float(),p.model.get_output_embeddings().weight[p.verb_ids].float())
        inputs = [r['input_ids'] for r in bench_prepared]
        shapes = []
        with torch.no_grad():
            baseline = raw_forward(inputs,1536)
            for label, test_inputs, length in [('fixed_B4_L1536',inputs,1536),
                    ('trimmed_B4',inputs,max(map(len,inputs))),('single_row_L1536',inputs[:1],1536)]:
                elapsed=[]
                for repeat in range(7):
                    torch.cuda.synchronize(); begin=time.perf_counter()
                    actual=raw_forward(test_inputs,length)
                    torch.cuda.synchronize()
                    if repeat>=2: elapsed.append(time.perf_counter()-begin)
                delta=float((actual.log_softmax(-1)-baseline[:len(test_inputs)].log_softmax(-1)).abs().max())
                shapes.append(dict(case=label,batch=len(test_inputs),length=length,seconds=elapsed,
                                   median_seconds=statistics.median(elapsed),max_unmasked_logp_difference=delta))
        def stats(xs):
            ordered=sorted(xs)
            return dict(n=len(xs),mean=statistics.mean(xs),median=statistics.median(xs),
                        p95=ordered[min(len(xs)-1,math.ceil(.95*len(xs))-1)],min=min(xs),max=max(xs),sum=sum(xs))
        measured = [r for r in records if r['measured']]
        result=dict(scope='fresh short diagnostic; not historical trajectory, training, or acceptance',
            created_utc=datetime.now(timezone.utc).isoformat(), checkpoint='checkpoint_0032',
            probe_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            seeds=[882610,882611,882612,882613], sampling_seed=20261107,
            warmup_decisions=32,measured_decisions=256,total_fresh_decisions=288,
            load_seconds=load_seconds,measured_seconds=measured_seconds,
            measured_decisions_per_second=256/measured_seconds,timing={k:stats(v) for k,v in timers.items()},
            forward_calls=forward_calls,verb_input_tokens=stats([r['verb_input_tokens'] for r in measured]),
            target_input_tokens=stats([r['target_input_tokens'] for r in measured if r['target_input_tokens']]),
            target_required_decisions=branch_count,actions=dict(actions),shape_experiments=shapes,
            actor_forward_backward_32_rows_seconds=backward,actor_forward_backward_measured_median=statistics.median(backward[1:]),
            gpu_peak_allocated_bytes=peak,torch=torch.__version__,cuda=torch.version.cuda,
            gpu=torch.cuda.get_device_name(),model_metadata=metadata,
            actor_digest_before=initial,actor_digest_after=p.digest(),reference_digest_before=ref_initial,
            reference_digest_after=p.reference_digest(),optimizer_steps=0,checkpoint_saved=False,
            generate_call_count=0,generate_guard_enabled=True,finished=True)
        assert result['actor_digest_before']==result['actor_digest_after']
        assert result['reference_digest_before']==result['reference_digest_after']
        (out/'profile.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
        print(json.dumps({k:v for k,v in result.items() if k not in ['forward_calls','model_metadata']},ensure_ascii=False),flush=True)
    finally:
        for env in envs:
            env.close()


if __name__=='__main__':
    main()
