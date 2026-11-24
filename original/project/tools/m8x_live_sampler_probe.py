"""Post-release CUDA equivalence on eight fixed training observations only.

Never launches a service or steps a world. Run only after the GPU is idle.
The publication gate is rebuilt before importing/loading the GPU actor.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from agent.m8x_release import canonical, verify_release
from agent.m8x_identity import sha
from agent.m6_live_session import live_seed


def require_idle_gpu():
    active = subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,used_gpu_memory',
                                      '--format=csv,noheader'],text=True).strip()
    if active:
        raise RuntimeError('GPU compute work left untouched: '+active)


def require_accepted(report):
    if not (report.get('protocol') == 'm8_exploration_reliable_v1' and report.get('phase') == 'test'
            and report.get('decision') == 'accept' and report.get('release_authorized') is True
            and report.get('evidence_complete') is True):
        raise ValueError('Complete accepted M8X test publication is required')


def preflight(verification, project):
    report = json.loads(Path(verification).read_text(encoding='utf-8'))
    require_accepted(report)
    config = report['config']
    checkpoint = Path(config['output']) / report['selected_checkpoint']
    # Rebuild all original evidence and bind artifact metadata before loading weights.
    verify_release(verification=verification, project=project, checkpoint=checkpoint,
                   model=config['model'], actor_sha256=report['selected_actor_sha256'],
                   policy_metadata=json.loads((checkpoint/'policy.json').read_text(encoding='utf-8')),
                   execution_identity=report['execution_identity'])
    return report, checkpoint


def select_pairs(rows, config):
    """Exactly first two rows of update 1 per env; never search by outcome."""
    groups = {i: [] for i in range(4)}
    for index, row in enumerate(rows):
        if row['update'] != 1:
            continue
        env = row['env']
        if env not in groups:
            raise ValueError('Expected exactly four training environments')
        if len(groups[env]) < 2:
            seed = row['seed']
            if not (config['train_seed_start'] <= seed < config['train_seed_start'] + config['train_seed_count']) or seed in config['validation_seeds'] + config['test_seeds']:
                raise ValueError('Only registered training seeds are permitted')
            groups[env].append((index, row))
        if all(len(v) == 2 for v in groups.values()):
            break
    for pair in groups.values():
        if len(pair) != 2:
            raise ValueError('Incomplete first training pair')
        first, second = pair[0][1], pair[1][1]
        if (first['seed'] != second['seed'] or first['terminal'] or first['truncated']
                or first['transition']['after']['tick'] != second['observation']['tick']
                or first['observation']['tick'] >= second['observation']['tick']):
            raise ValueError('Training observations must be consecutive in one journey')
    return [groups[i] for i in range(4)]


def training_pairs(report):
    path = Path(report['publication_request']['training_run'])/'transitions.jsonl'
    # Full publication rebuild already hashes this same trace; retain it in evidence.
    with path.open(encoding='utf-8') as stream:
        pairs = select_pairs((json.loads(line) for line in stream), report['config'])
    return pairs, str(path), sha(path)


def write_new(path, report):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verification', required=True)
    parser.add_argument('--project', default=str(ROOT))
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    if Path(args.output).exists(): raise FileExistsError(args.output)
    report, checkpoint = preflight(args.verification, args.project)
    pairs, source, source_sha = training_pairs(report)
    require_idle_gpu()
    import torch
    from agent.m8x_policy import Policy
    from agent.m8x_identity import execution_identity
    from agent.m8x_server import sample_stochastic
    if not torch.cuda.is_available(): raise RuntimeError('Actual CUDA is required')
    class CudaStream:
        def __init__(self, seed): self.generator = torch.Generator(device='cuda').manual_seed(seed)
        def snapshot(self): return self.generator.get_state().cpu().numpy().tobytes()
    torch.set_num_threads(8)
    c = report['config']
    policy = Policy(c['model'], checkpoint=checkpoint, parent_checkpoint=c['parent_checkpoint'], max_tokens=1536, with_reference=False)
    policy.eval()
    actor = policy.digest()
    verify_release(verification=args.verification, project=args.project, checkpoint=checkpoint,
                   model=c['model'], actor_sha256=actor, policy_metadata=policy.metadata(),
                   execution_identity=execution_identity(c, torch))
    contexts = [dict(session_id='m8x-probe-'+uuid.uuid4().hex, episode=0, world_seed=p[0][1]['seed']) for p in pairs]
    offline = [CudaStream(live_seed(c)) for c in contexts]
    online = [CudaStream(live_seed(c)) for c in contexts]
    fixtures = [[] for _ in pairs]
    comparisons = []
    with torch.no_grad():
        for step in range(2):
            prepared = policy.prepare([p[step][1]['observation'] for p in pairs], salt='m6')
            before = [hashlib.sha256(s.snapshot()).hexdigest() for s in offline]
            vd = policy.verb_distribution(prepared)
            verbs = [int(torch.multinomial(vd.probs[i], 1, generator=s.generator).item()) for i,s in enumerate(offline)]
            td = policy.target_distribution(prepared, verbs)
            targets = [int(torch.multinomial(td.probs[i], 1, generator=s.generator).item()) for i,s in enumerate(offline)]
            for i,pair in enumerate(pairs):
                index,row = pair[step]
                actual = sample_stochastic(policy, policy.prepare([row['observation']], salt='m6'), online[i], torch)
                expected = dict(action=prepared[i]['branches'][verbs[i]][targets[i]], verb_index=verbs[i], target_index=targets[i],
                    verb_probabilities=vd.probs[i].tolist(), target_probabilities=td.probs[i].tolist(),
                    rng_before_sha256=before[i], rng_after_sha256=hashlib.sha256(offline[i].snapshot()).hexdigest(),
                    agent_context=contexts[i], rng_seed=str(live_seed(contexts[i])), tick=row['observation']['tick'])
                equal = (actual['verb']==verbs[i] and actual['target']==targets[i] and actual['action']==expected['action']
                    and actual['verb_probabilities']==expected['verb_probabilities'] and actual['target_probabilities']==expected['target_probabilities']
                    and online[i].snapshot()==offline[i].snapshot())
                comparisons.append(dict(env=i, step=step, exact_equal=equal, singleton_target=len(prepared[i]['branches'][verbs[i]])==1))
                fixtures[i].append(dict(source_index=index, observation=row['observation'], expected=expected))
    passed = all(r['exact_equal'] for r in comparisons) and policy.digest()==actor
    result = dict(protocol='m8x_live_probe_v1', passed=passed, actual_cuda=True, actor_sha256=actor,
        verification_sha256=sha(args.verification), checkpoint=str(checkpoint.resolve()), training_trace=source,
        training_trace_sha256=source_sha, contexts=contexts, fixtures=fixtures, comparisons=comparisons,
        weights_unchanged=policy.digest()==actor, physical_forward_shape=[4,1536], temperature=1, encoding_salt='m6',
        source_sha256={p:sha(ROOT/p) for p in ('tools/m8x_live_sampler_probe.py','agent/m8x_server.py','agent/m6_live_session.py')},
        new_world_steps=0, optimizer_steps=0, held_out_observations=0, http_executed=False,
        scope='CUDA B4 offline versus actual online sampler on eight fixed training states; not survival or HTTP evidence')
    write_new(args.output, result)
    print(json.dumps(dict(passed=passed, observations=8, actual_cuda=True)))
    if not passed: raise SystemExit(1)


if __name__ == '__main__': main()
