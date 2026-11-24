"""GPU regression and latency probe; zero optimizer steps or checkpoint writes."""
import argparse
import gc
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.benchmark_m8s3 import setup, selected_logps, write


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--states', required=True)
    ap.add_argument('--output', required=True)
    ap.add_argument('--legacy', action='store_true')
    ap.add_argument('--profile', action='store_true')
    args = ap.parse_args()
    setup()
    import torch
    if args.legacy:
        from agent.m8s3_policy import BatchedSemanticPolicy as Policy
    else:
        from agent.m8s3_batch_policy import CanonicalBatchPolicy as Policy
    c = SimpleNamespace(**json.loads(Path(args.config).read_text()))
    p = Policy(c.model, checkpoint=c.initial_checkpoint, parent_checkpoint=c.parent_checkpoint,
               seed=c.rng_seed, reset_critic=True, with_reference=True)
    before = p.digest()
    rows = p.prepare(json.loads(Path(args.states).read_text()))
    calls = p.cache_stats['forward_calls']
    with torch.no_grad():
        p.distributions(rows[:4], [0] * 4)
    root_calls = p.cache_stats['forward_calls'] - calls
    assert root_calls == 1, f'Four same-bucket states need one backbone call, got {root_calls}'
    verbs = []
    targets = []
    for i, row in enumerate(rows):
        legal = [v for v, b in enumerate(row['branches']) if b]
        v = legal[i % len(legal)]
        verbs.append(v)
        targets.append(i % len(row['branches'][v]))
    old = selected_logps(p, rows, verbs, targets)
    errors = []
    for size in (1, 3, 4):
        current = selected_logps(p, rows, verbs, targets, size, list(reversed(range(len(rows)))))
        errors.append(max(abs(a - b) for a, b in zip(old, current)))
    grad = selected_logps(p, rows, verbs, targets, grad=True)
    grad_error = max(abs(a - b) for a, b in zip(old, grad))
    def step():
        p.model.zero_grad(set_to_none=True)
        vd, td = p.distributions(rows[:4], [0] * 4)
        (-vd.logits[:, 0].mean() - td.logits[:, 0].mean()).backward()
    step()
    torch.cuda.synchronize()
    times = []
    for _ in range(5):
        start = time.perf_counter()
        step()
        torch.cuda.synchronize()
        times.append(time.perf_counter() - start)
    if args.profile:
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]) as prof:
            step()
        Path(args.output + '.profile.txt').write_text(prof.key_averages().table(sort_by='self_cpu_time_total', row_limit=30))
    nav = next(r for r in rows if len(r['branches'][1]) > 1)
    p.model.zero_grad(set_to_none=True)
    vd, td = p.distributions([nav], [1])
    (-vd.logits[0, 1] - td.logits[0, 0]).backward()
    norm = float(torch.nn.utils.clip_grad_norm_([q for q in p.model.parameters() if q.requires_grad], 1.))
    result = dict(root_calls=root_calls, reorder_error=max(errors), grad_mode_error=grad_error,
                  navigation_grad_norm=norm, microbatch_seconds=times, actor_unchanged=p.digest() == before,
                  peak_allocated_bytes=torch.cuda.max_memory_allocated(), optimizer_steps=0,
                  formal_training_started=False, cache_stats=p.cache_stats)
    write(args.output, result)
    print(json.dumps(result), flush=True)
    assert max(errors + [grad_error]) <= 1e-6
    assert torch.isfinite(torch.tensor(norm)) and norm > 0 and result['actor_unchanged']


if __name__ == '__main__':
    main()
