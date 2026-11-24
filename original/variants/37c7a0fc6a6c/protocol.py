"""Independent scalar audits; no simulator or tensor imports."""
import math

def audit_gae(rows,bootstrap,gamma,lam):
    assert rows and rows[-1]['boundary']
    carry=0.;advantages=[None]*len(rows);terminal_indices=[]
    for i in reversed(range(len(rows))):
        r=rows[i]
        assert type(r['duration']) is int and r['duration']>0
        assert r['boundary']==(r['terminal'] or i==len(rows)-1)
        n=0. if r['terminal'] else rows[i+1]['value'] if i+1<len(rows) else bootstrap
        d=gamma**r['duration']
        carry=r['reward']+d*n-r['value']+d*lam**r['duration']*(not r['boundary'])*carry
        assert math.isfinite(carry)
        assert abs(carry-r['advantage'])<1e-10 and abs(carry+r['value']-r['return'])<1e-10
        if r['terminal']:
            assert n==0. and abs(carry-(r['reward']-r['value']))<1e-10
            terminal_indices.append(i)
        advantages[i]=carry
    return dict(advantages=advantages,terminal_indices=sorted(terminal_indices),passed=True)

def optimizer_gate(before,after,actor_steps,critic_steps):
    for name,increment in [('actor',actor_steps),('critic',critic_steps)]:
        assert after[name] and increment>0
        expected=(max(before[name]) if before[name] else 0)+increment
        assert all(s==expected for s in after[name]),(name,expected,after[name])

def resource_gate(samples):
    assert len(samples)>=3
    baseline=samples[1]
    gpu=max(s['allocated']-baseline['allocated'] for s in samples[2:])
    rss=max(s['rss']-baseline['rss'] for s in samples[2:])
    return dict(passed=gpu<=512*1024**2 and rss<=4*1024**3,
        gpu_growth_bytes=gpu,rss_growth_bytes=rss,gpu_limit_bytes=512*1024**2,rss_limit_bytes=4*1024**3)
