"""Paired scheduling and fixed held-out evaluation, shared by rehearsal and live runs."""
from concurrent.futures import ThreadPoolExecutor
from .protocol import ARMS, STEPS, physical_order


def together(function, arms=ARMS):
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures={arm:pool.submit(function,arm) for arm in arms}
        return {a:f.result() for a,f in futures.items()}


def run_pair(device, workers, replicate, record):
    """Only this controller requests physical slots. GPU inference/updates run in parallel."""
    results={}
    for phase in ('train','eval'):
        priming=device('prime',name='prime_'+phase)
        together(lambda a: workers[a]('start',phase=phase,windows=priming))
        active=set(ARMS)
        episodes=dict.fromkeys(ARMS,0)
        for i in range(STEPS):
            decisions=together(lambda a: workers[a]('act'),active)
            for a in physical_order(i,replicate):
                if a not in active:continue
                d=decisions[a]
                req=d['request']
                if d['terminal'] != (req is None):
                    raise ValueError('Invalid terminal decision')
                receipt=None
                if req is not None:
                    if (req['arm'],req['phase'],req['episode'],req['index'])!=(a,phase,episodes[a],i):
                        raise ValueError('Unexpected physical transaction')
                    if req['pulses'] and i%3!=2:raise ValueError('Outside scheduled stimulation opportunity')
                    receipt=device('slot',req=req)
                workers[a]('accept',receipt=receipt)
                record(dict(phase=phase,arm=a,index=i,terminal=d['terminal'],request=req,receipt=receipt))
                if d['terminal']:
                    if phase=='eval':
                        active.remove(a)
                    elif i+1<STEPS:
                        episodes[a]+=1
                        w=device('prime',name=f'reset_{a}_{i}')
                        workers[a]('start',phase=phase,windows=w,episode=episodes[a])
            if not active:break
        results[phase]=together(lambda a:workers[a]('finish'))
    c,d=(results['eval'][a] for a in ARMS)
    return dict(results=results,ordered_minus_shuffled={k:c[k]-d[k] for k in c},
                independent_samples=1,inference='single_chip_paired_pilot')
