"""Summarize preregistered paired pilots. One sample contributes one difference."""
import argparse
import json
from math import comb
from pathlib import Path
from statistics import median


def lower95(x):
    """Distribution-free one-sided lower confidence bound for the population median."""
    n=len(x)
    ranks=[k for k in range(1,n+1) if sum(comb(n,j) for j in range(k))/2**n<=.05]
    return sorted(x)[max(ranks)-1] if ranks else None


def summarize(dirs):
    paired=[];incomplete=[];seen=set();chips=set();samples=set()
    encodings=set()
    for folder in dirs:
        b=json.loads((folder/'binding.json').read_text(encoding='utf-8'))
        encodings.add(b.get('feedback_encoding'))
        if len(encodings)>1:
            raise ValueError('Cannot pool samples with different feedback and reward definitions')
        if b['replicate'] in seen or b['chip_id'] in chips or b['organoid_id'] in samples:
            raise ValueError('Repeated registration, chip or biological sample')
        seen.add(b['replicate']);chips.add(b['chip_id']);samples.add(b['organoid_id'])
        if not (folder/'paired_result.json').exists():
            incomplete.append(b['replicate']);continue
        r=json.loads((folder/'paired_result.json').read_text())
        left,right=(r['results']['eval'][a] for a in ('ordered','shuffled'))
        delta={k:left[k]-right[k] for k in left}
        if r['ordered_minus_shuffled']!=delta:raise ValueError('Mismatched paired summary')
        paired.append(delta)
    stats={}
    for k in ('explored','health','hunger','thirst','health_min','hunger_min','thirst_min','death','ticks'):
        values=[x[k] for x in paired]
        if values:
            stats[k]=dict(differences=values,median=median(values),lower95=lower95(values))
    registered=seen==set(range(8)) and len(paired)==8 and not incomplete
    stable=(stats['explored']['lower95']>0 and all(stats[k]['lower95']>=0 for k in ('health','hunger','thirst'))
            if registered else None)
    return dict(independent_completed=len(paired),incomplete=incomplete,statistics=stats,
                eight_sample_joint_criterion=stable)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('runs',nargs='+',type=Path)
    print(json.dumps(summarize(p.parse_args().runs),indent=2))
