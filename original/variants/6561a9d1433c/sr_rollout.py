"""Terminal-safe variable-duration returns and finite offline window progression."""
import math
from agent.rl_contract import gae
from batch_boundary import bootstrap
from sr_protocol import PROTOCOL

def audit(rows,boot):
    carry=0.
    for i in reversed(range(len(rows))):
        r=rows[i]
        if type(r['duration']) is not int or r['duration']<1 or r['boundary']!=(r['terminal'] or i==len(rows)-1):raise ValueError('duration/boundary')
        nextv=0. if r['terminal'] else rows[i+1]['value'] if i+1<len(rows) else boot
        gamma=PROTOCOL['gamma']**r['duration'];carry=r['reward']+gamma*nextv-r['value']+gamma*PROTOCOL['gae_lambda']**r['duration']*(not r['boundary'])*carry
        if not math.isfinite(carry) or abs(carry-r['advantage'])>1e-10 or abs(carry+r['value']-r['return'])>1e-10:raise ValueError('independent GAE mismatch')
    return True

def finish(rows,next_value):
    value,pair=bootstrap(rows,None,lambda _:next_value(),lambda p:p[0])
    rows[-1]['boundary']=True;nv=[0. if r['terminal'] else rows[i+1]['value'] if i+1<len(rows) else value for i,r in enumerate(rows)]
    adv,ret=gae([r['reward'] for r in rows],[r['value'] for r in rows],nv,[r['duration'] for r in rows],
        [r['terminal'] for r in rows],[r['boundary'] for r in rows],gamma_tick=PROTOCOL['gamma'],lambda_tick=PROTOCOL['gae_lambda'])
    for r,a,v,n in zip(rows,adv,ret,nv):r.update(advantage=a,**{'return':v},next_value=n)
    audit(rows,value);return dict(value=value,prepared=None if pair is None else pair[1],terminal=rows[-1]['terminal'])

def validate_sequence(rows,order,*,training):
    if not rows or len(order)!=len(set(order)) or len(rows)>len(order)-1 or training and len(rows)!=len(order)-1:raise ValueError('length/wrap')
    previous=None
    for i,r in enumerate(rows):
        if r['index']!=i or r['window']!=order[i]:raise ValueError('window identity')
        if previous is not None and r['seed']!=(previous['reset_seed'] if previous['terminal'] else previous['seed']):raise ValueError('world continuity')
        if r['terminal']:
            if training:
                if r['reset_seed']!=r['seed']+1:raise ValueError('native reset seed')
            elif r['reset_seed'] is not None or i!=len(rows)-1:raise ValueError('evaluation continues after death')
        elif r['reset_seed'] is not None:raise ValueError('unrequested reset')
        previous=r
    if not training and len(rows)<len(order)-1 and not rows[-1]['terminal']:raise ValueError('early nonterminal truncation')
    return True
