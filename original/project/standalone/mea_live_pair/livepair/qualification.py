"""Three-site pilot. Only measured state2 vectors enter the distance classifier."""
from itertools import permutations, product
import random

import numpy as np
from mea_state2_r2.features import digest, transform
from .protocol import RATE, check_window, reward

SITES='HWF'
BLOCKS=3
VERSION='mea.site_response.v1'


def order_for(binding_id):
    rng=random.Random(digest([VERSION,binding_id]))
    return [rng.sample(list(SITES),len(SITES)) for _ in range(BLOCKS)]


def _predictions(x,y):
    predicted=np.full(y.shape,3,dtype=int)
    for test in range(BLOCKS):
        train=np.arange(BLOCKS)!=test
        centers=np.asarray([x[train][y[train]==c].mean(axis=0) for c in range(3)])
        distances=np.square(x[test,:,None,:]-centers).sum(axis=-1)
        winners=np.isclose(distances,distances.min(axis=1,keepdims=True),atol=1e-12,rtol=0.)
        unique=winners.sum(axis=1)==1
        predicted[test,unique]=winners[unique].argmax(axis=1)
    return predicted


def classify_blocks(labels,features):
    if len(labels)!=BLOCKS or any(sorted(block)!=sorted(SITES) for block in labels):
        raise ValueError('Three complete independently shuffled blocks of H/W/F required')
    x=np.asarray(features,dtype=float)
    if x.shape!=(BLOCKS,3,4) or not np.isfinite(x).all() or (np.abs(x)>1).any():
        raise ValueError('Nine normalized, finite four-feature state2 vectors required')
    y=np.asarray([[SITES.index(s) for s in block] for block in labels])
    pred=_predictions(x,y)
    matrix=np.zeros((3,4),dtype=int)
    np.add.at(matrix,(y.ravel(),pred.ravel()),1)
    correct=int(np.trace(matrix))
    ge=0
    # Enumerate the same within-block assignments used by the acquisition design.
    for perm in product(permutations(range(3)),repeat=BLOCKS):
        shuffled=np.asarray(perm)
        ge+=int(np.count_nonzero(_predictions(x,shuffled)==shuffled)>=correct)
    return dict(correct=correct,accuracy=correct/9,per_site_correct=matrix.diagonal().tolist(),
                confusion=matrix.tolist(),rows=list(SITES),columns=[*SITES,'undecided'],
                permutations=216,p_exact=ge/216)


def assess(labels,post,pre):
    observed=classify_blocks(labels,post)
    control=classify_blocks(labels,pre)
    return dict(passed=(observed['correct']>=7 and min(observed['per_site_correct'])>=2
                and observed['p_exact']<=.05 and control['p_exact']>.05),
                post=observed,pre=control)


def trial_row(receipt,req,norm):
    reward(0.,1,False,req,receipt)
    pre=receipt['pre_window']
    check_window(pre,req['binding'])
    if (pre['origin']!=req['id'] or pre['start']!=receipt['anchor']-3*RATE
            or req['pulses']!=3 or req['site'] not in SITES):
        raise ValueError('Qualification pre-window or site mismatch')
    return dict(block=req['episode'],site=req['site'],index=req['index'],request_id=req['id'],
                receipt_id=receipt['id'],anchor=receipt['anchor'],pre_id=pre['id'],post_id=receipt['window']['id'],
                post=list(transform(receipt['window']['features'],norm)),pre=list(transform(pre['features'],norm)))


def analysis(binding_id,rows):
    schedule=order_for(binding_id)
    expected=[(i,site,3*i+j) for i,block in enumerate(schedule) for j,site in enumerate(block)]
    if [(r['block'],r['site'],r['index']) for r in rows]!=expected:
        raise ValueError('Missing or out-of-order qualification trials')
    for key in ('request_id','receipt_id','pre_id','post_id'):
        if len({r[key] for r in rows})!=9:
            raise ValueError('Reused '+key)
    if any(b['anchor']-a['anchor']<20*RATE for a,b in zip(rows,rows[1:])):
        raise ValueError('Trial coverage overlap or short hardware interval')
    def by_block(key):
        return [[r[key] for r in rows[k:k+3]] for k in range(0,9,3)]
    return assess(schedule,by_block('post'),by_block('pre'))


def require_pass(report,binding_id):
    if report.get('schema')!=VERSION or report.get('binding')!=binding_id or report.get('passed') is not True:
        raise ValueError('Three-site response qualification did not pass')
    quiet=report.get('quiet_receipt')
    if (not isinstance(quiet,str) or len(quiet)!=64 or type(report.get('attempts')) is not int
            or not 9<=report['attempts']<=30 or any(r['receipt_id']==quiet for r in report['trials'])):
        raise ValueError('Confirmed quiet receipt and complete physical attempt count required')
    measured=analysis(binding_id,report['trials'])
    if (not measured['passed'] or report.get('analysis')!=measured or report.get('order')!=order_for(binding_id)):
        raise ValueError('Qualification differs from registered analysis')
