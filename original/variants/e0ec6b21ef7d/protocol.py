"""Pure protocol shared by the electrode controller and the two GPU workers."""
import math
import random
import re
from datetime import datetime
from mea_binary_r2.encoding import Intent

from mea_state2_r2.features import digest, transform, vector
from mea_state2_r2.protocol import BETA, GAMMA, PROFILE

ARMS = ('ordered', 'shuffled')
HOSTS = ('192.0.2.16', '192.0.2.23')
BLOCK = 4
STEPS = 16
RATE = 20000
PROTOCOL = 'mea.live.paired.v1'


def binding(b):
    for key in ('chip_id', 'organoid_id', 'session_id', 'device_hostname'):
        if not isinstance(b.get(key), str) or not re.fullmatch(r'[A-Za-z0-9_-]{1,64}', b[key]):
            raise ValueError('Missing or invalid ' + key)
    if b['device_model'] != 'MaxOne' or b['acquisition_host'] not in ('192.0.2.21', '192.0.2.101'):
        raise ValueError('Acquisition machine or chip model unbound')
    t = datetime.fromisoformat(b['started_at'])
    if t.utcoffset() is None:
        raise ValueError('Original biological start requires timezone')
    if type(b['replicate']) is not int or b['replicate'] < 0:
        raise ValueError('Invalid replicate')
    return t.timestamp()


def host_map(replicate):
    return dict(zip(ARMS, HOSTS if replicate % 2 == 0 else reversed(HOSTS)))


def physical_order(index, replicate):
    return ARMS if (index + replicate) % 2 == 0 else tuple(reversed(ARMS))


def window(bid, start, features, origin):
    row = dict(binding=bid, start=start, end=start+RATE, features=list(vector(features)), origin=origin)
    return dict(row, id=digest(row))


def check_window(w, bid):
    if (w['binding'] != bid or w['id'] != digest({k:v for k,v in w.items() if k != 'id'})
            or type(w['start']) is not int or w['start'] < 0 or w['end'] != w['start']+RATE):
        raise ValueError('Window provenance, integrity or coverage differs')
    vector(w['features'])


class Buffer:
    """Only expose a previously completed block, with identical block latency."""
    def __init__(self, arm, seed, initial, bid):
        self.arm, self.rng, self.bid = arm, random.Random(seed), bid
        if arm not in ARMS or len(initial) != BLOCK:
            raise ValueError('Four measured priming windows required')
        self.pending, self.available, self.last = [], [], -1
        for w in initial:
            self.push(w)

    def push(self, w):
        check_window(w, self.bid)
        if w['start'] < self.last:
            raise ValueError('Reused or overlapping acquisition')
        self.last = w['end']
        self.pending.append(w)
        if len(self.pending) == BLOCK:
            if self.available:
                raise ValueError('Unconsumed previous block')
            order = list(range(BLOCK))
            if self.arm == 'shuffled':
                while any(i == j for i,j in enumerate(order)):
                    self.rng.shuffle(order)
            self.available = [self.pending[i] for i in order]
            self.pending = []

    def peek(self):
        if not self.available:
            raise ValueError('Incomplete previous block')
        return self.available[0]

    def take(self):
        self.peek()
        return self.available.pop(0)


def transaction(bid, arm, phase, episode, index, context, intent):
    row = dict(binding=bid, arm=arm, phase=phase, episode=episode, index=index,
               context=context, site=intent.site, pulses=intent.pulses, profile=PROFILE.digest)
    return dict(row, id=digest(row))


def scheduled_intent(encoder, reserves, terminal, index):
    intent=encoder.decide(reserves,terminal)
    if intent.pulses and index%3!=2:
        return Intent(None,0,True,())
    return intent


def reward(r1, ticks, terminal, req, receipt):
    if not math.isfinite(r1) or type(ticks) is not int or ticks < 1:
        raise ValueError('Invalid native R1 or tick count')
    if terminal:
        if receipt is not None or req is not None:
            raise ValueError('Terminal must have no output')
        return dict(r1=r1, r2=None, contribution=0., total=r1)
    if (req['id'] != digest({k:v for k,v in req.items() if k != 'id'})
            or receipt['request'] != req or receipt['status'] != 'accepted'
            or receipt['id'] != digest({k:v for k,v in receipt.items() if k != 'id'})):
        raise ValueError('Unmatched or unconfirmed delivery')
    check_window(receipt['window'], req['binding'])
    n = req['pulses']
    anchor = receipt['anchor']
    if (n not in (0,3) or receipt['frames'] != [anchor+200*i for i in range(n)]
            or receipt['window']['start'] != anchor+240000
            or receipt['window']['origin'] != req['id']
            or receipt['observed'] != ('output' if n else 'quiet')):
        raise ValueError('Output or feature timing differs')
    sign = -1 if n else 1
    contribution = BETA * sign * math.expm1(ticks*math.log(GAMMA))/math.expm1(math.log(GAMMA))
    return dict(r1=r1, r2=sign, contribution=contribution, total=r1+contribution)


def snapshot(w, config, context, key):
    check_window(w, config['binding_sha256'])
    row = dict(schema=PROTOCOL, request_id=key, context_id=context, window=w,
               binding_sha256=w['binding'], normalization_id=config['normalization_id'],
               normalized=list(transform(w['features'], config['normalization'])))
    return dict(row, snapshot_id=digest(row))


def metrics(rows):
    if not rows or (len(rows) != STEPS and not rows[-1]['terminal']):
        raise ValueError('Incomplete evaluation horizon')
    first = rows[0]['transition']['before']
    last = rows[-1]['transition']['after']
    stats = dict(explored=last['visited_count']-first['visited_count'], decisions=len(rows),
                 death=int(rows[-1]['terminal']), ticks=sum(row['duration'] for row in rows))
    for field in ('health', 'hunger', 'thirst'):
        stats[field] = last[field]
        stats[field+'_min'] = min([first[field]]+[r['transition']['after'][field] for r in rows])
    return stats
