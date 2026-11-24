"""Action-contingent feedback. Timings are onset offsets in 20 kHz device frames."""
import math
import random
from mea_binary_r2.encoding import pulse_shape

from mea_state2_r2.features import digest
from mea_state2_r2.protocol import PROFILE

MODE = 'game.predictability.v1'
FLOOR = 85
SITES = ('H', 'W', 'F')
FIELDS = ('health', 'thirst', 'hunger')


def outcome(transition):
    before, after, trace = (transition[k] for k in ('before', 'after', 'trace'))
    ticks = transition['elapsed_ticks']
    if (type(ticks) is not int or ticks < 1 or len(trace) != ticks or
            [r['tick'] for r in trace] != list(range(before['tick'] + 1, after['tick'] + 1))):
        raise ValueError('Incomplete action trace')
    delta = after['visited_count'] - before['visited_count']
    if type(delta) is not int or delta < 0 or delta != transition['new_discovered']:
        raise ValueError('Exploration counters disagree')
    minimum = {}
    for k in FIELDS:
        values = [before[k], *(r[k] for r in trace), after[k]]
        if any(type(x) not in (float, int) or not math.isfinite(x) or not 0 <= x <= 100 for x in values):
            raise ValueError('Invalid game reserve ' + k)
        minimum[k] = min(values)
    return dict(new_cells=delta, minimum=minimum, floor=FLOOR,
                correct=delta > 0 and min(minimum.values()) >= FLOOR and not after['dead'])


def pattern(binding, phase, episode, index, correct):
    if index % 3 != 2:
        return 'quiet', []
    if correct:
        return 'regular', [['H', 0], ['H', 4000], ['H', 8000]]
    # Equal count and duration. Randomize the intermediate onset and visit
    # three distinct qualified sites. Both arms share the randomization rule.
    rng = random.Random(digest([MODE, binding, phase, episode, index]))
    sites = rng.sample(SITES, 3)
    middle = 4000 + rng.choice((-1, 1)) * rng.randint(1000, 2000)
    return 'irregular', [[s, t] for s, t in zip(sites, (0, middle, 8000))]


def request(binding, arm, phase, episode, index, context, transition):
    result = outcome(transition)
    kind, events = pattern(binding, phase, episode, index, result['correct'])
    row = dict(binding=binding, arm=arm, phase=phase, episode=episode, index=index,
               context=context, profile=PROFILE.digest, encoding=MODE, outcome=result,
               pattern=kind, events=events, pulses=len(events))
    return dict(row, id=digest(row))


def verify_request(req):
    o = req['outcome']
    if (req.get('encoding') != MODE or req['id'] != digest({k:v for k,v in req.items() if k!='id'})
            or req['profile'] != PROFILE.digest or req['phase'] not in ('train','eval')
            or req['arm'] not in ('ordered','shuffled') or type(req['index']) is not int
            or not 0 <= req['index'] < 16 or type(req['episode']) is not int or req['episode'] < 0
            or type(o['correct']) is not bool or type(o['new_cells']) is not int or o['new_cells'] < 0
            or o['floor'] != FLOOR or set(o['minimum']) != set(FIELDS)
            or any(type(x) not in (int,float) or not 0 <= x <= 100 for x in o['minimum'].values())
            or o['correct'] != (o['new_cells'] > 0 and min(o['minimum'].values()) >= FLOOR)):
        raise ValueError('Invalid nonterminal feedback request')
    kind, events = pattern(req['binding'], req['phase'], req['episode'], req['index'], o['correct'])
    if req['pattern'] != kind or req['events'] != events or req['pulses'] != len(events):
        raise ValueError('Feedback schedule differs from the registered game outcome')
    return events


def receipt_sign(req, receipt):
    events = verify_request(req)
    expected = [[s, receipt['anchor']+offset] for s,offset in events]
    if (receipt.get('verified_site_frames') != expected or receipt['frames'] != [f for _,f in expected]
            or receipt['observed'] != req['pattern']):
        raise ValueError('Measured feedback class, site or time differs')
    return {'regular': 1, 'irregular': -1, 'quiet': 0}[req['pattern']]


def sequence(req, lsb_mv):
    """Route a finite registered pulse shape, leaving all DACs neutral between events."""
    events = verify_request(req)
    if not events:
        return []
    phase = PROFILE.phase_us // 50
    pulse = pulse_shape(lsb_mv, PROFILE)
    result = []
    for i, (site, offset) in enumerate(events):
        if i:
            result.extend([('delay',offset-events[i-1][1]-2*phase), ('event_index', i)])
        dac = SITES.index(site)
        for kind,value in pulse:
            result.append(('site_dac',(dac,value)) if kind=='dac' else (kind,value))
    return result
