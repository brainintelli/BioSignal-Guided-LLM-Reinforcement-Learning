"""Frozen design of the second oddball protocol: trial plan, device chunks, context counts, budgets.

Pure functions only; importing this module never contacts a device. One trial is
one marker plus, for a real trial, one biphasic positive-first pulse on every
electrode of one site at the same instant.
"""
from decimal import Decimal
import hashlib
import json
import math
import random

SCHEMA = 'mea.oddball.v2'
SITES = ('H', 'W', 'F')            # site k is driven by DAC channel k
SHAM = 'Z'                         # marker label of a zero-output trial
RATE = 20000                       # frames per second; one frame is 50 us
TRIAL_FRAMES = 2 * RATE            # marker-to-marker interval inside a chunk
CHUNK = 10                         # trials per device sequence
PHASE_FRAMES = 4                   # 200 us per phase
LADDER_MV = (300, 450, 600)        # vendor reference range for Pt-black electrodes on MaxOne
MAX_SITE_ELECTRODES = 5
HISTORY = 8                        # trials looked back for the context count
BIN_NAMES = ('low', 'mid', 'high')
BIN_EDGES = ((0, 1), (2, 4), (5, 8))
ROUNDS = 3
SCREEN_PER_SITE = 20               # a screen only decides whether the confirmation block runs
SCREEN_SHAM = 20
CONFIRM_PER_SITE = 30              # fresh trials; the on-site evoked decision uses only these
CONFIRM_SHAM = 30
REHEARSAL_TRIALS = 40              # four chunks, three chunk boundaries, no output
PRE_SECONDS = 10.                  # connected, zero output, before the first marker
POST_SECONDS = 10.                 # after the last pulse, units already off
REST_SECONDS = 60.                 # no recording, no output, after every screen block
MAX_REAL_PULSES = 1400             # site pulses in one session: 3 screens, 1 confirmation, 12 main blocks
NEW_BLOCK_CUTOFF_S = 88 * 60       # no stimulation block starts later than this
STIM_CUTOFF_S = 95 * 60            # no real pulse later than this
HARD_STOP_S = 120 * 60
RECOVERY_SECONDS = 600.
GAP_FRAMES = (int(1.8 * RATE), int(3.0 * RATE))  # allowed marker gap across chunks


def digest(value):
    raw = json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)
    return hashlib.sha256(raw.encode()).hexdigest()


def _shuffled(seed, name, items):
    items = list(items)
    random.Random(f'{seed}:{name}').shuffle(items)
    return items


def _block(index, name, stage, kind, trials, **extra):
    return dict(index=index, name=name, stage=stage, kind=kind, trials=trials, **extra)


def ladder(ceiling_mv):
    if ceiling_mv not in LADDER_MV:
        raise ValueError('Ceiling must be one of %s mV' % (LADDER_MV,))
    return [a for a in LADDER_MV if a <= ceiling_mv]


def plan(seed, ceiling_mv):
    """Every file of the session in execution order.

    Screens run from the lowest amplitude until one qualifies; only the
    confirmation block of that amplitude runs, and the main blocks use it.
    """
    if type(seed) is not int or not 0 <= seed < 2**63:
        raise ValueError('Integer seed required')
    steps = ladder(ceiling_mv)
    blocks = []
    index = 2
    for stage, per_site, sham in (('screen', SCREEN_PER_SITE, SCREEN_SHAM), ('confirm', CONFIRM_PER_SITE, CONFIRM_SHAM)):
        trials = [s for s in SITES for _ in range(per_site)] + [None] * sham
        for amplitude in steps:
            name = f'{stage}{amplitude}'
            blocks.append(_block(index, name, stage, stage, _shuffled(seed, name, trials), amplitude_mv=amplitude))
            index += 1
    for rnd in range(1, ROUNDS + 1):
        for kind in _shuffled(seed, f'round{rnd}', SITES + ('uniform',)):
            if kind == 'uniform':
                trials = [s for s in SITES for _ in range(30)] + [None] * 10
            else:
                trials = [kind] * 70 + [s for s in SITES if s != kind for _ in range(10)] + [None] * 10
            name = f'main_r{rnd}_{kind}'
            blocks.append(_block(index, name, 'main', 'uniform' if kind == 'uniform' else 'biased',
                                 _shuffled(seed, name, trials), round=rnd,
                                 frequent=None if kind == 'uniform' else kind))
            index += 1
    first_main = next(b for b in blocks if b['stage'] == 'main')
    # The rehearsal sends the start of the first main block with every DAC command at neutral.
    blocks.insert(0, _block(1, 'rehearsal', 'rehearsal', 'rehearsal',
                            list(first_main['trials'][:REHEARSAL_TRIALS]), mirrors=first_main['name']))
    body = dict(schema=SCHEMA, seed=seed, ceiling_mv=ceiling_mv, ladder_mv=steps, sites=list(SITES),
                trial_frames=TRIAL_FRAMES, chunk=CHUNK, phase_frames=PHASE_FRAMES, history=HISTORY,
                bins=[list(e) for e in BIN_EDGES], blocks=blocks)
    return dict(body, sha256=digest(body))


def check_plan(value):
    if value != plan(value.get('seed'), value.get('ceiling_mv')):
        raise ValueError('Plan differs from the registered design for its seed and ceiling')
    return value


def pulse_bits(lsb_mv, target_mv):
    """DAC offset for one phase, rounded down like the previous sessions."""
    if type(lsb_mv) not in (int, float) or not math.isfinite(lsb_mv) or lsb_mv <= 0:
        raise ValueError('Positive finite DAC step required')
    if target_mv not in LADDER_MV:
        raise ValueError('Unregistered amplitude')
    bits = int(Decimal(str(target_mv)) / Decimal(str(lsb_mv)))
    if not 1 <= bits <= 511:
        raise ValueError('Target amplitude cannot be represented')
    return bits


def event_id(block, trial):
    return block['index'] * 1000 + trial


def event_properties(block, trial):
    site = block['trials'][trial]
    return dict(blk=block['index'], i=trial, site=site or SHAM)


def chunks(block):
    """Trial index ranges, one per device sequence."""
    n = len(block['trials'])
    return [range(a, min(a + CHUNK, n)) for a in range(0, n, CHUNK)]


def chunk_codes(block, trials, bits):
    """Sequence operations for one chunk. Only delays consume time on the device.

    `bits == 0` keeps every DAC command at the neutral code, which is how the
    rehearsal exercises the full command count without any output.
    """
    if type(bits) is not int or not 0 <= bits <= 511:
        raise ValueError('Invalid DAC offset')
    if (bits == 0) != (block['stage'] == 'rehearsal'):
        raise ValueError('Zero amplitude is reserved for the rehearsal')
    trials = list(trials)
    codes = []
    for position, trial in enumerate(trials):
        site = block['trials'][trial]
        codes.append(('marker', trial))
        used = 0
        if site is not None:
            channel = SITES.index(site)
            codes += [('dac', (channel, 512 - bits)), ('delay', PHASE_FRAMES),
                      ('dac', (channel, 512 + bits)), ('delay', PHASE_FRAMES),
                      ('dac', (channel, 512))]
            used = 2 * PHASE_FRAMES
        if position + 1 < len(trials):
            codes.append(('delay', TRIAL_FRAMES - used))
    return codes


def chunk_seconds(trials):
    """From the first marker of a chunk to the end of its last pulse."""
    return ((len(trials) - 1) * TRIAL_FRAMES + 2 * PHASE_FRAMES) / RATE


def real_pulses(block, trials=None):
    if block['stage'] == 'rehearsal':
        return 0
    trials = range(len(block['trials'])) if trials is None else trials
    return sum(block['trials'][i] is not None for i in trials)


def stimulation_seconds(block):
    return len(block['trials']) * TRIAL_FRAMES / RATE


def recent_counts(sites):
    """For each real trial, how many of the previous HISTORY trials used the same site."""
    return [None if s is None else sum(p == s for p in sites[max(0, i - HISTORY):i])
            for i, s in enumerate(sites)]


def bin_of(count):
    for name, (lo, hi) in zip(BIN_NAMES, BIN_EDGES):
        if lo <= count <= hi:
            return name
    raise ValueError('Context count outside the registered history')


def worst_case_pulses(value):
    """Every screen fails except the last, which is confirmed, and all main blocks run."""
    by_stage = lambda stage: [real_pulses(b) for b in value['blocks'] if b['stage'] == stage]
    return sum(by_stage('screen')) + max(by_stage('confirm')) + sum(by_stage('main'))


def summary(value):
    rows = []
    for b in value['blocks']:
        counts = {s: b['trials'].count(s) for s in SITES}
        rows.append(dict(index=b['index'], name=b['name'], stage=b['stage'], trials=len(b['trials']),
                         sham=b['trials'].count(None), real=real_pulses(b), **counts))
    return rows
