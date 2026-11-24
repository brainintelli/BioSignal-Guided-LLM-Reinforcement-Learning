"""Frozen oddball v2 stimulation design: multi-electrode clusters, 300/450/600 mV ladder, confirm block.

Pure functions only; importing this module never contacts hardware.
All parameters stay strictly within MaxOne Pt-Black vendor reference (300-600 mV, 200 us/phase,
positive phase first, inter-pulse interval >= 300 ms).
"""
from decimal import Decimal
import hashlib
import json
import math
import random

SCHEMA = 'mea.oddball.v2'
SITES = ('H', 'W', 'F')             # site k is driven by DAC channel k (3-5 clustered electrodes)
SHAM = 'Z'
RATE = 20000                        # 20 kHz (50 us/frame)
TRIAL_FRAMES = 2 * RATE             # 2.0 s marker-to-marker (>= 300 ms vendor minimum)
CHUNK = 10                          # trials per device sequence
PHASE_FRAMES = 4                    # 200 us per phase
AMPLITUDES_MV = (300, 450, 600)     # 3-level ladder within Pt-Black 300-600 mV reference
CLUSTER_MIN = 3
CLUSTER_TARGET = 5
CLUSTER_RADIUS_UM = 40.0
SITE_MIN_SPACING_UM = 500.0
LOCAL_ROI_INNER_UM = 100.0
LOCAL_ROI_DENSE_UM = 500.0
LOCAL_ROI_OUTER_UM = 1000.0
MIN_LOCAL_CHANNELS = 20

HISTORY = 8
BIN_NAMES = ('low', 'mid', 'high')
BIN_EDGES = ((0, 1), (2, 4), (5, 8))
ROUNDS = 3
BLOCK_TRIALS = 100

SCREEN_PER_SITE = 20                # 20 real per site + 20 sham = 80 trials (160 s) per amplitude
SCREEN_SHAM = 20
SCREEN_REST_SECONDS = 60.0          # quiet recovery between ladder amplitudes

CONFIRM_PER_SITE = 30               # independent confirm block: 30 per site + 30 sham = 120 trials (240 s)
CONFIRM_SHAM = 30

REHEARSAL_TRIALS = 40
PRE_SECONDS = 10.0
POST_SECONDS = 10.0
MAX_REAL_PULSES = 1400              # worst case: 3*60 (screen) + 90 (confirm) + 1080 (main) = 1350 <= 1400
NEW_BLOCK_CUTOFF_S = 88 * 60
STIM_CUTOFF_S = 95 * 60
HARD_STOP_S = 120 * 60
RECOVERY_SECONDS = 600.0
GAP_FRAMES = (int(1.8 * RATE), int(3.0 * RATE))


def digest(value):
    raw = json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)
    return hashlib.sha256(raw.encode('utf-8')).hexdigest()


def _shuffled(seed, name, items):
    items = list(items)
    random.Random(f'{seed}:{name}').shuffle(items)
    return items


def _block(index, name, stage, kind, trials, **extra):
    return dict(index=index, name=name, stage=stage, kind=kind, trials=trials, **extra)


def plan(seed):
    """Build deterministic v2 session plan (rehearsal, 3-level screen, confirm, 3 main rounds)."""
    if type(seed) is not int or not 0 <= seed < 2**63:
        raise ValueError('Integer seed required')
    blocks = []
    index = 2
    screen_trials = [s for s in SITES for _ in range(SCREEN_PER_SITE)] + [None] * SCREEN_SHAM
    for amplitude in AMPLITUDES_MV:
        name = f'screen{amplitude}'
        blocks.append(_block(index, name, 'screen', 'screen', _shuffled(seed, name, screen_trials),
                             amplitude_mv=amplitude))
        index += 1
    confirm_trials = [s for s in SITES for _ in range(CONFIRM_PER_SITE)] + [None] * CONFIRM_SHAM
    blocks.append(_block(index, 'confirm', 'confirm', 'confirm', _shuffled(seed, 'confirm', confirm_trials)))
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
    blocks.insert(0, _block(1, 'rehearsal', 'rehearsal', 'rehearsal',
                            list(first_main['trials'][:REHEARSAL_TRIALS]), mirrors=first_main['name']))
    body = dict(
        schema=SCHEMA,
        seed=seed,
        sites=list(SITES),
        trial_frames=TRIAL_FRAMES,
        chunk=CHUNK,
        phase_frames=PHASE_FRAMES,
        amplitudes_mv=list(AMPLITUDES_MV),
        cluster_min=CLUSTER_MIN,
        cluster_target=CLUSTER_TARGET,
        cluster_radius_um=CLUSTER_RADIUS_UM,
        history=HISTORY,
        bins=[list(e) for e in BIN_EDGES],
        max_real_pulses=MAX_REAL_PULSES,
        blocks=blocks,
    )
    return dict(body, sha256=digest(body))


def check_plan(value):
    if value != plan(value.get('seed')):
        raise ValueError('v2 plan differs from the registered design for its seed')
    return value


def pulse_bits(lsb_mv, target_mv):
    """DAC offset for one phase, rounded down."""
    if type(lsb_mv) not in (int, float) or not math.isfinite(lsb_mv) or lsb_mv <= 0:
        raise ValueError('Positive finite DAC step required')
    if target_mv not in AMPLITUDES_MV:
        raise ValueError('Unregistered v2 amplitude')
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
    n = len(block['trials'])
    return [range(a, min(a + CHUNK, n)) for a in range(0, n, CHUNK)]


def chunk_codes(block, trials, bits):
    """Sequence operations for one chunk; same DAC pulse drives all clustered units of the site."""
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
    return ((len(trials) - 1) * TRIAL_FRAMES + 2 * PHASE_FRAMES) / RATE


def real_pulses(block, trials=None):
    if block['stage'] == 'rehearsal':
        return 0
    trials = range(len(block['trials'])) if trials is None else trials
    return sum(block['trials'][i] is not None for i in trials)


def stimulation_seconds(block):
    return len(block['trials']) * TRIAL_FRAMES / RATE


def recent_counts(sites):
    return [None if s is None else sum(p == s for p in sites[max(0, i - HISTORY):i])
            for i, s in enumerate(sites)]


def bin_of(count):
    for name, (lo, hi) in zip(BIN_NAMES, BIN_EDGES):
        if lo <= count <= hi:
            return name
    raise ValueError('Context count outside the registered history')
