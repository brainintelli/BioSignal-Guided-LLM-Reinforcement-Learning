"""Four recording features; no torch, device API, reward or future context."""
from collections import Counter
import hashlib
import json
import math
from statistics import median

SCHEMA = 'mea.features.state2.v2'
NAMES = ('log_rate', 'active_fraction', 'temporal_concentration', 'spatial_entropy')
FLOORS = (.1, .05, .05, .05)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def transition_context(observation, terminal, encoder):
    """A terminal receipt has no legal policy actions and must bypass policy encoding."""
    return 'terminal:'+digest(observation) if terminal else encoder(observation)


def vector(values):
    if not isinstance(values, (list, tuple)) or len(values) != 4:
        raise ValueError('Four features required')
    if any(type(x) not in (int, float) or not math.isfinite(x) for x in values):
        raise ValueError('Finite feature values required')
    return tuple(float(x) for x in values)


def extract(events, start, end, channels, rate):
    if (type(start) is not int or type(end) is not int or start < 0 or type(rate) is not int
            or rate <= 0 or end - start != rate or rate % 10):
        raise ValueError('Complete one-second window at integer sampling rate required')
    if not isinstance(channels, (tuple, list)) or len(channels) < 2 or len(set(channels)) != len(channels):
        raise ValueError('Unique fixed ROI channels required')
    if any(type(c) is not int or c < 0 for c in channels) or not isinstance(events, (tuple, list)):
        raise ValueError('Missing events or invalid channels')
    allowed = set(channels)
    times = Counter()
    sites = Counter()
    previous = None
    for row in events:
        if not isinstance(row, (tuple, list)) or len(row) != 3:
            raise ValueError('Frame/channel/amplitude rows required')
        frame, channel, amplitude = row
        if (type(frame) is not int or type(channel) is not int or not start <= frame < end
                or channel not in allowed or type(amplitude) not in (int, float) or not math.isfinite(amplitude)):
            raise ValueError('Invalid event coordinates or amplitude')
        key = (frame, channel, amplitude)
        if previous is not None and key < previous:
            raise ValueError('Events must be sorted')
        previous = key
        times[(frame - start) // (rate // 10)] += 1
        sites[channel] += 1
    count = len(events)
    if not count:
        return (0., 0., 0., 0.)
    def entropy(counts):
        return -math.fsum((n / count) * math.log(n / count) for n in counts)
    bounded = lambda x: min(1., max(0., x))
    return (math.log1p(count / len(channels)), len(sites) / len(channels),
            bounded(1 - entropy(times.values()) / math.log(10)),
            bounded(entropy(sites.values()) / math.log(len(channels))))


def fit(rows):
    rows = [vector(row) for row in rows]
    if not rows:
        raise ValueError('Baseline observations required')
    centers = [median(row[j] for row in rows) for j in range(4)]
    scales = [max(1.4826 * median(abs(row[j] - centers[j]) for row in rows), FLOORS[j]) for j in range(4)]
    return dict(schema=SCHEMA, names=list(NAMES), centers=centers, scales=scales,
                baseline_windows=len(rows), floors=list(FLOORS), clip=3., decimals=3)


def transform(values, norm):
    if norm.get('schema') != SCHEMA or norm.get('names') != list(NAMES):
        raise ValueError('Feature schema mismatch')
    centers, scales = vector(norm['centers']), vector(norm['scales'])
    if any(s < floor for s, floor in zip(scales, FLOORS)) or norm['clip'] != 3. or norm['decimals'] != 3:
        raise ValueError('Normalization contract mismatch')
    return tuple(round(min(3., max(-3., (x - c) / s)) / 3, 3)
                 for x, c, s in zip(vector(values), centers, scales))


def validate_snapshot(row, context_id, normalization_id):
    required = {'schema', 'names', 'source', 'association', 'source_pin', 'phase', 'window_index',
                'window_start', 'window_end', 'channel_map_id', 'events_sha256', 'context_id',
                'features', 'normalized', 'normalization_id', 'snapshot_id'}
    if not isinstance(row, dict) or set(row) != required or row['schema'] != SCHEMA or row['names'] != list(NAMES):
        raise ValueError('Exact versioned snapshot required')
    if row['context_id'] != context_id or row['normalization_id'] != normalization_id:
        raise ValueError('Snapshot context/normalization mismatch')
    if row['source'] != 'recording' or row['association'] != 'historical_no_game_causality':
        raise ValueError('Only historical recording accepted')
    if row['snapshot_id'] != digest({k: v for k, v in row.items() if k != 'snapshot_id'}):
        raise ValueError('Snapshot digest mismatch')
    vector(row['features'])
    values = vector(row['normalized'])
    if any(abs(v) > 1 or round(v, 3) != v for v in values):
        raise ValueError('Normalized values out of contract')
    return values
