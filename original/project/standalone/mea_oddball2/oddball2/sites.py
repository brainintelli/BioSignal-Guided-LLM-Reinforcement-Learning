"""Stimulation sites inside active tissue, and the electrodes driven together at each site.

The first protocol took the three most distant stable electrodes, which put one
site where no active channel lay within 500 um. Here a site centre needs active
readout channels around it, and the three centres stay far enough apart that
their pulses remain distinguishable.
"""
from itertools import combinations
import math

PITCH_UM = 17.5
COLUMNS, ROWS = 220, 120           # MaxOne electrode grid; electrode = row * COLUMNS + column
NEAR_UM = (100., 500.)             # annulus in which active channels are counted
MIN_NEAR = 20
MIN_SEPARATION_UM = 500.
THIN_UM = 100.                     # candidate centres closer than this are represented by the better one
NEIGHBOUR_UM = 40.                 # electrodes this close to the centre may be driven with it
MIN_ACTIVE_HZ = .05


def electrode_xy(electrode):
    if type(electrode) is not int or not 0 <= electrode < COLUMNS * ROWS:
        raise ValueError('Electrode outside the MaxOne grid')
    row, column = divmod(electrode, COLUMNS)
    return column * PITCH_UM, row * PITCH_UM


def neighbours(electrode):
    """Other electrodes within NEIGHBOUR_UM of this one, nearest first."""
    row, column = divmod(electrode, COLUMNS)
    reach = int(NEIGHBOUR_UM // PITCH_UM)
    found = []
    for dr in range(-reach, reach + 1):
        for dc in range(-reach, reach + 1):
            distance = math.hypot(dr, dc) * PITCH_UM
            if (dr or dc) and distance <= NEIGHBOUR_UM and 0 <= row + dr < ROWS and 0 <= column + dc < COLUMNS:
                found.append((distance, (row + dr) * COLUMNS + column + dc))
    return [e for _, e in sorted(found)]


def choose(rows, active_xy):
    """Three centres among the stable active electrodes `rows` of the quiet baseline.

    `active_xy` holds the position of every active channel of that recording.
    The triple maximises the smallest count of active channels around a centre,
    then the total count, then the smallest distance between centres.
    """
    scored = []
    for row in rows:
        near = sum(NEAR_UM[0] <= math.hypot(row['x_um'] - x, row['y_um'] - y) <= NEAR_UM[1] for x, y in active_xy)
        if near >= MIN_NEAR:
            scored.append(dict(row, near_active=near))
    scored.sort(key=lambda r: (-r['near_active'], -r['min_spikes_per_20s'], r['electrode']))
    apart = lambda a, b: math.hypot(a['x_um'] - b['x_um'], a['y_um'] - b['y_um'])
    kept = []
    for row in scored:
        if all(apart(row, other) >= THIN_UM for other in kept):
            kept.append(row)
    best = None
    for triple in combinations(kept, 3):
        gaps = [apart(a, b) for a, b in combinations(triple, 2)]
        if min(gaps) < MIN_SEPARATION_UM:
            continue
        counts = [r['near_active'] for r in triple]
        score = (min(counts), sum(counts), min(gaps))
        if best is None or score > best[0]:
            best = (score, triple)
    if best is None:
        raise ValueError('Fewer than three sites with %d active channels within %d-%d um and %d um apart'
                         % (MIN_NEAR, *NEAR_UM, MIN_SEPARATION_UM))
    chosen = dict(zip('HWF', best[1]))        # `kept` is ordered, so H has the most active surroundings
    return dict(sites=chosen, qualified_centres=len(scored), considered_centres=len(kept),
                distances_um={a + '-' + b: apart(chosen[a], chosen[b]) for a, b in combinations('HWF', 2)},
                rule=dict(near_um=list(NEAR_UM), min_near=MIN_NEAR, min_separation_um=MIN_SEPARATION_UM,
                          thin_um=THIN_UM, neighbour_um=NEIGHBOUR_UM, min_active_hz=MIN_ACTIVE_HZ))


def active_electrodes(rec, bad_channels, seconds=60.):
    """Electrode -> position of the channels that fired at MIN_ACTIVE_HZ or more in the quiet baseline.

    Also checks that this chip numbers its electrodes like the MaxOne grid, because the
    neighbours of a centre are found from electrode numbers alone.
    """
    import numpy as np
    xy = rec.channel_xy_um
    if rec.duration < seconds or xy is None:
        raise ValueError('One minute of quiet baseline with physical coordinates required')
    for channel, electrode in rec.mapping:
        wanted = electrode_xy(electrode)
        if max(abs(xy[channel][0] - wanted[0]), abs(xy[channel][1] - wanted[1])) > .5:
            raise ValueError('Electrode %d is not where the MaxOne grid puts it' % electrode)
    counts = np.bincount(rec.spike_channels[rec.spike_times < seconds], minlength=1024)
    bad = set(bad_channels)
    return {e: xy[c] for c, e in rec.mapping if c not in bad and counts[c] >= MIN_ACTIVE_HZ * seconds}


def surroundings(centres, active):
    """Per site: active electrodes in the counting annulus of its centre and not next to any centre.

    `centres` maps site -> (x, y); `active` maps electrode -> (x, y).
    """
    out = {s: [] for s in centres}
    for electrode, (x, y) in active.items():
        distance = {s: math.hypot(x - cx, y - cy) for s, (cx, cy) in centres.items()}
        if min(distance.values()) < NEAR_UM[0]:
            continue
        for s in centres:
            if distance[s] <= NEAR_UM[1]:
                out[s].append(electrode)
    return out
