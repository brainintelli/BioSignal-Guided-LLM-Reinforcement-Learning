"""Spike counts from raw samples: blank the pulse, filter causally, threshold.

The device's own detector filters with a symmetric kernel, so a pulse leaves
false spikes from 1 ms before the marker to 2.5 ms after it. The registered
gates are therefore recounted here. Real and sham trials go through exactly
the same steps:

1. Samples from -0.25 to +2.5 ms around the marker, and any sample at an ADC
   rail with 0.25 ms before and 3 ms after it, are replaced by a straight line.
2. A causal 4th-order Butterworth band-pass (300-3000 Hz) is applied, so
   nothing after the marker can leak to before it.
3. A spike is a local minimum below -5 sigma, at least 2 ms after the previous
   one on that channel. Sigma is the robust noise of the quiet warmup file.
   (The filter rebounds 1.2 ms after a large spike with a fifth of its size;
   a shorter dead time would count such spikes twice.)
4. Nothing is counted from -1 to +5 ms, nor within 2.5 ms after a blanked rail.
"""
import numpy as np

from . import analysis

LOW_HZ, HIGH_HZ, ORDER = 300., 3000., 4
BLANK = (-5, 50)           # frames around the marker replaced before filtering
RAIL = (2, 1021)           # raw values at or beyond these are saturated
RAIL_PAD = (5, 60)         # frames blanked before and after a saturated sample
QUIET = (-20, 100)         # frames around the marker in which nothing is counted
SETTLE = 50                # frames after a blanked rail in which nothing is counted
REFRACTORY = 40            # frames between two spikes on one channel
SIGMAS = 5.
LEAD = 1200                # 60 ms of signal before the marker: baseline and filter settling
EDGE = 20                  # samples next to a blanked run that anchor the straight line
NOISE_STARTS = (6., 16., 26., 36., 46., 56.)
NOISE_SECONDS = 2.
NOISE_SKIP = 400


def band_pass(fs, low=LOW_HZ, high=HIGH_HZ, order=ORDER):
    """Second-order sections [b0, b1, b2, a1, a2] of a Butterworth band-pass, unit gain mid-band."""
    edges = 2 * fs * np.tan(np.pi * np.asarray([low, high]) / fs)
    width, centre = edges[1] - edges[0], np.sqrt(edges[0] * edges[1])
    k = np.arange(order)
    half = np.exp(1j * np.pi * (2 * k + order + 1) / (2 * order)) * width / 2
    root = np.sqrt(half * half - centre * centre)
    analog = np.concatenate([half + root, half - root])
    poles = (2 * fs + analog) / (2 * fs - analog)
    poles = poles[poles.imag > 0]
    if len(poles) != order:
        raise ValueError('Band-pass design did not yield conjugate pole pairs')
    sos = np.asarray([[1., 0., -1., -2 * p.real, abs(p) ** 2] for p in poles])
    z = np.exp(1j * 2 * np.arctan(centre / (2 * fs)))
    gain = np.prod([abs((b0 + b1 / z + b2 / z ** 2) / (1 + a1 / z + a2 / z ** 2)) for b0, b1, b2, a1, a2 in sos])
    sos[0, :3] /= gain
    return sos


def causal_filter(sos, x):
    """Filter along the last axis from a resting state. Uses scipy when it is installed."""
    x = np.asarray(x, dtype=float)
    try:
        from scipy.signal import sosfilt
    except ImportError:
        return convolved(sos, x)
    return sosfilt(np.column_stack([sos[:, :3], np.ones(len(sos)), sos[:, 3:]]), x, axis=-1)


def recursive(sos, x):
    """The filter's difference equations, one sample at a time."""
    y = np.asarray(x, dtype=float)
    for b0, b1, b2, a1, a2 in sos:
        out = np.empty_like(y)
        z1 = np.zeros(y.shape[:-1])
        z2 = np.zeros(y.shape[:-1])
        for t in range(y.shape[-1]):
            sample = y[..., t]
            value = b0 * sample + z1
            z1 = b1 * sample - a1 * value + z2
            z2 = b2 * sample - a2 * value
            out[..., t] = value
        y = out
    return y


_impulse = {}


def convolved(sos, x):
    """The same output without scipy: a signal of n samples sees only the first n samples of the impulse response."""
    n = x.shape[-1]
    key = (sos.tobytes(), n)
    if key not in _impulse:
        unit = np.zeros(n)
        unit[0] = 1.
        if len(_impulse) > 4:
            _impulse.clear()
        _impulse[key] = np.fft.rfft(recursive(sos, unit), 2 * n)
    return np.fft.irfft(np.fft.rfft(x, 2 * n, axis=-1) * _impulse[key], 2 * n, axis=-1)[..., :n]


def thresholds(rec, channels, sos):
    """Detection threshold per channel in ADC steps: SIGMAS times the robust noise of a quiet file."""
    starts = [t for t in NOISE_STARTS if t + NOISE_SECONDS <= rec.duration]
    if len(starts) < 3:
        raise ValueError('Quiet recording too short for a noise estimate')
    rows = []
    for start in starts:
        raw = rec.sample(channels, start, NOISE_SECONDS).astype(float)
        filtered = causal_filter(sos, raw - np.median(raw, axis=1, keepdims=True))[:, NOISE_SKIP:]
        rows.append(np.median(np.abs(filtered), axis=1) / .6745)
    sigma = np.median(rows, axis=0)
    if not np.all(sigma > 0):
        raise ValueError('A readout channel shows no noise in the quiet recording')
    return {int(c): float(SIGMAS * s) for c, s in zip(channels, sigma)}


def _runs(mask):
    edges = np.flatnonzero(np.diff(np.concatenate(([False], mask, [False])).astype(np.int8)))
    return list(zip(edges[::2], edges[1::2]))


def _bridge(row, start, stop):
    left = np.median(row[max(0, start - EDGE):start]) if start else row[min(stop, len(row) - 1)]
    right = np.median(row[stop:stop + EDGE]) if stop < len(row) else left
    row[start:stop] = left + (right - left) * np.arange(stop - start) / (stop - start)


def trial_spikes(raw, threshold, sos, centre):
    """channels x frames of spike flags for one trial; `centre` is the marker's column in `raw`."""
    work = raw.astype(float)
    frames = work.shape[1]
    blank = np.zeros(work.shape, dtype=bool)
    blank[:, centre + BLANK[0]:centre + BLANK[1]] = True
    counted = np.ones(work.shape, dtype=bool)
    counted[:, centre + QUIET[0]:centre + QUIET[1]] = False
    saturated = (raw <= RAIL[0]) | (raw >= RAIL[1])
    railed = np.flatnonzero(saturated.any(axis=1))
    for c in railed:
        for i in np.flatnonzero(saturated[c]):
            blank[c, max(0, i - RAIL_PAD[0]):i + RAIL_PAD[1] + 1] = True
    plain = np.setdiff1d(np.arange(len(work)), railed)
    start, stop = centre + BLANK[0], centre + BLANK[1]
    if len(plain):
        left = np.median(work[plain, start - EDGE:start], axis=1)
        right = np.median(work[plain, stop:stop + EDGE], axis=1)
        work[plain, start:stop] = left[:, None] + (right - left)[:, None] * np.arange(stop - start) / (stop - start)
    for c in railed:
        for a, b in _runs(blank[c]):
            _bridge(work[c], a, b)
            counted[c, a:min(frames, b + SETTLE)] = False
    work -= np.median(work[:, :centre + QUIET[0]], axis=1, keepdims=True)
    y = causal_filter(sos, work)
    spikes = np.zeros(work.shape, dtype=bool)
    spikes[:, 1:-1] = ((y[:, 1:-1] < -threshold[:, None]) & (y[:, 1:-1] <= y[:, :-2]) & (y[:, 1:-1] < y[:, 2:])
                       & counted[:, 1:-1])
    for c in np.flatnonzero(spikes.sum(axis=1) > 1):
        last = -REFRACTORY
        for i in np.flatnonzero(spikes[c]):
            if i - last < REFRACTORY:
                spikes[c, i] = False
            else:
                last = i
    return spikes


def window_counts(rec, frames, channels, threshold, sos, window=analysis.RESPONSE):
    """trials x channels: spikes in `window` (frames after the marker) on every channel."""
    out = np.zeros((len(frames), len(channels)), dtype=int)
    for i, frame in enumerate(frames):
        raw = rec.sample(channels, (frame - LEAD) / rec.fs, (LEAD + window[1] + EDGE) / rec.fs)
        spikes = trial_spikes(raw, threshold, sos, LEAD)
        out[i] = spikes[:, LEAD + window[0]:LEAD + window[1]].sum(axis=1)
    return out


def recount(rec, qc, roi, threshold_by_channel, sos):
    """Counts from raw samples for a block's trials: one dict per trial, like `r` of the on-site rows."""
    channels = analysis.union(roi)
    threshold = np.asarray([threshold_by_channel[c] for c in channels])
    counts = window_counts(rec, [t['frame'] for t in qc['trials']], channels, threshold, sos)
    column = {c: i for i, c in enumerate(channels)}
    index = {s: [column[c] for c in roi['sites'][s]] for s in roi['sites']}
    return [{s: int(row[cols].sum()) for s, cols in index.items()} for row in counts]
