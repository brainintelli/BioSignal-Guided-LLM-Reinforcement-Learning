"""Offline stand-ins: a simulated clock and a rig that writes MaxWell-layout H5 files.

The simulated chip has three sites of three neighbouring electrodes each, far
enough apart that every site has its own 24 readout channels.
"""
import json
from types import SimpleNamespace

import h5py
import numpy as np

from mea_stim_v2.recording import H5Recording
from oddball2 import design, sites as geometry

RATE = design.RATE
CONFIG = SimpleNamespace(threshold_h5_dataset='settings/spike_threshold')
SESSION = 'oddball2-test'
CENTRES = {'H': (20, 20), 'W': (20, 100), 'F': (90, 60)}        # (row, column) on the electrode grid
STEPS = (-16, -8, 0, 8, 16)                                      # readout spacing: 140 um


def _electrode(row, column):
    return row * geometry.COLUMNS + column


ELECTRODES = {s: [_electrode(r, c), _electrode(r, c + 1), _electrode(r + 1, c)] for s, (r, c) in CENTRES.items()}
READOUT_ELECTRODES = {s: [_electrode(r + dr, c + dc) for dr in STEPS for dc in STEPS if dr or dc]
                      for s, (r, c) in CENTRES.items()}
LAYOUT = [e for s in design.SITES for e in ELECTRODES[s]] + [e for s in design.SITES for e in READOUT_ELECTRODES[s]]
CHANNELS = len(LAYOUT)                                           # channel number = position in LAYOUT
CHANNEL_OF = {e: c for c, e in enumerate(LAYOUT)}
SITE_CHANNELS = {s: [CHANNEL_OF[e] for e in ELECTRODES[s]] for s in design.SITES}
READOUT = {s: [CHANNEL_OF[e] for e in READOUT_ELECTRODES[s]] for s in design.SITES}
ROI = dict(sites={s: list(v) for s, v in READOUT.items()})
DIP = 30                                                         # ADC steps below baseline for a raw spike


def xy(channel):
    return geometry.electrode_xy(LAYOUT[channel])


def write_h5(path, first, frames, events, spikes, patches, chip='chip-test', noise=()):
    """events: (frame, id, message dict); spikes: (frames, channels); patches: (channel, start, stop, value);
    noise: (channel, start, values) written before the patches."""
    with h5py.File(path, 'w') as f:
        f['wellplate/id'] = [chip.encode()]
        d = f.create_group('data_store/data0000')
        for key, value in dict(sampling=20000, gain=512, hpf=300, lsb=6.294e-6, spike_threshold=5).items():
            d['settings/' + key] = [value]
        d['settings/mapping'] = np.array([(c, LAYOUT[c], *xy(c)) for c in range(CHANNELS)],
                                         dtype=[('channel', 'i4'), ('electrode', 'i4'), ('x', 'f8'), ('y', 'f8')])
        d['groups/routed/channels'] = np.arange(1024)
        d.create_dataset('groups/routed/frame_nos', data=np.arange(first, first + frames, dtype='u8'),
                         compression='gzip', compression_opts=1, shuffle=True)
        raw = d.create_dataset('groups/routed/raw', (1024, frames), dtype='u2', chunks=(1, 4000), fillvalue=512,
                               compression='gzip', compression_opts=1)
        for channel, start, values in noise:
            raw[channel, start - first:start - first + len(values)] = values
        for channel, start, stop, value in patches:
            raw[channel, start - first:stop - first] = value
        kind = np.dtype([('frameno', 'u8'), ('eventtype', 'u4'), ('eventid', 'u4'), ('eventmessage', 'S200')])
        d['events'] = np.array([(frame, 1, number, json.dumps({k: str(v) for k, v in message.items()}).encode())
                                for frame, number, message in events], dtype=kind)
        rows = np.zeros(len(spikes[0]), dtype=[('frameno', 'u8'), ('channel', 'i4'), ('amplitude', 'f4')])
        rows['frameno'], rows['channel'] = spikes
        d['spikes'] = rows


class Clock:
    def __init__(self):
        self.now = 0.

    def wall(self):
        return 1_800_000_000. + self.now

    def mono(self):
        return 500. + self.now

    def sleep(self, seconds):
        assert seconds >= 0
        self.now += seconds


class FakeRig:
    """Evoked spikes appear on the driven site's readout channels and shrink with recent pulses there.

    gain: evoked spikes per pulse without recent use, or a function of (file name, site).
    raw: also write the spikes of the main blocks into the raw samples, and noise into the warmup,
    so that the offline recount from raw samples can run.
    """
    lsb_mv = 2.981901

    def __init__(self, root, clock, gain=60., habituation=.25, tamper=None, background=None, seed=5, raw=False):
        self.root, self.clock = root, clock
        self.gain = gain if callable(gain) else (lambda name, site: gain)
        self.habituation = habituation
        self.tamper, self.background, self.raw = tamper, background or {}, raw
        self.rng = np.random.default_rng(seed)
        self.site_channels = {s: list(v) for s, v in SITE_CHANNELS.items()}
        self.faulted = False
        self.on = False
        self.current = None
        self.sent, self.unit_log, self.files = [], [], []
        self.cleaned = False

    def frame(self):
        return 7_000_000 + int(round(self.clock.now * RATE))

    def prepare(self):
        self.clock.sleep(12)
        return dict(units=dict(H=[5, 6, 7], W=[9, 10, 11], F=[21, 22, 23]), channels=self.site_channels,
                    lsb_mv=self.lsb_mv)

    def open(self, name):
        assert self.current is None
        (self.root / name).mkdir()
        self.current = dict(name=name, first=self.frame(), sends=[])
        self.clock.sleep(.2)

    def build(self, block, bits):
        self.clock.sleep(1.)
        return [dict(block=block, trials=list(trials), codes=design.chunk_codes(block, trials, bits))
                for trials in design.chunks(block)]

    def units(self, on):
        self.on = on
        self.unit_log.append((self.current and self.current['name'], on))
        self.clock.sleep(.01)

    def send(self, handle, event, gate):
        gate(event)
        assert self.on and self.current
        self.current['sends'].append((self.frame() + 100, handle))
        self.sent.append(dict(event, age=self.clock.now))
        self.clock.sleep(.002)

    def close(self):
        cur, self.current = self.current, None
        self.clock.sleep(.2)
        name, first, frames = cur['name'], cur['first'], self.frame() - cur['first']
        events, pulses = [], []
        for cursor, handle in cur['sends']:
            block = handle['block']
            counts = design.recent_counts(block['trials'])
            level = {}
            for kind, value in handle['codes']:
                if kind == 'marker':
                    trial = value
                    message = dict(design.event_properties(block, trial), session=SESSION)
                    events.append((cursor, design.event_id(block, trial), message))
                elif kind == 'dac':
                    channel, code = value
                    if code != 512 and level.get(channel, 512) == 512:
                        pulses.append(dict(frame=cursor, site=design.SITES[channel], count=counts[trial], trial=trial))
                    level[channel] = code
                else:
                    cursor += value
        patches = [(c, p['frame'], p['frame'] + 800, 1023) for p in pulses for c in self.site_channels[p['site']]]
        rate = 2. * self.background.get(name, 1.)
        n = self.rng.poisson(rate * frames / RATE * CHANNELS)
        spike_frames = [first + self.rng.integers(0, frames, n)]
        spike_channels = [self.rng.integers(0, CHANNELS, n)]
        for p in pulses:
            k = self.rng.poisson(self.gain(name, p['site']) * np.exp(-self.habituation * p['count']))
            spike_frames.append(p['frame'] + self.rng.integers(120, 950, k))
            spike_channels.append(self.rng.choice(READOUT[p['site']], k))
        spike_frames, spike_channels = np.concatenate(spike_frames), np.concatenate(spike_channels)
        noise = []
        if self.raw and name == 'warmup':
            readout = [c for s in design.SITES for c in READOUT[s]]
            noise = [(c, first + int(start * RATE), (512 + np.rint(self.rng.normal(0, 2, 2 * RATE))).astype('u2'))
                     for c in readout for start in (6, 16, 26, 36, 46, 56)]
        if self.raw and name.startswith('main') and events:
            markers = np.asarray([e[0] for e in events])
            nearest = markers[np.clip(np.searchsorted(markers, spike_frames + 1300) - 1, 0, len(markers) - 1)]
            inside = (spike_frames >= nearest - 1300) & (spike_frames < nearest + 1100) & (spike_channels >= 9)
            patches += [(int(c), int(f), int(f) + 4, 512 - DIP) for f, c in zip(spike_frames[inside], spike_channels[inside])]
        if self.tamper:
            self.tamper(name, pulses, patches, events)
        path = self.root / name / (name + '.raw.h5')
        write_h5(path, first, frames, events, (spike_frames, spike_channels), patches, noise=noise)
        self.files.append(name)
        return str(path)

    def read(self, path):
        return H5Recording(path, CONFIG)

    def verify(self, rec):
        if rec.metadata['chip_id'] != 'chip-test':
            raise ValueError('Wrong chip')

    def cleanup(self):
        self.cleaned = True
        return []
