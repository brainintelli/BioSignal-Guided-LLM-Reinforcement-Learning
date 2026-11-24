"""Offline stand-ins: a simulated clock and a rig that writes MaxWell-layout H5 files."""
import json
from types import SimpleNamespace

import h5py
import numpy as np

from mea_stim_v2.recording import H5Recording
from oddball import design

RATE = design.RATE
CONFIG = SimpleNamespace(threshold_h5_dataset='settings/spike_threshold')
SESSION = 'oddball-test'
CHANNELS = 80
SITE_CHANNELS = {'H': 10, 'W': 20, 'F': 30}


def xy(channel):
    return ((channel % 10) * 150., (channel // 10) * 150.)


def write_h5(path, first, frames, events, spikes, patches, chip='chip-test'):
    """events: (frame, id, message dict); spikes: (frames, channels); patches: (channel, start, stop, value)."""
    with h5py.File(path, 'w') as f:
        f['wellplate/id'] = [chip.encode()]
        d = f.create_group('data_store/data0000')
        for key, value in dict(sampling=20000, gain=512, hpf=300, lsb=6.294e-6, spike_threshold=5).items():
            d['settings/' + key] = [value]
        d['settings/mapping'] = np.array([(c, c + 1000, *xy(c)) for c in range(CHANNELS)],
                                         dtype=[('channel', 'i4'), ('electrode', 'i4'), ('x', 'f8'), ('y', 'f8')])
        d['groups/routed/channels'] = np.arange(1024)
        d.create_dataset('groups/routed/frame_nos', data=np.arange(first, first + frames, dtype='u8'),
                         compression='gzip', compression_opts=1, shuffle=True)
        raw = d.create_dataset('groups/routed/raw', (1024, frames), dtype='u2', chunks=(1, 4000), fillvalue=512,
                               compression='gzip', compression_opts=1)
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
    """Evoked spikes shrink with the number of recent pulses at the same site."""
    lsb_mv = 2.981901

    def __init__(self, root, clock, gain=60., habituation=.25, tamper=None, background=None, seed=5):
        self.root, self.clock = root, clock
        self.gain, self.habituation = gain, habituation
        self.tamper, self.background = tamper, background or {}
        self.rng = np.random.default_rng(seed)
        self.site_channels = dict(SITE_CHANNELS)
        self.readout = [c for c in range(CHANNELS) if c not in SITE_CHANNELS.values()]
        self.faulted = False
        self.on = False
        self.current = None
        self.sent, self.unit_log, self.files = [], [], []
        self.cleaned = False

    def frame(self):
        return 7_000_000 + int(round(self.clock.now * RATE))

    def prepare(self):
        self.clock.sleep(12)
        return dict(units=dict(H=5, W=9, F=21), channels=self.site_channels, lsb_mv=self.lsb_mv)

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
        first, frames = cur['first'], self.frame() - cur['first']
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
        patches = [(self.site_channels[p['site']], p['frame'], p['frame'] + 800, 1023) for p in pulses]
        rate = 2. * self.background.get(cur['name'], 1.)
        n = self.rng.poisson(rate * frames / RATE * CHANNELS)
        spike_frames = [first + self.rng.integers(0, frames, n)]
        spike_channels = [self.rng.integers(0, CHANNELS, n)]
        for p in pulses:
            k = self.rng.poisson(self.gain * np.exp(-self.habituation * p['count']))
            spike_frames.append(p['frame'] + self.rng.integers(300, 5000, k))
            spike_channels.append(self.rng.choice(self.readout, k))
        if self.tamper:
            self.tamper(cur['name'], pulses, patches, events)
        path = self.root / cur['name'] / (cur['name'] + '.raw.h5')
        write_h5(path, first, frames, events, (np.concatenate(spike_frames), np.concatenate(spike_channels)), patches)
        self.files.append(cur['name'])
        return str(path)

    def read(self, path):
        return H5Recording(path, CONFIG)

    def verify(self, rec):
        if rec.metadata['chip_id'] != 'chip-test':
            raise ValueError('Wrong chip')

    def cleanup(self):
        self.cleaned = True
        return []
