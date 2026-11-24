"""Read-only, bounded-memory reader for closed MaxOne raw HDF5 recordings.

No SDK, device transport, or hardware gates are imported here. Raw rows are
physical channel indices; mapping row order never determines raw row order.
"""
from pathlib import Path
import json
import math
import re

import h5py
import numpy as np


class RecordingError(ValueError):
    pass


def _scalar(group, key):
    if key not in group:
        raise RecordingError('Missing measured metadata: ' + key)
    a = np.asarray(group[key][()])
    if a.size != 1:
        raise RecordingError('Expected scalar measured metadata: ' + key)
    value = a.reshape(-1)[0]
    return value.decode('utf-8').rstrip('\x00') if isinstance(value, bytes) else value.item() if hasattr(value, 'item') else value


def _event_pair(value):
    # MaxLab stores token properties as JSON strings. Accept existing integer
    # fixtures too, but never coerce booleans, floats, signs or whitespace.
    if type(value) is int and value >= 0:
        return value
    if isinstance(value, str) and re.fullmatch(r'[0-9]+', value):
        return int(value, 10)
    raise RecordingError('Event pair must be a nonnegative integer or ASCII decimal string')


class H5Recording:
    FRAME_CHUNK = 1_000_000

    def __init__(self, path, config):
        self.path = Path(path).resolve()
        self._file = None
        try:
            stat = self.path.stat()
            self.source = {'path': str(self.path), 'size_bytes': stat.st_size, 'mtime_ns': stat.st_mtime_ns}
            self._file = h5py.File(self.path, 'r')
            f = self._file
            d = f['data_store/data0000']
            if len(f['data_store']) != 1:
                raise RecordingError('Exactly one recording data store required')
            threshold_path = config.threshold_h5_dataset
            if not isinstance(threshold_path, str) or not threshold_path.startswith('settings/') or '..' in threshold_path:
                raise RecordingError('Explicit threshold readback dataset required')
            self.metadata = {
                'chip_id': str(_scalar(f, 'wellplate/id')),
                'fs': float(_scalar(d, 'settings/sampling')),
                'gain': float(_scalar(d, 'settings/gain')),
                'hpf': float(_scalar(d, 'settings/hpf')),
                'lsb_uv': float(_scalar(d, 'settings/lsb')) * 1e6,
                'threshold_value': float(_scalar(d, threshold_path)),
                'threshold_h5_dataset': threshold_path,
            }
            if any(not math.isfinite(self.metadata[k]) or self.metadata[k] <= 0 for k in ('fs','gain','hpf','lsb_uv','threshold_value')):
                raise RecordingError('Nonfinite/nonpositive measured metadata')
            self.fs = self.metadata['fs']
            m = d['settings/mapping'][:]
            if not m.dtype.names or not {'channel','electrode'} <= set(m.dtype.names):
                raise RecordingError('Mapping lacks channel/electrode')
            self.mapping = sorted((int(x['channel']), int(x['electrode'])) for x in m)
            self.channels = np.array([c for c, _ in self.mapping], dtype=int)
            if not len(m) or len(set(self.channels)) != len(m) or len({e for _,e in self.mapping}) != len(m):
                raise RecordingError('Empty/duplicate mapping')
            if any(not 0 <= c < 1024 or not 0 <= e < 26400 for c,e in self.mapping):
                raise RecordingError('Mapping outside physical device range')
            self._raw = d['groups/routed/raw']
            channel_axis=d.get('groups/routed/channels')
            if channel_axis is not None and not np.array_equal(channel_axis[:],np.arange(1024)):
                raise RecordingError('Raw channel axis differs from physical channel indices')
            frames = d['groups/routed/frame_nos']
            if frames.ndim != 1 or frames.dtype.kind not in 'ui' or len(frames) == 0:
                raise RecordingError('Missing/invalid actual frames')
            if self._raw.ndim != 2 or self._raw.shape != (1024,len(frames)):
                raise RecordingError('Raw must have 1024 physical-channel rows and one column per actual frame')
            self.frame_count = len(frames)
            self.first_frame = int(frames[0])
            self.last_frame = int(frames[-1])
            previous = None
            gaps = 0
            for offset in range(0, len(frames), self.FRAME_CHUNK):
                block = np.asarray(frames[offset:offset+self.FRAME_CHUNK], dtype=np.uint64)
                if previous is not None and int(block[0]) != previous+1:
                    gaps += 1
                gaps += int(np.count_nonzero(block[1:] != block[:-1]+np.uint64(1)))
                previous = int(block[-1])
            if gaps:
                raise RecordingError('Noncontiguous actual frames: ' + str(gaps))
            self.duration = self.frame_count / self.fs
            self.metadata.update(frame_gaps=gaps, first_frame=self.first_frame, last_frame=self.last_frame,
                                 frames=self.frame_count, duration_seconds=self.duration,
                                 raw_axis_layout='physical_channel_index', raw_rows=1024)
            ev = d['events'][:]
            if ev.dtype.names is None or not {'frameno','eventid','eventtype','eventmessage'} <= set(ev.dtype.names):
                raise RecordingError('Event schema missing required fields')
            self.events = []
            for e in ev:
                frame = int(e['frameno'])
                if not self.first_frame <= frame <= self.last_frame:
                    raise RecordingError('Event outside actual frame coverage')
                message = e['eventmessage']
                if isinstance(message, bytes): message = message.decode('utf-8').rstrip('\x00')
                data = json.loads(str(message))
                if not isinstance(data, dict): raise RecordingError('Event message must be a JSON object')
                pair = _event_pair(data['pair']) if 'pair' in data else None
                self.events.append({'id':int(e['eventid']), 'eventtype':int(e['eventtype']),
                                    'frame':frame, 'time_s':(frame-self.first_frame)/self.fs,
                                    'arm':data.get('arm'), 'pair':pair, 'phase':data.get('phase'),
                                    'session':data.get('session'), 'message':data})
            if any(b['frame'] <= a['frame'] for a,b in zip(self.events,self.events[1:])):
                raise RecordingError('Events not strictly chronological')
            sp = d['spikes']
            if sp.dtype.names is None or not {'frameno','channel'} <= set(sp.dtype.names):
                raise RecordingError('Spike schema missing required fields')
            times, channels, excluded, out_of_range = [], [], 0, 0
            for offset in range(0,len(sp),self.FRAME_CHUNK):
                block = sp[offset:offset+self.FRAME_CHUNK]
                covered=(block['frameno']>=self.first_frame) & (block['frameno']<=self.last_frame)
                out_of_range += int((~covered).sum())
                valid = np.isin(block['channel'],self.channels) & covered
                excluded += int((~valid).sum())
                # Subtract integer frames before float conversion, including large counters.
                times.append((block['frameno'][valid].astype(np.uint64)-np.uint64(self.first_frame)).astype(float)/self.fs)
                channels.append(block['channel'][valid].astype(int))
            self.spike_times = np.concatenate(times) if times else np.empty(0,dtype=float)
            self.spike_channels = np.concatenate(channels) if channels else np.empty(0,dtype=int)
            self.metadata['invalid_spikes_excluded'] = excluded
            self.metadata['outside_frame_coverage'] = out_of_range
            self.metadata['native_spikes'] = len(sp)
            after = self.path.stat()
            if (after.st_size,after.st_mtime_ns) != (stat.st_size,stat.st_mtime_ns):
                raise RecordingError('Recording changed while opening; closed file required')
        except Exception as error:
            self.close()
            if isinstance(error,RecordingError): raise
            raise RecordingError(str(error)) from error

    def sample(self, channels, t, seconds=1):
        if self._file is None:
            raise RecordingError('Recording is closed')
        stat=self.path.stat()
        if (stat.st_size,stat.st_mtime_ns)!=(self.source['size_bytes'],self.source['mtime_ns']):
            raise RecordingError('Recording changed after opening; closed immutable file required')
        if not all(math.isfinite(v) for v in (t,seconds)) or seconds <= 0 or t < 0:
            raise RecordingError('Invalid sample interval')
        # Event times are exact frame offsets; rounding avoids int(2000.999...) truncation.
        start = int(round(t*self.fs))
        count = int(round(seconds*self.fs))
        stop = start+count
        if count < 1 or start < 0 or stop > self.frame_count or t+seconds > self.duration+1e-10:
            raise RecordingError('Insufficient actual frame coverage for requested window')
        c = np.asarray(channels)
        if c.ndim != 1 or not len(c) or not np.issubdtype(c.dtype,np.integer) or len(set(c.tolist())) != len(c) or not np.all(np.isin(c,self.channels)):
            raise RecordingError('Requested channels must be unique mapped physical channels')
        order = np.argsort(c)
        selected = np.asarray(self._raw[c[order].tolist(),start:stop])
        if selected.shape != (len(c),count): raise RecordingError('Truncated sample window')
        return selected[np.argsort(order)]

    def close(self):
        if self._file is not None:
            self._file.close()
            self._file = None

    def __enter__(self): return self
    def __exit__(self,*args): self.close()
