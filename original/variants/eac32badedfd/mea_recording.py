"""Offline evidence only. A finalized H5 cannot supply a live coverage watermark."""
from collections import Counter
import hashlib
import math
from pathlib import Path

from .mea_protocol import canonical, digest, integer, number, strict_json


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''): h.update(block)
    return h.hexdigest()


def channel_map(value):
    if not isinstance(value, list) or not value:
        raise ValueError('empty mapping')
    for pair in value:
        if not isinstance(pair, list) or len(pair) != 2: raise ValueError('mapping pair')
        integer(pair[0], 'channel'); integer(pair[1], 'electrode')
    if any(len({p[i] for p in value}) != len(value) for i in (0, 1)):
        raise ValueError('mapping not one-to-one')
    return sorted(value)


def audit_recording(metadata, frames, h5_spikes, live_spikes, live_status, guard_seconds=1):
    result = dict(schema='mea.recording.audit.v1', offline_integrity_passed=False,
                  online_coverage_proven=False, upstream_live_loss=None,
                  biological_identity_complete=False, r2_reward_enabled=False, alpha=1,
                  errors=[], guard_seconds=guard_seconds)
    errors = result['errors']
    try:
        mapping = channel_map(metadata['mapping'])
        if any(channel_map(metadata[k]) != mapping for k in
               ('current_mapping_before', 'current_mapping_after')):
            errors.append('current_mapping_mismatch')
        channels = {p[0] for p in mapping}
        raw_channels = metadata['raw_channels']
        if len(set(raw_channels)) != len(raw_channels) or set(raw_channels) != channels:
            errors.append('raw_channel_coverage')
        rate = number(metadata['sample_rate_hz'], 'rate')
        lsb = number(metadata['lsb_volts'], 'lsb')
        guard = number(guard_seconds, 'guard')
        well = integer(metadata['well_id'], 'well')
        if rate <= 0 or lsb <= 0 or guard < 0: raise ValueError('invalid rate/lsb/guard')
        if not len(frames): raise ValueError('empty raw recording')
        start, end = int(frames[0]), int(frames[-1]) + 1
        if start < 0 or metadata['raw_shape'] != [len(raw_channels), len(frames)]:
            errors.append('raw_shape')
        gaps = sum(int(b) != int(a) + 1 for a, b in zip(frames, frames[1:]))
        result.update(raw_interval=[start, end], raw_frame_count=len(frames), raw_discontinuities=gaps)
        if gaps: errors.append('raw_frame_discontinuity')
        duration = len(frames)/rate
        result['raw_duration_seconds'] = duration
        if 'requested_seconds' in metadata:
            requested = number(metadata['requested_seconds'], 'requested seconds')
            # Engineering tolerance for acknowledgement/buffer boundaries, not biological timing.
            result['duration_tolerance_seconds'] = dict(short=0.5, long=2.0)
            if requested <= 0 or not requested-0.5 <= duration <= requested+2.0:
                errors.append('recording_duration_mismatch')
        if 'start_time_unix_ms' in metadata and 'stop_time_unix_ms' in metadata:
            wall_duration = (metadata['stop_time_unix_ms']-metadata['start_time_unix_ms'])/1000
            result['h5_wall_duration_seconds'] = wall_duration
            if not math.isfinite(wall_duration) or abs(wall_duration-duration) > 0.5:
                errors.append('h5_time_duration_mismatch')
        lo, hi = start + math.ceil(guard * rate), end - math.ceil(guard * rate)
        if hi <= lo: raise ValueError('empty audit interval')
        result.update(audit_interval=[lo, hi], audited_seconds=(hi-lo)/rate,
                      channel_map_id=digest(mapping), mapped_channels=len(mapping))
        if live_status.get('clean_end') is not True or live_status.get('coordinator_ok') is not True:
            errors.append('acquisition_not_clean')
        for key in ('sdk_open_status', 'sdk_close_status', 'sdk_errors'):
            if type(live_status.get(key)) is not int or live_status[key] != 0:
                errors.append(key)
        if live_status.get('event_count') != len(live_spikes): errors.append('live_event_count')

        def select(rows, live):
            selected = {}; excluded = Counter(unmapped=0, outside_raw=0, guard=0)
            for row in rows:
                if len(row) != (4 if live else 3): raise ValueError('invalid spike width')
                frame = integer(row[0], 'frame'); channel = integer(row[1], 'channel')
                if live and integer(row[2], 'well') != well:
                    raise ValueError('unexpected well')
                amplitude = number(row[-1], 'amplitude')
                if channel not in channels: excluded['unmapped'] += 1; continue
                if not start <= frame < end: excluded['outside_raw'] += 1; continue
                if not lo <= frame < hi: excluded['guard'] += 1; continue
                key = (frame, channel)
                if key in selected: errors.append('live_duplicate' if live else 'h5_duplicate')
                selected[key] = amplitude
            return selected, dict(excluded)

        expected, result['h5_exclusions'] = select(h5_spikes, False)
        actual, result['live_exclusions'] = select(live_spikes, True)
        missing, extra = expected.keys() - actual.keys(), actual.keys() - expected.keys()
        common = expected.keys() & actual.keys()
        amp_mismatch = [k for k in common if not math.isclose(expected[k], actual[k], rel_tol=1e-6, abs_tol=1e-5)]
        result.update(h5_audited_events=len(expected), live_audited_events=len(actual),
                      missing_in_live=len(missing), extra_in_live=len(extra),
                      amplitude_mismatches=len(amp_mismatch), matched_events=len(common)-len(amp_mismatch),
                      missing_examples=sorted(missing)[:10], extra_examples=sorted(extra)[:10],
                      amplitude_tolerance=dict(relative=1e-6, absolute_native=1e-5))
        if missing or extra or amp_mismatch: errors.append('event_disagreement')
        result['baseline_metrics'] = dict(
            event_rate_hz=len(expected)/result['audited_seconds'],
            observed_active_channels=len({k[1] for k in expected}),
            active_fraction=len({k[1] for k in expected})/len(channels))
    except (ValueError, TypeError, KeyError, OverflowError) as exc:
        errors.append(str(exc))
    result['offline_integrity_passed'] = not errors
    return result


def load_h5(path):
    """Read routed frame indices and spikes, not the potentially huge voltage array."""
    import h5py
    with h5py.File(path, 'r') as f:
        stores = list(f['data_store'])
        if len(stores) != 1: raise ValueError('expected one data store')
        root = f['data_store'][stores[0]]
        groups = list(root['groups'])
        if len(groups) != 1: raise ValueError('expected one raw group')
        raw = root['groups'][groups[0]]
        settings = root['settings']
        def scalar(ds):
            value = ds[()].reshape(-1)[0]
            return value.decode() if isinstance(value, bytes) else value.item()
        meta = dict(chip_id=str(scalar(f['wellplate/id'])), well_id=int(scalar(root['well_id'])),
                    sample_rate_hz=float(scalar(settings['sampling'])),
                    lsb_volts=float(scalar(settings['lsb'])),
                    mapping=[[int(r['channel']), int(r['electrode'])] for r in settings['mapping']],
                    raw_channels=[int(v) for v in raw['channels']], raw_shape=list(raw['raw'].shape),
                    start_time_unix_ms=int(scalar(root['start_time'])),
                    stop_time_unix_ms=int(scalar(root['stop_time'])),
                    maxlab_version=str(scalar(f['mxw_version'])),
                    spike_threshold=float(scalar(settings['spike_threshold'])),
                    raw_group=groups[0], organoid_id=None, organoid_type=None, culture_days=None)
        frames = raw['frame_nos'][:]
        spikes = [[int(r['frameno']), int(r['channel']), float(r['amplitude'])] for r in root['spikes']]
    return meta, frames, spikes


def load_live(path):
    events, status, ended = [], {}, False
    with open(path, 'rb') as f:
        for seq, line in enumerate(f):
            if ended or not line.endswith(b'\n') or len(line) > 2_000_000:
                raise ValueError('trailing/truncated/oversized log')
            row = strict_json(line)
            if row.get('seq') != seq: raise ValueError('log sequence')
            kind = row.get('type')
            if seq == 0:
                if kind != 'session' or row.get('protocol') != 'mea.filtered.p1.v1':
                    raise ValueError('log session')
                status.update(row)
            elif kind == 'events':
                events.extend(row['events'])
                if len(events) > 5_000_000: raise ValueError('event capacity')
            elif kind == 'end':
                ended = True; status.update(row)
            else: raise ValueError('log packet type')
    if not ended or status.get('event_count') != len(events): raise ValueError('log EOF/count')
    return events, status


def export_recording(path, metadata, h5_spikes, report, session_id):
    if report.get('offline_integrity_passed') is not True: raise ValueError('audit failed')
    lo, hi = report['audit_interval']; mapping = channel_map(metadata['mapping'])
    channels = {p[0] for p in mapping}
    events = sorted([f, c, a * metadata['lsb_volts'] * 1e6]
                    for f, c, a in h5_spikes if lo <= f < hi and c in channels)
    session = dict(type='session', protocol='mea.spikes.v1', session_id=session_id, seq=0,
                   source='recording', sample_rate_hz=metadata['sample_rate_hz'],
                   well_id=metadata['well_id'], start_frame=lo,
                   channel_map_id=digest(mapping), channels=mapping)
    with open(path, 'x', encoding='utf-8', newline='\n') as f:
        f.write(canonical(session)+'\n'); seq = 1
        for i in range(0, len(events), 1000):
            f.write(canonical(dict(type='spikes', session_id=session_id, seq=seq, events=events[i:i+1000]))+'\n')
            seq += 1
        f.write(canonical(dict(type='watermark', session_id=session_id, seq=seq,
                               end_frame=hi, complete=True, dropped=0))+'\n')
