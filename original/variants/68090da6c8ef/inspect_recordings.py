"""Inspect at most four closed H5 files; never change acquisition or file locks."""
import datetime
import json
import os
from pathlib import Path
import h5py


def open_h5():
    result = set()
    for process in Path('/proc').iterdir():
        if not process.name.isdigit() or int(process.name) == os.getpid():
            continue
        try:
            for fd in (process / 'fd').iterdir():
                try:
                    target = os.readlink(fd)
                    if target.endswith('.h5'):
                        result.add(target)
                except OSError:
                    pass
        except OSError:
            pass
    return result


root = Path('/home/researcher/brainintelli/process/2026-1116-22554/261116/22554/Network')
opened = open_h5()
paths = sorted(root.glob('*/data.raw.h5'), key=lambda p: p.stat().st_mtime_ns, reverse=True)
report = dict(schema='mea.r2.recording_inventory.v1',
              checked_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
              stimulation_status='unstimulated_as_reported_by_user',
              scope='read-only metadata; not a recording integrity audit or R2 validation',
              files=[], open_h5=sorted(opened), h5py_version=h5py.__version__)
closed_read = 0
for path in paths[:12]:
    before = path.stat()
    row = dict(path=str(path), size_bytes=before.st_size, mtime_ns=before.st_mtime_ns)
    report['files'].append(row)
    if str(path) in opened or str(path) in open_h5():
        row['status'] = 'skipped_open_file'
        continue
    if closed_read >= 4:
        row['status'] = 'skipped_read_limit'
        continue
    closed_read += 1
    try:
        with h5py.File(path, 'r') as f:
            def scalar(ds):
                value = ds[()].reshape(-1)[0]
                return value.decode() if isinstance(value, bytes) else value.item()
            row['root_keys'] = list(f.keys())
            if 'wellplate/id' in f:
                row['chip_id'] = str(scalar(f['wellplate/id']))
            row['stores'] = []
            for name, store in f.get('data_store', {}).items():
                data = dict(name=name, keys=list(store.keys()))
                row['stores'].append(data)
                for field in ('well_id', 'start_time', 'stop_time'):
                    if field in store:
                        data[field] = scalar(store[field])
                for field in ('sampling', 'lsb'):
                    if 'settings/' + field in store:
                        data[field] = scalar(store['settings/' + field])
                if 'settings/mapping' in store:
                    data['mapping_shape'] = list(store['settings/mapping'].shape)
                if 'spikes' in store:
                    data['spikes_shape'] = list(store['spikes'].shape)
                    data['spikes_dtype'] = str(store['spikes'].dtype)
                data['raw_groups'] = []
                for group_name, group in store.get('groups', {}).items():
                    data['raw_groups'].append(dict(name=group_name, datasets={
                        key: list(value.shape) for key, value in group.items() if isinstance(value, h5py.Dataset)}))
        after = path.stat()
        row['metadata_stable_during_read'] = (before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns)
        row['status'] = 'metadata_read'
    except Exception as exc:
        row['status'] = 'read_failed'
        row['error'] = type(exc).__name__ + ': ' + str(exc)
print(json.dumps(report, ensure_ascii=False, indent=2))
