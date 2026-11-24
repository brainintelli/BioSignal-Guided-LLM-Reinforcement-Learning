import dataclasses
import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest

import h5py
import numpy as np

from mea_stim_v2.config import Config


def fixture(root):
    route = root / 'route.cfg'
    # Route files encode all mapping pairs in the first whitespace token.
    route.write_text(''.join(f'{i}({1000+i})' for i in range(60)))
    config = Config(session_id='test-session', chip_id='28073', organoid_id='test-org',
                    started_utc='2026-11-17T06:13:00Z', device_hostname='192.0.2.21',
                    route_file=str(route), route_sha256=hashlib.sha256(route.read_bytes()).hexdigest(),
                    stim_electrode=1059, threshold_value=5., threshold_h5_dataset='settings/spike_threshold')
    path = root / 'test.raw.h5'
    with h5py.File(path, 'w') as f:
        f['wellplate/id'] = [b'28073']
        d = f.create_group('data_store/data0000')
        for key, val in [('sampling', 20000), ('gain', 512), ('hpf', 300),
                         ('lsb', 3e-6), ('spike_threshold', 5)]:
            d['settings/' + key] = [val]
        mapping = np.zeros(60, dtype=[('channel', 'i4'), ('electrode', 'i4')])
        mapping['channel'] = np.arange(59, -1, -1)
        mapping['electrode'] = mapping['channel'] + 1000
        d['settings/mapping'] = mapping
        f0 = 2**40 + 99
        d['groups/routed/frame_nos'] = np.arange(f0, f0 + 4000, dtype='u8')
        raw = d.create_dataset('groups/routed/raw', (1024, 4000), dtype='u2', chunks=(1, 1000), compression='gzip')
        raw[2, :] = np.arange(4000) % 1024
        raw[57, :] = 57
        ev = np.zeros(1, dtype=[('frameno', 'u8'), ('eventid', 'i4'), ('eventtype', 'i4'), ('eventmessage', 'S256')])
        ev[0] = (f0+2001, 8201, 1, json.dumps({'session':'test-session', 'arm':'sham', 'pair':0, 'phase':'pilot'}).encode())
        d['events'] = ev
        sp = np.zeros(2, dtype=[('frameno', 'u8'), ('channel', 'i4')])
        sp[0] = (f0+100, 2)
        sp[1] = (f0+5000, 2)
        d['spikes'] = sp
    return path, config


class RecordingTests(unittest.TestCase):
    def test_physical_channels_and_fractional_frame_window_are_exact(self):
        from mea_stim_v2.recording import H5Recording, RecordingError
        with tempfile.TemporaryDirectory() as td:
            path, config = fixture(Path(td))
            with H5Recording(path, config) as rec:
                a = rec.sample([2, 57], rec.events[0]['time_s'], .001)
                self.assertEqual(a.shape, (2, 20))
                self.assertEqual(a[0, 0], 2001 % 1024)
                self.assertEqual(a[1, 0], 57)
                self.assertEqual(rec.metadata['threshold_value'], 5)
                self.assertEqual(rec.metadata['invalid_spikes_excluded'], 1)
                self.assertEqual(rec.events[0]['session'], 'test-session')
                with self.assertRaises(RecordingError): rec.sample([2], .199, .002)
                with self.assertRaises(RecordingError): rec.sample([900], 0, .001)

    def test_missing_threshold_never_uses_config_value_as_readback(self):
        from mea_stim_v2.recording import H5Recording, RecordingError
        with tempfile.TemporaryDirectory() as td:
            path, config = fixture(Path(td))
            with h5py.File(path, 'r+') as f: del f['data_store/data0000/settings/spike_threshold']
            with self.assertRaises(RecordingError): H5Recording(path, config)

    def test_frame_gap_is_rejected(self):
        from mea_stim_v2.recording import H5Recording, RecordingError
        with tempfile.TemporaryDirectory() as td:
            path, config = fixture(Path(td))
            with h5py.File(path, 'r+') as f: f['data_store/data0000/groups/routed/frame_nos'][1999] += 2
            with self.assertRaises(RecordingError): H5Recording(path, config)

    def test_missing_metadata_fails_closed(self):
        from mea_stim_v2.recording import H5Recording, RecordingError
        for field in ('gain', 'hpf', 'lsb'):
            with tempfile.TemporaryDirectory() as td:
                path, config = fixture(Path(td))
                with h5py.File(path, 'r+') as f: del f['data_store/data0000/settings/' + field]
                with self.assertRaises(RecordingError): H5Recording(path, config)

    def test_raw_mapping_rows_are_not_treated_as_physical_indices(self):
        from mea_stim_v2.recording import H5Recording, RecordingError
        with tempfile.TemporaryDirectory() as td:
            path, config = fixture(Path(td))
            with h5py.File(path,'r+') as f:
                del f['data_store/data0000/groups/routed/raw']
                f.create_dataset('data_store/data0000/groups/routed/raw',shape=(60,4000),dtype='u2')
            with self.assertRaises(RecordingError):H5Recording(path,config)

    def test_changed_recording_is_rejected_before_sampling(self):
        from mea_stim_v2.recording import H5Recording, RecordingError
        with tempfile.TemporaryDirectory() as td:
            path, config = fixture(Path(td))
            with H5Recording(path,config) as rec:
                stat=path.stat();os.utime(path,ns=(stat.st_atime_ns,stat.st_mtime_ns+1_000_000_000))
                with self.assertRaises(RecordingError):rec.sample([2],0,.001)


if __name__ == '__main__': unittest.main()
