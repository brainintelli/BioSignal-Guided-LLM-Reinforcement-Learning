"""Verify both closed raw H5 copies, then recompute long-recording admission locally."""
from pathlib import Path
import datetime
import hashlib
import json
import math
import subprocess
import sys
import h5py
import numpy as np

OUT = Path(__file__).resolve().parent
PROJECT = OUT.parents[1]
ROOT = OUT / 'live_22021_20261117_032840'
PIN = '8bec759ec7a6826f7cffc0f831a4e3a0' + '856c9d6426149394ea40199766887ec0'
def read(p): return json.loads(p.read_text(encoding='utf-8'))
def sha(p):
    with p.open('rb') as f: return hashlib.file_digest(f, 'sha256').hexdigest()

assert sha(ROOT / 'manifest.json') == PIN
manifest = read(ROOT / 'manifest.json')
verified = []
for name, item in manifest['files'].items():
    path = (ROOT / name).resolve()
    assert path.is_relative_to(ROOT.resolve()), name
    assert path.stat().st_size == item['size'], 'Incomplete backup: ' + name
    assert sha(path) == item['sha256'], name
    verified.append(name)
print('All source files verified:', len(verified), flush=True)
run = ROOT / manifest['accepted_run']
with (OUT / 'local_long_admission_verify.log').open('x') as log:
    result = subprocess.run([sys.executable, '-s', '-m', 'tools.mea_h5_admit', 'verify', str(run),
                             '--chip-id', '22021'], cwd=PROJECT, stdout=log, stderr=subprocess.STDOUT, timeout=240)
assert result.returncode == 0, 'Independent local H5 admission failed'
admission = read(run / 'admission.json')
review = read(run / 'candidate_review.json')
lo, hi = admission['audit_interval']
independent = []
with h5py.File(run / 'baseline.raw.h5', 'r') as f:
    store = f['data_store'][next(iter(f['data_store']))]
    mapping = store['settings/mapping'][:]
    spikes = store['spikes'][:]
    scale = float(np.asarray(store['settings/lsb'][()]).reshape(-1)[0]) * 1e6
    for expected in review['candidates']:
        channels = mapping['channel'][mapping['electrode'] == expected['electrode_id']]
        selected = spikes[(spikes['frameno'] >= lo) & (spikes['frameno'] < hi) & np.isin(spikes['channel'], channels)]
        assert len(selected) == expected['events']
        p90 = float(np.quantile(np.abs(selected['amplitude'].astype(float)) * scale, .9, method='linear')) if len(selected) else None
        if p90 is not None:
            assert math.isclose(p90, expected['p90_uv'], rel_tol=1e-12, abs_tol=1e-12)
            assert 1.2 * p90 / 1000 < float(review['dac_lsb_mv'])
        independent.append(dict(electrode=expected['electrode_id'], event_count=len(selected), p90_uv=p90))
completion = read(ROOT / 'completion.json')
assert completion['stimulation_sent'] is False and completion['hardware_recording_stopped']
assert completion['alpha'] == 1 and completion['r2_reward_enabled'] is False and completion['realtime_training'] is False
assert completion['live_watch']['passed']
result = dict(passed=True, verified_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    source_manifest_sha256=PIN, verified_source_files=len(verified), full_raw_copies_local=True,
    original_strict_audit_retained=True, local_long_admission_verified=True,
    independently_recomputed_candidate_quantiles=independent,
    source_host='192.0.2.101', chip_id='22021', total_recorded_seconds=659.99,
    long_seconds=admission['raw_duration_seconds'], long_raw_frames=admission['raw_frame_count'],
    mapped_analysis_channels=admission['mapped_channels'], matched_events=admission['matched_events'],
    raw_discontinuities=admission['raw_discontinuities'], stimulation_sent=False,
    model_training_performed=False, alpha=1, r2_reward_enabled=False, realtime_training=False)
with (OUT / 'backup_verification.json').open('x', encoding='utf-8') as f:
    json.dump(result, f, indent=2)
    f.write('\n')
print(json.dumps(result, indent=2), flush=True)
