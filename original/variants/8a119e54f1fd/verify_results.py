"""Independent recomputation from H5; does not import production analysis code."""
from collections import defaultdict
from pathlib import Path
import hashlib
import json
import math
import statistics
import h5py
import numpy as np

root = Path(__file__).resolve().parent
source = root/'source/22554_baseline.raw.h5'
result = json.loads((root/'analysis_routed_v1/analysis.json').read_text(encoding='utf-8'))
strict = json.loads((root/'analysis_v1/analysis.json').read_text(encoding='utf-8'))
checksum = hashlib.sha256()
with source.open('rb') as stream:
    for block in iter(lambda: stream.read(8*1024*1024), b''):
        checksum.update(block)
assert checksum.hexdigest() == '45a341c80d7ac60eb2b9ba846dce926f9e7fd4b6fb27f0b7b8c963798640f84c'
with h5py.File(source, 'r') as file:
    store = file['data_store/data0000']; frames = store['groups/routed/frame_nos'][:]
    channels = set(int(c) for c in store['settings/mapping']['channel'])
    assert len(frames) == 6001600 and np.all(frames[1:] == frames[:-1]+1)
    assert file['wellplate/id'][0] == b'22554' and len(channels) == 850
    start, end = int(frames[0]), int(frames[-1])+1
    events = [(int(e['frameno']), int(e['channel']), float(e['amplitude'])) for e in store['spikes']]
outside = [i for i,e in enumerate(events) if not start <= e[0] < end]
unmapped = [i for i,e in enumerate(events) if e[1] not in channels]
excluded = sorted(set(outside) | set(unmapped))
assert len(outside) == 2 and len(unmapped) == 133 and len(excluded) == 135
assert excluded == result['scope_exclusions']['event_indices']
assert not strict['qc']['structural_passed'] and strict['candidates']['profile'] is None
assert not result['source_qc']['structural_passed'] and result['qc']['structural_passed']
valid = [e for e in events if start <= e[0] < end and e[1] in channels]
assert len(valid) == 57517
windows = []
for i in range(59):
    lo = start + 20000 + i*100000; hi = lo + 100000
    selected = [e for e in valid if lo <= e[0] < hi]
    bin_channels = defaultdict(set)
    for frame,channel,_ in selected:
        bin_channels[(frame-lo)//200].add(channel)
    coincidence = sum(len(bin_channels[(frame-lo)//200]) >= 2 for frame,channel,_ in selected)
    values = [len(selected)/4250, len({e[1] for e in selected})/850,
              coincidence/len(selected) if selected else 0]
    expected = result['windows'][i]
    for key,value in zip(('rate_hz_per_channel','active_channel_fraction','coincident_event_fraction'), values):
        assert abs(expected[key] - value) < 1e-14
    assert expected['spike_count'] == len(selected)
    windows.append([math.log1p(values[0]), *values[1:]])
centers = [statistics.median(w[j] for w in windows[:29]) for j in range(3)]
scales = [1.4826*statistics.median(abs(w[j]-centers[j]) for w in windows[:29]) for j in range(3)]
profile = result['candidates']['profile']
assert profile['split'] == dict(fit=[0,29], embargo=[29,31], validation=[31,59])
assert max(abs(a-b) for a,b in zip(centers,profile['center'])) < 1e-14
assert max(abs(a-b) for a,b in zip(scales,profile['scale'])) < 1e-14
normalized = [[(w[j]-centers[j])/scales[j] for j in range(3)] for w in windows]
def distance(a,b): return math.sqrt(sum((x-y)**2 for x,y in zip(a,b))/3)
nearest_fit = sorted(min(distance(normalized[i], normalized[k]) for k in range(i)) for i in range(1,29))
rank = .95*(len(nearest_fit)-1); base = math.floor(rank)
normalizer = nearest_fit[base] + (rank-base)*(nearest_fit[base+1]-nearest_fit[base])
assert abs(normalizer-profile['novelty_distance_p95']) < 1e-13
max_error = 0.
for i,score in enumerate(result['candidates']['scores'], 31):
    baseline = math.exp(-.5*distance(normalized[i], [0,0,0])**2)
    nearest = min(distance(normalized[i], normalized[k]) for k in range(i))
    novelty = min(nearest/normalizer,1.)
    max_error = max(max_error, abs(baseline-score['baseline_proximity']), abs(novelty-score['novelty']))
    assert score['window_index'] == i and score['history_last_window'] == i-1
assert max_error < 1e-12
assert profile['alpha'] == 1 and profile['r2_training_enabled'] is False
for directory in (root/'analysis_v1', root/'analysis_routed_v1'):
    manifest = json.loads((directory/'manifest.json').read_text())
    for name,checksum_expected in manifest.items():
        assert hashlib.sha256((directory/name).read_bytes()).hexdigest() == checksum_expected
report = dict(passed=True, source_sha256=checksum.hexdigest(), raw_frames=6001600,
              original_events=len(events), excluded_events=len(excluded), analyzed_scope_events=len(valid),
              checked_windows=59, checked_validation_scores=28, max_score_error=max_error,
              source_qc_failed_preserved=True, training_enabled=False,
              checked_artifact_manifests=True, implementation_imported=False,
              limits=['No independent biological validation', 'No proof of online acquisition completeness'])
(root/'independent_verification.json').write_text(json.dumps(report,indent=2)+'\n', encoding='utf-8')
print(json.dumps(report,indent=2))
