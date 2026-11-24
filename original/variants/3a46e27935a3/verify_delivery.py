"""Verify the full local training delivery against independently received hashes."""
from pathlib import Path
import datetime
import hashlib
import json
import zipfile

ROOT=Path(__file__).resolve().parent
ARCHIVE_SHA='f9d7d5b7939d4663c899f474c06bfdccd1de6966a7ce38078f8208fa69c226d3'
MANIFEST_SHA='36e73972fb68b6e5d6790f1235e20d158022a72c75b4a6774317cf0f9897d1f2'
SOURCE_SHA='49302e7794da71b5c6038827c182eb7cdf8edf9eca4f2d1c5faf49d2af7fe507'
FINAL_ACTOR='c59ab701d4777491a9877aeef98ca88af5089d3465b2029e7aa767d156ea9604'

def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def read(p):return json.loads(p.read_text(encoding='utf-8'))

archive=ROOT/'training_delivery_28080.zip'
assert archive.stat().st_size==528720314
assert sha(archive)==ARCHIVE_SHA
dest=ROOT/'training_delivery';dest.mkdir(exist_ok=False)
with zipfile.ZipFile(archive) as z:
    for n in z.namelist():assert (dest/n).resolve().is_relative_to(dest.resolve())
    z.extractall(dest)
assert sha(dest/'accepted_training_manifest.json')==MANIFEST_SHA
assert sha(dest/'source_manifest.json')==SOURCE_SHA
manifest=read(dest/'accepted_training_manifest.json');out=dest/'training28080_01'
assert {p.relative_to(out).as_posix() for p in out.rglob('*') if p.is_file()}==set(manifest)
for n,item in manifest.items():
    p=(out/n).resolve();assert p.is_relative_to(out.resolve())
    assert p.stat().st_size==item['size'] and sha(p)==item['sha256'],n
report=read(out/'report.json');verification=read(out/'independent_verification.json');status=read(dest/'training_status.json')
assert verification['passed'] is True and status['phase']=='accepted'
assert status['train_exit']==status['verify_exit']==0
assert verification['collection_pid']!=verification['verification_pid']
assert verification['source_manifest_sha256']==SOURCE_SHA
assert len(verification['rounds'])==len(report['rounds'])==8
assert verification['decisions']==report['decisions']==1024
assert report['source']['chip_id']=='28080' and report['source']['analysis_channels']==831
assert report['alpha']==1 and report['r2_enabled'] is False and report['realtime_training'] is False
assert report['heldout_used'] is False and report['biological_effect_verified'] is False
assert report['final']['actor']==FINAL_ACTOR and report['final']['actor']!=report['initial']['actor']
assert report['final']['critic']!=report['initial']['critic']
assert all(report['final'][k]==report['initial'][k] for k in ('base','reference'))
error=max(v[k] for v in verification['rounds'] for k in ('old_error','new_error','bootstrap_error'))
assert error<=1e-6
assert set(verification['rounds'][-1]['optimizer_steps_after']['actor'])=={16}
assert set(verification['rounds'][-1]['optimizer_steps_after']['critic'])=={128}
source=ROOT/'dataset28080_metadata/long_20261116_run01'
recording_sha=sha(source/'recording.jsonl');assert recording_sha==report['source']['inputs']['recording.jsonl']
indices=[];spikes=0;ticks=0;terminals=0;counts=[]
for number in range(1,9):
    rd=out/f'round_{number:02d}'
    rows=[json.loads(s) for s in (rd/'rollout.jsonl').read_text(encoding='utf-8').splitlines()]
    assert len(rows)==128
    for row in rows:
        i=row['index'];s=row['snapshot'];reward=row['reward_record']
        assert s['index']==i and s['recording_sha256']==recording_sha and s['mapped_channels']==831
        assert row['action_source']=='model'
        assert row['prepared']['critic_input'][-4:]==s['normalized_values']
        assert reward['alpha']==1 and reward['r2_enabled'] is False and reward['r2_valid'] is False and reward['r2_raw'] is None
        assert row['reward']==reward['total_reward']==row['r1_score']['reward']
        indices.append(i);spikes+=int(s['feature_values'][0]);counts.append(s['feature_values'][0])
        ticks+=row['duration'];terminals+=int(row['terminal'])
assert indices==list(range(1024)) and ticks==report['elapsed_ticks'] and terminals==report['terminals']
receipt=dict(passed=True,verified_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    archive_sha256=ARCHIVE_SHA,accepted_manifest_sha256=MANIFEST_SHA,source_manifest_sha256=SOURCE_SHA,
    verified_training_files=len(manifest),full_checkpoints_local=True,decisions=1024,rounds=8,
    elapsed_ticks=ticks,terminals=terminals,training_window_spikes=spikes,
    current_window_spike_count_range=[min(counts),max(counts)],maximum_statistic_error=error,
    final_actor_sha256=FINAL_ACTOR,heldout_used=False,raw_h5_copies=['192.0.2.101','192.0.2.15'],
    raw_h5_local=False,biological_effect_verified=False)
with (ROOT/'local_delivery_verification.json').open('x',encoding='utf-8') as f:json.dump(receipt,f,indent=2)
print(json.dumps(receipt,indent=2))
