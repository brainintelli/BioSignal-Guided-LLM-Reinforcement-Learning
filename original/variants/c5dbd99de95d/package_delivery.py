"""Check evidence/source/link integrity and package the current rules delivery."""
from pathlib import Path
from html.parser import HTMLParser
from urllib.parse import urlsplit, unquote
import hashlib
import json
import zipfile

REPORT = Path(__file__).resolve().parent
PROJECT = REPORT.parent.parent

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))

source = read(REPORT/'source_manifest.json')
for name, expected in source.items():
    assert sha(PROJECT/name) == expected, name
evidence = read(REPORT/'remote_evidence/evidence_manifest.json')
for name, expected in evidence.items():
    assert sha(REPORT/'remote_evidence'/name) == expected, name
summary = read(REPORT/'summary.json')
assert summary['reload_logp_max'] == 0 and summary['replay']['passed']
assert summary['optimizer_steps'] == 0 and not summary['formal_training_started']
assert summary['replay']['counts']['complete_transitions_compared'] == 20
checks = read(REPORT/'project_verification/checks.json')
assert len(checks['checks']) == 13
assert all(c['passed'] and c['exit_code'] == 0 for c in checks['checks'])
for name, expected in [('project_verification/python.log','OK (skipped=21)'),('remote_evidence/unit_tests_final.log','OK (skipped=3)')]:
    log = (REPORT/name).read_text(encoding='utf-8-sig')
    assert 'Ran 192 tests' in log and expected in log

class Links(HTMLParser):
    count = 0
    def handle_starttag(self, tag, attrs):
        for key, value in attrs:
            if key in ('src','href') and value and not urlsplit(value).scheme and not value.startswith('#'):
                assert (REPORT/unquote(urlsplit(value).path)).resolve().is_file(), value
                self.count += 1

links = Links()
links.feed((REPORT/'report.html').read_text(encoding='utf-8'))
status = dict(profile='open4', world_version=6, environment_version='m8_survival_v2',
              model_contract='open_world_semantic_survival_v3', default_prefix_layout='system_only',
              formal_training_started=False, optimizer_steps=0, checkpoint_writes=0,
              model_release_available=False, report='output/rules_v2_20261108/report.html')
current = PROJECT/'output/current/current_rules.json'
current.write_text(json.dumps(status,ensure_ascii=False,indent=2),encoding='utf-8')
verification = dict(source_files_verified=len(source), raw_evidence_files_verified=len(evidence),
                    report_links_verified=links.count, actor_reload_logp_error=0,
                    replay_records_verified=20, formal_training_started=False, optimizer_steps=0, checkpoint_writes=0)
(REPORT/'delivery_verification.json').write_text(json.dumps(verification,indent=2),encoding='utf-8')
files = {PROJECT/name for name in source}
files.update(p for p in REPORT.rglob('*') if p.is_file() and p.name!='delivery_manifest.json' and '__pycache__' not in p.parts)
files.update(PROJECT/name for name in ['README.md','docs/CURRENT_STATE.md','docs/WORKFLOW.md','docs/SURVIVAL_V2_STATE_REVIEW.md','tools/audit_m8s3_replay.py'])
files.add(current)
manifest = {p.relative_to(PROJECT).as_posix():sha(p) for p in sorted(files)}
manifest_path = REPORT/'delivery_manifest.json'
manifest_path.write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
archive = PROJECT/'output/m8s3_delivery.zip'
with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
    for p in [*sorted(files),manifest_path]:
        z.write(p,p.relative_to(PROJECT).as_posix())
receipt = dict(file=str(archive), bytes=archive.stat().st_size, files=len(files)+1, sha256=sha(archive))
(PROJECT/'output/m8s3_delivery.sha256.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8')
print(json.dumps({**verification,'package':receipt},ensure_ascii=False))
