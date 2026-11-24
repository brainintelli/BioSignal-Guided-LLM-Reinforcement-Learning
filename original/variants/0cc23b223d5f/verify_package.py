"""Read-only artifact validation; no MaxLab or device connections."""
import hashlib
import json
import tarfile
from pathlib import Path

ROOT = Path('./github_program/BrainDesktop')
RELEASE = ROOT / 'output/stimulation-p90-guard-20261117'
BUNDLE = RELEASE / 'BrainStimulation'
ARCHIVE = RELEASE / 'BrainStimulation-ubuntu22.04-python.tar.gz'
OUT = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


manifest = json.loads((BUNDLE / 'SOURCE-MANIFEST.json').read_text(encoding='utf-8'))
for name, digest in manifest['source_sha256'].items():
    assert sha(ROOT / name) == digest, name
assert sha(ARCHIVE) == (RELEASE / 'SHA256SUMS').read_text().split()[0]
with tarfile.open(ARCHIVE, 'r:gz') as archive:
    files = [m for m in archive.getmembers() if m.isfile()]
    for member in files:
        assert hashlib.sha256(archive.extractfile(member).read()).hexdigest() == sha(RELEASE / member.name), member.name

runner = 'standalone/stimulation/brain_stimulation/runner.py'
assert sha(ROOT / runner) == sha(BUNDLE / 'brain_stimulation/runner.py')
assert 'quantizes to zero' in (BUNDLE / 'brain_stimulation/runner.py').read_text(encoding='utf-8')
for name, count in [('standalone_tests.log', 21), ('schedule_tests.log', 5)]:
    content = (OUT / name).read_text(encoding='utf-8-sig')
    assert f'Ran {count} tests' in content and content.rstrip().endswith('OK'), name
assert 'build complete' in (OUT / 'docs_build.log').read_text(encoding='utf-8-sig')
plan = Path('./LLM湿件生存游戏/docs/MEA_TWO_HOUR_PREPARATION_20261117.md')
result = dict(
    schema='mea.stimulation.preparation.verification.v1',
    archive=str(ARCHIVE), archive_sha256=sha(ARCHIVE),
    source_files_verified=len(manifest['source_sha256']), archive_files_verified=len(files),
    standalone_tests=21, schedule_tests=5, docs_build_passed=True,
    desktop_recording_mode_tests_executed=0,
    plan_sha256=sha(plan),
    hardware_connected=False, stimulation_sent=False, acquisition_started=False,
    training_started=False, r2_reward_enabled=False, alpha=1,
    limitations=[
        'No physical waveform or H5 verification in this preparation.',
        'Nominal nonzero-to-zero DAC guard only; no per-electrode P90 or peak-calibration enforcement.',
        'No two-hour hard-deadline or game closed-loop controller implemented.',
        'Generic example amplitude 100 mV is not an approved configuration for this experiment.',
        'Two unchanged desktop recording-mode tests could not run: Docker lacks PySide6 and local venv points to a missing Python.',
    ],
)
(OUT / 'verification.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
print(json.dumps(result, ensure_ascii=False, indent=2))
