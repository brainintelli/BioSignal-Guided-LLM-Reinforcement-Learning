"""Consolidate the observed regression retry without erasing its failed attempt."""
import hashlib
import json
from pathlib import Path
import re

folder=Path(__file__).resolve().parent
initial=folder/'remote_checks_initial.json'
if not initial.exists():initial.write_bytes((folder/'remote_checks.json').read_bytes())
checks=json.loads(initial.read_text())
log=(folder/'remote_python.log').read_text(encoding='utf-8')
assert 'Ran 147 tests' in log and 'OK (skipped=3)' in log
assert not re.search(r'(?m)(SCRIPT ERROR:|ERROR:|FAIL failures=|FAILED \()',log)
assert hashlib.sha256((folder.parents[1]/'tests/test_m8x_fast_policy.py').read_bytes()).hexdigest()=='bb583388765fd722d945348bdd24d40d65aecbd63333acfcd5b1d32444f2e09c'
for row in checks:
    if row['name']=='python':
        row.update(passed=True,exit_code=0,seconds=9.314,log='remote_python.log',
                   test_count=147,passed_tests=144,skipped=3,attempt=2,
                   initial_failure='Missing output/runtime_tests; see remote_python_initial.log',
                   exit_evidence='Netcatty terminal_execute returned [exit code: 0] for the retry; logs agree')
assert all(r['passed'] for r in checks)
(folder/'remote_checks.json').write_text(json.dumps(dict(passed=True,checks=checks,
    initial_checks='remote_checks_initial.json',
    engine_logs_remote='/home/researcher/meadowmind-gpu/diagnostics/rl_speed_20261107_regression',
    final_process_check='No benchmark, formal trainer/server, regression process or GPU compute process observed after completion'),indent=2))
attempts=json.loads((folder/'attempts.json').read_text())
attempts.update(
    comparison_v2='Completed in 866.394 seconds with all six gates true and comparison.json written. Netcatty foreground transport disconnected while the remote process continued; completion verified from persisted results and subsequent process/GPU checks, not from the lost foreground exit status.',
    local_full_regression='10 Godot suites passed. Editor import user:// error and HTTP WinError 10013 occurred within the sandbox. Two Windows PowerShell cases passed locally. Original failed logs retained.',
    remote_regression='Import and 10 Godot suites passed. Initial Python attempt had 19 missing output/runtime_tests errors. Created the expected fixture root and reran unchanged tests: 147 run, 144 passed, 3 skipped, observed exit 0. Five new optimization tests passed; original application code unchanged.',
    replay='All 576 new real transitions replayed in local Godot; inputs, legal actions, next states and rewards matched. 72 frozen source hashes plus measured optimization sources matched.')
(folder/'attempts.json').write_text(json.dumps(attempts,indent=2),encoding='utf-8')
print(json.dumps({'regression_passed':True,'tests_run':147,'tests_passed':144,'tests_skipped':3,'engine_suites':10}))
