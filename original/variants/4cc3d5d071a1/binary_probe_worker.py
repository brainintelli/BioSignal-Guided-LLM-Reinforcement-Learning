"""One bounded technical PPO smoke; no automatic retry or efficacy evaluation."""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback

root = Path(sys.argv[1]).resolve()
campaign = Path(sys.argv[2]).resolve()
status_path = root / 'ppo_smoke_job.json'
status = dict(pid=os.getpid(), started=datetime.now(timezone.utc).isoformat(),
              complete=False, passed=False, technical_update_only=True, r2_source='simulation')
with status_path.open('x') as stream:
    json.dump(status, stream)
try:
    command = [sys.executable, str(root / 'binary_ppo_probe.py'),
               '--campaign', str(campaign), '--binary-package', str(root),
               '--source', '/home/researcher/mea-28080-shuffle-20261117/experiment01/b03_shuffled_after/train/round_01/rollout.jsonl',
               '--output', str(root / 'ppo_smoke01')]
    with (root / 'ppo_smoke.log').open('x') as log:
        result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
    status['exit_code'] = result.returncode
    if result.returncode:
        raise RuntimeError('Technical probe failed; preserve log and do not silently retry')
    receipt = json.loads((root / 'ppo_smoke01' / 'result.json').read_text())
    status['passed'] = receipt.get('passed') is True
except BaseException as exc:
    status.update(error=type(exc).__name__ + ': ' + str(exc), traceback=traceback.format_exc())
finally:
    status.update(complete=True, finished=datetime.now(timezone.utc).isoformat())
    temp = status_path.with_suffix('.tmp')
    with temp.open('w') as stream:
        json.dump(status, stream)
        stream.flush()
        os.fsync(stream.fileno())
    temp.replace(status_path)
raise SystemExit(0 if status['passed'] else 1)
