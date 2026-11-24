"""One-shot audit/export worker; no training, device calls or automatic retries."""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback


def stamp():
    return datetime.now(timezone.utc).isoformat()


def run(root):
    status_path = root / 'audit_export_job.json'
    state = dict(started=stamp(), pid=os.getpid(), complete=False, passed=False, phase='starting')
    with status_path.open('x') as stream:
        json.dump(state, stream)
    def save():
        temp = status_path.with_suffix('.tmp')
        with temp.open('w') as stream:
            json.dump(state, stream)
            stream.flush()
            os.fsync(stream.fileno())
        temp.replace(status_path)
    try:
        status = json.loads((root / 'experiment01' / 'parallel_status.json').read_text())
        job = json.loads((root / 'job.json').read_text())
        if status.get('accepted_cells') != 32 or status.get('passed') is not True or job.get('accepted_complete') is not True:
            raise ValueError('All32 accepted required before audit/export')
        env = os.environ | {'PYTHONOPTIMIZE': '0', 'PYTHONDONTWRITEBYTECODE': '1'}
        for phase, script, output in [('audit', 'independent_audit.py', 'independent_audit.json'),
                                      ('export', 'export_evidence.py', 'evidence_bundle.zip')]:
            state.update(phase=phase, phase_started=stamp())
            save()
            with (root / ('audit_export_' + phase + '.log')).open('x') as log:
                result = subprocess.run([sys.executable, str(root / script), '--root', str(root),
                                         '--output', str(root / output)],
                                        env=env, stdout=log, stderr=subprocess.STDOUT)
            state[phase + '_exit_code'] = result.returncode
            if result.returncode:
                raise RuntimeError(phase + ' failed; preserve outputs and inspect log')
        state.update(passed=True, phase='finished')
    except BaseException as exc:
        state.update(error=type(exc).__name__ + ': ' + str(exc), traceback=traceback.format_exc())
    finally:
        state.update(complete=True, finished=stamp())
        save()
    return 0 if state['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(run(Path(sys.argv[1]).resolve()))
