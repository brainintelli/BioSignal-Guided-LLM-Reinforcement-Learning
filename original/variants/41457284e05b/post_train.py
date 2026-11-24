from pathlib import Path
import subprocess
import time
from resources import PYTHON,environment
HERE=Path(__file__).resolve().parent
while True:
    status=(HERE/'training.log').read_text(errors='replace')
    if 'Traceback' in status:raise RuntimeError('Training scheduler needs attention')
    if status.rstrip().endswith('all complete'):break
    time.sleep(4)
for name in ('verify_training.py','schedule.py'):
    subprocess.run([PYTHON,'-u',HERE/'evaluation'/name],env=environment(),check=True)
print('training audit and frozen test complete',flush=True)
