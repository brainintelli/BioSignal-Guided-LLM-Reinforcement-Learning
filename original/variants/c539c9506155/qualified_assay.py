"""Run the registered single-site assay only after the actual final readout route is stable."""
from pathlib import Path
import json
import signal
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import one_hz
original_neutral=one_hz.neutral

def neutral(dev,chs,folder,wire):
    result=original_neutral(dev,chs,folder,wire)
    measured=json.loads((folder/'neutral-check.json').read_bytes())
    if measured['quiet_noise_fraction']<.95:
        raise ValueError('Final connected zero-output route has less than 95% stable readout channels')
    return result

if __name__=='__main__':
    def stop(*_):raise TimeoutError('Original time limit')
    for s in (signal.SIGALRM,signal.SIGTERM,signal.SIGHUP):signal.signal(s,stop)
    signal.alarm(760);root=Path(sys.argv[1]);select=json.loads((root/'selected-scan.json').read_bytes())
    assert select['scan'].startswith('recovery/independent-')
    assert json.loads((root/select['scan']/'qc.json').read_bytes())['passed']
    one_hz.neutral=neutral
    one_hz.run(root)
