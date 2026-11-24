"""Run the offline tests of this package. Contacts no device and writes only temporary files.

    python selfcheck.py              every test
    python selfcheck.py test_rig     one test file

The sibling directories mea_live_pair, mea_online, mea_binary_r2, mea_state2_r2
and mea_stimulation_v2 must lie next to this one.
"""
from pathlib import Path
import sys
import unittest

HERE = Path(__file__).resolve().parent
SIBLINGS = ('mea_live_pair', 'mea_online', 'mea_binary_r2', 'mea_state2_r2', 'mea_stimulation_v2')

if __name__ == '__main__':
    missing = [name for name in SIBLINGS if not (HERE.parent / name).is_dir()]
    if missing:
        raise SystemExit('Missing next to this directory: ' + ', '.join(missing))
    local = HERE.parents[1] / '.tools' / 'mea_baseline_deps'
    paths = [HERE, HERE / 'tests'] + [HERE.parent / name for name in SIBLINGS] + ([local] if local.is_dir() else [])
    sys.path[:0] = [str(p) for p in paths]
    sys.dont_write_bytecode = True
    pattern = (sys.argv[1] if len(sys.argv) > 1 else 'test_*') + '.py'
    suite = unittest.defaultTestLoader.discover(str(HERE / 'tests'), pattern=pattern, top_level_dir=str(HERE / 'tests'))
    result = unittest.TextTestRunner(verbosity=1).run(suite)
    raise SystemExit(0 if result.wasSuccessful() else 1)
