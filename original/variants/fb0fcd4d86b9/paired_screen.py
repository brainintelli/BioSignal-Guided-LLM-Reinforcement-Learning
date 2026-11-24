"""Sensitivity analysis using the original acquisition software's paired readout."""
import json
import os
import runpy
from pathlib import Path
from register import out,rd,store

plan=rd(out/'protocol.json')
plan.update(ref_offset=-3,offsets=[-3,12],
            screen='Sensitivity analysis fixed before PPO: original single-pulse readout pair [-3,-2) and [12,13). Apply the identical per-electrode criteria to this pair, with >=50 passing remote channels; use the same retained panel for pre and post within each raw file. Repeated replay is not repeated acquisition.',
            shuffle='Exactly two donor trials per training fold. Swap donors WITHIN phase, preserving phase labels, feature multiset and R2. Replay train source A pre/post then B pre/post. Identical prime. Both measured arms evaluated on identical chronologically ordered windows of the third file.')
store(out/'paired-protocol.json',plan)
v=runpy.run_path(str(Path(os.environ['TEMP'])/'codex_live22554_1119.py'))
with v['connect']() as c:
    data=json.loads(v['execute'](c,(out/'extract_hwf.py').read_text(encoding='utf8'),[json.dumps(plan)],timeout=240))
store(out/'paired-windows.json',data)
print({s:(v['candidates'],len(v['channels']),[x['features'] for x in v['windows']]) for s,v in data.items()})
