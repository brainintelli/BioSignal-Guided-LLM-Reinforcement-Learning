"""Evaluate registered maps with the frozen runner; parallelize the last seed if GPUs free up."""
from pathlib import Path
import json
import sys

HERE=Path(__file__).resolve().parent

if __name__=='__main__':
    seed=int(sys.argv[1])
    if len(sys.argv)==2 and seed==json.loads((HERE/'design.json').read_text())['seeds'][-1]:
        from split_last import main
        main(seed)
    else:
        sys.path.insert(0,str(HERE.parent))
        from freeze import main
        main(seed, Path(sys.argv[2]) if len(sys.argv)>2 else HERE)
