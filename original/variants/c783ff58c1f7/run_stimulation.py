#!/usr/bin/env python3
"""Entry point for an exported BrainStimulation directory."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
if not (Path(__file__).resolve().parent / "brain_stimulation/plan.py").is_file():
    sys.exit("请先在仓库根目录运行 python3 scripts/release/package_stimulation.py，再使用导出包。")

from brain_stimulation.cli import main

if __name__ == "__main__":
    sys.exit(main())
