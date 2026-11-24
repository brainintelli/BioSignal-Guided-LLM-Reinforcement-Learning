"""Independent-process replay using the registered canonical batch policy."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools import audit_m8s3_diagnostic as audit
from tools.benchmark_m8s3_batch import make, verify


if __name__ == '__main__':
    audit.make = make
    audit.verify = verify
    audit.main()
