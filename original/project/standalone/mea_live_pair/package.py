"""Build a no-credentials, no-binding deployment bundle from the checked source."""
from hashlib import sha256
from pathlib import Path
import sys
from zipfile import ZipFile, ZIP_DEFLATED


def build(out):
    src=Path(__file__).resolve().parent
    with ZipFile(out,'x',compression=ZIP_DEFLATED) as z:
        for f in (src/'launch.py',src/'control.py',src/'quiet_baseline.py',src/'select_sites.py',src/'livepair/rtx4090_probe.json',
                  *sorted((src/'livepair').glob('*.py'))):
            z.write(f,f.relative_to(src))
        for folder,packages in dict(mea_online=['mea_online'],mea_binary_r2=['mea_binary_r2'],
                 mea_state2_r2=['mea_state2_r2'],mea_stimulation_v2=['mea_stim_v2','vendor']).items():
            for pkg in packages:
                for f in sorted((src.parent/folder/pkg).rglob('*.py')):
                    z.write(f,f.relative_to(src.parent))
    print(sha256(out.read_bytes()).hexdigest())


if __name__=='__main__':build(Path(sys.argv[1]))
