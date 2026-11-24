import importlib
import tempfile
from pathlib import Path
import unittest


class SourceIdentityTests(unittest.TestCase):
    def test_missing_extra_and_changed_files_cannot_match_a_snapshot(self):
        m=importlib.import_module('agent.m8x_identity')
        root=Path(__file__).resolve().parents[1]/'output/runtime_tests'
        root.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(dir=root) as directory:
            base=Path(directory)
            for name in m.CORE_FILES:
                path=base/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text(name)
            expected=m.source_identity(base)
            m.verify_source_identity(base,expected)
            with self.assertRaises(ValueError):m.verify_source_identity(base,{**expected,'other.py':'a'*64})
            missing=dict(expected);missing.pop(next(iter(missing)))
            with self.assertRaises(ValueError):m.verify_source_identity(base,missing)
            (base/'agent/m8x_contract.py').write_text('changed')
            with self.assertRaises(ValueError):m.verify_source_identity(base,expected)


if __name__=='__main__':unittest.main()
