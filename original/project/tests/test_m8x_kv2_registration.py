"""Registration must bind actual bytes, reject drift and preserve prior runs."""
import importlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


class KV2RegistrationTests(unittest.TestCase):
    def temp(self):
        base=Path(__file__).resolve().parents[1]/'output/runtime_tests'
        base.mkdir(parents=True,exist_ok=True)
        return tempfile.TemporaryDirectory(dir=base)
    def module(self):
        self.assertIsNotNone(importlib.util.find_spec('agent.m8x_kv2_registration'))
        return importlib.import_module('agent.m8x_kv2_registration')

    def fixture(self,m,root):
        config=json.loads((Path(__file__).resolve().parents[1]/m.CONFIG).read_text())
        for name in m.SOURCES:
            path=root/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text('bound fixture '+name)
        (root/m.CONFIG).write_text(json.dumps(config))
        weights=root/'initial.bin';weights.write_bytes(b'weights-before')
        return config,{'initial_adapter':str(weights)}

    def test_source_and_weight_mutations_are_detected(self):
        m=self.module()
        with self.temp() as tmp:
            root=Path(tmp);config,inputs=self.fixture(m,root)
            reg=m.seal(root,root/'experiment',config,{'runtime':'fixture'},inputs)
            m.verify(root,reg)
            file=root/m.SOURCES[0];saved=file.read_bytes();file.write_bytes(b'changed')
            with self.assertRaises(ValueError):m.verify(root,reg)
            file.write_bytes(saved);Path(inputs['initial_adapter']).write_bytes(b'changed')
            with self.assertRaises(ValueError):m.verify(root,reg)

    def test_registration_cannot_overwrite_or_claim_started_training(self):
        m=self.module()
        with self.temp() as tmp:
            root=Path(tmp);config,inputs=self.fixture(m,root)
            reg=m.seal(root,root/'experiment',config,{},inputs)
            self.assertEqual(reg['formal_training_status'],'not_started')
            self.assertFalse(reg['release_authorized'])
            self.assertTrue((root/'experiment/frozen_source.zip').is_file())
            with self.assertRaises(FileExistsError):m.seal(root,root/'experiment',config,{},inputs)
            reg['config']['actor_lr']*=2
            with self.assertRaises(ValueError):m.verify(root,reg)

    def test_invalid_speed_gate_or_old_output_rejected_before_writing(self):
        m=self.module()
        with self.temp() as tmp:
            root=Path(tmp);config,inputs=self.fixture(m,root)
            config['performance']['minimum_total_speedup']=1.1
            with self.assertRaises(ValueError):m.seal(root,root/'experiment',config,{},inputs)
            self.assertFalse((root/'experiment').exists())

if __name__=='__main__':unittest.main()
