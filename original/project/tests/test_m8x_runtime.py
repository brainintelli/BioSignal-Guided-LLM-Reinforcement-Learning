"""Runtime records must identify executed implementations, not installed names."""
import functools
import importlib
import tempfile
from pathlib import Path
import unittest


class RuntimeBindingTests(unittest.TestCase):
    def module(self): return importlib.import_module('agent.m8x_runtime')

    def test_resolves_actual_fallback_closure_through_wrapper(self):
        def original(x): return x
        def optimized(x): return x + 1
        def decorate(torch_function, implementation):
            @functools.wraps(torch_function)
            def wrapped(x): return implementation(x)
            return wrapped
        m=self.module()
        self.assertIs(m.resolved_implementation(decorate(original, optimized)), optimized)
        self.assertIs(m.resolved_implementation(decorate(original, original)), original)
        with self.assertRaises(ValueError): m.resolved_implementation(original)

    def test_source_digest_changes_on_actual_binary_change_not_cache(self):
        m=self.module()
        root=Path(__file__).resolve().parents[1]/'output/runtime_tests'
        root.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(dir=root) as directory:
            base=Path(directory);package=base/'kernel';package.mkdir()
            (package/'__init__.py').write_text('version=1')
            binary=base/'kernel_cuda.so';binary.write_bytes(b'old-binary')
            before=m.package_files(package,binary)
            self.assertEqual(set(before),{'kernel/__init__.py','kernel_cuda.so'})
            (package/'__pycache__').mkdir()
            (package/'__pycache__/ignored.pyc').write_bytes(b'generated-cache')
            self.assertEqual(before,m.package_files(package,binary))
            binary.write_bytes(b'new-binary')
            self.assertNotEqual(before,m.package_files(package,binary))


if __name__=='__main__':unittest.main()
