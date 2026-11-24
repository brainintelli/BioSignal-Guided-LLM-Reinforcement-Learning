"""Bind optional GPU implementations without modifying the shared runtime."""
import hashlib
import importlib
import importlib.metadata
import importlib.util
import inspect
from pathlib import Path


def resolved_implementation(function):
    seen = set()
    while callable(function) and id(function) not in seen:
        seen.add(id(function))
        if inspect.isfunction(function):
            closure = inspect.getclosurevars(function).nonlocals
            implementation = closure.get('implementation')
            if callable(implementation):
                return implementation
        function = getattr(function, '__wrapped__', None)
    raise ValueError('Cannot identify the executed Transformers fallback implementation')


def package_files(package_directory, binary=None):
    root = Path(package_directory).resolve()
    files = {root.name + '/' + p.relative_to(root).as_posix(): p
             for p in sorted(root.rglob('*.py')) if '__pycache__' not in p.parts}
    if binary is not None:
        path = Path(binary).resolve()
        files[path.name] = path
    result = {}
    for name, path in files.items():
        with path.open('rb') as stream:
            result[name] = hashlib.file_digest(stream, 'sha256').hexdigest()
    return result


def kernel_identity():
    from transformers.models.qwen3_5 import modeling_qwen3_5 as qwen
    implementations = {}
    for name in ('causal_conv1d_fn', 'causal_conv1d_update',
                 'torch_chunk_gated_delta_rule', 'torch_recurrent_gated_delta_rule'):
        function = resolved_implementation(getattr(qwen, name))
        source = Path(inspect.getsourcefile(function))
        implementations[name] = dict(module=function.__module__, name=function.__name__,
                                    source_sha256=hashlib.sha256(source.read_bytes()).hexdigest())
    packages = {}
    for name, distribution in (('causal_conv1d', 'causal-conv1d'), ('fla', 'fla-core')):
        spec = importlib.util.find_spec(name)
        if spec is None:
            packages[name] = None
            continue
        package = importlib.import_module(name)  # Broken extensions must fail, never silently fall back.
        binary = importlib.import_module('causal_conv1d_cuda').__file__ if name == 'causal_conv1d' else None
        packages[name] = dict(version=importlib.metadata.version(distribution),
                              files_sha256=package_files(Path(package.__file__).parent, binary))
        prefix = 'causal_conv1d' if name == 'causal_conv1d' else 'fla'
        relevant = ('causal_conv1d_fn', 'causal_conv1d_update') if name == 'causal_conv1d' else (
            'torch_chunk_gated_delta_rule', 'torch_recurrent_gated_delta_rule')
        packages[name]['executed_by_transformers'] = {
            key: implementations[key]['module'].startswith(prefix + '.') for key in relevant}
    return dict(implementations=implementations, **packages)
