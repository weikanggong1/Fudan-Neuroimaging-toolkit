"""Stdlib binding checks; no tensor math, compiler, writes or hidden library lookup."""
import hashlib
from pathlib import Path


def identity(path):
    path = Path(path)
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            digest.update(block)
    return {'bytes': path.stat().st_size, 'sha256': digest.hexdigest()}


def check_sources(root, workspace, plan):
    root, workspace = Path(root), Path(workspace)
    result = {}
    for relative, expected in plan['source17'].items():
        actual = identity(root / 'repo' / relative)
        if actual != expected:
            raise RuntimeError('production source changed: ' + relative)
        result['production/' + relative] = actual
    for name, expected in plan['prototype_sources'].items():
        actual = identity(workspace / name)
        if actual != expected:
            raise RuntimeError('new frozen C24 source changed: ' + name)
        result['prototype/' + name] = actual
    for name, expected in plan['old_C72_files'].items():
        actual = identity(root / expected['fnit_relative_path'])
        if actual != {key: expected[key] for key in ('bytes', 'sha256')}:
            raise RuntimeError('old C72 frozen file changed: ' + name)
        result['old_C72/' + name] = actual
    return result


def check_runtime(torch, plan):
    if torch.__version__ != '2.5.1' or torch._C._GLIBCXX_USE_CXX11_ABI or torch.get_default_dtype() != torch.float32:
        raise RuntimeError('accepted Torch2.5.1/ABI0/default FP32 required')
    root = Path(torch.__file__).resolve().parent
    if identity(root / 'lib/libtorch_cpu.so')['sha256'] != plan['libtorch_cpu_sha256']:
        raise RuntimeError('Torch CPU runtime bytes changed')
    for name, digest in plan['headers'].items():
        if identity(root / 'include' / name)['sha256'] != digest:
            raise RuntimeError('Torch header changed: ' + name)


def flags(torch):
    return {'CUDA_initialized': torch.cuda.is_initialized(),
            'oneDNN': bool(torch.backends.mkldnn.enabled),
            'matmul_TF32': bool(torch.backends.cuda.matmul.allow_tf32),
            'cuDNN_TF32': bool(torch.backends.cudnn.allow_tf32),
            'CPU_autocast': bool(torch.is_autocast_enabled('cpu')),
            'grad_enabled': torch.is_grad_enabled(), 'default_dtype': str(torch.get_default_dtype())}
