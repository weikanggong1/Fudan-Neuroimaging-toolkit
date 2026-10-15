"""recon-all CUDA内核的环境编译器选择；不改变算法、精度或CUDA状态。"""
from __future__ import annotations
import hashlib
import importlib
import os
from pathlib import Path
import re
import subprocess
import sys
import threading

_lock = threading.RLock()
_last = None


def _probe(candidate, source, cuda_release):
    candidate = candidate.resolve()
    if not candidate.is_file() or not os.access(candidate, os.X_OK):
        raise ValueError('所选TRITON_PTXAS_PATH不存在或不可执行；不回退其他编译器')
    version = subprocess.run([str(candidate), '--version'], capture_output=True,
                             text=True, check=True, timeout=10).stdout.strip()
    match = re.search(r'release\s+(\d+\.\d+)', version)
    if match is None:
        raise RuntimeError('无法核验ptxas的CUDA版本')
    release = match.group(1)
    if source == 'declared_environment_prefix' and release != cuda_release:
        raise RuntimeError('Conda ptxas版本与torch.version.cuda不匹配；请使用声明环境或显式配置已验证编译器')
    return candidate, {'status': 'available', 'selection_source': source,
                       'compiler_cuda_release': release, 'torch_cuda_release': cuda_release,
                       'matches_torch_cuda': release == cuda_release,
                       'compiler_version_text': version,
                       'compiler_sha256': hashlib.sha256(candidate.read_bytes()).hexdigest(),
                       'changes_precision': False, 'gpu_execution_verified': False}


def _select(device, compiler_path):
    if not str(device).startswith('cuda'):
        return None, {'status': 'not_required', 'device_type': 'cpu'}
    import torch
    cuda_release = torch.version.cuda
    if compiler_path is not None:
        return _probe(Path(compiler_path), 'explicit_parameter', cuda_release)
    if os.environ.get('TRITON_PTXAS_PATH'):
        return _probe(Path(os.environ['TRITON_PTXAS_PATH']),
                      'explicit_environment', cuda_release)
    choices = [Path(sys.prefix) / 'bin/ptxas',
               Path(sys.prefix) / 'targets/x86_64-linux/bin/ptxas']
    # 两个声明位置可能分别指向残留工具链和当前CUDA工具链；只选择核验通过者。
    for candidate in choices:
        if not candidate.is_file():
            continue
        try:
            return _probe(candidate, 'declared_environment_prefix', cuda_release)
        except (OSError, ValueError, RuntimeError, subprocess.SubprocessError):
            continue
    raise RuntimeError('CUDA recon-all需要当前Python环境中可执行且匹配torch.version.cuda的ptxas；请按environment.yml安装，不搜索系统软件目录')


def inspect_triton_compiler(*, device: str = 'cuda:0',
                            compiler_path: str | Path | None = None) -> dict:
    """只读核验编译器版本/SHA，不初始化CUDA、不改变环境或Triton缓存。

    device默认cuda:0，CPU返回not_required；compiler_path默认None，尊重显式
    TRITON_PTXAS_PATH，否则依次核验当前Python环境两处ptxas，选择匹配Torch
    CUDA的可执行程序。显式参数/环境设置核验失败时不回退。
    返回无私有绝对路径的JSON字典，含选择来源、CUDA版本、完整版本和SHA。
    无/非法编译器、默认候选版本不符或探测失败抛异常。显式候选记录版本
    差异，不据此认定驱动兼容；没有原软件算法等价CLI。
    """
    _, info = _select(device, compiler_path)
    return info


def configure_triton_compiler(*, device: str = 'cuda:0',
                              compiler_path: str | Path | None = None) -> dict:
    """首次FNIT JIT前选择环境ptxas；已初始化Torch CUDA的API也可调用。

    参数/核验/返回结构沿inspect_triton_compiler；另含configured和既有编译器
    缓存状态。只设置TRITON_PTXAS_PATH，不动TF32、autocast、allocator或算法。
    若本进程Triton已缓存其他路径/版本，或本函数已核验的程序路径/文件内容
    改变，抛RuntimeError要求新解释器；不清第三方JIT/磁盘缓存。必须在科学
    JIT前串行调用，不与其他线程的编译并发改配置。重复选择同程序仍核验实际
    缓存，保留首次选择来源；核验失败恢复此前环境值，但不改Triton缓存。
    此接口没有影像输入/输出、坐标空间或原软件CLI；真实GPU成功仍须同输入
    执行验证。示例：configure_triton_compiler(device="cuda:0",  # 目标GPU
    compiler_path=None)  # 使用显式环境设置或声明环境中的匹配编译器
    """
    global _last
    with _lock:
        candidate, info = _select(device, compiler_path)
        if candidate is None:
            return info
        if (_last is not None and
                (str(candidate) != _last[0] or
                 info['compiler_sha256'] != _last[1]['compiler_sha256'])):
            raise RuntimeError('本进程已核验的ptxas路径或文件内容改变；请使用新Python解释器，不复用既有JIT kernel')
        compiler = importlib.import_module('triton.backends.nvidia.compiler')
        path_cache = compiler._path_to_binary.cache_info().currsize
        version_cache = compiler.get_ptxas_version.cache_info().currsize
        if path_cache:
            cached_path, cached_release = compiler._path_to_binary('ptxas')
            if (Path(cached_path).resolve() != candidate or
                    cached_release != info['compiler_cuda_release']):
                raise RuntimeError('本进程Triton已缓存其他ptxas；请在新Python解释器中重试，不复用失败kernel')
        if version_cache and compiler.get_ptxas_version().strip() != info['compiler_version_text']:
            raise RuntimeError('本进程Triton已缓存其他编译器版本；请使用新Python解释器')
        previous_environment = os.environ.get('TRITON_PTXAS_PATH')
        try:
            os.environ['TRITON_PTXAS_PATH'] = str(candidate)
            actual_path, actual_release = compiler._path_to_binary('ptxas')
            if (Path(actual_path).resolve() != candidate or
                    actual_release != info['compiler_cuda_release'] or
                    compiler.get_ptxas_version().strip() != info['compiler_version_text']):
                raise RuntimeError('Triton实际编译器与已核验选择不同；请使用新Python解释器，不继续科学计算')
        except Exception:
            if previous_environment is None:
                os.environ.pop('TRITON_PTXAS_PATH', None)
            else:
                os.environ['TRITON_PTXAS_PATH'] = previous_environment
            raise
        if _last is not None:
            info['selection_source'] = _last[1]['selection_source']
        info.update(configured=True, preexisting_path_cache=bool(path_cache),
                    preexisting_version_cache=bool(version_cache),
                    caches_cleared=False, idempotent_reuse=_last is not None)
        _last = (str(candidate), dict(info))
        return info
