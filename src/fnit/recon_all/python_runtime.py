"""独立 Python recon-all 的依赖与资源预检；不发现或执行原版程序。"""
from __future__ import annotations
import argparse
import hashlib
import importlib
import importlib.metadata
import json
from pathlib import Path

_DEPENDENCIES = {
    'torch': 'torch', 'numpy': 'numpy', 'scipy': 'scipy', 'numba': 'numba',
    'nibabel': 'nibabel', 'h5py': 'h5py', 'SimpleITK': 'SimpleITK',
    'tifffile': 'tifffile', 'imagecodecs': 'imagecodecs',
    'triton': 'triton', 'nvidia-ml-py': 'pynvml',
}


def check_python_runtime(*, require_nvml: bool = False,
                         assets_dir: str | Path | None = None) -> dict:
    """检查完整 Python CUDA 配方的依赖，并可校验固定 core 资产。

    require_nvml默认False：NVML仅监测，缺失不影响计算；True时缺失报失败。
    assets_dir默认None，不读资源；传目录时复用固定core大小/SHA校验，不下载。
    返回JSON兼容字典：status、dependencies（版本/可加载状态/文件哈希）、
    missing、incompatible、assets及whole_isolation。失败用status='failed'表示；
    CLI另返回非零。依赖导入失败不回退原程序，不初始化CUDA、不查询PATH。
    报告不含安装绝对路径、服务器地址或账号；不代替真实整例隔离验证。
    """
    result = {'status': 'passed', 'dependencies': {}, 'missing': [], 'incompatible': [],
              'assets': {'status': 'not_requested'}, 'whole_isolation': 'not_assessed',
              'native_program_discovery': False}
    for distribution, module_name in _DEPENDENCIES.items():
        required = distribution != 'nvidia-ml-py' or require_nvml
        row = {'required': required, 'module': module_name}
        try:
            module = importlib.import_module(module_name)
            try:
                version = importlib.metadata.version(distribution)
            except importlib.metadata.PackageNotFoundError:
                version = getattr(module, '__version__', None)
                if version is None and distribution == 'SimpleITK':
                    version = module.Version_VersionString()
            file = getattr(module, '__file__', None)
            row.update(status='loadable', version=version,
                       module_sha256=hashlib.sha256(Path(file).read_bytes()).hexdigest() if file else None)
            if distribution == 'SimpleITK' and version != '2.5.6':
                result['incompatible'].append('SimpleITK must be 2.5.6 for the fixed N4 recipe')
            if distribution == 'numpy' and version and int(version.split('.')[0]) >= 2:
                result['incompatible'].append('numpy must be below 2 for the declared environment')
        except (ImportError, OSError, RuntimeError) as error:
            # 不把异常文本中的私有安装路径导出；详细trace留给调用方私有日志。
            row.update(status='unavailable', error_type=type(error).__name__)
            if required:
                result['missing'].append(distribution)
        result['dependencies'][distribution] = row
    if assets_dir is not None:
        from .assets import validate_core_assets, CORE_ASSETS, ASSET_FILES
        try:
            validate_core_assets(assets_dir)
            result['assets'] = {'status': 'verified', 'files': {
                name: {'bytes': ASSET_FILES[name][0], 'sha256': ASSET_FILES[name][1]}
                for name in CORE_ASSETS}}
        except (OSError, ValueError):
            result['assets'] = {'status': 'failed'}
    if result['missing'] or result['incompatible'] or result['assets']['status'] == 'failed':
        result['status'] = 'failed'
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--require-nvml', action='store_true', help='显存监测必须可导入NVML')
    parser.add_argument('--assets-dir', type=Path, help='只校验固定core模板，不下载')
    parser.add_argument('--output', type=Path, help='将typed报告写到该文件')
    args = parser.parse_args(argv)
    report = check_python_runtime(require_nvml=args.require_nvml, assets_dir=args.assets_dir)
    text = json.dumps(report, ensure_ascii=False, indent=2) + '\n'
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
    print(text, end='')
    return 0 if report['status'] == 'passed' else 1

if __name__ == '__main__':
    raise SystemExit(main())
