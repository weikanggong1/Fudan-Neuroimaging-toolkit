"""在 headcw 复核本轮固定输入、资源及程序；不读取许可证或参考结果影像。"""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform


def fingerprint(path: Path) -> dict:
    """path 为可读文件；返回 size_bytes 与 SHA-256，读取失败时抛异常。"""
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return {'size_bytes': path.stat().st_size, 'sha256': digest.hexdigest()}


def main() -> None:
    """使用已声明的服务器路径；写出新的 runtime_fingerprints_e036f57.json。"""
    root = Path('/cwStorage/home/gongwk/Notebook_code/freesurfer_synth/work/recon_standard_20260929')
    directory = root / 'volume_parity_20260930'
    expected_path = directory / 'provenance.json'
    expected = json.loads(expected_path.read_text())
    build_path = directory / 'build_full_14_20260929.json'
    build = json.loads(build_path.read_text())
    report = {'code_commit': 'e036f57b62b99d2af4cd8853ab2f1e6d2a9f8c68',
              'fingerprint_host': platform.node(), 'utc': datetime.now(timezone.utc).isoformat(),
              'expected_manifest_sha256': fingerprint(expected_path)['sha256'],
              'build_manifest_sha256': fingerprint(build_path)['sha256'], 'mismatches': []}
    folders = {'weights': root / 'weights', 'assets': root / 'assets',
               'binaries': root / 'fnit_main_env/bin',
               'reference_binaries': Path('/public/software/apps/Freesurfer/8.2.0-1/bin')}
    for group, folder in folders.items():
        entries = expected[group]
        if group == 'binaries':
            entries = {name: {'sha256': digest} for name, digest in
                       build['installed_program_sha256'].items()}
        report[group] = {}
        for name, previous in entries.items():
            actual = fingerprint(folder / name)
            report[group][name] = actual
            if any(actual[key] != value for key, value in previous.items()):
                report['mismatches'].append(f'{group}/{name}')
    report['inputs'] = {}
    for name, previous in expected['inputs'].items():
        sid = name.split('_')[0].replace('sub', 'sub-')
        path = root.parents[1] / 'examples/data' / f'{sid}_T1w.nii.gz'
        actual = fingerprint(path)
        report['inputs'][name] = actual
        if actual != previous:
            report['mismatches'].append(name)
    output = directory / 'runtime_fingerprints_e036f57.json'
    with output.open('x') as stream:
        stream.write(json.dumps(report, indent=2) + '\n')
    if report['mismatches']:
        raise ValueError(report['mismatches'])


if __name__ == '__main__':
    main()
