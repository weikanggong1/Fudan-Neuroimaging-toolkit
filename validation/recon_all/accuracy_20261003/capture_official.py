"""冻结独立官方benchmark的程序、权重、图谱与版本。

输入fs_home是已经安装的官方参考目录；names_path包含过去真实整例出现的
program_names、当前声明assets和weight_names。输出新JSON，记录SHA-256、
大小、缺失资源、版本、主机及采集墙钟。只读取公开软件资源，不读取许可证。
这不是安装工具，不能用于FNIT生产调用。后续发现新的实际程序须补录哈希。
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time


def digest(path):
    """对公开程序/资产流式计算SHA-256，路径不可读时抛异常。"""
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def capture(*, fs_home, names_path, output_path):
    """用具名路径参数采集；目标已存在则拒绝覆盖，返回清单字典。"""
    start = time.perf_counter()
    home, output = Path(fs_home).resolve(), Path(output_path)
    if output.exists():
        raise FileExistsError(output)
    names = json.loads(Path(names_path).read_text())
    programs, resources, missing = {}, {}, []
    for name in names['program_names']:
        p = home / 'bin' / name
        if not p.is_file():
            raise FileNotFoundError('observed official program unavailable: ' + name)
        programs[str(p)] = digest(p)
    for p in (home / 'SetUpFreeSurfer.sh', home / 'build-stamp.txt',
              home / 'python/bin/python3'):
        if p.is_file():
            programs[str(p)] = digest(p)
    for relative in names['assets'] + ['models/' + n for n in names['weight_names']]:
        p = home / relative
        if not p.is_file():
            missing.append(relative)
            continue
        resources[str(p)] = {'sha256': digest(p), 'bytes': p.stat().st_size}
    env = dict(os.environ, FREESURFER_HOME=str(home))
    version = subprocess.run([str(home / 'bin/recon-all'), '-version'], env=env,
                             capture_output=True, text=True, check=True)
    report = {'scope': 'official isolated benchmark only', 'host': os.uname().nodename,
              'created_utc': datetime.now(timezone.utc).isoformat(),
              'reported_version': version.stdout.strip(), 'programs': programs,
              'resources': resources, 'declared_resources_missing_in_official': missing,
              'names_sha256': digest(names_path), 'script_sha256': digest(__file__),
              'seconds': time.perf_counter() - start,
              'coverage': 'observed previous fixed-version commands; extend if new commands appear',
              'physical_isolation_verified': False}
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x') as stream:
        json.dump(report, stream, indent=2)
        stream.write('\n')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fs-home', type=Path, required=True)
    parser.add_argument('--names', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = capture(fs_home=args.fs_home, names_path=args.names, output_path=args.output)
    print(json.dumps({'programs': len(result['programs']), 'resources': len(result['resources']),
                      'missing_declared_resources': result['declared_resources_missing_in_official'],
                      'version': result['reported_version']}))
