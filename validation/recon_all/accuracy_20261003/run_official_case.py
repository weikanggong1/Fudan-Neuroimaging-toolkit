"""在独立benchmark目录，从原始T1和空目录运行固定官方recon-all。

输入--config为prepare_baseline生成的official JSON；--output配置禁止覆盖。
核验原始T1 SHA和官方程序清单，固定CPU总线程4/ITK1/RNG1234；不使用GPU。
输出launch.json、command.log、completion.json；成功还绑定recon-all.log及done
的SHA和墙钟（含校验/加载/写出）。失败非零退出并写异常，目录存在不代表完成。
该工具只用于官方参考，生产流程不得调用或读取这些输出。
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
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def run(*, config_path):
    """只运行声明的官方程序；返回完成凭据，错误写报告后抛异常。"""
    start = time.perf_counter()
    config = json.loads(Path(config_path).read_text())
    destination = Path(config['diagnostic_root'])
    destination.mkdir(parents=True, exist_ok=False)
    report = {'execution_status': 'failed', 'code_version': config['code_version'],
              'started_utc': datetime.now(timezone.utc).isoformat(),
              'input_sha256': config['input_sha256'], 'source_kind': 'official',
              'strict_reproduction': 'not_assessed', 'overall_metric_equivalence': 'not_assessed'}
    try:
        subject = Path(config['subjects_dir']) / config['case']
        if subject.exists():
            raise FileExistsError('official subject must start from an empty directory')
        if Path(config['case']).name != config['case'] or config['case'] in ('.', '..'):
            raise ValueError('invalid case id')
        if (config['threads'], config['itk_threads'], config['seed']) != (4, 1, 1234):
            raise ValueError('official resource protocol differs')
        if digest(config['input']) != config['input_sha256']:
            raise ValueError('official raw input SHA differs')
        if not Path(config['fs_license']).is_file():
            raise FileNotFoundError('private official license unavailable')
        manifest = json.loads(Path(config['program_manifest']).read_text())
        for path, sha in manifest['programs'].items():
            if digest(path) != sha:
                raise ValueError('official program changed: ' + Path(path).name)
        home = Path(config['fs_home'])
        executable = home / 'bin/recon-all'
        if str(executable) not in manifest['programs']:
            raise ValueError('official recon-all not covered by frozen manifest')
        Path(config['subjects_dir']).mkdir(parents=True, exist_ok=True)
        env = dict(os.environ)
        env.pop('PYTHONPATH', None)
        env.pop('PYTORCH_NO_CUDA_MEMORY_CACHING', None)
        for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                    'NUMEXPR_NUM_THREADS', 'TF_NUM_INTRAOP_THREADS'):
            env[key] = '4'
        env.update(FREESURFER_HOME=str(home), FS_LICENSE=config['fs_license'],
                   SUBJECTS_DIR=config['subjects_dir'], CUDA_VISIBLE_DEVICES='',
                   ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS='1', TF_NUM_INTEROP_THREADS='1',
                   PATH=str(home / 'bin') + os.pathsep + env.get('PATH', ''))
        setup = subprocess.run(
            ['/bin/bash', '-c', 'source "$1/SetUpFreeSurfer.sh" >/dev/null && env -0',
             'official-setup', str(home)], env=env, capture_output=True, check=True)
        env = {key.decode(): value.decode() for entry in setup.stdout.split(b'\0')
               if entry for key, _, value in [entry.partition(b'=')]}
        # 只保留在内存；启动记录不写全部环境，防止泄露无关凭据。
        env.update(CUDA_VISIBLE_DEVICES='', ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS='1')
        version = subprocess.run([str(executable), '-version'], env=env,
                                 capture_output=True, text=True, check=True)
        if manifest['reported_version'].strip() != version.stdout.strip():
            raise ValueError('official reported version differs from manifest')
        command = [str(executable), '-all', '-i', config['input'], '-s', config['case'],
                   '-sd', config['subjects_dir'], '-openmp', '4', '-itkthreads', '1',
                   '-rng-seed', '1234']
        launch = {**config, 'command': command, 'host': os.uname().nodename,
                  'loadavg': os.getloadavg(), 'script_sha256': digest(__file__),
                  'program_manifest_sha256': digest(config['program_manifest']),
                  'reported_version': version.stdout.strip(),
                  'config_sha256': digest(config_path), 'environment': {
                      k: env[k] for k in ('FREESURFER_HOME', 'CUDA_VISIBLE_DEVICES',
                                         'OMP_NUM_THREADS', 'ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS')},
                  'timing_scope': 'preflight/loading/transfer/compute/output IO',
                  'started_utc': report['started_utc']}
        (destination / 'launch.json').write_text(json.dumps(launch, indent=2) + '\n')
        with (destination / 'command.log').open('w') as log:
            status = subprocess.call(command, env=env, stdout=log, stderr=subprocess.STDOUT)
        report['child_exit_code'] = status
        if status:
            raise RuntimeError('official child returned ' + str(status))
        done, log = subject / 'scripts/recon-all.done', subject / 'scripts/recon-all.log'
        if not done.is_file() or not log.is_file():
            raise RuntimeError('official completion records unavailable')
        report.update(execution_status='complete', exit_code=0, subject=str(subject),
                      done_sha256=digest(done), log_sha256=digest(log),
                      program_manifest_sha256=digest(config['program_manifest']))
    except BaseException as error:
        report.update(error=repr(error), exit_code=report.get('child_exit_code') or 1)
        raise
    finally:
        report.update(command_wall_seconds=time.perf_counter() - start,
                      finished_utc=datetime.now(timezone.utc).isoformat())
        (destination / 'completion.json').write_text(json.dumps(report, indent=2) + '\n')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(config_path=args.config), indent=2))
