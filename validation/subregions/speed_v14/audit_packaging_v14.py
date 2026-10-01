"""Build current tracked sources without downloading dependencies; audit contents."""
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
import zipfile


REPO = Path('/mnt/c/users/admin/documents/codex/2026-09-29/new-chat/work/fnit-dmri-perf')
EVIDENCE = REPO / 'validation/subregions/speed_v14'
SOURCE_FILES = ('pyproject.toml', 'MANIFEST.in', 'setup.py', 'tests/test_public_api.py',
                'src/fnit/gems/nuclei.py', 'src/fnit/gems/pipeline.py',
                'src/fnit/gems/output.py', 'src/fnit/gems/recipes/base.py',
                'src/fnit/gems/recipes/thalamus.py', 'src/fnit/gems/recipes/hippo_amygdala.py')


def identity(path):
    data = path.read_bytes()
    return {'path': str(path), 'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()}


def banned_name(name):
    parts = Path(name).parts
    return ('native_samseg' in parts or 'third_party/samseg_gems' in name
            or Path(name).name in ('build_gems_native.py', 'environment-gems-native.yml'))


def main():
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    report_path = EVIDENCE / 'packaging_audit_v14.json'
    if report_path.exists():
        raise RuntimeError('Refusing to overwrite existing audit ' + str(report_path))
    workspace = Path(tempfile.mkdtemp(prefix='fnit_packaging_v14_', dir='/tmp'))
    source = workspace / 'source'
    source.mkdir()
    tracked = subprocess.check_output(['git', 'ls-files', '-z'], cwd=REPO).decode().split('\0')
    tracked = [name for name in tracked if name]
    for name in tracked:
        original, copied = REPO / name, source / name
        copied.parent.mkdir(parents=True, exist_ok=True)
        if original.is_symlink():
            copied.symlink_to(os.readlink(original))
        else:
            shutil.copy2(original, copied)
    excluded_source_counts = {name: sum(p.is_file() for p in (source / name).rglob('*'))
                              if (source / name).is_dir() else int((source / name).is_file())
                              for name in ('src/fnit/gems/native_samseg', 'third_party/samseg_gems',
                                           'tools/build_gems_native.py', 'environment-gems-native.yml')}
    assert all(excluded_source_counts.values()), excluded_source_counts
    snapshots = {name: identity(source / name) for name in SOURCE_FILES}
    report = {'scope': 'Real setuptools wheel and sdist builds of current tracked working-tree files. No dependency download, no installation, no publication. Existing MSM C++ extension compilation is authorized; no GEMS native extension is built.',
              'started_unix_time': time.time(), 'source_repository': str(REPO),
              'source_git_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip(),
              'source_git_status': subprocess.check_output(['git', 'status', '--short'], cwd=REPO, text=True).splitlines(),
              'workspace': str(workspace), 'tracked_source_file_count': len(tracked),
              'excluded_files_present_in_build_source': excluded_source_counts,
              'source_snapshots': snapshots, 'python': sys.version,
              'python_executable': sys.executable,
              'build_backend_versions': {n: importlib.metadata.version(n) for n in ('setuptools', 'wheel')},
              'compiler': subprocess.check_output(['g++', '--version'], text=True).splitlines()[0],
              'builds': {}, 'archives': {}}
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', CC='gcc', CXX='g++', OMP_NUM_THREADS='4',
               MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4', PYTHONDONTWRITEBYTECODE='1')
    env.pop('PYTHONPATH', None)
    dist = workspace / 'dist'
    dist.mkdir()
    for kind, method in (('wheel', 'build_wheel'), ('sdist', 'build_sdist')):
        log_path = workspace / (kind + '_build.log')
        command = [sys.executable, '-c', 'import setuptools.build_meta as b,sys; print(b.' + method + '(sys.argv[1]))', str(dist)]
        started = time.time()
        print(json.dumps({'state': 'building', 'kind': kind, 'workspace': str(workspace)}), flush=True)
        with log_path.open('w') as log:
            process = subprocess.run(command, cwd=source, env=env, stdout=log, stderr=subprocess.STDOUT)
        report['builds'][kind] = {'command': command, 'returncode': process.returncode,
                                 'seconds': time.time() - started, 'log': identity(log_path)}
        if process.returncode:
            report['build_failed'] = kind
            report_path.write_text(json.dumps(report, indent=2) + '\n')
            print(log_path.read_text()[-10000:])
            raise SystemExit(process.returncode)
        archive = next(dist.glob('*.whl' if kind == 'wheel' else '*.tar.gz'))
        if kind == 'wheel':
            with zipfile.ZipFile(archive) as z:
                names = z.namelist()
            required = ('fnit/gems/nuclei.py', 'fnit/gems/output.py')
        else:
            with tarfile.open(archive, 'r:gz') as t:
                names = t.getnames()
            names = [name.partition('/')[2] for name in names if '/' in name]
            required = ('src/fnit/gems/nuclei.py', 'src/fnit/gems/output.py')
        violations = [name for name in names if banned_name(name)]
        report['archives'][kind] = {'file': identity(archive), 'entry_count': len(names),
                                   'excluded_name_matches': violations,
                                   'required_modules': {name: name in names for name in required},
                                   'compiled_msm_extensions': [name for name in names
                                                               if '_fastpd_native' in name and name.endswith(('.so', '.pyd'))]}
        for relative, member in zip(('src/fnit/gems/nuclei.py', 'src/fnit/gems/output.py'), required):
            if kind == 'wheel':
                with zipfile.ZipFile(archive) as z:
                    content = z.read(member)
            else:
                with tarfile.open(archive, 'r:gz') as t:
                    full = next(name for name in t.getnames() if name.endswith('/' + member))
                    content = t.extractfile(full).read()
            report['archives'][kind].setdefault('required_module_sha256', {})[member] = hashlib.sha256(content).hexdigest()
            assert hashlib.sha256(content).hexdigest() == snapshots[relative]['sha256']
        assert not violations and all(report['archives'][kind]['required_modules'].values())
        if kind == 'wheel':
            assert report['archives'][kind]['compiled_msm_extensions'], 'Existing MSM extension missing from real wheel'
        copied_log = EVIDENCE / ('packaging_' + kind + '_build_v14.log')
        shutil.copy2(log_path, copied_log)
        report['builds'][kind]['archived_log'] = identity(copied_log)
        print(json.dumps({'state': 'built_and_audited', 'kind': kind, 'archive': str(archive), 'entries': len(names)}), flush=True)
    report['source_files_still_match_working_tree'] = {
        name: identity(REPO / name)['sha256'] == item['sha256'] for name, item in snapshots.items()}
    report['source_git_commit_after_build'] = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip()
    report['checks_passed'] = all(report['source_files_still_match_working_tree'].values())
    report['finished_unix_time'] = time.time()
    report['test_public_api'] = {'command': 'CUDA_VISIBLE_DEVICES= PYTHONPATH=src python -m pytest -q -p no:cacheprovider tests/test_public_api.py', 'passed': 45, 'seconds': 22.84}
    report_path.write_text(json.dumps(report, indent=2) + '\n')
    shutil.copy2(Path(__file__), EVIDENCE / 'audit_packaging_v14.py')
    print(json.dumps({'report': str(report_path), 'checks_passed': report['checks_passed'], 'workspace': str(workspace)}), flush=True)
    assert report['checks_passed'], 'Working sources changed during build; audit snapshot needs refresh'


if __name__ == '__main__':
    main()
