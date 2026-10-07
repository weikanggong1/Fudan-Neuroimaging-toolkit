"""隔离的 white.preaparc 四臂同输入诊断；私有计划和结果不得发布。"""
from __future__ import annotations

import argparse
import contextlib
import fcntl
import hashlib
import json
import os
import platform
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import traceback

ARM_NAMES = ('official_A', 'official_B', 'conda_standard', 'conda_fast')


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def binding(record):
    path = Path(record['path'])
    if not path.is_absolute() or not path.is_file():
        raise ValueError(f'absolute regular file required: {path}')
    before = path.stat()
    result = {'path': str(path), 'size': before.st_size, 'sha256': digest(path)}
    after = path.stat()
    if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns) != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns):
        raise RuntimeError(f'file changed while reading: {path}')
    if result['size'] != record['size'] or result['sha256'] != record['sha256']:
        raise ValueError(f'binding mismatch: {path}')
    return result


def source_binding(root):
    """SHA256 of sorted UTF-8 relative-path NUL size NUL file-SHA LF records."""
    src = Path(root) / 'src'
    if not src.is_dir():
        raise ValueError('source_root/src missing')
    records = []
    for path in sorted(src.rglob('*')):
        if '__pycache__' in path.parts or path.suffix == '.pyc':
            continue
        if path.is_symlink():
            raise ValueError(f'source symlink refused: {path}')
        if path.is_file():
            meta = {'path': str(path), 'size': path.stat().st_size, 'sha256': digest(path)}
            binding(meta)
            records.append(f"{path.relative_to(src).as_posix()}\0{meta['size']}\0{meta['sha256']}\n")
    return hashlib.sha256(''.join(records).encode()).hexdigest()


def relative_path(value):
    path = Path(value)
    if path.is_absolute() or '..' in path.parts or not path.parts or path.parts[0] not in ('mri', 'surf'):
        raise ValueError(f'invalid subject destination: {value}')
    return path


def command(binary, subject, hemi):
    return [binary, '--adgws-in', str(subject / f'surf/autodet.gw.stats.{hemi}.dat'),
            '--wm', str(subject / 'mri/wm.mgz'), '--threads', '4', '--invol',
            str(subject / 'mri/brain.finalsurfs.mgz'), f'--{hemi}', '--i',
            str(subject / f'surf/{hemi}.orig'), '--o',
            str(subject / f'surf/{hemi}.white.preaparc'), '--white', '--seg',
            str(subject / 'mri/aseg.presurf.mgz'), '--restore-255', '--nsmooth', '5',
            '--rip-bg-no-annot', '--rip-bg', '--rip-bg-lof', '--restore-255',
            '--outvol', str(subject / 'mri/mrisps.wpa.mgz')]


@contextlib.contextmanager
def lock(path, timeout):
    if not Path(path).is_absolute():
        raise ValueError('shared_lock must be absolute')
    with open(path, 'a') as handle:
        start = time.monotonic()
        while True:
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.monotonic() - start >= timeout:
                    raise TimeoutError('shared lock wait expired')
                time.sleep(min(0.2, timeout))
        try:
            yield time.monotonic() - start
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def validate_output(plan, output):
    """Reject protected output locations before creating any files."""
    resolved = Path(output).resolve()
    required = {f"surf/{plan['hemi']}.orig", f"surf/autodet.gw.stats.{plan['hemi']}.dat", 'mri/wm.mgz', 'mri/brain.finalsurfs.mgz', 'mri/aseg.presurf.mgz'}
    inputs = [Path(item['path']).resolve() for item in plan['inputs'] if item['relative_path'] in required]
    if not inputs:
        raise ValueError('required inputs missing for output boundary check')
    common = Path(os.path.commonpath(inputs))
    if len(inputs) == 1 or common.is_file():
        common = common.parent
    protected = [Path(plan['source_root']).resolve(), common]
    protected.extend(Path(arm['assets_dir']).resolve() for arm in plan['arms'])
    protected.extend(Path(arm['binary']['path']).resolve().parent for arm in plan['arms'])
    protected.extend(Path(item['path']).resolve().parent for item in plan['inputs'])
    for directory in protected:
        if resolved == directory or resolved.is_relative_to(directory):
            raise ValueError(f'output inside protected tree: {directory}')


def official_repeat_binding(arms):
    return all(arms[0][key] == arms[1][key] for key in ('binary', 'assets_dir', 'assets', 'license_path'))


def runtime_metadata():
    cpu = None
    cpuinfo = Path('/proc/cpuinfo')
    if cpuinfo.is_file():
        cpu = next((line.split(':', 1)[1].strip() for line in cpuinfo.read_text().splitlines() if line.startswith('model name')), None)
    return {'host': platform.node(), 'cpu_model': cpu, 'cpu_affinity': sorted(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else None, 'threads': 4, 'python': {'path': str(Path(sys.executable).resolve()), 'sha256': digest(sys.executable), 'version': sys.version}, 'cuda_disabled': True, 'LD_LIBRARY_PATH_private': os.environ.get('LD_LIBRARY_PATH'), 'dynamic_library_verification': 'not_verified: inherited LD_LIBRARY_PATH and system loader dependencies are outside the explicit program/asset manifest'}


def run(plan_path, output):
    started = time.monotonic()
    output = Path(output)
    if not output.is_absolute():
        raise ValueError('--output must be absolute')
    plan_hash = digest(plan_path)
    plan = json.loads(Path(plan_path).read_text())
    if digest(plan_path) != plan_hash:
        raise RuntimeError('plan changed before output boundary check')
    validate_output(plan, output)
    output.mkdir(parents=False, exist_ok=False)
    report = {'schema': 'fnit-prewhite-same-input-receipt-v1', 'runner_sha256': digest(__file__), 'status': 'running', 'arms': [], 'comparisons': {}, 'lock_wait_seconds': None}
    try:
        report['runtime'] = runtime_metadata()
        report['expected_sourceSHA'] = plan['srcSHA']
        report['plan_sha256'] = plan_hash
        if plan['threads'] != 4 or plan['hemi'] not in ('lh', 'rh'):
            raise ValueError('threads=4 and hemi=lh/rh required')
        if not official_repeat_binding(plan['arms']):
            raise ValueError('official A/B must repeat the identical program and resources')
        if [arm['name'] for arm in plan['arms']] != list(ARM_NAMES):
            raise ValueError(f'exact sequential arms required: {ARM_NAMES}')
        if not 0 < plan['timeout_seconds'] <= 86400 or not 0 < plan['lock_timeout_seconds'] <= 3600:
            raise ValueError('timeouts outside finite supported range')
        required = {f"surf/{plan['hemi']}.orig", f"surf/autodet.gw.stats.{plan['hemi']}.dat", 'mri/wm.mgz', 'mri/brain.finalsurfs.mgz', 'mri/aseg.presurf.mgz'}
        destinations = [str(relative_path(item['relative_path'])) for item in plan['inputs']]
        if len(set(destinations)) != len(destinations) or not required.issubset(destinations):
            raise ValueError('missing or duplicate command inputs')
        if any(name.endswith('.white.preaparc') or name == 'mri/mrisps.wpa.mgz' for name in destinations):
            raise ValueError('output supplied as input')
        root = Path(plan['source_root'])
        if not root.is_absolute():
            raise ValueError('source_root must be absolute')
        with lock(plan['shared_lock'], plan['lock_timeout_seconds']) as waited:
            report['lock_wait_seconds'] = waited
            def verify_all():
                if digest(plan_path) != plan_hash:
                    raise RuntimeError('plan changed')
                actual_source_sha = source_binding(root)
                report['sourceSHA'] = actual_source_sha
                if actual_source_sha != plan['srcSHA']:
                    raise RuntimeError('source binding changed')
                for item in plan['inputs']:
                    binding(item)
                for arm in plan['arms']:
                    binding(arm['binary'])
                    if not os.access(arm['binary']['path'], os.X_OK):
                        raise ValueError('binary is not executable')
                    if not Path(arm['assets_dir']).is_absolute() or not Path(arm['assets_dir']).is_dir():
                        raise ValueError('absolute assets directory required')
                    if not Path(arm['license_path']).is_absolute() or not Path(arm['license_path']).is_file():
                        raise ValueError('license file missing (contents are never read)')
                    if not arm['assets']:
                        raise ValueError('explicit bound asset files required')
                    for asset in arm['assets']:
                        binding(asset)
                        if Path(asset['path']).resolve() == Path(arm['license_path']).resolve():
                            raise ValueError('license contents must not be hashed')
            verify_all()
            sys.dont_write_bytecode = True
            sys.path.insert(0, str(root / 'src'))
            import fnit
            from fnit.recon_all.mris_remove_intersection_python import mark_intersections
            import nibabel.freesurfer as fs
            import numpy as np
            if not Path(fnit.__file__).resolve().is_relative_to((root / 'src').resolve()):
                raise RuntimeError('fnit imported from unexpected source')
            reference_faces = fs.read_geometry(next(item['path'] for item in plan['inputs'] if item['relative_path'] == f"surf/{plan['hemi']}.orig"))[1]
            meshes = {}
            for arm in plan['arms']:
                verify_all()
                arm_start = time.monotonic()
                subject = output / arm['name'] / 'subject'
                subject.mkdir(parents=True)
                (subject / 'mri').mkdir()
                (subject / 'surf').mkdir()
                copied = []
                for item in plan['inputs']:
                    destination = subject / relative_path(item['relative_path'])
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(item['path'], destination)
                    record = dict(item, path=str(destination))
                    copied.append(binding(record))
                entry = {'name': arm['name'], 'input_before': copied, 'binary_before': binding(arm['binary']), 'license_exists': True}
                report['arms'].append(entry)
                env = dict(os.environ)
                for name in ('LD_PRELOAD', 'LD_AUDIT', 'FS_LICENSE', 'SUBJECTS_DIR'):
                    env.pop(name, None)
                env['PATH'] = os.pathsep.join(dict.fromkeys((str(Path(sys.executable).resolve().parent), str(Path(arm['binary']['path']).resolve().parent), '/usr/bin', '/bin')))
                entry['runtime_environment'] = {'PATH': env['PATH'], 'CUDA_VISIBLE_DEVICES': '', 'threads': 4, 'LD_LIBRARY_PATH_private': env.get('LD_LIBRARY_PATH')}
                env.update(NUMBA_NUM_THREADS='4', CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4', MKL_NUM_THREADS='4', NUMEXPR_NUM_THREADS='4', ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS='4', OMP_DYNAMIC='FALSE', FREESURFER_HOME=arm['assets_dir'], FS_LICENSE=arm['license_path'], SUBJECTS_DIR=str(subject.parent))
                argv = command(arm['binary']['path'], subject, plan['hemi'])
                entry['argv'] = argv
                exec_start = time.monotonic()
                log = subject.parent / 'command.log'
                with log.open('wb') as stream:
                    try:
                        process = subprocess.Popen(argv, cwd=subject / 'mri', env=env, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
                        entry['exitcode'] = process.wait(timeout=plan['timeout_seconds'])
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait()
                        entry['exitcode'] = process.returncode
                        entry['timed_out'] = True
                entry['exec_seconds'] = time.monotonic() - exec_start
                entry['log_sha256'] = digest(log)
                entry['binary_after'] = binding(arm['binary'])
                entry['input_after'] = [binding(record) for record in copied]
                verify_all()
                if entry.get('timed_out') or entry['exitcode'] != 0:
                    raise RuntimeError(f"arm failed: {arm['name']}")
                surface = subject / f"surf/{plan['hemi']}.white.preaparc"
                vertices, faces = fs.read_geometry(str(surface))
                finite = bool(np.isfinite(vertices).all())
                valid_faces = bool(faces.ndim == 2 and faces.shape[1] == 3 and faces.min() >= 0 and faces.max() < len(vertices))
                if not finite or not valid_faces:
                    raise RuntimeError('invalid surface coordinates or face indices')
                edges = np.sort(np.concatenate((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]])), axis=1)
                unique_edges, edge_counts = np.unique(edges, axis=0, return_counts=True)
                entry['geometry'] = {'finite': finite, 'vertices': len(vertices), 'faces': len(faces), 'euler': int(len(vertices) - len(unique_edges) + len(faces)), 'closed': bool(np.all(edge_counts == 2)), 'faces_preserved': bool(np.array_equal(faces, reference_faces)), 'self_intersecting_faces': mark_intersections(vertices, faces)[1], 'coords_f4_sha256': hashlib.sha256(np.asarray(vertices, dtype='<f4').tobytes()).hexdigest(), 'faces_i4_sha256': hashlib.sha256(np.asarray(faces, dtype='<i4').tobytes()).hexdigest()}
                entry['outputs'] = []
                for product in (surface, subject / 'mri/mrisps.wpa.mgz'):
                    entry['outputs'].append({'path': str(product), 'size': product.stat().st_size, 'sha256': digest(product)})
                meshes[arm['name']] = (vertices, faces)
                entry['arm_wall_seconds'] = time.monotonic() - arm_start
            base_vertices, base_faces = meshes['official_A']
            for name in ARM_NAMES[1:]:
                vertices, faces = meshes[name]
                corresponding = vertices.shape == base_vertices.shape and np.array_equal(faces, base_faces)
                comparison = {'ordered_faces_match': bool(corresponding), 'units': 'mm'}
                if corresponding:
                    distances = np.linalg.norm(vertices - base_vertices, axis=1)
                    comparison.update(max_mm=float(distances.max()), p99_mm=float(np.percentile(distances, 99)), mean_mm=float(distances.mean()), coords_f4_bytes_identical=bool(np.asarray(vertices, dtype='<f4').tobytes() == np.asarray(base_vertices, dtype='<f4').tobytes()), num_different_vertices=int(np.count_nonzero(np.any(vertices != base_vertices, axis=1))), num_different_coords=int(np.count_nonzero(vertices != base_vertices)))
                else:
                    comparison['status'] = 'vertex_correspondence_not_proven'
                report['comparisons'][name + '_to_official_A'] = comparison
            verify_all()
            report['status'] = 'completed'
            report['zero_intersection_gate'] = all(arm['geometry']['self_intersecting_faces'] == 0 for arm in report['arms'])
    except Exception as error:
        report.update(status='failed', error=str(error), traceback=traceback.format_exc())
    finally:
        report['wall_seconds_including_lock_wait'] = time.monotonic() - started
        report['wall_seconds_excluding_lock_wait'] = report['wall_seconds_including_lock_wait'] - (report['lock_wait_seconds'] or 0)
        (output / 'receipt.json').write_text(json.dumps(report, indent=2) + '\n')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    report = run(args.plan, args.output)
    print(json.dumps({'status': report['status'], 'receipt': str(args.output / 'receipt.json')}))
    raise SystemExit(0 if report['status'] == 'completed' else 1)


if __name__ == '__main__':
    main()
