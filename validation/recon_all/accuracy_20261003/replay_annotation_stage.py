"""从带SHA的FNIT自产检查点复制并重放双侧annotation；仅阶段评估。

调用者必须在固定GPU共享锁和自有进程清理包装内运行。本脚本不取锁，
不读取官方输出，不更改原检查点，不运行原始T1整例。基线不传startup参数；
候选可明确传 --startup-wait-seconds 30，缺少对应API时报错，不回退重试。
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import importlib
import inspect
import json
import math
import os
import re
from pathlib import Path
import shutil
import struct
import sys
import time
import traceback

GPU_UUID = 'GPU-26e41f63-1a65-6b3e-5370-fa9a2934ca8e'
HEMISPHERES = ('lh', 'rh')
ATLASES = ('aparc', 'aparc.a2009s', 'aparc.DKTatlas')
SUBJECT_INPUTS = ('mri/aseg.presurf.mgz',) + tuple(
    relative for hemi in HEMISPHERES for relative in
    (f'surf/{hemi}.smoothwm', f'surf/{hemi}.sphere.reg', f'label/{hemi}.cortex.label'))
ASSET_INPUTS = tuple(f'average/{hemi}.{prefix}.atlas.acfb40.noaparc.i12.2016-08-02.gcs'
                     for hemi in HEMISPHERES for prefix in ('DKaparc', 'CDaparc', 'DKTaparc')) + (
                         'lib/bem/ic4.tri', 'lib/bem/ic7.tri')
THREAD_KEYS = ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS',
               'ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS', 'NUMBA_NUM_THREADS')



def declared_gpu_uuid(value):
    """Require a full physical GPU UUID; aliases and multi-device lists are rejected."""
    if not isinstance(value, str) or re.fullmatch(
            r'GPU-[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}', value) is None:
        raise argparse.ArgumentTypeError('gpu-uuid must be one full GPU-xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx UUID')
    return value


def normalized_actual_uuid(value):
    # Torch device properties may omit the GPU- prefix; never accept a short UUID.
    if not isinstance(value, str):
        raise ValueError('actual device UUID must be a string')
    full = value if value.startswith('GPU-') else 'GPU-' + value
    return declared_gpu_uuid(full)[4:].lower()


def worker_device_receipt(group, expected, require_both):
    receipt = {'expected_uuid': expected, 'workers': {}, 'errors': []}
    workers = group.get('workers', {}) if isinstance(group, dict) else {}
    for hemi in HEMISPHERES:
        worker = workers.get(hemi, {})
        if worker.get('status') != 'complete':
            if require_both:
                receipt['errors'].append(hemi + ': completed worker report missing')
            continue
        actual = worker.get('actual_cuda_device', {}).get('uuid')
        row = {'actual_uuid': actual, 'matches_declared': False}
        receipt['workers'][hemi] = row
        try:
            row['normalized_uuid'] = normalized_actual_uuid(actual)
            row['matches_declared'] = row['normalized_uuid'] == normalized_actual_uuid(expected)
            if not row['matches_declared']:
                raise ValueError('worker device differs from declared GPU UUID')
        except (ValueError, argparse.ArgumentTypeError) as error:
            receipt['errors'].append(hemi + ': ' + str(error))
    receipt['verified'] = not receipt['errors'] and len(receipt['workers']) == 2
    return receipt


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def file_receipt(path):
    path = Path(path)
    return {'path': str(path.resolve()), 'size_bytes': path.stat().st_size, 'sha256': sha(path)}


def write(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


def protected_output(output, checkpoint, assets):
    """防止在原输入内部写入或把原输入包含在新输出内。"""
    for original in (checkpoint, assets):
        if output == original or output in original.parents or original in output.parents:
            raise ValueError('output must be separate from original checkpoint and assets')
    if output.exists():
        raise FileExistsError('output must be a new directory')


def manifest_bindings(manifest, checkpoint, assets):
    """验证15项数据/资产schema；后5项源码只保留来源，不限制候选源码。"""
    entries = manifest.get('replay_files')
    if not isinstance(entries, list) or manifest.get('input_file_count', 15) != 15 or len(entries) < 15:
        raise ValueError('input manifest requires first15 data/assets replay_files')
    original = Path(manifest['subject_checkpoint'])
    original_assets = manifest.get('assets_root') or manifest.get(
        'reconstructed_request_template', {}).get('kwargs', {}).get('assets')
    if not original_assets:
        raise ValueError('manifest must declare assets_root or reconstructed_request_template.kwargs.assets')
    original_assets = Path(original_assets)
    bindings = []
    seen = set()
    for entry in entries[:15]:
        if not isinstance(entry, dict) or not {'path', 'size_bytes', 'sha256'} <= entry.keys():
            raise ValueError('invalid data file receipt')
        checksum = entry['sha256']
        if not isinstance(checksum, str) or len(checksum) != 64 or any(c not in '0123456789abcdef' for c in checksum):
            raise ValueError('invalid SHA256 in data receipt')
        if not isinstance(entry['size_bytes'], int) or isinstance(entry['size_bytes'], bool) or entry['size_bytes'] < 0:
            raise ValueError('invalid file size')
        historical = Path(entry['path'])
        try:
            relative = str(historical.relative_to(original))
            kind, actual = 'subject', checkpoint / relative
            if relative not in SUBJECT_INPUTS:
                raise ValueError('manifest subject file outside annotation inputs')
        except ValueError as error:
            if original == historical or original in historical.parents:
                raise error
            relative = str(historical.relative_to(original_assets))
            kind, actual = 'asset', assets / relative
            if relative not in ASSET_INPUTS:
                raise ValueError('manifest asset outside fixed annotation assets')
        identity = (kind, relative)
        if identity in seen:
            raise ValueError('duplicate data receipt')
        seen.add(identity)
        bindings.append({'kind': kind, 'relative': relative, 'historical_receipt': entry,
                         'actual': actual})
    expected = {('subject', name) for name in SUBJECT_INPUTS} | {('asset', name) for name in ASSET_INPUTS}
    if seen != expected:
        raise ValueError('manifest does not bind all7 subject files and all8 assets')
    return bindings


def verify_bindings(bindings, copied_subject=None):
    receipts = []
    for binding in bindings:
        path = (copied_subject / binding['relative'] if copied_subject is not None
                and binding['kind'] == 'subject' else binding['actual'])
        actual = file_receipt(path)
        expected = binding['historical_receipt']
        if actual['size_bytes'] != expected['size_bytes'] or actual['sha256'] != expected['sha256']:
            raise ValueError('annotation input receipt differs: ' + str(path))
        receipts.append({'kind': binding['kind'], 'relative': binding['relative'], **actual})
    return receipts


def array_sha(array, dtype):
    import numpy as np
    canonical = np.ascontiguousarray(array, dtype=dtype)
    return hashlib.sha256(canonical.tobytes()).hexdigest()


def mesh_receipts(subject):
    import nibabel.freesurfer.io as fsio
    receipts = {}
    for hemi in HEMISPHERES:
        previous_faces = None
        for name in ('smoothwm', 'sphere.reg'):
            path = subject / 'surf' / f'{hemi}.{name}'
            coordinates, faces = fsio.read_geometry(str(path))
            row = {'file': file_receipt(path), 'vertices': len(coordinates), 'faces': len(faces),
                   'coordinate_array_sha256_le_f64': array_sha(coordinates, '<f8'),
                   'ordered_faces_sha256_le_i64': array_sha(faces, '<i8'),
                   'space': 'surface RAS, millimetres; sphere.reg retains ordered vertex correspondence'}
            if previous_faces is not None and row['ordered_faces_sha256_le_i64'] != previous_faces:
                raise ValueError(f'{hemi} smoothwm and sphere.reg ordered topology differ')
            previous_faces = row['ordered_faces_sha256_le_i64']
            receipts[f'{hemi}.{name}'] = row
    return receipts


def annotation_receipts(subject, destination, meshes):
    import nibabel.freesurfer.io as fsio
    import numpy as np
    destination.mkdir()
    receipts = {}
    for hemi in HEMISPHERES:
        for atlas in ATLASES:
            key = f'{hemi}.{atlas}'
            path = subject / 'label' / (key + '.annot')
            original_ids, ctab, names = fsio.read_annot(str(path), orig_ids=True)
            table_indices, second_ctab, second_names = fsio.read_annot(str(path), orig_ids=False)
            with path.open('rb') as stream:
                vertex_count = struct.unpack('>i', stream.read(4))[0]
                vertex_labels = np.frombuffer(stream.read(vertex_count * 8), dtype='>i4').reshape(vertex_count, 2)
            order = vertex_labels[:, 0]
            if vertex_count != meshes[f'{hemi}.smoothwm']['vertices']:
                raise ValueError('annotation vertex count differs from ordered input mesh: ' + key)
            if not np.array_equal(order, np.arange(vertex_count)):
                raise ValueError('annotation records are not in input vertex index order: ' + key)
            if not np.array_equal(original_ids, vertex_labels[:, 1]):
                raise ValueError('fsio labels differ from raw ordered record IDs: ' + key)
            if not np.array_equal(ctab, second_ctab) or names != second_names:
                raise ValueError('fsio color-table semantic read differs: ' + key)
            encoded_names = np.asarray(names, dtype='S')
            # Full numeric vectors + named color table enable paired semantic comparison.
            saved = destination / (key + '.npz')
            np.savez_compressed(saved, vertex_indices=np.asarray(order, dtype='<i8'),
                                original_annotation_ids=np.asarray(original_ids, dtype='<i8'),
                                label_table_indices=np.asarray(table_indices, dtype='<i8'),
                                color_table=np.asarray(ctab, dtype='<i8'), names=encoded_names)
            unique, counts = np.unique(original_ids, return_counts=True)
            table = [{'row': row, 'name_hex': bytes(name).hex(),
                      'name': bytes(name).decode('utf-8', errors='replace'),
                      'rgba_transparency_and_packed_id': [int(value) for value in ctab[row]]}
                     for row, name in enumerate(names)]
            receipts[key] = {'file': file_receipt(path), 'semantic_vectors': file_receipt(saved),
                             'vertex_count': vertex_count, 'vertex_order': '0..nvertices-1 verified',
                             'label_original_ids_sha256_le_i64': array_sha(original_ids, '<i8'),
                             'label_table_indices_sha256_le_i64': array_sha(table_indices, '<i8'),
                             'color_table_sha256_le_i64': array_sha(ctab, '<i8'),
                             'color_table_names_sha256': hashlib.sha256(
                                 json.dumps([bytes(name).hex() for name in names]).encode()).hexdigest(),
                             'color_table': table,
                             'label_counts': [{'id': int(value), 'vertices': int(count)}
                                              for value, count in zip(unique, counts)]}
    return receipts


def numba_cache_receipt():
    value = os.environ.get('NUMBA_CACHE_DIR')
    if not value:
        return {'status': 'undeclared', 'scope': 'default cache location/coldness not inferred'}
    directory = Path(value)
    return {'path': str(directory.resolve()), 'exists': directory.exists(),
            'existing_files': [{'relative': str(path.relative_to(directory)), 'bytes': path.stat().st_size}
                               for path in sorted(directory.rglob('*')) if path.is_file()]
                              if directory.is_dir() else []}


def imported_sources():
    result = {}
    for name, module in list(sys.modules.items()):
        if name == 'fnit' or name.startswith('fnit.'):
            filename = getattr(module, '__file__', None)
            if filename and Path(filename).is_file():
                result[name] = file_receipt(filename)
    return result


def run(args):
    validation_start = time.monotonic()
    checkpoint, assets, output = args.checkpoint.resolve(), args.assets.resolve(), args.output.resolve()
    protected_output(output, checkpoint, assets)
    output.mkdir(parents=True, exist_ok=False)
    report_path = output / 'annotation.stage.json'
    report = {'status': 'validating', 'started_utc': now(), 'started_monotonic': validation_start,
              'checkpoint': str(checkpoint), 'assets': str(assets), 'output': str(output),
              'scope': 'FNIT-produced checkpoint annotation stage only; not continuous chain or original-T1 whole case',
              'caller_lock_requirement': 'Caller must hold shared GPU lock and monitor/clean all owned descendants.',
              'threads': {'total': 4, 'workers': 2, 'per_worker': 2},
              'gpu_uuid_declared': args.gpu_uuid, 'device': 'cuda:0', 'initialized_parent': args.initialized_parent,
              'initialized_parent_scope': 'Optional FP32 scalar+sync reproduces an initialized API; '
                                          'it does not reconstruct historical parent1.946GB/context/model history.',
              'startup_wait_seconds_argument': args.startup_wait_seconds,
              'reference_policy': 'No official images/results are consumed. Assets are declared existing resources; '
                                  'no download or redistribution is performed.',
              'precision_policy': 'FP32 scalar and TF32 true; no autocast; cache disabled',
              'torch_memory_stats_status': 'unavailable_cache_disabled', 'timings': {}}
    write(report_path, report)
    subject = output / 'subject'
    bindings = None
    group = None
    bootstrap = None
    try:
        declared_gpu_uuid(args.gpu_uuid)
        if os.environ.get('CUDA_VISIBLE_DEVICES') != args.gpu_uuid:
            raise ValueError('declared GPU UUID must exactly match the sole CUDA_VISIBLE_DEVICES entry')
        report['thread_environment'] = {key: os.environ.get(key) for key in THREAD_KEYS}
        if any(os.environ.get(key) != '4' for key in THREAD_KEYS):
            raise ValueError('fresh parent native thread environment must be4 for all declared libraries')
        if os.environ.get('PYTORCH_NO_CUDA_MEMORY_CACHING') != '1':
            raise ValueError('fresh child must declare disabled allocator before Torch import')
        report['numba_cache_before'] = numba_cache_receipt()
        manifest = json.loads(args.input_manifest.read_text())
        report['input_manifest'] = file_receipt(args.input_manifest)
        report['historical_source_receipts_not_candidate_gate'] = manifest['replay_files'][15:]
        bindings = manifest_bindings(manifest, checkpoint, assets)
        report['original_inputs_before'] = verify_bindings(bindings)
        report['timings']['validation_seconds'] = time.monotonic() - validation_start
        tick = time.monotonic()
        subject.mkdir()
        for name in ('mri', 'surf', 'label', 'stats'):
            if not (checkpoint / name).is_dir():
                raise FileNotFoundError('checkpoint requires directory ' + name)
            shutil.copytree(checkpoint / name, subject / name, symlinks=False)
        (subject / 'scripts').mkdir()
        report['copied_inputs_before'] = verify_bindings(bindings, subject)
        report['removed_copied_target_annotations'] = []
        # A complete checkpoint may contain old FNIT outputs; clear only this run's six targets.
        for hemi in HEMISPHERES:
            for atlas in ATLASES:
                path = subject / 'label' / f'{hemi}.{atlas}.annot'
                if path.exists():
                    report['removed_copied_target_annotations'].append(file_receipt(path))
                    path.unlink()
        report['timings']['copy_and_copied_hash_seconds'] = time.monotonic() - tick
        write(report_path, report)
        tick = time.monotonic()
        import torch
        import nibabel
        import numpy
        import fnit
        from fnit.recon_all.hemisphere_parallel import run_hemisphere_group
        from fnit.recon_all.profiling import configure_cuda_allocator
        # CPU-only import records exact annotation algorithms also used by fresh exec workers.
        importlib.import_module('fnit.recon_all.native_free')
        importlib.import_module('fnit.recon_all.gcsa_label_python')
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_num_interop_threads(1)
        torch.set_num_threads(4)
        report['allocator'] = configure_cuda_allocator('cuda:0', 'disabled')
        report['runtime'] = {'python': sys.executable, 'python_version': sys.version,
                             'interpreter': file_receipt(Path(sys.executable).resolve()),
                             'torch_version': torch.__version__, 'cuda_build': torch.version.cuda,
                             'nibabel_version': nibabel.__version__, 'numpy_version': numpy.__version__,
                             'fnit_path': fnit.__file__, 'torch_threads': torch.get_num_threads(),
                             'torch_interop_threads': torch.get_num_interop_threads()}
        report['imported_fnit_sources_before'] = imported_sources()
        report['script'] = file_receipt(__file__)
        report['copied_meshes_before'] = mesh_receipts(subject)
        report['original_meshes_before'] = mesh_receipts(checkpoint)
        if {k: (v['coordinate_array_sha256_le_f64'], v['ordered_faces_sha256_le_i64'])
            for k, v in report['copied_meshes_before'].items()} != {
                k: (v['coordinate_array_sha256_le_f64'], v['ordered_faces_sha256_le_i64'])
                for k, v in report['original_meshes_before'].items()}:
            raise ValueError('copied mesh coordinate/topology differs from original checkpoint')
        report['timings']['import_and_mesh_verification_seconds'] = time.monotonic() - tick
        group_options = {'device': 'cuda:0', 'threads': 4, 'workers': 2, 'profile_stages': True,
                         'kwargs': {'assets': str(assets)}}
        if args.startup_wait_seconds is not None:
            if 'startup_wait_seconds' not in inspect.signature(run_hemisphere_group).parameters:
                raise TypeError('candidate startup_wait_seconds API unsupported; no fallback/retry')
            group_options['startup_wait_seconds'] = args.startup_wait_seconds
        parameter = inspect.signature(run_hemisphere_group).parameters.get('startup_wait_seconds')
        report['source_default_startup_wait_seconds'] = parameter.default if parameter is not None else None
        report['group_call_arguments'] = group_options
        if args.initialized_parent:
            tick = time.monotonic()
            bootstrap = torch.empty(1, dtype=torch.float32, device='cuda:0')
            torch.cuda.synchronize(torch.device('cuda:0'))
            properties = torch.cuda.get_device_properties(torch.device('cuda:0'))
            report['parent_actual_device'] = {'name': properties.name,
                                               'uuid': str(getattr(properties, 'uuid', 'unavailable'))}
            report['parent_actual_device']['matches_declared'] = (
                normalized_actual_uuid(report['parent_actual_device']['uuid']) ==
                normalized_actual_uuid(args.gpu_uuid))
            if not report['parent_actual_device']['matches_declared']:
                raise ValueError('initialized parent device differs from declared GPU UUID')
            report['timings']['parent_bootstrap_seconds'] = time.monotonic() - tick
        report['parent_cuda_initialized_before_group'] = torch.cuda.is_initialized()
        report['actual_parent_precision'] = {'matmul_tf32': torch.backends.cuda.matmul.allow_tf32,
                                             'cudnn_tf32': torch.backends.cudnn.allow_tf32,
                                             'autocast_enabled': bool(torch.is_autocast_enabled())}
        report['status'] = 'running_annotation'
        write(report_path, report)
        tick = time.monotonic()
        try:
            group = run_hemisphere_group(subject, 'annotation', **group_options)
            report['group_execution_completed'] = True
        except BaseException as error:
            group = getattr(error, 'report', None)
            report['group_execution_completed'] = False
            raise
        finally:
            report['timings']['group_wall_seconds'] = time.monotonic() - tick
            if group is not None:
                write(output / 'annotation.group.json', group)
                report['group_report'] = file_receipt(output / 'annotation.group.json')
            report['worker_device_validation'] = worker_device_receipt(
                group, args.gpu_uuid, report.get('group_execution_completed', False))
        if report['worker_device_validation']['errors']:
            raise ValueError('worker GPU UUID validation failed: ' +
                             '; '.join(report['worker_device_validation']['errors']))
        tick = time.monotonic()
        report['copied_inputs_after'] = verify_bindings(bindings, subject)
        report['original_inputs_after'] = verify_bindings(bindings)
        report['copied_meshes_after'] = mesh_receipts(subject)
        if report['copied_meshes_after'] != report['copied_meshes_before']:
            raise ValueError('annotation mutated copied input mesh bytes/coordinates/ordered topology')
        report['annotations'] = annotation_receipts(subject, output / 'annotation_semantics',
                                                   report['copied_meshes_before'])
        report['output_complete'] = len(report['annotations']) == 6
        report['input_meshes_stable'] = True
        report['timings']['semantic_output_and_post_hash_seconds'] = time.monotonic() - tick
        report.update(status='stage_complete', whole_case_complete=False,
                      official_equivalence='not_assessed', paired_stage_equivalence='requires other arm comparison')
        return_code = 0
    except BaseException as error:
        report.update(status='failed', error=repr(error), traceback=traceback.format_exc(),
                      whole_case_complete=False)
        print(report['traceback'], file=sys.stderr, flush=True)
        return_code = 1
    finally:
        # CPU-only postfailure receipts; do not recover CUDA or clear its error state.
        if bindings is not None:
            try:
                report['original_inputs_final'] = verify_bindings(bindings)
                if subject.exists():
                    report['copied_inputs_final'] = verify_bindings(bindings, subject)
            except BaseException as error:
                report['final_input_verification_error'] = repr(error)
                report['status'] = 'failed'
                return_code = 1
        report['numba_cache_after'] = numba_cache_receipt()
        report['imported_fnit_sources_after'] = imported_sources()
        changed_sources = [name for name, before in report.get('imported_fnit_sources_before', {}).items()
                           if report['imported_fnit_sources_after'].get(name) != before]
        report['changed_imported_sources'] = changed_sources
        if changed_sources:
            report['status'] = 'failed'
            return_code = 1
        report['retained_stage_files'] = [file_receipt(path) for path in sorted(
            (subject / 'scripts').glob('*')) if path.is_file()] if subject.exists() else []
        report['finished_utc'] = now()
        report['timings']['validation_to_before_summary_write_seconds'] = time.monotonic() - validation_start
        write(report_path, report)
        # Include validation/copy/import/stage/semantics and initial final JSON output in this wall clock.
        report['timings']['validation_through_summary_write_seconds'] = time.monotonic() - validation_start
        report['timing_scope'] = ('From validation entry through first final summary write; final timing receipt '
                                  'rewrite excluded. Outer run_monitored wall clock includes interpreter and all writes.')
        report['exit_code'] = return_code
        write(report_path, report)
    return return_code


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True, help='FNIT自产冻结subject目录')
    parser.add_argument('--input-manifest', type=Path, required=True, help='replay_files前15数据/资产SHA清单')
    parser.add_argument('--assets', type=Path, required=True, help='已核查许可、带固定SHA的existing assets目录')
    parser.add_argument('--output', type=Path, required=True, help='新独立阶段产物目录')
    parser.add_argument('--gpu-uuid', type=declared_gpu_uuid, default=GPU_UUID,
                        help='完整物理GPU UUID；默认原GPU0，必须与唯一CUDA_VISIBLE_DEVICES完全一致')
    parser.add_argument('--initialized-parent', action='store_true', help='保留FP32标量并同步，仅复现父API初始化')
    parser.add_argument('--startup-wait-seconds', type=float, help='仅候选传具名startup_wait_seconds，例如30')
    args = parser.parse_args()
    if args.startup_wait_seconds is not None and (not math.isfinite(args.startup_wait_seconds)
                                                 or args.startup_wait_seconds < 0):
        parser.error('startup-wait-seconds must be finite and >=0')
    return run(args)


if __name__ == '__main__':
    raise SystemExit(main())
