"""冻结同输入annotation调度回归；私有原始报告与typed公开摘要分别保存。"""
from __future__ import annotations
import argparse
import hashlib
import inspect
import json
import os
from pathlib import Path
import shutil
import time


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--subject', type=Path, required=True)
    parser.add_argument('--resources', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--source-manifest', type=Path, required=True)
    parser.add_argument('--case', required=True)
    parser.add_argument('--variant', choices=('baseline', 'candidate'), required=True)
    parser.add_argument('--api-initialized', action='store_true')
    args = parser.parse_args()
    command_tick = time.monotonic()
    # Import/JIT/load costs stay in the outer command; UUID selected before process import.
    import nibabel.freesurfer.io as fs
    import numpy as np
    import torch
    import fnit.recon_all.native_free as pipeline
    from fnit.recon_all.hemisphere_parallel import run_hemisphere_group
    from fnit.recon_all.profiling import configure_cuda_allocator
    from fnit.recon_all.thread_budget import thread_budget
    required_inputs = ([f'surf/{hemi}.{suffix}' for hemi in ('lh', 'rh')
        for suffix in ('orig', 'sphere.reg', 'smoothwm', 'sulc', 'curv')]
        + [f'label/{hemi}.cortex.label' for hemi in ('lh', 'rh')]
        + ['mri/aseg.presurf.mgz'])
    missing = [relative for relative in required_inputs if not (args.subject / relative).is_file()]
    if missing:
        raise ValueError('annotation benchmark input incomplete: ' + ', '.join(missing))
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    source = Path(pipeline.__file__).resolve().parents[3]
    manifest = json.loads(args.source_manifest.read_text())
    for relative, expected in manifest['files'].items():
        if sha(source / relative) != expected:
            raise ValueError('deployed source does not match immutable source manifest')
    resource_manifest = json.loads((args.resources / 'manifest.json').read_text())
    checked = []
    for row in resource_manifest['files']:
        relative = row['path']
        if relative.startswith(('assets/', 'weights/')):
            path = args.resources / relative
            if path.stat().st_size != row['size'] or sha(path) != row['sha256']:
                raise ValueError('resource size/SHA mismatch')
            checked.append({'resource': relative, 'size_bytes': row['size'], 'sha256': row['sha256']})
    inputs = {f'surf/{h}.{suffix}': sha(args.subject / f'surf/{h}.{suffix}')
              for h in ('lh', 'rh') for suffix in ('orig', 'sphere.reg', 'smoothwm', 'sulc', 'curv')}
    inputs['mri/aseg.presurf.mgz'] = sha(args.subject / 'mri/aseg.presurf.mgz')
    for h in ('lh', 'rh'):
        inputs[f'label/{h}.cortex.label'] = sha(args.subject / f'label/{h}.cortex.label')
    private = args.output / 'subject'
    shutil.copytree(args.subject, private)
    # 仅保留计算输入，历史整例报告不能被解释为本次运行证明。
    shutil.rmtree(private / 'scripts', ignore_errors=True)
    (private / 'scripts').mkdir()
    (private / 'fnit-native-free-run.json').unlink(missing_ok=True)
    for h in ('lh', 'rh'):
        for atlas in ('aparc', 'aparc.a2009s', 'aparc.DKTatlas'):
            (private / f'label/{h}.{atlas}.annot').unlink(missing_ok=True)
    prep_seconds = time.monotonic() - command_tick
    allocator = configure_cuda_allocator('cuda:0', 'enabled')
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    if args.api_initialized:
        torch.zeros(1, dtype=torch.float32, device='cuda:0')
        torch.cuda.synchronize('cuda:0')
    initialized_before_group = torch.cuda.is_initialized()
    kwargs = {'assets': str(args.resources / 'assets'), 'binaries': {},
              'surface_placement_backend': 'python', 'topology_backend': 'python-ga',
              'annotation_gibbs_backend': 'numba'}
    if 'profile_stages' in inspect.signature(pipeline._hemisphere_operation).parameters:
        kwargs['profile_stages'] = True
    # Candidate guard is the real production guard; baseline has no corresponding implementation.
    from contextlib import nullcontext
    guard_context = nullcontext(None)
    if args.variant == 'candidate':
        from fnit.recon_all.execution_recipe import original_stage_guard
        guard_context = original_stage_guard(enabled=True)
    tick = time.monotonic()
    with torch.cuda.device('cuda:0'), thread_budget(threads=4), guard_context as guard:
        group = run_hemisphere_group(private, 'annotation', device='cuda:0', threads=4,
            workers=2, profile_stages=True, kwargs=kwargs, cuda_allocator_cache='inherit')
        torch.cuda.synchronize('cuda:0')
    wall = time.monotonic() - tick
    (args.output / 'group.private.json').write_text(json.dumps(group, indent=2))
    monitor = group['device_process_tree']
    typed = {'scope': 'frozen same-input annotation only; not raw-T1 whole run',
             'case': args.case, 'variant': args.variant, 'source_commit': manifest['commit'],
             'benchmark_script_sha256': sha(__file__),
             'source_sha256': manifest['files'], 'inputs_sha256': inputs,
             'resources_verified': checked, 'resource_count': len(checked),
             'execution_status': group['status'], 'parent_cuda_initialized_before_group': initialized_before_group,
             'allocator_selection': allocator, 'group_complete_wall_seconds': wall,
             'preparation_seconds': prep_seconds,
             'command_body_seconds': time.monotonic() - command_tick,
             'command_scope': 'entry includes import, source/resource validation, fixture prep and complete group; interpreter startup is not measured by this harness',
             'thread_budget': 4, 'hemisphere_threads': 2,
             'program_guard': guard, 'workers': {}, 'annotations': {},
             'timing': {key: group[key] for key in ('group_wall_seconds', 'private_copy_seconds',
                'publish_seconds', 'cleanup_seconds', 'worker_sum_seconds', 'worker_span_seconds', 'overlap_seconds')},
             'gpu_observation': {key: monitor.get(key) for key in ('status', 'target_gpu_uuid',
                'sampling_interval_seconds', 'max_observed_interval_seconds', 'peak_tree_total_bytes',
                'query_backend', 'peak_target_device_used_bytes',
                'peak_target_compute_process_sum_bytes', 'scope')},
             'overall_equivalence': 'not_assessed', '138_whole_validation': 'not_run'}
    typed['gpu_observation']['failed_sample_count'] = len(monitor.get('failed_samples', []))
    for h, worker in group['workers'].items():
        typed['workers'][h] = {key: worker.get(key) for key in ('status', 'actual_cuda_device',
            'cuda_allocator', 'precision_actual_at_entry', 'precision_actual_after_operation',
            'stage_program_boundary', 'stage', 'thread_budget')}
        typed['workers'][h]['substeps'] = group['values'][h]['stages']
    for h in ('lh', 'rh'):
        for atlas in ('aparc', 'aparc.a2009s', 'aparc.DKTatlas'):
            name = f'{h}.{atlas}.annot'
            labels, table, names = fs.read_annot(private / 'label' / name, orig_ids=True)
            typed['annotations'][name] = {'sha256': sha(private / 'label' / name),
                'vertex_count': int(len(labels)), 'label_count': int(len(np.unique(labels))),
                'dtype': str(labels.dtype), 'table_sha256': hashlib.sha256(table.tobytes()).hexdigest()}
    typed['input_unchanged'] = all(sha(args.subject / relative) == expected for relative, expected in inputs.items())
    (args.output / 'summary.public.json').write_text(json.dumps(typed, indent=2))


if __name__ == '__main__':
    main()
