"""Prepared single complete 33-class arm; execution requires separate approval."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import sys
import time

from controller_safety import child_subprocess_lock_wrapper


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    process_start = time.perf_counter()
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('root', 'source', 'plan', 'bindings', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--arm', choices=('baseline', 'candidate'), required=True)
    parser.add_argument('--device', choices=('cpu', 'cuda:0'), required=True)
    parser.add_argument('--approved-whole', action='store_true')
    args = parser.parse_args()
    assert args.approved_whole, 'prepared only; separate coordinator approval required'
    os.umask(0o077)
    plan = json.loads(args.plan.read_text())
    bindings = json.loads(args.bindings.read_text())
    assert bindings['public_PLAN_sha256'] == sha(args.plan)
    assert sha(__file__) == plan['validation_sources']['whole_worker.py']
    assert sha(Path(__file__).with_name('controller_safety.py')) == plan['validation_sources']['controller_safety.py']
    expected = plan['production_sources'][args.arm]
    def sources():
        return {name: sha(args.source / 'fnit/synthseg_parc' / name) for name in expected}
    assert sources() == expected
    def support_sources():
        return {name: sha(args.source / name) for name in plan['common_support_sha256']}
    assert support_sources() == plan['common_support_sha256']
    args.output.mkdir(mode=0o700)  # Never overwrite a prior arm.
    cpu = args.device == 'cpu'
    lock_fd = int(os.environ['FNIT_VALIDATION_LOCK_FD'])
    group = plan['whole_CPU'] if cpu else plan['whole_GPU']
    assert os.fstat(lock_fd).st_ino == (args.root / group['common_lock']).stat().st_ino
    if cpu:
        resource.setrlimit(resource.RLIMIT_AS, (32_000_000_000, 32_000_000_000))
    sys.path.insert(0, str(args.source))
    import fnit
    import torch
    from fnit import SynthSeg
    from fnit.synthseg_parc import segment
    assert Path(fnit.__file__).resolve() == (args.source / 'fnit/__init__.py').resolve()
    assert torch.__version__ == '2.5.1'
    torch.set_num_threads(8)
    torch.set_num_interop_threads(8)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = True
    assert not torch.is_autocast_enabled('cpu') and not torch.is_autocast_enabled('cuda')
    def flags():
        return {'oneDNN': torch.backends.mkldnn.enabled, 'grad': torch.is_grad_enabled(),
                'matmul_TF32': torch.backends.cuda.matmul.allow_tf32,
                'cuDNN_TF32': torch.backends.cudnn.allow_tf32,
                'CPU_autocast': torch.is_autocast_enabled('cpu'),
                'CUDA_autocast': torch.is_autocast_enabled('cuda'),
                'dtype': str(torch.get_default_dtype()), 'threads': torch.get_num_threads()}
    initial = flags()
    resources = {}
    for name, entry in bindings['resources'].items():
        path = args.root / entry['fnit_relative']
        actual = {'bytes': path.stat().st_size, 'sha256': sha(path)}
        assert actual == {key: entry[key] for key in actual}
        resources[name] = actual
    raw_t1 = args.root / bindings['resources']['raw_t1']['fnit_relative']
    weights = (args.root / bindings['resources']['segmentation_weight']['fnit_relative']).parent
    report = {'schema': 'fnit_C24_complete33_worker/v1', 'status': 'running',
              'arm': args.arm, 'device': args.device, 'source_before': sources(),
              'common_support_before': support_sources(),
              'worker_sha256': sha(__file__), 'validation_helper_before': sha(Path(__file__).with_name('controller_safety.py')), 'private_bindings_sha256': sha(args.bindings), 'PLAN_sha256': sha(args.plan), 'resources_before': resources,
              'affinity': sorted(os.sched_getaffinity(0)), 'flags_before': initial,
              'threads': torch.get_num_threads(), 'interop_threads': torch.get_num_interop_threads(),
              'hostname': os.uname().nodename, 'load_before': list(os.getloadavg()),
              'forward_rows': [], 'columns_rows': [], 'compile_calls': 0, 'copy_calls': 0, 'SGEMM_calls': 0,
              'compiler_probe_calls': 0, 'build_input_calls': 0,
              'C72': {'layer_calls': 0, 'copy_calls': 0, 'SGEMM_calls': 0, 'compile_calls': 0, 'compile_seconds': []},
              'C24': {'layer_calls': 0, 'copy_calls': 0, 'SGEMM_calls': 0, 'compile_calls': 0, 'compile_seconds': []},
              'cache_directory_exists_before': None, 'observations_are_Module_hooks': False}
    originals = []
    def replace(owner, name, function):
        originals.append((owner, name, getattr(owner, name)))
        setattr(owner, name, function)
    original_forward = segment.SegmentUNet.forward
    def observed_forward(model, image):
        row = {'input_shape': list(image.shape), 'input_dtype': str(image.dtype),
               'input_device': str(image.device), 'flags': flags(),
               'parameter_devices': sorted({str(value.device) for value in model.parameters()}),
               'parameter_dtypes': sorted({str(value.dtype) for value in model.parameters()})}
        report['forward_rows'].append(row)
        output = original_forward(model, image)
        row.update(output_dtype=str(output.dtype), output_device=str(output.device), output_shape=list(output.shape))
        return output
    replace(segment.SegmentUNet, 'forward', observed_forward)
    def old_cache_identities():
        return {entry['fnit_relative']: {'bytes': (args.root / entry['fnit_relative']).stat().st_size,
                                        'sha256': sha(args.root / entry['fnit_relative'])}
                for entry in bindings['C72_cache_files']}
    old_cache_expected = {entry['fnit_relative']: {'bytes': entry['bytes'], 'sha256': entry['sha256']}
                          for entry in bindings['C72_cache_files']}
    assert old_cache_identities() == old_cache_expected
    report['C72_cache_before'] = old_cache_expected
    if cpu:
        # Both versions already use C72. Count C72 and C24 separately without
        # installing Module hooks; CUDA imports neither optional module.
        from fnit.synthseg_parc import cpu_columns, _cpu_columns_build as build
        replace(build.subprocess, 'Popen', child_subprocess_lock_wrapper(build.subprocess.Popen, lock_fd))
        cache = Path(os.environ['FNIT_SYNTHSEG_C24_CPU_CACHE'])
        report['cache_directory_exists_before'] = cache.exists()
        def observe_layer(cls, kind):
            original = cls.forward
            def observed(engine, layer, image):
                row = {'kind': kind, 'shape': list(image.shape), 'dtype': str(image.dtype), 'device': str(image.device)}
                report['columns_rows'].append(row)
                report[kind]['layer_calls'] += 1
                original_copy, original_gemm = engine.copy, engine.gemm
                def copy(*values):
                    report[kind]['copy_calls'] += 1
                    return original_copy(*values)
                def gemm(*values):
                    report[kind]['SGEMM_calls'] += 1
                    return original_gemm(*values)
                engine.copy, engine.gemm = copy, gemm
                try:
                    result = original(engine, layer, image)
                    row['returned_candidate'] = result is not None
                    return result
                finally:
                    engine.copy, engine.gemm = original_copy, original_gemm
            replace(cls, 'forward', observed)
        observe_layer(cpu_columns._Columns, 'C72')
        if args.arm == 'candidate':
            from fnit.synthseg_parc import cpu_columns_c24
            observe_layer(cpu_columns_c24._ColumnsC24, 'C24')
        original_compile = build._compile
        def observed_compile(command):
            source = [Path(value).name for value in command if value.endswith('.cpp')]
            assert len(source) == 1 and source[0] in ('_columns_reuse.cpp', '_columns_c24.cpp')
            kind = 'C24' if source[0] == '_columns_c24.cpp' else 'C72'
            report['compile_calls'] += 1
            report[kind]['compile_calls'] += 1
            compile_started = time.perf_counter()
            try:
                return original_compile(command)
            finally:
                report[kind]['compile_seconds'].append(time.perf_counter() - compile_started)
        replace(build, '_compile', observed_compile)
        for name, key in (('_compiler', 'compiler_probe_calls'), ('_build_inputs', 'build_input_calls')):
            original = getattr(build, name)
            def observed(*values, original=original, key=key, **keywords):
                report[key] += 1
                return original(*values, **keywords)
            replace(build, name, observed)
    target = torch.device(args.device)
    if not cpu:
        assert torch.cuda.is_available()
        properties = torch.cuda.get_device_properties(target)
        assert str(properties.uuid).lower().removeprefix('gpu-') == bindings['GPU_uuid'].lower().removeprefix('gpu-')
        torch.cuda.set_per_process_memory_fraction(20_000_000_000 / properties.total_memory, target)
        torch.cuda.reset_peak_memory_stats(target)
        report['gpu'] = {'uuid': str(properties.uuid), 'name': properties.name,
                         'allocator_budget_bytes': 20_000_000_000}
    def sync():
        if not cpu:
            torch.cuda.synchronize(target)
    error = None
    try:
        report['preflight_seconds'] = time.perf_counter() - process_start
        sync()
        start = time.perf_counter()
        model = SynthSeg(weights=weights, device=args.device, threads=8, cudnn_tf32=True)
        sync()
        report['construct_seconds'] = time.perf_counter() - start
        assert flags() == initial
        start = time.perf_counter()
        result = model(raw_t1, keep_geometry=False)
        sync()
        report['API_seconds'] = time.perf_counter() - start
        report['precision'] = result.precision
        report['model_devices'] = sorted({str(value.device) for value in model.segmenter.model.parameters()})
        report['model_dtypes'] = sorted({str(value.dtype) for value in model.segmenter.model.parameters()})
        # Save complete scientific output before diagnostic-count assertions.
        start = time.perf_counter()
        result.segmentation.save(args.output / 'segmentation.nii.gz')
        result.write_volumes_csv(raw_t1, args.output / 'volumes.csv')
        report['save_seconds'] = time.perf_counter() - start
        report['saved_outputs'] = {name: {'bytes': (args.output / name).stat().st_size,
                                  'sha256': sha(args.output / name)}
                                  for name in ('segmentation.nii.gz', 'volumes.csv')}
        assert len(report['forward_rows']) == 2
        assert report['model_dtypes'] == ['torch.float32']
        expected_forward = plan['forward_expectations'][args.device]
        for row in report['forward_rows']:
            assert row['input_shape'] == expected_forward['input_shape']
            assert row['output_shape'] == expected_forward['output_shape']
            assert row['input_device'] == row['output_device'] == expected_forward['device']
            assert row['parameter_devices'] == [expected_forward['device']]
            assert row['parameter_dtypes'] == [expected_forward['dtype']]
            assert row['input_dtype'] == row['output_dtype'] == expected_forward['dtype']
            assert row['flags'] == expected_forward['flags']
        assert report['model_devices'] == [expected_forward['device']]
        assert report['precision']['requested_cudnn_tf32'] is True
        assert len(report['precision']['forwards']) == 2
        for row in report['precision']['forwards']:
            assert row['device'] == expected_forward['device']
            assert row['model_dtypes'] == [expected_forward['dtype']]
            assert row['input_dtype'] == row['output_dtype'] == expected_forward['dtype']
            assert row['matmul_tf32'] == expected_forward['flags']['matmul_TF32']
            assert row['cudnn_tf32'] == expected_forward['flags']['cuDNN_TF32']
            assert all(row['autocast'][name]['enabled'] is False for name in ('cpu', 'cuda'))
        report['actual_forward_device_dtype_precision_gate'] = True
        if cpu:
            assert report['C72']['layer_calls'] == 2
            assert report['C72']['copy_calls'] == report['C72']['SGEMM_calls'] == 28
            assert report['C72']['compile_calls'] == 0
            assert report['C24']['layer_calls'] == (2 if args.arm == 'candidate' else 0)
            assert report['C24']['copy_calls'] == report['C24']['SGEMM_calls'] == (12 if args.arm == 'candidate' else 0)
            assert all(row['returned_candidate'] for row in report['columns_rows'])
            assert report['build_input_calls'] == report['compiler_probe_calls'] == (4 if args.arm == 'candidate' else 2)
            report['copy_calls'] = report['C72']['copy_calls'] + report['C24']['copy_calls']
            report['SGEMM_calls'] = report['C72']['SGEMM_calls'] + report['C24']['SGEMM_calls']
        else:
            assert not report['columns_rows'] and report['compile_calls'] == 0
        if not cpu:
            optional = [name for name in sys.modules if name in
                        ('fnit.synthseg_parc.cpu_columns', 'fnit.synthseg_parc._cpu_columns_build',
                         'fnit.synthseg_parc.cpu_columns_c24', 'fnit.synthseg_parc._cpu_columns_c24_build')]
            report['optional_CPU_modules_imported'] = optional
            assert not optional
            report['gpu'].update(allocated_peak_bytes=torch.cuda.max_memory_allocated(target),
                                 reserved_peak_bytes=torch.cuda.max_memory_reserved(target))
            assert max(report['gpu']['allocated_peak_bytes'], report['gpu']['reserved_peak_bytes']) <= 20_000_000_000
        else:
            assert not torch.cuda.is_initialized()
        report['status'] = 'complete'
    except BaseException as caught:
        error = caught
        report.update(status='failed', error_type=type(caught).__name__, error=str(caught))
    finally:
        for owner, name, original in reversed(originals):
            setattr(owner, name, original)
        report['validation_helper_after'] = sha(Path(__file__).with_name('controller_safety.py'))
        report['validation_helper_unchanged'] = report['validation_helper_before'] == report['validation_helper_after'] == plan['validation_sources']['controller_safety.py']
        report['C72_cache_after'] = old_cache_identities()
        report['C72_cache_unchanged'] = report['C72_cache_after'] == old_cache_expected
        report['flags_after'] = flags()
        report['source_after'] = sources()
        report['common_support_after'] = support_sources()
        report['common_support_unchanged'] = (report['common_support_before'] == report['common_support_after'] == plan['common_support_sha256'])
        report['resources_after'] = {name: {'bytes': (args.root / entry['fnit_relative']).stat().st_size,
                                          'sha256': sha(args.root / entry['fnit_relative'])}
                                     for name, entry in bindings['resources'].items()}
        report['source_unchanged'] = report['source_before'] == report['source_after'] == expected
        report['resources_unchanged'] = report['resources_before'] == report['resources_after']
        report['flags_restored'] = initial == report['flags_after']
        report['RSS_maximum_bytes'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
        report['load_after'] = list(os.getloadavg())
        report['worker_seconds'] = time.perf_counter() - process_start
        report['valid_complete_arm'] = (error is None and report['validation_helper_unchanged'] and report.get('actual_forward_device_dtype_precision_gate', False) and report['C72_cache_unchanged'] and report['source_unchanged']
            and report['resources_unchanged'] and report['common_support_unchanged'] and report['flags_restored']
            and report['RSS_maximum_bytes'] <= 32_000_000_000)
        (args.output / 'WHOLE.json').write_text(json.dumps(report, indent=2) + '\n')
    if error is not None:
        raise error
    assert report['valid_complete_arm']


if __name__ == '__main__':
    main()
