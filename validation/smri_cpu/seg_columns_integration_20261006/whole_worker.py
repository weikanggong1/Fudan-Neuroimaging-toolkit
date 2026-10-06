"""Prepared single complete 33-class arm; execution requires separate approval."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import sys
import time


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    process_start = time.perf_counter()
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('root', 'source', 'plan', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--arm', choices=('baseline', 'candidate'), required=True)
    parser.add_argument('--device', choices=('cpu', 'cuda:0'), required=True)
    parser.add_argument('--approved-whole', action='store_true')
    args = parser.parse_args()
    assert args.approved_whole, 'prepared only; separate coordinator approval required'
    os.umask(0o077)
    plan = json.loads(args.plan.read_text())
    assert sha(__file__) == plan['validation_sources']['whole_worker.py']
    expected = plan['production_sources'][args.arm]
    def sources():
        return {name: sha(args.source / 'fnit/synthseg_parc' / name) for name in expected}
    assert sources() == expected
    def support_sources():
        return {name: sha(args.source / name) for name in plan['common_support_sha256']}
    assert support_sources() == plan['common_support_sha256']
    args.output.mkdir(mode=0o700)  # Never overwrite a prior arm.
    cpu = args.device == 'cpu'
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
    for name, entry in plan['resources'].items():
        path = args.root / entry['fnit_relative']
        actual = {'bytes': path.stat().st_size, 'sha256': sha(path)}
        assert actual == {key: entry[key] for key in actual}
        resources[name] = actual
    raw_t1 = args.root / plan['resources']['raw_t1']['fnit_relative']
    weights = (args.root / plan['resources']['segmentation_weight']['fnit_relative']).parent
    report = {'schema': 'fnit_columns_complete33_worker/v1', 'status': 'running',
              'arm': args.arm, 'device': args.device, 'source_before': sources(),
              'common_support_before': support_sources(),
              'worker_sha256': sha(__file__), 'PLAN_sha256': sha(args.plan), 'resources_before': resources,
              'affinity': sorted(os.sched_getaffinity(0)), 'flags_before': initial,
              'threads': torch.get_num_threads(), 'interop_threads': torch.get_num_interop_threads(),
              'hostname': os.uname().nodename, 'load_before': list(os.getloadavg()),
              'forward_rows': [], 'columns_rows': [], 'compile_calls': 0, 'copy_calls': 0, 'SGEMM_calls': 0,
              'compiler_probe_calls': 0, 'build_input_calls': 0,
              'cache_directory_exists_before': None, 'observations_are_Module_hooks': False}
    originals = []
    def replace(owner, name, function):
        originals.append((owner, name, getattr(owner, name)))
        setattr(owner, name, function)
    original_forward = segment.SegmentUNet.forward
    def observed_forward(model, image):
        row = {'input_shape': list(image.shape), 'input_dtype': str(image.dtype),
               'input_device': str(image.device), 'flags': flags()}
        report['forward_rows'].append(row)
        output = original_forward(model, image)
        row.update(output_dtype=str(output.dtype), output_device=str(output.device))
        return output
    replace(segment.SegmentUNet, 'forward', observed_forward)
    if cpu and args.arm == 'candidate':
        # CPU-only observers. The GPU arm never imports either optional module.
        from fnit.synthseg_parc import cpu_columns, _cpu_columns_build as build
        cache = Path(os.environ['FNIT_SYNTHSEG_CPU_CACHE'])
        report['cache_directory_exists_before'] = cache.exists()
        original_columns = cpu_columns._Columns.forward
        def observed_columns(engine, layer, image):
            row = {'shape': list(image.shape), 'dtype': str(image.dtype), 'device': str(image.device)}
            report['columns_rows'].append(row)
            original_copy, original_gemm = engine.copy, engine.gemm
            def copy(*values):
                report['copy_calls'] += 1
                return original_copy(*values)
            def gemm(*values):
                report['SGEMM_calls'] += 1
                return original_gemm(*values)
            engine.copy, engine.gemm = copy, gemm
            try:
                result = original_columns(engine, layer, image)
                row['returned_candidate'] = result is not None
                return result
            finally:
                engine.copy, engine.gemm = original_copy, original_gemm
        replace(cpu_columns._Columns, 'forward', observed_columns)
        for name, key in (('_compile', 'compile_calls'), ('_compiler', 'compiler_probe_calls'),
                          ('_build_inputs', 'build_input_calls')):
            original = getattr(build, name)
            def observed(*values, original=original, key=key, **keywords):
                report[key] += 1
                return original(*values, **keywords)
            replace(build, name, observed)
    target = torch.device(args.device)
    if not cpu:
        assert torch.cuda.is_available()
        properties = torch.cuda.get_device_properties(target)
        assert str(properties.uuid).lower().removeprefix('gpu-') == plan['GPU_uuid'].lower().removeprefix('gpu-')
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
        if cpu and args.arm == 'candidate':
            assert len(report['columns_rows']) == 2
            assert all(row['returned_candidate'] for row in report['columns_rows'])
            assert report['copy_calls'] == report['SGEMM_calls'] == 28
            assert report['build_input_calls'] == report['compiler_probe_calls'] == 2
        else:
            assert not report['columns_rows'] and report['compile_calls'] == 0
        if not cpu:
            optional = [name for name in sys.modules if name in
                        ('fnit.synthseg_parc.cpu_columns', 'fnit.synthseg_parc._cpu_columns_build')]
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
        report['flags_after'] = flags()
        report['source_after'] = sources()
        report['common_support_after'] = support_sources()
        report['common_support_unchanged'] = (report['common_support_before'] == report['common_support_after'] == plan['common_support_sha256'])
        report['resources_after'] = {name: {'bytes': (args.root / entry['fnit_relative']).stat().st_size,
                                          'sha256': sha(args.root / entry['fnit_relative'])}
                                     for name, entry in plan['resources'].items()}
        report['source_unchanged'] = report['source_before'] == report['source_after'] == expected
        report['resources_unchanged'] = report['resources_before'] == report['resources_after']
        report['flags_restored'] = initial == report['flags_after']
        report['RSS_maximum_bytes'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
        report['load_after'] = list(os.getloadavg())
        report['worker_seconds'] = time.perf_counter() - process_start
        report['valid_complete_arm'] = (error is None and report['source_unchanged']
            and report['resources_unchanged'] and report['common_support_unchanged'] and report['flags_restored']
            and report['RSS_maximum_bytes'] <= 32_000_000_000)
        (args.output / 'WHOLE.json').write_text(json.dumps(report, indent=2) + '\n')
    if error is not None:
        raise error
    assert report['valid_complete_arm']


if __name__ == '__main__':
    main()
