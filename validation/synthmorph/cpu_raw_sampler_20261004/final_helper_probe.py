"""Bind unchanged replay to the actual integrated helper and old v29 oracle."""
import argparse
import ast
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import resource
import sys
import time

import numpy as np
import torch
from fnit.synthmorph import _cpu_preprocessing as integrated, spatial


def file_sha(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--baseline-source', type=Path, required=True)
parser.add_argument('--integrated-source-sha256', required=True)
parser.add_argument('--helper-source-sha256', required=True)
parser.add_argument('--mode', choices=('full', 'cold256'), default='full')
args, replay_arguments = parser.parse_known_args()
assert file_sha(args.baseline_source) == '9f531efabf8c992e2f1ee0d785ad6655bb09570d75a6e071f7c644e18a1b236a'
assert file_sha(spatial.__file__) == '0dd342cf0a9c94cd09a0396097adc40afbb838cfbce2f0963a28f417249d93a5'
spec = importlib.util.spec_from_file_location('fnit.synthmorph._validated_raw_v29', args.baseline_source)
baseline = importlib.util.module_from_spec(spec); spec.loader.exec_module(baseline)
helper_path = Path(integrated.__file__).with_name('_cpu_raw_sampler.py')
assert file_sha(integrated.__file__) == args.integrated_source_sha256
assert file_sha(helper_path) == args.helper_source_sha256
helper_tree = ast.parse(helper_path.read_text())
kernel_node = next(node for node in helper_tree.body
                   if isinstance(node, ast.FunctionDef) and node.name == '_sample_impl')
kernel_ast_sha = hashlib.sha256(ast.dump(kernel_node, include_attributes=False).encode()).hexdigest()
assert kernel_ast_sha == '1b93ed9279559e0493658349a38fdd24717205004efacef0794ac31d7bf01cbe'
bindings = {'integrated_preprocessing_sha256': file_sha(integrated.__file__),
            'final_helper_sha256': file_sha(helper_path), 'driver_sha256': file_sha(__file__),
            'baseline_preprocessing_sha256': file_sha(args.baseline_source),
            'unchanged_spatial_sha256': file_sha(spatial.__file__),
            'ordered_kernel_AST_sha256': kernel_ast_sha}
routes = []


class ActualHelper:
    __file__ = str(helper_path)
    def network_transform(self, *arguments, **keywords):
        result = integrated.network_transform(*arguments, **keywords)
        from fnit.synthmorph import _cpu_raw_sampler as helper
        info = helper.backend_info(); routes.append(info)
        if result[0, 0].numel() >= 32768:
            assert info['backend'] == 'numba' and info['requested_threads'] == 8, info
        return result
    @property
    def _sample(self):
        from fnit.synthmorph import _cpu_raw_sampler as helper
        return helper._KERNEL


if args.mode == 'full':
    # Reuse the existing scoring/ABBA script. The oracle is still original v29.
    import benchmark_raw_sampler as replay
    replay._cpu_preprocessing = baseline
    replay.candidate = ActualHelper()
    sys.argv = [sys.argv[0], *replay_arguments]
    replay.main()
    output_index = replay_arguments.index('--output') + 1
    target = Path(replay_arguments[output_index]) / 'report.private.json'
    report = json.loads(target.read_text())
    report['final_source_binding'] = bindings
    report['source_sha256']['candidate_helper'] = report['source_sha256'].pop('prototype')
    report['actual_backend_records'] = routes
    report['scope'] = __doc__
    # Separate untimed real-input preservation calls; no timer instrumentation.
    from fnit.synthmorph import _cpu_raw_sampler as helper
    import numba
    reference_index = replay_arguments.index('--reference-root') + 1
    reference = Path(replay_arguments[reference_index])
    checks = []
    with torch.inference_mode():
        for extent in (192, 256):
            folder = reference / f'joint_{extent}' / 'artifacts'
            for image in (0, 1):
                volume = torch.from_numpy(np.array(np.load(folder / f'source_data_{image}.npy', mmap_mode='r'), dtype=np.float32, copy=True))[None, None]
                matrix = torch.as_tensor(np.load(folder / f'source_to_network_{image}.npy'), dtype=torch.float32)
                coords = spatial.grid((extent,) * 3, 'cpu')
                locations = coords + spatial._dense_from_grid(matrix, coords)
                volume_before = hashlib.sha256(volume.numpy().tobytes()).hexdigest()
                locations_before = hashlib.sha256(locations.numpy().tobytes()).hexdigest()
                previous = numba.get_num_threads()
                result = helper.try_sample(volume, locations)
                info = helper.backend_info()
                assert info['backend'] == 'numba' and info['requested_threads'] == 8, info
                assert numba.get_num_threads() == previous
                assert hashlib.sha256(volume.numpy().tobytes()).hexdigest() == volume_before
                assert hashlib.sha256(locations.numpy().tobytes()).hexdigest() == locations_before
                assert hashlib.sha256(result.numpy().tobytes()).hexdigest() == report['cases'][str(extent)]['images'][str(image)]['raw_vs_v29']['expected_sha256']
                checks.append({'extent': extent, 'image': image, 'input_preserved': True,
                               'locations_preserved': True, 'thread_mask_restored': True,
                               'input_array_sha256': volume_before, 'locations_array_sha256': locations_before,
                               'actual_backend': info})
    report['real_input_and_locations_preservation'] = checks
    # One default-policy finite contract, using the actual helper on this host.
    # No call changes the caller's or any worker's floating-point environment.
    bits = np.resize(np.array([0, 0x80000000, 1, 0x80000001, 0x007fffff, 0x807fffff], np.uint32), (1, 1, 3, 4, 5))
    tiny = torch.from_numpy(bits.view(np.float32))
    matrix = torch.eye(4); matrix[:3, 3] = .5
    coords = spatial.grid((32,) * 3, 'cpu')
    locations = coords + spatial._dense_from_grid(matrix, coords)
    tiny_before, locations_before = tiny.numpy().view(np.uint32).copy(), locations.numpy().view(np.uint32).copy()
    previous = numba.get_num_threads()
    with torch.inference_mode():
        actual = helper.try_sample(tiny, locations)
        expected = baseline.network_transform(tiny, matrix, shape=(32,) * 3)
    info = helper.backend_info()
    assert info['backend'] == 'numba' and info['requested_threads'] == 8, info
    assert numba.get_num_threads() == previous
    assert np.array_equal(actual.numpy().view(np.uint32), expected.numpy().view(np.uint32))
    assert np.array_equal(tiny.numpy().view(np.uint32), tiny_before)
    assert np.array_equal(locations.numpy().view(np.uint32), locations_before)
    report['default_policy_subnormal_signedzero_contract'] = {
        'different_bits': 0, 'input_preserved': True, 'locations_preserved': True,
        'thread_mask_restored': True, 'floating_policy_changed': False,
        'actual_backend': info}
    target.write_text(json.dumps(report, indent=2) + '\n')
else:
    # One first integrated call. Its timer includes the actual lazy helper import.
    cold_parser = argparse.ArgumentParser()
    cold_parser.add_argument('--reference', type=Path, required=True)
    cold_parser.add_argument('--completed-report', type=Path, required=True)
    cold_parser.add_argument('--output', type=Path, required=True)
    cold = cold_parser.parse_args(replay_arguments)
    torch.set_num_threads(8); torch.set_num_interop_threads(1)
    load_before = list(os.getloadavg())
    volume = torch.from_numpy(np.array(np.load(cold.reference / 'source_data_0.npy', mmap_mode='r'), dtype=np.float32, copy=True))[None, None]
    matrix = np.load(cold.reference / 'source_to_network_0.npy')
    expected = json.loads(cold.completed_report.read_text())['cases']['256']['images']['0']['raw_vs_v29']['expected_sha256']
    with torch.inference_mode():
        started = time.perf_counter(); old = baseline.network_transform(volume, matrix, shape=(256,) * 3)
        old_seconds = time.perf_counter() - started
        assert hashlib.sha256(old.numpy().tobytes()).hexdigest() == expected
        del old
        assert 'fnit.synthmorph._cpu_raw_sampler' not in sys.modules
        started = time.perf_counter(); result = ActualHelper().network_transform(volume, matrix, shape=(256,) * 3)
        new_seconds = time.perf_counter() - started
        assert hashlib.sha256(result.numpy().tobytes()).hexdigest() == expected
    report = {'scope': 'actual integrated 256 first call includes lazy helper import and fresh JIT',
              'hostname': platform.node(), 'affinity': sorted(os.sched_getaffinity(0)),
              'threads': torch.get_num_threads(), 'interop_threads': torch.get_num_interop_threads(),
              'versions': {'torch': torch.__version__, 'numpy': np.__version__,
                           'numba': routes[0]['numba_version']},
              'load_before': load_before, 'load_after': list(os.getloadavg()),
              'CUDA_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
              'source_file_sha256': file_sha(cold.reference / 'source_data_0.npy'),
              'matrix_file_sha256': file_sha(cold.reference / 'source_to_network_0.npy'),
              'final_source_binding': bindings, 'v29_first_call_seconds': old_seconds,
              'final_first_call_seconds_including_lazy_import_JIT': new_seconds,
              'raw_output_sha256': expected, 'actual_backend_records': routes,
              'whole_probe_peak_RSS_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
              'production_adopted': False}
    cold.output.write_text(json.dumps(report, indent=2) + '\n')
