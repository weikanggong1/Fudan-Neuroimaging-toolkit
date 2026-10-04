"""Real saved CPUjoint preprocessing replay; no CNN or original suite calls."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import resource
import statistics
import time

import numba
import numpy as np
import torch
from fnit.synthmorph import _cpu_preprocessing, spatial
import raw_sampler_numba as candidate


def file_sha(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''): digest.update(block)
    return digest.hexdigest()


def score(actual, expected):
    actual = np.asarray(actual, dtype=np.float32)
    expected = np.asarray(expected, dtype=np.float32)
    assert actual.shape == expected.shape, (actual.shape, expected.shape)
    a, b = actual.reshape(-1), expected.reshape(-1)
    different_values = different_bits = 0
    maximum = squared = 0.0
    for start in range(0, len(a), 1024 * 1024):
        av, bv = a[start:start + 1024 * 1024], b[start:start + 1024 * 1024]
        different_values += int(np.count_nonzero(av != bv))
        different_bits += int(np.count_nonzero(av.view(np.uint32) != bv.view(np.uint32)))
        delta = av.astype(np.float64) - bv.astype(np.float64)
        maximum = max(maximum, float(np.abs(delta).max(initial=0)))
        squared += float(np.dot(delta, delta))
    return {'values': int(a.size), 'different_values': different_values,
            'different_bits': different_bits, 'max_abs': maximum,
            'RMSE': (squared / max(a.size, 1)) ** .5,
            'actual_sha256': hashlib.sha256(np.ascontiguousarray(actual).tobytes()).hexdigest(),
            'expected_sha256': hashlib.sha256(np.ascontiguousarray(expected).tobytes()).hexdigest()}


def edge_cases():
    # Contract fixtures only; the performance rows below use complete real data.
    generator = torch.Generator().manual_seed(72)
    volume = torch.randn((2, 3, 3, 4, 10), generator=generator)[..., ::2]
    before = volume.clone()
    translations = [-1e30, -1.5, -.5, 0., .5, 2., 4., 4.5, 1e30]
    records = []
    for translation in translations:
        matrix = torch.eye(4)
        matrix[:3, 3] = torch.tensor([translation, -.5, .5])
        for fill in [0., None, 3.25]:
            expected = _cpu_preprocessing.network_transform(volume, matrix, shape=(5, 6, 7), fill_value=fill)
            actual = candidate.network_transform(volume, matrix, shape=(5, 6, 7), fill_value=fill)
            row = score(actual.numpy(), expected.numpy())
            assert row['different_bits'] == 0, (translation, fill, row)
            records.append({'translation': translation, 'fill': fill, **row})
    assert torch.equal(before, volume)
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference-root', type=Path, required=True)
    parser.add_argument('--raw192', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    assert file_sha(_cpu_preprocessing.__file__) == '9f531efabf8c992e2f1ee0d785ad6655bb09570d75a6e071f7c644e18a1b236a'
    assert file_sha(spatial.__file__) == '0dd342cf0a9c94cd09a0396097adc40afbb838cfbce2f0963a28f417249d93a5'
    torch.set_num_threads(8); torch.set_num_interop_threads(1)
    numba.set_num_threads(8)
    report = {'scope': __doc__, 'hostname': platform.node(), 'affinity': sorted(os.sched_getaffinity(0)),
              'threads': 8, 'interop_threads': 1,
              'versions': {'torch': torch.__version__, 'numpy': np.__version__, 'numba': numba.__version__},
              'source_sha256': {'baseline': file_sha(_cpu_preprocessing.__file__),
                                'spatial': file_sha(spatial.__file__), 'prototype': file_sha(candidate.__file__),
                                'worker': file_sha(__file__)},
              'load_before': list(os.getloadavg()), 'cases': {},
              'CUDA_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
              'official_raw_scope': '192 two full images; 256 raw oracle not saved, full normalized oracle available'}
    with torch.inference_mode():
        # The first real call precedes edge fixtures so its timer includes JIT.
        for extent in (192, 256):
            folder = args.reference_root / ('joint_' + str(extent)) / 'artifacts'
            case = {'images': {}, 'abba': []}
            for image in (0, 1):
                data_file, matrix_file = folder / f'source_data_{image}.npy', folder / f'source_to_network_{image}.npy'
                expected_file = folder / f'network_input_{image}.npy'
                source = np.load(data_file, mmap_mode='r')
                matrix = np.load(matrix_file)
                volume = torch.from_numpy(np.array(source, dtype=np.float32, copy=True))[None, None]
                original_input = hashlib.sha256(volume.numpy().tobytes()).hexdigest()
                shape = (extent,) * 3
                before = numba.get_num_threads()
                started = time.perf_counter(); raw_old = _cpu_preprocessing.network_transform(volume, matrix, shape=shape)
                old_first = time.perf_counter() - started
                started = time.perf_counter(); raw_new = candidate.network_transform(volume, matrix, shape=shape)
                new_first = time.perf_counter() - started
                assert numba.get_num_threads() == before
                metrics = score(raw_new.numpy(), raw_old.numpy())
                shifted = raw_new - raw_new.min()
                normalized = shifted / shifted.max()
                normalized_metrics = score(normalized.permute(0, 2, 3, 4, 1).numpy(), np.load(expected_file, mmap_mode='r'))
                row = {'source_file_sha256': file_sha(data_file), 'matrix_file_sha256': file_sha(matrix_file),
                       'official_normalized_file_sha256': file_sha(expected_file),
                       'source_shape': list(source.shape), 'source_dtype': str(source.dtype),
                       'raw_vs_v29': metrics, 'normalized_vs_official': normalized_metrics,
                       'first_v29_seconds': old_first, 'first_candidate_seconds': new_first,
                       'candidate_first_includes_fresh_JIT': extent == 192 and image == 0}
                assert metrics['different_bits'] == 0 and normalized_metrics['different_bits'] == 0, row
                if extent == 192:
                    raw_file = args.raw192 / f'resampled_{image}.npy'
                    raw_expected = np.load(raw_file, mmap_mode='r')
                    row['official_raw_file_sha256'] = file_sha(raw_file)
                    row['raw_vs_official'] = score(raw_new[0].permute(1, 2, 3, 0).numpy(), raw_expected)
                    assert row['raw_vs_official']['different_bits'] == 0, row
                del raw_old, raw_new, shifted, normalized
                pair = []
                for arm in ('v29', 'numba', 'numba', 'v29'):
                    function = _cpu_preprocessing.network_transform if arm == 'v29' else candidate.network_transform
                    started = time.perf_counter(); output = function(volume, matrix, shape=shape)
                    seconds = time.perf_counter() - started
                    pair.append({'arm': arm, 'seconds': seconds})
                    # Hash outside the timed sampler; no auxiliary array saves in the timer.
                    assert hashlib.sha256(output.numpy().tobytes()).hexdigest() == metrics['expected_sha256']
                    del output
                assert hashlib.sha256(volume.numpy().tobytes()).hexdigest() == original_input
                case['abba'].append({'image': image, 'records': pair})
                row['input_preserved'] = True
                case['images'][str(image)] = row
                print('completed', extent, image, 'bits', metrics['different_bits'], flush=True)
            timings = {arm: [r['seconds'] for pairs in case['abba'] for r in pairs['records'] if r['arm'] == arm]
                       for arm in ('v29', 'numba')}
            case['warm_median_seconds_per_image'] = {arm: statistics.median(values) for arm, values in timings.items()}
            report['cases'][str(extent)] = case
        report['boundary_contract_cases'] = edge_cases()
    assembly = '\n'.join(candidate._sample.inspect_asm().values())
    llvm = '\n'.join(candidate._sample.inspect_llvm().values())
    report['arithmetic_evidence'] = {'fastmath': False,
        'fused_multiply_add_asm_present': bool(re.search(r'\bvfm(?:add|sub)|\bfmadd', assembly)),
        'LLVM_sha256': hashlib.sha256(llvm.encode()).hexdigest(),
        'assembly_sha256': hashlib.sha256(assembly.encode()).hexdigest()}
    report['load_after'] = list(os.getloadavg())
    report['peak_RSS_bytes'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    report['normalized_values_checked'] = sum(row['normalized_vs_official']['values'] for c in report['cases'].values() for row in c['images'].values())
    report['raw_values_checked_vs_v29'] = sum(row['raw_vs_v29']['values'] for c in report['cases'].values() for row in c['images'].values())
    report['production_adopted'] = False
    (args.output / 'report.private.json').write_text(json.dumps(report, indent=2) + '\n')
    print('passed', report['normalized_values_checked'], 'full normalized values', flush=True)


if __name__ == '__main__': main()
