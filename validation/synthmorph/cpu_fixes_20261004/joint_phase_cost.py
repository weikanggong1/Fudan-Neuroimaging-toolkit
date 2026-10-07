"""Time finite v29 CPU stages on certified saved original tensors; no full CNN."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
from fnit.synthmorph import models, spatial
from fnit.synthmorph._cpu_preprocessing import network_transform
from fnit.synthmorph import _cpu_eigen


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def timed(function, *args, **kwargs):
    start = time.perf_counter()
    result = function(*args, **kwargs)
    return result, time.perf_counter() - start


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', required=True)
    parser.add_argument('--extent', type=int, required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    torch.set_num_threads(8)
    torch.set_num_interop_threads(1)
    reference = Path(args.reference)
    rows = []
    with torch.inference_mode():
        for index in (0, 1):
            data_path = reference / ('source_data_' + str(index) + '.npy')
            matrix_path = reference / ('source_to_network_' + str(index) + '.npy')
            data = torch.from_numpy(np.load(data_path).astype(np.float32))[None, None]
            matrix = np.load(matrix_path)
            expected = np.load(reference / ('network_input_' + str(index) + '.npy'))
            sampler_rows = []
            # Alternating orders limit drift within this finite control.
            for name in ('original', 'ordered', 'ordered', 'original'):
                function = spatial.transform if name == 'original' else network_transform
                raw, seconds = timed(function, data, matrix, shape=(args.extent,) * 3)
                normalized = raw - raw.min()
                normalized = normalized / normalized.max()
                normalized = normalized.permute(0, 2, 3, 4, 1).numpy()
                sampler_rows.append({'sampler': name, 'seconds': seconds,
                                     'normalized_reference_exact': bool(np.array_equal(normalized, expected)),
                                     'normalized_values_different': int(np.count_nonzero(normalized != expected))})
                del raw, normalized
            rows.append({'input': index, 'source_data_sha256': digest(data_path),
                         'matrix_sha256': digest(matrix_path), 'samplers': sampler_rows})
        features = [torch.from_numpy(np.load(reference / ('Conv3D_conv3d_8_output_' + str(index) + '.npy'))).permute(0, 4, 1, 2, 3)
                    for index in (0, 1)]
        centers, masses, times = [], [], {}
        for index, feature in enumerate(features):
            (center, mass), seconds = timed(models._cpu_joint_barycenter, feature, (args.extent,) * 3)
            centers.append(center); masses.append(mass)
            times['mass_and_barycenter_' + str(index)] = seconds
        start = time.perf_counter()
        weights = (masses[0] / models._cpu_inner_sum(masses[0]).unsqueeze(-1)) * (masses[1] / models._cpu_inner_sum(masses[1]).unsqueeze(-1))
        times['weights'] = time.perf_counter() - start
        fit1, times['fit_0'] = timed(models._cpu_joint_fit_affine, centers[0], centers[1], weights)
        fit2, times['fit_1'] = timed(models._cpu_joint_fit_affine, centers[1], centers[0], weights)
        inverse, times['inverse'] = timed(models._cpu_joint_inverse, fit2)
        average = .5 * (fit1 + inverse)
        _, times['sqrt_first_process_cached_binary'] = timed(models._cpu_joint_matrix_sqrt, average)
        _, times['sqrt_second_call'] = timed(models._cpu_joint_matrix_sqrt, average)
        _, times['center_composition'] = timed(models._cpu_joint_center_affine, average, (args.extent,) * 3)
    report = {'scope': __doc__, 'extent': args.extent, 'preprocessing': rows, 'small_affine_stages_seconds': times,
              'timing_scope': 'finite saved-real-data operator timing; input reading excluded; fresh process, existing compiled Eigen cache; not full-registration speed',
              'source_sha256': {Path(module.__file__).name: digest(module.__file__) for module in (models, spatial, _cpu_eigen)},
              'worker_sha256': digest(__file__)}
    Path(args.output).write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
