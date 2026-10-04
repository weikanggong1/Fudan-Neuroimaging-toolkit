"""Source-bound preprocessing comparison against actual original eager inputs."""
import argparse
import hashlib
import json
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from fnit.synthmorph import pipeline, spatial
try:
    from first_layer_torch_probe import metrics
except ModuleNotFoundError:
    from first_layer_torch_probe_v1 import metrics


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('moving', 'fixed', 'reference', 'output'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--extent', type=int, required=True)
    parser.add_argument('--raw-reference')
    parser.add_argument('--production-raw', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(8)
    torch.set_num_interop_threads(1)
    reference = Path(args.reference)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    rows = {}
    with torch.inference_mode():
        for index, filename in enumerate((args.moving, args.fixed)):
            image = nib.load(filename)
            native = pipeline._image_data(image, 'cpu')
            original_data = np.load(reference / ('source_data_' + str(index) + '.npy'), mmap_mode='r')
            original_matrix = np.load(reference / ('source_to_network_' + str(index) + '.npy'))
            original_affine = np.load(reference / ('source_vox2world_' + str(index) + '.npy'))
            expected = np.load(reference / ('network_input_' + str(index) + '.npy'))
            original_half = np.load(reference / ('VxmAffineFeatureDetector_vxm_affine_feature_detector_input_' + str(index) + '.npy'))
            matrix, _ = pipeline.network_space(image, (args.extent,) * 3)
            sampler = spatial.transform
            if args.production_raw:
                from fnit.synthmorph._cpu_preprocessing import network_transform
                sampler = network_transform
            raw = sampler(torch.from_numpy(native)[None, None], matrix, shape=(args.extent,) * 3)
            shifted = raw - raw.min()
            normalized = shifted / shifted.max()
            normalized_array = normalized.permute(0, 2, 3, 4, 1).numpy()
            np.save(output / ('normalized_' + str(index) + '.npy'), normalized_array)
            row = {'decoded_float32_values_exact': bool(np.array_equal(native, original_data.astype(np.float32))),
                   'native_shape': list(image.shape), 'decode_dtype_original': str(original_data.dtype),
                   'source_affine_max_abs': float(np.abs(image.affine - original_affine).max()),
                   'network_matrix_max_abs': float(np.abs(matrix - original_matrix).max()),
                   'normalized': metrics(normalized_array, expected),
                   'original_integer_downsample_exact': bool(np.array_equal(expected[:, ::2, ::2, ::2], original_half)),
                   'input_file_sha256': digest(filename),
                   'normalized_sha256': digest(output / ('normalized_' + str(index) + '.npy'))}
            if args.raw_reference:
                original_raw = np.load(Path(args.raw_reference) / ('resampled_' + str(index) + '.npy'))
                row['resampling'] = metrics(raw[0].permute(1, 2, 3, 0).numpy(), original_raw)
                same_raw = torch.from_numpy(original_raw)[None].permute(0, 4, 1, 2, 3)
                same_shift = same_raw - same_raw.min()
                same_normalized = same_shift / same_shift.max()
                row['normalization_on_identical_raw'] = metrics(same_normalized.permute(0, 2, 3, 4, 1).numpy(), expected)
            rows[str(index)] = row
    report = {'scope': __doc__, 'extent': args.extent, 'rows': rows,
              'worker_sha256': digest(__file__), 'pipeline_sha256': digest(pipeline.__file__),
              'spatial_sha256': digest(spatial.__file__),
              'original_live_report_sha256': digest(reference / 'report.private.json')}
    (output / 'report.private.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(rows, indent=2))


if __name__ == '__main__':
    main()
