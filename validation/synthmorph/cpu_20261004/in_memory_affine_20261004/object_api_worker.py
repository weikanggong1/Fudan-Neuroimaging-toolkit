"""One real, materialized SpatialImage affine API call; no reference inference."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time

import nibabel as nib
import numpy as np
import torch
import fnit
from fnit.synthmorph import SynthMorph


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def array_digest(value):
    data = np.ascontiguousarray(value)
    return hashlib.sha256(data.tobytes()).hexdigest()


def image_summary(image):
    return {
        'class': type(image).__name__, 'shape': list(image.shape),
        'data_dtype': str(np.asanyarray(image.dataobj).dtype),
        'array_sha256': array_digest(np.asanyarray(image.dataobj)),
        'header_sha256': hashlib.sha256(image.header.binaryblock).hexdigest(),
        'affine': image.affine.tolist(),
        'zooms': list(map(float, image.header.get_zooms())),
        'qform': image.get_qform().tolist(),
        'sform': image.get_sform().tolist(),
        'qform_code': int(image.header['qform_code']),
        'sform_code': int(image.header['sform_code']),
        'units': list(image.header.get_xyzt_units()),
        'extension_codes': [int(x.get_code()) for x in image.header.extensions],
    }


def materialize(source):
    # Preserve real acquired voxel values and geometry; this is not an ArrayProxy.
    data = np.array(np.asanyarray(source.dataobj), copy=True)
    image = type(source)(data, source.affine.copy(), header=source.header.copy())
    if not isinstance(image.dataobj, np.ndarray) or image.get_filename() is not None:
        raise RuntimeError('not a fully materialized, filename-free image')
    before, after = image_summary(source), image_summary(image)
    if before != after:
        raise RuntimeError('materialization changed acquired data or stored geometry')
    return image, after


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('moving', 'fixed', 'weights', 'source-manifest', 'output'):
        parser.add_argument('--' + name, required=True)
    args = parser.parse_args()
    torch.set_num_threads(8)
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise RuntimeError('this verification requires hidden CUDA')
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    source = Path(fnit.__file__).parent
    manifest = json.loads(Path(args.source_manifest).read_text())
    source_sha = {}
    for file in (source / '_nib.py', source / '_transforms.py',
                 source / '_world_resampling.py', *sorted((source / 'synthmorph').glob('*.py'))):
        member = str(file.relative_to(source))
        got = digest(file)
        if manifest['files']['src/fnit/' + member] != got:
            raise RuntimeError('frozen source mismatch: ' + member)
        source_sha[member] = got
    expected_inputs = {
        'moving': 'afd1a20fe75fdea44313f0eda05020b916c87234e7a2045f7ccc6bb7c6e90b19',
        'fixed': '73e3866d4e54f9cb253868daab4bf90303a97bc193e8bda21e2e60c53a5dea21',
    }
    for name, expected in expected_inputs.items():
        if digest(getattr(args, name)) != expected:
            raise RuntimeError('real input hash mismatch: ' + name)
    weight = Path(args.weights) / 'synthmorph.affine.2.h5'
    expected_weight = '1ac5304b683036e5177f5b4ad38fa09fcbbe7883e742d6fa5bdaedd0e619ced6'
    if weight.stat().st_size != 51455312 or digest(weight) != expected_weight:
        raise RuntimeError('official affine weight size/hash mismatch')
    report = {
        'scope': 'one real full-grid materialized SpatialImage affine API verification; not cold CLI timing',
        'source_label': manifest['source_label'],
        'head_commit': manifest['head_commit'],
        'source_archive_sha256': manifest['archive_sha256'],
        'includes_reviewed_uncommitted_changes': manifest['includes_reviewed_uncommitted_changes'],
        'source_sha256': source_sha, 'worker_sha256': digest(__file__),
        'input_sha256': expected_inputs,
        'weight': {'file': weight.name, 'size': weight.stat().st_size, 'sha256': expected_weight},
        'model': 'affine', 'extent': 256, 'hyper': 0.5, 'steps': 7,
        'device': 'cpu', 'torch_threads': torch.get_num_threads(),
        'torch_interop_threads': torch.get_num_interop_threads(),
        'cpu_affinity': sorted(os.sched_getaffinity(0)),
        'environment': {key: os.environ.get(key) for key in (
            'CUDA_VISIBLE_DEVICES', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS',
            'OPENBLAS_NUM_THREADS', 'NUMBA_NUM_THREADS', 'ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS')},
    }
    start = time.perf_counter()
    raw = [nib.load(args.moving), nib.load(args.fixed)]
    report['image_load_seconds'] = time.perf_counter() - start
    start = time.perf_counter()
    pairs = [materialize(image) for image in raw]
    moving, fixed = [pair[0] for pair in pairs]
    report['input_materialize_and_check_seconds'] = time.perf_counter() - start
    report['inputs'] = dict(zip(('moving', 'fixed'), [pair[1] for pair in pairs]))
    report['arguments_are_materialized_spatial_images'] = all(
        isinstance(image, nib.spatialimages.SpatialImage) and
        isinstance(image.dataobj, np.ndarray) and image.get_filename() is None
        for image in (moving, fixed))
    flags = lambda: {'matmul_tf32': torch.backends.cuda.matmul.allow_tf32,
                     'cudnn_tf32': torch.backends.cudnn.allow_tf32}
    report['flags_before'] = flags()
    start = time.perf_counter()
    register = SynthMorph(weights={'affine': str(weight)}, device='cpu', model='affine', extent=256)
    report['model_load_seconds'] = time.perf_counter() - start
    report['flags_after_constructor'] = flags()
    precision = []
    start = time.perf_counter()
    result = register(moving, fixed, compute_inverse=True, precision_report=precision)
    report['api_seconds'] = time.perf_counter() - start
    report['precision_report'] = precision
    report['flags_after_call'] = flags()
    report['input_unchanged_after_call'] = all(
        image_summary(image) == report['inputs'][name]
        for name, image in (('moving', moving), ('fixed', fixed)))
    if not report['input_unchanged_after_call']:
        raise RuntimeError('public call mutated input data or geometry')
    report['outputs'] = {}
    start = time.perf_counter()
    for name, image in (('moved', result.moved), ('fixed_moved', result.fixed_moved)):
        if image is None:
            raise RuntimeError('missing full bidirectional output: ' + name)
        file = output / (name + '.nii.gz')
        image.save(file)
        report['outputs'][name] = image_summary(image)
        report['outputs'][name]['file_sha256'] = digest(file)
    for name, transform in (('forward', result.transform), ('inverse', result.inverse)):
        if transform is None:
            raise RuntimeError('missing bidirectional transform: ' + name)
        file = output / (name + '.lta')
        transform.save(file)
        report['outputs'][name] = {'matrix': transform.matrix.tolist(),
                                   'matrix_array_sha256': array_digest(transform.matrix),
                                   'file_sha256': digest(file)}
    report['save_and_output_hash_seconds'] = time.perf_counter() - start
    report['status'] = 'complete'
    (output / 'report.private.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')


if __name__ == '__main__':
    main()
