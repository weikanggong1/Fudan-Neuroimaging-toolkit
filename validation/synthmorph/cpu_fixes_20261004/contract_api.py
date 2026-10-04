"""Complete materialized-image CPU API and returned-affine image contract."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time

import nibabel as nib
import numpy as np
import torch
from fnit.synthmorph import SynthMorph, apply_transform
from fnit._transforms import load_lta


def digest_array(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('moving', 'fixed', 'weights', 'output', 'cli'):
        p.add_argument('--' + name, required=True)
    p.add_argument('--model', choices=('rigid', 'affine'), default='affine')
    p.add_argument('--extent', type=int, default=256)
    args = p.parse_args()
    torch.set_num_threads(8)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    images = []
    for name in ('moving', 'fixed'):
        image = nib.load(getattr(args, name))
        materialized = nib.Nifti1Image(np.array(np.asanyarray(image.dataobj), copy=True),
                                     image.affine.copy(), image.header.copy())
        if not isinstance(materialized.dataobj, np.ndarray) or materialized.get_filename() is not None:
            raise RuntimeError('input is not a materialized image')
        images.append(materialized)
    before = [digest_array(image.dataobj) for image in images]
    started = time.perf_counter()
    register = SynthMorph(weights=args.weights, model=args.model, device='cpu', extent=args.extent)
    load_seconds = time.perf_counter() - started
    started = time.perf_counter()
    result = register(*images)
    api_seconds = time.perf_counter() - started
    report = {'scope': 'complete real materialized-image API and returned-affine image contract',
              'model': args.model, 'extent': args.extent, 'host': os.uname().nodename,
              'model_load_seconds': load_seconds, 'api_seconds': api_seconds,
              'source_sha256': {}, 'rows': []}
    import fnit.synthmorph.pipeline as pipeline
    import fnit.synthmorph.models as models
    import fnit.synthmorph.spatial as spatial
    for name, module in [('pipeline', pipeline), ('models', models), ('spatial', spatial)]:
        report['source_sha256'][name] = hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()
    for i, (name, transformation, image) in enumerate([
        ('moved', result.transform, result.moved),
        ('fixed_moved', result.inverse, result.fixed_moved),
    ]):
        reapplied = apply_transform(images[i], transformation, device='cpu')
        cli = nib.load(Path(args.cli) / (name + '.nii.gz'))
        suffix = 'forward' if i == 0 else 'inverse'
        cli_transform = load_lta(Path(args.cli) / (suffix + '.lta'))
        row = {'image': name,
               'array_sha256': digest_array(image.dataobj),
               'same_returned_affine_array': bool(np.array_equal(image.dataobj, reapplied.dataobj)),
               'same_returned_affine_header': image.header.binaryblock == reapplied.header.binaryblock,
               'same_cli_array': bool(np.array_equal(image.dataobj, np.asanyarray(cli.dataobj))),
               'same_cli_header': image.header.binaryblock == cli.header.binaryblock,
               'cli_world_matrix_max_abs': float(np.abs(transformation.matrix - cli_transform.matrix).max())}
        image.save(output / (name + '.nii.gz'))
        transformation.save(output / (suffix + '.lta'))
        row['gate_passed'] = all(row[key] for key in ['same_returned_affine_array',
            'same_returned_affine_header', 'same_cli_array', 'same_cli_header']) and row['cli_world_matrix_max_abs'] <= 1e-12
        report['rows'].append(row)
    report['input_unchanged'] = before == [digest_array(image.dataobj) for image in images]
    report['all_gates_passed'] = report['input_unchanged'] and all(row['gate_passed'] for row in report['rows'])
    (output / 'report.private.json').write_text(json.dumps(report, indent=2) + '\n')
    if not report['all_gates_passed']:
        raise RuntimeError('complete materialized-image/returned-affine contract failed')


if __name__ == '__main__':
    main()
