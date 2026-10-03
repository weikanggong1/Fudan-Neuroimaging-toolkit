"""核对两次本轮原软件运行的脑图、完整motion及配准文件；不复用计算。"""
import argparse
import hashlib
import json
from pathlib import Path

import nibabel as nib
import numpy as np


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--first-root', type=Path, required=True)
    parser.add_argument('--retry-root', type=Path, required=True)
    parser.add_argument('--report-out', type=Path, required=True)
    args = parser.parse_args()
    relative = ['masks/epi_brain.nii.gz', 'masks/epi_mask.nii.gz',
        'anat/T1_brain.nii.gz', 'anat/T1_mask.nii.gz', 'feat/example_func.nii.gz',
        'feat/mc/prefiltered_func_data_mcf.nii.gz', 'feat/mc/prefiltered_func_data_mcf.par',
        'feat/filtered_func_data.nii.gz']
    relative += [path.relative_to(args.first_root).as_posix()
                 for path in sorted((args.first_root / 'reg_fnirt').iterdir())
                 if path.name.endswith(('.mat', '.nii.gz', '.txt'))]
    matrices = sorted((args.first_root / 'feat/mc/prefiltered_func_data_mcf.mat').glob('MAT_*'))
    if len(matrices) != 490:
        raise ValueError('The original run must contain all490 motion matrices')
    relative += [path.relative_to(args.first_root).as_posix() for path in matrices]
    rows = {}
    for name in relative:
        first, second = (directory / name for directory in (args.first_root, args.retry_root))
        if not first.is_file() or not second.is_file():
            raise ValueError('A compared original result is absent')
        hashes = {'first': sha256(first), 'retry': sha256(second)}
        row = {'file_sha256': hashes, 'saved_file_identical': hashes['first'] == hashes['retry']}
        if name.endswith('.nii.gz'):
            a, b = (nib.load(path) for path in (first, second))
            row['same_shape'] = a.shape == b.shape
            row['same_saved_dtype'] = a.get_data_dtype() == b.get_data_dtype()
            row['affine_exact_equal'] = bool(np.array_equal(a.affine, b.affine))
            row['header_binary_equal'] = a.header.binaryblock == b.header.binaryblock
            if row['saved_file_identical']:
                row['decoded_values_bitwise_equal'] = True
            else:
                x, y = (np.asanyarray(image.dataobj) for image in (a, b))
                row['decoded_values_bitwise_equal'] = bool(x.shape == y.shape and x.dtype == y.dtype
                    and np.array_equal(np.ascontiguousarray(x).view(np.uint8),
                                       np.ascontiguousarray(y).view(np.uint8)))
                del x, y
        else:
            x, y = (np.loadtxt(path, ndmin=1) for path in (first, second))
            row['same_shape'] = x.shape == y.shape
            row['decoded_values_bitwise_equal'] = bool(x.shape == y.shape and np.array_equal(x.view(np.uint8), y.view(np.uint8)))
        rows[name] = row
    passed = all(row['decoded_values_bitwise_equal'] and row['same_shape']
                 and row.get('affine_exact_equal', True) and row.get('header_binary_equal', True)
                 for row in rows.values())
    report = {'schema_version': 1, 'validated_on': '2026-10-02',
        'all_scientific_values_and_geometry_exact_equal': passed,
        'compared_files': len(rows), 'motion_matrices': 490, 'files': rows,
        'first_registration_report_sha256': sha256(args.first_root / 'reg_fnirt/registration.public.json'),
        'retry_registration_report_sha256': sha256(args.retry_root / 'reg_fnirt/registration.public.json'),
        'verification_script_sha256': sha256(__file__),
        'computational_reuse_between_whole_clean_runs': False,
        'scope': 'The first incomplete original clean run and the fresh complete retry independently estimated anatomy, all490 motion matrices, T1 affine/FNIRT and EPI BBR. Supplemental original preproc and matched surface reuse the first run registration; the complete clean comparison uses the retry. This check establishes their actual numerical relationship without changing either run or its timing.',
        'privacy': 'Standard scientific output names and anonymous hashes only; images and matrices stay private.'}
    args.report_out.parent.mkdir(parents=True, exist_ok=True)
    args.report_out.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'passed': passed, 'files': len(rows)}))
    if not passed:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
