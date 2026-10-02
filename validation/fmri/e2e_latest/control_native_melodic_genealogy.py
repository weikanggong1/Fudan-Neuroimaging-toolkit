"""原MELODIC同参数独立复跑：补线程系谱并逐项核验原科学数值输出。"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

import nibabel as nib
import numpy as np


def sha256(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def compare_values(first, second):
    if first.shape != second.shape or first.dtype != second.dtype:
        raise ValueError('Original/control output shape or decoded dtype differs')
    if not np.isfinite(first).all() or not np.isfinite(second).all():
        raise ValueError('Nonfinite original/control output')
    bitwise = np.array_equal(np.ascontiguousarray(first).view(np.uint8),
                             np.ascontiguousarray(second).view(np.uint8))
    error = first.astype(np.float64) - second.astype(np.float64)
    return {'values': int(first.size), 'decoded_dtype': str(first.dtype),
            'decoded_bitwise_equal': bool(bitwise),
            'numerically_unequal_values': int(np.count_nonzero(error)),
            'max_absolute_difference': float(np.abs(error).max()),
            'rmse': float(np.sqrt(np.square(error).mean()))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--native-root', type=Path, required=True)
    parser.add_argument('--case-json', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    args = parser.parse_args()
    output = args.output_root
    if output.exists():
        raise FileExistsError('Use a new genealogy-control output directory')
    output.mkdir(parents=True)
    case = json.loads(args.case_json.read_text())
    source_records = args.native_root / 'aroma_fnirt/commands.private.json'
    records = json.loads(source_records.read_text())
    positions = [index for index, row in enumerate(records) if row['phase'] == 'melodic']
    if len(positions) != 1 or not records[positions[0]]['output_validation_passed']:
        raise ValueError('The original timed MELODIC entry is not fully validated')
    original = records[positions[0]]['argv']
    directories = [Path(value.split('=', 1)[1]) for value in original if value.startswith('--outdir=')]
    if len(directories) != 1:
        raise ValueError('Ambiguous original MELODIC output directory')
    reference = directories[0]
    target = output / 'melodic.ica'
    control = [('--outdir=' + str(target)) if value.startswith('--outdir=') else value for value in original]
    changed = [index for index, values in enumerate(zip(original, control)) if values[0] != values[1]]
    if len(changed) != 1:
        raise ValueError('Only the output destination may change in this control')
    original_input_record = json.loads((args.native_root / 'aroma_fnirt/official_denoising.public.json').read_text())['input_sha256']
    input_hashes = {value.split('=', 1)[0][2:]: sha256(value.split('=', 1)[1])
                    for value in original if value.startswith(('--in=', '--mask='))}
    if input_hashes != {'in': original_input_record['bold'], 'mask': original_input_record['brain_mask']}:
        raise ValueError('Control inputs differ from the original execution-time hashes')
    environment = dict(os.environ, FSLDIR=case['fsl_root'], FSLOUTPUTTYPE='NIFTI_GZ',
                       OMP_NUM_THREADS='8', MKL_NUM_THREADS='8', OPENBLAS_NUM_THREADS='8')
    environment['LD_LIBRARY_PATH'] = case['fsl_root'] + '/lib:' + environment.get('LD_LIBRARY_PATH', '')
    environment['PATH'] = case['fsl_root'] + '/bin:' + environment.get('PATH', '')
    tracer = shutil.which('strace')
    if not tracer:
        raise FileNotFoundError('Genealogy control needs strace')
    calls = ['execve', 'clone', 'fork', 'vfork', 'exit', 'exit_group', 'wait4', 'waitid']
    clone3 = subprocess.run([tracer, '-e', 'trace=clone3', '-o', '/dev/null', '/bin/true'],
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE).returncode == 0
    if clone3:
        calls.append('clone3')
    trace = output / 'melodic.exec_genealogy.private.log'
    invoked = [tracer, '-f', '-s', '4096', '-e', 'trace=' + ','.join(calls), '-o', str(trace), *control]
    (output / 'command.private.json').write_text(json.dumps({
        'original_scientific_argv': original, 'control_scientific_argv': control,
        'actual_strace_invoked_argv': invoked, 'original_record_pointer': [positions[0], 'argv']}, indent=2) + '\n')
    (output / 'status.private.json').write_text(json.dumps({'stage': 'melodic', 'status': 'running'}) + '\n')
    started = time.perf_counter()
    with (output / 'melodic.private.log').open('wb') as log:
        result = subprocess.run(invoked, env=environment, stdout=log, stderr=subprocess.STDOUT)
    wall = time.perf_counter() - started
    (output / 'original_process.public.json').write_text(json.dumps({
        'launcher_exit_code': result.returncode, 'control_wall_seconds_including_io': wall,
        'supported_genealogy_calls': calls, 'clone3_supported': clone3}, indent=2) + '\n')
    if result.returncode not in (0, 255):
        raise RuntimeError('Original MELODIC control returned an unexpected code')
    (output / 'status.private.json').write_text(json.dumps({'stage': 'full_output_comparison', 'status': 'running'}) + '\n')
    image_records = {}
    original_images = sorted(path.relative_to(reference).as_posix() for path in reference.rglob('*.nii.gz'))
    control_images = sorted(path.relative_to(target).as_posix() for path in target.rglob('*.nii.gz'))
    # This file is created by official_denoising.py using nibabel after
    # MELODIC exits. The strong verifier independently rebuilds it from all
    # control thresholds; it is not an output of this native command.
    if 'melodic_IC_thr.nii.gz' in original_images:
        original_images.remove('melodic_IC_thr.nii.gz')
    if original_images != control_images or not original_images:
        raise ValueError('Original/control scientific NIfTI file sets differ')
    for relative in original_images:
        first, second = [nib.load(directory / relative) for directory in (reference, target)]
        image_records[relative] = {**compare_values(np.asanyarray(first.dataobj), np.asanyarray(second.dataobj)),
            'header_binary_equal': first.header.binaryblock == second.header.binaryblock,
            'affine_exact_equal': bool(np.array_equal(first.affine, second.affine)),
            'stored_dtype_equal': first.get_data_dtype() == second.get_data_dtype(),
            'shape': list(first.shape), 'original_sha256': sha256(reference / relative),
            'control_sha256': sha256(target / relative)}
    numerical_records = {}
    for first in sorted(reference.rglob('*')):
        if not first.is_file() or first.name.endswith('.nii.gz') or first.stat().st_size == 0:
            continue
        relative = first.relative_to(reference).as_posix()
        try:
            a = np.loadtxt(first, ndmin=1)
        except (ValueError, UnicodeError):
            continue
        if a.size:
            second = target / relative
            if not second.is_file():
                raise ValueError('A scientific numeric text output is missing from the control')
            try:
                b = np.loadtxt(second, ndmin=1)
            except (ValueError, UnicodeError) as error:
                raise ValueError('A control numeric text output cannot be decoded') from error
            numerical_records[relative] = {**compare_values(a, b), 'original_sha256': sha256(first),
                                            'control_sha256': sha256(second)}
    required = {'melodic_mix', 'melodic_FTmix'}
    if not required.issubset(numerical_records):
        raise ValueError('Original/control mixing and spectral arrays are missing')
    all_bits = all(row['decoded_bitwise_equal'] for row in [*image_records.values(), *numerical_records.values()])
    all_headers = all(row['header_binary_equal'] and row['affine_exact_equal'] and row['stored_dtype_equal']
                      for row in image_records.values())
    report = {'schema_version': 1, 'validated_on': '2026-10-02',
        'original_timed_command_wall_seconds': records[positions[0]]['command_wall_seconds'],
        'control_wall_seconds_including_io': wall, 'control_excluded_from_pipeline_wall': True,
        'launcher_exit_code': result.returncode, 'exit_code_rewritten': False,
        'scientific_arguments_identical_except_output_destination': True,
        'image_scope': 'Native MELODIC command outputs; the original later Python melodic_IC_thr merge is verified separately by verify_native_melodic_control.py.',
        'all_scientific_decoded_values_bitwise_equal': all_bits, 'all_image_headers_exact_equal': all_headers,
        'image_file_count': len(image_records), 'numeric_text_file_count': len(numerical_records),
        'image_comparisons': image_records, 'numeric_text_comparisons': numerical_records,
        'input_sha256': input_hashes,
        'source_original_command_record_sha256': sha256(source_records),
        'control_private_command_record_sha256': sha256(output / 'command.private.json'),
        'control_full_genealogy_trace_sha256': sha256(trace),
        'original_launcher_sha256': sha256(control[0]), 'driver_sha256': sha256(__file__),
        'supported_genealogy_calls': calls, 'clone3_supported': clone3,
        'strict_original_thread_genealogy_inferred_from_control': False,
        'scope': 'Separate original MELODIC execution with identical scientific inputs/parameters and full available clone/fork/wait trace. It validates its own genealogy and exact scientific-output identity; it does not retrospectively create missing clone evidence for the old timed trace.',
        'privacy': 'Anonymous scientific output names, scalar comparisons and hashes only; images, argv and genealogy stay private.'}
    (output / 'control.public.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    (output / 'status.private.json').write_text(json.dumps({'stage': 'complete', 'status': 'passed' if all_bits and all_headers else 'difference'}) + '\n')
    print(json.dumps({'control_complete': True, 'bitwise': all_bits, 'headers': all_headers,
                      'images': len(image_records), 'numeric_text': len(numerical_records)}), flush=True)
    if not all_bits or not all_headers:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
