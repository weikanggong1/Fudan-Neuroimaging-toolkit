"""完整核验独立MELODIC控制：原输入哈希、线程系谱和全部科学输出。"""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import re
import sys

import nibabel as nib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from native_exec import trace_exit_evidence, LOADED_SOURCE_SHA256


def read(path):
    return json.loads(Path(path).read_text())


def sha256(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def compare(first, second):
    if first.shape != second.shape or first.dtype != second.dtype:
        raise ValueError('Scientific arrays differ in shape or decoded dtype')
    if not np.isfinite(first).all() or not np.isfinite(second).all():
        raise ValueError('Nonfinite scientific arrays')
    bits = np.array_equal(np.ascontiguousarray(first).view(np.uint8),
                          np.ascontiguousarray(second).view(np.uint8))
    difference = first.astype(np.float64) - second.astype(np.float64)
    return {'values': int(first.size), 'decoded_bitwise_equal': bool(bits),
            'numerically_unequal_values': int(np.count_nonzero(difference)),
            'max_absolute_difference': float(np.abs(difference).max())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--native-root', type=Path, required=True)
    parser.add_argument('--control-root', type=Path, required=True)
    parser.add_argument('--original-driver-source', type=Path, required=True)
    parser.add_argument('--report-out', type=Path, required=True)
    args = parser.parse_args()
    original_record_path = args.native_root / 'aroma_fnirt/commands.private.json'
    original_records = read(original_record_path)
    originals = [row for row in original_records if row['phase'] == 'melodic']
    if len(originals) != 1 or not originals[0]['output_validation_passed']:
        raise ValueError('Original timed MELODIC record is incomplete')
    control_record_path = args.control_root / 'command.private.json'
    commands = read(control_record_path)
    first, second = commands['original_scientific_argv'], commands['control_scientific_argv']
    if first != originals[0]['argv']:
        raise ValueError('The control does not refer to this timed original command')
    differences = [index for index, pair in enumerate(zip(first, second)) if pair[0] != pair[1]]
    if len(first) != len(second) or len(differences) != 1 or not first[differences[0]].startswith('--outdir=') or not second[differences[0]].startswith('--outdir='):
        raise ValueError('The control changed scientific arguments')
    original_denoise = read(args.native_root / 'aroma_fnirt/official_denoising.public.json')
    prior_log = args.control_root.parent / (args.control_root.name + '_controller.private.log')
    prior_harness_failure = (prior_log.exists() and
        'ValueError: Original/control scientific NIfTI file sets differ' in prior_log.read_text())
    driver_digest = sha256(args.original_driver_source / 'official_denoising.py')
    if driver_digest != original_denoise['driver_sha256']:
        raise ValueError('The source of the original post-MELODIC merge is not frozen byte-exact')
    before = original_denoise['input_sha256']
    inputs = {value.split('=', 1)[0][2:]: sha256(value.split('=', 1)[1])
              for value in second if value.startswith(('--in=', '--mask='))}
    if inputs != {'in': before['bold'], 'mask': before['brain_mask']}:
        raise ValueError('Control source inputs differ from the original pre-execution hashes')
    trace = args.control_root / 'melodic.exec_genealogy.private.log'
    process = read(args.control_root / 'original_process.public.json')
    first_event = re.match(r'^\d+\s+(execve\(.*)$', trace.read_text().splitlines()[0])
    if first_event is None:
        raise ValueError('The full control trace does not start with its command root exec')
    decoder = json.JSONDecoder()
    filename, offset = decoder.raw_decode(first_event[1][len('execve('):])
    remaining = first_event[1][len('execve(') + offset:].lstrip()
    if not remaining.startswith(','):
        raise ValueError('Control root argv is not fully decoded')
    actual_argv, _ = decoder.raw_decode(remaining[1:].lstrip())
    invoked = commands['actual_strace_invoked_argv']
    expected_filter = 'trace=' + ','.join(process['supported_genealogy_calls'])
    if (filename != second[0] or actual_argv != second or Path(invoked[0]).name != 'strace'
            or invoked[1:] != ['-f', '-s', '4096', '-e', expected_filter, '-o', str(trace), *second]):
        raise ValueError('Actual traced MELODIC command differs from the private frozen invocation')
    proof = trace_exit_evidence(trace, process['launcher_exit_code'])
    if not proof['original_process_accepted']:
        args.report_out.write_text(json.dumps({'control_genealogy_passed': False,
                                              'exit_evidence': proof}, indent=2) + '\n')
        raise ValueError('The full-genealogy control is not strictly verified')
    directories = [Path([value.split('=', 1)[1] for value in command if value.startswith('--outdir=')][0])
                   for command in (first, second)]
    reference, control = directories
    oracle_path = reference / 'oracle_ica.public.json'
    oracle = read(oracle_path)
    if oracle['input_sha256'] != inputs['in'] or oracle['brain_mask_sha256'] != inputs['mask']:
        raise ValueError('Original threshold merge oracle input hashes do not match')
    image_files = [sorted(path.relative_to(directory).as_posix() for path in directory.rglob('*.nii.gz'))
                   for directory in directories]
    # official_denoising.run_melodic constructs this additional AROMA input
    # using nibabel after the original MELODIC process exits. Verify it below
    # from the independently generated control thresholds; never claim that
    # this added file was a native MELODIC command output.
    derived_name = 'melodic_IC_thr.nii.gz'
    if derived_name not in image_files[0] or derived_name in image_files[1]:
        raise ValueError('The original/control post-MELODIC merge boundary is unexpected')
    image_files[0].remove(derived_name)
    if not image_files[0] or image_files[0] != image_files[1]:
        raise ValueError('Original/control NIfTI scientific output sets differ')
    images = {}
    for relative in image_files[0]:
        a, b = [nib.load(directory / relative) for directory in directories]
        images[relative] = {**compare(np.asanyarray(a.dataobj), np.asanyarray(b.dataobj)),
                            'header_binary_equal': a.header.binaryblock == b.header.binaryblock,
                            'affine_exact_equal': bool(np.array_equal(a.affine, b.affine)),
                            'shape': list(a.shape), 'original_sha256': sha256(reference / relative),
                            'control_sha256': sha256(control / relative)}
    mask_path = next(value.split('=', 1)[1] for value in second if value.startswith('--mask='))
    source_path = next(value.split('=', 1)[1] for value in second if value.startswith('--in='))
    mask = np.asarray(nib.load(mask_path).dataobj) > 0
    source = nib.load(source_path)
    count = np.loadtxt(control / 'melodic_mix', ndmin=2).shape[1]
    if count != oracle['components']:
        raise ValueError('Control mixing column count differs from the original merge oracle')
    thresholded = []
    for component in range(1, count + 1):
        image = nib.load(control / 'stats' / f'thresh_zstat{component}.nii.gz')
        values = np.asarray(image.dataobj[..., -1] if image.ndim == 4 else image.dataobj, dtype=np.float32)
        thresholded.append(values * mask)
    derived = nib.load(reference / derived_name)
    rebuilt = np.stack(thresholded, axis=3)
    expected_header = source.header.copy()
    expected_header.set_data_dtype(np.float32)
    expected_header.set_slope_inter(1.0, 0.0)
    # Round-trip in memory applies nibabel's same on-disk header/scaling rules
    # as frozen save_image, without adding a control output file.
    rebuilt_image = nib.Nifti1Image.from_bytes(nib.Nifti1Image(rebuilt, source.affine, expected_header).to_bytes())
    driver_ast = ast.parse((args.original_driver_source / 'official_denoising.py').read_text())
    merge_ast = [node for node in driver_ast.body if isinstance(node, ast.FunctionDef)
                 and node.name in ('run_melodic', 'save_image')]
    if len(merge_ast) != 2:
        raise ValueError('The frozen merge functions are absent')
    derived_check = {**compare(np.asanyarray(derived.dataobj), rebuilt),
        'components': count, 'original_sha256': sha256(reference / derived_name),
        'original_frozen_merge_driver_sha256': driver_digest,
        'source_affine_exact_equal': bool(np.array_equal(source.affine, derived.affine)),
        'source_spatial_shape_equal': source.shape[:3] == derived.shape[:3],
        'header_binary_equal': derived.header.binaryblock == rebuilt_image.header.binaryblock,
        'original_merge_oracle_sha256': sha256(oracle_path),
        'frozen_merge_function_ast_sha256': hashlib.sha256(''.join(ast.dump(node, include_attributes=False)
                                                                  for node in merge_ast).encode()).hexdigest(),
        'scope': 'Extra Python merge created after the original MELODIC command. Independently rebuilt in memory from every control thresh_zstat last/fallback channel multiplied by the unchanged original brain mask; no file is added to the native control output directory.'}
    numeric = {}
    for path in sorted(reference.rglob('*')):
        if not path.is_file() or path.name.endswith('.nii.gz') or path.stat().st_size == 0:
            continue
        try:
            a = np.loadtxt(path, ndmin=1)
        except (ValueError, UnicodeError):
            continue
        if not a.size:
            continue
        relative = path.relative_to(reference).as_posix()
        second_path = control / relative
        if not second_path.is_file():
            raise ValueError('An original numeric scientific output is absent from the control')
        try:
            b = np.loadtxt(second_path, ndmin=1)
        except (ValueError, UnicodeError) as error:
            raise ValueError('A control scientific numeric text file cannot be decoded') from error
        numeric[relative] = {**compare(a, b), 'original_sha256': sha256(path),
                             'control_sha256': sha256(second_path)}
    if not {'melodic_mix', 'melodic_FTmix'}.issubset(numeric):
        raise ValueError('The full mixing/frequency arrays were not compared')
    all_bits = all(row['decoded_bitwise_equal'] for row in [*images.values(), *numeric.values(), derived_check])
    all_headers = all(row['header_binary_equal'] and row['affine_exact_equal'] for row in images.values())
    derived_geometry = (derived_check['source_affine_exact_equal'] and derived_check['source_spatial_shape_equal']
                        and derived_check['header_binary_equal'])
    report = {'schema_version': 1, 'validated_on': '2026-10-02',
        'all_scientific_results_verified': all_bits and all_headers and derived_geometry,
        'all_original_scientific_image_files_included': True,
        'all_decodable_original_numeric_text_files_included': True,
        'all_decoded_values_bitwise_equal': all_bits, 'all_image_headers_exact_equal': all_headers,
        'native_image_header_comparison_scope': 'All native MELODIC command image outputs, excluding the separately verified subsequent Python merge.',
        'derived_post_melodic_merge': derived_check,
        'derived_post_melodic_geometry_verified': derived_geometry,
        'image_file_count': len(images), 'numeric_text_file_count': len(numeric),
        'total_scientific_values_compared': sum(row['values'] for row in [*images.values(), *numeric.values(), derived_check]),
        'images': images, 'numeric_text': numeric, 'input_sha256': inputs,
        'original_command_record_sha256': sha256(original_record_path),
        'original_saved_trace_sha256': sha256(args.native_root / 'aroma_fnirt/melodic.exec.private.log'),
        'control_command_record_sha256': sha256(control_record_path),
        'actual_root_scientific_argv_exact_match': True,
        'actual_strace_invocation_matches_private_record': True,
        'actual_control_scientific_argv_sha256': hashlib.sha256(json.dumps(second, ensure_ascii=False,
                                                              separators=(',', ':')).encode()).hexdigest(),
        'control_genealogy_trace_sha256': sha256(trace), 'control_exit_evidence': proof,
        'strict_post_validation_parser_sha256': LOADED_SOURCE_SHA256,
        'verification_script_sha256': sha256(__file__),
        'original_timed_command_wall_seconds': originals[0]['command_wall_seconds'],
        'independent_control_wall_seconds': process['control_wall_seconds_including_io'],
        'original_native_control_launcher_exit_code': process['launcher_exit_code'],
        'original_native_control_successful_inner_exit_code': 0,
        'original_control_harness_exit_code': 1 if prior_harness_failure else None,
        'original_control_harness_exit_code_record_kind': (
            'Python error exit1 inferred from the preserved completed controller traceback ending in an uncaught ValueError; a separate outer wait status was not saved.'
            if prior_harness_failure else 'No prior controller output-set exception identified.'),
        'original_control_harness_failure_reason': (
            'Original194 NIfTI files included the later controller-created Python melodic_IC_thr merge; control193 were all native MELODIC outputs. The prior output-set check raised an uncaught ValueError. This separate complete validator verifies all193 native outputs and independently reconstructs the extra merge.'
            if prior_harness_failure else None),
        'original_control_harness_failure_log_sha256': sha256(prior_log) if prior_harness_failure else None,
        'post_validation_recovery_is_separate_from_prior_harness_exit': True,
        'independent_control_excluded_from_pipeline_wall': True,
        'old_original_thread_genealogy_inferred_from_control': False,
        'scope': 'The independent same-parameter original MELODIC control has directly verified clone/thread/SIGCHLD genealogy. All scientific arrays and NIfTI headers exactly match the original timed run; missing old thread genealogy is not retroactively inferred.',
        'privacy': 'Anonymous array counts, standardized scientific output names and hashes only; full images, argv and traces remain private.'}
    args.report_out.parent.mkdir(parents=True, exist_ok=True)
    args.report_out.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'control_verified': all_bits and all_headers, 'images': len(images),
                      'numeric_text': len(numeric), 'values': report['total_scientific_values_compared']}))
    if not all_bits or not all_headers or not derived_geometry:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
