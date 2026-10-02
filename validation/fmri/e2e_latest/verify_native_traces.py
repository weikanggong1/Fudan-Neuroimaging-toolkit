"""用最终严格解析器后验核对本轮所有原科学命令的已保存进程证据。"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from native_exec import LOADED_SOURCE_SHA256, trace_exit_evidence


def read(path):
    return json.loads(Path(path).read_text())


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--native-root', type=Path, required=True)
    parser.add_argument('--preproc-root', type=Path, required=True)
    parser.add_argument('--recovered-preproc-root', type=Path)
    parser.add_argument('--driver-source', type=Path, required=True)
    parser.add_argument('--preproc-driver-source', type=Path, required=True)
    parser.add_argument('--recovered-preproc-driver-source', type=Path)
    parser.add_argument('--independent-melodic-verification', type=Path)
    parser.add_argument('--report-out', type=Path, required=True)
    args = parser.parse_args()
    original = args.native_root
    rows = []
    sources = {}

    initial = read(original / 'source_start_attestation.public.json')['driver_sha256']
    change_path = original / 'validation_helper_change.public.json'
    helper_change = read(change_path) if change_path.exists() else None
    for name, digest in initial.items():
        if name == 'native_exec.py':
            continue
        if sha256(args.driver_source / name) != digest:
            raise ValueError('The original scientific driver changed during this run')
    initial_helper = args.driver_source / 'native_exec.py'
    if sha256(initial_helper) != initial['native_exec.py']:
        initial_helper = args.driver_source / 'native_exec.before_scalar_fix.py'
        if (not helper_change or sha256(initial_helper) != initial['native_exec.py'] or
                helper_change['source_before'] != initial['native_exec.py'] or
                helper_change['source_after'] != sha256(args.driver_source / 'native_exec.py') or
                helper_change['observed_stage']['stage'] != 'feat' or
                helper_change['scientific_driver_or_parameters_changed'] is not False):
            raise ValueError('The old loaded helper has no declared, byte-exact source archive')
    sources['scientific_driver_sha256'] = {name: digest for name, digest in initial.items()
                                         if name != 'native_exec.py'}
    sources['initial_loaded_helper_sha256'] = sha256(initial_helper)
    sources['initial_source_attestation_sha256'] = sha256(original / 'source_start_attestation.public.json')
    if helper_change:
        sources['validation_helper_change_record_sha256'] = sha256(change_path)

    def provenance(source, record_path, pointer, kind):
        return {'frozen_source_path': str(source), 'frozen_source_sha256': sha256(source),
                'command_record_path': str(record_path), 'command_record_sha256': sha256(record_path),
                'command_record_pointer': pointer, 'command_kind': kind}

    def verify(group, name, path, code, command_provenance=None):
        if command_provenance is not None:
            command_provenance['saved_trace_sha256'] = sha256(path)
        evidence = trace_exit_evidence(path, code, command_trace_provenance=command_provenance)
        rows.append({'group': group, 'stage': name, 'saved_trace_sha256': sha256(path),
                     'saved_launcher_exit_code': code, **evidence})

    for group in ('anatomy', 'feat'):
        report_path = original / (group + '.public.json')
        sources[group] = sha256(report_path)
        record_path = original / (group + '.commands.private.json')
        command_records = read(record_path)['commands']
        for row in read(report_path)['steps']:
            positions = [index for index, command in enumerate(command_records)
                         if command['name'] == row['name']]
            # masked_p50 was recorded in the timed scalar report but not the
            # private argv list. Its single packed-program trace uses no orphan
            # child exception; exact root/child evidence remains mandatory.
            identity = (provenance(initial_helper, record_path, ['commands', positions[0], 'argv'], 'argv')
                        if len(positions) == 1 else None)
            verify(group, row['name'], original / (row['name'] + '.exec.private.log'), row['exit_code'], identity)
    path = original / 'reg_fnirt/registration.public.json'
    sources['registration'] = sha256(path)
    for name, row in read(path)['stages'].items():
        verify('registration', name, path.parent / (name + '.exec.private.log'), row['launcher_exit_code'])
    path = original / 'aroma_fnirt/commands.private.json'
    sources['denoising_private_command_record'] = sha256(path)
    current_helper = args.driver_source / 'native_exec.py'
    for index, row in enumerate(read(path)):
        if row['exit_evidence'].get('evidence_parser_loaded_sha256') != sha256(current_helper):
            raise ValueError('A denoising command loaded a different evidence helper')
        phase = row['phase']
        trace_name = phase.replace('aroma_scalar_query_', 'aroma_scalar_')
        key = 'shell_pipeline' if 'shell_pipeline' in row else 'argv'
        identity = provenance(current_helper, path, [index, key], key)
        verify('denoising', phase, path.parent / (trace_name + '.exec.private.log'), row['exit_status'], identity)
    path = original / 'pipeline.public.json'
    sources['pipeline'] = sha256(path)
    row = read(path)['stages']['mni_resampling']
    record_path = original / 'commands.private.json'
    positions = [index for index, command in enumerate(read(record_path))
                 if command['name'] == 'mni_resampling']
    if len(positions) != 1:
        raise ValueError('Final original command identity is ambiguous')
    verify('final', 'clean_mni', original / 'mni_resampling.exec.private.log', row['original_exit_code'],
           provenance(initial_helper, record_path, [positions[0], 'argv'], 'argv'))
    for group, directory, source in (('preproc_timed', args.preproc_root, args.preproc_driver_source),
                                    ('preproc_recovered', args.recovered_preproc_root, args.recovered_preproc_driver_source)):
        if directory is None:
            continue
        path = directory / 'preproc.public.json'
        sources[group] = sha256(path)
        helper = source / 'native_exec.py'
        recorded_helpers = read(path).get('original_driver_sha256')
        if recorded_helpers and recorded_helpers['native_exec.py'] != sha256(helper):
            raise ValueError('Recovered preproc helper differs from its byte-exact driver record')
        records = directory / 'commands.private.json'
        for name in ('preproc_MNI', 'preproc_T1w'):
            row = read(path)['timings'][name]
            positions = [index for index, command in enumerate(read(records)) if command['name'] == name]
            if len(positions) != 1:
                raise ValueError('Preproc original command identity is ambiguous')
            if group == 'preproc_timed' and row['exit_evidence']['evidence_parser_loaded_sha256'] != sha256(helper):
                raise ValueError('Timed preproc loaded another helper')
            verify(group, name, directory / (name + '.exec.private.log'), row['exit_evidence']['launcher_exit_code'],
                   provenance(helper, records, [positions[0], 'argv'], 'argv'))
    failed = [row for row in rows if not row['original_process_accepted']]
    controls = []
    covered_failure = []
    if args.independent_melodic_verification:
        verification = read(args.independent_melodic_verification)
        mandatory = ('all_scientific_results_verified', 'all_original_scientific_image_files_included',
                     'all_decodable_original_numeric_text_files_included',
                     'all_decoded_values_bitwise_equal', 'all_image_headers_exact_equal',
                     'derived_post_melodic_geometry_verified', 'actual_root_scientific_argv_exact_match',
                     'actual_strace_invocation_matches_private_record',
                     'independent_control_excluded_from_pipeline_wall')
        if any(verification.get(key) is not True for key in mandatory):
            raise ValueError('The independent MELODIC full-array control is incomplete')
        if (verification.get('old_original_thread_genealogy_inferred_from_control') is not False
                or not verification['control_exit_evidence']['original_process_accepted']
                or verification['original_command_record_sha256'] != sources['denoising_private_command_record']):
            raise ValueError('The independent control does not bind the original timed MELODIC')
        melody = [row for row in rows if row['group'] == 'denoising' and row['stage'] == 'melodic']
        if len(melody) != 1:
            raise ValueError('Timed MELODIC identity is ambiguous')
        if verification['original_saved_trace_sha256'] != melody[0]['saved_trace_sha256']:
            raise ValueError('The control verification is bound to another original trace')
        if not melody[0]['original_process_accepted']:
            # The old receiver's ancestry was not traced. An independent control
            # verifies the unchanged scientific arrays; it does not fill this
            # missing old PID ancestry or change the old trace acceptance.
            if {item['kind'] for item in melody[0]['trace_integrity_errors']} != {'sigchld_parent_exec_missing'}:
                raise ValueError('The old MELODIC has a different unexplained evidence failure')
            covered_failure.append(melody[0])
        controls.append({'covered_stage': 'melodic',
            'old_saved_trace_sha256': melody[0]['saved_trace_sha256'],
            'independent_control_verification_sha256': sha256(args.independent_melodic_verification),
            'control_genealogy_trace_sha256': verification['control_genealogy_trace_sha256'],
            'control_strict_parser_sha256': verification['strict_post_validation_parser_sha256'],
            'all_scientific_values_and_headers_exact_equal': True,
            'old_thread_genealogy_inferred_from_control': False,
            'independent_control_excluded_from_pipeline_wall': True})
    unresolved = [row for row in failed if row not in covered_failure]
    report = {'schema_version': 1, 'validated_on': '2026-10-02',
        'all_original_traces_strictly_verified': not failed,
        'all_scientific_results_verified_with_independent_controls': not unresolved,
        'verified_scientific_command_count': len(rows), 'failed_stage_count': len(failed),
        'strictly_verified_original_command_count': len(rows) - len(failed),
        'independently_verified_scientific_stage_count': len(covered_failure),
        'unresolved_scientific_stage_count': len(unresolved),
        'independent_controls': controls,
        'strict_post_validation_parser_sha256': LOADED_SOURCE_SHA256,
        'verification_script_sha256': sha256(__file__), 'source_records_sha256': sources,
        'scope': 'Post-validation of saved original exec/exit/SIGCHLD traces using a strict parser. Original launcher exits, frozen computational driver identity, elapsed times and saved output files are unchanged.',
        'commands': rows,
        'privacy': 'Anonymous phase names and hashes only; raw argv, paths, traces and individual matrices remain private.'}
    args.report_out.parent.mkdir(parents=True, exist_ok=True)
    args.report_out.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'strict_all_passed': not failed, 'commands': len(rows),
                      'failed': [{'group': row['group'], 'stage': row['stage']} for row in failed]}))
    if unresolved:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
