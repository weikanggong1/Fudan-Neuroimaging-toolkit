"""汇总本轮原 SynthStrip/FSL/ICA-AROMA 连续 clean 链的匿名记录。"""
import argparse
import hashlib
import json
from pathlib import Path


def read(path):
    return json.loads(Path(path).read_text())


def sha256(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--native-root', type=Path, required=True)
    parser.add_argument('--driver-source', type=Path, required=True)
    parser.add_argument('--report-out', type=Path, required=True)
    parser.add_argument('--candidate-revision', required=True)
    args = parser.parse_args()
    root = args.native_root
    reports = {
        'pipeline': root / 'pipeline.public.json',
        'anatomy': root / 'anatomy.public.json',
        'feat': root / 'feat.public.json',
        'registration': root / 'reg_fnirt/registration.public.json',
        'denoising': root / 'aroma_fnirt/official_denoising.public.json'}
    content = {name: read(path) for name, path in reports.items()}
    if read(root / 'status.private.json') != {'stage': 'complete', 'status': 'passed'}:
        raise RuntimeError('The new continuous original run is not complete')
    initial_source = read(root / 'source_start_attestation.public.json')
    start = initial_source['driver_sha256']
    end = {name: sha256(args.driver_source / name) for name in start}
    helper_change = None
    if start != end:
        change_path = root / 'validation_helper_change.public.json'
        if not change_path.is_file():
            raise RuntimeError('Undeclared original benchmark source change')
        helper_change = read(change_path)
        changed_files = {name for name in start if start[name] != end[name]}
        if (changed_files != {'native_exec.py'} or
                helper_change['source_before'] != start['native_exec.py'] or
                helper_change['source_after'] != end['native_exec.py'] or
                helper_change['scientific_driver_or_parameters_changed'] is not False):
            raise RuntimeError('Only the declared validation evidence parser repair is permitted')
    commands = read(root / 'aroma_fnirt/commands.private.json')
    # Private argv / shell_pipeline never enter the public report.
    denoise_commands = [{key: value for key, value in record.items()
                         if key not in ('argv', 'shell_pipeline')} for record in commands]
    evidence = []
    for name in ('anatomy', 'feat'):
        for row in content[name]['steps']:
            if 'exit_evidence' in row:
                evidence.append({'step': row['name'], **row['exit_evidence']})
    for row in denoise_commands:
        if 'exit_evidence' in row:
            evidence.append({'step': row['phase'], **row['exit_evidence']})
    evidence.append({'step': 'mni_resampling',
                     **content['pipeline']['stages']['mni_resampling']['exit_evidence']})
    if not evidence or not all(row['original_process_accepted'] for row in evidence):
        raise RuntimeError('A native computation has no successful exit evidence')
    registration = content['registration']['stages']
    if not all(stage['launcher_exit_code'] == 0 or
               (stage['launcher_exit_code'] == 255 and stage['actual_child_exit_code'] == 0)
               for stage in registration.values()):
        raise RuntimeError('Native registration lacks successful-child evidence')
    report = {
        'schema_version': 1, 'validated_on': '2026-10-02',
        'candidate_revision': args.candidate_revision,
        'protocol': 'Fresh raw BOLD/SBRef and archived same-subject T1 -> original SynthStrip, FSL FAST/MCFLIRT/FLIRT/BBR/FNIRT/MELODIC/ICA-AROMA, independent NumPy joint nuisance regression, original FSL applywarp -> native and MNI clean.',
        'outputs': ['clean_native', 'clean_mni'],
        'preproc_outputs_included': False,
        'no_computational_stage_reused': True,
        'input_sha256': content['pipeline']['input_sha256'],
        'output_sha256': content['pipeline']['output_sha256'],
        'whole_workflow_wall_seconds': content['pipeline']['whole_workflow_wall_seconds'],
        'whole_workflow_stages': content['pipeline']['stages'],
        'anatomy_commands': content['anatomy']['steps'],
        'feat_commands': content['feat']['steps'],
        'registration_commands': registration,
        'ica': content['denoising']['ica'],
        'aroma': content['denoising']['aroma'],
        'confounds': content['denoising']['confounds'],
        'denoising_timing_seconds': content['denoising']['timing_seconds'],
        'denoising_commands': denoise_commands,
        'software': {'fsl_installation_version': content['anatomy']['fsl_installation_version'],
                     'fsl_component_packages': content['anatomy']['component_packages'],
                     'fsl_melodic': content['denoising']['software'],
                     'configured_cpu_threads': content['anatomy']['threads'],
                     'original_synthstrip_device': 'cuda:0',
                     'native_fsl_device': 'CPU'},
        'exit_proof': {'trace_enabled': True,
                       'all_abnormal_launchers_have_inner_success': True,
                       'commands_with_exec_sigchld_evidence': len(evidence),
                       'abnormal_launcher_count': sum(len(row['exit255_records']) for row in evidence),
                       'exit_codes_rewritten': False},
        'source_sha256_before': start, 'source_sha256_after': end,
        'initial_source_observation_phase': initial_source.get('observed_phase', 'anatomy'),
        'source_unchanged': start == end,
        'scientific_driver_source_unchanged': all(start[name] == end[name] for name in start if name != 'native_exec.py'),
        'validation_only_helper_change': helper_change,
        'recorded_loaded_evidence_parser_sha256': sorted({row['evidence_parser_loaded_sha256'] for row in evidence if 'evidence_parser_loaded_sha256' in row}),
        'component_sha256': content['anatomy']['component_sha256'],
        'underlying_report_sha256': {name: sha256(path) for name, path in reports.items()},
        'aggregation_script_sha256': sha256(__file__),
        'timing_boundary': content['pipeline']['timing_boundary'],
        'limits': [
            'The complete chain outputs clean only; current FNIT volume also outputs T1w/MNI preproc. Use matching branch/stage boundaries rather than raw total ratios.',
            'Each native command timer includes original startup, read/write and exec/exit tracing; validations are outside command timers but inside the continuous workflow.',
            'Configured CPU threads are environment limits, not a claim that every original FSL command uses eight active cores.',
            'One subject and one new run on a shared host; no stable speedup claim.',
            'Same archived T1 reconstruction as previous benchmark, not verified scanner-raw T1.'],
        'privacy': 'Anonymous scalar checks and hashes only; no image arrays, raw paths, argv or subject IDs.'}
    resource = root / 'resource_observation.public.json'
    if resource.exists():
        report['resource_observation_sha256'] = sha256(resource)
    args.report_out.parent.mkdir(parents=True, exist_ok=True)
    args.report_out.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'native_complete': True,
                      'seconds': report['whole_workflow_wall_seconds'],
                      'exit_proof': report['exit_proof']}))


if __name__ == '__main__':
    main()
