"""只读确认 staged volume 原件来自哪个真实 attempt；相同字节不重标 producer。"""
import argparse
import hashlib
import json
from pathlib import Path


def sha256(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fnit-root', type=Path, required=True)
    args = parser.parse_args()
    root = args.fnit_root
    cohort = root / 'workspaces/fnit_surface_ten_public_20261003'
    run = root / 'runs/fmri_surface_backends_20261003'
    diagnostic = cohort / 'candidate_v2/CON01'
    formal = cohort / 'candidate_v4/CON01'
    intermediate = root / 'workspaces/fnit_surface_backend_demo_freesurfer_20261003/surface_source19/derivatives'
    diagnostic_report_path = diagnostic / 'report/report.public.json'
    formal_report_path = formal / 'report/report.public.json'
    diagnostic_report = json.loads(diagnostic_report_path.read_text())
    formal_report = json.loads(formal_report_path.read_text())
    if diagnostic_report.get('source_revision') != '01de7f30579d79ef6cfad0ea51de713828dc4206' or diagnostic_report.get('status') != 'failed':
        raise ValueError('diagnostic source/attempt no longer matches its saved real failure record')
    if formal_report.get('source_revision') != '1128bc52c7a0233266e5b8a8d7dc0b382994e676' or formal_report.get('status') != 'complete':
        raise ValueError('formal source/attempt is not the actual completed1128 CON01')
    names = sorted(path.relative_to(intermediate).as_posix() for path in intermediate.rglob('*') if path.is_file())
    if len(names) != 7:
        raise ValueError('actual intermediate ready-volume closure is no longer the seven declared files')
    guard = {'diagnostic_report': diagnostic_report_path, 'formal_report': formal_report_path}
    for label, base in (('diagnostic_v2',diagnostic / 'derivatives'), ('original_intermediate',intermediate),
                        ('provided_directory_cpu',run/'provided-dir-cpu-source1128-prepared-v2/derivatives'),
                        ('provided_zip_cpu',run/'provided-zip-cpu-source1128-prepared-v2/derivatives'),
                        ('formal_v4',formal / 'derivatives')):
        for name in names:
            guard[label + '/' + name] = base / name
    before = {name: sha256(path) for name,path in guard.items()}
    data_names = [name for name in names if name != 'dataset_description.json']
    for name in data_names:
        expected = before['diagnostic_v2/' + name]
        if any(before[label+'/' + name] != expected for label in ('original_intermediate','provided_directory_cpu','provided_zip_cpu')):
            raise ValueError('provided staged MRI/sidecar bytes do not match the original diagnostic v2 producer')
    metadata_fields = {}
    for label, base in (('diagnostic_v2',diagnostic/'derivatives'), ('original_intermediate',intermediate),
                        ('provided_directory_cpu',run/'provided-dir-cpu-source1128-prepared-v2/derivatives'),
                        ('provided_zip_cpu',run/'provided-zip-cpu-source1128-prepared-v2/derivatives')):
        description = json.loads((base/'dataset_description.json').read_text())
        metadata_fields[label] = {'DatasetType': description['DatasetType'], 'raw_link_form': 'relative' if not Path(description['DatasetLinks']['raw']).is_absolute() else 'absolute'}
    after = {name: sha256(path) for name,path in guard.items()}
    if before != after:
        raise ValueError('actual producer/consumer inputs changed during readonly lineage verification')
    result = {'status': 'passed', 'scope': 'Readonly actual-file SHA lineage check. Provided-directory and provided-ZIP full CPU surface reuse diagnostic-v2 completed volume from source01de7f30; that original whole surface attempt failed later. Formal source1128 remains separately identified, even where MRI bytes happen to match.',
        'diagnostic_volume_producer_revision': diagnostic_report['source_revision'],
        'diagnostic_original_whole_attempt_status': diagnostic_report['status'],
        'formal_producer_revision': formal_report['source_revision'], 'formal_whole_attempt_status': formal_report['status'],
        'MRI_and_volume_sidecar_sha256_from_diagnostic_v2': {name:before['diagnostic_v2/'+name] for name in data_names},
        'intermediate_and_both_CPU_copies_MRI_sidecars_exact': True,
        'formal_v4_MRI_file_bytes_equal_to_diagnostic_v2': {name:before['formal_v4/'+name]==before['diagnostic_v2/'+name] for name in data_names if name.endswith('.nii.gz')},
        'owned_dataset_description_scope': 'Only DatasetLinks.raw is relocated to each owned derivative root; original dataset descriptions remain unchanged.',
        'dataset_description_summary': metadata_fields, 'input_sha256_before': before, 'input_sha256_after': after,
        'input_guards_equal': True, 'lineage_validator_sha256': sha256(__file__)}
    with (run/'backend_volume_lineage1128.public.json').open('x') as stream:
        stream.write(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print('ACTUAL_VOLUME_LINEAGE_PASSED',flush=True)


if __name__ == '__main__':
    main()
