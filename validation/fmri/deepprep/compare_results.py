"""由两套真实运行报告生成计时对照；不同处理范围不计算加速比。"""
import csv
import hashlib
import json
from pathlib import Path


def main():
    root = Path(__file__).resolve().parent
    paths = [root.parent / 'fmri_volume.public.json',
             root.parent / 'fmri_surface.public.json', root / 'benchmark.public.json']
    volume, surface, deepprep = [json.loads(p.read_text(encoding='utf-8')) for p in paths]
    inputs = {p['file'].split('_')[-1]: p for p in deepprep['input']['inputs']}
    bold = inputs['bold.nii.gz']
    t1w = inputs['T1w.nii.gz']
    if volume['input_sha256']['bold'] != surface['input_sha256']['bold'] or volume['input_sha256']['bold'] != bold['sha256']:
        raise ValueError('BOLD inputs differ')
    if volume['data']['bold_shape'] != [88, 88, 64, 490] or surface['checks']['cifti_shape'] != [490, 91282]:
        raise ValueError('Unexpected FNIT frame count or output geometry')
    rows = []
    for report, space in [(volume, 'MNI152NLin6Asym 2 mm'), (surface, 'fsLR32k + 91k CIFTI')]:
        rows.append(dict(method='FNIT', mode=report['stage'], output_space=space,
                         wall_seconds=report['timing_seconds']['public_api_including_output_save'],
                         timing_scope='Public API, including output save; excludes imports and validation',
                         includes_anatomical_reconstruction=False,
                         starts_from_completed_volume=report['stage'] == 'surface',
                         source_revision=report['source_revision']))
    for run in deepprep['runs']:
        if run['status'] != 'complete' or run['exit_code'] != 0 or not all(p.get('finite', True) for p in run['validated_outputs']):
            raise ValueError('DeepPrep did not complete and pass output checks')
        rows.append(dict(method='DeepPrep', mode=run['mode'],
                         output_space='MNI152NLin6Asym 2 mm' if run['mode'] == 'volume' else 'fsaverage6',
                         wall_seconds=run['pipeline_wall_seconds'],
                         timing_scope='Container launch through pipeline exit, including reconstruction and QC',
                         includes_anatomical_reconstruction=True, starts_from_completed_volume=False,
                         source_revision=deepprep['software']['upstream_source_commit']))
    result = dict(schema_version=1, benchmark_date='2026-09-30',
                  source_report_sha256={p.relative_to(root.parent).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
                  same_bold_file=True, bold_sha256=bold['sha256'], frames=490,
                  same_t1w_file=volume['input_sha256']['t1w'] == t1w['sha256'],
                  matched_end_to_end_comparison=False, rows=rows,
                  fnit_volume_plus_surface_seconds=sum(row['wall_seconds'] for row in rows if row['method'] == 'FNIT'),
                  numerical_agreement_between_methods=None,
                  limits=['T1w inputs differ: archive reconstruction input versus full original T1w.',
                          'FNIT uses SBRef; DeepPrep has no SBRef input.',
                          'FNIT surface starts from completed volume and existing recon-all geometry; DeepPrep starts from raw inputs.',
                          'Surface spaces differ: fsLR32k/91k versus fsaverage6.',
                          'FNIT includes ICA-AROMA and confound regression; DeepPrep exports preprocessed BOLD and confounds.',
                          'Shared H100, different CPU quotas, timing and memory definitions; one observation per mode.',
                          'FNIT timings are frozen to the recorded source revision, not later main revisions.'])
    (root / 'comparison.public.json').write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    with (root / 'timings.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)
    print('Verified same 490-frame BOLD; wrote four scoped observations. No speed ratio calculated.')


if __name__ == '__main__':
    main()
