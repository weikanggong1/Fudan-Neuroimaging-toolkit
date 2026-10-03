"""Compare complete CIFTI outputs of two separately executed original protocols.

This is postprocessing only. It does not run volume, registration or projection,
fit intensity scales or add smoothing. Paths and arrays stay private; the public
report contains scalar errors, axis checks and original execution/file hashes.
"""
import argparse
import json
from pathlib import Path

from compare_surface_e2e import read_json, sha256, support_modules, temporal_cifti


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ('matched-manifest', 'full-manifest', 'matched-execution-report',
                  'full-execution-report', 'report-out'):
        parser.add_argument('--' + field, type=Path, required=True)
    args = parser.parse_args()
    matched, full = read_json(args.matched_manifest), read_json(args.full_manifest)
    matched_execution, full_execution = (read_json(path) for path in
        (args.matched_execution_report, args.full_execution_report))
    if (matched_execution.get('original_scientific_worker_exit_code') != 0
            or matched_execution.get('validation_complete') is not True
            or full_execution.get('exit_code') != 0
            or full_execution.get('validation_complete') is not True):
        raise ValueError('both original scientific executions must have completed with full QC')
    tr, frames = full_execution['input_repetition_time_seconds'], full_execution['input_frames']
    projection, _, _ = support_modules()
    result = projection.compare_cifti(matched['dtseries'], full['dtseries'], tr, frames)
    result['temporal'] = temporal_cifti(matched['dtseries'], full['dtseries'])
    report = {
        'schema_version': 1, 'validation_complete': True,
        'scope': 'Two original software protocols: fresh matched native FSL volume with independent '
                 'official FS/newMSM surface versus continuously executed full fMRIPrep25.2.4 with '
                 'official newMSM. No FNIT imaging output is included; no intensity fit, smoothing '
                 'or time selection is applied. This output comparison has no new pipeline wall time. '
                 'Raw-input identity is bound separately by the end-to-end execution evidence.',
        'comparison_script_sha256': sha256(__file__),
        'validation_support_sha256': {'run_projection_candidate.py': sha256(projection.__file__),
                                     'compare_surface_e2e.py': sha256(Path(__file__).with_name('compare_surface_e2e.py'))},
        'matched_original_execution_report_sha256': sha256(args.matched_execution_report),
        'full_original_execution_report_sha256': sha256(args.full_execution_report),
        'matched_original_cifti_sha256': sha256(matched['dtseries']),
        'full_original_cifti_sha256': sha256(full['dtseries']),
        'frames': frames, 'tr_seconds': tr, 'output': result,
    }
    args.report_out.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'validation_complete': True, 'mean_temporal_r': result['temporal']['mean_temporal_r'],
                      'rmse': result['rmse'], 'relative_rmse': result['relative_rmse']}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
