"""Build a path-redacted FastVBM report from collected private run metadata.

No MRI, weights, templates, credentials, or private commands are published.
This reads completed results; it never invokes FNIT or the original software.
"""
import argparse
import hashlib
import json
from pathlib import Path


def redact(value):
    """Keep parameter names and file basenames while removing server locations."""
    if isinstance(value, dict):
        return {redact(key): redact(item) for key, item in value.items()}
    if isinstance(value, list):
        return [redact(item) for item in value]
    if isinstance(value, str):
        if value.startswith('/'):
            return Path(value).name
        option, separator, argument = value.partition('=')
        if separator and argument.startswith('/'):
            return option + separator + Path(argument).name
    return value


def queue_summary(record):
    fields = ('job_sha256', 'hostname', 'max_cpu_threads', 'cpu_affinity',
              'status', 'started_utc', 'finished_utc', 'wall_seconds', 'returncode',
              'maximum_sampled_tree_rss_bytes', 'maximum_sampled_tree_threads',
              'load_before', 'load_after')
    result = {key: record[key] for key in fields if key in record}
    result['job_id'] = record['job']['id']
    samples = record.get('resource_samples', 0)
    result['resource_samples_count'] = len(samples) if isinstance(samples, list) else samples
    result['job_command_basenames_only'] = redact(record['job']['argv'])
    result['thread_environment'] = {key: item for key, item in record['environment'].items()
                                    if key != 'PYTHONPATH'}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--collected', required=True, type=Path,
                        help='private JSON containing completed queue, step, score, and source-hash records')
    parser.add_argument('--output', required=True, type=Path,
                        help='public JSON destination; private paths are replaced by basenames')
    args = parser.parse_args()
    collected = json.loads(args.collected.read_text())
    official_fnirt = collected['task4_vbm_official_cpu_v2/official_synthstrip_fast_fnirt/reference_steps.private.json']
    morph_native = collected['task4_vbm_morph_official_cpu_v1/artifacts/reference_steps.private.json']
    morph_final = collected['task4_vbm_morph_official_cpu_v2/artifacts/reference_steps.private.json']
    upstream_seconds = sum(row['wall_seconds'] for row in official_fnirt['stages'][:3])
    native_morph_seconds = next(row['wall_seconds'] for row in morph_native['stages']
                               if row['name'] == 'synthmorph')
    final_postprocessing_seconds = sum(row.get('wall_seconds', 0) for row in morph_final['stages'])
    comparison_keys = {
        'fnirt_vs_original_fnirt': 'task4_vbm_scores_v1/fnirt_vs_official_fnirt.private.json',
        'synthmorph_vs_original_synthmorph': 'task4_vbm_scores_v3/synthmorph_vs_official_synthmorph.private.json',
        'synthmorph_vs_original_fnirt_different_estimators': 'task4_vbm_scores_v1/synthmorph_vs_official_fnirt.private.json',
    }
    queue_keys = {
        'candidate_fnirt_complete_cli': 'task4_vbm_cpu_queue_v1/candidate_fnirt/record.json',
        'candidate_synthmorph_complete_cli': 'task4_vbm_cpu_queue_v1/candidate_synthmorph/record.json',
        'original_fnirt_complete_chain': 'task4_vbm_official_cpu_v2_queue/official_synthstrip_fast_fnirt_cpu8/record.json',
        'original_synthmorph_branch_upstream_reused': 'task4_vbm_morph_official_cpu_v1_queue/official_synthmorph_reuse_upstream/record.json',
        'original_synthmorph_postprocessing_replay_only': 'task4_vbm_morph_official_cpu_v2_queue/synthmorph_source_grid_postprocessing_replay/record.json',
        'jacobian_definition_diagnostic': 'task4_vbm_jacobian_diagnostic_v1_queue/same_original_warp_jacobian_definition/record.json',
        'independent_coordinate_validation': 'task4_vbm_conversion_validate_v2_queue/independent_source_coordinate_check/record.json',
        'retained_reference_startup_failure': 'task4_vbm_cpu_queue_v1/official_synthstrip_fast_fnirt/record.json',
        'retained_old_coordinate_validator_failure': 'task4_vbm_conversion_validate_v1_queue/independent_source_coordinate_check/record.json',
    }
    report = {
        'schema_version': 1,
        'date': '2026-10-04',
        'scope': 'single real raw T1, thirteen maps, same CPU affinity and configured eight-thread budget; CPU observations on a shared node',
        'canonical_index_live_updated_utc': '2026-10-04T02:16:12.096774+00:00',
        'protocol': {
            'host': 'nodecw10', 'cpu': 'Intel Xeon Gold 6418H',
            'baseline_commit': '1d31e7baaebbb644ab199471f7fe6282721455fd',
            'candidate_frozen_source_namespace': 'task5_candidate_cpu_v2/src',
            'root_final_integrated_source_full_vbm_retested': False,
            'cpu_affinity': [3, 7, 11, 15, 19, 23, 27, 31], 'configured_threads': 8,
            'cuda_visible_devices': '', 'exclusive_node': False,
            'sample_count': 1, 'complete_pipeline_ab_ba_repeats': False,
            'complete_pipeline_order': ['candidate_fnirt', 'candidate_synthmorph', 'original_fnirt', 'original_synthmorph_branch'],
            'io_scope': 'new processes; complete candidate CLI and original FNIRT chain include startup, reads and output writes; no filesystem cache flush',
            'numba_compilation_scope': 'candidate CPU fsl execution loads the previously compiled disk cache; first installation JIT is not separately timed here',
            'reference_versions': {'FSL': '6.0.7.4', 'FreeSurfer': '8.2.0-1'},
            'native_calls_only_in_reference_workers': True,
            'standard_bet_fsl_vbm_pipeline': False,
            'gpu_fast_vbm_full_pipeline_rerun_in_this_subreport': False,
        },
        'input': {
            'dataset': 'OpenNeuro ds003138', 'version': '1.0.1', 'license': 'CC0',
            'name': 'case01_T1w.nii.gz', 'shape': [224, 288, 288],
            'sha256': official_fnirt['input_sha256'],
            'template_shape': [91, 109, 91], 'template_voxels': 902629,
            'template_sha256': official_fnirt['template_sha256'],
            'template_distribution': 'existing private GM template used only for validation; not redistributed',
            'reference_mask_sha256': official_fnirt['reference_mask_sha256'],
            'synthstrip_weights_sha256': official_fnirt['synthstrip_weights_sha256'],
            'synthmorph_deform_weights_sha256': morph_final['weights_sha256'],
            'assets_downloaded_or_redistributed_in_this_subtask': False,
        },
        'queue_measurements': {key: queue_summary(collected[source]) for key, source in queue_keys.items()},
        'candidate_reports': {
            name: redact(collected['task4_vbm_cpu_v1/candidate_' + name + '/fast_vbm_report.json'])
            for name in ('fnirt', 'synthmorph')
        },
        'original_reference_steps': {
            'fnirt': redact(official_fnirt),
            'synthmorph_native_run_with_superseded_rounded_geometry_postprocessing': redact(morph_native),
            'synthmorph_final_postprocessing_replay_actual_nifti_geometry': redact(morph_final),
        },
        'synthmorph_reference_timing': {
            'original_upstream_stages_seconds': upstream_seconds,
            'original_native_synthmorph_seconds': native_morph_seconds,
            'final_prepare_and_postprocessing_stage_seconds': final_postprocessing_seconds,
            'reconstructed_complete_stage_sum_seconds': upstream_seconds + native_morph_seconds + final_postprocessing_seconds,
            'scope': 'sum of separately measured original stages; upstream and native field reused; not a fresh complete CLI wall time and not a formal end-to-end speed ratio',
        },
        'comparisons': {key: collected[source] for key, source in comparison_keys.items()},
        'jacobian_diagnostic': redact(collected['task4_vbm_jacobian_diagnostic_v1/artifacts/report.private.json']),
        'independent_field_conversion': collected['task4_vbm_conversion_validate_v2/report.private.json'],
        'header_findings': {
            'shapes_dtypes_affines_and_sforms_equal_for_all_thirteen': True,
            'native_qform_maximum_absolute_error_mm': 2.9763615183586722e-9,
            'native_nonspatial_pixdim_5_to_7': {'candidate': [0, 0, 0], 'original': [1, 1, 1]},
            'warped_and_modulated_nonspatial_pixdim_4': {'candidate': 1, 'original': 2.4000000953674316},
            'spatial_pixdim_1_to_3_equal': True,
            'all_header_fields_bitwise_equal': False,
        },
        'source_sha256': {
            'candidate_frozen_relative_modules': collected['candidate_source_sha256'],
            'validation_frozen_relative_files': collected['validation_source_sha256'],
        },
        'figure_scope': collected['task4_vbm_figures_v2/figure_scope.json'],
        'conclusions': {
            'fnirt_chain_numerically_equivalent': False,
            'synthmorph_chain_all_thirteen_bitwise_equal': False,
            'synthmorph_three_template_maps_brain_nrmse_below_1e_minus_3': True,
            'candidate_native_brain_mask_seg_and_mixeltype_values_exact': True,
            'candidate_native_pveseg_different_voxels': 2,
            'flirt_whole_source_grid_maximum_world_error_mm': 0.01720926551803969,
            'jacobian_error_sources': ['analytic spline derivatives versus dense finite differences', 'different estimated nonlinear fields'],
            'jacobian_errors_orthogonal_or_additive_rmse': False,
            'pending': ['complete pipeline AB-BA repeats and wider real subject coverage', 'FLIRT/FNIRT spatial equivalence', 'remaining FAST PVE/pveseg differences', 'nonspatial NIfTI header equality'],
        },
        'history': {
            'rounded_original_warp_source_affine_maximum_error_mm': 0.0007610124330312829,
            'postprocessing_correction': 'reuse the same original RAS field; convert with actual moving/fixed NIfTI geometry rather than rounded geometry in the warp extension',
            'retained_startup_failure_reason': 'reference_env.sh was invoked directly although it is not executable; no original algorithm ran',
            'retained_coordinate_validator_failure_reason': 'old harness used nonexistent Surfa Format.disp_vox; corrected to Format.disp_crs without changing original/candidate inference',
        },
    }
    figures = args.output.parent/'figures'
    report['figure_sha256'] = {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                              for path in sorted(figures.glob('*.png'))}
    serialized = json.dumps(report, indent=2, allow_nan=False) + '\n'
    forbidden = ('/cwStorage/', '/public/', '/home/', '/mnt/', 'FS_LICENSE', 'gongwk@')
    if any(token in serialized for token in forbidden):
        raise ValueError('private location or license environment escaped redaction')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(serialized)
    print(json.dumps({'output': str(args.output), 'comparisons': len(comparison_keys),
                      'maps_per_comparison': 13, 'bytes': len(serialized.encode())}))


if __name__ == '__main__':
    main()
