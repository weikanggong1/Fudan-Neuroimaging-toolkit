"""Summarize immutable real Double-control reports with accurate dtype labels."""
import argparse
import hashlib
import json
from pathlib import Path


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scalar-root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--step4-root',type=Path)
    parser.add_argument('--saved-direction-audit',type=Path)
    args=parser.parse_args();records={}
    for model in ['points64_projection32_gaussian32','points64_projection64_gaussian32','points64_projection64_gaussian64']:
        py_path=args.scalar_root/(model+'-python/summary.public.json')
        na_path=args.scalar_root/(model+'-native/summary.public.json')
        py=json.loads(py_path.read_text());na=json.loads(na_path.read_text())
        if na['python_summary_sha256']!=digest(py_path):raise RuntimeError('actual producer/scorer report binding differs')
        if na['raw_adapter_sha256']!=py['raw_adapter_sha256']:raise RuntimeError('actual adapter binding differs')
        points={}
        for step,row in na['same_points'].items():
            item={'native_minus_CPU_same_point_cost':row['native_minus_raw_cost'],
                  'same_point_full_gradient_relative_l2':row['same_point_projected_full_gradient']['relative_l2'],
                  'raw_priors_max_abs':row['raw_priors_vs_native_float_drawer']['max_abs'],
                  'coverage_different':row['coverage_different'],'original_scientific_gate':row['gate']}
            if 'trajectory_at_distinct_accepted_points' in row:
                trajectory=row['trajectory_at_distinct_accepted_points']
                item['distinct_accepted_points']={'actual_coordinate_dtype':trajectory['points']['dtype'],
                    'coordinate_difference_to_saved_native_Double':trajectory['points'],
                    'native_minus_CPU_cost':trajectory['native_minus_raw_cost'],
                    'CPU_actual_coordinate_deformation':trajectory['raw_actual_FP32_deformation'],
                    'CPU_theoretical_deformation':trajectory['raw_theoretical_deformation'],
                    'native_returned_deformation':trajectory['native_deformation']}
            points[step]=item
        records[model]={'points':points,'all_scientific_gates_passed':na['all_scientific_gates_passed'],
             'actual_CPU_precision_policy':py['actual_private_CPU_precision_policy'],
             'bounded_update_observation_seconds':py['limited_replay_observation_seconds'],
             'frozen_GEMS_source_sha256':py['source_sha256'],
             'producer_sha256':py['program_sha256'],'adapter_sha256':py['raw_adapter_sha256'],
             'original_producer_summary_sha256':digest(py_path),'original_native_summary_sha256':digest(na_path)}
    report={'status':'completed_three_real_CPU_Double_bounded3_controls',
        'scope':'Validation-only private frozen CPU closure; no production/GPU defaults, EM, new native optimizer updates, recipe or final segmentation',
        'controls':records,'original_reports_preserved_without_overwrite':True,
        'metadata_errata':{
          'native_points_rounded_to_FP32_exact':'Legacy scorer compares Double candidate coordinates to intentionally rounded native FP32 coordinates. False does not assess Double coordinate match. Use explicit Double coordinate error here.',
          'raw_actual_FP32_deformation':'Legacy label only. Value is trace actual displacement of the real Double stored points, renamed CPU_actual_coordinate_deformation here.',
          'same_image_alphas_reference_canmove_epsilon_projection_source':'Frozen function source and captured input values are shared. Projection64 recomputes QR in Double; this legacy flag does not prove identical projection matrices.',
          'raw_initial.intentional_change':'Legacy producer mentions raw mass only. These controls also deliberately change point/leaf-gradient storage, projection computation and/or Gaussian likelihood according to actual_CPU_precision_policy.'},
        'scorer_tolerances_unchanged':{'absolute_cost':.01,'gradient_relative_l2':1e-5,'raw_prior_max_abs':1e-6,'coverage':'exact'},
        'contracts_summary_sha256':digest(args.scalar_root/'contracts.public.json'),
        'program_sha256':digest(__file__)}
    if args.step4_root:
        if args.saved_direction_audit is None:raise ValueError('step4 alpha comparison requires saved-state audit provenance')
        saved=json.loads(args.saved_direction_audit.read_text())
        py_path=args.step4_root/'python_summary.public.json';na_path=args.step4_root/'native_summary.public.json'
        py=json.loads(py_path.read_text());na=json.loads(na_path.read_text());row=py['modes']['native_definition_cpu']['rows'][-1]
        if saved['capture_sha256']!=na['capture_sha256'] or saved['native_binary_sha256']!=na['native_binding_sha256']:
            raise RuntimeError('saved direction audit uses different native/input')
        gate=py['resume_state_provenance']['resume_cost_gradient_gate'];item=na['same_points']['4']
        report['one_additional_step4']={'added_steps':py['added_steps'],'actual_CPU_precision_policy':py['actual_private_CPU_precision_policy'],
            'resume_cost_exact':gate['cost_exact'],'resume_gradient_exact':gate['gradient_exact'],
            'actual_private_alpha':row['alpha'],'native_alpha_inferred_in_previous_saved_state_audit':saved['rows'][3]['native_alpha_inferred_from_deformation'],
            'saved_direction_audit_sha256':digest(args.saved_direction_audit),
            'same_point_scientific_gate':item['gate'],'same_point_gradient_relative_l2':item['same_point_projected_full_gradient']['relative_l2'],
            'native_minus_CPU_same_point_cost':item['native_minus_raw_cost'],'coverage_different':item['coverage_different'],
            'raw_prior_max_abs':item['raw_priors_vs_native_float_drawer']['max_abs'],
            'Double_accepted_coordinate_error':item['trajectory_at_distinct_accepted_points']['points'],
            'one_update_observation_seconds':py['limited_replay_observation_seconds'],
            'producer_summary_sha256':digest(py_path),'native_summary_sha256':digest(na_path)}
    args.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'status':report['status'],'all_controls_gates_passed':all(x['all_scientific_gates_passed'] for x in records.values())}))


if __name__=='__main__':main()
