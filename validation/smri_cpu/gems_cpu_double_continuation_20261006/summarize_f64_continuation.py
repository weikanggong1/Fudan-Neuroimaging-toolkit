"""Corrected scalar aggregate for finite CPU Double accepted4→37 continuation."""
import argparse
import hashlib
import json
from pathlib import Path


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text())


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ['scalars','prefix3','prefix4','path-audit','direction-audit','output']:
        parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args();p=read(args.scalars/'python_summary.public.json');n=read(args.scalars/'native_summary.public.json')
    contracts=read(args.scalars/'portable_contracts.public.json');p3=read(args.prefix3);p4=read(args.prefix4)
    path=read(args.path_audit);directions=read(args.direction_audit)
    if (p['added_steps']!=33 or p['modes']['native_definition_cpu']['steps']!=37
            or p['resume_summary_sha256']!=sha(args.prefix4) or p['raw_adapter_sha256']!=p3['raw_adapter_sha256']
            or p['actual_private_CPU_precision_policy']!=p3['actual_private_CPU_precision_policy']
            or p['source_sha256']!=p3['source_sha256'] or p['source_sha256']!=p4['source_sha256']
            or n['python_summary_sha256']!=sha(args.scalars/'python_summary.public.json')
            or n['native_optimizer_calls']!=0 or not n['all_scientific_gates_passed']
            or not contracts['all_passed']):raise RuntimeError('finite continuation/score/source binding failed')
    for audit in [path,directions]:
        if (audit['continuation_summary_sha256']!=sha(args.scalars/'python_summary.public.json')
                or audit['prefix3_summary_sha256']!=sha(args.prefix3)
                or audit['prefix4_summary_sha256']!=sha(args.prefix4)
                or audit['native_summary_sha256']!=n['reused_actual_native37_summary_sha256']):
            raise RuntimeError('read-only saved trajectory binding failed')
    same=n['same_points']['37'];trajectory=same['trajectory_at_distinct_accepted_points'];resume=p['resume_state_provenance']
    report={'scope':'one real public T1-derived synthetic-label recipe stage; fixed likelihood and smoothed alphas; no EM/full recipe or production adoption',
        'synthetic_stage_meaning':'Recipe maps coarse segmentation labels derived from the real T1 into its working label image, then resamples/crops/masks that image and uses fixed Gaussians. It is not a simulated-MRI benchmark.',
        'actual_CPU_precision_policy':p['actual_private_CPU_precision_policy'],
        'bound_frozen_source_sha256':p['source_sha256'],
        'production_difference':'This frozen diagnostic additionally depends on CPU epsilon, Double reference/interpolation, separate FP32 ownership, raw atlas mass, actual Double QR/Gaussian/point-gradient/history and private native-definition L-BFGS. It does not validate a one-line normalization edit in current production.',
        'portable_serialization_contracts':contracts['contracts'],
        'resume_accepted4':resume,
        'limited_updates':{'reused_prior_steps':4,'new_steps':p['added_steps'],'total_saved_steps':37,
            'stop_reason':p['modes']['native_definition_cpu']['stop_reason'],
            'reason_counts':{reason:sum(r['reason']==reason for r in p['modes']['native_definition_cpu']['rows']) for reason in sorted({r['reason'] for r in p['modes']['native_definition_cpu']['rows']})},
            'algorithmic_convergence_rows':sum(r['algorithmic_convergence'] for r in p['modes']['native_definition_cpu']['rows']),
            'diagnostic_budget_exhausted_rows':sum(r['reason']=='diagnostic_budget_exhausted' for r in p['modes']['native_definition_cpu']['rows']),
            'maximum_saved_trial_count_per_step':max(len(r['records']) for r in p['modes']['native_definition_cpu']['rows']),
            'saved_nonfinite_trial_cost_count':sum(t['cost'] is None for r in p['modes']['native_definition_cpu']['rows'] for t in r['records']),
            'native_optimizer_calls':0,'new_EM_or_full_recipe':False},
        'same_point37':{'native_cost':same['native_cost'],'CPU_cost':same['raw_cpu_cost'],
            'native_minus_CPU_cost':same['native_minus_raw_cost'],'projected_full_gradient':same['same_point_projected_full_gradient'],
            'raw_priors':same['raw_priors_vs_native_float_drawer'],'coverage_different':same['coverage_different'],'gate':same['gate'],
            'tolerances':n['tolerances'],'owner_cell_IDs_directly_compared':False},
        'different_accepted_geometry37':{'actual_Double_coordinate_difference':trajectory['points'],
            'native_minus_CPU_cost':trajectory['native_minus_raw_cost'],
            'CPU_actual_coordinate_deformation':trajectory['raw_actual_FP32_deformation'],
            'CPU_actual_coordinate_dtype':'torch.float64','native_returned_maximal_deformation':trajectory['native_deformation'],
            'Jacobian':same['Jacobian_at_distinct_accepted_points'],'trajectory_equivalence_accepted':False},
        'saved_path_audit':{'earliest_alpha_difference_above_1e-8':path['earliest_alpha_difference_above_1e-8'],
            'maximum_coordinate_error_over37_saved_steps':path['maximum_coordinate_difference_over_saved_steps'],
            'step4':directions['rows'][3],'step5':directions['rows'][4],'step6':directions['rows'][5],
            'native_alpha_and_history_are_inferences_not_internal_trace':True,
            'native_increment_reconstruction_max_residual_reused':max(r['native_prior_source_direction_increment_residual_reused']['max_abs'] for r in directions['rows']),
            'no_new_objective_or_optimizer_call_for_read_only_audits':True},
        'observed_timing_seconds':{'first3':p3['limited_replay_observation_seconds'],
            'one_update3_to4':p4['limited_replay_observation_seconds'],'33_updates4_to37':p['limited_replay_observation_seconds'],
            'sum_of_three_individual_report_clocks':sum(r['limited_replay_observation_seconds'] for r in [p3,p4,p]),
            'continuation_report_declared_sum_with_earlier_segments':p['sum_segment_observation_seconds_excluding_resume_import_gate_pause'],
            'accepted4_restore_cost_gradient_gate_and_cold_cache':resume['resume_cost_gradient_gate']['observation_seconds'],
            'accepted3_restore_gate_and_cold_JIT':p4['resume_state_provenance']['resume_cost_gradient_gate']['observation_seconds'],
            'not_a_fresh_uninterrupted_benchmark':True,'not_an_ABBA_speed_acceptance':True},
        'legacy_field_errata':{'raw_actual_FP32_deformation':'In this frozen scorer this is actual Double-coordinate displacement; the corrected name here is CPU_actual_coordinate_deformation.',
            'native_points_rounded_to_FP32_exact':'Intentional native FP32 projection comparison is not a test of Double trajectory; compare actual Double error.',
            'same_image_alphas_reference_canmove_epsilon_projection_source':'Same source/input does not mean the actual Double QR matrix equals the previous FP32 projection.',
            'raw_initial.intentional_change':'Legacy raw-mass label omits Double points/gradient/QR/Gaussian controls; actual policy is stated above.'},
        'bindings':{'program_sha256':sha(__file__),'Python_summary_sha256':sha(args.scalars/'python_summary.public.json'),
            'native_same_point_summary_sha256':sha(args.scalars/'native_summary.public.json'),
            'portable_contract_summary_sha256':sha(args.scalars/'portable_contracts.public.json'),
            'prefix3_summary_sha256':sha(args.prefix3),'prefix4_summary_sha256':sha(args.prefix4),
            'read_only_path_summary_sha256':sha(args.path_audit),'read_only_direction_summary_sha256':sha(args.direction_audit),
            'native37_saved_summary_sha256':n['reused_actual_native37_summary_sha256'],
            'raw_adapter_sha256':p['raw_adapter_sha256'],'optimizer_candidate_sha256':p['candidate_sha256'],
            'continuation_runner_sha256':p['program_sha256'],'portable_helper_sha256':p['resume_helper_sha256'],
            'native_scorer_sha256':n['program_sha256'],'capture_sha256':p['capture_sha256'],'native_binary_sha256':n['native_binding_sha256']}}
    args.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'same_point37_pass':all(same['gate'].values()),'trajectory_equivalence_accepted':False,'added_steps':p['added_steps']}))


if __name__=='__main__':main()
