"""自产全局先验及有序ETable→候选输入，保存后核对固定首缺陷参考。"""
from __future__ import annotations
import time
_ENTRY=time.perf_counter()
import argparse,hashlib,inspect,json,os,sys
from pathlib import Path
import nibabel as nib
import nibabel.freesurfer.io as fsio
import numpy as np
from fnit.recon_all.topology_candidate_inputs import prepare_candidate_input_context, prepare_defect_candidate_inputs
from fnit.recon_all.topology_edge_mri_score import source_topology_ras_to_voxel
from fnit.recon_all.thread_budget import thread_budget
_IMPORT=time.perf_counter()-_ENTRY


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def global_priors_from_checkpoint(values):
    """自产producer的既定origin→spacing schema；不读取参考PDF或mean。"""
    geometry=values['k_pdf_geometry']
    return dict(curvature=dict(k1_bins=values['k1_bins'],k2_bins=values['k2_bins'],
        k1_counts=values['k1_counts'],k2_counts=values['k2_counts'],
        k1_start=geometry[0],k2_start=geometry[1],k1_width=geometry[2],k2_width=geometry[3],
        joint=values['k_pdf'][:,:,0]),normal_dot=dict(bins=values['dot_bins'],
        counts=values['dot_counts'],bin_size=values['dot_bin_size'][0]))


def exported_fields(context,defect):
    """完整source-consumed静态kwargs及自产校准/局部场；没有网格after输入。"""
    kwargs=defect['candidate_kwargs'];local=kwargs['local_priors'];cal=defect['calibration']
    result={k:v for k,v in local.items() if isinstance(v,np.ndarray)}
    result.update({name+'_calibration':value for name,value in cal.items() if isinstance(value,np.ndarray)})
    result.update(gray_white_pdf=kwargs['gray_white_pdf'],joint_curvature_pdf=kwargs['joint_curvature_pdf'],
        joint_origin=kwargs['joint_origin_mm_inv'],joint_spacing=kwargs['joint_spacing_mm_inv'],
        matching_means=np.asarray([local['white_mean'],local['gray_mean'],kwargs['k1_mean'],kwargs['k2_mean']],np.float32),
        source_weights=np.asarray([kwargs[name] for name in ('mri_weight','volume_unmri_weight','normal_dot_weight','quadratic_curvature_weight')],np.float64))
    for name in ('k1_histogram','k2_histogram','normal_dot_histogram'):
        for field in ('bins','counts','bin_size'):
            result[name+'_'+field]=np.asarray(kwargs[name][field])
    return result


def run(*,input_surface,brain,global_checkpoint,global_report,etable_checkpoint,etable_report,
        creation_reference,likelihood_reference,statistics_reference,match_reference,
        source_manifest,output,threads):
    if output.exists():raise FileExistsError('new empty stage output required')
    started=time.perf_counter()
    prior=json.loads(global_report.read_text());table_receipt=json.loads(etable_report.read_text())
    if (prior['status']!='complete_global_priors' or prior['candidate_checkpoint_sha256']!=sha(global_checkpoint)
            or table_receipt['candidate_checkpoint_sha256']['cold']!=sha(etable_checkpoint)
            or not table_receipt['all_candidate_outputs_computed_without_reference']
            or not table_receipt['strict_native_etable_and_retess_inputs']):
        raise ValueError('bound own complete global priors and same-input ordered ETable required')
    with np.load(global_checkpoint,allow_pickle=False) as z:global_fields={k:z[k].copy() for k in z.files}
    with np.load(etable_checkpoint,allow_pickle=False) as z:table={k:z[k].copy() for k in z.files}
    raw,faces=fsio.read_geometry(str(input_surface));raw=np.asarray(raw,np.float32);faces=np.asarray(faces,np.int32)
    if sha(input_surface)!=prior['input_sha256']['nofix']:
        raise ValueError('same own nofix input SHA required')
    image=nib.load(brain);volume=np.asarray(image.dataobj);matrix=source_topology_ras_to_voxel(volume_header=image.header)
    if sha(brain)!=table_receipt['input_sha256']['brain']:
        raise ValueError('same own MRI SHA as ETable producer required')
    order=table['source_order'];translation=np.argsort(order).astype(np.int32)
    hull=order[table['hull_vertices']].astype(np.int32)
    load_seconds=time.perf_counter()-started
    with thread_budget(threads=threads):
        context=prepare_candidate_input_context(vertices=raw,faces=faces,
            defect_vertices=global_fields['defect_labels']!=0,vertex_translation=translation,
            volume=volume,ras_to_voxel=matrix,global_priors=global_priors_from_checkpoint(global_fields))
        # This benchmark explicitly covers the first defect before any final
        # corrected geometry changes. The public API always requires live xyz.
        current=context.preparation['original_vertices'][order].copy()
        args=dict(context=context,original_defect_hull=hull,candidate_edges=table['edge_vertices'],
            current_corrected_original_vertices=current)
        first=prepare_defect_candidate_inputs(**args);second=prepare_defect_candidate_inputs(**args)
    from fnit.recon_all.topology_candidate_pipeline import evaluate_candidate
    consumer_bound=inspect.signature(evaluate_candidate).bind(
        surface_state={},retessellation_inputs={},**first['candidate_kwargs'])
    shared_mri=(consumer_bound.arguments['volume'] is context.volume
                and consumer_bound.arguments['ras_to_voxel'] is context.ras_to_voxel
                and not context.volume.flags.writeable
                and not context.ras_to_voxel.flags.writeable)
    if not shared_mri:raise ValueError('complete read-only original MRI consumer inputs required')
    fields=exported_fields(context,first);repeat_fields=exported_fields(context,second)
    output.mkdir(parents=True)
    np.savez_compressed(output/'candidate.private.npz',**fields,current_corrected_original=current,
        original_white=context.preparation['white_values'],original_gray=context.preparation['gray_values'],
        original_hull=hull,corrected_edges=table['edge_vertices'],source_matrix=matrix)
    before_reference_seconds=time.perf_counter()-started
    from benchmark_recon_topology_volume_prepare_fields import load_volume_preparation_capture
    from benchmark_recon_topology_mri_likelihood import load_statistics_capture
    from benchmark_recon_topology_match_fields import load_match_capture
    from benchmark_recon_topology_volume_priors import difference
    before=load_volume_preparation_capture(capture_path=creation_reference)
    after=load_volume_preparation_capture(capture_path=likelihood_reference)
    stat=load_statistics_capture(capture_path=statistics_reference)
    match=load_match_capture(capture_path=match_reference)
    comparisons=dict(corrected_current=difference(current,before['original_vertices']),
        full_ordered_ETable=difference(table['edge_vertices'],before['potential_edges']),
        white_calibration=difference(first['calibration']['white_smooth'],before['white_pdf']),
        gray_calibration=difference(first['calibration']['gray_smooth'],before['gray_pdf']),
        white_bins=difference(first['calibration']['bins'],before['white_bins']),
        gray_bins=difference(first['calibration']['bins'],before['gray_bins']),
        white_gray_joint_PDF=difference(fields['gray_white_pdf'],stat['gray_white_pdf']),
        joint_curvature_PDF=difference(fields['joint_curvature_pdf'],stat['k_pdf']),
        joint_origin_spacing=difference(np.r_[fields['joint_origin'],fields['joint_spacing']],stat['k_pdf_geometry']),
        matching_means=difference(fields['matching_means'],match['parameters']))
    for name,reference_name in [('defect_intensity_volume','intensity'),('white_log_probability','white_log_pdf'),('gray_log_probability','gray_log_pdf')]:
        comparisons[name]=difference(fields[name],after[reference_name])
    comparisons['initial_edge_mask']=difference(fields['initial_edge_mask'],before['intensity'])
    comparisons['algorithm_volume_geometry']=difference(fields['algorithm_volume_geometry'],before['algorithm_volume_geometry'])
    for name,source_prefix in [('k1_histogram','k1'),('k2_histogram','k2'),('normal_dot_histogram','dot')]:
        for field,source_field in [('bins','bins'),('counts','counts'),('bin_size','bin_size')]:
            comparisons[name+'_'+field]=difference(fields[name+'_'+field].reshape(-1),stat[source_prefix+'_'+source_field].reshape(-1))
    matrix_comparison=difference(matrix.astype(np.float64),before['ras_to_voxel'])
    repeated={key:difference(value,repeat_fields[key]) for key,value in fields.items()}
    exact=all(x['bitwise_equal'] for x in comparisons.values());repeat_exact=all(x['bitwise_equal'] for x in repeated.values())
    modules=['topology_candidate_inputs','topology_candidate_pipeline','topology_edge_mri_score','topology_intensity_histogram','topology_curvature_histogram','topology_mri_likelihood','topology_defect_volume_priors','thread_budget']
    report=dict(schema_version=1,status='complete_scoring_inputs' if exact and repeat_exact else 'failed_scoring_inputs',
        public_case='ds000114 sub-07 ses-test LH first defect only',
        nvertices=len(raw),nfaces=len(faces),candidate_edges=len(table['edge_vertices']),
        comparisons=comparisons,source_consumed_outputs_bitwise_equal=exact,
        own_per_defect_repeat=repeated,own_per_defect_repeat_bitwise_equal=repeat_exact,
        internal_matrix_diagnostic=matrix_comparison,
        all_internal_bitwise_reproduction=exact and matrix_comparison['bitwise_equal'],
        input_sha256={k:sha(p) for k,p in dict(nofix=input_surface,brain=brain,global_checkpoint=global_checkpoint,
            global_report=global_report,ETable_checkpoint=etable_checkpoint,ETable_report=etable_report).items()},
        reference_sha256={k:sha(p) for k,p in dict(creation=creation_reference,likelihood=likelihood_reference,
            statistics=statistics_reference,matching=match_reference).items()},
        actual_module_sha256={name:sha(sys.modules['fnit.recon_all.'+name].__file__) for name in modules},
        source_manifest_sha256=sha(source_manifest),script_sha256=sha(__file__),
        own_checkpoint_sha256=sha(output/'candidate.private.npz'),
        timing=dict(context_api=context.seconds,first_per_defect=first['stage_seconds'],repeat_per_defect=second['stage_seconds'],
            loading=load_seconds,own_IO_before_reference=before_reference_seconds,
            complete_driver_including_IO_comparison=time.perf_counter()-started,import_seconds=_IMPORT),
        cpu_affinity=sorted(os.sched_getaffinity(0)),threads=threads,GPU_used=False,GPU_memory_bytes=None,
        original_calibration_and_current_corrected_geometry_separate=True,
        consumer_required_argument_binding=True,consumer_mri_shared_readonly=shared_mri,
        first_defect_current_geometry_generated_from_own_initial_base_preparation=True,
        shared_original_preparation_calls=1,independent_per_defect_calls=2,
        reference_after_candidate_saved=True,reference_outputs_production_inputs=False,
        full_raw_ETable_region_order_producer='not_assessed; explicit stage input producer provenance retained',
        multiple_defect_live_state='contracts only',full_GA='not_assessed',whole_recon_all='not_assessed',
        overall_metric_equivalence='not_assessed',production_default_changed=False)
    dest=output/'report.json';dest.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    (output/'COMPLETE.json').write_text(json.dumps(dict(status=report['status'],report_sha256=sha(dest)))+'\n')
    print(json.dumps(dict(status=report['status'],output_strict=exact,all_internal_strict=report['all_internal_bitwise_reproduction'])))
    return 0 if exact and repeat_exact else 1


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('input-surface','brain','global-checkpoint','global-report','etable-checkpoint','etable-report',
                 'creation-reference','likelihood-reference','statistics-reference','match-reference','source-manifest','output'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--threads',type=int,default=4)
    raise SystemExit(run(**vars(p.parse_args())))
