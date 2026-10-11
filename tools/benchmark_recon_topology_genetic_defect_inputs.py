"""冻结自产准备字段+显式区域→完整边输入；候选保存后再读独立参考。"""
from __future__ import annotations
import argparse,hashlib,json,os,time,sys
from pathlib import Path
import nibabel as nib
import numpy as np
from fnit.recon_all.topology_genetic_defect_inputs import (ordered_genetic_defect_inputs,
    prepare_ordered_genetic_defect_context)
from fnit.recon_all.thread_budget import thread_budget
from benchmark_recon_topology_retess_fields import load_retessellation_capture

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def difference(a,b):
    a,b=np.asarray(a),np.asarray(b)
    if a.shape!=b.shape or a.dtype!=b.dtype:
        return dict(shape_equal=a.shape==b.shape,dtype_equal=a.dtype==b.dtype,byte_equal=False)
    d=np.abs(a.astype(np.float64)-b.astype(np.float64))
    return dict(shape_equal=True,dtype_equal=True,byte_equal=a.tobytes()==b.tobytes(),
        different_elements=int(np.count_nonzero(a!=b)),maximum_absolute_error=float(d.max(initial=0)),
        p99_absolute_error=float(np.percentile(d,99)) if d.size else 0.)

def arrays(result):
    keys=['edge_vertices','edge_used','edge_scores','ordering','source_edge_indices',
        'candidate_source_vertices','inside_vertices','inside_discarded','inside_status_source','border_vertices','hull_vertices',
        'old_edge_vertices','old_edge_table_indices','old_edge_segments','source_order']
    values={k:result[k] for k in keys}
    values['segment_offsets']=np.r_[0,np.cumsum([len(row) for row in result['segment_rows']],dtype=np.int64)]
    values['segment_values']=np.concatenate(result['segment_rows']).astype(np.int32) if result['segment_rows'] else np.empty(0,np.int32)
    if result['use_overlap']:
        values.update(overlap_offsets=result['overlap_offsets'],overlap_values=result['overlap_values'],
            overlap_complete=result['overlap_complete'])
    return values

def run(*,prepared_checkpoint,prepared_report,regions_json,brain,
        source_manifest,reference_edges,reference_retess,reference_edges_repeat,output,threads,
        use_overlap=False,reference_use_overlap=False,reuse_context=False):
    if output.exists():raise FileExistsError('fresh benchmark output required')
    started=time.perf_counter()
    producer=json.loads(prepared_report.read_text())
    if producer['candidate_checkpoint_sha256']!=sha(prepared_checkpoint):
        raise ValueError('own prepared checkpoint producer SHA mismatch')
    with np.load(prepared_checkpoint,allow_pickle=False) as z:
        # Candidate edges/scores stored in this old producer are intentionally
        # not input to the new computation. Only own stage preparation fields.
        prepared={k:z[k].copy() for k in ['canonical','faces','labels','status','translation',
            'matrix','smoothed_original','original_normals','white_values','gray_values']}
    original_image=nib.load(brain);volume=np.asarray(original_image.dataobj)
    image_sha=sha(brain)
    if producer['input_sha256']['brain']!=image_sha:raise ValueError('same own MRI required')
    region_document=json.loads(regions_json.read_text())
    if region_document['prepared_checkpoint_sha256']!=sha(prepared_checkpoint):
        raise ValueError('explicit own region source SHA mismatch')
    regions=region_document['ordered_regions']
    kw=dict(canonical_vertices=prepared['canonical'],original_vertices=prepared['smoothed_original'],
        original_normals=prepared['original_normals'],faces=prepared['faces'],defect_labels=prepared['labels'],
        ordered_regions=regions,retention_status=prepared['status'],vertex_translation=prepared['translation'],
        white_values=prepared['white_values'],gray_values=prepared['gray_values'],volume=volume,
        ras_to_voxel=prepared['matrix'],defect_number=region_document['defect_number'],use_overlap=use_overlap)
    loaded=time.perf_counter()-started;output.mkdir(parents=True)
    values=[];times=[]
    context=None
    with thread_budget(threads=threads) as budget:
        if reuse_context:
            context=prepare_ordered_genetic_defect_context(**{k:kw[k] for k in
                ['canonical_vertices','faces','defect_labels','ordered_regions','vertex_translation']})
            kw['context']=context
        for name in ['cold','repeat']:
            result=ordered_genetic_defect_inputs(**kw);v=arrays(result)
            tick=time.perf_counter();np.savez_compressed(output/(name+'.private.npz'),**v)
            times.append(dict(run=name,seconds=result['seconds'],write_seconds=time.perf_counter()-tick))
            values.append(v)
    # Independent reference is only opened after both computed artifacts exist.
    dtype=np.dtype([('a','<i4'),('b','<i4'),('score','<f4'),('used','<i2'),('padding','<i2')])
    ref=np.fromfile(reference_edges,dtype=dtype);ref_repeat=np.fromfile(reference_edges_repeat,dtype=dtype)
    retess=load_retessellation_capture(capture_path=reference_retess)
    reference=dict(edge_vertices=np.column_stack((ref['a'],ref['b'])).astype(np.int32),
        edge_scores=ref['score'].copy(),edge_used=ref['used'].astype(np.int32),
        retess_edge_vertices=retess['edge_fields'][:,:2],retess_edge_used=retess['edge_fields'][:,2],
        overlap_offsets=retess['overlap_offsets'].astype(np.int64),overlap_values=retess['overlaps'],
        overlap_complete=retess['edge_fields'][:,3].astype(bool),ordering=retess['ordering'],
        inside_vertices=retess['inside_fields'][:,0],border_vertices=retess['border_vertices'],hull_vertices=retess['hull_vertices'])
    def comparisons(v):
        d={k:difference(v[k],reference[k]) for k in ['edge_vertices','edge_used','edge_scores',
            'ordering','inside_vertices','border_vertices','hull_vertices']}
        if use_overlap and reference_use_overlap:
            d.update({k:difference(v[k],reference[k]) for k in ['overlap_offsets','overlap_values','overlap_complete']})
        d['retess_edge_vertices']=difference(v['edge_vertices'],reference['retess_edge_vertices'])
        d['retess_edge_used']=difference(v['edge_used'],reference['retess_edge_used'])
        d['inside_status_source']=difference(v['inside_status_source'],retess['inside_fields'][:,1])
        return d
    comparisons_by_run=[comparisons(v) for v in values]
    repeated={k:difference(values[0][k],values[1][k]) for k in values[0]}
    source_names=['topology_genetic_defect_inputs','topology_preflight_python','topology_edge_mri_score','topology_edge_segments','thread_budget']
    report=dict(status='complete_stage_diagnostic',scope='own frozen preparation plus explicit source-order regions -> ETable/ES segments/overlap; no GA or whole recon-all',
        all_candidate_outputs_computed_without_reference=True,
        prepared_checkpoint_fields_consumed=list(prepared),prepared_checkpoint_candidate_edges_or_scores_consumed=False,
        prepared_region_provenance=region_document['provenance'],
        static_context_reused=reuse_context,
        static_context_prepare_seconds=context.seconds if context else None,
        static_context_statistics=dict(neighbor_entries=sum(map(len,context.neighbors)),
            unique_original_edges=len(context.original_pairs),unique_base_edges=len(context.base_pairs),
            owned_array_bytes=context.source_order.nbytes) if context else None,
        strict_native_etable_and_retess_inputs=(use_overlap==reference_use_overlap and all(c['byte_equal'] for d in comparisons_by_run for c in d.values())),
        candidate_use_overlap=use_overlap,reference_use_overlap=reference_use_overlap,
        same_overlap_policy=use_overlap==reference_use_overlap,
        native_overlap_comparison='not_assessed' if not (use_overlap and reference_use_overlap) else 'assessed',
        source_disabled_overlap_capture_entries=len(reference['overlap_values']) if not reference_use_overlap else None,
        native_segment_rows_comparison='not_assessed',full_producer_source_comparison='not_assessed',
        self_repeat_all_computed_fields_byte_equal=all(c['byte_equal'] for c in repeated.values()),
        source_reference_edge_repeat_byte_equal=ref.tobytes()==ref_repeat.tobytes(),
        comparisons_by_run=comparisons_by_run,self_repeat=repeated,
        edge_count=len(values[0]['edge_vertices']),old_edge_count=len(values[0]['old_edge_vertices']),
        source_max_segments_rows=len(result['segment_rows']),
        total_overlap_entries=len(values[0]['overlap_values']) if use_overlap else None,
        incomplete_overlap_rows=int(np.count_nonzero(~values[0]['overlap_complete'])) if use_overlap else None,
        source_max_overlap_edges=result['max_overlap_edges'],
        source_manifest_sha256=sha(source_manifest),script_sha256=sha(__file__),
        actual_module_sha256={name:sha(sys.modules['fnit.recon_all.'+name].__file__) for name in source_names},
        input_sha256={k:sha(path) for k,path in dict(prepared_checkpoint=prepared_checkpoint,prepared_report=prepared_report,
            regions=regions_json,brain=brain).items()},reference_capture_sha256={k:sha(path) for k,path in dict(
            sorted_edges=reference_edges,sorted_edges_repeat=reference_edges_repeat,retess_before=reference_retess).items()},
        candidate_checkpoint_sha256={name:sha(output/(name+'.private.npz')) for name in ['cold','repeat']},
        timings=times,loading_seconds=loaded,full_harness_including_IO_comparison_seconds=time.perf_counter()-started,
        thread_budget=budget,affinity=sorted(os.sched_getaffinity(0)),gpu_used=False,gpu_memory_bytes=None,
        complete_ga=False,production_default_changed=False,overall_metric_equivalence='not_assessed',final_mesh_quality='not_assessed')
    dest=output/'report.private.json';dest.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    (output/'COMPLETE.json').write_text(json.dumps(dict(status=report['status'],report_sha256=sha(dest)),indent=2)+'\n')
    print(json.dumps({k:report[k] for k in ['status','strict_native_etable_and_retess_inputs',
        'self_repeat_all_computed_fields_byte_equal','edge_count','old_edge_count','total_overlap_entries','timings']}))
    return report

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['prepared-checkpoint','prepared-report','regions-json','brain','source-manifest',
        'reference-edges','reference-retess','reference-edges-repeat','output']:
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--threads',type=int,default=4)
    p.add_argument('--use-overlap',action='store_true',help='显式启用候选重叠；默认原标准GA关闭')
    p.add_argument('--reference-use-overlap',action='store_true',help='仅参考实际已开启时声明；不能从0列表推断')
    p.add_argument('--reuse-context',action='store_true',help='同表面原邻接与边集只建立一次')
    run(**vars(p.parse_args()))

if __name__=='__main__':main()
