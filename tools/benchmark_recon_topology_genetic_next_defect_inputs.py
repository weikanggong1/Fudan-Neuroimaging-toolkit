"""自产原准备+自产初始邻接→当前边表 ABBA；输出后才读取参考。"""
from __future__ import annotations
import time
_START=time.perf_counter()
import argparse,hashlib,json,os,resource,sys
from pathlib import Path
import numpy as np
import nibabel as nib
from fnit.recon_all.topology_genetic_defect_inputs import (
    ordered_genetic_defect_inputs,prepare_ordered_genetic_defect_context)
from fnit.recon_all.topology_genetic_next_defect_inputs import next_ordered_genetic_defect_inputs
from fnit.recon_all.topology_retessellation_python import source_base_neighbors
from fnit.recon_all.thread_budget import thread_budget
from benchmark_recon_topology_genetic_defect_inputs import arrays,difference
_IMPORT=time.perf_counter()-_START


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def own_initial_base(*,context,canonical_vertices,defect_labels,vertex_translation):
    """成熟源有序base邻接的自产初始化；不读取任何reference state。"""
    rows=source_base_neighbors(source_neighbors=context.neighbors,labels=defect_labels,
        translation=vertex_translation)
    rows=[np.asarray(row,np.int32) for row in rows]
    n=len(defect_labels);top=np.zeros((n,7),np.int32)
    top[:,3]=[len(row) for row in rows]
    active=top[:,3]>0;top[active,:2]=1;top[:,2]=top[:,3]
    return dict(canonical_vertices=canonical_vertices[context.source_order].copy(),
        vertex_topology=top,neighbor_storage=rows)


def run(*,prepared_checkpoint,prepared_report,regions_json,brain,source_manifest,
        reference_checkpoint,reference_report,reference_edges,output,threads):
    if output.exists():raise FileExistsError('fresh current-base benchmark output required')
    started=time.perf_counter();producer=json.loads(prepared_report.read_text())
    if producer['candidate_checkpoint_sha256']!=sha(prepared_checkpoint):
        raise ValueError('own prepared checkpoint producer SHA mismatch')
    with np.load(prepared_checkpoint,allow_pickle=False) as z:
        prepared={k:z[k].copy() for k in ('canonical','faces','labels','status','translation',
            'matrix','smoothed_original','original_normals','white_values','gray_values')}
    image=nib.load(brain);volume=np.asarray(image.dataobj)
    if producer['input_sha256']['brain']!=sha(brain):raise ValueError('same own MRI required')
    region_document=json.loads(regions_json.read_text())
    if region_document['prepared_checkpoint_sha256']!=sha(prepared_checkpoint):
        raise ValueError('own explicit region source SHA mismatch')
    kw=dict(canonical_vertices=prepared['canonical'],original_vertices=prepared['smoothed_original'],
        original_normals=prepared['original_normals'],faces=prepared['faces'],defect_labels=prepared['labels'],
        ordered_regions=region_document['ordered_regions'],retention_status=prepared['status'],
        vertex_translation=prepared['translation'],white_values=prepared['white_values'],
        gray_values=prepared['gray_values'],volume=volume,ras_to_voxel=prepared['matrix'],
        defect_number=region_document['defect_number'],use_overlap=False)
    loading=time.perf_counter()-started;output.mkdir(parents=True)
    values=[];runs=[]
    with thread_budget(threads=threads) as budget:
        context=prepare_ordered_genetic_defect_context(**{k:kw[k] for k in
            ('canonical_vertices','faces','defect_labels','ordered_regions','vertex_translation')})
        tick=time.perf_counter()
        state=own_initial_base(context=context,canonical_vertices=kw['canonical_vertices'],
            defect_labels=kw['defect_labels'],vertex_translation=kw['vertex_translation'])
        initial_seconds=time.perf_counter()-tick
        offsets=np.r_[0,np.cumsum([len(row) for row in state['neighbor_storage']],dtype=np.int64)]
        storage=np.concatenate(state['neighbor_storage']).astype(np.int32)
        np.savez_compressed(output/'own_initial_state.private.npz',canonical_vertices=state['canonical_vertices'],
            vertex_topology=state['vertex_topology'],neighbor_offsets=offsets,neighbor_storage=storage)
        before_signature=hashlib.sha256(state['canonical_vertices'].tobytes()+state['vertex_topology'].tobytes()
            +offsets.tobytes()+storage.tobytes()).hexdigest()
        for name,mode in [('old-a1','old'),('next-b1','next'),('next-b2','next'),('old-a2','old')]:
            tick=time.perf_counter()
            result=(ordered_genetic_defect_inputs(**kw,context=context) if mode=='old' else
                next_ordered_genetic_defect_inputs(**kw,context=context,current_surface_state=state))
            elapsed=time.perf_counter()-tick;v=arrays(result)
            write_started=time.perf_counter();dest=output/(name+'.private.npz')
            np.savez_compressed(dest,**v);write_seconds=time.perf_counter()-write_started
            runs.append(dict(run=name,implementation=mode,complete_api_wall_seconds=elapsed,
                returned_seconds=result['seconds'],write_seconds=write_seconds,
                checkpoint_sha256=sha(dest),dynamic_base=result.get('dynamic_base')))
            values.append(v)
        after_storage=np.concatenate(state['neighbor_storage']).astype(np.int32)
        after_signature=hashlib.sha256(state['canonical_vertices'].tobytes()+state['vertex_topology'].tobytes()
            +offsets.tobytes()+after_storage.tobytes()).hexdigest()
    # After-only comparison. Neither old own outputs nor original reference
    # EDGE rows were input to the new current-state computation above.
    old_producer=json.loads(reference_report.read_text())
    if old_producer['candidate_checkpoint_sha256']['cold']!=sha(reference_checkpoint):
        raise ValueError('frozen own old-stage report/checkpoint SHA mismatch')
    with np.load(reference_checkpoint,allow_pickle=False) as z:old_reference={k:z[k] for k in z.files}
    dtype=np.dtype([('a','<i4'),('b','<i4'),('score','<f4'),('used','<i2'),('padding','<i2')])
    ref=np.fromfile(reference_edges,dtype=dtype)
    native=dict(edge_vertices=np.column_stack((ref['a'],ref['b'])).astype(np.int32),
        edge_scores=ref['score'].copy(),edge_used=ref['used'].astype(np.int32))
    pair=[{k:difference(values[0][k],v[k]) for k in values[0]} for v in values[1:]]
    own=[{k:difference(v[k],old_reference[k]) for k in old_reference} for v in values]
    original=[{k:difference(v[k],native[k]) for k in native} for v in values]
    names=('topology_genetic_next_defect_inputs','topology_genetic_defect_inputs',
        'topology_preflight_python','topology_edge_mri_score','topology_edge_segments',
        'topology_retessellation_python','thread_budget')
    report=dict(status='complete_current_base_stage',dataset='public ds000114',subject='sub-07 ses-test',hemisphere='lh',
        scope='same own frozen MRI preparation plus own source-base neighbors -> current ETable ABBA; no sequential GA',
        reference_inputs_consumed=False,native_program_called=False,
        own_initial_state_producer='mature source_base_neighbors from original ordered adjacency and own labels/mapping',
        own_initial_state_fields=list(state),initial_state_bytes_unchanged=before_signature==after_signature,
        own_initial_state_sha256=sha(output/'own_initial_state.private.npz'),
        all_ABBA_outputs_byte_equal=all(c['byte_equal'] for d in pair for c in d.values()),
        frozen_own_old_stage_all_fields_byte_equal=all(c['byte_equal'] for d in own for c in d.values()),
        frozen_original_etable_fields_byte_equal=all(c['byte_equal'] for d in original for c in d.values()),
        ABBA_comparison=pair,frozen_own_old_stage_comparison=own,original_etable_comparison=original,
        runs=runs,import_seconds=_IMPORT,loading_seconds=loading,static_context_prepare_seconds=context.seconds,
        own_initial_state_prepare_seconds=initial_seconds,
        full_harness_with_loading_context_IO_comparison_seconds=time.perf_counter()-started,
        edge_count=len(values[0]['edge_vertices']),old_edge_count=len(values[0]['old_edge_vertices']),
        source_manifest_sha256=sha(source_manifest),script_sha256=sha(__file__),
        actual_module_sha256={name:sha(sys.modules['fnit.recon_all.'+name].__file__) for name in names},
        input_sha256={k:sha(p) for k,p in dict(prepared_checkpoint=prepared_checkpoint,prepared_report=prepared_report,
            regions_json=regions_json,brain=brain).items()},
        after_only_reference_sha256={k:sha(p) for k,p in dict(old_checkpoint=reference_checkpoint,
            old_report=reference_report,original_edges=reference_edges).items()},
        frozen_old_source_manifest_sha256=old_producer['source_manifest_sha256'],
        dynamic_noninitial_native_comparison='not_assessed',native_segment_rows_comparison='not_assessed',
        use_overlap=False,native_overlap_comparison='not_assessed (source standard policy disabled)',
        initial_state_full_copied_attributes_comparison='not_assessed (only consumed canonical/topology/storage)',
        thread_budget=budget,affinity=sorted(os.sched_getaffinity(0)),shared_host=True,
        process_maxrss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        gpu_used=False,gpu_memory_bytes=None,complete_ga=False,final_mesh_quality='not_assessed',
        overall_metric_equivalence='not_assessed',production_default_changed=False)
    path=output/'report.public.json';path.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    (output/'COMPLETE.json').write_text(json.dumps(dict(status=report['status'],report_sha256=sha(path)),indent=2)+'\n')
    print(json.dumps({k:report[k] for k in ('status','edge_count','old_edge_count','all_ABBA_outputs_byte_equal',
        'frozen_own_old_stage_all_fields_byte_equal','frozen_original_etable_fields_byte_equal','runs')}))
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('prepared-checkpoint','prepared-report','regions-json','brain','source-manifest',
                 'reference-checkpoint','reference-report','reference-edges','output'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--threads',type=int,default=4)
    run(**vars(p.parse_args()))


if __name__=='__main__':main()
