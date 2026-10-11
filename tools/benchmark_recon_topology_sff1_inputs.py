"""从raw nofix/qsphere自产SFF1前字段，参考球面只在候选保存后比较。"""
from __future__ import annotations
import time
_START=time.perf_counter()
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import nibabel.freesurfer.io as fsio
import numpy as np
from fnit.recon_all.topology_genetic_base_prepare import prepare_genetic_base_sff1_inputs
from fnit.recon_all.topology_preflight_python import center_sphere,project_and_smooth_sphere,defect_component_labels
from fnit.recon_all.thread_budget import thread_budget
_IMPORT=time.perf_counter()-_START


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def array_sha(value):return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def difference(first,second):
    first,second=np.asarray(first),np.asarray(second)
    if first.shape!=second.shape:return dict(shape_equal=False,bitwise_equal=False)
    error=np.abs(first.astype(np.float64)-second.astype(np.float64))
    return dict(shape_equal=True,dtype_equal=first.dtype==second.dtype,
        different_elements=int(np.count_nonzero(first!=second)),
        max_absolute=float(error.max(initial=0)),
        p99_absolute=float(np.percentile(error,99)) if error.size else 0.,
        bitwise_equal=first.dtype==second.dtype and first.tobytes()==second.tobytes(),
        first_sha256=array_sha(first),second_sha256=array_sha(second))


def run(*,input_surface,input_qsphere,source_manifest,input_manifest,
        output,threads,centered_reference=None,compute_sff_and_merge=False):
    if output.exists():raise FileExistsError('fresh output required')
    started=time.perf_counter()
    xyz,faces=fsio.read_geometry(str(input_surface))
    sphere,sfaces=fsio.read_geometry(str(input_qsphere))
    xyz,faces=np.asarray(xyz,np.float32),np.asarray(faces,np.int32)
    if xyz.shape!=sphere.shape or not np.array_equal(faces,sfaces):raise ValueError('same-order nofix/qsphere required')
    with thread_budget(threads=threads):
        precenter=project_and_smooth_sphere(sphere,faces)
        canonical,iterations=center_sphere(precenter)
        labels=defect_component_labels(canonical,faces)
        args=dict(vertices=xyz,faces=faces,canonical_vertices=canonical,defect_vertices=labels!=0,
            precenter_vertices=precenter)
        first=prepare_genetic_base_sff1_inputs(**args)
        second=prepare_genetic_base_sff1_inputs(**args)
        curvature=merged=curvature_repeat=merge_repeat=None
        if compute_sff_and_merge:
            from fnit.recon_all.topology_global_sff import global_second_fundamental_form_state
            from fnit.recon_all.topology_defect_merge import merge_neighboring_defects
            curvature=global_second_fundamental_form_state(**first['sff_arguments'])
            curvature_repeat=global_second_fundamental_form_state(**second['sff_arguments'])
            merge_args=dict(canonical_vertices=canonical,vertex_normals=first['canonical_normals'],
                initial_vertex_normals=first['initial_segment_vertex_normals'],
                original_vertex_area=first['raw_original_areas'],faces=faces,component_labels=labels)
            merged=merge_neighboring_defects(**merge_args)
            merge_repeat=merge_neighboring_defects(**merge_args)
    output.mkdir(parents=True)
    fields={'sff_'+name:value for name,value in first['sff_arguments'].items()}
    fields.update(raw_vertices=xyz,faces=faces,canonical_vertices=canonical,precenter_vertices=precenter,defect_labels=labels,
        initial_segment_vertex_normals=first['initial_segment_vertex_normals'],
        canonical_normals=first['canonical_normals'],raw_all_vertex_normals=first['raw_all_vertex_normals'],
        raw_original_areas=first['raw_original_areas'],
        one_ring_offsets=first['one_ring_offsets'],one_ring_neighbors=first['one_ring_neighbors'])
    if curvature is not None:
        fields.update({'curvature_'+name:curvature[name] for name in
                       ['k1','k2','h','gaussian_curvature','tangent_e1','tangent_e2']})
        for name in ['labels','source_order','vertex_translation','source_marks',
                     'initial_vertex_val','merge_events']:
            fields['merged_'+name]=merged[name]
        for name in ['vertices','border']:
            rows=[row[name] for row in merged['regions']]
            fields['merged_'+name+'_offsets']=np.asarray([0]+list(np.cumsum([len(row) for row in rows])),np.int32)
            fields['merged_'+name]=np.concatenate(rows) if rows else np.empty(0,np.int32)
        fields['merged_areas']=np.asarray([row['area'] for row in merged['regions']],np.float32)
    np.savez_compressed(output/'sff1_inputs.private.npz',**fields)
    repeat={name:difference(value,second['sff_arguments'][name]) for name,value in first['sff_arguments'].items()}
    repeat['raw_original_areas']=difference(first['raw_original_areas'],second['raw_original_areas'])
    repeat['initial_segment_vertex_normals']=difference(first['initial_segment_vertex_normals'],second['initial_segment_vertex_normals'])
    calculated_repeat={}
    if curvature is not None:
        calculated_repeat.update({'curvature_'+name:difference(curvature[name],curvature_repeat[name])
            for name in ['k1','k2','h','gaussian_curvature','tangent_e1','tangent_e2']})
        calculated_repeat.update({'merged_'+name:difference(merged[name],merge_repeat[name]) for name in
            ['labels','source_order','vertex_translation','source_marks','initial_vertex_val','merge_events']})
    reference=None
    # Only AFTER independent fields are written; reference is never an API input.
    if centered_reference is not None:
        ref,rfaces=fsio.read_geometry(str(centered_reference))
        reference=dict(vertices=difference(canonical,np.asarray(ref,np.float32)),
            faces=difference(faces,np.asarray(rfaces,np.int32)),file_sha256=sha(centered_reference))
    modules=['topology_genetic_base_prepare','topology_preflight_python','normal_pending_python',
        'inflate_topology','place_surface_normals','thread_budget']
    if curvature is not None:modules+=['topology_global_sff','topology_candidate_sff','topology_vnl_svd','topology_defect_merge']
    report=dict(status='complete_prepared_sff1_inputs',candidate_repeat=repeat,
        candidate_repeat_exact=all(r['bitwise_equal'] for r in repeat.values()),
        centered_reference=reference,normal_pending_counts=first['normal_pending_counts'],
        nvertices=len(xyz),nfaces=len(faces),inside_vertices=int(np.count_nonzero(labels)),
        ndefects=int(labels.max(initial=0)),center_iterations=iterations,
        timing=dict(import_seconds=_IMPORT,first_api_seconds=first['seconds'],repeat_api_seconds=second['seconds'],
            complete_tool_excluding_import_and_final_JSON_seconds=time.perf_counter()-started),
        input_sha256=dict(nofix=sha(input_surface),qsphere=sha(input_qsphere)),
        candidate_checkpoint_sha256=sha(output/'sff1_inputs.private.npz'),
        input_manifest_sha256=sha(input_manifest),source_manifest_sha256=sha(source_manifest),
        script_sha256=sha(__file__),actual_module_sha256={name:sha(sys.modules['fnit.recon_all.'+name].__file__) for name in modules},
        all_consumed_state_self_generated=True,reference_after_only=True,
        SFF1_curvature_not_computed=curvature is None,full_base_not_validated=True,complete_ga=False,
        sff_and_merge_calculated=curvature is not None,calculated_repeat=calculated_repeat,
        calculated_repeat_exact=all(row['bitwise_equal'] for row in calculated_repeat.values()),
        sff_timing=None if curvature is None else dict(first_seconds=curvature['seconds'],repeat_seconds=curvature_repeat['seconds']),
        merge_timing=None if merged is None else dict(first_seconds=merged['seconds'],repeat_seconds=merge_repeat['seconds']),
        merged_defects=None if merged is None else len(merged['regions']),
        merge_event_count=None if merged is None else len(merged['merge_events']),
        overall_metric_equivalence='not_assessed',gpu_used=False,gpu_memory_bytes=None,
        threads=threads,affinity=sorted(os.sched_getaffinity(0)),host=platform.node(),shared_host=True,
        OS_isolation='not_validated')
    (output/'report.private.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    (output/'COMPLETE.json').write_text(json.dumps(dict(status=report['status'],
        report_sha256=sha(output/'report.private.json')),indent=2)+'\n')
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ['input-surface','input-qsphere','source-manifest','input-manifest','output']:
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--centered-reference',type=Path)
    parser.add_argument('--threads',type=int,default=4)
    parser.add_argument('--compute-sff-and-merge',action='store_true',help='自产全局SFF1、原序合并与映射；不读取DVS参考字段')
    report=run(**vars(parser.parse_args()))
    print(json.dumps({name:report[name] for name in ['status','candidate_repeat_exact','timing']},allow_nan=False))


if __name__=='__main__':main()
