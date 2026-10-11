"""公开 nofix/qsphere→自产完整 K/dot 先验，保存后才读取同域参考。"""
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
from fnit.recon_all.topology_global_sff import global_second_fundamental_form_state
from fnit.recon_all.topology_global_priors import global_topology_priors
from fnit.recon_all.topology_preflight_python import (
    center_sphere, project_and_smooth_sphere, defect_component_labels)
from fnit.recon_all.smooth_surface_python import average_positions

_IMPORT=time.perf_counter()-_START


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def difference(candidate, reference):
    a,b=np.asarray(candidate),np.asarray(reference)
    if a.shape!=b.shape or a.dtype!=b.dtype:
        return dict(shape_equal=a.shape==b.shape,dtype_equal=a.dtype==b.dtype,
            candidate_shape=list(a.shape),reference_shape=list(b.shape),bitwise_equal=False)
    delta=np.abs(a.astype(np.float64)-b.astype(np.float64))
    return dict(shape_equal=True,dtype_equal=True,
        differing_values=int(np.count_nonzero(a!=b)),
        differing_bytes=int(np.count_nonzero(np.frombuffer(a.tobytes(),np.uint8)!=
                                             np.frombuffer(b.tobytes(),np.uint8))),
        bitwise_equal=a.tobytes()==b.tobytes(),
        maximum_absolute_error=float(np.max(delta,initial=0)),
        p99_absolute_error=float(np.percentile(delta,99)) if delta.size else 0.)


def curvature_pdf_geometry(histogram):
    """FNITFIT1/既有scorer顺序：xstart,ystart,xsize,ysize，不是spacing先行。"""
    return np.asarray([histogram['k1_start'],histogram['k2_start'],
                       histogram['k1_width'],histogram['k2_width']],np.float32)


def produce(*, raw_vertices, faces, qsphere_vertices):
    """只消费自身未修复网格/qsphere；完整normal/SFF/PDF算法均复用成熟接口。"""
    started=time.perf_counter()
    stage_started=started
    precenter=project_and_smooth_sphere(qsphere_vertices,faces)
    canonical,iterations=center_sphere(precenter)
    labels=defect_component_labels(canonical,faces)
    sphere_seconds=time.perf_counter()-stage_started
    metric=prepare_genetic_base_sff1_inputs(vertices=raw_vertices,faces=faces,
        canonical_vertices=canonical,defect_vertices=labels!=0,precenter_vertices=precenter)
    sff=global_second_fundamental_form_state(**metric['sff_arguments'])
    stage_started=time.perf_counter()
    original=average_positions(vertices=raw_vertices,faces=faces,iterations=2,device='cpu')
    smoothing_seconds=time.perf_counter()-stage_started
    prior=global_topology_priors(principal_curvature_fields=sff,
        smoothed_original_vertices=original,faces=faces,
        neighbor_offsets=metric['one_ring_offsets'],neighbors=metric['one_ring_neighbors'],
        vertex_rip=metric['sff_arguments']['vertex_rip'],surface_status='sphere')
    histogram,dot=prior['curvature'],prior['normal_dot']
    fields=dict(raw_vertices=raw_vertices,faces=faces,canonical_vertices=canonical,
        smoothed_original_vertices=original,smoothed_metric_vertices=prior['smoothed_metric'].positions,
        smoothed_metric_normals=prior['smoothed_metric'].normals,
        sff1_k1=sff['k1'],sff1_k2=sff['k2'],defect_labels=labels,
        one_ring_offsets=metric['one_ring_offsets'],one_ring_neighbors=metric['one_ring_neighbors'],
        k1_bins=histogram['k1_bins'],k2_bins=histogram['k2_bins'],
        k1_counts=histogram['k1_counts'],k2_counts=histogram['k2_counts'],
        k1_bin_size=np.asarray([histogram['k1_width']],np.float32),
        k2_bin_size=np.asarray([histogram['k2_width']],np.float32),
        k_pdf_geometry=curvature_pdf_geometry(histogram),
        k_pdf=histogram['joint'][:,:,None],dot_bins=dot['bins'],dot_counts=dot['counts'],
        dot_bin_size=np.asarray([dot['bin_size']],np.float32),
        dot_raw_counts=dot['raw_counts'],dot_projected_values=dot['projected_values'])
    timing=dict(sphere_preparation=sphere_seconds,mixed_normal_and_neighborhood=metric['seconds'],
        full_SFF1=sff['seconds'],two_original_averages=smoothing_seconds,
        **prior['step_seconds'],api_seconds=time.perf_counter()-started)
    return fields,dict(timing=timing,center_iterations=iterations,
        defect_vertices=int(np.count_nonzero(labels)),unmerged_defects=int(labels.max(initial=0)),
        initial_normal_phase_available=metric['initial_segment_vertex_normals'] is not None,
        mixed_normal_pending=metric['normal_pending_counts'],
        dot_normal_pending=prior['smoothed_metric'].pending_counts.tolist(),
        dot_metric_changed_vertices=int(len(prior['smoothed_metric'].changed_vertices)),
        dot_metric_unresolved_vertices=int(len(prior['smoothed_metric'].unresolved_vertices)),
        sampled_dot_edges=dot['sampled_edges'])


def compare_priors(*, candidate, reference):
    """FNITFIT1 在 SFF2 后保存 current=smoothed ORIGINAL，raw 仅核输入哈希。"""
    domains=dict(faces=difference(candidate['faces'],reference['faces']),
        smoothed_original_vertices=difference(candidate['smoothed_original_vertices'],
                                               reference['vertex_fields'][:,:3]),
        captured_current_is_smoothed_original=difference(candidate['smoothed_original_vertices'],
                                                         reference['vertex_fields'][:,3:6]),
        one_ring_offsets=difference(candidate['one_ring_offsets'],reference['neighbor_offsets']),
        one_ring_neighbors=difference(candidate['one_ring_neighbors'],reference['neighbors']))
    exact=all(row['bitwise_equal'] for row in domains.values())
    names=('k1_bins','k2_bins','k1_counts','k2_counts','k1_bin_size','k2_bin_size',
           'k_pdf_geometry','k_pdf','dot_bins','dot_counts','dot_bin_size')
    comparisons={name:difference(candidate[name],reference[name]) for name in names} if exact else {}
    return domains,comparisons


def run(*, input_surface, input_qsphere, reference, reference_repeat,
        reference_receipt, source_manifest, output, threads):
    """隔离benchmark：原始数据两次独立生成，先写候选再进行同域对照。"""
    started=time.perf_counter();output=Path(output)
    if output.exists():raise FileExistsError('new empty output directory required')
    from fnit.recon_all.thread_budget import thread_budget
    with thread_budget(threads=threads):
        xyz,triangles=fsio.read_geometry(str(input_surface))
        qsphere,qfaces=fsio.read_geometry(str(input_qsphere))
        if not np.array_equal(triangles,qfaces):raise ValueError('same ordered nofix/qsphere faces required')
        xyz,triangles,qsphere=np.asarray(xyz,np.float32),np.asarray(triangles,np.int32),np.asarray(qsphere,np.float32)
        first,meta=produce(raw_vertices=xyz,faces=triangles,qsphere_vertices=qsphere)
        second,repeat_meta=produce(raw_vertices=xyz,faces=triangles,qsphere_vertices=qsphere)
    output.mkdir(parents=True)
    np.savez_compressed(output/'candidate.private.npz',**first)
    repeat={key:difference(first[key],second[key]) for key in first}
    # All production calculations and checkpoint writes finish before reference reads.
    from benchmark_recon_topology_mri_likelihood import load_statistics_capture
    ref=load_statistics_capture(capture_path=reference)
    ref2=load_statistics_capture(capture_path=reference_repeat)
    receipt=json.loads(Path(reference_receipt).read_text())
    input_hashes=dict(nofix=sha(input_surface),qsphere=sha(input_qsphere))
    reference_input_hashes_match=input_hashes==receipt['reference_input_sha256']
    domains,comparisons=compare_priors(candidate=first,reference=ref)
    domain_exact=all(row['bitwise_equal'] for row in domains.values())
    prior_names=('k1_bins','k2_bins','k1_counts','k2_counts','k1_bin_size','k2_bin_size',
                 'k_pdf_geometry','k_pdf','dot_bins','dot_counts','dot_bin_size')
    ref_repeat={name:difference(ref[name],ref2[name]) for name in prior_names}
    consumed=['topology_global_priors','topology_global_dot_prior','topology_global_sff',
        'topology_curvature_histogram','topology_intensity_histogram','topology_genetic_base_prepare',
        'topology_preflight_python','normal_pending_python','place_surface_normals','smooth_surface_python',
        'inflate_topology','topology_candidate_sff','topology_vnl_svd','thread_budget']
    strict=domain_exact and reference_input_hashes_match and all(row['bitwise_equal'] for row in comparisons.values())
    repeated=all(row['bitwise_equal'] for row in repeat.values())
    report=dict(schema_version=1,status='complete_global_priors' if strict and repeated else 'failed_strict_priors',
        case='public ds000114 sub07 LH same frozen nofix/qsphere',
        nvertices=len(xyz),nfaces=len(triangles),domain_comparisons=domains,
        domain_correspondence_exact=domain_exact,prior_comparisons=comparisons,
        scoring_priors_strict_exact=strict,candidate_repeat=repeat,candidate_repeat_exact=repeated,
        reference_repeat=ref_repeat,reference_repeat_exact=sha(reference)==sha(reference_repeat),
        metadata=meta,repeat_metadata=repeat_meta,
        timing=dict(import_seconds=_IMPORT,
            complete_tool_excluding_import_and_final_JSON_seconds=time.perf_counter()-started),
        input_sha256=input_hashes,reference_input_sha256=receipt['reference_input_sha256'],
        reference_input_hashes_match=reference_input_hashes_match,
        reference_capture_sha256=sha(reference),reference_repeat_capture_sha256=sha(reference_repeat),
        reference_receipt_sha256=sha(reference_receipt),reference_receipt=receipt,
        candidate_checkpoint_sha256=sha(output/'candidate.private.npz'),
        source_manifest_sha256=sha(source_manifest),script_sha256=sha(__file__),
        actual_module_sha256={name:sha(sys.modules['fnit.recon_all.'+name].__file__) for name in consumed},
        all_consumed_fields_self_generated=True,reference_after_only=True,
        source_reference_SFF2_fields_consumed=False,original_SFF2_recomputed=False,
        full_ga_complete=False,production_default_changed=False,
        raw_original_state_not_modified=True,overall_metric_equivalence='not_assessed',
        cpu_affinity=sorted(os.sched_getaffinity(0)),threads=threads,host=platform.node(),
        python=platform.python_version(),numpy=np.__version__,
        gpu_used=False,gpu_memory_bytes=None,shared_host=True,isolated_deployment='not_validated')
    (output/'report.private.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    (output/'COMPLETE.json').write_text(json.dumps(dict(status=report['status'],
        report_sha256=sha(output/'report.private.json')),indent=2)+'\n')
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('input-surface','input-qsphere','reference','reference-repeat','reference-receipt','source-manifest','output'):
        parser.add_argument('--'+name,required=True,type=Path)
    parser.add_argument('--threads',type=int,default=4)
    result=run(**vars(parser.parse_args()))
    print(json.dumps(dict(status=result['status'],strict=result['scoring_priors_strict_exact'])))
    return 0 if result['status']=='complete_global_priors' else 1


if __name__=='__main__':raise SystemExit(main())
