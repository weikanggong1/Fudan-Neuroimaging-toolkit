"""raw nofix/MRI自产三组median强度，再只读校准入口的真实字段比较。"""
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
import nibabel as nib
import nibabel.freesurfer.io as fsio
import numpy as np

from fnit.recon_all.topology_genetic_base_prepare import prepare_nofix_border_fields
from fnit.recon_all.topology_preflight_python import center_sphere,project_and_smooth_sphere,defect_component_labels
from fnit.recon_all.topology_edge_mri_score import source_topology_ras_to_voxel
from fnit.recon_all.thread_budget import thread_budget
from benchmark_recon_topology_calibration_fields import load_calibration_capture

_IMPORT=time.perf_counter()-_START


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def array_sha(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def difference(a,b):
    a,b=np.asarray(a),np.asarray(b)
    if a.shape!=b.shape:return dict(shape_equal=False,candidate_shape=list(a.shape),reference_shape=list(b.shape))
    error=np.abs(a.astype(np.float64)-b.astype(np.float64))
    return dict(shape_equal=True,dtype_equal=a.dtype==b.dtype,
        different_elements=int(np.count_nonzero(a!=b)),max_absolute=float(error.max(initial=0)),
        p99_absolute=float(np.percentile(error,99)) if error.size else 0.,
        bitwise_equal=a.dtype==b.dtype and a.tobytes()==b.tobytes(),
        candidate_array_sha256=array_sha(a),reference_array_sha256=array_sha(b))


def run(*,input_surface,input_qsphere,brain,reference_capture,reference_repeat,
        producer_benchmark,observer_build_receipt,observer_run_receipt,source_manifest,output,threads):
    if output.exists():raise FileExistsError('fresh diagnostic output required')
    started=time.perf_counter();raw,faces=fsio.read_geometry(str(input_surface))
    sphere,sfaces=fsio.read_geometry(str(input_qsphere))
    raw,faces=np.asarray(raw,np.float32),np.asarray(faces,np.int32)
    if raw.shape!=sphere.shape or not np.array_equal(faces,sfaces):raise ValueError('same ordered meshes required')
    image=nib.load(brain);volume=np.asarray(image.dataobj);matrix=source_topology_ras_to_voxel(volume_header=image.header)
    with thread_budget(threads=threads):
        canonical,_=center_sphere(project_and_smooth_sphere(sphere,faces));labels=defect_component_labels(canonical,faces)
        # Only outside val is the regression target. Zero is an explicit
        # diagnostic BEFORE input for ripped vertices, not invented defect.area.
        args=dict(vertices=raw,faces=faces,defect_vertices=labels!=0,
            initial_border_values=np.zeros(len(raw),np.float32),volume=volume,ras_to_voxel=matrix)
        first=prepare_nofix_border_fields(**args);second=prepare_nofix_border_fields(**args)
    output.mkdir(parents=True)
    np.savez_compressed(output/'candidate.private.npz',**{k:v for k,v in first.items() if isinstance(v,np.ndarray)},labels=labels)
    candidate_wall=time.perf_counter()-started
    # Reference is only opened once the independent candidate is saved.
    native=load_calibration_capture(capture_path=reference_capture)
    repeat=load_calibration_capture(capture_path=reference_repeat)
    if native['header'][6]!=0 or repeat['header'][6]!=0:raise ValueError('actual pre-calibration phase0 required')
    selected=labels==0
    comparisons={name:difference(first[name][selected],native['vertex_fields'][selected,6+i])
        for i,name in enumerate(['val','val2','val2bak'])}
    comparisons['white_all_vertices']=difference(first['val2'],native['vertex_fields'][:,7])
    comparisons['gray_all_vertices']=difference(first['val2bak'],native['vertex_fields'][:,8])
    # Same input identity checks do not use normals after smoothing as raw
    # ground truth. Those are distinct phases and are deliberately not compared.
    identity=dict(faces=difference(faces,native['faces']),all_defect_mask=difference(labels!=0,native['flags'][:,0]!=0))
    own_repeat={name:difference(first[name],second[name]) for name in ['val','val2','val2bak','raw_vertex_normals']}
    native_repeat={name:difference(value,repeat[name]) for name,value in native.items()}
    report=dict(status='complete_raw_nofix_border_fields_stage',comparisons=comparisons,input_identity=identity,
        strict_compared_fields_passed=all(v.get('bitwise_equal',False) for v in comparisons.values()),
        candidate_repeat=own_repeat,candidate_repeat_exact=all(v['bitwise_equal'] for v in own_repeat.values()),
        native_repeat=native_repeat,native_repeat_exact=all(v['bitwise_equal'] for v in native_repeat.values()),
        nvertices=len(raw),nfaces=len(faces),outside_vertices=int(selected.sum()),inside_vertices=int((~selected).sum()),
        initial_border_values_in_defects='explicit diagnostic zero input; defect.area not generated or validated',
        source_scalar_fields_not_consumed_from_reference=True,
        full_prebase_source_capture='pending authorization',raw_mri_to_base_fully_validated=False,
        complete_ga=False,production_default_changed=False,overall_metric_equivalence='not_assessed',
        timing=dict(import_seconds=_IMPORT,first_api_seconds=first['seconds'],repeat_api_seconds=second['seconds'],
            candidate_wall_including_input_IO_and_preflight_seconds=candidate_wall,
            complete_tool_seconds_excluding_import_and_final_JSON=time.perf_counter()-started),
        candidate_input_sha256={k:sha(v) for k,v in dict(orig_nofix=input_surface,qsphere_nofix=input_qsphere,brain=brain).items()},
        reference_sha256=dict(capture=sha(reference_capture),repeat=sha(reference_repeat)),
        producer_benchmark_sha256=sha(producer_benchmark),observer_build_receipt_sha256=sha(observer_build_receipt),
        observer_run_receipt_sha256=sha(observer_run_receipt),source_manifest_sha256=sha(source_manifest),
        source_manifest=json.loads(source_manifest.read_text()),candidate_checkpoint_sha256=sha(output/'candidate.private.npz'),
        script_sha256=sha(__file__),actual_imported_module_sha256={name:sha(sys.modules['fnit.recon_all.'+name].__file__) for name in
            ['topology_genetic_base_prepare','topology_edge_mri_score','topology_preflight_python','smooth_surface_python',
             'place_surface_normals','place_surface_border','mri_em_register','thread_budget']},
        host=platform.node(),affinity=sorted(os.sched_getaffinity(0)),threads=threads,gpu_used=False,
        gpu_memory_bytes=None,shared_host=True,OS_isolation='not_validated')
    (output/'report.private.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    (output/'COMPLETE.json').write_text(json.dumps(dict(status=report['status'],report_sha256=sha(output/'report.private.json')),indent=2)+'\n')
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['input-surface','input-qsphere','brain','reference-capture','reference-repeat','producer-benchmark',
                 'observer-build-receipt','observer-run-receipt','source-manifest','output']:
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--threads',type=int,default=4)
    r=run(**vars(p.parse_args()));print(json.dumps(dict(status=r['status'],passed=r['strict_compared_fields_passed'],timing=r['timing']),allow_nan=False))


if __name__=='__main__':main()
