"""原 nofix/qsphere/MRI独立生成完整base；不读取任何参考状态。"""
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
from fnit.recon_all.topology_genetic_base_prepare import prepare_nofix_genetic_base
from fnit.recon_all.topology_edge_mri_score import source_topology_ras_to_voxel
from fnit.recon_all.thread_budget import thread_budget
from benchmark_recon_topology_candidate_state import dump_state_checkpoint
_IMPORT=time.perf_counter()-_START


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(*,input_surface,input_qsphere,brain,hemisphere,input_manifest,source_manifest,output,threads):
    if output.exists():raise FileExistsError('fresh output required')
    started=time.perf_counter()
    xyz,faces,geometry=fsio.read_geometry(str(input_surface),read_metadata=True)
    sphere,sfaces=fsio.read_geometry(str(input_qsphere))
    xyz,faces,sphere=np.asarray(xyz,np.float32),np.asarray(faces,np.int32),np.asarray(sphere,np.float32)
    if xyz.shape!=sphere.shape or not np.array_equal(faces,sfaces):raise ValueError('same ordered nofix/qsphere required')
    image=nib.load(brain);volume=np.asarray(image.dataobj)
    matrix=source_topology_ras_to_voxel(volume_header=image.header)
    # Standard fresh triangle reader metadata. Volume geometry is read from
    # the own nofix tail; no native after state or DVS object is opened here.
    metadata=dict(use_real_ras=0,hemisphere=0 if hemisphere=='lh' else 1,
        status=0,origxyz_status=0,volume_geometry=dict(geometry))
    with thread_budget(threads=threads):
        first=prepare_nofix_genetic_base(vertices=xyz,faces=faces,qsphere_vertices=sphere,
            volume=volume,ras_to_voxel=matrix,source_metadata=metadata)
    output.mkdir(parents=True)
    dump_state_checkpoint(output/'base_state.private.npz',first['surface_state'])
    np.savez_compressed(output/'base_maps.private.npz',vertex_translation=first['vertex_translation'],
        face_translation=first['face_translation'],source_order=first['source_order'],
        original_defect_labels=first['merge']['labels'],original_retention_status=first['retention_status'],
        original_initial_vertex_val=first['merge']['initial_vertex_val'])
    modules=['topology_genetic_base_prepare','topology_genetic_base','topology_defect_merge',
        'topology_global_sff','topology_candidate_sff','topology_vnl_svd','normal_pending_python',
        'inflate_topology','topology_preflight_python','smooth_surface_python','topology_edge_mri_score',
        'place_surface_normals','thread_budget']
    report=dict(status='complete_independent_raw_nofix_base_candidate',
        full_base_strict_validation='not_assessed',reference_state_consumed=False,
        all_computed_state_self_generated=True,complete_ga=False,production_default_changed=False,
        overall_metric_equivalence='not_assessed',OS_isolation='not_validated',
        nvertices=len(xyz),nfaces_input=len(faces),nfaces_base=first['surface_state']['nfaces'],
        merged_defects=len(first['merge']['regions']),merge_events=len(first['merge']['merge_events']),
        inside_vertices=int(np.count_nonzero(first['merge']['labels'])),
        retained_inside=int(np.count_nonzero(first['retention_status']==1)),
        discarded_inside=int(np.count_nonzero(first['retention_status']==-1)),
        metadata_origin='own nofix geometry tail; fresh triangle use_real_ras/status/hemisphere definitions',
        normal_pending_counts=first['metric']['normal_pending_counts'],
        timing=dict(import_seconds=_IMPORT,full_candidate_api_seconds=first['seconds'],
            metric_seconds=first['metric']['seconds'],sff1_seconds=first['sff1']['seconds'],
            merge_seconds=first['merge']['seconds'],field_prepare_seconds=first['prepared']['stage_seconds'],
            tool_including_IO_excluding_import_final_JSON_seconds=time.perf_counter()-started),
        input_sha256={key:sha(path) for key,path in dict(nofix=input_surface,qsphere=input_qsphere,brain=brain).items()},
        input_manifest_sha256=sha(input_manifest),source_manifest_sha256=sha(source_manifest),
        actual_module_sha256={name:sha(sys.modules['fnit.recon_all.'+name].__file__) for name in modules},
        script_sha256=sha(__file__),checkpoint_sha256={name:sha(output/name) for name in
            ['base_state.private.npz','base_maps.private.npz']},
        threads=threads,affinity=sorted(os.sched_getaffinity(0)),host=platform.node(),shared_host=True,
        gpu_used=False,gpu_memory_bytes=None)
    (output/'report.private.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    (output/'COMPLETE.json').write_text(json.dumps(dict(status=report['status'],
        report_sha256=sha(output/'report.private.json')),indent=2)+'\n')
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ['input-surface','input-qsphere','brain','input-manifest','source-manifest','output']:
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--hemisphere',choices=['lh','rh'],required=True)
    parser.add_argument('--threads',type=int,default=4)
    result=run(**vars(parser.parse_args()))
    print(json.dumps({name:result[name] for name in ['status','timing','nfaces_base','merged_defects']}))


if __name__=='__main__':main()
