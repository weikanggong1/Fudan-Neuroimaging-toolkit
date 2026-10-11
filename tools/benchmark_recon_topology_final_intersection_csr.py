"""真实自产压缩CURRENT/TMP网格：原默认与显式源CSR完整文件API对照。"""
from __future__ import annotations
import time
_START=time.perf_counter()
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import nibabel.freesurfer.io as fsio
import numpy as np
from fnit.recon_all.thread_budget import thread_budget
from fnit.recon_all.mris_remove_intersection_python import run_remove_intersection
from fnit.recon_all.place_surface_smoothing import _ordered_neighbors
from benchmark_recon_topology_candidate_state import load_state_checkpoint
_IMPORT=time.perf_counter()-_START


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(*, compact_state, input_surface, source_manifest, output, threads=4):
    """自产compact完整NPZ+同源几何头→fresh control/CSR文件API与安全数值。

    只操作本次自产 pre-GA 基底CURRENT/TMP域，原参考after不进入计算。
    compact_state完整state含有序neighbor storage和vnum；input_surface
    为同源nofix，仅读取几何头。两条完整API均严格source+独立几何双
    复检，不通过则退出失败不发布替代网格。零交只证明zero分支/数据
    贯通；非零100SOAP真实阶段及全部25缺陷完成面另验。线程预算固定，
    GPU隐藏由启动者设置。返回只读大网格兼容报告，不声称整例通过。
    """
    if output.exists():raise FileExistsError('fresh diagnostic directory required')
    start=time.perf_counter();state=load_state_checkpoint(compact_state)
    _,_,geometry=fsio.read_geometry(str(input_surface),read_metadata=True)
    vertices=np.asarray(state['current_vertices'],np.float32)
    faces=np.asarray(state['faces'])[:state['nfaces']]
    if np.asarray(state['vertex_rip']).any():raise ValueError('already compacted nonripped state required')
    rows=[np.asarray(row)[:int(top[3])] for row,top in zip(state['neighbor_storage'],state['vertex_topology'])]
    offsets=np.r_[0,np.cumsum([len(row) for row in rows])].astype(np.int32)
    neighbors=np.concatenate(rows).astype(np.int32) if rows else np.empty(0,np.int32)
    face_indices,face_valid,_=_ordered_neighbors(faces,len(vertices))
    unbacked=0
    for index,row in enumerate(rows):
        face_supported=set(face_indices[index,face_valid[index]].tolist())
        unbacked+=sum(int(vertex) not in face_supported for vertex in row)
    loading=time.perf_counter()-start;output.mkdir(parents=True)
    source=output/'own_compacted_CURRENT.surface'
    tick=time.perf_counter();fsio.write_geometry(str(source),vertices,faces,
        create_stamp='FNIT self-generated real pre-GA compact CURRENT/TMP',volume_info=geometry)
    initial_write=time.perf_counter()-tick;reports=[];status='running'
    with thread_budget(threads=threads):
        for mode in ('default_faces','explicit_ordered_CSR'):
            kwargs={} if mode=='default_faces' else dict(ordered_neighbor_offsets=offsets,ordered_neighbors=neighbors)
            tick=time.perf_counter()
            try:
                report=run_remove_intersection(input_path=source,output_path=output/(mode+'.surface'),
                    marking_backend='source_numba',require_zero=True,report_path=output/(mode+'.private.json'),**kwargs)
            except BaseException as error:
                (output/'FAILURE.json').write_text(json.dumps(dict(status='failed_complete_file_API',mode=mode,error_class=type(error).__name__,seconds=time.perf_counter()-tick),indent=2)+'\n')
                raise
            reports.append(dict(mode=mode,wall_seconds=time.perf_counter()-tick,
                input_sha256=report['input_sha256'],output_sha256=report['output_sha256'],cleanup=report['cleanup'],
                source_intersections_after=report['source_intersecting_faces_after_independent_recheck'],
                geometric_intersections_after=report['geometric_intersecting_faces_after_independent_recheck'],
                read_seconds=report['read_seconds'],compute_seconds=report['compute_seconds'],
                recheck_seconds=report['recheck_seconds'],write_seconds=report['write_seconds']))
    a,fa=fsio.read_geometry(str(output/'default_faces.surface'));b,fb=fsio.read_geometry(str(output/'explicit_ordered_CSR.surface'))
    coordinates=a.astype(np.float32).tobytes()==b.astype(np.float32).tobytes();faces_equal=fa.tobytes()==fb.tobytes()
    samefiles=reports[0]['output_sha256']==reports[1]['output_sha256']
    zero=all(r['source_intersections_after']==0 and r['geometric_intersections_after']==0 for r in reports)
    if not (coordinates and faces_equal and samefiles and zero):raise RuntimeError('same-input cleanup compatibility gate failed')
    modules=('place_surface_final_cleanup','mris_remove_intersection_python','topology_finish_geometry','topology_compact')
    import importlib
    result=dict(status='passed_actual_compact_CURRENT_sameinput_file_API',completed_25_defects_input=False,
        full_ga_complete=False,overall_metric_equivalence='not_assessed',full_mesh_quality='not_assessed',
        original_program_reference='not_assessed',nonzero_cleanup_real_regression='not_assessed' if all(r['cleanup']['intersecting_faces_before']==0 for r in reports) else 'compared',
        vertices=len(vertices),faces=len(faces),ordered_neighbor_entries=len(neighbors),unbacked_directed_neighbor_entries=unbacked,
        same_input_coordinate_bytes=coordinates,same_ordered_faces=faces_equal,same_surface_bytes=samefiles,
        coordinate_max_mm=float(np.max(np.abs(a-b),initial=0)),coordinate_P99_mm=float(np.percentile(np.abs(a-b),99)),
        strict_source_and_independent_geometric_recheck=zero,results=reports,
        loading_seconds=loading,initial_surface_write_seconds=initial_write,import_seconds=_IMPORT,
        tool_wall_seconds=time.perf_counter()-start,threads=threads,CPU_affinity=sorted(os.sched_getaffinity(0)),
        gpu_used=False,gpu_peak_bytes=None,input_sha256=dict(compact_state=digest(compact_state),input_surface=digest(input_surface),source_manifest=digest(source_manifest)),
        module_sha256={name:digest(importlib.import_module('fnit.recon_all.'+name).__file__) for name in modules},
        script_sha256=digest(__file__))
    (output/'report.private.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    (output/'COMPLETE.json').write_text(json.dumps(dict(status=result['status'],report_sha256=digest(output/'report.private.json')),indent=2)+'\n')
    print(json.dumps(dict(status=result['status'],vertices=len(vertices),unbacked_directed_neighbor_entries=unbacked,seconds=result['tool_wall_seconds'])))
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('compact-state','input-surface','source-manifest','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--threads',type=int,default=4)
    run(**vars(parser.parse_args()))
