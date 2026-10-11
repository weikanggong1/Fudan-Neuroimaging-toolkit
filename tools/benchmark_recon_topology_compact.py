"""自产真实基底上的finish/compact稳定性诊断；不冒充全25缺陷后参考回归。"""
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

from fnit.recon_all.thread_budget import thread_budget
from fnit.recon_all.topology_compact import compact_ripped_surface_elements
from fnit.recon_all.topology_finish_geometry import finish_topology_geometry
from benchmark_recon_topology_candidate_state import load_state_checkpoint, dump_state_checkpoint, compare_states
_IMPORT=time.perf_counter()-_START


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _json(value):
    if isinstance(value,np.ndarray):return value.tolist()
    if isinstance(value,np.generic):return value.item()
    raise TypeError(type(value).__name__)


def _rows(container,prefix):
    off=container[prefix+'_offsets'];values=container[prefix]
    if off.ndim!=1 or values.ndim!=1 or off[0]!=0 or off[-1]!=len(values):raise ValueError('complete own merged rows required')
    return [values[a:b].copy() for a,b in zip(off[:-1],off[1:])]


def run(*,input_state,input_maps,input_regions,input_surface,input_manifest,source_manifest,
        hemisphere,output,threads,iterations=5):
    """真实自产pre-GA基底→完整finish+compact两重复+compact幂等检查。

    本诊断输入是rawBase自产checkpoint及同源merged regions，不是已完成
    25缺陷的候选；只说明实际大网格接口稳定/可调用，不判最终closedmesh。
    原nofix仅提供自产几何头，reference after或原程序输出不进入计算。
    输出保存完整私有NPZ、JSON/完成收据，所有CLI参数具名。
    """
    if output.exists():raise FileExistsError('fresh output required')
    begin=time.perf_counter();state=load_state_checkpoint(input_state)
    with np.load(input_maps,allow_pickle=False) as m:maps={key:m[key].copy() for key in m.files}
    with np.load(input_regions,allow_pickle=False) as p:regions={key:p[key].copy() for key in p.files}
    _,_,geometry=fsio.read_geometry(str(input_surface),read_metadata=True)
    metadata=dict(use_real_ras=0,status=0,origxyz_status=0,
        hemisphere=0 if hemisphere=='lh' else 1,volume_geometry=dict(geometry))
    if not np.array_equal(regions['merged_labels'],maps['original_defect_labels']):
        raise ValueError('independent own regions and rawBase labels differ')
    if not np.array_equal(regions['merged_vertex_translation'],maps['vertex_translation']):
        raise ValueError('independent own merged mapping and rawBase mapping differ')
    inside=_rows(regions,'merged_vertices');border=_rows(regions,'merged_border')
    loading=time.perf_counter()-begin;results=[];compacts=[];finish=[];files=[]
    output.mkdir(parents=True)
    with thread_budget(threads=threads):
        for repeat in range(2):
            tick=time.perf_counter()
            before=compact_ripped_surface_elements(surface_state=state,metadata=metadata)
            before_wall=time.perf_counter()-tick
            tick=time.perf_counter()
            f=finish_topology_geometry(surface_state=state,defect_vertices=inside,border_vertices=border,
                vertex_translation=maps['vertex_translation'],normal_status='surface',iterations=iterations)
            finish_wall=time.perf_counter()-tick
            tick=time.perf_counter()
            c=compact_ripped_surface_elements(surface_state=f['surface_state'],metadata=metadata)
            compact_wall=time.perf_counter()-tick
            tick=time.perf_counter()
            idem=compact_ripped_surface_elements(surface_state=c['surface_state'],metadata=c['metadata'])
            idem_wall=time.perf_counter()-tick
            idempotence=compare_states(c['surface_state'],idem['surface_state'])
            cv=c['kept_source_vertices'];cf=c['kept_source_faces']
            own_field_check={name:dict(bitwise_equal=np.array_equal(
                c['surface_state'][name].view(np.uint32),np.asarray(f['surface_state'][name])[cv].view(np.uint32)))
                for name in ('current_vertices','original_vertices','temporary_vertices','canonical_vertices','vertex_normals')}
            face_same=np.array_equal(c['surface_state']['faces'],c['vertex_translation'][np.asarray(state['faces'])[:state['nfaces']][cf]])
            target=output/('compact.'+str(repeat)+'.private.npz')
            write=time.perf_counter();dump_state_checkpoint(target,c['surface_state']);files.append(target)
            write_wall=time.perf_counter()-write
            results.append(dict(repeat=repeat,before_compact_wall_seconds=before_wall,
                finish_wall_seconds=finish_wall,finish_subtimers=dict(metric=f['metric_geometry']['seconds'],soap=f['soap']['seconds']),
                after_compact_wall_seconds=compact_wall,idempotent_compact_wall_seconds=idem_wall,
                checkpoint_write_seconds=write_wall,idempotence=idempotence,
                remapping_geometry=own_field_check,face_remapping_exact=bool(face_same),
                kept_vertices=len(cv),kept_faces=len(cf),removed_vertices=c['removed_vertices'],removed_faces=c['removed_faces'],
                metric_pending_counts=f['metric_geometry']['pending_counts'],
                metric_changed_vertices_count=len(f['metric_geometry']['changed_vertices']),
                soap_iterations=f['soap']['iterations'],soap_moving_vertices=f['soap']['moving_vertices'],
                all_marks_cleared=bool(not f['surface_state']['vertex_marks'].any())))
            finish.append(f['surface_state']);compacts.append(c['surface_state'])
    finish_repeat=compare_states(finish[0],finish[1]);compact_repeat=compare_states(compacts[0],compacts[1])
    # Reuse the mature state comparator's byte_equal contract.  Geometry's
    # local remapping checks below intentionally use their own bitwise key.
    own_pass=all(row['byte_equal'] for group in (finish_repeat,compact_repeat) for row in group.values())
    own_pass &= all(r['face_remapping_exact'] and all(v['byte_equal'] for v in r['idempotence'].values())
                    and all(v['bitwise_equal'] for v in r['remapping_geometry'].values()) for r in results)
    modules=('topology_compact','topology_finish_geometry','topology_genetic_base','normal_pending_python','thread_budget')
    report=dict(status='own_real_pre_GA_base_finish_compact_passed' if own_pass else 'failed_own_stability',
        original_stage_strict_reproduction='not_assessed',completed_25_defects_input=False,
        scope='self-generated public real pre-GA base large-mesh compatibility; not source final compaction or final GA mesh',
        reference_after_consumed=False,full_ga_complete=False,final_mesh_quality='not_assessed',overall_metric_equivalence='not_assessed',
        results=results,finish_self_repeat=finish_repeat,compact_self_repeat=compact_repeat,
        nvertices_input=len(state['original_vertices']),nfaces_input=state['nfaces'],defect_regions=len(inside),
        iterations=iterations,loading_seconds=loading,import_seconds=_IMPORT,
        tool_wall_before_final_JSON_seconds=time.perf_counter()-begin,
        input_sha256={key:sha(path) for key,path in dict(state=input_state,maps=input_maps,regions=input_regions,
            surface=input_surface,input_manifest=input_manifest,source_manifest=source_manifest).items()},
        module_sha256={name:sha(sys.modules['fnit.recon_all.'+name].__file__) for name in modules},
        script_sha256=sha(__file__),checkpoint_sha256={f.name:sha(f) for f in files},
        threads=threads,CPU_affinity=sorted(os.sched_getaffinity(0)),host=platform.node(),
        shared_host=True,gpu_used=False,gpu_peak_bytes=None,independent_install_validation='not_assessed')
    (output/'report.private.json').write_text(json.dumps(report,default=_json,allow_nan=False,indent=2)+'\n')
    (output/'COMPLETE.json').write_text(json.dumps(dict(status=report['status'],report_sha256=sha(output/'report.private.json')),indent=2)+'\n')
    if not own_pass:raise RuntimeError('real pre-GA own stability/remapping gate failed')
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('input-state','input-maps','input-regions','input-surface','input-manifest','source-manifest','output'):
        parser.add_argument('--'+key,type=Path,required=True)
    parser.add_argument('--hemisphere',choices=('lh','rh'),required=True)
    parser.add_argument('--threads',type=int,default=4)
    parser.add_argument('--iterations',type=int,default=5)
    report=run(**vars(parser.parse_args()));print(json.dumps(dict(status=report['status'],seconds=report['tool_wall_before_final_JSON_seconds'])))


if __name__=='__main__':main()
