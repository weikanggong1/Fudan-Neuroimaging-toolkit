"""自产3a完整merged域retention；DVS下游rip只在保存候选后比较。"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import time
import numpy as np
from benchmark_recon_topology_candidate_state import load_state_capture
from fnit.recon_all.topology_preflight_python import (
    defect_regions, defect_retention_status, defect_hull_labels,
)


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _difference(a, b):
    a, b = np.asarray(a), np.asarray(b)
    if a.shape != b.shape:
        return dict(shape_equal=False, byte_equal=False)
    error = np.abs(a.astype(np.float64)-b.astype(np.float64))
    return dict(shape_equal=True, dtype_equal=a.dtype==b.dtype,
                byte_equal=a.dtype==b.dtype and a.tobytes()==b.tobytes(),
                differing_values=int(np.count_nonzero(a!=b)),
                max_abs=float(error.max(initial=0)),
                p99_abs=float(np.percentile(error,99)) if error.size else 0.)


def run(*, own_inputs, original_surface, reference_state, reference_repeat,
        reference_program_sha256, output, threads):
    """自产raw/canonical/merged NPZ→全N状态/有序hull与独立下游状态诊断。

    own_inputs只含本链自产raw_vertices/完整faces/canonical_vertices、
    merged_labels/有序inside/borderCSR和vertex_translation。original_surface
    是同一真实nofix文件，用nibabel核完整有序XYZ/F不借参考补输入。
    reference_state/reference_repeat为隔离源码程序FNITDVS1第一record，
    只在两个候选保存后打开。原程序SHA必填64hex，跨主机生成明确记录。
    output必须新目录，threads正整数为Numba/BLAS预算，GPU不初始化。
    返回机器可读阶段报告，保存完整私有NPZ；不生产GA最终面、不改QC。
    必需字段、几何或域非法时抛异常并保留已写产物。没有原版调用。
    """
    start=time.perf_counter()
    output=Path(output)
    if output.exists():raise FileExistsError(output)
    if not isinstance(threads,int) or isinstance(threads,bool) or threads<1:
        raise ValueError('positive explicit thread budget required')
    if len(reference_program_sha256)!=64 or any(c not in '0123456789abcdef' for c in reference_program_sha256):
        raise ValueError('actual reference program SHA256 required')
    from fnit.recon_all.thread_budget import thread_budget
    import nibabel as nib
    with thread_budget(threads=threads):
        with np.load(own_inputs,allow_pickle=False) as z:
            data={key:z[key].copy() for key in z.files}
        raw,faces,canonical=(data[key] for key in ('raw_vertices','faces','canonical_vertices'))
        actual,actual_faces=nib.freesurfer.read_geometry(original_surface)
        if (not np.array_equal(actual.astype(np.float32),raw)
                or not np.array_equal(actual_faces,faces)):
            raise ValueError('own raw MRI-derived geometry does not match original nofix')
        regions=[]
        for a,b,c,d in zip(data['merged_vertices_offsets'][:-1],data['merged_vertices_offsets'][1:],
                           data['merged_border_offsets'][:-1],data['merged_border_offsets'][1:]):
            regions.append(dict(vertices=data['merged_vertices'][int(a):int(b)],
                                border=data['merged_border'][int(c):int(d)]))
        labels=data['merged_labels'];timing=dict(input_reading=time.perf_counter()-start)
        tick=time.perf_counter()
        status=defect_retention_status(canonical,raw,faces,labels,ordered_regions=regions)
        timing['first_complete_status_api']=time.perf_counter()-tick
        tick=time.perf_counter()
        repeated=defect_retention_status(canonical,raw,faces,labels,ordered_regions=regions)
        timing['repeat_complete_status_api']=time.perf_counter()-tick
        tick=time.perf_counter()
        # Only legacy component-BFS domains are used here. Their output remains
        # unchanged; source merged order is compared separately, never replaced.
        component_labels=data['defect_labels']
        legacy=defect_retention_status(canonical,raw,faces,component_labels)
        explicit_bfs=defect_retention_status(canonical,raw,faces,component_labels,
                                           ordered_regions=defect_regions(component_labels,faces))
        timing['default_bfs_compatibility_pair']=time.perf_counter()-tick
        tick=time.perf_counter()
        hull=defect_hull_labels(labels,faces,ordered_regions=regions)
        timing['complete_hull_api']=time.perf_counter()-tick
        output.mkdir(parents=True)
        tick=time.perf_counter()
        np.savez_compressed(output/'candidate.private.npz',retention=status,
            repeated_retention=repeated,hull_labels=hull,
            vertex_translation=data['merged_vertex_translation'],
            source_status=(status==-1).astype(np.int32))
        timing['candidate_writing']=time.perf_counter()-tick
        # The two native AFTER states first become visible at this point.
        reference,selection,info=load_state_capture(reference_state)
        reference2,selection2,info2=load_state_capture(reference_repeat)
        mapping=selection['vertex_translation']
        # FNITDVS1 captures the prefix through the first defect's largest
        # source index, not necessarily the entire raw-to-corrected mapping.
        if (mapping.ndim!=1 or len(mapping)>len(raw) or np.any(mapping<0)
                or np.any(mapping>=len(reference['vertex_rip']))):
            raise ValueError('reference translation is not a valid captured source prefix')
        reference_discard=reference['vertex_rip'][mapping].astype(np.int32)
        counts=[]
        for region in regions:
            inside=region['vertices']
            counts.append(dict(inside_vertices=len(inside),border_vertices=len(region['border']),
                retained=int(np.count_nonzero(status[inside]==1)),
                discarded=int(np.count_nonzero(status[inside]==-1))))
    checks=dict(self_repeat=_difference(status,repeated),
                default_bfs_compatibility=_difference(legacy,explicit_bfs),
                merged_source_mapping_prefix=_difference(data['merged_vertex_translation'][:len(mapping)],mapping),
                after_base_discard_prefix_diagnostic=_difference((status[:len(mapping)]==-1).astype(np.int32),reference_discard),
                reference_repeat_mapping=_difference(mapping,selection2['vertex_translation']),
                reference_repeat_rip=_difference(reference['vertex_rip'],reference2['vertex_rip']))
    source=Path(__file__).resolve().parents[1]/'src/fnit/recon_all/topology_preflight_python.py'
    report=dict(scope='same own-produced raw canonical merged input; retention then downstream base-rip diagnosis',
        declared_case='public ds000114 sub-06 historical3a LH; not 79a41cdd sub07',
        execution_complete=True,strict_interface_compatibility=checks['default_bfs_compatibility']['byte_equal'],
        strict_self_repeat=checks['self_repeat']['byte_equal'],checks=checks,
        vertices=len(raw),faces=len(faces),regions=counts,
        outside_vertices=int(np.count_nonzero(labels==0)),
        retained_vertices=int(np.count_nonzero(status==1)),discarded_vertices=int(np.count_nonzero(status==-1)),
        reference_phase=info,reference_repeat_phase=info2,
        source_phase_evidence='base constructor sets rip only for DISCARD; first DVS record is downstream; whole raw/base producer separately assessed',
        reference_generated_same_host=False,reference_after_used_as_input=False,
        reference_mapping_covered_source_vertices=len(mapping),
        reference_mapping_complete=len(mapping)==len(raw),
        complete_retention_reference_reproduction='not_assessed',
        input_sha256=dict(own_checkpoint=_sha(own_inputs),original_nofix=_sha(original_surface)),
        reference_sha256=dict(first=_sha(reference_state),repeat=_sha(reference_repeat),program=reference_program_sha256),
        source_sha256=dict(preflight=_sha(source),benchmark=_sha(__file__)),
        candidate_checkpoint_sha256=_sha(output/'candidate.private.npz'),
        seconds=dict(timing,full_script_before_report=time.perf_counter()-start),
        cpu_affinity=sorted(os.sched_getaffinity(0)),threads=threads,target_cuda='hidden',gpu_memory=None,
        complete_ga=False,final_mesh_quality='not_assessed',overall_metric_equivalence='not_assessed')
    (output/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    if not report['strict_interface_compatibility'] or not report['strict_self_repeat']:
        raise RuntimeError('retention compatibility/repeat changed; report retained')
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('own-inputs','original-surface','reference-state','reference-repeat','output'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--reference-program-sha256',required=True)
    p.add_argument('--threads',type=int,required=True)
    print(json.dumps(run(**vars(p.parse_args())),indent=2,allow_nan=False))


if __name__=='__main__':main()
