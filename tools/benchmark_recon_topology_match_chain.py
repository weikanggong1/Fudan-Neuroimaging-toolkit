"""首个真实缺陷：现有25次smooth/40次match的隔离与连续几何回归。"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import nibabel as nib
from benchmark_recon_topology_match_fields import load_match_capture
from fnit.recon_all.topology_defect_smooth import defect_smooth_type2
from fnit.recon_all.topology_defect_mri_match import defect_mri_match


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _compare(candidate, reference):
    delta = candidate.astype(np.float64)-reference.astype(np.float64)
    distance = np.linalg.norm(delta, axis=1)
    return dict(different_float_elements=int(np.count_nonzero(candidate.view(np.uint32)!=reference.view(np.uint32))),
                different_vertices=int(np.count_nonzero(distance)),
                max_mm=float(distance.max(initial=0)), p99_mm=float(np.percentile(distance, 99)),
                mean_mm=float(distance.mean()), exact=bool(np.array_equal(candidate.view(np.uint32),reference.view(np.uint32))))


def run(*, before, smooth, after, mri, output):
    """只用before自产25→40；另列冻结smooth输入的算子诊断。

    before/smooth/after为三phase源观测，仅before提供连续链原状态；smooth
    单独用于冻结同输入match测试并作为平滑事后比较；after只事后比较。
    mri是原brain.mgz，nibabel读三维uint8；output为不存在的新诊断目录。
    报告包含所有输入/源码SHA、完整有序顶点差异/冷与重复API墙钟，加载
    与写出另列，不称整例、最终GA质量或独占性能门。失败保留原文件。
    """
    start=time.perf_counter();before,smooth,after,mri,output=map(Path,(before,smooth,after,mri,output))
    if output.exists():raise FileExistsError('fresh smooth/match diagnosis required')
    a,b,c=[load_match_capture(capture_path=p) for p in (before,smooth,after)]
    if [int(f['header'][7]) for f in (a,b,c)] != [0,1,2]:raise ValueError('before/smooth/after source phases required')
    for key in ['faces','topology','neighbor_offsets','neighbors','incident_offsets','incident_faces','incident_corners','selected_vertices','selected_faces','parameters','ras_to_voxel']:
        if not all(np.array_equal(a[key],q[key]) for q in (b,c)):raise ValueError('source patch changed during match '+key)
    if tuple(a['header'][8:])!=(2,1):raise ValueError('current mature operators cover source smooth2/match1')
    image=nib.load(str(mri));volume=np.asanyarray(image.dataobj)
    if volume.ndim!=3 or volume.dtype!=np.uint8:raise ValueError('same source uint8 3D brain required')
    selected=a['selected_vertices'][:int(a['header'][6])]
    rows=[a['neighbors'][a['neighbor_offsets'][v]:a['neighbor_offsets'][v]+a['topology'][v,0]].tolist() for v in selected]
    loading_seconds=time.perf_counter()-start;output.mkdir(parents=True)
    result=[]
    for label in ['cold','repeat']:
        tick=time.perf_counter()
        smoothed=defect_smooth_type2(xyz=a['vertex_fields'][:,:3],faces=a['faces'],inside_vertices=selected,
            runtime_neighbors=rows,k1_mean=float(a['parameters'][2]),k2_mean=float(a['parameters'][3]))
        smooth_seconds=time.perf_counter()-tick
        tick=time.perf_counter()
        matched=defect_mri_match(xyz=smoothed,faces=a['faces'],inside_vertices=selected,
            runtime_neighbors=rows,volume=volume,ras_to_voxel=a['ras_to_voxel'],
            white_mean=float(a['parameters'][0]),gray_mean=float(a['parameters'][1]),iterations=40)
        chain_match_seconds=time.perf_counter()-tick
        tick=time.perf_counter()
        same_input=defect_mri_match(xyz=b['vertex_fields'][:,:3],faces=a['faces'],inside_vertices=selected,
            runtime_neighbors=rows,volume=volume,ras_to_voxel=a['ras_to_voxel'],
            white_mean=float(a['parameters'][0]),gray_mean=float(a['parameters'][1]),iterations=40)
        same_input_seconds=time.perf_counter()-tick
        tick=time.perf_counter();np.savez_compressed(output/(label+'.npz'),smooth=smoothed,chain=matched,match_frozen_input=same_input)
        result.append(dict(label=label,smooth_api_seconds=smooth_seconds,chain_match_api_seconds=chain_match_seconds,
            frozen_match_api_seconds=same_input_seconds,write_seconds=time.perf_counter()-tick,
            smooth=_compare(smoothed,b['vertex_fields'][:,:3]),continuous_25_to_40=_compare(matched,c['vertex_fields'][:,:3]),
            frozen_match=_compare(same_input,c['vertex_fields'][:,:3])))
    code=Path(__file__).resolve().parents[1]
    report=dict(scope='frozen same-input stage and candidate before→25smooth→40match, not whole GA/recon-all',
        header=a['header'].tolist(),parameters=a['parameters'].tolist(),loading_seconds=loading_seconds,
        algorithm_input_sha256={k:_sha(p) for k,p in [('before',before),('mri',mri)]},
        comparison_reference_sha256={'smooth':_sha(smooth),'after':_sha(after)},
        code_sha256={p:_sha(code/p) for p in ['tools/benchmark_recon_topology_match_chain.py','tools/benchmark_recon_topology_match_fields.py','src/fnit/recon_all/topology_defect_smooth.py','src/fnit/recon_all/topology_defect_mri_match.py']},
        results=result,report_write_not_in_api=True,total_before_report_seconds=time.perf_counter()-start)
    (output/'report.private.json').write_text(json.dumps(report,indent=2)+'\n')
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ['before','smooth','after','mri','output']:
        parser.add_argument('--'+name,type=Path,required=True)
    print(json.dumps(run(**vars(parser.parse_args())),indent=2))


if __name__=='__main__':main()
