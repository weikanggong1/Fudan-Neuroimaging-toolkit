"""只读完整源相交查询的阳性面对，区分双面冻结与局部可动状态。"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time


def observe(*, frozen_source: Path, overlay_source: Path, diagnostic_case: Path,
            output_directory: Path, threads: int = 4) -> dict:
    """终态(N,3)float32 surface RAS/mm，完整源候选不裁剪；输出聚合JSON。

    frozen_source/overlay_source与完整cleanup诊断相同，diagnostic_case
    只读自产white/标签和terminal快照。新输出目录必须不存在；threads
    为正整数。用sys.setprofile在原evaluate返回时只读已完成的源双方向
    谓词和MHT桶结果，不更改判定/顺序。面对明细NPZ留服务器，不公开
    顶点或影像。原white另做同规则QC，用于诊断上游已存在的相交。
    """
    if threads < 1 or output_directory.exists():
        raise ValueError("positive threads and new output directory required")
    output_directory.mkdir(parents=True)
    wall=time.perf_counter();sys.path.insert(0,str(frozen_source))
    import fnit.recon_all
    fnit.recon_all.__path__.insert(0,str(overlay_source))
    import numpy as np
    import nibabel.freesurfer as fs
    import numba
    numba.set_num_threads(threads)
    from fnit.recon_all.place_surface_intersection_marking import mark_source_intersections
    white_path=diagnostic_case/'subject/surf/lh.white'
    white,faces=fs.read_geometry(str(white_path));white=white.astype(np.float32);faces=faces.astype(np.int32)
    terminal_path=diagnostic_case/'cleanup-result.private.npy'
    terminal=np.load(terminal_path)
    if terminal.shape!=white.shape or terminal.dtype!=np.float32:
        raise ValueError('ordered float32 terminal required')
    ripped=np.ones(len(white),bool)
    ripped[fs.read_label(str(diagnostic_case/'subject/label/lh.cortex+hipamyg.label')).astype(np.int64)]=False
    pairs=[];flags={}
    marker_file=str(Path(sys.modules[mark_source_intersections.__module__].__file__))
    def observer(frame,event,arg):
        if event!='return' or frame.f_code.co_filename!=marker_file:
            return
        if frame.f_code is mark_source_intersections.__code__:
            flags['face']=frame.f_locals['face_marks'].copy()
        elif frame.f_code.co_name=='evaluate':
            values=frame.f_locals
            first,second=values['first'],values['second']
            forward,reverse=values['forward'],values['reverse']
            buckets=values['bucket_cache']
            for slot in np.flatnonzero(forward|reverse):
                fa,fb=int(first[slot]),int(second[slot])
                if not buckets[fa].isdisjoint(buckets[fb]):
                    pairs.append((fa,fb,int(forward[slot]),int(reverse[slot])))
    tick=time.perf_counter();previous=sys.getprofile()
    try:
        sys.setprofile(observer)
        marks,count=mark_source_intersections(terminal,faces,predicate_backend='numba')
    finally:sys.setprofile(previous)
    seconds=time.perf_counter()-tick
    a=np.asarray(pairs,np.int64).reshape(-1,4)
    face_frozen=np.all(ripped[faces],axis=1)
    double_frozen=face_frozen[a[:,0]] & face_frozen[a[:,1]]
    one_frozen=face_frozen[a[:,0]] | face_frozen[a[:,1]]
    private=output_directory/'positive_pairs.npz';np.savez(private,pairs=a,face_flags=flags['face'])
    # 同规则检查前置白面；它不是额外pial清理的输入。
    tick=time.perf_counter();white_marks,white_count=mark_source_intersections(white,faces,predicate_backend='numba')
    result={'schema_version':1,'status':'complete_readonly_pairs',
            'scope':'full source widephase/narrowphase observer; no geometry change or additional cleanup',
            'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'source_marker_sha256':hashlib.sha256(Path(marker_file).read_bytes()).hexdigest(),
            'white_file_sha256':hashlib.sha256(white_path.read_bytes()).hexdigest(),
            'terminal_file_sha256':hashlib.sha256(terminal_path.read_bytes()).hexdigest(),
            'source_intersecting_faces':int(count),'source_marked_vertices':int(marks.sum()),
            'valid_positive_pairs':len(a),'pairs_with_both_faces_all_ripped':int(double_frozen.sum()),
            'pairs_with_any_face_all_ripped':int(one_frozen.sum()),
            'pairs_both_directions_positive':int(np.count_nonzero((a[:,2]!=0)&(a[:,3]!=0))),
            'positive_pair_artifact_sha256':hashlib.sha256(private.read_bytes()).hexdigest(),
            'terminal_qc_with_observer_seconds':seconds,'white_source_intersecting_faces':int(white_count),
            'white_source_marked_vertices':int(white_marks.sum()),'white_qc_seconds':time.perf_counter()-tick,
            'threads':threads,'gpu_computation':False,'gpu_memory_bytes':None,
            'diagnostic_wall_seconds':time.perf_counter()-wall,
            'original_source_return_and_geometry_unchanged':True,
            'producer_mesh_quality':'failed' if count else 'passed',
            'overall_metric_equivalence':'not_assessed','whole_pipeline_speedup':None}
    (output_directory/'report.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('frozen-source','overlay-source','diagnostic-case','output-directory'):
        p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--threads',type=int,default=4);a=p.parse_args()
    observe(frozen_source=a.frozen_source,overlay_source=a.overlay_source,diagnostic_case=a.diagnostic_case,
            output_directory=a.output_directory,threads=a.threads)

if __name__=='__main__':main()
