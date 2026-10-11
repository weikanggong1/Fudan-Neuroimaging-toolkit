"""冻结自产white/pial的只读穿越检查；不跑原始T1或读取官方表面。"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import resource
import sys
import time


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(*, installed_directory: Path, overlay_directory: Path, diagnostic_case: Path,
        input_hashes: Path, terminal_sha256: str, protocol: Path, output: Path,
        threads: int = 4) -> dict:
    """读取带SHA的七前置/末态，输出穿越、接触、空间、来源及墙钟数值。

    installed_directory是冻结安装包；overlay_directory仅含本轮helper。
    diagnostic_case含subject七前置和cleanup-result.private.npy，皆只读。
    input_hashes为七相对路径/SHA JSON；terminal_sha256固定自产末态。
    protocol为运行前固定的检测语义/后续候选约束，原样hash绑定，不
    用结果选参数。output须不存在；threads正整数默认4，只设置Numba。
    返回JSON无坐标/私人路径/主机标识；外层冷CLI另包括解释器加载和写入。
    不解码MRI体素、不优化顶点、不调用外部程序、不读参考坐标或初始化GPU。
    哈希、网格/标签域、协议、输出存在性不符则抛异常，不宣称质量通过。
    """
    started = time.perf_counter()
    if isinstance(threads,bool) or not isinstance(threads,int) or threads<1:
        raise ValueError('positive integer thread budget required')
    if output.exists(): raise FileExistsError(output)
    expected = json.loads(input_hashes.read_text())
    required = {'surf/lh.white','surf/autodet.gw.stats.lh.dat','label/lh.cortex.label',
                'label/lh.cortex+hipamyg.label','mri/brain.finalsurfs.mgz','mri/wm.mgz','mri/aseg.presurf.mgz'}
    if not isinstance(expected,dict) or set(expected)!=required:
        raise ValueError('the exact seven frozen pial input hashes required')
    before = {name:sha(diagnostic_case/'subject'/name) for name in sorted(required)}
    if before!=expected: raise ValueError('frozen seven inputs differ')
    terminal_path = diagnostic_case/'cleanup-result.private.npy'
    if sha(terminal_path)!=terminal_sha256: raise ValueError('frozen own terminal differs')
    settings = json.loads(protocol.read_text())
    if settings.get('coordinate_space')!='surface_RAS' or settings.get('proper_crossing_minimum_length_mm')!=0.0:
        raise ValueError('predeclared strict crossing protocol required')
    sys.path.insert(0,str(installed_directory))
    import fnit.recon_all
    fnit.recon_all.__path__.insert(0,str(overlay_directory))
    import numpy as np
    import nibabel.freesurfer as fs
    import nibabel
    import numba
    import scipy
    numba.set_num_threads(threads)
    from fnit.recon_all.white_pial_crossing_python import WhitePialCrossingIndex
    white,faces = fs.read_geometry(str(diagnostic_case/'subject/surf/lh.white'))
    white=white.astype(np.float32);faces=faces.astype(np.int32)
    terminal=np.load(terminal_path)
    if terminal.dtype!=np.float32 or terminal.shape!=white.shape or not np.isfinite(terminal).all():
        raise ValueError('finite float32 terminal on own ordered white mesh required')
    cortex=np.zeros(len(white),bool);rip=np.ones(len(white),bool)
    for name,target,value in (('lh.cortex.label',cortex,True),('lh.cortex+hipamyg.label',rip,False)):
        indices=fs.read_label(str(diagnostic_case/'subject/label'/name)).astype(np.int64)
        if indices.size and (indices.min()<0 or indices.max()>=len(white)):
            raise ValueError('label outside ordered vertex domain')
        target[indices]=value
    fixed=rip|~cortex
    index=WhitePialCrossingIndex(white_vertices=white,faces=faces,fixed_vertices=fixed)
    proper,contact,legal,summary=index.query(terminal)
    # 真实局部/全局同一集合回归，查询自产末态相交面的全部顶点关联面。
    from fnit.recon_all.place_surface_intersection_marking import mark_source_intersections
    marked,source_faces=mark_source_intersections(terminal,faces,predicate_backend='numba')
    affected=np.flatnonzero(marked[faces].any(axis=1))
    local=index.query(terminal,affected)
    selected_faces=set(map(int,affected))
    local_equal=[local[i]=={pair for pair in whole if pair[0] in selected_faces}
                 for i,whole in enumerate((proper,contact,legal))]
    if not all(local_equal): raise RuntimeError('local crossing/contact sets differ from full restriction')
    after={name:sha(diagnostic_case/'subject'/name) for name in sorted(required)}
    if before!=after or sha(terminal_path)!=terminal_sha256:
        raise RuntimeError('readonly frozen inputs changed')
    result={'schema_version':1,'execution_status':'complete_readonly_crossing_check',
            'scope':'frozen own pial terminal; no replacement or raw-T1 whole execution',
            'input_sha256':before,'input_hashes_unchanged':True,'terminal_sha256':terminal_sha256,
            'script_sha256':sha(__file__),'protocol':settings,'protocol_sha256':sha(protocol),
            'source_sha256':{name:sha(module.__file__) for name,module in sorted(sys.modules.items())
                             if name.startswith('fnit.recon_all.') and getattr(module,'__file__',None) and str(module.__file__).endswith('.py')},
            'mesh':{'vertices':len(white),'faces':len(faces),'fixed_vertices':int(fixed.sum()),
                    'outside_cortex_elements_different_from_white_at_entry':int(np.count_nonzero(terminal[~cortex]!=white[~cortex]))},
            'initial_crossing_qc':summary,'local_query':local[3],
            'local_full_proper_contact_legal_sets_equal':local_equal,
            'source_self_intersecting_faces':int(source_faces),'affected_faces':len(affected),
            'proper_crossing_pair_set_sha256':hashlib.sha256(json.dumps(sorted(proper),separators=(',',':')).encode()).hexdigest(),
            'exact_contact_pair_set_sha256':hashlib.sha256(json.dumps(sorted(contact),separators=(',',':')).encode()).hexdigest(),
            'legal_fixed_contact_pair_set_sha256':hashlib.sha256(json.dumps(sorted(legal),separators=(',',':')).encode()).hexdigest(),
            'versions':{'python':platform.python_version(),'numpy':np.__version__,'numba':numba.__version__,
                        'nibabel':nibabel.__version__,'scipy':scipy.__version__},
            'threads':threads,'cpu_affinity':sorted(os.sched_getaffinity(0)),
            'cuda_visible_devices':os.environ.get('CUDA_VISIBLE_DEVICES'),'gpu_computation':False,
            'gpu_memory_bytes':None,'original_stage_program_called':False,
            'overall_metric_equivalence':'not_assessed','whole_pipeline_speedup':None,
            'initial_nonzero_crossings_are_not_a_no_new_gate_pass':True,
            'peak_process_cpu_rss_bytes':int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024),
            'wall_seconds_before_report_write':time.perf_counter()-started}
    output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('installed-directory','overlay-directory','diagnostic-case','input-hashes','protocol','output'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--terminal-sha256',required=True);p.add_argument('--threads',type=int,default=4)
    result=run(**vars(p.parse_args()))
    print(json.dumps({'status':result['execution_status'],'proper_crossings':result['initial_crossing_qc']['proper_crossing_pairs']}))


if __name__=='__main__':main()
