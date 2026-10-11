"""一环/all 模式候选状态闭环；复用几何、评分、RP 更新和选择性 DVS。

本模块不建立基底/全局先验或种群，不调用原版程序。中间候选评分
合法不代表最终表面通过质量门；外层 GA 与最终发布由调用者控制。
"""
from __future__ import annotations

from collections.abc import Mapping
import time
import numpy as np

from .topology_candidate_pipeline import evaluate_candidate
from .topology_candidate_state import DefectVertexSnapshot, restore_candidate_state
from .topology_candidate_statistics import update_candidate_statistics


def _one_ring_state(state):
    """只接受 source base/已 full-restore 的一环计数，不猜多环存储。"""
    if not isinstance(state, Mapping):
        raise ValueError('explicit complete surface state required')
    required=('original_vertices','vertex_normals','vertex_rip','vertex_marks',
              'vertex_topology','neighbor_storage','incident_faces','incident_corners',
              'faces','nfaces','curvbak','k1','k2','tangent_e1','tangent_e2',
              'flattened_vertices','face_area','vertex_area')
    if any(k not in state for k in required):
        raise ValueError('complete DVS and candidate attributes required')
    n=len(state['original_vertices']);t=np.asarray(state['vertex_topology'])
    if (t.shape!=(n,7) or t.dtype!=np.int32 or np.any(t[:,:2]<0)
            or np.any(t[:,:2]>1) or np.any(t[:,0]>t[:,1])
            or np.any(t[:,2]!=t[:,3])
            or np.any(t[t[:,0]==0,2]!=0)):
        raise ValueError('validated nsizeCur/Max 0 or 1 and vtotal=vnum required')
    for rows,count,name in [('neighbor_storage',t[:,3],'neighbors'),
                            ('incident_faces',t[:,6],'incident faces'),
                            ('incident_corners',t[:,6],'incident corners')]:
        if len(state[rows])!=n or any(len(row)<int(size) for row,size in zip(state[rows],count)):
            raise ValueError('complete ordered '+name+' storage required')
    nf=state['nfaces'];f=np.asarray(state['faces'])
    if (isinstance(nf,(bool,np.bool_)) or not isinstance(nf,(int,np.integer))
            or nf<0 or nf>len(f) or f.ndim!=2 or f.shape[1:]!=(3,)
            or f.dtype.kind not in 'iu'):
        raise ValueError('explicit logical nfaces and full face capacity required')
    return n,t,int(nf)


def _candidate_state(*,initial,result):
    """已有完整候选→源 all-mode 元数据；只改真实加边及追加面字段。"""
    validated=_one_ring_state(initial)
    return _candidate_state_validated(initial=initial,result=result,validated=validated)


def _candidate_state_validated(*,initial,result,validated):
    """同一次闭环内复用已完成的一环校验；不跨候选保存校验结果。

    仅由本模块调用。validated为评分前_one_ring_state的(n,topology,nfaces)，
    evaluate_candidate不修改initial。仍完整核对候选面、邻接和入射前缀；
    独立公开装配入口先校验，显式state_context路径也保留自身完整校验。
    """
    n,old,nf=validated;g=result['geometry'];rows=g['neighbors']
    if len(rows)!=n or not np.array_equal(g['faces'][:nf],np.asarray(initial['faces'])[:nf]):
        raise ValueError('candidate must preserve complete ordered base face prefix')
    if len(g['removed_vertices']):
        raise ValueError('all-mode candidate unexpectedly removed a vertex')
    top=old.copy();neighbors=[];incidents=[];corners=[]
    # Source rows are short ordered integer arrays. Python list equality uses
    # the same elementwise value comparison without allocating a NumPy boolean
    # array and reducing it separately for each of 3*N rows. The unusual
    # non-vector case retains NumPy's existing exact shape/value semantics.
    def same_row(a,b):
        if a.ndim==b.ndim==1:
            return a.tolist()==b.tolist()
        return np.array_equal(a,b)
    all_incidents=np.asarray(g['incident_faces'],np.int32)
    all_corners=np.asarray(g['incident_corners'],np.uint8)
    for v,row in enumerate(rows):
        row=np.asarray(row,np.int32)
        oldrow=np.asarray(initial['neighbor_storage'][v])[:old[v,3]]
        if len(row)<len(oldrow) or not same_row(row[:len(oldrow)],oldrow):
            raise ValueError('all-mode adjacency must preserve source row and append only')
        # mrisAddEdge: modVnum then vtotal=vnum, nsizeCur=nsizeMax=1.
        # v2num/v3num are not changed by this operation.
        if len(row)!=len(oldrow):
            top[v,0:2]=1;top[v,2:4]=len(row)
            neighbors.append(row.copy())
        else:
            neighbors.append(initial['neighbor_storage'][v])
        a,b=map(int,g['incident_offsets'][v:v+2])
        inc=all_incidents[a:b]
        cor=all_corners[a:b]
        oldinc=np.asarray(initial['incident_faces'][v])[:old[v,6]]
        oldcor=np.asarray(initial['incident_corners'][v])[:old[v,6]]
        if (len(inc)<len(oldinc) or not same_row(inc[:len(oldinc)],oldinc)
                or not same_row(cor[:len(oldcor)],oldcor)):
            raise ValueError('source append-only ordered face/corner incidence required')
        top[v,6]=len(inc);incidents.append(inc.copy());corners.append(cor.copy())
    return _publish_candidate_attributes(initial=initial,result=result,top=top,
        neighbors=neighbors,incidents=incidents,corners=corners,n=n,nf=nf)


def _publish_candidate_attributes(*,initial,result,top,neighbors,incidents,corners,n,nf):
    """两种装配入口复用原属性写回；不缓存动态曲率/反馈/坐标。"""
    g=result['geometry']
    output=dict(initial)
    output.update(original_vertices=g['original_vertices'],vertex_normals=result['vertex_normals'],
        faces=g['faces'],nfaces=len(g['faces']),face_area=result['face_area'],
        vertex_topology=top,neighbor_storage=neighbors,incident_faces=incidents,incident_corners=corners,
        vertex_marks=np.asarray(g['marks'],np.int32).copy(),
        flattened_vertices=result['flattened_vertices'],curvbak=result['vertex_curvbak'],
        k1=result['sff']['k1'],k2=result['sff']['k2'],
        tangent_e1=result['sff']['tangent_e1'],tangent_e2=result['sff']['tangent_e2'])
    if 'face_rip' in initial:
        output['face_rip']=np.concatenate((np.asarray(initial['face_rip'])[:nf],
            np.zeros(len(g['faces'])-nf,np.asarray(initial['face_rip']).dtype)))
    # These are the computed cache normals, kept separate from signed FACE.area.
    if 'face_normals' in g:output['face_normals']=g['face_normals']
    # Source SFF writes H/K only after a nonempty local fit. Do not update
    # zero-vtotal/skipped rows or synthesize missing initial scalar fields.
    if 'fit_kind' in result['sff']:
        fitted=np.asarray(g['selected_vertices'])[np.asarray(result['sff']['fit_kind'])!=0]
        for key in ('H','K'):
            if key not in initial:continue
            values=np.asarray(initial[key]).copy()
            if values.dtype!=np.float32 or values.shape!=(n,):
                raise ValueError('explicit float32 initial H/K state required')
            k1=np.asarray(output['k1'])[fitted];k2=np.asarray(output['k2'])[fitted]
            values[fitted]=(k1*k2 if key=='K' else (k1+k2)/np.float32(2)).astype(np.float32)
            output[key]=values
    return output


def materialize_candidate_state(*, initial, result, state_context=None):
    """已有 complete candidate→完整新状态；不重新评分、统计或恢复。

    initial/result 分别是原 full 一环 state 和 evaluate_candidate 返回。
    复用同一装配核心，保持有序 base 前缀、追加邻接和 incidence；
    保留未评分属性/currentXYZ，更新真实 K/E/fx/curvbak 和已存在的
    H/K（仅非空 SFF rows）、面 rip 容量及面法向缓存。面积缓存不
    替代 FACE.area。state_context 默认 None 走旧完整装配；显式传
    prepare_candidate_state_context 的同基底上下文时全行校验后复用
    批量前缀检查及独立行视图。无文件/线程切换；缺状态或未支持
    动态删除/多环明确 ValueError。结果尚不是 final mesh QA。
    """
    if state_context is not None:
        from .topology_candidate_state_cache import materialize_candidate_state_cached
        return materialize_candidate_state_cached(initial=initial,result=result,context=state_context)
    return _candidate_state(initial=initial,result=result)


def evaluate_candidate_cycle(*, surface_state, snapshot, retessellation_inputs,
        candidate_inputs, use_counts, vertex_fitness, state_context=None):
    """完整显式初态→一次评分/RP/原 DVS 恢复/边复位，返回新状态。

    surface_state 包含原 full base producer/DVS 的完整字段和源计数；
    坐标为 float32(N,3) surface RAS/mm。snapshot 为同顶点版本原序
    record_candidate_state 结果，不能来自不同网格或参考 after。
    retessellation_inputs 为成熟重连全部具名参数，限 mode='all'；
    当前 logical faces 必须与 base_faces 逐元素一致。candidate_inputs
    为 evaluate_candidate 的 MRI/校准/先验/权重关键字（不含两个 state
    参数）。state_context 默认 None 保持旧路径，显式只缓存完整
    有序基底拓扑；每次仍检查当前全部有效行/面前缀，动态属性不缓存。
    RP use_counts int32(K,)、vertex_fitness float32(K,) 与
    inside 映射严格同序。没有自动零初态、默认参考或隐式种群。

    调用成熟完整评分→成熟 update_candidate_statistics→成熟 full DVS
    restore，再将 NEW3→NOT0、BOTH4→ORIGINAL2。DVS 不恢复 K/E/fx/
    curvbak/RP/rip/marks/面内容；logical nfaces 回缩，追加面容量保留。
    返回 surface_state、edge_used、use_counts、vertex_fitness、完整
    candidate、statistics/restore 报告及耗时。异常不发布新状态，
    输入不修改；nsize>1、some 模式或不完整存储明确 ValueError。
    中间候选的 incidence penalty 与最终质量门分开，不执行最终写出。
    """
    start=time.perf_counter();validated=_one_ring_state(surface_state);nf=validated[2]
    if not isinstance(snapshot,DefectVertexSnapshot) or not isinstance(candidate_inputs,Mapping):
        raise ValueError('explicit own DVS snapshot and candidate parameter mapping required')
    r=dict(retessellation_inputs)
    if r.get('mode','all')!='all':
        raise ValueError('only verified source all-mode cycle supported')
    if (snapshot.nfaces!=nf or not np.array_equal(np.asarray(surface_state['faces'])[:nf],r['base_faces'])):
        raise ValueError('same recorded logical base faces required')
    initial=dict(surface_state)
    initial['faces']=np.asarray(initial['faces'])[:nf]
    initial['face_area']=np.asarray(initial['face_area'])[:nf]
    tick=time.perf_counter()
    result=evaluate_candidate(surface_state=initial,retessellation_inputs=r,**dict(candidate_inputs))
    times=dict(candidate_complete=time.perf_counter()-tick)
    # The complete candidate does not mutate its initial state. Reuse this
    # call's validated base metadata instead of checking the same 3*N row
    # capacities twice. Each next candidate and each independent public
    # materialization still validates its current state from scratch.
    if state_context is None:
        current=_candidate_state_validated(initial=surface_state,result=result,validated=validated)
    else:
        current=materialize_candidate_state(initial=surface_state,result=result,state_context=state_context)
    tick=time.perf_counter()
    score=result['fitness']['composition']['fitness']
    stats=update_candidate_statistics(vertex_marks=current['vertex_marks'],vertex_curvbak=current['curvbak'],
        candidate_edges=r['candidate_edges'],edge_used=result['geometry']['edge_used'],
        translated_defect_vertices=r['inside_vertices'],discarded_defect_vertices=r['inside_discarded'],
        translated_border_vertices=r['border_vertices'],use_counts=use_counts,vertex_fitness=vertex_fitness,
        candidate_fitness=score)
    current['vertex_marks']=stats['vertex_marks'];times['statistics_update']=time.perf_counter()-tick
    tick=time.perf_counter();restored,restore=restore_candidate_state(surface_state=current,snapshot=snapshot,mode='full')
    times['selective_full_restore']=time.perf_counter()-tick
    used=np.asarray(result['geometry']['edge_used']).copy()
    used[used==3]=0;used[used==4]=2
    times['complete_cycle']=time.perf_counter()-start
    return dict(surface_state=restored,edge_used=used,use_counts=stats['use_counts'],vertex_fitness=stats['vertex_fitness'],
        fitness=float(score),candidate=result,statistics=stats,restore=restore,seconds=times,
        complete_ga=False,final_mesh_quality='not_assessed')


class CandidateCycleEvaluator:
    """串行 GA 回调适配器；成功才发布状态，不重做基底或冻结初始化。

    构造参数与 evaluate_candidate_cycle 相同但没有 RP 数组；
    state_context 默认 None，可显式传同基底已验证上下文。调用仅
    接 ordering/discarded_defect_vertices/vertex_rip/use_counts/
    vertex_fitness 五关键字，返回同形反馈与 fitness；state/K/E 等保留
    在实例中。外层删除后的 rip/status 必须由下一调用显式给定，不能
    在构造时冻结删除策略。只能同一线程按源顺序调用，不支持并发。
    """
    def __init__(self,*,surface_state,snapshot,retessellation_inputs,candidate_inputs,state_context=None):
        _one_ring_state(surface_state)
        self.surface_state=dict(surface_state);self.snapshot=snapshot
        self.retessellation_inputs=dict(retessellation_inputs);self.candidate_inputs=dict(candidate_inputs)
        self.last_result=None;self.state_context=state_context

    def __call__(self,*,ordering,discarded_defect_vertices,vertex_rip,use_counts,vertex_fitness):
        rip=np.asarray(vertex_rip)
        if (rip.dtype.kind not in 'biu' or rip.shape!=np.asarray(self.surface_state['vertex_rip']).shape
                or rip.size and (rip.min() < -128 or rip.max() > 127)):
            raise ValueError('explicit full source signed-char integer/bool vertex_rip required')
        # Surface state preserves actual signed flag values. The retessellation
        # predicate asks only whether the flag is nonzero; do not return that
        # bool mask to the population controller in place of source storage.
        stored_rip=rip.astype(np.int32,copy=True)
        state=dict(self.surface_state);state['vertex_rip']=stored_rip
        args=dict(self.retessellation_inputs,ordering=np.asarray(ordering).copy(),
            inside_discarded=np.asarray(discarded_defect_vertices).copy(),vertex_rip=stored_rip!=0)
        result=evaluate_candidate_cycle(surface_state=state,snapshot=self.snapshot,
            retessellation_inputs=args,candidate_inputs=self.candidate_inputs,
            use_counts=use_counts,vertex_fitness=vertex_fitness,state_context=self.state_context)
        self.surface_state=result['surface_state']
        self.retessellation_inputs['edge_used']=result['edge_used']
        self.last_result=result
        return dict(fitness=result['fitness'],vertex_rip=np.asarray(result['surface_state']['vertex_rip']).astype(np.int32,copy=True),
            use_counts=result['use_counts'],vertex_fitness=result['vertex_fitness'])
