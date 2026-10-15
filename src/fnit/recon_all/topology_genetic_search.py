"""连接自产单缺陷边输入、完整候选闭环、种群搜索和最终写回。"""
from __future__ import annotations

from collections.abc import Mapping
import time
import numpy as np

from .gcsa_permutation import VnlRandom
from .topology_candidate_cycle import CandidateCycleEvaluator, _one_ring_state
from .topology_candidate_state import record_candidate_state
from .topology_retessellation_dynamic import prepare_retessellation_base
from .topology_face_detection_cache import prepare_face_detection_context
from .topology_ga_population import run_population_search
from .topology_ga_finalize import finalize_defect_patch


class TopologyRandomStream:
    """复用已验证VNL流并做double区间缩放；一个半球只初始化一次。

    audit_callback默认None；可显式具名(index,low,high,value)观测每次缩放后的
    double调用，返回值忽略，异常传播，不记录prime、不改变生成器或调用顺序。
    seed为有符号32位整数，先消费setRandomSeed对应的一次OpenRan1。
    实例以具名或位置low/high调用，必须是有限实数且low<=high；
    即使区间宽零也消费一次随机量，返回Python double。draws只计
    prime之后的调用。不用NumPy RNG、不重置每代或每个缺陷、不改
    全局随机状态；非法seed/区间ValueError。不保证调用方此前阶段
    没有消费随机数；需要共享流时显式传同一实例。无独立原CLI。
    """
    def __init__(self, *, seed, audit_callback=None):
        if audit_callback is not None and not callable(audit_callback):
            raise ValueError("audit_callback must be callable or None")
        self.audit_callback = audit_callback
        a = np.asarray(seed)
        if a.ndim or a.dtype.kind not in 'iu' or not -(1 << 31) <= int(a) < (1 << 31):
            raise ValueError('explicit signed int32 seed required')
        self.generator = VnlRandom(int(a))
        self.generator.open_ran1()
        self.seed = int(a)
        self.draws = 0

    def __call__(self, low, high):
        bounds = []
        for value in (low, high):
            a = np.asarray(value)
            if a.ndim or a.dtype.kind not in 'iuf' or not np.isfinite(a):
                raise ValueError('finite scalar real random bounds required')
            bounds.append(float(a))
        lo, hi = bounds
        if hi < lo:
            raise ValueError('random high must not be below low')
        value = self.generator.open_ran1()
        self.draws += 1
        scaled = lo + (hi-lo)*value
        if self.audit_callback is not None:
            self.audit_callback(index=self.draws-1, low=lo, high=hi, value=scaled)
        return scaled


def source_population_limits(*, edge_count, population_size, max_unchanged):
    """完整ETable计数→源大缺陷种群裁剪；不裁掉候选边或精度。

    三项为非bool整数：edge_count>=0，population_size为1..1000，
    max_unchanged>=0。默认值由上层明确提供。边数>200000用至多3个
    patch和1代未改善；>100000用至多10和1；>50000使用原整除减半
    patch、整除5未改善门。返回两项整数；小缺陷保持调用者值。
    非法输入ValueError，不运行算法/修改全局状态；源1个patch且
    仍生成下一代的未定义分支由成熟种群控制器明确拒绝。
    """
    values=[]
    for value, lo, hi in ((edge_count,0,np.iinfo(np.int32).max),
                         (population_size,1,1000),(max_unchanged,0,np.iinfo(np.int32).max)):
        a=np.asarray(value)
        if a.ndim or a.dtype.kind not in 'iu' or not lo<=int(a)<=hi:
            raise ValueError('valid explicit integer population limits required')
        values.append(int(a))
    n,p,u=values
    if n>200000:p=max(min(p,3),1);u=min(u,1)
    elif n>100000:p=max(min(p,10),1);u=min(u,1)
    elif n>50000:p=max(p//2,1);u=u//5
    return dict(population_size=p,max_unchanged=u)


def _selection_groups(defect_inputs, edge_count):
    """ETable域分组→原ES域行；零旧边返回源NULL对应的None/None。

    空ES必须同时为零行分组，对应segmentIntersectingEdges(nes=0)在
    分配前返回NULL。其他输入保留全部空槽；非空ES零有效组由排序器
    报错，不能转换为NULL或跳过候选。两个索引域不能混用。
    """
    old=np.asarray(defect_inputs['old_edge_table_indices'])
    if (old.ndim!=1 or old.dtype.kind not in 'iu' or np.any(old<0)
            or np.any(old>=edge_count) or len(np.unique(old))!=len(old)):
        raise ValueError('complete unique ES-to-ETable mapping required')
    inverse=np.full(edge_count,-1,np.int32);inverse[old]=np.arange(len(old),dtype=np.int32)
    groups=[]
    for row in defect_inputs['segment_rows']:
        a=np.asarray(row)
        if a.size==0 and isinstance(row,(tuple,list)):a=a.astype(np.int32)
        if a.ndim!=1 or a.dtype.kind not in 'iu' or np.any(a<0) or np.any(a>=edge_count):
            raise ValueError('ordered segment rows must use ETable indices')
        mapped=inverse[a]
        if np.any(mapped<0):raise ValueError('segment references an edge outside the old ES')
        groups.append(mapped.astype(np.int32,copy=True))
    if len(old)==0:
        # Fixed d932 source, segmentIntersectingEdges: nes == 0 returns NULL.
        # No SEGMENTATION/max_segments slots exist in that source branch.
        if groups:raise ValueError('zero old ES requires zero segment rows')
        return None,None
    return old.astype(np.int32,copy=True),groups


def prepare_genetic_defect_evaluator(*, surface_state, defect_inputs, candidate_inputs,
        face_detection_cache=False, blocker_scan_backend='python', distance_backend='serial', distance_device=None, state_backend="legacy"):
    """自产当前corrected完整状态/有序边/先验→源DVS及可复用评分回调。

    surface_state为完整一环state，含同域canonical_vertices；shape、空间
    和持久字段与candidate cycle一致。defect_inputs是自产ordered_genetic_
    defect_inputs输出；边、inside/border/hull和DISCARD已在corrected域，
    不将原域再错误translation。candidate_inputs为完整自产MRI评分字典，
    本函数不加载文件、不推测缺少初态，不接收参考路径。前三参必填。
    face_detection_cache默认False，仅接受bool/np.bool_；True时为本缺陷完整
    保留面准备一次独立不可变索引，候选评分及最终几何共用同一对象。
    下一缺陷重新准备，不缓存动态邻接、面判定或候选结果。
    blocker_scan_backend默认python；仅接受python/numba。numba将同一成熟
    球面边谓词的完整首阻挡顺序扫描放入编译循环，无自动回退或fastmath。
    搜索、最终评分及最终几何共用同一显式选择，不改变dtype或判定规则。
    state_backend默认legacy，显式packed需任务5完整状态API，搜索、最终评分与
    几何发布均传同一选择；缺实现明确失败，不退回legacy，不启用旧缓存。
    distance_device默认None；GPU后端必须显式可用cuda:N，CPU选择不接设备。
    GPU仅使用已验证的gpu_defect_signed_distance，失败不回退；原CPU路径保留。
    distance_backend默认serial，接受serial/parallel/gpu_defect_signed_distance；复制candidate_inputs
    后用本具名参数选择距离核，搜索及最终评分共用。不修改调用方字典，
    不更改Numba线程mask；parallel预算须由调用方thread_budget明确限制。
    最终纯几何不执行距离；unMRI跳过状态由实际候选distance_policy报告。

    保留当前logical faces和实际源邻接，生成一次不可变retess缓存与
    DVS（profile=chull）。返回evaluator、初始分组与overlap控制参数、
    prepare_seconds；开启overlap消费自产CSR，关闭显式disabled但给
    种群接口空CSR，不称真实重叠完整计算。此处不评候选/改输入。
    非法类型/索引/域或未知模式ValueError，子函数异常传播。
    """
    start=time.perf_counter()
    from .topology_ga_backend_policy import validate_state_backend, state_backend_arguments
    validate_state_backend(state_backend)
    if not isinstance(face_detection_cache,(bool,np.bool_)):
        raise ValueError('explicit bool face_detection_cache required')
    if not isinstance(blocker_scan_backend,str) or blocker_scan_backend not in ('python','numba'):
        raise ValueError('blocker_scan_backend must be python or numba')
    from .topology_distance_policy import validate_distance_policy
    distance_device=validate_distance_policy(distance_backend=distance_backend,distance_device=distance_device)
    n,top,nf=_one_ring_state(surface_state)
    if not isinstance(defect_inputs,Mapping) or not isinstance(candidate_inputs,Mapping):
        raise ValueError('explicit self-generated defect and candidate mappings required')
    canonical=np.asarray(surface_state.get('canonical_vertices'))
    if canonical.shape!=(n,3) or canonical.dtype!=np.float32 or not np.isfinite(canonical).all():
        raise ValueError('complete same-domain float32 canonical_vertices required')
    edges=np.asarray(defect_inputs['edge_vertices']);count=len(edges)
    numbers,groups=_selection_groups(defect_inputs,count)
    active=defect_inputs['use_overlap']
    if not isinstance(active,(bool,np.bool_)):
        raise ValueError('explicit bool use_overlap policy required')
    if active:
        overlap=dict(use_overlap=True,overlap_offsets=np.asarray(defect_inputs['overlap_offsets']).copy(),
            overlapping_edges=np.asarray(defect_inputs['overlap_values']).copy(),
            overlap_list_incomplete=~np.asarray(defect_inputs['overlap_complete'],np.bool_))
        rows=defect_inputs['overlaps'];complete=defect_inputs['overlap_complete']
    else:
        overlap=dict(use_overlap=False,overlap_offsets=np.zeros(count+1,np.int64),
            overlapping_edges=np.empty(0,np.int32),overlap_list_incomplete=np.zeros(count,np.bool_))
        rows=complete=None
    prepared=prepare_retessellation_base(canonical=canonical,
        base_faces=np.asarray(surface_state['faces'])[:nf],
        neighbors=[np.asarray(row)[:degree]for row,degree in zip(surface_state['neighbor_storage'],top[:,3])])
    r=dict(canonical=prepared.canonical,base_faces=prepared.base_faces,neighbors=prepared.neighbors,
        base_context=prepared,candidate_edges=edges.copy(),edge_used=np.asarray(defect_inputs['edge_used']).copy(),
        ordering=np.asarray(defect_inputs['ordering']).copy(),
        inside_vertices=np.asarray(defect_inputs['inside_vertices']).copy(),
        inside_discarded=np.asarray(defect_inputs['inside_discarded']).copy(),
        border_vertices=np.asarray(defect_inputs['border_vertices']).copy(),
        hull_vertices=np.asarray(defect_inputs['hull_vertices']).copy(),
        vertex_rip=np.asarray(surface_state['vertex_rip'])!=0,fixed_mapping=None,
        mode='all',overlaps=rows,overlap_complete=complete,
        blocker_scan_backend=blocker_scan_backend)
    if face_detection_cache:
        r['face_detection_context']=prepare_face_detection_context(
            base_faces=prepared.base_faces,vertex_count=n)
    snapshot=record_candidate_state(surface_state=surface_state,defect_vertices=r['inside_vertices'],
        convex_hull_vertices=r['hull_vertices'],border_vertices=r['border_vertices'],
        vertex_translation=np.arange(n,dtype=np.int32),profile='chull')
    selected_inputs=dict(candidate_inputs,distance_backend=distance_backend,distance_device=distance_device)
    evaluator=CandidateCycleEvaluator(surface_state=surface_state,snapshot=snapshot,
        retessellation_inputs=r,candidate_inputs=selected_inputs,
        **state_backend_arguments(state_backend=state_backend,target=CandidateCycleEvaluator))
    return dict(evaluator=evaluator,segment_edge_numbers=numbers,segment_rows=groups,
        overlap_arguments=overlap,overlap_policy='computed'if active else'disabled',
        prepare_seconds=time.perf_counter()-start,
        face_detection_cache=bool(face_detection_cache),blocker_scan_backend=blocker_scan_backend,
        distance_backend=distance_backend,distance_device=distance_device,state_backend=state_backend)


def search_and_finalize_defect(*, surface_state, defect_inputs, candidate_inputs,
        draw_random_number, population_size, max_unchanged, niters,
        initial_selection, vertex_eliminate, deletion_count, elimination_threshold,
        check_surface_intersection, intersection_checker, face_detection_cache=False,
        blocker_scan_backend='python', distance_backend='serial', distance_device=None, state_backend="legacy"):
    """一个自产完整缺陷→完整种群搜索、连续持久状态与最优最终几何。

    前三个输入与prepare_genetic_defect_evaluator一致。RNG回调必须
    为连续callable(low,high)→double；同一半球跨defect共享，不能每次
    重置seed。显式population_size/max_unchanged/niters/initial_selection/
    vertex_eliminate和跨缺陷deletion_count/float32 elimination_threshold
    与成熟种群接口相同；源标准值10/10/−1/True/True、首1/4由调用方
    明确提供。源大缺陷裁剪先计算。可选相交策略同finalize接口。
    face_detection_cache默认False，blocker_scan_backend默认python，其余全参必填；策略与prepare
    相同，完整API耗时包括工厂。RP每个新defect按源calloc独立零初始化，
    只在此边界做。
    distance_backend默认serial，显式parallel仅改变已有距离核的x slab
    执行分配。返回requested选择及最终评分实际distance_policy；跳过
    unMRI时实际backend为None，不将请求等同执行，不观测OS线程数。

    返回完整最终surface_state、search、finalization、跨缺陷删除状态、
    prepare/search/finalization及完整API秒数、actual_policy。保留连续
    K/H/E/fx/curvbak、RP和源DVS选择性恢复，不从参考或初态重建每个
    candidate。输入不改，不调用原版，不改CUDA/精度/全局线程。
    缺状态/类型/未定义分支ValueError，评分/RNG/子核异常传播。
    只完成单缺陷；全部缺陷后的SOAP/压缩/相交/网格QC另行执行，
    complete_ga=False，整体指标等效not_assessed，不生成标准文件。
    """
    start=time.perf_counter()
    if not callable(draw_random_number):raise ValueError('explicit continuous random callback required')
    prepared=prepare_genetic_defect_evaluator(surface_state=surface_state,
        defect_inputs=defect_inputs,candidate_inputs=candidate_inputs,
        face_detection_cache=face_detection_cache,blocker_scan_backend=blocker_scan_backend,
        distance_backend=distance_backend,distance_device=distance_device,state_backend=state_backend)
    evaluator=prepared['evaluator'];r=evaluator.retessellation_inputs
    limits=source_population_limits(edge_count=len(r['candidate_edges']),
        population_size=population_size,max_unchanged=max_unchanged)
    tick=time.perf_counter()
    search=run_population_search(edge_count=len(r['candidate_edges']),**limits,niters=niters,
        initial_selection=initial_selection,segment_edge_numbers=prepared['segment_edge_numbers'],
        segment_rows=prepared['segment_rows'],**prepared['overlap_arguments'],
        draw_random_number=draw_random_number,evaluate_patch=evaluator,
        translated_defect_vertices=r['inside_vertices'],discarded_defect_vertices=r['inside_discarded'],
        vertex_rip=np.asarray(evaluator.surface_state['vertex_rip'],np.int32),
        use_counts=np.zeros(len(r['inside_vertices']),np.int32),
        vertex_fitness=np.zeros(len(r['inside_vertices']),np.float32),
        vertex_eliminate=vertex_eliminate,deletion_count=deletion_count,
        elimination_threshold=elimination_threshold)
    search_seconds=time.perf_counter()-tick
    # Elimination can occur after the last evaluation. Preserve its rip state,
    # then the finalizer resets only this defect to its recorded best status.
    state=dict(evaluator.surface_state);state['vertex_rip']=search['vertex_rip'].copy()
    tick=time.perf_counter()
    final=finalize_defect_patch(surface_state=state,snapshot=evaluator.snapshot,
        retessellation_inputs=evaluator.retessellation_inputs,candidate_inputs=evaluator.candidate_inputs,
        best_ordering=search['best_ordering'],best_discarded=search['best_discarded'],
        best_fitness=search['best_fitness'],use_counts=search['use_counts'],vertex_fitness=search['vertex_fitness'],
        check_surface_intersection=check_surface_intersection,intersection_checker=intersection_checker,
        state_backend=state_backend)
    final_seconds=time.perf_counter()-tick
    return dict(surface_state=final['surface_state'],search=search,finalization=final,
        deletion_count=search['deletion_count'],elimination_threshold=search['elimination_threshold'],
        actual_policy=dict(**limits,niters=niters,initial_selection=initial_selection,
            vertex_eliminate=vertex_eliminate,overlap=prepared['overlap_policy']),
        seconds=dict(prepare=prepared['prepare_seconds'],search=search_seconds,
            finalization=final_seconds,complete_api=time.perf_counter()-start),
        face_detection_cache=bool(face_detection_cache),blocker_scan_backend=blocker_scan_backend,
        distance_backend=distance_backend,distance_device=distance_device,state_backend=state_backend,
        final_distance_policy=final.get('final_evaluation',{}).get('candidate',{}).get('distance_policy'),
        complete_ga=False,whole_surface_quality='not_assessed',overall_metric_equivalence='not_assessed')
