"""独立拓扑 GA 种群控制；复用排序、交叉、变异与剔除，评分显式注入。"""
from __future__ import annotations

from collections.abc import Sequence, Mapping
import math
import time
import numpy as np

from .topology_ga_ordering import crossover_ordering, mutate_ordering, _permutation
from .topology_ga_selection import plan_generation_selection, choose_distinct_parent, _draw
from .topology_fitness_search import rank_patch_fitness
from .topology_vertex_elimination import eliminate_worst_vertices


def _int(value, name, low, high):
    a=np.asarray(value)
    if a.ndim or a.dtype.kind not in 'iu' or not low<=int(a)<=high:
        raise ValueError(f'{name} requires integer in [{low},{high}]')
    return int(a)


def _overlap_arguments(n, use_overlap, offsets, edges, incomplete):
    offsets,edges,incomplete=map(np.asarray,(offsets,edges,incomplete))
    if (not isinstance(use_overlap,(bool,np.bool_)) or offsets.shape!=(n+1,)
            or offsets.dtype.kind not in 'iu' or edges.ndim!=1 or edges.dtype.kind not in 'iu'
            or offsets[0]!=0 or offsets[-1]!=len(edges) or np.any(offsets<0)
            or np.any(offsets[1:]<offsets[:-1]) or np.any(edges<0) or np.any(edges>=n)
            or incomplete.shape!=(n,) or incomplete.dtype!=np.bool_):
        raise ValueError('same-edge ordered overlap CSR and explicit bool flags required')
    return dict(use_overlap=bool(use_overlap),overlap_offsets=offsets,
        overlapping_edges=edges,overlap_list_incomplete=incomplete)


def generate_initial_ordering(*, edge_count, population_index, initial_selection,
        segment_edge_numbers, segment_rows, use_overlap, overlap_offsets,
        overlapping_edges, overlap_list_incomplete, draw_random_number):
    """有序边域/已有边分组→一个初始完整排列；每个初始候选评分前调用。

    edge_count为非负int32范围整数，population_index为0..999整数。
    initial_selection/use_overlap为bool；segment_edge_numbers为整数(M,)
    分组边到候选边索引，segment_rows为有序整数行的Sequence，含源
    max_segments的空行。两者同时None表示源无segmentation；非None时
    保留空槽及nsegments=非空行数，不压缩或重新排序。其他三重叠参数
    为同E域CSR(E+1,)/(L,)/bool(E,)；draw_random_number为必填
    callable(low,high)，返回闭区间double，调用者负责随机流及seed。

    i=0不抽样，返回identity。其他候选按源选择第(i%nsegments+1)个
    非空组，随后做11轮、每轮nsegments次的nint随机交换。源seg_order
    只访问最前nsegments槽，保留中间空组的行为；最后补齐未出现的
    identity边。是否再变异由最后一次shuffle的r与i+1比较，不能用
    所选组号代替r。无初始分组时直接调用成熟的0.1变异。

    返回dict：ordering=int32(E,)；segment_order=int32(S,)；
    selected_segment_index=int或None，mutated=bool，random_draws为
    float64(D,3)[low,high,value]，完整保留所有shuffle/变异调用。
    全参必填，无坐标/单位/设备/精度默认。输入不改，不执行原版。
    非法形状/分组、无非空组、随机结果抛ValueError，回调异常传播，
    不回滚外部随机流。不是分组producer或完整GA；对应generateOrdering。
    """
    n=_int(edge_count,'edge_count',0,np.iinfo(np.int32).max)
    index=_int(population_index,'population_index',0,999)
    if not isinstance(initial_selection,(bool,np.bool_)) or not callable(draw_random_number):
        raise ValueError('explicit bool initial_selection and random callback required')
    overlap=_overlap_arguments(n,use_overlap,overlap_offsets,overlapping_edges,overlap_list_incomplete)
    numbers,groups=None,None
    if (segment_edge_numbers is None)!=(segment_rows is None):
        raise ValueError('segment mapping and rows must both be explicit or both None')
    if segment_rows is not None:
        numbers=np.asarray(segment_edge_numbers)
        if (numbers.ndim!=1 or numbers.dtype.kind not in 'iu'
                or np.any(numbers<0) or np.any(numbers>=n)
                or len(np.unique(numbers))!=len(numbers) or not isinstance(segment_rows,Sequence)):
            raise ValueError('unique old-to-candidate edge indices and complete ordered segment rows required')
        groups=[];seen=set()
        for row in segment_rows:
            a=np.asarray(row)
            if a.size==0 and isinstance(row,(list,tuple)):a=a.astype(np.int32)
            if (a.ndim!=1 or a.dtype.kind not in 'iu' or np.any(a<0)
                    or np.any(a>=len(numbers)) or len(np.unique(a))!=len(a)
                    or seen.intersection(a.tolist())):
                raise ValueError('disjoint in-range ordered segment edge rows required')
            seen.update(a.tolist());groups.append(a.astype(np.int32,copy=True))
    draws=[]
    def draw(low,high):
        if low!=0:raise RuntimeError('source ordering draw must start at zero')
        value=_draw(draw_random_number,float(high))
        draws.append((float(low),float(high),value));return value
    ordering=np.arange(n,dtype=np.int32)
    segment_order=np.empty(0,np.int32);chosen=None;mutated=False
    if index:
        if initial_selection and groups is not None:
            active=[i for i,row in enumerate(groups) if len(row)]
            count=len(active)
            if not count:raise ValueError('source initial selection with zero segments is undefined')
            chosen=active[index%count]
            selected=numbers[groups[chosen]].tolist()
            segment_order=np.arange(count,dtype=np.int32)
            last=0
            for _ in range(11):
                for slot in range(count):
                    last=math.floor(draw(0.,float(count-1))+.5)
                    segment_order[slot],segment_order[last]=segment_order[last],segment_order[slot]
            added=set(selected)
            for group in segment_order:
                for edge in numbers[groups[int(group)]]:
                    edge=int(edge)
                    if edge not in added:selected.append(edge);added.add(edge)
            selected.extend(int(edge) for edge in ordering if int(edge) not in added)
            ordering=_permutation(np.asarray(selected,np.int32),'initial ordering')
            mutated=last!=index+1
        else:mutated=True
        if mutated:
            ordering=mutate_ordering(ordering=ordering,mutation_probability=.1,
                draw_random_number=draw,**overlap)['ordering']
    return dict(ordering=ordering,segment_order=segment_order,
        selected_segment_index=chosen,mutated=mutated,
        random_draws=np.asarray(draws,np.float64).reshape(-1,3))


def _rank(values):
    ranks=rank_patch_fitness(values)
    if not np.array_equal(np.sort(ranks),np.arange(len(values))):
        raise ValueError('source epsilon ranks collide; inverse rank table is undefined')
    inverse=np.empty(len(values),np.int32);inverse[ranks]=np.arange(len(values),dtype=np.int32)
    return ranks,inverse


def run_population_search(*, edge_count, population_size, max_unchanged, niters,
        initial_selection, segment_edge_numbers, segment_rows, use_overlap,
        overlap_offsets, overlapping_edges, overlap_list_incomplete,
        draw_random_number, evaluate_patch, translated_defect_vertices,
        discarded_defect_vertices, vertex_rip, use_counts, vertex_fitness,
        vertex_eliminate, deletion_count, elimination_threshold):
    """显式缺陷/RP初态及完整评分回调→初始种群、各代、最终最佳排序。

    edge_count非负整数；population_size为1..1000，max_unchanged非负
    int32，niters为-1(不限)或非负整数。超大缺陷配方裁剪由上游先完成。
    初始分组、重叠和随机参数与generate_initial_ordering完全一致。
    translated_defect_vertices为整数(K,)原缺陷顺序到当前网格映射；
    discarded_defect_vertices为bool(K,)，vertex_rip为int32(N,) signed
    char值，use_counts为非负int32(K,)，vertex_fitness有限float32(K,)。
    vertex_eliminate为bool；deletion_count正整数和非负float32标量
    elimination_threshold是跨缺陷状态，上游首值1/4，之后必须传回。

    evaluate_patch是必填callable，以具名ordering、discarded_defect_vertices、
    vertex_rip、use_counts、vertex_fitness输入，必须返回finite double
    fitness及更新的同shape vertex_rip/use_counts/vertex_fitness。该回调
    必须完成geometry→五项评分→RP统计→选择性DVS恢复→边复位，并
    自己连续保留不属DVS的K/E/fx/curvbak；不能每次重建全零基底。
    本控制器不读取参考，也不生成缺失几何/先验。所有参数必填。

    返回best_ordering/int32(E,)、best_discarded/bool(K,)的源RP快照、
    best_fitness/F64、最终种群orderings(P,E)/fitness(P,)/ranks(P,)，
    传回RP、rip、deletion_count/threshold，evaluation_trace及
    generation_trace/termination/counts/seconds。严格改善保存RP快照；
    同代末max用>=允许较晚同分best_i，但不改早先RP最佳排序。
    精英不重评分；先完成replacement评分才抽父代补选，失败cross再
    mutation且评分；FEQUAL=abs(diff)<FLT_EPSILON。停止时的5代剔除
    回退、>=10每3代剔除、源static count/threshold均保留显式状态。

    输入数组不改，外部回调/RNG异常传播且不回滚其状态。形状/排名
    /非法draw/不存在第二父代抛ValueError，非有限评分抛ValueError。
    不执行最终重新评分、最终retessellation/面定向/匹配/未用顶点清理；
    返回的是搜索结果，完整producer必须随后执行这些必要步骤和QC。
    该接口属于mris_fix_topology内部GA，无独立原CLI，不改GPU/线程/精度。
    """
    started=time.perf_counter()
    n=_int(edge_count,'edge_count',0,np.iinfo(np.int32).max)
    p=_int(population_size,'population_size',1,1000)
    unchanged_limit=_int(max_unchanged,'max_unchanged',0,np.iinfo(np.int32).max)
    iteration_limit=_int(niters,'niters',-1,np.iinfo(np.int32).max)
    count=_int(deletion_count,'deletion_count',1,np.iinfo(np.int32).max)
    if (not isinstance(vertex_eliminate,(bool,np.bool_))
            or not callable(draw_random_number) or not callable(evaluate_patch)):
        raise ValueError('explicit elimination bool, random and complete evaluation callbacks required')
    threshold=np.asarray(elimination_threshold)
    if threshold.ndim or threshold.dtype!=np.float32 or not np.isfinite(threshold) or threshold<0:
        raise ValueError('nonnegative finite float32 persistent elimination_threshold required')
    threshold=np.float32(threshold)
    discarded=np.asarray(discarded_defect_vertices)
    mapped=np.asarray(translated_defect_vertices)
    rip,used,feedback=map(np.asarray,(vertex_rip,use_counts,vertex_fitness))
    if (discarded.ndim!=1 or discarded.dtype!=np.bool_ or mapped.shape!=discarded.shape
            or mapped.dtype.kind not in 'iu' or rip.ndim!=1 or rip.dtype!=np.int32
            or np.any(rip<-128) or np.any(rip>127) or used.shape!=discarded.shape
            or used.dtype!=np.int32 or np.any(used<0) or feedback.shape!=discarded.shape
            or feedback.dtype!=np.float32 or not np.isfinite(feedback).all()
            or np.any(mapped[~discarded]<0) or np.any(mapped[~discarded]>=len(rip))):
        raise ValueError('complete explicit same-domain defect/rip/RP state required')
    discarded,rip,used,feedback=[a.copy() for a in (discarded,rip,used,feedback)]
    overlap=_overlap_arguments(n,use_overlap,overlap_offsets,overlapping_edges,overlap_list_incomplete)
    if p<2 and unchanged_limit and iteration_limit!=0:
        raise ValueError('source one-patch generation indexes a missing second ranked parent')
    evaluations=[];generations=[];best_ordering=None;best_discarded=None
    best_fitness=-1_000_000.;best_i=0;removed=final_removed=0;best_patch=0
    unchanged=generation=0;last_euthanasia=-1
    mutation_attempts=crossover_attempts=mutation_improvements=crossover_improvements=0

    def evaluate(ordering,phase,slot):
        nonlocal rip,used,feedback,best_fitness,best_ordering,best_discarded
        nonlocal best_i,final_removed,best_patch,unchanged
        result=evaluate_patch(ordering=ordering.copy(),discarded_defect_vertices=discarded.copy(),
            vertex_rip=rip.copy(),use_counts=used.copy(),vertex_fitness=feedback.copy())
        if not isinstance(result,Mapping) or any(name not in result for name in
                ('fitness','vertex_rip','use_counts','vertex_fitness')):
            raise ValueError('complete fitness/rip/RP evaluation result required')
        score=np.asarray(result['fitness'])
        r,u,f=map(np.asarray,(result['vertex_rip'],result['use_counts'],result['vertex_fitness']))
        if (score.ndim or score.dtype.kind not in 'iuf' or not np.isfinite(score)
                or r.shape!=rip.shape or r.dtype!=np.int32 or np.any(r<-128) or np.any(r>127)
                or u.shape!=used.shape or u.dtype!=np.int32 or np.any(u<used)
                or f.shape!=feedback.shape or f.dtype!=np.float32 or not np.isfinite(f).all()):
            raise ValueError('finite score and same-order updated signed-char rip/int32 RP/F32 feedback required')
        rip,used,feedback=r.copy(),u.copy(),f.copy();score=float(score)
        improved=best_ordering is None or score>best_fitness
        evaluations.append(dict(generation=generation,phase=phase,slot=slot,
            fitness=score,strict_improvement=improved))
        if improved:
            best_fitness=score;best_i=slot;best_ordering=ordering.copy();best_discarded=discarded.copy()
            if len(evaluations)>1:final_removed=removed;best_patch=len(evaluations)
            if phase!='initial':unchanged=0
        return dict(ordering=ordering.copy(),fitness=score),improved

    population=[]
    for slot in range(p):
        ordering=generate_initial_ordering(edge_count=n,population_index=slot,
            initial_selection=initial_selection,segment_edge_numbers=segment_edge_numbers,
            segment_rows=segment_rows,draw_random_number=draw_random_number,**overlap)['ordering']
        record,_=evaluate(ordering,'initial',slot);population.append(record)
    last_fitness=best_fitness;termination='unchanged_limit'
    while unchanged<unchanged_limit:
        if generation==iteration_limit:termination='iteration_limit';break
        values=np.asarray([record['fitness'] for record in population],np.float64)
        _,ranks=_rank(values)
        elite=max(1,math.floor(.1*p+.5));replacements=min(math.floor(.1*p+.5),p-elite)
        following=[dict(ordering=population[int(i)]['ordering'].copy(),fitness=population[int(i)]['fitness'])
            for i in ranks[:elite]]
        previous_best=best_fitness
        for parent in ranks[:replacements]:
            mutation_attempts+=1
            ordering=mutate_ordering(ordering=population[int(parent)]['ordering'],
                mutation_probability=.1,draw_random_number=draw_random_number,**overlap)['ordering']
            record,improved=evaluate(ordering,'replacement',len(following));following.append(record)
            mutation_improvements+=int(improved)
        # 源selection补选draw在全部replacement之后，不能为提前planning改变随机消费。
        selection=plan_generation_selection(fitness=values,draw_random_number=draw_random_number)
        selected=selection['selected_parents']
        for first in selected:
            crossover_attempts+=1
            second=choose_distinct_parent(selected_parents=selected,first_parent=first,
                draw_random_number=draw_random_number)['parent_index']
            ordering=crossover_ordering(first_parent_ordering=population[int(first)]['ordering'],
                second_parent_ordering=population[second]['ordering'],
                first_parent_draw=_draw(draw_random_number,1.))
            slot=len(following);record,improved=evaluate(ordering,'crossover',slot)
            if improved:crossover_improvements+=1
            else:
                mutation_attempts+=1
                ordering=mutate_ordering(ordering=ordering,mutation_probability=.1,
                    draw_random_number=draw_random_number,**overlap)['ordering']
                record,improved=evaluate(ordering,'crossover_mutation',slot)
                mutation_improvements+=int(improved)
            following.append(record)
        if len(following)!=p:raise RuntimeError('complete source generation did not fill population')
        population=following;values=np.asarray([record['fitness'] for record in population],np.float64)
        _rank(values)
        # 源世代末使用>=；较晚同分best_i不重写RP的strict-improvement快照。
        best_i=0;best_fitness=float(values[0])
        for i,value in enumerate(values):
            if value>=best_fitness:best_i=i;best_fitness=float(value)
        if abs(previous_best-best_fitness)<float(np.finfo(np.float32).eps):unchanged+=1
        else:last_euthanasia=-1;unchanged=0
        row=dict(generation=generation,best_fitness=best_fitness,best_population_index=best_i,
            unchanged_before_elimination=unchanged,elimination_fraction=None,reported_deleted=0)
        if vertex_eliminate:
            fraction=None
            if unchanged>=unchanged_limit:
                if last_euthanasia<0:last_euthanasia=generation;last_fitness=best_fitness
                elif last_euthanasia+5<=generation:
                    termination='euthanasia_wait_limit';generations.append(row);break
                if last_fitness<best_fitness:last_euthanasia=generation
                if unchanged>=unchanged_limit:unchanged-=5
                fraction=.2
            elif generation>=10 and generation%3==0:fraction=.1
            if fraction is not None:
                deletion=eliminate_worst_vertices(discarded_defect_vertices=discarded,
                    translated_defect_vertices=mapped,vertex_rip=rip,use_counts=used,
                    vertex_fitness=feedback,fraction=fraction,deletion_count=count,threshold=threshold)
                discarded,rip,threshold=(deletion['discarded_defect_vertices'],deletion['vertex_rip'],deletion['threshold'])
                deleted=deletion['source_reported_deleted'];removed+=deleted
                row.update(elimination_fraction=fraction,reported_deleted=deleted,
                    unused_deleted_entries=deletion['unused_deleted_entries'].tolist(),
                    fitness_deleted_entries=deletion['fitness_deleted_entries'].tolist())
                if fraction==.2:
                    if deleted==0:termination='no_vertices_reported_deleted';generations.append(row);break
                    if count==np.iinfo(np.int32).max:raise OverflowError('persistent deletion_count exceeds int32')
                    count+=1
        generations.append(row);generation+=1
    fitness=np.asarray([record['fitness'] for record in population],np.float64)
    ranks=rank_patch_fitness(fitness)
    return dict(best_ordering=best_ordering,best_discarded=best_discarded,best_fitness=best_fitness,
        population_orderings=np.stack([record['ordering'] for record in population]),
        population_fitness=fitness,population_ranks=ranks,best_population_index=best_i,
        discarded_defect_vertices=discarded,vertex_rip=rip,use_counts=used,vertex_fitness=feedback,
        deletion_count=count,elimination_threshold=threshold,evaluation_trace=evaluations,
        generation_trace=generations,termination=termination,generations=generation,
        counts=dict(evaluated=len(evaluations),removed_vertices=removed,best_removed_vertices=final_removed,
            best_patch_number=best_patch,mutation_attempts=mutation_attempts,
            mutation_improvements=mutation_improvements,crossover_attempts=crossover_attempts,
            crossover_improvements=crossover_improvements),seconds=time.perf_counter()-started,
        final_tessellation_complete=False,complete_ga=False,overall_metric_equivalence='not_assessed')
