"""拓扑 GA 的有序交叉与变异；独立 NumPy/Python，随机数显式注入。"""
from __future__ import annotations

import numpy as np

# 排名已经有成熟实现；控制器直接复用，不另写排序或 epsilon 规则。
from .topology_fitness_search import rank_patch_fitness


def _permutation(value, name):
    data=np.asarray(value)
    if data.ndim!=1 or data.dtype.kind not in 'iu':
        raise ValueError(f'{name} requires a one-dimensional integer permutation')
    n=len(data)
    if n>np.iinfo(np.int32).max or np.any(data<0) or np.any(data>=n):
        raise ValueError(f'{name} outside source int32 edge domain')
    data=data.astype(np.int32,copy=True)
    if n and not np.all(np.bincount(data,minlength=n)==1):
        raise ValueError(f'{name} must contain each edge exactly once')
    return data


def _real(value, name):
    data=np.asarray(value)
    if data.ndim or data.dtype.kind not in 'iuf' or not np.isfinite(data):
        raise ValueError(f'{name} requires a finite real scalar')
    return float(data)


def crossover_ordering(*, first_parent_ordering, second_parent_ordering,
        first_parent_draw):
    """同一边域的两父代完整排列 → 新 int32(N,) 子代排列。

    first_parent_ordering/second_parent_ordering 为整数(N,) 的0…N-1排列，
    同一有序候选边域。first_parent_draw 为原randomNumber(0,1)的一次
    有限double结果，须在[0,1]；<0.5从第一父代起步，其他从第二父代。
    函数不生成随机数，不假定NumPy RNG等同原随机流。原nadded控制
    会在首次一个条目之后形成换父代/继续父代的两次尝试，失败的重复
    条目跳过也会推进该控制状态；不能简化成逐条交替或拼接两半。

    返回新int32(N,)完整排列，不修改输入，不复制fitness/rank/模式等
    patch属性。无默认值、坐标或物理单位；非法类型/排列/长度/draw
    抛ValueError，内部状态无法推进抛RuntimeError，不执行原版程序，
    不改变CUDA/TF32/线程。对应mrisCrossoverDefectPatches内部步骤。
    """
    parents=(_permutation(first_parent_ordering,'first_parent_ordering'),
             _permutation(second_parent_ordering,'second_parent_ordering'))
    if len(parents[0])!=len(parents[1]):
        raise ValueError('parent permutations must share the complete edge domain')
    draw=_real(first_parent_draw,'first_parent_draw')
    if not 0<=draw<=1:
        raise ValueError('first_parent_draw outside randomNumber(0,1)')
    n=len(parents[0]);child=np.empty(n,np.int32);added=np.zeros(n,bool)
    current=0 if draw<0.5 else 1
    positions=[0,0];nadded=0;output=0;attempts=0
    while output<n:
        attempts+=1
        if attempts>8*n+8:
            raise RuntimeError('valid crossover permutations did not progress')
        if nadded>=1:
            nadded=0
            if (current==0 and positions[1]<n) or positions[0]>=n:
                current=1
            elif positions[0]<n:
                current=0
        else:
            nadded+=1
        index=positions[current]
        if index>=n:
            continue
        parent=parents[current]
        while index<n and added[parent[index]]:
            index+=1
        if index>=n:
            continue
        positions[current]=index+1
        edge=int(parent[index]);child[output]=edge;added[edge]=True;output+=1
    return child


def mutate_ordering(*, ordering, mutation_probability, use_overlap,
        overlap_offsets, overlapping_edges, overlap_list_incomplete,
        draw_random_number):
    """完整边排列与显式重叠CSR → 新排列、原临时位置表和随机调用轨迹。

    ordering为整数(N,)完整排列；mutation_probability为[0,1]实数。
    use_overlap为bool。overlap_offsets为整数(N+1,)从0到M非降序CSR；
    overlapping_edges为整数(M,)重叠边索引，可有重复，顺序须属于同边域。
    overlap_list_incomplete为bool(N,)，对应源ET_OVERLAP_LIST_INCOMPLETE。
    draw_random_number为必填callable(low,high)，返回该闭区间内有限
    double；控制器负责原随机流/版本/种子，函数不隐式选择NumPy RNG。

    每个当前位置i先调用(0,1)一次，严格p<mutation_probability才交换。
    关闭重叠、无重叠或列表不完整：再调用(0,N-0.1)，截断为位置。
    完整重叠：调用(0,count-0.0001)，截断取重叠边，再查询源临时位置表。
    固定源NTRY=0，不检查边used、不额外重试。任意交换分支不更新源
    临时位置表；完整重叠分支才更新两项，因此不能在每次交换后重建它。

    返回dict：ordering=int32(N,)新排列；source_position_map=int32(N,)
    原临时dp_indices（可能过期，不是最终逆排列）；
    stale_position_edges=int64(S,)最终过期项；random_draws=float64(D,6)
    每行[i,当前edge,kind,low,high,value]，kind0决定、1任意、2重叠交换。
    无参数默认、坐标或单位。类型/形状/CSR/索引/非有限/越界draw抛
    ValueError；随机回调异常原样传播。输入数组不改，但失败时不回滚
    外部随机流。空排列不调用回调。不读写文件、不执行原版、不修改
    GPU/TF32/线程；没有独立原CLI，对应mrisMutateDefectPatch内部步骤。
    """
    result=_permutation(ordering,'ordering');n=len(result)
    probability=_real(mutation_probability,'mutation_probability')
    if not 0<=probability<=1 or not isinstance(use_overlap,(bool,np.bool_)):
        raise ValueError('mutation probability in [0,1] and bool use_overlap required')
    offsets=np.asarray(overlap_offsets);edges=np.asarray(overlapping_edges)
    incomplete=np.asarray(overlap_list_incomplete)
    if (offsets.dtype.kind not in 'iu' or offsets.shape!=(n+1,)
            or edges.dtype.kind not in 'iu' or edges.ndim!=1
            or offsets[0]!=0 or offsets[-1]!=len(edges)
            or np.any(offsets[1:]<offsets[:-1]) or np.any(offsets<0)
            or np.any(edges<0) or np.any(edges>=n)
            or incomplete.dtype!=np.bool_ or incomplete.shape!=(n,)):
        raise ValueError('complete ordered overlap CSR and bool incomplete flags required')
    if not callable(draw_random_number):
        raise ValueError('explicit draw_random_number(low,high) callable required')
    position=np.empty(n,np.int32);position[result]=np.arange(n,dtype=np.int32)
    draws=[]
    def draw(i,edge,kind,upper):
        value=_real(draw_random_number(0.0,float(upper)),'randomNumber result')
        if not 0<=value<=upper:
            raise ValueError('randomNumber result outside requested source interval')
        draws.append((i,edge,kind,0.0,upper,value))
        return value
    for index in range(n):
        edge=int(result[index])
        if draw(index,edge,0,1.0)>=probability:
            continue
        begin,end=int(offsets[edge]),int(offsets[edge+1])
        complete=bool(use_overlap and end>begin and not incomplete[edge])
        if complete:
            overlap_index=int(draw(index,edge,2,end-begin-0.0001))
            target_edge=int(edges[begin+overlap_index])
            target=int(position[target_edge])
        else:
            target=int(draw(index,edge,1,n-0.1))
        result[index],result[target]=result[target],result[index]
        if complete:
            position[result[index]]=index
            position[result[target]]=target
    inverse=np.empty(n,np.int32);inverse[result]=np.arange(n,dtype=np.int32)
    return dict(ordering=result,source_position_map=position,
                stale_position_edges=np.flatnonzero(position!=inverse),
                random_draws=np.asarray(draws,np.float64).reshape(-1,6))
