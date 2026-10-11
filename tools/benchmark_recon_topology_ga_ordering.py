"""完整GA交叉/变异的冻结输入回归；原随机调用轨迹只作显式输入。"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import time
import numpy as np
from fnit.recon_all import topology_ga_ordering as implementation
from fnit.recon_all.thread_budget import thread_budget
from benchmark_recon_topology_candidate_displacement import difference


def load_ordering_capture(*, capture_path):
    """严格读取FNITGAO1小端完整算子状态，返回独立typed数组。

    magic8，u32(kind,N,M,D,use_overlap,alias_mask)，F64参数：
    mutation为pmutation，crossover为真实首次父代draw。
    kind0=mutation：before/after/position_map i32(N)、offsets i32(N+1)、
    overlaps i32(M)、incomplete i32(N)、draws F64(D,3)。
    kind1=crossover：parent1/parent2/after i32(N)、draws F64(D,3)。
    draws每行为low/high/value。after仅比较；丢字段、未知尾部、非法
    domain/flag/CSR/permutation或alias_mask非零抛ValueError。不同源
    kind分别定义字段，无占位parent2或不可用位置表。
    """
    raw=Path(capture_path).read_bytes()
    if len(raw)<40 or raw[:8]!=b'FNITGAO1':
        raise ValueError('complete FNITGAO1 capture required')
    kind,n,m,d,overlap,alias=map(int,np.frombuffer(raw,'<u4',6,8))
    if kind not in (0,1) or overlap not in (0,1) or alias or n>np.iinfo(np.int32).max:
        raise ValueError('unsupported operator kind, flags, alias or edge count')
    offset=32
    def take(dtype,count,shape=None):
        nonlocal offset
        dtype=np.dtype(dtype);size=dtype.itemsize*count
        if offset+size>len(raw):
            raise ValueError('truncated complete GA ordering fields')
        out=np.frombuffer(raw,dtype,count,offset).copy();offset+=size
        return out.reshape(shape) if shape is not None else out
    parameter=take('<f8',1)[0]
    fields=dict(kind=kind,use_overlap=bool(overlap))
    if kind==0:
        fields.update(mutation_probability=parameter,ordering_before=take('<i4',n),ordering_after=take('<i4',n),
            position_map_after=take('<i4',n),overlap_offsets=take('<i4',n+1),
            overlapping_edges=take('<i4',m),overlap_list_incomplete=take('<i4',n))
        flags=fields['overlap_list_incomplete']
        if np.any(flags<0) or np.any(flags>1):
            raise ValueError('logical overlap incomplete flag required')
        fields['overlap_list_incomplete']=flags.astype(bool)
        offsets=fields['overlap_offsets'];edges=fields['overlapping_edges']
        if (offsets[0]!=0 or offsets[-1]!=m or np.any(np.diff(offsets.astype(np.int64))<0)
                or np.any(edges<0) or np.any(edges>=n)):
            raise ValueError('complete overlap CSR required')
        implementation._permutation(fields['ordering_before'],'ordering_before')
        implementation._permutation(fields['ordering_after'],'ordering_after')
        if (np.any(fields['position_map_after']<0) or np.any(fields['position_map_after']>=n)
                or not np.isfinite(parameter) or not 0<=parameter<=1):
            raise ValueError('source position map and probability invalid')
    else:
        if m or overlap or d!=1:
            raise ValueError('crossover requires no overlap domain and exactly one draw')
        fields.update(first_parent_draw=parameter,first_parent_ordering=take('<i4',n),second_parent_ordering=take('<i4',n),
                      ordering_after=take('<i4',n))
        for name in ['first_parent_ordering','second_parent_ordering','ordering_after']:
            implementation._permutation(fields[name],name)
    fields['random_draws']=take('<f8',3*d,(d,3))
    if offset!=len(raw):
        raise ValueError('unknown trailing GA ordering data')
    draws=fields['random_draws']
    if (not np.isfinite(draws).all() or np.any(draws[:,0]>draws[:,1])
            or np.any(draws[:,2]<draws[:,0]) or np.any(draws[:,2]>draws[:,1])):
        raise ValueError('finite source random intervals and in-range results required')
    if kind==1 and (not np.array_equal(draws[0,:2],[0.,1.]) or parameter!=draws[0,2]):
        raise ValueError('crossover parameter must equal its actual first-parent draw')
    return fields


def evaluate_ordering(*, fields):
    """完整before+显式source draws → 自产结果，不读取after作输入。

    fields遵循上面schema，来自原参考或自产控制器。随机调用次数/请求
    区间必须全部吻合，不允许新增draw或忽略尾部。返回ordering及有原
    定义的position_map、请求low/high/value；失败抛ValueError。原随机
    算法/种子是否正确属于控制器另一个验收步骤，此处没有隐式RNG。
    """
    draws=fields['random_draws']
    if fields['kind']==1:
        if draws.shape!=(1,3) or not np.array_equal(draws[0,:2],[0.,1.]):
            raise ValueError('source crossover requires exactly randomNumber(0,1)')
        order=implementation.crossover_ordering(
            first_parent_ordering=fields['first_parent_ordering'],
            second_parent_ordering=fields['second_parent_ordering'],first_parent_draw=draws[0,2])
        return dict(ordering=order,random_draws=draws.copy())
    position=0
    def random_number(low,high):
        nonlocal position
        if position>=len(draws) or not np.array_equal(draws[position,:2],[low,high]):
            raise ValueError('source RNG call count or requested interval differs')
        value=draws[position,2];position+=1;return value
    result=implementation.mutate_ordering(ordering=fields['ordering_before'],
        mutation_probability=float(fields['mutation_probability']),use_overlap=fields['use_overlap'],
        overlap_offsets=fields['overlap_offsets'],overlapping_edges=fields['overlapping_edges'],
        overlap_list_incomplete=fields['overlap_list_incomplete'],draw_random_number=random_number)
    if position!=len(draws):
        raise ValueError('unused source RNG draws remain')
    return dict(ordering=result['ordering'],position_map=result['source_position_map'],
                random_draws=result['random_draws'][:,3:],stale_position_edges=result['stale_position_edges'])


def run(*, capture, native_repeat, observer_receipt, output, threads):
    """冻结真实GA算子输入 → 同输入精度/重复性/读写计时JSON和私有NPZ。

    capture/native_repeat为两次独立固定源捕获；observer_receipt绑定
    实际source/program/input。output必须不存在；threads是正整数，CLI
    默认4，结束恢复。参考after仅比较；报告不称真实原随机流自产，
    更不称完整GA/整例性能。严格失败保留报告，CLI退出非零。
    """
    if output.exists():
        raise FileExistsError('new isolated regression required')
    if isinstance(threads,bool) or not isinstance(threads,int) or threads<1:
        raise ValueError('positive thread budget required')
    started=time.perf_counter();fields=load_ordering_capture(capture_path=capture)
    repeat=load_ordering_capture(capture_path=native_repeat)
    with thread_budget(threads=threads):
        tick=time.perf_counter();candidate=evaluate_ordering(fields=fields);first=time.perf_counter()-tick
        tick=time.perf_counter();again=evaluate_ordering(fields=fields);warm=time.perf_counter()-tick
    comparisons=dict(ordering=difference(candidate['ordering'],fields['ordering_after']),
                     random_draws=difference(candidate['random_draws'],fields['random_draws']))
    if fields['kind']==0:
        comparisons['position_map']=difference(candidate['position_map'],fields['position_map_after'])
    self_repeat={key:difference(value,again[key]) for key,value in candidate.items()}
    native={key:difference(value,repeat[key]) for key,value in fields.items()}
    output.mkdir(parents=True);np.savez_compressed(output/'candidate.private.npz',**candidate)
    sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    report=dict(status='complete_frozen_ga_ordering_regression',
        strict_operator_passed=all(row['bitwise_equal'] for row in comparisons.values()),
        candidate_repeat_exact=all(row['bitwise_equal'] for row in self_repeat.values()),
        native_captured_fields_repeat_exact=all(row['bitwise_equal'] for row in native.values()),
        comparisons=comparisons,self_repeat=self_repeat,native_repeat=native,
        kind='mutation' if fields['kind']==0 else 'crossover',complete_edges=len(candidate['ordering']),
        recorded_random_calls=len(fields['random_draws']),
        source_stale_position_edges=len(candidate.get('stale_position_edges',[])),
        seconds=dict(first_complete_api=first,repeat_complete_api=warm,
                     tool_load_api_npz_before_json_write=time.perf_counter()-started),
        module_sha256=sha(implementation.__file__),benchmark_sha256=sha(__file__),
        inputs_sha256=dict(capture=sha(capture),native_repeat=sha(native_repeat),observer_receipt=sha(observer_receipt)),
        observer_receipt=json.loads(observer_receipt.read_text()),
        cpu_threads=threads,cpu_affinity=sorted(os.sched_getaffinity(0)),host=platform.node(),
        python=platform.python_version(),numpy=np.__version__,gpu_used=False,
        process_gpu_memory_bytes=None,original_component_seconds=None,original_component_speedup=None,
        scope='complete frozen operator inputs with captured source RNG calls; not RNG producer, GA or whole pipeline',
        overall_equivalence='not_assessed')
    (output/'report.private.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ['capture','native-repeat','observer-receipt','output']:
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--threads',type=int,default=4)
    report=run(**vars(parser.parse_args()))
    if not all(report[key] for key in ['strict_operator_passed','candidate_repeat_exact','native_captured_fields_repeat_exact']):
        raise SystemExit(1)
