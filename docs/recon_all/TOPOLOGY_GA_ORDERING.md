# 拓扑 GA 的交叉、变异与排名

## 1．功能简介

`crossover_ordering` 和 `mutate_ordering` 用独立 NumPy/Python 处理拓扑
GA 中的完整候选边排列。排名直接复用已有 `rank_patch_fitness`。
它们不执行、链接或发现原版程序，不负责网格重连、fitness 计算或
整个 GA 的种群调度。

交叉保留原 `nadded` 控制，第一次加入一条边后，换父代与继续父代各有
一次尝试。变异保留逐位置随机决定、完整/不完整重叠分支及 `NTRY=0`。
任意交换分支不更新源临时位置表，完整重叠分支才更新两项；这个状态
差异会改变后续交换，不能将它当作无影响的缓存重建。

## 2．Python 调用、输入与输出

```python
import numpy as np
from fnit.recon_all.gcsa_permutation import VnlRandom
from fnit.recon_all.topology_ga_ordering import (
    crossover_ordering, mutate_ordering, rank_patch_fitness,
)

# 读取FNIT自产控制器检查点；这里不是原软件after参考输出。
population_state = np.load(file="fnit_ga_population.npz", allow_pickle=False)
random_stream = VnlRandom(seed=1234)  # 示例单算子的可重复流；完整GA还须保存消费位置

def draw_random_number(lower_bound, upper_bound):
    # 复用已有VNL随机数代码；完整GA的seed/prime/既往消费顺序另行验证。
    unit_random_number = random_stream.open_ran1()
    return float(unit_random_number * (upper_bound - lower_bound) + lower_bound)

child_ordering = crossover_ordering(
    first_parent_ordering=population_state["first_parent_ordering"],  # 整数(N,)完整排列
    second_parent_ordering=population_state["second_parent_ordering"],  # 同一边域的整数(N,)排列
    first_parent_draw=draw_random_number(0.0, 1.0),  # 一次明确的父代选择draw
)
mutation_result = mutate_ordering(
    ordering=child_ordering,  # 整数(N,)完整候选边排列
    mutation_probability=float(population_state["mutation_probability"]),  # [0,1]明确概率
    use_overlap=bool(population_state["use_overlap"]),  # 是否使用重叠边表
    overlap_offsets=population_state["overlap_offsets"],  # 整数(N+1,)CSR偏移
    overlapping_edges=population_state["overlapping_edges"],  # 整数(M,)有序重叠边
    overlap_list_incomplete=population_state["overlap_list_incomplete"],  # bool(N,)列表不完整标志
    draw_random_number=draw_random_number,  # 必填(low,high)→有限double回调
)
population_ranks = rank_patch_fitness(
    fitness=population_state["fitness"],  # 完整(P,)候选double评分；已成熟排名函数
)
```

这只是单独算子的具名示例，不能仅凭相同 seed 称它与原 GA 随机流一致。
生产控制器须显式维护种子、初始化、当前状态、既往消费位置及半球隔离。
冻结同输入诊断可直接注入原函数的实际 low/high/value 调用轨迹。

| 参数或输出 | 类型、结构和语义 |
| --- | --- |
| 两个 `*_parent_ordering` | 整数 `(N,)`，各自恰含一次 `0…N-1`；须属于同一个候选边表，长度相同 |
| `first_parent_draw` | 有限实数 `[0,1]`；严格 `<0.5` 从第一父代起步 |
| 交叉返回值 | 新 int32 `(N,)` 完整排列；父代输入不变，不复制 patch 的评分、模式或 rank |
| `ordering` | 整数 `(N,)` 完整边排列；每轮当前位置会受前面的交换影响 |
| `mutation_probability` | 有限实数 `[0,1]`；每个位置先 draw，再严格 `p<probability` 决定交换 |
| `use_overlap` | bool；关闭时走任意位置交换 |
| `overlap_offsets` | 整数 `(N+1,)`，从 0 到 M，非降序 CSR |
| `overlapping_edges` | 整数 `(M,)`，同一边域的边索引；保留源列表顺序，不隐式去重 |
| `overlap_list_incomplete` | bool `(N,)`，对应源 `ET_OVERLAP_LIST_INCOMPLETE=0x0001` 逻辑值 |
| `draw_random_number` | callable `(low,high)`，返回请求区间内有限实数；不隐式改用 NumPy RNG |
| 变异返回 `ordering` | 新 int32 `(N,)` 完整排列 |
| `source_position_map` | 新 int32 `(N,)` 源临时 `dp_indices`；可能过期，不应称最终逆排列 |
| `stale_position_edges` | int64 `(S,)`，最终临时表与真实逆排列不一致的边索引，用于诊断 |
| `random_draws` | float64 `(D,6)`，行 `[i,edge,kind,low,high,value]`；kind=0决定、1任意、2完整重叠交换 |
| `rank_patch_fitness` | 已有有限 double `(P,)`→int32 `(P,)`；逐对 `abs(diff)<FLT_EPSILON` 同分时原序较早者优先 |

所有参数必填，没有阈值、设备、概率或随机种子的隐含默认。这些数组
使用同一个边索引域，没有坐标和物理单位。关闭重叠时，检查点仍须
明确提供有效的空 CSR。输入不修改；类型、长度、重复/越界索引、CSR、
非有限或越界 draw 抛 `ValueError`。外部随机回调异常原样传播，失败
不会回滚外部随机流。空排列不调用变异回调。GPU、TF32 和线程保持
调用方设置；没有 FP16/BF16。

## 3．命令行调用

它们是控制器数组 API，没有独立生产 CLI。契约测试：

```bash
PYTHONPATH=src python -m unittest discover \
  -s tests/recon_all \
  -p 'test_topology_ga_ordering*.py' \
  -v
```

真实冻结参考工具已提供，实际捕获尚未完成：

```bash
python tools/benchmark_recon_topology_ga_ordering.py \
  --capture isolated_reference/repeat0/mutation0.bin \
  --native-repeat isolated_reference/repeat1/mutation0.bin \
  --observer-receipt isolated_reference/receipt.json \
  --output new_diagnostic/ga_mutation \
  --threads 4
```

`capture`/`native_repeat` 是两次固定源的完整 `FNITGAO1` 捕获。
`observer_receipt` 绑定实际输入、源代码、构建和程序版本。`output`
必须不存在，输出 JSON 和私有 NPZ；`threads` CLI 默认 4，须为正整数，
结束恢复。工具核对每一次随机请求区间和消费数量，不增加或忽略 draw。
after 只作比较，严格失败保留报告并非零退出。

`FNITGAO1` 为小端：magic8、u32 `(kind,N,M,D,use_overlap,alias_mask)`、
F64 参数；mutation 参数为 pmutation，crossover 参数为真实父代 draw。
mutation 保存 before/after/临时位置表、重叠 CSR、逻辑标志和全部
F64 `(D,3)` low/high/value；crossover 保存两父代、after 和一条 draw。
两个 kind 不构造无定义的占位字段。当前 API 不支持目的 patch 与父代
对象别名；捕获发现此情况会报错。截断、未知尾字节或非法字段也报错。

## 4．原软件调用与源码

对应固定 d932 `mrisurf_defect.cpp` 内部 `mrisCrossoverDefectPatches`、
`mrisMutateDefectPatch` 和 `defectPatchRank`，没有这些辅助函数的独立
官方 CLI。完整原软件参考命令：

```bash
mris_fix_topology -ga -seed 1234 -threads 4 reference_subject lh
```

原程序仅在隔离 benchmark 环境运行。参考 observer 会透明转发一次原
`randomNumber` 调用，保存其请求区间和实际返回值；每类前三次调用后
有意停止，exit0 不等于完成拓扑 GA，不能写作整例重建完成。

现场只读核对：`utils.cpp` 的 `randomNumber` 使用 double 范围缩放；
`numerics.cpp` 的 `OpenRan1` 将 VNL `drand64` 结果写 float 再返回。
不能用 NumPy 默认 RNG 宣称相同随机序列。具体初始化/调用位置仍需
全控制器验证。

## 5．当前精度、耗时与图

本地 11 项契约测试通过，最近一组 0.023 秒：8 项覆盖有序父代切换、
重复候选跳过、每位置 draw、任意/完整/不完整重叠、临时位置表语义、
空输入和失败事务；3 项覆盖完整二进制字段、类型、别名拒绝、尾部、
截断及随机区间/消费数量。这不是影像 benchmark。

[机器可读报告](../../validation/recon_all/optimizations/20261010_topology_ga_ordering/local_contracts.json)
绑定实际测试的 HEAD 和八个实现／测试／工具文件 SHA。
[复现脚本](../../validation/recon_all/optimizations/20261010_topology_ga_ordering/run_local_contracts.py)
记录合同耗时；源码锚点检查只读固定源，不编译或上传它。

固定 d932 源的透明插桩锚点已通过本地只读核对，8 个源随机调用位置
完整保留。新增原位构建/真实捕获尚未执行，headcw 上传授权等待中。
因此没有当前真实算子误差、速度比、误差图或整例提速结论。阶段
严格复现尚未评估；整体指标等效为 `not_assessed`，不改默认拓扑调度。

实现只用主页已有 NumPy，复用已有 VNL 随机组件与排名，不增加依赖。
未做新的干净环境安装或整例隔离验证；未采样显存，不能报告零显存。

## 6．更新记录与 benchmark

- 2026-10-10：新增独立有序交叉和变异，保留源控制流及临时表更新边界。
- 保留显式 RNG 回调，不伪造 source seed 等价；排名直接复用已验证函数。
- 新增完整条件 schema/冻结同输入工具与透明 reference observer；真实
  验证、独立完整 GA 控制器及原始 T1 整例仍待完成。

## 7．参考文献与源码库

- [固定原算子源码](https://github.com/freesurfer/freesurfer/blob/d932c45b7941662ea380a05efef580568b98d41a/utils/mrisurf_defect.cpp#L11619-L11855)。
- [独立 NumPy/Python 实现](../../src/fnit/recon_all/topology_ga_ordering.py)。
- [复用的排名函数](../../src/fnit/recon_all/topology_fitness_search.py)。
- [复用的 VNL 随机组件](../../src/fnit/recon_all/gcsa_permutation.py)。
- Fischl B, Liu A, Dale AM. Automated manifold surgery. IEEE TMI, 2001.
  [DOI](https://doi.org/10.1109/42.906426)。
