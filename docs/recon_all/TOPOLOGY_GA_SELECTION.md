# 拓扑 GA 的独立父代选择

## 1. 功能简介

`plan_generation_selection()`用NumPy完成旧种群的排名、统计、精英保留、变异副本和交叉父代表；`choose_distinct_parent()`按同一有序表抽取不同的第二父代。交叉和变异本体复用[已有独立算子](TOPOLOGY_GA_ORDERING.md)。这些函数没有原版程序或源码库运行依赖，尚未接入完整生产GA。

```mermaid
flowchart LR
  A[旧种群有序fitness] --> B[排名和统计]
  B --> C[精英与变异副本]
  B --> D[交叉父代表]
  D --> E[不同第二父代]
  E --> F[已有交叉与候选评分]
```

## 2. Python 调用、输入和输出

```python
from fnit.recon_all.topology_ga_selection import (
    plan_generation_selection,
    choose_distinct_parent,
)

selection_report = plan_generation_selection(
    fitness=population_fitness,  # 当前完整种群的一维double评分，候选顺序不变
    draw_random_number=controller_random_number,  # 控制器持有版本、种子和连续随机状态
)
parent_report = choose_distinct_parent(
    selected_parents=selection_report["selected_parents"],  # 保留重复槽位的int32父代表
    first_parent=int(selection_report["selected_parents"][0]),  # 第一个交叉候选的第一父代
    draw_random_number=controller_random_number,  # 与前一步同一随机流，不能重新播种
)
```

`population_fitness`和`controller_random_number`由外层控制器提供。完整原随机流的初始化及先前消费仍需连续验证；示例不会自行换成NumPy随机数生成器。

| 输入 | 格式及意义 |
| --- | --- |
| `fitness` | 有限实数`(P,)`，2≤P≤1000，转float64；无单位的候选评分 |
| `draw_random_number` | 必填`callable(low, high)`，返回指定闭区间内有限double；无隐式默认RNG |
| `selected_parents` | 非负整数`(C,)`，有序重复槽影响抽样概率 |
| `first_parent` | 父代表中的非负整数索引；表中须有另一种父代 |

选择报告返回`rank_by_patch`和`patch_by_rank`（int32 `(P,)`）、`elite_indices`（`(E,)`）、`replacement_parent_indices`（`(R,)`）、`eligible_parent_indices`（`(S,)`）、`selected_parents`（`(C,)`），以及F64概率、均值、标准差、缩放分母和归一化量。`counts`逐项列出P/E/R/S/C，`distinct_parent_count`记录父代多样性。两个函数的`random_draws`均为F64 `(D,3)`，列为low/high/value；第二父代报告还返回Python整数`parent_index`。没有影像、坐标空间、网格或文件输出。

固定比例为精英10%、可交叉候选50%、变异副本10%，四舍五入及最小数量按固定源码。`C=P-E-R`。同分复用FLT_EPSILON规则，不用另一种排序。补齐父代表时保留原绝对fitness指数；不将它改成前面配额计算的平移指数。

输入不修改。不合法形状、类型、非有限值和越界draw抛`ValueError`；统计溢出抛`FloatingPointError`。近似相等关系产生重复排名时，原逆排名表含未定义槽，因此明确报错，不猜排序。父代表只有一种父代时提前报错，避免原第二父代循环永久重试；合法多样表按原规则重试，随机回调须最终返回不同父代。失败不会回滚外部随机流。函数不改TF32、dtype、线程或CUDA配置。

## 3. 命令行调用

没有独立生产CLI。以下命令仅复跑本地合同，不是影像benchmark：

```bash
PYTHONPATH=src python validation/recon_all/optimizations/20261010_topology_ga_selection/run_local_contracts.py \
  --fixed-source /benchmark/fixed/utils/mrisurf_defect.cpp \
  --output /benchmark/new-run/ga-selection-contracts.json
```

`--fixed-source`是可选只读固定源码，核SHA和7个规则锚点，不编译或执行原版；`--output`必须是新路径，防止覆盖旧证据。新增依赖为空，使用主页Conda环境已有的Python/NumPy。

## 4. 对应原软件

属于`mris_fix_topology -ga`内部的`mrisComputeOptimalRetessellation_wkr`，没有等价的单独原CLI。参考完整命令的具名选项为：

```bash
mris_fix_topology -mgz -ga -seed 1234 -threads 4 reference_subject lh
```

这条命令仅在独立benchmark路径运行；被冻结版本、参数、输入和随机状态决定阶段对照，单独给相同seed不足以证明连续随机流相同。

## 5. 精度和耗时

本轮完成8个本地接口合同和7个只读源码规则锚点检查，详见[机器可读报告](../../validation/recon_all/optimizations/20261010_topology_ga_selection/local_contracts.json)。覆盖配额、平移指数差异、补选下溢、重复槽重试、排名歧义、输入不变和失败行为。

真实同输入参考、完整GA及原始T1整例尚未测完。这里不报告影像误差、阶段加速或显存为0；CPU数组合同的计时不能替代真实MRI benchmark。本函数不生成脑图，真实网格图仍由完整拓扑链验证报告提供。

## 6. 最近更新记录

- 2026-10-10：新增完整独立父代选择和第二父代抽样。保留原补选的绝对指数规则，明确拒绝原未定义排名表和单父代死循环输入。没有改生产默认。
- 初次10候选测试错误预期一次补选抽样；实际源配额已填满父代表。修正测试，并用13候选覆盖真实尾部补选分支；实现规则未因此改变。
- 完整候选几何、五项评分和缓存回归分别见[连续候选链](TOPOLOGY_CANDIDATE_PIPELINE.md)与[缓存](TOPOLOGY_PREPARED_BASE_CACHE.md)，不合并成完整GA通过结论。

## 7. 参考文献和源码

- [FreeSurfer固定源码：utils/mrisurf_defect.cpp](https://github.com/freesurfer/freesurfer/blob/d932c45b7941662ea380a05efef580568b98d41a/utils/mrisurf_defect.cpp)，12778–13048附近的种群计数、配额和第二父代抽样。
- [FreeSurfer宏定义](https://github.com/freesurfer/freesurfer/blob/d932c45b7941662ea380a05efef580568b98d41a/include/macros.h)，`FZERO`/`FEQUAL`使用FLT_EPSILON。
- [Fischl等，2001，Automated manifold surgery](https://doi.org/10.1109/42.906426)，皮层表面拓扑修复方法。
