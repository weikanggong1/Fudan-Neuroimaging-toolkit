# 独立拓扑 GA 的初始排序与种群控制

## 1. 功能简介

本模块将已有交叉、变异、排名、父代选择和顶点剔除串成独立 Python/NumPy 搜索。它保持原有评分顺序、随机消费和累积反馈，不执行原版程序。完整候选的几何与评分由显式回调提供；搜索退出后仍需最终重连、匹配、未用顶点清理与网格检查。

```mermaid
flowchart LR
  A[自产边表和有序边分组] --> B[初始排序和逐候选评分]
  B --> C[原序排名与精英保留]
  C --> D[先评分变异副本]
  D --> E[父代补选与交叉]
  E --> F[未改善时再变异和评分]
  F --> G[停滞检查和RP顶点剔除]
  G --> C
  G --> H[最佳ordering和status交给最终重连]
```

固定 `-ga` 启用 `initial_selection=1`、顶点剔除，默认10个候选、停滞上限10、`niters=-1`；默认关闭 `edge_table`。源文件的静态初值3会被实际入口参数10覆盖。初始排序使用边分组，不能只用随机变异代替。边分组producer仍复用已有模块，后续需完成连续真实回归。

## 2. Python 调用、输入与输出

```python
from fnit.recon_all.topology_ga_population import run_population_search

genetic_search_report = run_population_search(
    edge_count=len(candidate_edges),  # 同一版本有序边表的总长度
    population_size=10,  # 本次固定配方的种群大小
    max_unchanged=10,  # 标准入口允许停滞代数；由实际配方显式传入
    niters=-1,  # -1不限制代数；其他值按源在该代开始前停止
    initial_selection=True,  # 复用有序边分组生成初始排序
    segment_edge_numbers=old_to_candidate_edge_numbers,  # int(M,)分组边到候选边映射
    segment_rows=ordered_edge_segment_rows,  # 完整max_segments行，保留空槽
    use_overlap=False,  # 标准-ga缺省关闭重叠交换；仍明确传入有效空CSR
    overlap_offsets=empty_overlap_offsets,  # int(E+1,)，全0
    overlapping_edges=empty_overlap_edges,  # int(0,)空数组
    overlap_list_incomplete=empty_overlap_incomplete,  # bool(E,)全False
    draw_random_number=topology_random_draw,  # callable(low,high)，维护当前随机流
    evaluate_patch=complete_candidate_cycle,  # 完整几何/评分/统计/DVS/边复位回调
    translated_defect_vertices=ordered_inside_translation,  # int(K,)原序缺陷映射
    discarded_defect_vertices=initial_discarded_vertices,  # bool(K,)显式status==DISCARD
    vertex_rip=initial_corrected_vertex_rip,  # int32(N,)完整signed-char rip值
    use_counts=initial_vertex_use_counts,  # int32(K,)本缺陷的初始RP次数
    vertex_fitness=initial_vertex_feedback,  # float32(K,)本缺陷RP均值
    vertex_eliminate=True,  # 保留原配方的顶点剔除
    deletion_count=next_deletion_count,  # 源静态状态，首个缺陷1；跨缺陷传回
    elimination_threshold=next_elimination_threshold,  # float32标量，首值4；跨缺陷传回
)
```

示例中的边表、初态、分组、随机回调和完整候选回调均由上游提供。函数没有默认 seed，不将重新初始化的随机流称为原 GA 同一消费位置。诊断可以注入真实调用轨迹；生产随机流须包含前段已消费的随机数，并隔离半球状态。

| 输入 | 格式和意义 |
| --- | --- |
| `edge_count` | 非负int32范围整数；候选边索引域E |
| `population_size` | 整数1…1000；只有初始阶段可使用1，实际世代需要至少两个父代 |
| `max_unchanged` / `niters` | 非负整数 / -1或非负整数；上游先应用大缺陷配方裁剪 |
| `initial_selection` / `use_overlap` / `vertex_eliminate` | bool，显式控制初始分组、重叠交换和剔除 |
| `segment_edge_numbers` | 整数 `(M,)`，无重复的旧分组边→候选边映射 |
| `segment_rows` | Sequence；每行整数索引属于M，行互不重叠，保留源空槽；和mapping同时None表示无segmentation |
| 三个 `overlap_*` 输入 | 有序整数CSR `(E+1,)`、整数 `(L,)`、bool `(E,)`；不用时显式空CSR |
| `draw_random_number` | callable `(low,high)`→请求闭区间内有限实数double |
| `evaluate_patch` | 具名回调，完整输入/输出约定见下文，不执行原版阶段程序 |
| `translated_defect_vertices` | 整数 `(K,)`，原缺陷顺序→corrected顶点；已DISCARD的负映射不会查询 |
| `discarded_defect_vertices` / `vertex_rip` | bool `(K,)` / int32 `(N,)`，后者值域-128…127 |
| `use_counts` / `vertex_fitness` | 非负int32 `(K,)` / 有限float32 `(K,)`，RP次数和反馈，反馈不是候选fitness |
| `deletion_count` / `elimination_threshold` | 正int32范围整数 / 非负有限float32标量；跨缺陷传回，不隐式全局缓存 |

控制器数组没有坐标或物理单位；反馈及fitness沿用算子的定义，不对MRI或表面做变换。完整候选回调以具名 `ordering`、`discarded_defect_vertices`、`vertex_rip`、`use_counts`、`vertex_fitness` 接收副本，返回mapping：有限double `fitness`、更新的同shape/类型 `vertex_rip`、`use_counts`、`vertex_fitness`。它负责重连→匹配→五项评分→RP统计→选择性DVS恢复→边复位；K/E/fx/curvbak等不属于DVS的字段须在上下文中跨候选传递。不能把每次评分当作全新零状态。

| 输出 | 结构和意义 |
| --- | --- |
| `best_ordering` / `best_discarded` | int32 `(E,)` / bool `(K,)`，严格改善时保存的RP最佳快照 |
| `best_fitness` / `best_population_index` | double最佳值 / 当前种群索引；世代末较晚同分索引不改RP快照 |
| `population_orderings` / `population_fitness` / `population_ranks` | int32 `(P,E)` / float64 `(P,)` / int32 `(P,)`，当前完整种群及源epsilon排名 |
| `discarded_defect_vertices` / `vertex_rip` | 搜索结束时当前状态，不一定等于RP最佳status |
| `use_counts` / `vertex_fitness` | 最后一次评分后的完整RP累积量 |
| `deletion_count` / `elimination_threshold` | 下一个缺陷应传入的静态状态 |
| `evaluation_trace` | list；各次generation/phase/slot/fitness/strict_improvement，不把精英复制算作评分 |
| `generation_trace` | list；停滞代数、剔除fraction/计数和原缺陷序号 |
| `termination` / `generations` | 停滞/代数/剔除停止原因；源break发生在代数自增之前时保留该计数 |
| `counts` | 实际评分数、变异/交叉尝试及改善数、删除数、最佳patch序号 |
| `seconds` | 本搜索API墙钟，包含回调；不包含调用者加载、写出和最终重连 |
| `final_tessellation_complete` / `complete_ga` | 本接口始终False，最终步骤由完整producer完成 |
| `overall_metric_equivalence` | `not_assessed`，本接口不是整例指标验收 |

`generate_initial_ordering()`提供单个初始候选的数组API：上述边数、分组、重叠和随机参数，加 `population_index`（0…999）即可调用。返回新完整排列、`segment_order`、所选非空槽、是否变异和全部float64 `(D,3)` draw轨迹。i=0不抽样；其他候选保留11轮shuffle、原nint舍入和最后一次r决定是否变异的规则。

所有参数必填；不改输入数组、设备、TF32、autocast或线程，不增加依赖。非法类型/CSR/排名碰撞/非有限评分抛 `ValueError`；无第二种父代直接报错，不复现源无限重试。回调异常传播且外部状态、随机流不回滚，因此失败后不能无收据续跑。源有序Gauss-Seidel状态依赖保留，没有机械并行化候选。

## 3. 命令行调用

没有独立生产CLI。合同与只读源码核对：

```bash
PYTHONPATH=src python validation/recon_all/optimizations/20261010_topology_ga_population/run_local_contracts.py \
  --fixed-source /benchmark/fixed/utils/mrisurf_defect.cpp \
  --output /benchmark/new-run/population-contracts.json
```

`--fixed-source`可选，只读核固定SHA和14条控制锚点；`--output`必填，必须不存在。工具保存当前HEAD、各依赖源码SHA、测试耗时与未验证范围。

## 4. 原软件调用

对应固定d932源码的 `generateOrdering` 和 `mrisComputeOptimalRetessellation_wkr` 搜索部分，无独立辅助CLI。参考整步命令只在隔离路径运行：

```bash
mris_fix_topology -mgz -ga -seed 1234 -threads 4 reference_subject lh
```

评分回调复用[完整候选计算](TOPOLOGY_CANDIDATE_GEOMETRY.md)、[反馈统计](TOPOLOGY_CANDIDATE_STATISTICS.md)与[状态恢复](TOPOLOGY_CANDIDATE_STATE.md)。父代选择复用[独立选择](TOPOLOGY_GA_SELECTION.md)，不重复实现其概率与补选规则。

## 5. 当前精度、耗时和可视化

12个本地合同通过，实际0.093秒；覆盖初始空槽、11轮shuffle、精英不重评分、replacement先于父代补选、失败交叉再变异、同分索引、累积RP、两种剔除触发及失败行为。[机器可读报告](../../validation/recon_all/optimizations/20261010_topology_ga_population/local_contracts.json)绑定实际测试的代码与依赖SHA。这是控制合同，不是影像benchmark或速度比；首次合同中的shuffle期望误写为偶数交换，已修测试并保留记录，算法未更改。

完整候选与自产基底各有真实阶段结果；当前搜索控制器的真实参考随机流、连续多候选和最终完整GA尚未验证，因此没有本页的脑图、整例提速或严格复现结论。完整GA图和异常区域将在真实连续报告生成；不以控制示意图替代脑图。没有新增整例显存、安装或隔离运行实测，仍使用主页已声明的NumPy依赖。

## 6. 更新和benchmark记录

- 2026-10-10：补齐源initial_selection排序和种群搜索；只复用既有交叉/变异/排名/父代选择/顶点剔除。
- 保留精英复制、源候选评分顺序和显式跨缺陷静态状态；默认生产调度未改变。
- 必须补齐完整评分上下文、最终重连及真实多候选回归后，才能宣布GA与整例迁移完成。

## 7. 参考文献与源码

- [固定上游拓扑算法](https://github.com/freesurfer/freesurfer/blob/d932c45b7941662ea380a05efef580568b98d41a/utils/mrisurf_defect.cpp)，初始排序4765附近，搜索12130…13340附近。
- [独立实现](../../src/fnit/recon_all/topology_ga_population.py)。
- [Fischl等，2001，Automated manifold surgery](https://doi.org/10.1109/42.906426)。
