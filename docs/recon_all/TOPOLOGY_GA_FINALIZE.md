# 独立 GA 的最终缺陷写回

## 1. 功能简介

`finalize_defect_patch()`接收完整搜索末态，回填最优排序与DISCARD，重新评分和选择性恢复，再完整重连、MRI匹配，最后剔除未使用的缺陷顶点。它复用已有Python/Numba计算，没有调用原版阶段程序。第二次几何保留最终评分的K/E/fx/curvbak，不重算SFF或重复累加RP。

```mermaid
flowchart LR
  A[GA搜索末态和最优排序] --> B[回填最优DISCARD与rip]
  B --> C[最终评分与RP更新]
  C --> D[DVS选择性恢复与边复位]
  D --> E[重连和完整MRI匹配]
  E --> F[可选源相交计数与未使用顶点清理]
  F --> G[单个缺陷最终状态]
```

此接口只完成一个缺陷的最终写回；所有缺陷、最终压缩、remesh、完整网格质量和原始T1整例由上层流程完成。返回`complete_ga=False`，当前未改生产默认。

## 2. Python调用、输入与输出

```python
from fnit.recon_all.topology_ga_finalize import finalize_defect_patch

final_patch_report = finalize_defect_patch(
    surface_state=candidate_evaluator.surface_state,  # 自产连续评分末态，不能重置为初始基底
    snapshot=candidate_evaluator.snapshot,  # 同顶点版本、搜索之前记录的原DVS
    retessellation_inputs=candidate_evaluator.retessellation_inputs,  # 同序完整边、原面、邻接与缺陷域
    candidate_inputs=candidate_evaluator.candidate_inputs,  # 自产MRI、映射矩阵、全局和局部先验
    best_ordering=population_report["best_ordering"],  # RP保存的完整最优边排列
    best_discarded=population_report["best_discarded"],  # 与最优排列同时保存的DISCARD
    best_fitness=population_report["best_fitness"],  # 实际搜索double分数，不作为最终分数填值
    use_counts=population_report["use_counts"],  # 最后评分后的原序RP使用次数
    vertex_fitness=population_report["vertex_fitness"],  # 最后评分后的原序F32反馈均值
    check_surface_intersection=False,  # 固定-ga源默认关闭本项可选计数；全网格质量另行检查
    intersection_checker=None,  # 关闭可选计数时须显式None
)
```

变量来自[种群控制](TOPOLOGY_GA_POPULATION.md)与完整候选闭环；这段代码展示接线格式，不代表已验证的全GA生产入口。

| 参数 | 类型、格式与意义 |
| --- | --- |
| `surface_state` | 完整mapping；`original_vertices`、`vertex_normals`、`flattened_vertices`、`tangent_e1/e2`为float32 `(N,3)`；`k1/k2`、`curvbak`、`vertex_area`为float32 `(N,)`；`vertex_marks`为int32 `(N,)`，`vertex_rip`为bool或整数 `(N,)`，signed-char数值须在−128…127；`vertex_topology`为int32 `(N,7)`，列序nsizeCur/nsizeMax/vtotal/vnum/v2num/v3num/num；完整原序`neighbor_storage`、`incident_faces`、`incident_corners`各N行；`faces`为整数 `(capacity,3)`、逻辑`nfaces`及float32 `face_area`。本版限nsize0/1和vtotal=vnum。已有H/K、currentXYZ、val等额外字段按当前真实值保留 |
| `snapshot` | `DefectVertexSnapshot`；同基底原序DVS记录，仅恢复源实际保存项，不恢复K/E/fx/curvbak/area/rip/marks |
| `retessellation_inputs` | 完整mapping；`canonical`同序float32 `(N,3)`球面、逻辑原`base_faces`、原序邻接、整数 `(E,2)`候选边、源0…6 `edge_used`、完整排序、corrected内点/边界/凸包映射、bool DISCARD及rip掩膜。`mode='all'`；缓存与重叠参数沿用成熟重连定义。所有内点映射必须有效，负映射不作越界访问 |
| `candidate_inputs` | 完整评分mapping：uint8 `(X,Y,Z)`归一化MRI、4×4浮点`ras_to_voxel`，局部`local_priors`、白灰联合PDF、k1/k2校准均值、dot/K直方图、K联合PDF及origin/spacing，权重沿用成熟候选默认1/10/1/1。格式见[完整候选](TOPOLOGY_CANDIDATE_PIPELINE.md)，不读取参考after |
| `best_ordering` | 整数 `(E,)`完整边排列；使用RP最优记录，不用搜索最后一个候选代替 |
| `best_discarded` | bool `(K,)`，与最优排列同一时刻、原缺陷顺序 |
| `best_fitness` | 有限实数标量；保留与实际最后评分的比较，差异不会被改成历史值 |
| `use_counts` / `vertex_fitness` | 非负int32 `(K,)` / 有限float32 `(K,)`，实际RP累积量；最终评分更新一次，第二次几何不再更新 |
| `check_surface_intersection` | 必填bool，复现源可选相交计数策略，独立于完整生产网格质量门 |
| `intersection_checker` | 开启时为必填具名回调`(original_vertices, faces, patch_faces)→非负整数`，返回相交面数；关闭时须None。不提供隐式参考程序或忽略失败的替代回调 |

坐标采用同一半球surface RAS，单位mm；法向无量纲；主曲率/H为mm⁻¹，Gaussian K为mm⁻²；面/顶点面积mm²。`ras_to_voxel`是拓扑MRI-only tkregister RAS映射，不能换成placement footer桥。顶点图与索引对应同一版本，不在本函数压缩或重排。

| 返回项 | 内容 |
| --- | --- |
| `surface_state` | 完整最终坐标、法向、面/邻接/入射计数，保留最后评分的持久字段；新面FACE.area按源0，不拿几何缓存面积代替 |
| `edge_used` / `discarded_defect_vertices` | 最终源边状态及清理后原序bool DISCARD |
| `use_counts` / `vertex_fitness` | 最终评分更新后的RP，不计第二次几何 |
| `fitness` / `best_fitness` / `fitness_matches_best` | 实际最终分数、搜索记录及严格double相等标志 |
| `final_geometry` / `final_evaluation` / `cleanup` | 成熟几何、完整评分/恢复和新增剔除原序号；清理新增计数不包括本已DISCARD条目 |
| `source_intersections` | 开启时实际回调计数；未测为None，不能报告为零相交 |
| `seconds` | 最终评分/恢复、重连/MRI匹配、可选计数、清理及包含校验的API墙钟；不含外部加载/写出 |
| `final_tessellation_complete` | True，仅表示本缺陷最终写回完成 |
| `complete_ga` / `final_mesh_quality` / `overall_metric_equivalence` | False / `not_assessed` / `not_assessed`，不能据此发布完整GA或整例通过 |

`finalize_vertex_usage()`提供清理子接口：具名参数为`vertex_marks`、`vertex_rip`、`candidate_edges`、`edge_used`、`translated_defect_vertices`、`discarded_defect_vertices`、`translated_border_vertices`，格式见表。仅3/4边端点标FINAL_VERTEX=2，再按原内点顺序清标记并剔除未使用点，最后清边界；重复映射保留顺序。返回新marks/rip/DISCARD、新增剔除原条目int64数组和计数。

全部参数显式提供；不修改输入或全局线程、CUDA/TF32/autocast，不启用半精度。非法类型、排列、容量、映射、模式或非有限状态抛`ValueError`，子函数及回调异常传播，不发布失败的半成品。

## 3. 命令行调用

内部步骤没有独立生产CLI。本地控制合同和固定源码只读核对：

```bash
PYTHONPATH=src python validation/recon_all/optimizations/20261010_topology_ga_finalize/run_local_contracts.py \
  --fixed-source /benchmark/fixed/utils/mrisurf_defect.cpp \
  --output /benchmark/new-run/finalization-contracts.json
```

`--fixed-source`可选，提供时核SHA及12项控制顺序，不执行原程序；`--output`必填且须为新路径，记录实际HEAD和源码SHA。

真实冻结输入诊断采用同一完整候选工具的十个路径参数：

```bash
python tools/benchmark_recon_topology_ga_finalize.py \
  --before-retess /benchmark/captures/before-retess.bin \
  --initial-attributes /benchmark/captures/initial-attributes.bin \
  --initial-global-priors /benchmark/captures/global-priors.bin \
  --input-surface /benchmark/fnit-subject/surf/lh.orig.nofix \
  --input-qsphere /benchmark/fnit-subject/surf/lh.qsphere.nofix \
  --brain /benchmark/fnit-subject/mri/brain.mgz \
  --after-statistics-reference /benchmark/reference/patch-statistics.bin \
  --after-volume-reference /benchmark/reference/volume.bin \
  --fitness-reference /benchmark/reference/fitness.json \
  --output /benchmark/new-run/finalization \
  --threads 4
```

前三项为声明的冻结初态和全局先验；接下来三项为同版本FNIT自产表面、球面和MRI；三项`reference`只在候选输出全部生成之后核对首次候选；`output`须不存在；`threads`固定线程预算。这是隔离benchmark，不是读取官方中间结果的生产接口。工具每条序列先独立评分建立该排序的自身分数，再从连续末态最终写回；没有将参考分数当作best，也没有执行种群搜索。

## 4. 原软件调用

对应固定d932源码`mrisComputeOptimalRetessellation_wkr`的最终段，没有独立辅助CLI。隔离参考完整命令为：

```bash
mris_fix_topology -mgz -ga -seed 1234 -threads 4 reference_subject lh
```

源最终再评分后执行DVS恢复与边复位，然后再次重连、面发现/定向和完整25轮smooth2、40轮MRI匹配；没有第二次SFF、位移或RP更新。本实现保留该顺序。最优状态的域内rip设0/1，外部剔除计数不转成bool覆盖。

## 5. 当前精度、耗时和可视化

11项控制合同和12项固定源顺序核对通过，实际合同耗时0.015秒；覆盖最优状态回填、符号rip、第二次几何的持久字段、面面积语义、最终分数不回填、域内清理/重复索引、失败和相交策略。完整结果见[版本绑定JSON](../../validation/recon_all/optimizations/20261010_topology_ga_finalize/local_contracts.json)。小网格用于这些控制测试，不作为真实MRI benchmark。

公开ds000114的sub-07、ses-test、左半球在实际FNIT `79a41cdd`上游完成两条独立序列。候选初始属性和全局先验仍冻结；MRI、白灰校准和局部先验自产。固定排序下的完整最终API为8.380/8.358秒，60类数值输出和控制字段两次逐位重复；最终坐标、有序面和法向与本序列已经通过源首候选门的几何相同。实际分数均为−97.76077795028687，RP的38个使用条目从1累加到2，没有额外剔除顶点。

| 实测范围 | 序列A | 序列B |
| --- | ---: | ---: |
| 全状态准备 | 3.656秒 | 3.284秒 |
| 建立自身best的完整首候选 | 16.984秒，包含首次JIT | 4.371秒 |
| 最终评分与选择性恢复 | 4.422秒 | 4.385秒 |
| 再次完整几何匹配 | 3.920秒 | 3.934秒 |
| 最终API，含校验和清理 | 8.380秒 | 8.358秒 |
| 最终结果写出 | 3.251秒 | 3.268秒 |

包含导入、加载、一次校准、两条序列、读写和比较的新进程完整诊断为104.397秒；这不是单个半球GA或整例时间。CPU固定120–123、4线程，最大子进程RSS为1,537,044 KiB；本阶段隐藏CUDA，GPU占用未测为`null`，TF32不适用。测试冻结源码为`72ba5336`加逐文件哈希覆盖层；报告另列实际加载的全部FNIT模块，不能将之后的提交号回填为实测版本。

固定源码观测程序两次在首候选之后主动停止；输入准备、采样、统计和局部卷的捕获重复逐位相同。该程序6.228/5.112秒的观察时间对应不同工作范围，不能与上表构成加速比。没有捕获完整GA最终赢家或最终写回，当前严格源门仅覆盖首候选；最终段验证的是重复性和持久状态语义。

![真实T1与固定排序最终表面、位移和首候选参考误差](../../validation/recon_all/optimizations/20261010_topology_ga_finalize/figures/final_patch.png)

脑图左侧为T1叠加，中间为nofix到最终固定排序几何的位移，右侧为与源首候选相位的误差。右图零误差不代表完整GA最优解或完整拓扑通过。完整[数值报告、版本、资源及图像回执](../../validation/recon_all/optimizations/20261010_topology_ga_finalize/README.md)保留失败的v1依赖导入记录；v2仅补齐诊断包依赖，算法字节未改变。全GA、原始T1整例、600秒目标及整体指标等效尚未验收。

## 6. 更新与benchmark记录

- 2026-10-10：复用成熟评分、DVS/统计和几何，补上搜索后的最终写回。最终匹配保留源迭代，不重复构造零初态或读取参考after。
- 本轮同时发现GA与候选回调的rip接口差异；候选闭环保留signed值，重连只消费非零掩膜。完整H/K按实际SFF分支写回，最终几何不重新计算它们。
- 真实冻结输入v2：60类输出两次逐位重复，最终API8.380/8.358秒，完整诊断104.397秒。未把首候选源门称作完整最终段严格复现。
- 没有新增依赖；NumPy/Numba已在主页Conda安装路径。新单步未进入生产默认，完整安装/隔离/整例验收仍待实测。

## 7. 参考文献与源码

- [固定上游拓扑算法，最终段13350附近](https://github.com/freesurfer/freesurfer/blob/d932c45b7941662ea380a05efef580568b98d41a/utils/mrisurf_defect.cpp)。
- [独立最终写回实现](../../src/fnit/recon_all/topology_ga_finalize.py)。
- [Fischl等，2001，Automated manifold surgery](https://doi.org/10.1109/42.906426)。
