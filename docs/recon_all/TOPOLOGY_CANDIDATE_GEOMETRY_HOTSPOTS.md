# 拓扑候选：完整几何剖析与状态打包对照

## 1. 功能简介

这一轮以[已验证的有序邻接与状态装配](TOPOLOGY_CANDIDATE_HOTSPOTS.md)为基线，继续剖析完整候选几何与持久状态。入射面构造复用已有 `ordered_face_csr()`；曾尝试把所有有效入射前缀打包并一次比较；完整ABBA实测该版本慢6.74%，已撤回至准确的ce96实现。失败版本及其源码哈希保留，后续CSR复用另建冻结版本。

每个动态候选仍重新生成完整入射 CSR，并检查完整基底面、全部顶点的有效前缀和存储长度。邻接顺序、容量尾部、独立可写输出、RP/DVS和持久曲率保持原规则。

下图描述该轮被撤回的打包候选，现版恢复ce96逐行状态装配。

```mermaid
flowchart LR
  A[当前候选完整有序面] --> B[原参数及基底前缀检查]
  B --> C[复用稳定有序面CSR]
  C --> D[完整候选法向和评分]
  D --> E[全N行有效入射前缀打包]
  E --> F[一次检查face与corner]
  F --> G[独立复制完整行与保留容量]
  G --> H[原RP更新与选择性DVS恢复]
```

## 2. Python 调用、输入和输出

```python
from fnit.recon_all.topology_candidate_geometry import candidate_incidents
from fnit.recon_all.topology_candidate_cycle import materialize_candidate_state

incident_structure = candidate_incidents(
    faces=current_ordered_faces,  # 整数(F,3)，实际完整面表和角点顺序
    vertex_count=current_vertex_count,  # 非负整数N，包含孤立顶点
    base_context=own_prepared_base,  # 默认None；同版不可变基底可提供
)
next_surface_state = materialize_candidate_state(
    initial=own_current_surface_state,  # 自产完整一环状态和容量存储
    result=own_complete_candidate_result,  # 完整evaluate_candidate返回
)
```

`candidate_incidents()` 只处理整数索引，没有物理单位。返回 `incident_offsets:int32(N+1,)`、`incident_faces:int32(3F,)`、`incident_corners:int32(3F,)`；每顶点的切片为 `[offsets[v],offsets[v+1])`。稳定排序保留每个原面和角点的顺序、重复面、重复角点和孤立顶点。三个返回数组独立分配，不与输入或基底共享可写存储。

`base_context=None` 默认处理完整面表。显式上下文由 `prepare_retessellation_base()` 创建，仍核对其只读元信息、相同顶点数和完整基底前缀；不跨不同候选复用动态 CSR。错误形状、非整数面、非法 N、越界索引、3F超int32容量或不匹配基底抛 `ValueError`，不修改输入。

`materialize_candidate_state()` 的完整字段见[候选闭环](TOPOLOGY_CANDIDATE_CYCLE.md)。坐标为 float32 surface RAS/mm，K1/K2为 mm⁻¹，H为 mm⁻¹，K为 mm⁻²，面积为 mm²；拓扑与容量为有序整数。返回完整新字典，不重新评分或执行恢复。本次撤回的状态打包候选曾增加完整CSR批量前缀校验。现版状态装配恢复ce96的逐行完整检查和独立复制；本页失败版本的额外校验不是现版新增公共承诺。

邻接仍逐行完整检查 append-only；未改变的邻接保留原容量存储，新增行和所有输出入射行独立复制。H/K只按原 `fit_kind` 非空行写回，K/E、fx、curvbak、RP、rip和DVS选择性状态没有改变。错误状态、多环、动态删除或前缀改序明确失败，输入不变，不发布失败中间结果；不提供线程、精度或设备切换参数。

## 3. 命令行调用

内部函数没有独立生产CLI。本轮新工具复用既有完整worker，显式区分已验证基线与候选覆盖源码：

```bash
# 固定四线程；输入是隔离冻结阶段，不是原始T1整例。
OMP_NUM_THREADS=4 NUMBA_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
CUDA_VISIBLE_DEVICES="" python tools/benchmark_recon_topology_candidate_geometry_hotspots.py \
  --finalization-workspace frozen/finalization-v2 \
  --cycle-workspace frozen/candidate-cycle-v2 \
  --baseline-overlay-source frozen/ce96-baseline/src \
  --overlay-source candidate/src \
  --before-retess diagnosis/retess.before.bin \
  --initial-attributes diagnosis/match.before.bin \
  --initial-global-priors diagnosis/statistics.bin \
  --input-surface own-subject/surf/lh.orig.nofix \
  --input-qsphere own-subject/surf/lh.qsphere.nofix \
  --brain own-subject/mri/brain.mgz \
  --after-statistics-reference benchmark-only/patch_statistics.bin \
  --after-volume-reference benchmark-only/volume.bin \
  --fitness-reference benchmark-only/fitness.json \
  --output diagnosis/new-geometry-abba \
  --threads 4
```

前两个工作区保存原完整worker及依赖；`baseline-overlay-source` 保存实测 `ce96e724` 两个模块。候选覆盖目录包含原 smoothing、候选 geometry/cycle；逐进程核对实际加载SHA。三个 before/global 文件是相位明确的冻结输入；nofix、qsphere及uint8 brain自产 MRI 校准。三个 after 文件仅在原工具保存计算结果后比较，不生成候选。`output` 必须不存在，线程预算固定，`--worker`只为工具内部子进程使用。

输出包含四个完整子进程的 NPZ/控制记录、receipt、EXIT和ABBA JSON；失败保留已有文件。完整 fresh CLI 时钟包含所有子进程导入、JIT、加载、校验、完整候选/最终写回、读写和事后比较。透明函数/API子项不相加冒充整例。

## 4. 原软件调用

对应固定源 `mris_fix_topology -ga` 中完整入射拓扑、候选几何、`mrisDefectPatchFitness()` 与 `mrisRestoreVertexState()` 的内部准备；没有独立官方 CLI。原版程序仅生成 benchmark 参考，不进入此完整 Python 工具。时间对照为已验证FNIT控制，不用原版观察器墙钟充当候选速度。

固定参考源码 `d932c45b7941662ea380a05efef580568b98d41a`；本轮只发布自有代码、数据哈希和安全指标，不发布原源码主体、程序、库或许可证。

## 5. 最新真实精度、时间和脑图

公开 ds000114 sub07 ses-test LH，FNIT `79a41cdd` 上游；110534顶点、完整候选216860面，四核、总线程4、CUDA隐藏。全局属性及先验冻结，MRI校准自产。新的[完整只读剖析](../../validation/recon_all/optimizations/20261010_topology_candidate_geometry_hotspots/profile_v1.json)已 EXIT0，原首候选严格参考门和最终自身重复门通过。

| 实际剖析范围 | 秒数 | 边界 |
|---|---:|---|
| 完整两序列 API，含分调用cProfile | 131.369816 | 不是前后速度对照 |
| 首个完整 geometry | 17.460243 | 含7次JIT，编译父累计11.534秒 |
| 后三个热 geometry | 5.604564 / 6.581881 / 5.780470 | 都没有JIT调用；含完整算法和校验 |
| 热调用 candidate_incidents | 0.1368 / 0.1374 / 0.1291 | 不是几何主瓶颈 |
| 后三个完整 state 装配 | 1.538120 / 1.591068 / 1.513299 | 每次44万余数组转换、66万余tolist、22万余copy |

热几何的主要开销还包括完整面发现的集合/边入射字典和smooth/MRI-match重复构造全网格入射表。局部25轮/40轮数值内核约0.05/0.08秒，其公共全网格准备各约0.54秒。这些数字有cProfile开销，不作为未剖析的运行速度；此版本只优化获准两个模块，未修改重连判定或数值内核。

[完整ABBA](../../validation/recon_all/optimizations/20261010_topology_candidate_geometry_hotspots/abba_v1.json)已完成四个fresh child，顺序control/candidate/candidate/control：130.969718 / 136.471708 / 111.127621 / 101.005415秒。每个子进程保留完整两序列、所有162个数值/容量字段和控制JSON；全部字节一致，源首候选门及最终自身重复门均通过。

| 非剖析完整ABBA中位数 | control | candidate | 结果 |
|---|---:|---:|---|
| fresh CLI，含导入/JIT/加载/校验/读写 | 115.987567s | 123.799664s | 慢6.7353%，撤回state修改 |
| 完整两序列API | 102.502703s | 108.576975s | 慢5.926%，不是整例 |
| 六次入射构造合计 | 2.985697s | 0.331582s | 局部更快，未抵消状态成本 |
| 六次状态装配合计 | 5.885771s | 9.574910s | 打包完整行更慢 |

内部合计已包含在完整API，不能再累加。四次共享主机观测仍保留冷JIT及跨时段负载；不能用局部入射收益宣称完整候选加速。状态代码恢复至ce96，候选入射CSR进入下一独立版本验证。完整9缺陷基线仍使用原冻结源码，不回标本次模块。

同一冻结病例既有图像供核对几何和MRI相位，未因性能剖析重新生成：

![冻结真实候选与MRI参考](../../validation/recon_all/optimizations/20261010_topology_candidate_cycle/candidate_cycle.png)

本轮不使用GPU，显存为 `null`。新剖析记录进程最大RSS；ABBA工具未采样峰值RSS，不能称内存验收；无新增依赖，主页已有NumPy/Numba/nibabel。完整GA、最终完整网格质量、原始T1整例、10分钟目标和整体指标等效在本阶段不判定。

## 6. 更新与 benchmark

- 基线 `ce96e724`：已有完整162字段和控制轨迹字节门，原真实时钟及源码SHA保留。
- 本轮只读 profile v1：实际使用 ce96 两模块，逐六次geometry和六次state分别记录冷/热，完整API EXIT0；没有覆盖上一轮cProfile或ABBA。
- 本轮合同：18tests + 14subtests，8.90秒；新增重复角点/重复面完整乘数、CSR非法偏移失败事务、浮点错误旧前缀不量化后接受。是本地合同，不是MRI benchmark。
- 本轮候选：geometry `6d9e95ac…`、cycle `11224839…`；严格输出通过，但完整性能退化，cycle已撤回。源文件按实际未提交SHA绑定，发布HEAD不回填成测试版本。

## 7. 参考文献与原代码

- [FreeSurfer 固定mrisurf_defect.cpp](https://github.com/freesurfer/freesurfer/blob/d932c45b7941662ea380a05efef580568b98d41a/utils/mrisurf_defect.cpp)：完整候选、评分、状态恢复。
- [FreeSurfer 固定mrisurf.cpp](https://github.com/freesurfer/freesurfer/blob/d932c45b7941662ea380a05efef580568b98d41a/utils/mrisurf.cpp)：有序拓扑与面关联。
- Fischl et al. 2001. Automated manifold surgery. IEEE TMI. [DOI](https://doi.org/10.1109/42.906426)。
- [候选闭环](TOPOLOGY_CANDIDATE_CYCLE.md)、[完整几何](TOPOLOGY_CANDIDATE_GEOMETRY.md)、[第一轮热点](TOPOLOGY_CANDIDATE_HOTSPOTS.md)：成熟接口、状态与真实验证边界。
