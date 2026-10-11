# 全局法向先验与直方图平滑复用

## 1. 功能简介

`global_normal_dot_prior()` 用平滑 ORIGINAL 坐标、该网格的实际 metric 法向和有序一环邻域生成 100-bin 法向先验。它是拓扑修复的独立准备阶段，复用已有有序 float32 原语与 sigma=2 直方图平滑，不运行原版阶段程序。

```mermaid
flowchart LR
 A[平滑ORIGINAL及实际metric法向] --> B[原有序vnum一环取值]
 B --> C[100bin 原边界及0.01地板]
 C --> D[复用sigma2有序平滑]
 D --> E[法向先验PDF]
```

固定源的先验表达式为 `dx*nx + dy*ny*dz*nz`。这里保留该表达式和 float32 存储，不将它换成候选 likelihood 中的三项点积。其数值混合 mm 与 mm² 项，因此输出不标作三维投影距离。原始坐标上的全局 K 先验是[另一阶段](TOPOLOGY_GLOBAL_SFF.md)。

## 2. Python 调用、输入和输出

```python
from fnit.recon_all.topology_global_dot_prior import global_normal_dot_prior

normal_dot_prior = global_normal_dot_prior(
    smoothed_original_vertices=smoothed_original_xyz, # float32(N,3)，平滑ORIGINAL，surface RAS/mm
    metric_vertex_normals=smoothed_metric_normals,     # float32(N,3)，该网格metric后的实际法向
    neighbor_offsets=original_vnum_offsets,           # 整数(N+1)，原vnum一环CSR偏移
    neighbors=original_ordered_one_ring,              # 整数(E)，原访问顺序，不由faces重建
    vertex_rip=original_rip_flags,                    # bool/整数(N)，原中心rip标记
)
normal_bins = normal_dot_prior["bins"]               # float32(100)，原表达式的bin坐标
normal_pdf = normal_dot_prior["counts"]               # float32(100)，sigma2平滑后的PDF
```

所有参数必填。坐标及法向转 contiguous float32，必须同顶点顺序且有限；CSR 是 `vt.v[0:vnum]`，不能用 SFF 的 vtotal 替代。只跳过 rip 中心，原流程仍消费 ripped 邻点。上游平滑、metric、TMP/ORIGINAL 恢复不在这个接口中执行。

返回结构：

| 字段 | 类型、结构及含义 |
| --- | --- |
| `bins/counts/raw_counts` | float32(100)；raw_counts 已加空 bin 的 0.01 地板并按采样边数归一化，counts 再作平滑 |
| `bin_size/minimum_projection/maximum_projection` | float32 标量，原表达式的 bin 间隔与消费范围 |
| `sampled_edges` | 整数，非 rip 中心的消费边数 |
| `projected_values/consumed/bin_indices` | float32(E)、bool(E)、int32(E)，对应原 CSR 槽；未消费项为 NaN/False/−1 |
| `seconds` | 校验、首次 JIT、取值、计数和平滑的 API 墙钟，不含外部读写 |
| `scope/real_stage_validation/production_default_changed` | 明确 prepared 阶段、真实验证 pending、生产默认未改变 |

非法 shape/CSR/索引、空消费或零宽范围报 `ValueError`；有限输入导致非有限投影时报 `FloatingPointError`。源 clamp 允许 index=100，但数组只有 100 项；该越界情况明确报错，不能复制未定义写入。接口不自动删异常边、不生成均匀 PDF，也不修改任何输入。线程由调用方既定预算管理，不更改 TF32 或启用半精度。

## 3. 命令行调用

本阶段没有独立生产 CLI。隔离工具要求独立 before 与 after：

```bash
python tools/benchmark_recon_topology_global_sff.py \
  --input smoothed_original_metric_before.npz \
  --reference normal_dot_after.npz \
  --reference-repeat normal_dot_after_repeat.npz \
  --source-manifest frozen_source_manifest.json \
  --mode dot \
  --threads 4 \
  --output new_normal_dot_diagnostic
```

NPZ 必须恰好包含第 2 节的五个具名参数，不使用 pickle。`--mode dot` 指定本阶段，线程默认 4，参考重复可选，其余路径与新输出目录必填。自产运算结束后才读参考，报告不同值/字节、最大/P99 和重复性，候选完整数组留在私有验证目录。

## 4. 对应原软件步骤

对应固定 FreeSurfer `d932c45b7941662ea380a05efef580568b98d41a` 的 `mrisComputeNormalDotDistribution()`：恢复平滑 ORIGINAL → metric → vnum 分布 → HISTOsmooth(sigma=2) → 恢复 TMP → metric。这是 `mris_fix_topology` 内部步骤，没有独立官方 CLI。

独立 benchmark 可以在固定源码构建的 `mris_fix_topology -ga -seed 1234 -threads 4` 内观察真正入口与结果，生产不读观察结果。本轮复用已有公开同输入两次完整捕获，未新增或上传 global 观察器；实际程序 SHA 和输入 SHA 见下列验证收据。

## 5. 当前精度、时间与可视化

全局 SFF/HK、本先验与隔离 CLI 合计 43 个本地契约通过，JUnit 记录 10.513 秒，测试进程包装器 12.035 秒。法向先验覆盖源表达式、rip 中心/邻点差别、分 bin、零宽失败、100-bin 边界归一化、输入不变和自身重复。该结果是算子契约，不能替代真实 MRI benchmark。

共享 `_smooth_histogram()` 的旧循环固定 256 bins，不能直接消费本先验的 100 bins。本次只将三个循环/边界上限改为实际 `len(raw)`，17 项 sigma2 核、累加、边界重归一化保持原顺序；不重写已验证的强度直方图。这是共享函数的最小泛化，不改变原 256-bin 功能。

在既有公开真实 T1 首缺陷、472 校准顶点生成的两组 256-bin rawcounts 上，旧/新平滑、保存的校准结果以及新函数重复均逐字节相同。两组的不同值、不同字节、最大/P99 均为 0。

| 同输入平滑 | 旧函数 | 新函数 | 验证范围 |
| --- | ---: | ---: | --- |
| 白质 rawcounts | 0.004532 s | 0.004311 s | 256 输出与原存档相同 |
| 灰质 rawcounts | 0.004307 s | 0.004310 s | 256 输出与原存档相同 |

本次使用同 CPU 节点、四核亲和性，Python 3.11.16/NumPy 1.26.4；两个调用是共享节点单次观察，不计算速度比。完整平滑诊断脚本在报告写出前 0.040751 秒，包括载入、校验、重复和比较；不含冷解释器启动。报告见 [真实平滑兼容性](../../validation/recon_all/optimizations/20261010_topology_global_priors/reports/real_smoothing_v1.json)。它只验证共享平滑函数，没有替代 full normal-dot/global SFF/GA 的真实验收。此处没有执行 CUDA，显存未测；TF32 策略未变。

当前完整自产验证使用公开 ds000114 sub-07 ses-test 左侧的 110,534 顶点、221,116 面。自产首次 SFF1、两轮 ORIGINAL 平滑及实际 metric 法向后，663,348 条有向边生成的 dot bins、平滑 PDF 和 bin_size 全部逐位等于固定参考，两个参考捕获及两次自身计算均相同。metric pending `[0]`、未解决顶点和坐标改动均为 0。dot 子项 API 为 0.014224/0.010151 秒；完整 raw nofix/qsphere → K/dot 先验两次为 22.517566/18.659587 秒，均为共享四核观察，不将其子项外推为整例速度。

完整 11 项先验、脑图、17 契约与失败版本保留见 [自产全局先验](TOPOLOGY_GLOBAL_PRIORS.md) 和 [当前报告](../../validation/recon_all/optimizations/20261010_topology_global_priors_producer/README.md)。接口静态 `real_stage_validation` 字段仍描述最初 prepared 内核发布范围，当前真实证据由绑定源码与输入的 v3 报告提供。完整 GA 与整例指标未评估。

## 6. 最近更新及 benchmark

- 2026-10-10：新增 prepared 100-bin 法向先验，保留原表达式、原邻域、源访问顺序、边界和失败行为，不切旧默认。
- 同日：复用强度直方图平滑，并将 256-bin 限制最小泛化；旧 256-bin 两组固定数值 oracle 与真实白/灰 rawcounts 均通过字节回归。
- 首次全局 K/PDF 生成的 raw current 与法向先验的平滑 ORIGINAL 阶段分开。历史第二次全局 SFF 的统计捕获不能填作本轮 before 输入。
- 当前 producer v3：复用已有真实捕获，从自产 raw/qsphere 生成全部先验；dot 三项及其全局 K/PDF 组合均逐位一致，完整 GA 范围单列。

## 7. 原代码和参考文献

- [固定 metric、normal-dot 和全局曲率源码](https://github.com/freesurfer/freesurfer/blob/d932c45b7941662ea380a05efef580568b98d41a/utils/mrisurf_metricProperties.cpp)。
- [固定直方图平滑](https://github.com/freesurfer/freesurfer/blob/d932c45b7941662ea380a05efef580568b98d41a/utils/histo.cpp)。
- [拓扑 caller](https://github.com/freesurfer/freesurfer/blob/d932c45b7941662ea380a05efef580568b98d41a/utils/mrisurf_defect.cpp)。
- Fischl B et al. Automated manifold surgery. *NeuroImage* (2001). [DOI](https://doi.org/10.1006/nimg.2001.0781)。
