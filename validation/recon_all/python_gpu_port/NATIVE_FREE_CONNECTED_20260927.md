# 实验性纯 Python recon-all：同一 T1 整例对照（2026-09-27）

**结果：整例流程跑通，但与 FreeSurfer 8.2 不等价。** 输入原始 T1 后，候选独立生成体积分割、双侧 white/pial/sphere、16 种双侧顶点图、三个双侧皮层 atlas 和脑区统计。严格固定配置只通过 **6/138** 项，52 项缺失，80 项存在差异。因此这是一版可运行的近似核心流程，不是官方指标的替代结果。

完整逐文件、逐标签和逐脑区数值在 [comparison.json](native_free_sub01_20260927/comparison.json)；[strict_138.json](native_free_sub01_20260927/strict_138.json) 列出固定配置全部 138 项；[run.json](native_free_sub01_20260927/run.json) 和 [benchmark.json](native_free_sub01_20260927/benchmark.json) 保存阶段计时与哈希。

## 输入、环境和可复现边界

- 输入：仓库示例 `sub-01_T1w.nii.gz`，SHA-256 `f20410a4efd8e6a05cd04d55730a4a5492ecf9ad1b234fe0fd4661e448270c6a`。
- 节点：`gpucw1`，NVIDIA H100 PCIe 80 GB，使用 `cuda:1`、4 个 CPU 线程；PyTorch 2.5.1/CUDA 11.8、nibabel 5.4.2、SciPy 1.17.1。N4 调用另一 Python 环境中的 SimpleITK 2.4.0；其算子在 CPU 上运行。张量为 float32，未使用 float16/bfloat16；SynthSeg 卷积关闭 cuDNN TF32 以保持本例硬标签一致，其余 CUDA 路径按既有 TF32 设置运行。
- 对照：同一 T1 在同一节点的归档 FreeSurfer 8.2 `recon-all` 被试，官方日志 SHA-256 `6badf1a4c26e564a29f78a173547f4151920400ff4bda1047d15135c87998a66`。该官方运行不是本轮重新启动的配对冷启动，且共享节点负载未固定。
- 候选运行从空被试目录启动。`native_free.py` 只启动一次 Python N4 子进程，没有 FreeSurfer 可执行程序调用。测试时权重和模板从现有 bundle 的 `models/` 与数据目录读取；bundle 同时含有原生程序，因此本轮没有完成“只有权重和模板的全新安装”测试。公开入口以 `fnit-setup-weights --model recon-all` 和 `fnit-setup-recon-all-assets` 下载外部文件，不要求用户安装 FreeSurfer。
- 原整例运行 1781.889 秒。运行后发现 a2009s 体素编号偏移和未知 atlas 顶点的投影问题；用[可复现补算脚本](reproject_benchmark_subject.py)更新三种体素分区、`wmparc` 和其统计，实测增加 23.844 秒。报告的总时长 **1805.733 秒** 为两个实际运行阶段时间之和，并包含原流程被覆盖的第一次投影，属于保守计时。启动源码与最终修正源码 SHA-256 均在 `run.json`；新启动的流程已直接使用修正算法，无需补算。

## 时间

| 项目 | 候选 | 归档官方命令或整例 | 可比性 |
|---|---:|---:|---|
| 整例 | **1805.7 s（30.10 min）** | **6789.6 s（113.16 min）** | 候选/官方 0.266；输出不等价，不能解释为等价流程加速 |
| N4 + `nu` | 236.5 s | `mri_nu_correct.mni` 225.3 s | 大致对应 |
| SynthSeg | 7.1 s | `mri_synthseg` 192.3 s | 硬标签一致，软体积仍有微小差异 |
| GCA 仿射配准 | 905.4 s | `mri_em_register` 251.8 s | 候选 CPU 实现明显更慢 |
| 双侧近似表面 | 68.1 s | `mris_place_surface` 1505.6 s | 算法和输出不同，不作加速结论 |
| 六次皮层 atlas 标注 | 389.6 s | 官方日志无完全对应的单独命令汇总 | 不直接比较 |

所有候选阶段按实测顺序列出；CPU/GPU 为主要计算路径，混合步骤也使用 CPU 文件与网格处理。

| 候选阶段 | 秒 | 主要设备 |
|---|---:|---|
| `input_talairach` | 9.5 | CPU + CUDA |
| `n4` | 234.1 | CPU |
| `nu` | 2.4 | CPU |
| `T1_normalize` | 81.8 | CUDA |
| `brainmask` | 1.1 | CUDA |
| `SynthSeg` | 7.1 | CUDA |
| `mri_em_register` | 905.4 | CPU |
| `mri_ca_normalize` | 30.9 | CPU |
| `surface_lh` | 31.2 | CPU + CUDA |
| `surface_rh` | 36.9 | CPU + CUDA |
| `annot_lh_aparc` | 63.3 | CPU + CUDA |
| `annot_lh_aparc.a2009s` | 70.4 | CPU + CUDA |
| `annot_lh_aparc.DKTatlas` | 58.0 | CPU + CUDA |
| `annot_rh_aparc` | 60.7 | CPU + CUDA |
| `annot_rh_aparc.a2009s` | 77.3 | CPU + CUDA |
| `annot_rh_aparc.DKTatlas` | 60.0 | CPU + CUDA |
| `project_aparc_volumes` | 5.2 | CPU |
| `project_wmparc` | 1.9 | CPU |
| `brain_volume_stats` | 3.0 | CPU |
| `aseg_stats` | 17.1 | CPU |
| `wmparc_stats` | 15.3 | CPU |
| `stats_lh_aparc` | 0.7 | CPU + CUDA |
| `stats_lh_aparc.a2009s` | 0.7 | CPU + CUDA |
| `stats_lh_aparc.DKTatlas` | 0.7 | CPU + CUDA |
| `stats_rh_aparc` | 0.7 | CPU + CUDA |
| `stats_rh_aparc.a2009s` | 0.7 | CPU + CUDA |
| `stats_rh_aparc.DKTatlas` | 0.7 | CPU + CUDA |
| `corrected_aparc_ids` | 5.6 | CPU |
| `corrected_wmparc_ids` | 2.3 | CPU |
| `corrected_wmparc_stats` | 16.0 | CPU |

阶段和为 1800.5 秒；其余约 5.2 秒是调度、报告写盘等开销。归档官方日志还记录 `mris_register` 833.1 秒、`rca-surfreg` 834.7 秒；候选仅复制未配准的径向 sphere 作为 `sphere.reg`，不存在对应的等价计时。

## 体素分割和体积

全部可比体积的数值 affine 差为 0。`orig`、`nu`、`T1`、`synthseg.rca` 在 16,777,216 个体素上逐值相同；`brainmask` 有 35 个体素不同。下表显示下游实际差异：

| 体积 | 不同体素 | 说明 |
|---|---:|---|
| `aseg.mgz` | 107,249 | 用 SynthSeg 标签近似后续官方编辑；左/右白质 Dice 0.936/0.937，左/右皮层 0.871/0.848 |
| `aparc+aseg.mgz` | 312,356 | 共同标签的 Dice 中位数 0.345 |
| `aparc.a2009s+aseg.mgz` | 358,625 | 共同标签的 Dice 中位数 0.110 |
| `aparc.DKTatlas+aseg.mgz` | 314,950 | 共同标签的 Dice 中位数 0.378 |
| `wmparc.mgz` | 659,800 | 共同标签的 Dice 中位数 0.329 |
| `filled.mgz` | 90,420 | 来自近似白质填充 |

SynthSeg CSV 中共同的 33 个数值列，最大绝对差为 0.04 mm³；sTIV 为官方 1,280,294.20 mm³、候选 1,280,294.25 mm³。硬标签完全一致不代表最终 `aseg`、软体积和皮层指标一致。各结构的体素数与 Dice 均见 `comparison.json` 的 `per_label`。

## 表面、逐顶点指标和分区统计

双侧候选网格顶点数均高于官方，面数相对于球面拓扑的 `2V−4` 分别多 56/76；拓扑修复尚未实现。网格不具备一一对应顶点。下列空间最近距离和顶点图误差只做近似定位比较，**不是同源顶点一致性**。

| 半球与表面 | 官方/候选顶点 | 候选到官方最近顶点平均距离 | 官方到候选最近顶点平均距离 |
|---|---:|---:|---:|
| LH white | 106,622 / 117,432 | 1.000 mm | 0.898 mm |
| RH white | 105,541 / 120,276 | 0.980 mm | 0.842 mm |
| LH pial | 106,622 / 117,432 | 1.517 mm | 0.998 mm |
| RH pial | 105,541 / 120,276 | 1.521 mm | 1.018 mm |

| 顶点图 | LH 官方/候选平均 | RH 官方/候选平均 | LH/RH 空间最近点 MAE |
|---|---:|---:|---:|
| 厚度 `thickness` (mm) | 2.079 / 4.148 | 2.024 / 4.087 | 2.134 / 2.088 |
| 白质面积 `area` (mm²/vertex) | 0.706 / 0.530 | 0.711 / 0.534 | 0.383 / 0.385 |
| Pial 面积 `area.pial` | 0.872 / 1.394 | 0.876 / 1.371 | 0.883 / 0.885 |
| 中面面积 `area.mid` | 0.789 / 0.962 | 0.794 / 0.953 | 0.514 / 0.524 |
| 顶点体积 `volume` (mm³/vertex) | 1.682 / 2.583 | 1.658 / 2.564 | 1.541 / 1.556 |
| 白质曲率 `curv` | -0.025 / -0.044 | -0.023 / -0.036 | 0.074 / 0.072 |
| Pial 曲率 `curv.pial` | 0.054 / -0.830 | 0.054 / 0.008 | 1.023 / 0.173 |
| 脑沟图 `sulc` | 0.000 / 0.991 | 0.000 / 0.974 | 4.576 / 4.709 |

LH `curv.pial` 的最大空间最近点绝对差达 3460，提示近似 pial 网格存在严重局部曲率异常。额外的 mean/Gaussian、K1/K2 等逐顶点曲率图均在 `comparison.json` 中列出均值、总和、空间最近点 MAE/P95/最大差。六个 atlas 标注也有逐脑区顶点数及空间最近点 Dice：aparc 中位数 LH/RH 0.259/0.220，a2009s 0.044/0.028，DKT 0.257/0.223。

| 脑区统计文件 | 官方/候选区域行 | 共同区域厚度绝对误差中位数 | 面积绝对误差中位数 | 灰质体积绝对误差中位数 |
|---|---:|---:|---:|---:|
| LH aparc | 34 / 32 | 1.602 mm | 776.5 mm² | 4,336.5 mm³ |
| RH aparc | 34 / 33 | 1.741 mm | 736.0 mm² | 3,560.0 mm³ |
| LH a2009s | 74 / 67 | 1.712 mm | 296.0 mm² | 2,432.0 mm³ |
| RH a2009s | 74 / 70 | 1.583 mm | 445.5 mm² | 2,108.0 mm³ |
| LH DKT | 31 / 30 | 1.844 mm | 1,005.0 mm² | 5,738.0 mm³ |
| RH DKT | 31 / 31 | 1.751 mm | 891.0 mm² | 4,177.0 mm³ |

`aseg.stats` 为 45/30 区域行，`wmparc.stats` 为 70/65 行；共同区域的体积绝对误差中位数分别为 5.95/2279.5 mm³。`brainvol.stats` 的 `CortexVol` 是官方 353,949.8 mm³、候选 530,938.2 mm³，差 **+176,988.4 mm³**。所有共同区域的原值、候选值和逐列差，以及缺失区域名称，均保存在 `comparison.json` 的 `statistics` 下。

## 未完成的官方输出

严格 138 项仅有 `orig`、`orig/001`、`rawavg`、`nu`、`T1`、`synthseg.rca` 通过。52 项没有候选文件，包含辅助分割、ribbon/surface defects、BA/exvivo 标注、部分厚度与曲率辅助图及统计文件；其余 80 项未达到严格逐值/逐顶点与元数据门槛。面向核心输出的比较器比较 80 个实际文件，另列出 24 个相应缺失文件。两种分母不同，不能相减作覆盖率。

主要待改进环节是皮层拓扑修复、white/pial 几何放置与厚度、真实 sphere.reg 配准、官方后处理分割和剩余 atlas/统计输出。GCA 配准虽已由 Python 运行，实测比官方同名命令慢约 3.6 倍；当前整体时长优势来自省略或近似了大量表面与配准步骤。只有这些输出通过同输入验收后，才可将时间差称为等价重建的加速。
