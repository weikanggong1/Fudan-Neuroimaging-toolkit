# EPI→T1 边界配准（BBR）

## 功能简介

`register_bbr` 将一张 3D EPI 参考影像配准到同被试的 3D T1。函数在 T1 白质边界的内外各 2 mm 处采样 EPI 强度，优化 6 自由度刚体变换。边界目标采用 FSL FLIRT 的有符号 `tanh` 代价；无场图时不估计或校正 EPI 畸变。[FSL BBR 说明](https://fsl.fmrib.ox.ac.uk/fsl/docs/registration/flirt/bbr.html)解释了白质边界与 EPI 灰白质强度对比的要求；[epi_reg 说明](https://fsl.fmrib.ox.ac.uk/fsl/docs/registration/epi_reg.html)列出 T1、去颅骨 T1、白质分割等输入。

## Python 调用、输入输出与参数

| 参数 | 作用 |
|---|---|
| `epi` | 待配准的 **3D** EPI/SBRef NIfTI 路径或 NiBabel 影像；不接收 4D BOLD。输出变换的起点。 |
| `t1` | 同被试的 **3D** T1 NIfTI 路径或 NiBabel 影像；定义输出网格。建议先用 FNIT SynthStrip 去颅骨。 |
| `wmseg` | T1 网格的 3D 白质二值分割；形状及 affine 须与 `t1` 一致。可用 FNIT `TorchFAST` 的白质部分容积图 `pve_wm≥0.5` 生成。 |
| `init` | 可选的 EPI→T1、**FLIRT scaled-mm** 4×4 初始矩阵，可传文本文件路径或 NumPy 数组。`None` 时调用现有 FNIT `TorchFLIRT` 6 自由度 normmi 求初始矩阵，此时 `epi`、`t1` 必须是文件路径。 |
| `device` | `"cuda:0"`、`"cpu"` 等 PyTorch 设备；`None` 时优先 CUDA。GPU 默认允许 TF32，未使用 float16。 |
| `grid_search` | 默认 `True`，执行官方 `bbr.sch` 的粗网格与微网格。`False` 跳过两轮网格，仅在已有初始化附近优化；用于明确限定的调用，默认 benchmark 不关闭。 |
| `execution` | `"batched"`（默认）批量计算独立候选；`"reference"` 使用相同边界、成本、候选顺序和优化器逐项求值。 |
| `candidate_batch_size` | 独立候选每批上限，默认 128；程序再按边界点数量限制块大小。减小只影响内存与执行方式，不缩小搜索范围。 |

返回 `BBRResult`，包含 `moved`（T1 网格、float32 的 3D NiBabel NIfTI）、`matrix`（EPI→T1 的 FLIRT scaled-mm 4×4）、`moving_to_fixed_world`（同一变换的 RAS world 4×4）、`initial_cost`、`final_cost`、`boundary_points` 和 `runtime_seconds`。`phase_timings` 分别返回初始化、边界准备、粗级与细级 BBR、最终重采样耗时；`cost_evaluations`、`phase_cost_evaluations`、`host_result_transfers` 用于核对计算量和主机取值次数。`save(output=..., omat=...)` 分别写配准后的 NIfTI 和 FLIRT 格式文本矩阵。`matrix` 可直接交给接受 FLIRT `.mat` 的 FNIT 重采样函数；不能把它当作 NIfTI affine 或 RAS world 矩阵使用。应用到 4D BOLD 时，应把 BBR 与每帧运动矩阵及后续空间形变组合后只重采样一次。

```python
from fnit.fmri.bbr import register_bbr

result = register_bbr(
    epi="/absolute/path/example_func.nii.gz",   # 3D EPI/SBRef，待配准影像
    t1="/absolute/path/T1_brain.nii.gz",         # 3D 去颅骨 T1，目标空间与网格
    wmseg="/absolute/path/T1_wmseg.nii.gz",      # T1 网格白质二值掩膜
    init=None,                                    # EPI→T1 FLIRT 4×4 初始矩阵；None 自动估计
    device="cuda:0",                             # PyTorch 设备；无 GPU 可用 "cpu"
    grid_search=True,                             # 保留官方粗网格、微网格与局部优化
    execution="batched",                        # reference 可复核同一算法的串行求值
    candidate_batch_size=128,                     # 候选分块上限；内存紧张时减小
)
result.save(
    output="/absolute/path/example_func2T1.nii.gz",  # T1 网格中的 3D EPI 输出文件
    omat="/absolute/path/example_func2T1.mat",        # EPI→T1 FLIRT scaled-mm 矩阵文件
)
print(result.moving_to_fixed_world)               # EPI→T1 RAS world 4×4，供 RAS 变换链使用
print(result.final_cost)                          # 配准后的 BBR 代价值；越低越好
```

该函数只做刚体 BBR，不包括 SynthStrip、FAST、B0 场图校正、GDC 或 T1→MNI 的非线性配准。上游若已有 T1 白质分割和初始矩阵，显式传入即可复用，不需再次估计。

## 命令行调用

`register_bbr` 的独立入口是上面的 Python 函数，没有独立 BBR CLI。完整 BIDS 调用使用 [`fnit-fmri volume`](README.md#命令行调用)，通过 `--bbr-execution batched/reference` 选择同算法执行方式。

<a id="官方同输入命令"></a>

## 原软件调用

以下命令与上例使用同一张 EPI、去颅骨 T1 和白质边界；`fast` 与 `fslmaths` 两步只用于生成官方对照的分割。`bbr.sch` 来自所安装 FSL。该命令**不传场图，也不执行 GDC**。

```bash
fast -o t1_fast T1_brain.nii.gz
fslmaths t1_fast_pve_2 -thr 0.5 -bin T1_wmseg.nii.gz
flirt -in example_func.nii.gz -ref T1_brain.nii.gz \
  -dof 6 -cost normmi -omat init.mat
flirt -in example_func.nii.gz -ref T1_brain.nii.gz \
  -dof 6 -cost bbr -wmseg T1_wmseg.nii.gz \
  -init init.mat -schedule "$FSLDIR/etc/flirtsch/bbr.sch" \
  -omat example_func2T1.mat -out example_func2T1.nii.gz
```

官方 [FLIRT BBR](https://fsl.fmrib.ox.ac.uk/fsl/docs/registration/flirt/bbr.html)使用白质边界法线两侧的 EPI 强度，默认采样距离为 2 mm。当前实现按对应官方源码修正：

- 按 radiological storage、x 轴最快的顺序生成边界点，因此 `bbrstep=200` 与官方抽取同一类子集；2 mm 白质平滑采用官方核半径、零填充及逐偏移 float32 舍入，法线用 27 点加权梯度。
- 输入采用 FLIRT robust clamp 与强制 1 mm 级别的 blur 行为。成本插值使用 zero-valued corner，坐标、灰白质强度比、有符号 `tanh` 与归约保留官方对应精度；矩阵转换使用 header `pixdim`。
- 粗搜索包括 qform/sform 与已给初始矩阵两个起点、每个起点 729 个候选；之后按源 `Brent→Powell→Brent` 局部优化，再执行 729 个微网格候选和细级优化。保留网格枚举顺序、浮点参数解析和稳定的最小值选择。
- 输出复用 FNIT FLIRT 的三线性采样、边缘背景及 header 规则，避免另一套逐 slab 输出代码。

未降低 iteration、搜索范围或分辨率。Brent/Powell 存在前后依赖，局部优化仍需主机控制；候选批量化不改变这一依赖。官方 MISCMATHS 原源码的独立编译 oracle 用于单元测试，不是 FNIT 运行依赖。当前与官方是否数值一致以以下真实数据配对为准。

<a id="真实-ukb-数据对照"></a>

## 最新真实数据精度、耗时与脑图

2026-10-01 在共享 H100 PCIe 上，用一例真实 UKB 静息态 EPI 参考影像及同被试去颅骨 T1 比较。FSL 6.0.7.22 BBR 已重新运行，实际二进制子进程退出 0，矩阵、影像及 header 与固定参照逐位一致；安装包装器返回的 255 单独保留。两侧不做 GDC 或 B0 畸变校正。

### 相同初始化与白质掩膜

固定完全相同的 FSL normmi **初始**矩阵与 FSL FAST 白质分割。矩阵误差是在 T1 脑内点上计算 T1→EPI 逆变换的坐标距离；影像 r/MAE/RMSE 在 T1 脑内计算，强度单位为原 EPI 值。

| 指标 | 修改前 `7952b33` | 当前批量融合路径 |
|---|---:|---:|
| 首次 / 热调用 | 11.156 / 10.860 s | 3.581 / 1.409 s |
| 峰值 allocated / reserved | 0.060 / 0.080 GB | 0.844 / 1.103 GB |
| 逆变换位移 mean / median / p95 / RMS | 0.09056 / 0.09206 / 0.14117 / 0.09619 mm | 0.00242 / 0.00240 / 0.00400 / 0.00260 mm |
| 影像 Pearson r | 0.99966894 | 0.99999970 |
| 影像 MAE / RMSE | 87.097 / 167.529 | 2.646 / 5.014 |
| 成本求值次数 | 1754 | 2604 |

当前初始/最终成本为 0.4562233 / 0.3413588；官方最终矩阵代入**当前 FNIT 成本**为 0.3413580。修改前与当前边界/成本不同，不把两个版本的代价值直接相减。当前 schedule 补全后求值次数增加；速度提升来自独立候选批量化、缓存和融合采样，未删减搜索。修正后的非融合张量路径耗时 11.523 s、峰值分配 8.005 GB；其矩阵文件、影像、header、affine、缩放与融合结果逐位一致。

本轮 FSL BBR CPU 命令为 45.093 s，包含输入读写、启动及只记录 exec/exit 的追踪；FNIT 函数钟包含影像 ArrayProxy 读入/解压及 CPU 结果转换，排除最终写盘。GPU 与 CPU 均共享，计时边界也不同，不能据此发布稳定 CPU/GPU 加速倍数。“首次”是进程首个函数调用，CUDA 初始化已完成，未清空 Triton 磁盘缓存。

### FNIT FAST＋FLIRT＋BBR 完整配准链

同一 T1 重新生成 FNIT 白质分割和 FNIT 6-DOF normmi 初始化，再执行 BBR；精度参照仍为官方 FAST/init/BBR 结果。

| 指标 | 修改前 | 当前 |
|---|---:|---:|
| 首次 / 热调用 | 17.507 / 14.962 s | 6.932 / 5.072 s |
| 最终位移 median / p95 / RMS | 0.10059 / 0.15916 / 0.10645 mm | 0.06707 / 0.10897 / 0.07203 mm |
| 影像 Pearson r | 0.9995740 | 0.9997382 |
| 影像 MAE / RMSE | 98.212 / 190.026 | 80.034 / 148.914 |
| 峰值 allocated / reserved | 1.360 / 1.904 GB | 1.360 / 1.904 GB |

当前热调用中 FAST 为 1.329 s，初始 FLIRT 为 2.738 s，BBR 函数为 0.999 s。BBR 的阶段钟为边界准备 0.180 s、粗级搜索与优化 0.208 s、细级搜索与优化 0.551 s、最终采样 0.059 s；已显式提供初始化，函数内初始化钟仅计入读取。阶段钟采用 CPU wall，无额外 GPU fence，不能与包含它们的总钟再次相加。

### GPU profile 与回归门槛

profile 独立运行，插桩时间不进入上面的速度表。

| 完整 BBR profile | 修改前 | 当前 |
|---|---:|---:|
| CUDA kernels | 53,672 | 2,163 |
| `cudaStreamSynchronize` | 5,289 | 463 |
| H2D / D2H 事件 | 3,523 / 1,766 | 473 / 426 |
| CUDA 张量转 Python float | 1,754 | 2 |

剩余 D2H 主要用于有依赖的 Brent/Powell 主机分支；每次候选批次只返回成本向量。整张 GPU 的热调用平均利用率为 97.4% / 89.0%，包含其他作业，不能解释为 FNIT 自身利用率。报告保留完整链的 profile、所有阶段及环境。旧工具对两个单矩阵调用各把 `len(4×4)` 记为 4，共多计 6；当前成本次数采用 `BBRResult.cost_evaluations`，原测量哈希和计数更正注释保留。

回归测试覆盖官方 MISCMATHS Brent/Powell trial trace、边界/平滑/插值，以及真实 NIfTI 常见的 Fortran 布局。融合 kernel 前将驻留影像转换为连续布局；此前按 C 顺序误读 Fortran 数据的错误候选已排除。配对冷/热/profile 产物逐位一致；与 FSL 仍有上表所列误差。这是一例不含畸变校正的对照，不能验证 fieldmap BBR 或外推其他采集。匿名汇总与源码哈希见 [当前配准报告](../../validation/fmri/registration_gpu.current.public.json)。

BBR 本页没有单独发布新脑图；完整 490 帧空间输出图见[volume 精度与脑图](README.md#latest-real-benchmark)，其统计范围为完整流程，不能单独归因于 BBR。

## 最近版本与 benchmark 记录

| 版本/记录 | 变化与报告 |
|---|---|
| 2026-10-02 文档整理 | 保留当前参数、原命令、官方剩余误差与 CPU/reference；清理已替代路径的迁移说明，算法不变。 |
| 2026-10-01 批量融合路径 | 补齐原 schedule，独立候选分批和融合采样；与修正后张量路径的矩阵/影像/header 逐位相同，对 FSL 仍有上表误差，见[配准报告](../../validation/fmri/registration_gpu.current.public.json)。 |
| `7952b33` | 修改前真实配对基线，保留同一报告中的源码哈希、计数更正和完整链指标。 |

## 参考文献与原实现

- FSL FLIRT 源码：[`flirt`](https://git.fmrib.ox.ac.uk/fsl/flirt)，重点为 `costfns.cc`、`flirt.cc` 与 `flirtsch/bbr.sch`；核验版本见 [vendor 清单](../../src/fnit/_vendor_fsl/README.md)。
- 优化器：[FSL MISCMATHS](https://git.fmrib.ox.ac.uk/fsl/miscmaths) 的 `optimise.cc`；边界平滑/梯度：[FSL NEWIMAGE](https://git.fmrib.ox.ac.uk/fsl/newimage)。
- Greve DN, Fischl B. *Accurate and robust brain image alignment using boundary-based registration*. NeuroImage 48(1):63–72, 2009. [doi:10.1016/j.neuroimage.2009.06.060](https://pubmed.ncbi.nlm.nih.gov/19573611/)。
