# InvWarp CPU 迭代分配优化：源 v28

[返回功能页](README.md) · [旧新完整门槛](assets/cpu-allocation-v2-node8-20261004.public.json) · [nodecw8 官方配对](assets/cpu-allocation-v2-node8-official-20261004.public.json)

## 1. 功能简介

本次优化减少成熟 `_PullField.sample` 和 InvWarp 停止检查的 CPU 临时数组。公开 API、场坐标约定、求逆算法、最大迭代次数和停止阈值保持原合同；没有增加运行依赖。

系数取样只在 CPU prepared-source、query/matrix/sampled values 全部 FP64 且无梯度、forward AD 或 `torch.func` 的情况下，将本次新建的 affine 数组先加平移，再加插值位移。两次加法仍分别进行 FP64 舍入，输出不与输入或上一次结果共享存储。停止检查使用 CPU `aminmax`；NaN 回原 `abs().max()`，保留原结果。CUDA、混合类型和 AD 继续既有分支。

## 2. Python 调用与输入输出

没有新增公开参数。[完整参数表](README.md#2-python-调用输入输出与参数)沿用既有 API。输入是完整 3D native reference 与前向 FSL dense/intent-2007 系数，输出为 `reference.shape + (3,)` 的 float32 反场、全网格 `valid_fraction` 和求解 QC。CPU 计算中的关键坐标保持 FP64。

```bash
# 在首次导入 Torch/Numba 前设置线程上限；本次 CPU8 验证也使用 NUMBA_NUM_THREADS=8。
OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 NUMBA_NUM_THREADS=8 python inverse_case.py
```

```python
# inverse_case.py
import torch
from fnit import TorchInvWarp

torch.set_num_threads(8)  # 调用方设置 Torch 线程预算；1 核运行时同时将环境上限改为 1
native_reference_path = "/absolute/path/native_reference.nii.gz"  # 决定完整输出网格
forward_coefficient_path = "/absolute/path/forward_coeff.nii.gz"  # FNIRT 系数；保留原始 NIfTI header
inverse_output_path = "/absolute/path/inverse_warp.nii"  # 未压缩输出，与本次诊断的保存格式一致
inverse_result = TorchInvWarp(device="cpu").run(
    reference=native_reference_path,
    warp=forward_coefficient_path,
    output=inverse_output_path,
    warp_convention="relative",  # intent-2007 读取相对位移与嵌入 affine
    output_convention="relative",  # 保存 FSL scaled-mm 相对反场
    iterations=30,  # 保留最大次数，未缩短求解
    tolerance_mm=0.01,  # 原停止阈值
)
print(inverse_result.qc, inverse_result.valid_fraction)
```

## 3. 命令行调用

```bash
# FNIT 生产入口；--niter 为 FNIT 自己的最大固定点次数。
OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 NUMBA_NUM_THREADS=8 \
fnit invwarp --ref /absolute/path/native_reference.nii.gz \
  --warp /absolute/path/forward_coeff.nii.gz \
  --out /absolute/path/inverse_warp.nii --rel --niter 30 --device cpu
```

## 4. 原软件调用

```bash
# 官方参照独立执行；FSLOUTPUTTYPE=NIFTI 保持同样未压缩保存。
FSLOUTPUTTYPE=NIFTI OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 \
invwarp --ref=/absolute/path/native_reference.nii.gz \
  --warp=/absolute/path/forward_coeff.nii.gz \
  --out=/absolute/path/fsl_inverse_warp.nii --rel
```

官方 FSL 6.0.7.4 不接受 `--niter`，使用其原生求逆及停止设置。CPU8 亲和性和环境上限不表示原程序实际使用 8 核；正式报告另保存实际 CPU 时间。FNIT 固定点与原版求解器的差异见功能页。

## 5. 完整真实数据验收与阶段耗时

### 最新官方配对

nodecw8（Xeon Gold 6418H）上的新正式配对固定同一完整真实 T1 系数和输出网格。使用一次完整 pair warmup、三次 AB/BA 交替进程；已导入 API 另执行一次完整 warmup 和三次全量读取、求逆、保存。实际候选为严格 v2 冻结 overlay，其四个 Inv/Convert/Apply/normalizer runtime SHA 与组合 freeze28 相同；上述四个 runtime SHA 与当前实现相同。

| 线程上限 | 官方完整进程中位数 | FNIT 完整进程中位数 | FNIT 已导入 API 中位数（含读写） | 完整进程速度目标 |
|---|---:|---:|---:|---|
| 1 | 35.878 s | 50.899 s | 48.571 s | 未达到 |
| 8 | 36.679 s | 21.661 s | 18.441 s | 达到 |

双方亲和性与线程上限相同，原版实际 CPU 利用为约 0.98–0.99 核；8 表示资源预算。每预算的三次进程、实际 CPU 时间、输入/源码前后校验和版本见[正式机器报告](assets/cpu-allocation-v2-node8-official-20261004.public.json)。输入内容与旧新 gate 一致；每种预算的四次完整进程保存文件分别一致。CPU1 在已导入 API 内仍存在差距；以下单次分配诊断不是这组官方时钟的替代。

指定脑掩膜有 1,397,640 体素，相对官方的向量差 mean/median/p95/max 为 0.01723/0.01008/0.03361/7.31428 mm；全网格坐标分量 max 为 7.02610 mm、RMSE 为 0.32135 mm。两预算的全部 18,808,200 个值均为有限值，affine 一致。FNIT 固定点与原版原生求解器的差异保持既有范围。

### 旧新完整分配门槛

nodecw8 的两方使用相同物理核 `35,39,43,47,51,55,59,63`，CPU1 取 35，旧源是冻结 v26，新源是严格 v2 两文件 overlay。输入与 v26 官方病例逐文件核对相同。完整输出为 `162×215×180×3`，共有 18,808,200 个值；保留全部网格、30 次实际修正、阈值 0.01 mm 和未压缩 NIfTI 保存。

1/8 核各一次完整 instrumented API 的旧新保存文件 SHA-256、QC、whole-grid valid fraction、停止状态均相同；跨 1/8 核文件也相同。全部 658 个 src/tools Python 文件运行前后未变，实际导入来自相应冻结树。入口模块导入排除，函数内惰性导入/JIT 计入；没有对官方时钟计算速度比。

| 线程上限 | v26 完整 API | v2 完整 API | 累计 CPU 时间（user + system，v26 → v2） | minor faults（v26 → v2） |
|---|---:|---:|---:|---:|
| 1 | 47.165 s | 38.815 s | 46.660 → 38.270 s | 12,345,166 → 8,272,070 |
| 8 | 22.626 s | 19.642 s | 88.239 → 74.794 s | 11,846,859 → 8,295,091 |

| 线程上限 | 完整求解调用（v26 → v2） | NIfTI 保存（v26 → v2） | 峰值 RSS KiB（v26 → v2） |
|---|---:|---:|---:|
| 1 | 46.474 → 38.063 s | 0.664 → 0.742 s | 1,745,664 → 1,598,748 |
| 8 | 22.135 → 19.147 s | 0.474 → 0.483 s | 1,737,640 → 1,725,188 |

阶段是同一次完整调用内的直接计时，保存与求解分别记录。归一化的 32 次 inclusive 时钟位于求解内部，不能再相加。调度统计只记录主线程；8 核时不把主线程 runqueue 当作全进程总等待。

whole-grid `valid_fraction=0.760825`，指定脑掩膜中的 `valid_field_fraction_in_region` 使用另一分母，不能相互替代。`converged=False`、`iterations_used=30` 在两方保持相同。上述单次诊断显示分配与系统 CPU 成本下降；正式官方时钟见本节首表。nodecw10 的 v26 历史时钟保留其原节点和源码来源，不与此轮组成优化速度比。

私有病例仅发布聚合数字。可视化沿用[既有公开 CC0 T1 反场示例](README.md#公开-t1-示例与完整功能矩阵)，该脑图绑定旧 v17，其对应的误差保留原来源。源 v28 的[本地实际 CUDA 正确性门槛](../../validation/multimodal_cpu_20261004/cuda_invwarp_v28_local_20261004.public.json)包含 12 对/24 次保存，操作序列、峰值 allocation 和文件逐位一致；小网格不作 GPU 性能 benchmark。

源 v28 的[H100 完整真实回归](../../validation/multimodal_cpu_20261004/gpu_cpu_final_v28_ready_retry_20261004.public.json)另在同一 UUID、20 GB 上限和 TF32 设置下完成两组 AB/BA，共四进程，每进程完整 warmup 一次、测量三次。系数反场的 16 次保存均包含全部 18,808,200 个 float32 值；数值、header、extensions 和 affine 与起始 main 逐位一致，逐次峰值 allocation 均为 1,275,725,824 B。基线两组完整 API 中位数为 4.298/4.094 s，v28 为 3.870/3.993 s；进程启动和入口导入不计入 API，函数内惰性导入计入。同期完整六层、三阶段 TBSS 链的四张输出和对应调用显存峰值也一致：warmup 为 3,564,715,520 B，三次测量均为 3,565,562,368 B。输入与全部源码的前后 SHA、实际 UUID、逐调用设置均核对通过。这是资源变化后独立完成的源 v28 运行；共享 GPU 时钟不作稳定速度结论。

### 源 v28 的完整单核分步骤诊断

在同一正式输入、nodecw8 CPU35 和共享锁下，先完整运行一次预热，再 profile 一次完整读取、30 次修正和保存。预热、profile 与严格 v2 gate 的整个输出文件逐位一致，QC、停止状态和输入/源码前后校验通过；源 v28 的 659 个 src/tools Python 文件保持不变。[阶段时钟和 Torch operator self 记录](assets/cpu-allocation-v28-node8-profile-20261004.public.json)与上面的官方计时分开。

instrumented API 为 45.735 s，其中完整求解调用 45.185 s、保存 0.532 s；累计 CPU user/system 为 22.630/23.074 s，minor faults 为 10,071,965，major faults 为 0。主线程实际运行 45.697 s、runqueue 等待 0.027 s。该 profile 包含 profiler 与 wrapper 成本。

| 实际阶段 | 次数 | 本次 inclusive 时钟 |
|---|---:|---:|
| 完整 `PullField.sample` | 32 | 31.019 s |
| CPU 3D 插值 | 32 | 13.720 s |
| FP64 grid 归一化 | 32 | 3.694 s |
| 静态对角坐标准备 | 32 | 5.100 s |
| 其中：query finite `aminmax` | 32 | 1.013 s |
| embedded affine 矩阵乘法 | 32 | 3.273 s |
| embedded 平移就地加法 | 32 | 0.660 s |
| sampled 位移就地加法 | 32 | 1.854 s |
| correction 输出矩阵乘法 | 30 | 2.628 s |
| estimate 就地更新 | 30 | 3.473 s |
| 停止检查 | 30 | 1.116 s |
| 其中：停止量 `aminmax` | 30 | 1.108 s |

`sample` 包含插值、grid 归一化、静态坐标及 embedded 变换；finite scan 包含在静态坐标内，停止 `aminmax` 包含在停止检查内，表内时钟不能相加。32 次取样包含粗网格初始化、30 次完整迭代和最终完整 QC 取样。采样与 grid 归一化占本次完整 API 约 38.1%，finite scan 占约 2.2%，保存占约 1.2%；后续优化优先检查前两者。尚未根据该诊断更改采样器或新增 finite-bound 合同。

## 6. 更新与 benchmark 记录

| 来源 | 更新和证据 |
|---|---|
| 严格 v2，源 v28 | 两步 FP64 fresh-base 加法和 CPU 标量停止减少分配；160 项定向回归通过，覆盖实际 CUDA、NaN payload/Inf/signed zero、ULP 阈值、布局、dtype 提升、fresh/no-alias、梯度/dual/func 回退。nodecw8 完整 1/8 核保存文件与 QC/30 次停止一致；新官方完整进程 CPU1 未达、CPU8 达速度目标。 |
| 分配原型 v1 | nodecw10 完整 1/8 核文件一致，但不具有严格 v2 的全部类型和 AD guard；未收入生产。重载节点 wall time 与 nodecw8 分开保留。 |
| v26 静态坐标版本 | 完整真实官方 CPU1/8 warm+3 配对与 H100 保存文件门槛见 [v26 正式记录](../applywarp/cpu_static_diagonal_all26_official_20261004.public.json)。新诊断不重写该版时钟。 |

本次性能证据仅覆盖该完整系数反场。普通 ConvertWarp 默认单次查询仍用原算式；本次逆场时间不能写作 ConvertWarp 官方时间。新的 dense 功能回归与原矩阵路径回归属于接口证据，真实官方功能矩阵仍保留各自版本来源。

## 7. 参考文献与原软件代码库

- Andersson, Jenkinson & Smith，*Non-linear registration, aka spatial normalisation*，FMRIB Technical Report TR07JA2 (2007)，[原文](https://www.fmrib.ox.ac.uk/datasets/techrep/tr07ja2/tr07ja2.pdf)。
- [FSL FNIRT 源码（含 invwarp）](https://git.fmrib.ox.ac.uk/fsl/fnirt)；派生实现受 [FSL Software Licence 6.0](../../licenses/FSL-6.0.txt)约束。
- [完整公开聚合诊断](assets/cpu-allocation-v2-node8-20261004.public.json)；[原功能 API 与精度范围](README.md)。
