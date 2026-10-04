# 完整 105 帧 DWI→MNI：CPU 官方对照

## 1. 功能和病例来源

本例核对 `TorchApplyWarp` 将完整 EDDY 校正 DWI，通过同病例 FA 的 FNIRT 系数场重采样到 2 mm MNI 网格。固定 case 为 `applywarp_full_dwi_mni_coefficient`，CPU 使用冻结25；GPU 的最新门槛使用冻结26。两版本的普通 ApplyWarp 源码相同，26 的差异为仅重复 prepared CPU 查询使用的 ConvertWarp 静态坐标 helper。

服务器内核对了原始 AP 输入与真实处理报告的 SHA-256、实际 EDDY 产物、DWI/FA 相同网格和 affine，以及最新原版 TBSS 三阶段系数文件。既有 2 mm MNI 模板到系数的 1 mm 参考网格具有严格 `diag(2,2,2,1)` 的 voxel 变换。只公开聚合指标，病例路径、ID、影像及其哈希保留在服务器。

## 2. Python 调用、输入和输出

进程启动前设置相同的线程上限：

```bash
NUMBA_NUM_THREADS=8 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 \
  OPENBLAS_NUM_THREADS=8 python /path/benchmark_applywarp.py
```

```python
import torch
from fnit.applywarp import TorchApplyWarp

torch.set_num_threads(8)  # 本例CPU预算；进程启动前的OMP/BLAS上限也设为8
corrected_diffusion_path = "/path/eddy_corrected_dwi.nii.gz"  # 完整105帧，float32
standard_reference_path = "/path/MNI152_T1_2mm.nii.gz"  # 91×109×91、2mm MNI网格
fa_coefficient_path = "/path/FA_to_MNI_coeff.nii.gz"  # 同DWI网格FA产生的intent-2007系数
output_diffusion_path = "/path/dwi_in_MNI_2mm.nii"  # 全部105帧、float32
warp_model = TorchApplyWarp(device="cpu", frame_chunk_size=None)  # 默认分帧政策
warp_result = warp_model.run(
    input=corrected_diffusion_path,
    reference=standard_reference_path,
    warp=fa_coefficient_path,
    interpolation="trilinear",  # 与官方相同的三线性输入选项
    warp_convention="auto",  # 按系数intent解释；系数已包含初始化affine
    output_dtype="float",  # 输出float32
    output=output_diffusion_path,
)
print(warp_result.image.shape)
```

| 项目 | 实际输入或输出 |
|---|---|
| DWI | `104×104×72×105`，float32，完整 105 帧；与注册 FA 的空间网格和 affine 相同。 |
| 变换 | 完整原版 TBSS 三阶段产生的 intent-2007 FNIRT 系数，含初始化 affine；本例省略 premat/postmat。 |
| 参考 | 既有 `91×109×91`、2 mm MNI 模板，空间矩阵已现场核对。 |
| 输出 | `91×109×91×105`，94,776,045 个 float32 值，未压缩 `.nii`；保留输入 TR。 |

全部 API 字段和可选参数见 [功能页](README.md)。

## 3. 命令行调用

```bash
OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 NUMBA_NUM_THREADS=8 \
fnit applywarp --in /path/eddy_corrected_dwi.nii.gz \
  --ref /path/MNI152_T1_2mm.nii.gz --warp /path/FA_to_MNI_coeff.nii.gz \
  --interp trilinear --datatype float --device cpu \
  --out /path/dwi_in_MNI_2mm.nii
```

省略 `--frame-chunk-size` 使用默认政策，全部帧均处理。`.nii` 与 `.nii.gz` 是不同保存预算；本例双方使用 `.nii`。

## 4. 原软件调用

```bash
FSLOUTPUTTYPE=NIFTI OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 \
applywarp --in=/path/eddy_corrected_dwi.nii.gz \
  --ref=/path/MNI152_T1_2mm.nii.gz --warp=/path/FA_to_MNI_coeff.nii.gz \
  --interp=trilinear --datatype=float --out=/path/fsl_dwi_in_MNI_2mm.nii
```

实际参照为 FSL 6.0.7.4 的 native `applywarp`。FNIT 生产计算使用项目现有 PyTorch 实现；官方命令由独立 benchmark 执行。

## 5. 精度、完整耗时和分步骤

CPU1 使用物理核95；CPU8 使用 `33,37,41,45,49,53,57,61`。每组原版与 FNIT 串行执行一次完整操作，没有 warmup 或重复稳定速度估计。完整进程包含启动、导入、全输入读取、计算和全输出保存；FNIT 首次完整 API 是另一计时范围，排除进程启动及 Torch/adapter 设置，包含首次功能导入和惰性加载。

| CPU上限 | 官方完整进程 | FNIT完整进程 | FNIT首次完整API | 实际平均CPU核，官方/FNIT |
|---|---:|---:|---:|---:|
| 1 | 457.8428 s | 11.2433 s | 9.7215 s | 0.998 / 0.869 |
| 8 | 431.2784 s | 7.5706 s | 5.8086 s | 0.997 / 1.294 |

相同8核上限下，官方实际约使用一核。表中仅是本病例的单次完整观察，不给稳定加速比。

两预算所有 94,776,045 个输出值均为有限值；与官方的全网格最大差 0.3935547、MAE 0.0006471061、RMSE 0.0025949959、相对 L2 `2.08846×10⁻⁶`。逐值相同占 44.8212%。FNIT 的1/8线程保存文件互相完全相同，官方的1/8保存文件也互相完全相同。

保存比较包括 shape、float32 dtype、2 mm voxel/TR 3.6 s、`mm/sec`、qform/sform code4、slope1/intercept0 和 affine，这些字段与官方一致。参考正值区域有 857,785 个体素，作为 reference-nonzero support 单列；脑掩膜未在本 case 声明。区域内外完整指标见 [聚合公报](mni_dwi_105_20261004.public.json)。

### FNIT实际run链的独立步骤观察

另外各执行一次相同 `run()`，挂接计时器保留原读取、准备、采样和保存顺序。两个诊断输出均与各自正式候选的整个保存文件 SHA-256 一致。入口功能模块导入排除，调用内惰性导入/JIT 包含。官方二进制本次提供完整进程时钟。

| 阶段 | CPU1 | CPU8 |
|---|---:|---:|
| 输入解码、cast和有限值检查 | 3.0420 s | 3.0562 s |
| 系数、warp和查询准备 | 1.2306 s | 0.3607 s |
| └其中系数展开，已包含于上一行 | 0.7781 s | 0.2483 s |
| 全影像采样和输出构造 | 2.5033 s | 0.4589 s |
| NIfTI保存 | 1.4442 s | 1.4230 s |
| 完整instrumented API | 8.3166 s | 5.3805 s |

这些步骤来自另一单次诊断，保留独立计时边界；嵌套的系数展开不能再次加到总耗时。8线程时输入解码和保存占主要时间。

### 最新H100门槛

冻结26与起点 main 在相同 H100、共同锁及20GB allocation上限比较，全部16次完整保存的 array、header、extensions 和 affine 位级相同；每次峰值 allocation 为777,303,040 B，两版本相同。两组已warmup API中位数为基线5.8358/6.1069 s、候选5.4984/5.3950 s。GPU运行前其他作业占用61,606–66,014 MiB、利用率97–100%；这里报告共享GPU观察。完整证据见 [GPU26公报](../../validation/multimodal_cpu_20261004/gpu_inv_mni_v26_20261004.public.json)。

下图复用已公开的CC0 T1重采样例子，说明输出形式；本次临床DWI仅公开聚合数值。

![公开T1的官方和FNIT输出对照](assets/public_applywarp_comparison.png)

## 6. 最近记录

| 记录 | 操作和边界 |
|---|---|
| 本页CPU25、GPU26 | 同一FA pipeline的实际EDDY105帧，经最新原版完整TBSS系数到MNI2mm；CPU各一次完整新时钟。 |
| CPU50、GPU23 | 105帧在原`104×104×72`网格使用premat，无warp；81,768,960个值。仍保留原source和时钟。 |
| 2026-10-02历史MNI | 不同快照的历史完整DWI→MNI记录，保留原测量范围。 |

CPU50的原网格结果与本页MNI系数路径各自记录。当前源和完整数值见[JSON](mni_dwi_105_20261004.public.json)，CPU50见[报告](CPU_BENCHMARK_20261004.md)。

## 7. 原实现与参考

- [FNIT ApplyWarp源码](../../src/fnit/applywarp/core.py)、[官方配对adapter](../../tools/benchmark_multimodal_cpu_applywarp.py)。
- [FSL FNIRT User Guide：applywarp](https://fsl.fmrib.ox.ac.uk/fsl/docs/registration/fnirt/user_guide.html#applywarp)。
- [官方applywarp.cc](https://git.fmrib.ox.ac.uk/fsl/fugue/-/blob/master/applywarp.cc)和[FNIRT系数场实现](https://git.fmrib.ox.ac.uk/fsl/fnirt)。
- Andersson, Jenkinson & Smith. *Non-linear registration, aka spatial normalisation*. FMRIB Technical Report TR07JA2 (2007)，[原文](https://www.fmrib.ox.ac.uk/datasets/techrep/tr07ja2/tr07ja2.pdf)。
