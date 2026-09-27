# PyTorch MMORF

[返回首页](../../README.md) · [dMRI 参数图 pipeline](../dmri_pipeline/README.md) · [验证工件](../../validation/mmorf/README.md)

`run_mmorf` 用一套共享的非线性形变联合配准 T1 标量图和 diffusion tensor 图。`fnit-mmorf` 与 dMRI pipeline 的 MMORF 分支都调用这个函数；运行时不启动 FSL executable。

CUDA 路径使用 float32 并默认允许 TF32，不使用 float16 或 bfloat16。默认五层计划与 MMORF 0.3.2 配置一致：控制点分辨率为 32、32、16、8、4 mm，平滑为 8、8、4、2、1 mm，每层 5 次更新。当前实现包含 world-mm cubic 控制格点、reference-axis mm warp、标量/张量代价、tensor reorientation、Bkk/SPRED 正则和 LBFGS 优化路径，但仍未达到官方逐体素数值等价。

## 输入

| 参数 | 类型与 shape | 含义 |
|---|---|---|
| `moving_scalar` | 3D NIfTI 路径或 nibabel image | 个体标量图，通常为脑提取 T1w。 |
| `reference_scalar` | 3D NIfTI 路径或 nibabel image | 公共空间标量模板，同时定义输出 grid。 |
| `moving_tensor` | 4D NIfTI，`[X,Y,Z,6]` | 个体 diffusion tensor。六通道顺序必须为 `Dxx,Dxy,Dxz,Dyy,Dyz,Dzz`。 |
| `reference_tensor` | 4D NIfTI，`[X,Y,Z,6]` | 公共空间 diffusion tensor 模板。 |
| `moving_scalar_affine` | 4×4 array 或 `.mat` 路径，可省略 | `moving_scalar → reference_scalar` 的 FSL scaled-mm FLIRT 矩阵。 |
| `moving_tensor_affine` | 4×4 array 或 `.mat` 路径，可省略 | `moving_tensor → reference_scalar` 的 FSL scaled-mm FLIRT 矩阵。 |
| `reference_tensor_affine` | 4×4 array 或 `.mat` 路径，可省略 | `reference_tensor → reference_scalar` 的 FSL scaled-mm FLIRT 矩阵；已同网格时省略。 |
| `output_dir` | 目录 | 一个受试者的五个输出文件。 |
| `device` | `cpu`、`cuda` 或 `cuda:N` | 省略时由 FNIT 选择可用设备。 |
| `config` | `MMORFConfig`，可省略 | 默认使用上述五层计划。 |
| `overwrite` | bool | 默认 `False`；已有任一结果时停止。 |

所有图像必须为有限值。省略 affine 表示 FSL scaled-mm identity，不表示忽略 NIfTI affine；FNIT 仍把 voxel、world 和 FLIRT 坐标完整换算到公共空间。

## Python 单被试调用

```python
from fnit import run_mmorf

result = run_mmorf(
    moving_scalar="t1_brain.nii.gz",  # 输入：个体脑提取 3D T1w
    reference_scalar="MNI152_T1_1mm_brain.nii.gz",  # 输入：T1 模板及输出网格
    moving_tensor="dti_tensor.nii.gz",  # 输入：个体 [X,Y,Z,6] FSL tensor
    reference_tensor="FSL_HCP1065_tensor_1mm.nii.gz",  # 输入：公共空间六通道 tensor
    output_dir="mmorf",  # 输出：本受试者的五文件结果目录
    moving_scalar_affine="t1_to_MNI.mat",  # 输入：moving T1 -> reference 的 FSL scaled-mm 矩阵
    moving_tensor_affine="FA_to_MNI.mat",  # 输入：moving tensor -> reference 的 FSL scaled-mm 矩阵
    reference_tensor_affine=None,  # 输入：reference tensor 已与 T1 模板同网格，无需额外矩阵
    device="cuda:0",  # 运行设备：第一张 CUDA GPU
    config=None,  # 配置：使用 MMORFConfig 默认五层计划
    overwrite=False,  # 写盘策略：不覆盖已有文件
)
```

这次调用完成一名受试者的联合配准并写盘。`result` 为 `MMORFResult`：

| Python 属性 | 类型 | 内容 |
|---|---|---|
| `warp` | nibabel NIfTI | reference grid 上的三通道 relative pull displacement。 |
| `jacobian` | nibabel NIfTI | nonlinear warp 的 Jacobian determinant。 |
| `warped_scalar` | nibabel NIfTI | moving scalar 的 reference-grid cubic 重采样结果。 |
| `warped_tensor` | nibabel NIfTI | moving tensor 的六通道 cubic 重采样结果；这是 FNIT 便利输出。 |
| `qc` | dict | 设备、TF32、层级损失、closure 次数、耗时、峰值显存和非等价边界。 |

只在内存中计算时使用低层接口：

```python
from fnit import TorchMMORF

model = TorchMMORF(
    device="cuda:0",  # 运行设备
    config=None,  # 使用默认 MMORFConfig
)
result = model(
    moving_scalar="t1_brain.nii.gz",  # 输入：个体脑提取 T1w
    reference_scalar="MNI152_T1_1mm_brain.nii.gz",  # 输入：T1 模板及输出网格
    moving_tensor="dti_tensor.nii.gz",  # 输入：个体六通道 tensor
    reference_tensor="FSL_HCP1065_tensor_1mm.nii.gz",  # 输入：公共空间六通道 tensor
    moving_scalar_affine="t1_to_MNI.mat",  # 输入：moving T1 -> reference 矩阵
    moving_tensor_affine="FA_to_MNI.mat",  # 输入：moving tensor -> reference 矩阵
    reference_tensor_affine=None,  # 输入：reference tensor 已与 T1 模板同网格
)
```

应用已有 MMORF warp：

```python
from fnit import apply_mmorf_warp

warped_fa = apply_mmorf_warp(
    image="dti_FA.nii.gz",  # 输入：待重采样的 native FA
    reference="MNI152_T1_1mm_brain.nii.gz",  # 输入：输出网格
    warp=result.warp,  # 输入：reference-grid、reference-axis mm pull warp
    affine="FA_to_MNI.mat",  # 输入：FA -> reference 的 FSL scaled-mm 矩阵
    device="cuda:0",  # 运行设备
    interpolation="linear",  # 插值：连续参数图使用线性插值
)
```

`apply_mmorf_warp` 的 `image` 是待重采样图，`reference` 决定输出 grid，`warp` 必须与 reference 同 shape 和 affine，`affine` 是 image → reference 的 FLIRT 矩阵。`interpolation` 可为 `linear`、`nearest` 或 `cubic`；dMRI 参数图 pipeline 使用 linear，MMORF 标量输出使用 cubic。

## 命令行单被试调用

```bash
fnit-mmorf \
  --mov-scalar t1_brain.nii.gz \
  --ref-scalar MNI152_T1_1mm_brain.nii.gz \
  --mov-tensor dti_tensor.nii.gz \
  --ref-tensor FSL_HCP1065_tensor_1mm.nii.gz \
  --aff-mov-scalar t1_to_MNI.mat \
  --aff-mov-tensor FA_to_MNI.mat \
  -o mmorf \
  --device cuda:0
```

各行依次指定个体 T1、T1 模板、个体 tensor、tensor 模板、两个 input → reference FLIRT 初始化矩阵、输出目录和 PyTorch 设备。reference tensor 不在 T1 模板 grid 时再加 `--aff-ref-tensor ref_tensor_to_MNI.mat`。覆盖已有结果必须显式加 `--overwrite`。

## 输出目录与坐标定义

```text
mmorf/
├── mmorf_warp.nii.gz
├── mmorf_jacobian.nii.gz
├── mmorf_warped_scalar.nii.gz
├── mmorf_warped_tensor.nii.gz
└── mmorf_report.json
```

| 文件 | shape | dtype | 定义 |
|---|---|---|---|
| `mmorf_warp.nii.gz` | `[Xref,Yref,Zref,3]` | float32 | relative pull displacement；三通道为沿 reference image axes 的毫米位移。 |
| `mmorf_jacobian.nii.gz` | `[Xref,Yref,Zref]` | float32 | `det(I + ∂u/∂x)`，只含 nonlinear warp。 |
| `mmorf_warped_scalar.nii.gz` | `[Xref,Yref,Zref]` | float32 | moving scalar 的公共空间结果。 |
| `mmorf_warped_tensor.nii.gz` | `[Xref,Yref,Zref,6]` | float32 | 六通道 moving tensor 的公共空间采样结果。 |
| `mmorf_report.json` | JSON | — | 配置、设备、五层 QC、耗时和显存。 |

MMORF warp 与 FNIRT/applywarp 的 coefficient/dense-warp 文件不是同一种坐标合同，不能互换。设 `u_axis(x)` 为保存的三通道值，`Rref` 为 reference affine 去除 voxel scaling 后的轴方向矩阵，则公共 world 位移为 `Rref · u_axis(x)`；pull 位置再通过 input affine 的逆映射到 moving voxel。这个定义已用 anisotropic 和 oblique affine 单元测试验证。

官方 MMORF 只直接输出 warp 和 Jacobian；debug 路径可保存 warped scalar。`mmorf_warped_tensor.nii.gz` 是 FNIT 的便利输出，本次官方对照没有 warped-tensor oracle，因此不声明它逐体素等价。

## 对应的官方 MMORF 0.3.2 调用

将相同输入写入 `multimodal.ini`：

```ini
warp_res_init           = 32
warp_scaling            = 1 1 2 2 2
img_warp_space          = MNI152_T1_1mm_brain.nii.gz
lambda_reg              = 4.0e5 3.7e-1 3.1e-1 2.6e-1 2.2e-1
hires                   = 6
optimiser_lowres        = LM
optimiser_hires         = MM
optimiser_max_it_lowres = 5
optimiser_max_it_hires  = 5
warp_out                = official_warp
jac_det_out             = official_jacobian

img_ref_scalar      = MNI152_T1_1mm_brain.nii.gz
img_mov_scalar      = t1_brain.nii.gz
aff_ref_scalar      = identity.mat
aff_mov_scalar      = t1_to_MNI.mat
use_implicit_mask   = 0
use_mask_ref_scalar = 0 0 0 0 0
use_mask_mov_scalar = 0 0 0 0 0
mask_ref_scalar     = NULL
mask_mov_scalar     = NULL
fwhm_ref_scalar     = 8 8 4 2 1
fwhm_mov_scalar     = 8 8 4 2 1
lambda_scalar       = 1 1 1 1 1
estimate_bias       = 0
bias_res_init       = 32
lambda_bias_reg     = 1e9 1e9 1e9 1e9 1e9

img_ref_tensor      = FSL_HCP1065_tensor_1mm.nii.gz
img_mov_tensor      = dti_tensor.nii.gz
aff_ref_tensor      = identity.mat
aff_mov_tensor      = FA_to_MNI.mat
use_mask_ref_tensor = 0 0 0 0 0
use_mask_mov_tensor = 0 0 0 0 0
mask_ref_tensor     = NULL
mask_mov_tensor     = NULL
fwhm_ref_tensor     = 8 8 4 2 1
fwhm_mov_tensor     = 8 8 4 2 1
lambda_tensor       = 1 1 1 1 1
```

```bash
mmorf --config multimodal.ini
```

`run_mmorf` 的四幅输入和三份 affine 分别对应同名 `img_*` 与 `aff_*` 行；默认 `MMORFConfig` 对应 `warp_res_init`、`warp_scaling`、FWHM、`lambda_reg` 和五次更新。FNIT 输出目录和设备选项属于 Python 包接口。

## 源码对应与剩余差异

| MMORF 0.3.2 行为 | 当前 TorchMMORF |
|---|---|
| world-mm cubic B-spline warp | 已实现，并按整数 scaling 精确细化控制格点。 |
| robust world sample grid 和采样频率 | 已实现。 |
| cubic image coefficient/pull sampling | 已实现；使用 SciPy mirror prefilter 与 PyTorch 64-neighbour sampler。 |
| robust scalar normalization | 已实现。 |
| `0.5*(1+detJ)` symmetric quadrature | 已实现；按官方 gradient/Hessian 语义把该权重视为固定权重。 |
| full tensor Frobenius cost | 已实现。 |
| affine 与 local finite-strain tensor rotation | 已实现。 |
| 首层 Bkk、后续 SPRED | 已实现。 |
| coarse full / fine diagonal Gauss–Newton Hessian，Levenberg damping | 未等价；当前使用 LBFGS strong-Wolfe。 |
| analytic tensor-rotation gradient/Hessian | 当前由 PyTorch autograd 求导。 |
| analytic dense spline Jacobian | 当前输出使用 finite differences。 |

因此 `mmorf_report.json` 保持 `mmorf_numerically_equivalent=false`。文件角色、grid、dtype、warp 方向和单位已对齐；数值误差仍明显大于浮点舍入。

## 真实数据 benchmark

一例去标识真实 T1w、native DTI tensor 与九张 DTI/NODDI 参数图分别输入 FSL MMORF 0.3.2 和当前 TorchMMORF。两边使用同一 T1/tensor 模板及同一 FLIRT 初始化矩阵。公开仓库只保留汇总指标和去标识切片。

MNI 脑区内的直接输出比较：

| 输出 | Pearson r | MAE | RMSE |
|---|---:|---:|---:|
| warp，三通道合并 | 0.969381 | 0.393810 mm | 0.614697 mm |
| Jacobian | 0.947671 | 0.059807 | 0.097992 |
| warped T1 scalar | 0.932828 | 65.4819 | 107.3607 |

warp 三个分量的 `r` 为 `0.982270 / 0.957824 / 0.972309`。warp RMS 为 FNIT `1.9393 mm`、官方 `2.2881 mm`；Jacobian 范围为 FNIT `0.106–6.035`、官方 `0.234–3.866`。shape、affine 和 float32 dtype 均匹配。最终 warp NIfTI intent 也已改为与官方相同的 `none`。

九图比较使用相同 FNIT linear sampler、相同 affine 和相同 native input，只替换官方或 FNIT 估计的 warp，因此隔离 registration 差异：

| map | Pearson r | MAE | RMSE |
|---|---:|---:|---:|
| FA | 0.976514 | 0.017893 | 0.031733 |
| MD | 0.960073 | 8.03e-5 | 1.66e-4 |
| L1 | 0.955644 | 8.50e-5 | 1.79e-4 |
| L2 | 0.961048 | 8.26e-5 | 1.67e-4 |
| L3 | 0.964104 | 8.11e-5 | 1.60e-4 |
| MO | 0.919880 | 0.107869 | 0.170607 |
| ICVF | 0.955881 | 0.029394 | 0.058286 |
| OD | 0.954746 | 0.036146 | 0.066666 |
| ISOVF | 0.959405 | 0.040910 | 0.079039 |

固定官方 warp 后，FNIT cubic sampler 与官方 debug warped scalar 的 `r=0.993795`。官方 one-step warp 投影到当前 32 mm control lattice 后 dense `r=0.997249`。这两项排除了 warp 方向、单位、affine 顺序和格点表达能力作为主要剩余原因。

| 实现 | 设备 | 核心计算 | 注册调用含写盘 | 九图应用 | peak CUDA allocation |
|---|---|---:|---:|---:|---:|
| FSL MMORF 0.3.2 | H100 GPU | 925.13 s | 947.62 s | 未计入 | 未记录 |
| FNIT TorchMMORF | H100 GPU | 33.62 s | 47.80 s | 5.10 s | 12.318 GB |

计时来自共享节点，负载未隔离；两种优化器仍不数值等价，因此不把时间比写成“等价实现加速倍数”。完整数值、边界和源码哈希见[公开报告](../../validation/mmorf/report.public.json)。

![官方 warp 与 TorchMMORF 在同一真实病例上的 warped T1 和 FA](figures/mmorf_fsl_comparison.png)

图中两列结果使用同一 grid 和显示范围，右列为绝对差；T1 与 FA 都由相同 FNIT sampler 生成，只改变 warp。数值表使用完整 3D 数据，图只用于查看空间分布。生成脚本见 [`plot_mmorf.py`](../../validation/mmorf/plot_mmorf.py)。

## 来源与许可

实现对照仓库内 MMORF 0.3.2 commit `1c1c13b8368f05e1a79a6dafe919d6b61df36bd6` 的未修改源码快照。完整文件哈希见 [`_vendor_fsl/manifest.json`](../../src/fnit/_vendor_fsl/manifest.json)。该部分受 [`FSL Software Licence, Release 6.0`](../../licenses/FSL-6.0.txt) 约束，仅用于许可允许的非商业用途；FNIT 不是官方 FSL 发布。
