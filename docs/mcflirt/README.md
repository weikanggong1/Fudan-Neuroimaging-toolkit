# MCFLIRT 运动校正

`TorchMCFLIRT` 将四维 BOLD 的每一帧配准到 SBRef，或默认的中间帧。估计和重采样均在 FNIT 内完成，运行时不启动 FSL。六自由度变换使用 FSL scaled-mm 坐标，每个矩阵的方向是 input frame → reference。

实现按 [MCFLIRT 2111.0 源码](https://git.fmrib.ox.ac.uk/fsl/mcflirt/-/blob/2111.0/mcflirt.cc)的默认三阶段路径：参考图直接重采样到 8、4、4 mm；边缘使用 1 mm 降权；每阶段一次 Brent 坐标轮回，容差乘数依次为 0.8、0.8、0.1。8 mm 阶段沿时间传递上一帧的估计，后两阶段从各帧已有估计继续。使用外部参考图时，从第 0 帧开始。优化不使用脑掩膜。

FSL `p_normcorr_smoothed` 的 `num` 和 `numA` 在行、层之间连续累加。FNIT 保留此计数定义及其方差公式；替换成常见的加权 Pearson 会改变优化目标。参考金字塔没有额外 Gaussian 预滤波，最终样条采样也不增加 FLIRT `applyxfm` 的降采样滤波。

## 从 TorchFLIRT 复用的加速

`TorchMCFLIRT` 复用项目中成熟的 `TorchFLIRT` 变换组合、Brent 坐标优化、三线性插值和精确 float32 CUDA 运算。MCFLIRT 的目标函数与 FLIRT 常用的相关比、互信息不同；本模块继续使用上述 NCC 定义，并保留每行有效 x 范围及逐次 float32 加法生成采样坐标的顺序。

CUDA 路径将行范围计算、坐标生成、三线性采样、1 mm 边界降权和参考值读取合并到一个 Triton kernel，复用三个 float32 缓冲区供原 NCC 归约使用。这样减少每次 cost 的 GPU kernel 启动、临时张量和小张量的主机同步。变换仍先在 CPU 以 float64 组合、求逆，再转换为原实现使用的 12 个 float32 系数；融合采样使用逐步舍入的 float32 运算。

同一次调用还复用各帧的强度重心。若完整 float32 BOLD 的大小不超过 4 GiB 且不超过当前空闲显存的四分之一，CUDA 路径会在三个阶段间缓存已上传的帧；更大的输入逐帧上传。缓存保存完整 float32 数据。三阶段的执行顺序、第一阶段的相邻帧初值传递、Brent 求解和 NCC 归约沿用既有实现。

Python、命令行参数及输出结构均无需调整。融合采样需要 Triton；项目 [Conda 环境](../../environment.yml)已包含与 PyTorch 2.5.1 对应的 `triton==3.1.0`。CPU 路径，以及无法导入 Triton 时的 CUDA 采样，使用既有 PyTorch 张量实现。20 项 CUDA [采样合同测试](../../tests/mcflirt/test_fused_cost.py)已通过：融合采样的参考值、moving 值、权重及对应 NCC 与原张量路径逐元素 float32 位模式一致，覆盖正负平移、旋转、FOV 边界、接近零的方向和缓冲区复用。真实 BOLD 的精度与耗时见下文。

## 输入、参数与输出

`TorchMCFLIRT(device=None).run(...)` 或 `TorchMCFLIRT(...)(...)` 都可调用。

| 参数 | 含义 |
|---|---|
| `device` | `cpu`、`cuda`、`cuda:1` 等；不指定时优先 CUDA。计算使用 float32，变换组合和参数解析使用 float64；TF32 默认开启。 |
| `input_bold` | 四维 NIfTI 路径或 nibabel 图像，轴为 X×Y×Z×T。 |
| `reference` | 同网格的三维 NIfTI 路径或图像；`None` 使用第 `T//2` 帧。 |
| `stages` | 1、2 或 3；默认 3，对应 8/4/4 mm。 |
| `stage_iterations` | 每阶段的坐标轮回次数，默认 `(1, 1, 1)`；增加次数改变原程序默认优化流程。0 跳过该阶段。 |
| `resample` | 默认 `False` 只估计矩阵；`True` 还生成校正图。指定 `output` 也会重采样。 |
| `interpolation` | 最终采样 `linear` 或 `spline`。Python 和命令行均与原 MCFLIRT 一样默认 `linear`；UKB fMRI 路径显式选择 `spline`。优化阶段始终三线性。 |
| `output` | 校正图路径或前缀；不指定时只返回对象，不写文件。 `.nii`、`.nii.gz` 后缀从输出前缀去除，校正图保存为 `.nii.gz`。 |
| `mats` | 默认 `False`；为 `True` 时生成 `<prefix>.mat/MAT_0000` 等文件，需设置 `output`。 |
| `plots` | 默认 `False`；为 `True` 时生成 `<prefix>.par`，需设置 `output`。 |
| `rmsrel`、`rmsabs` | 默认 `False`；输出相邻帧、相对 identity 的 80 mm 球体 RMS 位移及均值，需设置 `output`。 |
| `overwrite` | 默认 `False`，保护已存在输出；为 `True` 时覆盖指定输出。 |

返回 `MCFLIRTResult`：

| 属性 | 内容 |
|---|---|
| `matrices` | T×4×4 的 NumPy float64 数组，逐帧 input→reference 的 FSL scaled-mm 矩阵。 |
| `parameters` | T×6 数组，依次为三个 Euler 转角（弧度）和三个平移（mm）。平移围绕 reference 的强度重心定义，对应原 `.par`。 |
| `reference` | 实际使用的三维 nibabel 参考图。 |
| `corrected` | 未请求重采样时为 `None`；否则为 reference 网格、保留输入时间轴和 TR 的 NIfTI。 |
| `cost_evaluations` | 本次 NCC 目标函数调用次数。 |
| `output_paths` | 已请求的输出文件路径字典；未指定 `output` 时为空字典。 |

整数图保存时沿用 NEWIMAGE 的向零截断；unsigned 16-bit 输入提升为 signed int32。uint32、int64 和 uint64 输入按原程序提升为 float32；int8 输出保存为 uint8，float64 输出保留 float64。这与把校正值四舍五入或始终保存 float32 不同。当前支持同网格三维参考、六自由度 Euler、默认 NCC、1–3 阶段和三线性/三阶样条最终采样。2D、小于 20 mm 的 z 向 FOV、`meanvol`、第四阶段 sinc 优化和其它 cost 不在本接口范围内，不能将上述结果推广为这些模式的等价性结论。

## 单被试 Python 调用

```python
from fnit import TorchMCFLIRT

# 以 SBRef 为参考校正 raw BOLD，写出校正图、每帧矩阵和六列运动参数。
motion_result = TorchMCFLIRT(device="cuda:1").run(
    input_bold="sub-01_task-rest_bold.nii.gz",
    reference="sub-01_task-rest_sbref.nii.gz",
    output="motion/prefiltered_func_data_mcf",
    mats=True,
    plots=True,
    rmsrel=True,
    rmsabs=True,
    interpolation="spline",
)
frame_to_reference_matrices = motion_result.matrices
motion_parameters_radians_mm = motion_result.parameters
motion_corrected_bold = motion_result.corrected
```

只估计变换、不写出或采样 BOLD：

```python
from fnit import TorchMCFLIRT

# fMRI pipeline 调用这个类取得矩阵，再完成规定的重采样和后续清理。
motion_result = TorchMCFLIRT(device="cuda:1").run(
    input_bold="sub-01_task-rest_bold.nii.gz",
    reference="sub-01_task-rest_sbref.nii.gz",
    resample=False,
)
```

## 单被试命令行及对应原命令

```bash
# 将 raw BOLD 配准到 SBRef，最终用样条采样；保存矩阵、参数及两类 RMS。
fnit mcflirt -in sub-01_task-rest_bold.nii.gz \
  -reffile sub-01_task-rest_sbref.nii.gz \
  -out motion/prefiltered_func_data_mcf \
  -mats -plots -rmsrel -rmsabs -spline_final --device cuda:1
```

对应的原 FSL benchmark 命令：

```bash
mcflirt -in sub-01_task-rest_bold.nii.gz \
  -reffile sub-01_task-rest_sbref.nii.gz \
  -out motion/prefiltered_func_data_mcf \
  -mats -plots -rmsrel -rmsabs -spline_final
```

省略 `-reffile` 使用中间帧；省略 `-spline_final` 使用三线性最终采样。`-mats`、`-plots` 和 RMS 开关只控制文件输出，不改变估计流程。

## 真实数据核对

本次优化用同一例真实 BOLD 的完整 490 帧，在共享 H100 PCIe、`cuda:0`、8 线程、TF32 开启的环境验证，三个阶段均为一次坐标轮回，最终使用样条采样。测量源码冻结于 `7456251842b5b6fac4affc95c83ba61265ad0a0a`；对照为冻结版本 `1eb9c417` 的完整 GPU 输出。输入、源码及输出 SHA-256 和计时边界见[优化报告](../../validation/mcflirt/gpu_optimization.public.json)。

| 核对内容 | 结果与计时范围 |
|---|---|
| 完整 490 帧 `TorchMCFLIRT.run` | 159.27 秒，45,972 次 cost；包含四维输入读取/解压、准备、估计、样条采样和输出 dtype 转换，未写盘。 |
| 上述调用中的最终样条采样 | 24.83 秒；由同步后的 `apply_motion_warp` 调用单独记录。 |
| 从 API 总时间扣除最终采样 | 134.45 秒；包含输入读取、准备、估计及 dtype 转换。 |
| API 加私有验证文件写盘 | 185.00 秒；另外保存完整精度 NumPy 数组、校正 NIfTI、矩阵文本及 `.par`，输出集合与原命令不同。 |
| 固定脑掩膜的 FEAT 后续处理 | 17.91 秒；包含均值/掩膜准备和保存、缩放、高通及结果写盘；与前项合计 202.91 秒。该测量调用这些子函数，范围不含脑提取及完整 `run_feat_core` 入口。 |
| CUDA 显存峰值 | allocated 1.150 GB，reserved 1.372 GB，均小于 20 GB；计算与帧缓存使用完整 float32。 |
| 与冻结 FNIT 的矩阵及参数文本 | 全部 490 帧写出的 `MAT_*` 和 `.par` 逐值相同，文件 SHA-256 相同。内存矩阵与冻结文本的最大差为 4.44×10⁻¹²，来自文本保存精度。 |
| 与冻结 FNIT 的校正图 | 全部 242,851,840 个体素时间点的解码值相同，RMSE 0；输出为 int32。 |
| 与冻结 FNIT 的固定掩膜高通结果 | 全部 242,851,840 个体素时间点的解码值相同，RMSE 0；输出为 float32。 |

这次验证确认优化保留冻结 FNIT 的完整运动输出。校正图与原 [GPU 对照报告](../../validation/mcflirt/full490_gpu.public.json)中的图像逐元素相同，因此沿用该报告相对原 FSL 的精度统计和下方脑图：

| 与原 FSL MCFLIRT 比较，完整 490 帧 | 结果 |
|---|---|
| 脑内 pull RMS | 逐帧均值 0.00881 mm、中位数 0.00816 mm、p95 0.01643 mm、最大 0.03008 mm；最坏为第 281 帧，其体素距离 p95 为 0.04915 mm。 |
| 校正图 | 脑内时间 r 均值 0.999578、中位数 0.999834；4D RMSE 8.21084，18.26% 的脑内体素时间点整数值相同。原和 FNIT 输出均为 int32。 |
| temporal SD 图 | EPI 空间 SD 图 r 为 0.99999784，RMSE 为 0.35877；原/FNIT 脑内平均 SD 为 239.3134/239.3228。 |

六列 `.par` 与原程序的对照也沿用上述完整 GPU 结果：

| 参数 | Pearson r | RMSE |
|---|---:|---:|
| x 转角 | 0.999600 | 0.0000831 rad |
| y 转角 | 0.996694 | 0.0001009 rad |
| z 转角 | 0.990908 | 0.0001198 rad |
| x 平移 | 0.999202 | 0.002909 mm |
| y 平移 | 0.999794 | 0.003377 mm |
| z 平移 | 0.999756 | 0.002953 mm |

冻结流程的[历史 profile](../../validation/fmri/feat_profile.public.json)曾记录运动 API 709.11 秒、其中采样 30.74 秒；它带有额外包装调用，与当前计时没有匹配共享 GPU 负载和缓存状态，作为瓶颈定位记录保留。原 MCFLIRT 历史独立命令为 326.10 秒，包含最终采样和写盘。以上计时展示各次实际运行的耗时和范围，未建立受控的固定加速比。历史 [CPU 490 帧估计](../../validation/mcflirt/full490_cpu.public.json)为 4 线程、387.60 秒、45,794 次 cost，仅包含解压、准备和估计；这不是本次优化后的 CPU 重测。旧的前 8 帧控制和采样诊断集中在[独立验证说明](../../validation/mcflirt/README.md)，其耗时不作为当前完整体积性能。

复测请使用冻结流程实际保存的参考图，并核对报告中的 SHA-256 和 NIfTI `pixdim`。本例原始 SBRef 与 FEAT 保存参考图的像素、affine 相同，z 向 voxel size 分别约为 2.3999999 与 2.3999996 mm；微小头信息差异仍会改变 float32 采样与平坦 NCC 最小值的选择。

原软件核对所用 MCFLIRT 为 2111.0、NEWIMAGE 2601.0、MISCMATHS 2412.6；对应 `costfns.cc`、`optimise.cc` 与项目审计版本一致。相对原 FSL 的矩阵和校正图仍有上表所列余差；这次逐元素一致性结论针对冻结 FNIT 输出。完整 fMRI 流程的处理边界见[全流程对照](../../validation/fmri/matched_native.md)。

## 真实数据图示

![原 FSL MCFLIRT 和 FNIT 的均值、temporal SD 与时间相关图](figures/motion_correction_mni.png)

同一例 490 帧 BOLD。均值、temporal SD（`ddof=0`）及逐体素时间 r 先在 EPI 空间计算，再用同一原 BBR 和 FNIRT pull 将三维指标图变换到 MNI，仅用于展示。均值和 SD 使用各自统一的色阶；时间 r 的色阶为 0.995–1，低于下限的体素显示为下限颜色。这里的 SD 不是将四维 BOLD 重采样到 MNI 后重新计算的 SD。没有额外空间平滑，只有获授权的去标识化模板空间 PNG 被公开。复现命令见[独立验证说明](../../validation/mcflirt/README.md)，图示实现为 [render_comparison.py](../../validation/mcflirt/render_comparison.py)。

## 最近版本更新记录

| 版本与日期 | 功能变化 | 真实数据 benchmark 与核对 |
|---|---|---|
| `7456251`，2026-10-01 | 复用 TorchFLIRT 的精确 float32 CUDA 运算，融合每次 NCC 的采样准备；复用重心，并在显存预算内缓存 float32 帧。Python/CLI 接口及三阶段估计顺序沿用既有实现。 | 完整 490 帧运动 API 为 **159.27 秒、45,972 次 cost**，包含读取、估计、最终样条采样和 dtype 转换，未写盘。矩阵/参数文本、校正图及固定掩膜高通结果与冻结 FNIT 逐值相同；详见[当前优化报告](../../validation/mcflirt/gpu_optimization.public.json)。 |
| 冻结状态 `1eb9c417`，2026-10-01 | 数值匹配修复后的基线：保留源 NCC 计数、逐次 float32 坐标累加及输出数据类型，保存完整 490 帧 GPU 对照。 | 相对原 FSL，脑内 pull RMS 均值 0.00881 mm，校正图脑内时间 r 均值 0.99957825。历史 **973.12 秒是 FEAT 阶段合计**，包含运动校正、掩膜准备、缩放和高通；MCFLIRT 估计与采样未单独计时。详见[冻结 GPU 报告](../../validation/mcflirt/full490_gpu.public.json)。 |

固定变换的 CPU/冻结 CUDA/融合 CUDA 代价微基准见[标量报告](../../validation/mcflirt/cost_optimization.public.json)及[复现说明](../../validation/mcflirt/README.md#单次代价函数微基准)。该测量已预热、采用共享 GPU 上的交替调用，范围仅为单次 cost；完整运动 API 的计时边界以上表和各自报告为准。

## 许可证与源码来源

本模块是基于 FSL 源码改写的 Python/PyTorch 实现，沿用 [FSL Software Licence, Release 6.0](../../licenses/FSL-6.0.txt) 的非商业使用条款。许可证要求在无财务回报的再分发中向接收者保留条款，并随产品提供原始及修改后的源代码。因此，本包同时提供改写后的 `src/fnit/mcflirt/` 和官方 tag `2111.0` 的[完整六文件源码快照](../../src/fnit/_vendor_fsl/sources/mcflirt-2111.0/)；原始文件未修改，FNIT 不编译或执行这些文件。

官方提交为 `fa24cb88fba970fb9adc959713fee57bf3706d3e`，Git tree 为 `ee503629cf36f06a698eb4bfc516e2ad2df8feff`。官方仓库、无前缀 `git archive --format=tar` 的 SHA-256、逐文件大小及 SHA-256 见[源码清单](../../src/fnit/_vendor_fsl/manifest.json)；共享 NEWIMAGE/MISCMATHS 来源及许可见[vendor 说明](../../src/fnit/_vendor_fsl/README.md)和[第三方声明](../../THIRD_PARTY_NOTICES.md)。这些来源记录不将 FNIT 标记为官方 FSL，也不扩大上文已测量的数值范围。

参考：[FSL MCFLIRT 文档](https://fsl.fmrib.ox.ac.uk/fsl/docs/registration/mcflirt.html)；Jenkinson M, Bannister P, Brady M, Smith S. Improved optimization for the robust and accurate linear registration and motion correction of brain images. *NeuroImage* 17:825–841, 2002. [DOI](https://doi.org/10.1006/nimg.2002.1132)。
