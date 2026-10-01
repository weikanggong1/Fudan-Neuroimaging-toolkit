# MCFLIRT 运动校正

`TorchMCFLIRT` 将四维 BOLD 的每一帧配准到 SBRef，或默认的中间帧。估计和重采样均在 FNIT 内完成，运行时不启动 FSL。六自由度变换使用 FSL scaled-mm 坐标，每个矩阵的方向是 input frame → reference。

实现按 [MCFLIRT 2111.0 源码](https://git.fmrib.ox.ac.uk/fsl/mcflirt/-/blob/2111.0/mcflirt.cc)的默认三阶段路径：参考图直接重采样到 8、4、4 mm；边缘使用 1 mm 降权；每阶段一次 Brent 坐标轮回，容差乘数依次为 0.8、0.8、0.1。8 mm 阶段沿时间传递上一帧的估计，后两阶段从各帧已有估计继续。使用外部参考图时，从第 0 帧开始。优化不使用脑掩膜。

FSL `p_normcorr_smoothed` 的 `num` 和 `numA` 在行、层之间连续累加。FNIT 保留此计数定义及其方差公式；替换成常见的加权 Pearson 会改变优化目标。参考金字塔没有额外 Gaussian 预滤波，最终样条采样也不增加 FLIRT `applyxfm` 的降采样滤波。

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

已使用同一例真实 BOLD/SBRef 做阶段控制。原参考为完整 490 帧 MCFLIRT 运行；局部控制使用前 8 帧，其中只比较前 7 帧，避免源程序最后一帧的 8 mm 初值规则造成不同实验条件。

| 核对内容 | 结果 |
|---|---|
| 原库 8/4 mm 参考重采样对照 | 数组最大绝对差 0，moving COG 一致。 |
| 同帧、同矩阵的原库 NCC | 8 个 cost 最大绝对差 1.79×10⁻⁷。 |
| CPU，前 7 帧脑内 pull RMS | 每帧 0.0029–0.0144 mm；最坏体素距离 p95 0.0192 mm。 |
| CUDA，前 7 帧脑内 pull RMS | 每帧 0.0011–0.0259 mm；最坏体素距离 p95 0.0390 mm。 |
| 前 8 帧 CPU 估计 | 6.15 秒，772 次 cost。 |
| 前 8 帧 CUDA 冷运行 | 35.94 秒，759 次 cost；包含首次 TorchInductor 编译，CUDA 分配显存峰值 0.305 GB。 |
| 同进程再次估计前 8 帧 | CPU 5.41 秒；CUDA 11.13 秒，分配显存峰值 0.0545 GB。各 device 两次矩阵最大差 0。 |
| 固定原矩阵文本的最终样条采样与 int32 转换 | 脑内 RMSE 0.2816，时间 r 均值 0.9999987、中位数 1；93.54% 的脑内体素时间点整数值相同。CPU 8 帧采样 1.49 秒。 |
| CPU，完整 490 帧估计 | 4 线程，387.60 秒，45,794 次 cost；包括路径输入的解压、准备和估计，不包括最终重采样和写盘。 |
| CPU，完整 490 帧脑内 pull RMS | 逐帧均值 0.00906 mm、中位数 0.00864 mm、p95 0.01699 mm、最大 0.02603 mm；最坏为第 413 帧，其体素距离 p95 为 0.04123 mm。 |

完整 490 帧与原 `.par` 的逐列对照：

| 参数 | Pearson r | RMSE |
|---|---:|---:|
| x 转角 | 0.999642 | 0.0000787 rad |
| y 转角 | 0.996269 | 0.0001071 rad |
| z 转角 | 0.991151 | 0.0001182 rad |
| x 平移 | 0.999087 | 0.003109 mm |
| y 平移 | 0.999773 | 0.003576 mm |
| z 平移 | 0.999727 | 0.003122 mm |

这些是共享 H100 服务器上的同输入真实控制结果。完整 CPU 对照覆盖全部 490 帧，包括最后一帧，没有截取片段造成的初值差异；详细匿名统计和哈希见[完整 CPU 估计](../../validation/mcflirt/full490_cpu.public.json)。原 MCFLIRT 独立命令为 326.10 秒，包含最终采样和写盘；本次 CPU 只估计矩阵，计时边界不同，不能据此声称加速。CPU 在前 8 帧暖运行控制中更快，完整 GPU 运行仍需单独测量。另见 [CPU/GPU 同进程重复控制](../../validation/mcflirt/first8_warm.public.json)和[固定原矩阵采样](../../validation/mcflirt/first8_sampling.public.json)。

实际安装的 MCFLIRT 为 2111.0，依赖 NEWIMAGE 2601.0 和 MISCMATHS 2412.6。对应的 `costfns.cc`、`optimise.cc` 与项目审计过的版本逐字一致；Euler 旋转、矩阵组合及分解函数去除空白和注释后也一致。NCC 在最优点附近很平坦，float32 累加/融合差异会改变 Brent 在相近 cost 中的选择，因此矩阵尚未逐元素相同。原矩阵文本精度也会影响整数截断；采样控制未取得原运行的内存矩阵，不能将所有余差归因于插值实现。完整 fMRI 处理的时序一致性、完整耗时和标准空间脑图见[全流程对照](../../validation/fmri/matched_native.md)。

参考：[FSL MCFLIRT 文档](https://fsl.fmrib.ox.ac.uk/fsl/docs/registration/mcflirt.html)；Jenkinson M, Bannister P, Brady M, Smith S. Improved optimization for the robust and accurate linear registration and motion correction of brain images. *NeuroImage* 17:825–841, 2002. [DOI](https://doi.org/10.1006/nimg.2002.1132)。
