# MCFLIRT 运动校正

## 1. 功能与流程

`TorchMCFLIRT` 将四维 BOLD 的每一帧配准到 SBRef，或默认的中间帧。估计和重采样均在 FNIT 内完成，运行时不启动 FSL。六自由度变换使用 FSL scaled-mm 坐标，每个矩阵的方向是 input frame → reference。

实现按 [MCFLIRT 2111.0 源码](https://git.fmrib.ox.ac.uk/fsl/mcflirt/-/blob/2111.0/mcflirt.cc)的默认三阶段路径：参考图直接重采样到 8、4、4 mm；边缘使用 1 mm 降权；每阶段一次 Brent 坐标轮回，容差乘数依次为 0.8、0.8、0.1。8 mm 阶段沿时间传递上一帧的估计，后两阶段从各帧已有估计继续。使用外部参考图时，从第 0 帧开始。优化不使用脑掩膜。

FSL `p_normcorr_smoothed` 的 `num` 和 `numA` 在行、层之间连续累加。FNIT 保留此计数定义及其方差公式；替换成常见的加权 Pearson 会改变优化目标。参考金字塔没有额外 Gaussian 预滤波，最终样条采样也不增加 FLIRT `applyxfm` 的降采样滤波。

```mermaid
flowchart LR
    A[完整四维 BOLD 与三维参考] --> B[8 / 4 / 4 mm 参考]
    B --> C[原 NCC 与 Brent 六自由度优化]
    C --> D[逐帧 input 到 reference 矩阵与六列参数]
    D --> E[可选原 linear / spline 最终采样]
    E --> F[校正 BOLD 与 MAT / par / RMS 文件]
```

### 从 TorchFLIRT 复用的实现

`TorchMCFLIRT` 复用项目中成熟的 `TorchFLIRT` 变换组合、Brent 坐标优化、三线性插值和精确 float32 CUDA 运算。MCFLIRT 的目标函数与 FLIRT 常用的相关比、互信息不同；本模块继续使用上述 NCC 定义，并保留每行有效 x 范围及逐次 float32 加法生成采样坐标的顺序。

CUDA 路径将行范围计算、坐标生成、三线性采样、1 mm 边界降权和参考值读取合并到一个 Triton kernel，复用三个 float32 缓冲区供原 NCC 归约使用。这样减少每次 cost 的 GPU kernel 启动、临时张量和小张量的主机同步。变换仍先在 CPU 以 float64 组合、求逆，再转换为原实现使用的 12 个 float32 系数；融合采样使用逐步舍入的 float32 运算。

同一次调用还复用各帧的强度重心。若完整 float32 BOLD 的大小不超过 4 GiB 且不超过当前空闲显存的四分之一，CUDA 路径会在三个阶段间缓存已上传的帧；更大的输入逐帧上传。缓存保存完整 float32 数据。三阶段的执行顺序、第一阶段的相邻帧初值传递、Brent 求解和 NCC 归约沿用既有实现。

本轮继续减少重复准备和内核提交：

- **CPU 变换缓存。** 每个固定重心的搜索保存最近一次各轴角度及旋转矩阵；只改变平移时直接复制三轴旋转的组合再加平移，只改变一个角度时复用另外两轴。角度按 float64 原始字节比较，正负零不会共用缓存。仍用共享 FLIRT 的逐轴旋转和 `I @ Rx @ Ry @ Rz` 运算顺序，不改三角函数和矩阵乘法。pull 系数复用固定 voxel size 的两份对角矩阵，每次仍按原顺序求逆、乘积，再缩窄到 float32。
- **NCC 的 CUDA graph。** 每次 `run` 为 8 mm、4 mm 参考分别保存采样缓冲区和 graph；两个 4 mm 阶段复用同一 workspace。graph 捕获并重放既有的编译 NCC 归约，继续使用原行、层累加顺序和连续计数。帧切换时复制完整 float32 moving 到固定地址。变换系数、融合采样和返回标量所需的同步仍按实际 cost 执行。
- **最终 motion-only 样条采样 graph。** CUDA 复用逐帧最终样条采样的固定缓冲区和 graph，保持原 Constant 样条、extraslice 边界和输出类型转换；CUDA 三线性与组合 nonlinear warp 仍由既有接口处理。CPU v4 的线性/样条路径使用有序 Numba 内核，逐步 float32 坐标、double 样条累加和每轴前滤波舍入保持不变；完整 180/490 帧 helper 已逐值核对，最终完整 API 验收范围见 [CPU 专页](CPU_BENCHMARK_20261004.md)。
- **按输出要求计算 RMS。** 仅请求 `rmsrel` 或 `rmsabs` 时计算对应 RMS 和均值，保留原 80 mm 球体定义及文本格式。

这些改动复用相同计算及固定数据，没有减少 cost 求值、放宽 Brent 容差、改变三阶段顺序或使用 float16。完整时序的精度及计时以本轮配对验证为准。

Python、命令行参数及输出结构均无需调整。融合采样需要 Triton；项目 [Conda 环境](../../environment.yml)已包含与 PyTorch 2.5.1 对应的 `triton==3.1.0`。CPU 成本采用专属有序行采样 helper，保持逐次 float32 坐标、插值和行/平面归约；计数的累计求和沿用 PyTorch CPU 的双精度累加及 float32 输出。无法导入 Triton 时的 CUDA cost 采样使用既有 PyTorch 张量实现。CPU 完整精度和当前性能状态见 [2026-10-04 CPU 专页](CPU_BENCHMARK_20261004.md)。2026-10-02 正常缓存配置的 H100 双卡 focused 检查 **74 项通过、0 跳过**，覆盖精确缓存、融合采样、原归约 graph 重放、最终采样及非默认 device/stream；实际源码、Git 及运行前后哈希核对见[测试报告](../../validation/mcflirt/review_tests_exact_latest.public.json)。该历史检查未覆盖本轮发现的无缓存配置。

### 2026-10-03 pipeline 接入暴露的成熟子函数 bug

公开 CON03 的完整 volume 接入在禁用 CUDA allocator 缓存时，于 NCC graph capture 报 `operation not permitted when stream is capturing`。项目固定 PyTorch 2.5.1 的无缓存模式直接执行 `cudaMalloc`，原 cost 与最终 spline 两处 graph 都会在捕获区间申请临时张量。

现在两处都在捕获前按 allocator 配置选路：正常缓存继续重放原 graph；无缓存配置复用相同采样缓冲区，直接执行原编译 NCC 归约与原样条函数。变换、cost 次数、三阶段顺序和计算精度保持原定义。

配置应在启动 Python **之前**完成。`PYTORCH_NO_CUDA_MEMORY_CACHING` 按变量是否存在判定，`1`、`0` 和空字符串都表示禁用缓存；要启用缓存须移除变量。该语义来自 [PyTorch 2.5.1 allocator 原实现](https://github.com/pytorch/pytorch/blob/v2.5.1/c10/cuda/CUDACachingAllocator.cpp#L2913)。不在已初始化的 CUDA 进程中切换它。

## 2. Python 调用、输入与输出

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

### 单被试 Python 调用

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

## 3. 命令行调用

```bash
# 将 raw BOLD 配准到 SBRef，最终用样条采样；保存矩阵、参数及两类 RMS。
fnit mcflirt -in sub-01_task-rest_bold.nii.gz \
  -reffile sub-01_task-rest_sbref.nii.gz \
  -out motion/prefiltered_func_data_mcf \
  -mats -plots -rmsrel -rmsabs -spline_final --device cuda:1
```

正常缓存可用 `env -u PYTORCH_NO_CUDA_MEMORY_CACHING fnit mcflirt ...` 启动；无缓存模式可在启动前设置 `PYTORCH_NO_CUDA_MEMORY_CACHING=1`。该配置只改变 graph 的执行方式，CLI 参数含义同 Python 表。

## 4. 对应原软件调用

对应的原 FSL benchmark 命令：

```bash
mcflirt -in sub-01_task-rest_bold.nii.gz \
  -reffile sub-01_task-rest_sbref.nii.gz \
  -out motion/prefiltered_func_data_mcf \
  -mats -plots -rmsrel -rmsabs -spline_final
```

省略 `-reffile` 使用中间帧；省略 `-spline_final` 使用三线性最终采样。`-mats`、`-plots` 和 RMS 开关只控制文件输出，不改变估计流程。

## 5. 真实数据精度、耗时与脑图

### 2026-10-04 完整 CPU 官方 benchmark

[CPU 专页](CPU_BENCHMARK_20261004.md)记录完整 180/490 帧、默认三阶段、最终样条及正常文件的原 MCFLIRT、冻结主仓库和 CPU 候选 v2。1/8 线程的完整矩阵、参数、影像及 header 与冻结版本逐位相同；速度尚未达标，8 线程 v2 正式计时出现回退。页面列出全体素官方误差、内部 profile、已执行及待执行功能，完整 GPU ABBA 由协调任务验收。

### 2026-10-03 公开 CON03 的无缓存修复回归

同一公开 [OpenNeuro ds001226 v5.0.1](https://openneuro.org/datasets/ds001226/versions/5.0.1) 健康控制 CON03 的完整 `64×64×42×180` BOLD，参考为成熟 FEAT `_save` 保存的原第 90 帧。物理 GPU 0 上按正常缓存、无缓存顺序运行两个独立 Python 进程，均为 PyTorch 2.5.1、8 CPU 线程、TF32、原 float32 运算、8/4/4 mm 三阶段及完整 spline 校正。输入、参考、实际源码及验证函数的前后 SHA 一致；完整记录见[匿名真实回归报告](../../validation/mcflirt/nocache_public_con03_20261003.public.json)。

| 完整 180 帧 `TorchMCFLIRT.run` | 正常 allocator 缓存 | 禁用 allocator 缓存 |
| --- | ---: | ---: |
| API 墙钟 | 52.151 s | 290.753 s |
| NCC cost 次数 | 17,465 | 17,465 |
| `180×4×4` 矩阵、`180×6` 参数 | 两组 float64 数组逐 bit 相同，最大差 0 | 同左 |
| 校正 BOLD | 两组完整 int16 数组逐 bit 相同，最大差 0 | 同左 |
| 本任务进程树显存采样峰值，十进制 GB | 1.327497 | 1.126171 |
| 同一峰值，GiB | 1.236328 | 1.048828 |

API 时间包括 NIfTI 解码、准备、该进程的正常延迟编译/graph 初始化、运动估计、最终采样和输出类型转换；包导入、参考准备、数组保存及保存后核验另计。两个进程共享私有 Inductor/Triton 磁盘缓存，缓存未清空；CUDA graph 由各自进程创建。测量发生在共享 H100 上，显存每 2 秒采样，完整 GPU 负载日志保留。此表是单次执行方式观察，计时不进入正式十例整链，也不据此给出通用性能比例。

本轮回归比较 FNIT 的两种 allocator 执行配置；此前完整 490 帧的原 FSL 精度和耗时对照保留如下。无缓存修复保持相同计算与输出，新增的小体积 policy/stream 控制属于回归测试。

### 前次正常缓存的完整 490 帧核对

本轮候选源码 `8a3f2276` 与优化前主线 `de239711` 在同一真实 BOLD 的全部 490 帧上按基线/候选/候选/基线顺序测量。使用共享 H100 PCIe 的物理 GPU 1、8 线程、TF32 和完整 float32，参考为冻结 FEAT 保存的 `example_func.nii.gz`。三个阶段均为一次坐标轮回，最终样条采样。独立进程的 API 计时包含四维读取/解压、准备、首次编译/graph 捕获、估计、最终采样和 dtype 转换，不含写盘。输入及源码哈希、GPU 快照、未舍入数组哈希见[本轮配对报告](../../validation/mcflirt/paired_exact_latest.public.json)。

| 相同计时边界，秒 | 基线第 1/2 回合 | 候选第 1/2 回合 | 两回合均值，基线 → 候选 |
|---|---:|---:|---:|
| 完整 `TorchMCFLIRT.run` | 86.37 / 109.46 | 52.47 / 63.48 | 97.91 → 57.98 |
| 其中最终样条采样 | 27.89 / 26.54 | 16.51 / 20.05 | 27.22 → 18.28 |
| API 扣除最终采样 | 58.47 / 82.92 | 35.96 / 43.43 | 70.70 → 39.69 |

本次 API 均值减少 **40.8%**。最后一行还包含输入读取、准备、编译/capture 和 dtype 转换，不能视为纯优化器时间。GPU 1 运行期间其他任务显存增加，各次共享负载不同；表中保留全部原始计时，该比例只描述本次配对观察。

| 完整 490 帧精度及资源核对 | 结果 |
|---|---|
| NCC 求值次数 | 四次均为 45,972，Brent 及阶段顺序未改变。 |
| 未舍入 float64 矩阵及六列参数 | 全部位模式一致，包括正负零；对应数组 SHA-256 相同。 |
| `MAT_*`、`.par` 文本 | 另一次完整验证中全部数值、文件 SHA-256 与冻结 FNIT 相同。 |
| 校正 int32 图 | 全部 242,851,840 个解码值一致，RMSE 和最大绝对差均为 0。 |
| 未截断 float32 样条输出 | 完整 490 帧共 242,851,840 个 uint32 位模式与原 eager 采样相同，RMSE 和最大差为 0；见[独立采样证明](../../validation/mcflirt/sampling_exact_latest.public.json)。 |
| 候选配对调用 CUDA 峰值 | allocated 1.154 GB，reserved 1.286 GB。 |

[额外完整验证](../../validation/mcflirt/gpu_exact_latest.public.json)记录 API 59.09 秒，其中样条采样 17.38 秒；加私有验证文件写出为 87.85 秒，固定掩膜后续 FEAT 子函数为 21.12 秒，合计 108.98 秒。这份验证额外写出完整精度数组，输出集合与原 FSL 命令不同。其默认读取头信息 TR=0.7350000143 秒，而冻结 FEAT 显式使用 0.735 秒；高通的 `sigma_volumes = 100 / (2 × TR)` 因而有微小差别，whole-image RMSE 为 1.48×10⁻⁵、最大差 0.00390625。将 TR 同样指定为 0.735 秒后，全部 242,851,840 个高通 float32 值恢复逐元素一致，RMSE 0，见[同 TR 控制](../../validation/mcflirt/matched_tr_exact_latest.public.json)。该差异来自验证参数，MCFLIRT 的矩阵和采样在两次控制中均保持一致。

固定矩阵的[组件 profile](../../validation/mcflirt/README.md#组件profile)进一步定位了重复构建和提交开销；它不能用各组件耗时相加推导 API 时间，端到端效果以上述配对结果为准。

历史融合版本 `cfb7beee` 在物理 GPU 0 上的[报告](../../validation/mcflirt/gpu_optimization_latest.public.json)记录运动 API 278.45 秒、最终采样 32.01 秒；`7456251` 在物理 GPU 1 上的[早期观察](../../validation/mcflirt/gpu_optimization.public.json)为 API 159.27 秒、采样 24.83 秒。这两次共享负载及缓存状态未匹配，分别保留为历史计时，不能作为本轮加速比的分母。

本轮重跑原 FSL `mcflirt` 2111.0，使用相同 raw BOLD 与保存参考，标准输出为校正 NIfTI、`MAT_*` 和 `.par`；加 `-report -verbose 1` 只记录 benchmark 进度。原命令总 wall 为 **331.54 秒**，包含估计、最终采样和标准文件写盘。原安装的外层 ELF 包装器返回 255，但完整图的 gzip CRC、网格、有限值及全部 490 个 MAT/参数均通过核验。独立真实首帧诊断确认 C++ 子进程正常退出 0，包装器第二次等待同一子进程得到 `ECHILD` 后退出 -1。全 490 帧的原退出码原样保留，报告分别标记 `completed_outputs=true` 和 `command_exit_ok=false`，见[原生命令报告](../../validation/mcflirt/native_exact_latest.public.json)。

以下精度统计及更新后的脑图均使用本轮原 FSL 输出，见[最新原生对照](../../validation/mcflirt/native_comparison_latest.public.json)。

| 与原 FSL MCFLIRT 比较，完整 490 帧 | 结果 |
|---|---|
| 脑内 pull RMS | 逐帧均值 0.00902 mm、中位数 0.00833 mm、p95 0.01648 mm、最大 0.04011 mm；最坏为第 296 帧，其体素距离 p95 为 0.06691 mm。 |
| 校正图 | 脑内时间 r 均值 0.999557、中位数 0.999825；4D RMSE 8.39578，17.76% 的脑内体素时间点整数值相同。原和 FNIT 输出均为 int32。 |
| temporal SD 图 | EPI 空间 SD 图 r 为 0.99999706，RMSE 为 0.42156；原/FNIT 脑内平均 SD 为 239.3432/239.3228。 |

六列 `.par` 与本轮原程序的对照：

| 参数 | Pearson r | RMSE |
|---|---:|---:|
| x 转角 | 0.999556 | 0.0000876 rad |
| y 转角 | 0.996263 | 0.0001071 rad |
| z 转角 | 0.990922 | 0.0001197 rad |
| x 平移 | 0.999256 | 0.002827 mm |
| y 平移 | 0.999773 | 0.003558 mm |
| z 平移 | 0.999754 | 0.002968 mm |

原报告消息可记录读取、8 mm 阶段、第二个 4 mm 阶段，以及第三阶段加最终采样/MAT/par 保存的合并区间。原程序没有分别标记第三阶段结束和采样开始，因此不推断纯估计或采样耗时；分步观察表见[验证说明](../../validation/mcflirt/README.md#原软件本轮运行与分步观察)。原命令 331.54 秒包含标准写盘，配对 API 57.98 秒不含写盘，二者不能直接作为同边界加速比。

冻结流程的[历史 profile](../../validation/fmri/feat_profile.public.json)曾记录运动 API 709.11 秒、其中采样 30.74 秒；它带有额外包装调用，与当前计时没有匹配共享 GPU 负载和缓存状态，作为瓶颈定位记录保留。原 MCFLIRT 历史独立命令为 326.10 秒，包含最终采样和写盘。以上计时展示各次实际运行的耗时和范围，未建立受控的固定加速比。历史 [CPU 490 帧估计](../../validation/mcflirt/full490_cpu.public.json)为 4 线程、387.60 秒、45,794 次 cost，仅包含解压、准备和估计；这不是本次优化后的 CPU 重测。旧的前 8 帧控制和采样诊断集中在[独立验证说明](../../validation/mcflirt/README.md)，其耗时不作为当前完整体积性能。

复测请使用冻结流程实际保存的参考图，并核对报告中的 SHA-256 和 NIfTI `pixdim`。本例原始 SBRef 与 FEAT 保存参考图的像素、affine 相同，z 向 voxel size 分别约为 2.3999999 与 2.3999996 mm；微小头信息差异仍会改变 float32 采样与平坦 NCC 最小值的选择。

原软件核对所用 MCFLIRT 为 2111.0、NEWIMAGE 2601.0、MISCMATHS 2412.6；对应 `costfns.cc`、`optimise.cc` 与项目审计版本一致。相对原 FSL 的矩阵和校正图仍有上表所列余差；本轮位模式核对的对象为优化前 FNIT。上轮完整 fMRI 流程的处理边界、阶段耗时和原软件对照见[全流程验证](../../validation/fmri/mcflirt_optimization.md)，该整链耗时尚未包括本轮 graph 优化。

### 前次正常缓存的原软件脑图

![原 FSL MCFLIRT 和 FNIT 的均值、temporal SD 与时间相关图](figures/motion_correction_mni.png)

同一例 490 帧 BOLD。本轮原生对照的均值、temporal SD（`ddof=0`）及逐体素时间 r 先在 EPI 空间计算，再用同一原 BBR 和 FNIRT pull 将三维指标图变换到 MNI，仅用于展示。均值和 SD 使用各自统一的色阶；时间 r 的色阶为 0.995–1，低于下限的体素显示为下限颜色。这里的 SD 不是将四维 BOLD 重采样到 MNI 后重新计算的 SD。没有额外空间平滑，只有获授权的去标识化模板空间 PNG 被公开。复现命令见[独立验证说明](../../validation/mcflirt/README.md)，图示实现为 [render_comparison.py](../../validation/mcflirt/render_comparison.py)。

## 6. 最近版本更新与 benchmark 记录

| 版本与日期 | 功能变化 | 真实数据 benchmark 与核对 |
|---|---|---|
| 2026-10-03 无缓存兼容修复 | 成熟 cost 与最终 spline 在 capture 前按 flag presence 选路；正常配置保留 graph，无缓存直接执行相同归约和样条函数。`0`、空字符串和 `1` 均禁用 allocator 缓存；配置在启动进程前设置。 | 公开 CON03 完整 180 帧、两个独立进程：17,465 次 cost 不变，矩阵、参数及校正 int16 图逐 bit 相同；API **52.151 / 290.753 s**，范围见[本轮报告](../../validation/mcflirt/nocache_public_con03_20261003.public.json)。原 `19c8e0a3` 无缓存整链失败记录保留，新正式整链从原数据另起。 |
| `8a3f2276`，2026-10-02 | 精确角度/旋转缓存和 pull 对角矩阵复用；每次调用按参考尺寸复用原 NCC graph；motion-only 样条采样 graph；按请求计算 RMS。 | 完整 490 帧两回合同卡配对，API 均值 **97.91 → 57.98 秒**；45,972 次 cost 不变，未舍入矩阵/参数和校正图逐 bit 相同，未截断样条 float32 全相同。同 TR 的高通结果全相同。原 FSL 本轮脑内时间 r 均值 0.999557；见[最新配对报告](../../validation/mcflirt/paired_exact_latest.public.json)。 |
| `cfb7beee`，2026-10-01 | 最终合并主线后复测；MCFLIRT 运行时源码哈希与初次融合版本一致。 | 历史完整 490 帧、固定 FEAT 参考，物理 GPU 0 上 API **278.45 秒、45,972 次 cost**，其中样条采样 32.01 秒；矩阵/参数文本和校正/高通图与冻结 FNIT 逐值相同。详见[历史融合报告](../../validation/mcflirt/gpu_optimization_latest.public.json)。 |
| 冻结状态 `1eb9c417`，2026-10-01 | 数值匹配修复后的基线：保留源 NCC 计数、逐次 float32 坐标累加及输出数据类型，保存完整 490 帧 GPU 对照。 | 相对原 FSL，脑内 pull RMS 均值 0.00881 mm，校正图脑内时间 r 均值 0.99957825。历史 **973.12 秒是 FEAT 阶段合计**，包含运动校正、掩膜准备、缩放和高通；MCFLIRT 估计与采样未单独计时。详见[冻结 GPU 报告](../../validation/mcflirt/full490_gpu.public.json)。 |

固定变换的 CPU/冻结 CUDA/融合 CUDA 代价微基准见[标量报告](../../validation/mcflirt/cost_optimization.public.json)及[复现说明](../../validation/mcflirt/README.md#单次代价函数微基准)。该测量已预热、采用共享 GPU 上的交替调用，范围仅为单次 cost；完整运动 API 的计时边界以上表和各自报告为准。

## 7. 参考文献、许可证与原实现

本模块是基于 FSL 源码改写的 Python/PyTorch 实现，沿用 [FSL Software Licence, Release 6.0](../../licenses/FSL-6.0.txt) 的非商业使用条款。许可证要求在无财务回报的再分发中向接收者保留条款，并随产品提供原始及修改后的源代码。因此，本包同时提供改写后的 `src/fnit/mcflirt/` 和官方 tag `2111.0` 的[完整六文件源码快照](../../src/fnit/_vendor_fsl/sources/mcflirt-2111.0/)；原始文件未修改，FNIT 不编译或执行这些文件。

官方提交为 `fa24cb88fba970fb9adc959713fee57bf3706d3e`，Git tree 为 `ee503629cf36f06a698eb4bfc516e2ad2df8feff`。官方仓库、无前缀 `git archive --format=tar` 的 SHA-256、逐文件大小及 SHA-256 见[源码清单](../../src/fnit/_vendor_fsl/manifest.json)；共享 NEWIMAGE/MISCMATHS 来源及许可见[vendor 说明](../../src/fnit/_vendor_fsl/README.md)和[第三方声明](../../THIRD_PARTY_NOTICES.md)。这些来源记录不将 FNIT 标记为官方 FSL，也不扩大上文已测量的数值范围。

参考：[FSL MCFLIRT 文档](https://fsl.fmrib.ox.ac.uk/fsl/docs/registration/mcflirt.html)；Jenkinson M, Bannister P, Brady M, Smith S. Improved optimization for the robust and accurate linear registration and motion correction of brain images. *NeuroImage* 17:825–841, 2002. [DOI](https://doi.org/10.1006/nimg.2002.1132)。

CUDA allocator 的配置语义与 graph 私有内存池见 [PyTorch 2.5.1 CUDACachingAllocator](https://github.com/pytorch/pytorch/blob/v2.5.1/c10/cuda/CUDACachingAllocator.cpp#L80) 及其 [`forceUncachedAllocator`](https://github.com/pytorch/pytorch/blob/v2.5.1/c10/cuda/CUDACachingAllocator.cpp#L2913)。
