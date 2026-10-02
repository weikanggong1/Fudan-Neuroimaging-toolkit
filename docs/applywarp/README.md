# TorchApplyWarp：FSL 变形场重采样

## 1. 功能简介

`TorchApplyWarp` 读取 FSL dense warp 或 FNIRT cubic coefficient，将 3D 图像或完整 4D 序列重采样到参考网格。计算由 PyTorch 在 CPU/CUDA 上完成，NIfTI 读写使用 nibabel，运行时不调用 FSL。

同一 warp 可应用于全部时间帧，也可先创建采样计划，再传播同网格上的 FA、MD、组织概率图或标签图。计划复用变换坐标、插值网格、有效掩膜和最近邻索引；4D 采样按 channel 分块，复用 GPU 输入 buffer，并将结果直接回传到最终 CPU 数组。分块改变数据流，不减少体素或时间帧。

CUDA 默认开启 TF32 全局开关；FSL 几何和 cubic coefficient 展开显式使用 float64，影像插值保留原 float32。输出 dtype 由输入类型和 `output_dtype` 决定；float64 输出仍使用 float32 影像插值。没有使用 float16/bfloat16，也没有新增依赖。

## 2. Python 调用、输入和输出

### 2.1 完整 4D 序列

```python
from fnit.applywarp import TorchApplyWarp

corrected_dwi_path = "corrected_dwi.nii.gz"  # 输入：(X, Y, Z, T)，所有帧使用同一空间变换
reference_template_path = "FMRIB58_FA_1mm.nii.gz"  # 输入：定义输出空间网格的参考图
fnirt_coefficients_path = "dwi_to_template_coefficients.nii.gz"  # 输入：已估计的 intent-2007 系数
warped_dwi_path = "corrected_dwi_in_template.nii.gz"  # 输出：(Xref, Yref, Zref, T)

warper = TorchApplyWarp(
    device="cuda:0",  # 第一张可见 GPU；也可指定 "cpu"
    frame_chunk_size=None,  # CUDA 自动选择分帧大小；CPU 保持一次全通道采样
)
warped_dwi_result = warper(
    input=corrected_dwi_path,
    reference=reference_template_path,
    warp=fnirt_coefficients_path,
    premat=None,  # 可选：input → warp source 的 FLIRT scaled-mm 矩阵
    postmat=None,  # 可选：warp reference → output reference 的 FLIRT scaled-mm 矩阵
    interpolation="trilinear",  # 连续强度图；离散标签选择 "nearest"
    warp_convention="auto",  # 对无专用 intent 的 dense field 自动判断约定
    output_dtype="float",  # 输出 float32
)
warped_dwi_result.save(output=warped_dwi_path)
```

需要在同一次调用中保存时，使用相同 `warper.run(input=..., reference=..., output=..., warp=...)`。`save()` 的路径参数名是 `output`。

### 2.2 同网格多图复用

```python
import nibabel as nib
from fnit.applywarp import TorchApplyWarp

native_fa_image = nib.load("native_FA.nii.gz")  # 输入：采样计划的源空间几何
reference_template_image = nib.load("FMRIB58_FA_1mm.nii.gz")  # 输入：输出网格和 header
fnirt_coefficients_path = "dwi_to_template_coefficients.nii.gz"

warper = TorchApplyWarp(device="cuda:0", frame_chunk_size=32)
sampling_plan = warper.prepare(
    input=native_fa_image,
    reference=reference_template_image,
    warp=fnirt_coefficients_path,
    interpolation="trilinear",
    warp_convention="auto",
)

warped_md_result = sampling_plan.apply(
    input="native_MD.nii.gz",  # 与 native_FA 的空间 shape、affine 和 pixdim 完全相同
    output_dtype="float",  # 每张图独立决定输出类型
)
warped_md_result.save(output="MD_in_template.nii.gz")

warped_series_result = sampling_plan.apply(
    input="corrected_dwi.nii.gz",  # 时间帧数可以不同，源空间网格须相同
    reference=reference_template_image,  # 可省略；提供时核验完整 header 和 extensions
    output_dtype="float",
    frame_chunk_size=16,  # 只覆盖本次采样的分帧大小
)
warped_series_result.save(output="dwi_in_template.nii.gz")
```

`prepare()` 捕获 warp、矩阵、插值方式及源/参考几何。源空间发生变化时，重新创建计划；修改 warp 后也重新创建计划。计划不能在创建后切换插值方式。`plan.apply(frame_chunk_size=None)` 沿用创建计划时的政策。

函数入口也支持分帧参数：

```python
from fnit.applywarp import applywarp

warped_image_result = applywarp(
    input="corrected_dwi.nii.gz",  # 输入：完整 4D 序列
    reference="FMRIB58_FA_1mm.nii.gz",  # 输入：输出参考图
    warp="dwi_to_template_coefficients.nii.gz",  # 输入：已有 FNIRT warp
    device="cuda:0",  # 计算设备
    frame_chunk_size=32,  # 每块最多 32 帧
    interpolation="trilinear",  # 输入图像插值方式
    output_dtype="float",  # 输出 float32
)
warped_image_result.save(output="dwi_in_template.nii.gz")
```

### 2.3 参数

| 参数 | 输入格式、默认值和含义 |
|---|---|
| `device` | 构造函数/函数入口参数；`None` 自动选择可用 CUDA，否则 CPU；也可显式 `cpu`、`cuda`、`cuda:N`。CLI 默认 CPU。 |
| `frame_chunk_size` | 构造函数/函数入口参数；`None` 使用自动政策，正整数指定每块帧数。`plan.apply()` 的正整数覆盖仅限当前调用。 |
| `input` | NIfTI 路径或 NIfTI image；3D `(X,Y,Z)` 或 4D `(X,Y,Z,T)`，数据须有限。 |
| `reference` | NIfTI 路径或 image；前三维、affine、空间 pixdim 和 header 定义输出。`plan.apply()` 可省略，沿用创建计划的参考图。 |
| `warp` | 可选 NIfTI 路径或 image；dense field 或 intent-2007 cubic coefficients。省略时仅应用矩阵。 |
| `premat` | 可选有限可逆 4×4 数组或文本文件；input → warp source，采用 FLIRT scaled-mm 坐标。 |
| `postmat` | 可选有限可逆 4×4 数组或文本文件；warp reference → output reference，同样采用 FLIRT scaled-mm 坐标。 |
| `interpolation` | 默认 `trilinear`；也支持 `nearest`、`nn`、`nearest-neighbour`、`nearest_neighbor`。warp field 本身始终三线性采样。 |
| `warp_convention` | 默认 `auto`；可选 `relative`/`rel`、`absolute`/`abs`。只控制无专用 intent 的 dense field。 |
| `output_dtype` | 默认 `None`，遵循下方类型规则；可指定 `char`/uint8、`short`/int16、`int`/int32、`float`/float32、`double`/float64。不属于 `prepare()` 参数。 |
| `output` | `run()` 或 `result.save()` 的输出路径，支持 `.nii`/`.nii.gz`；保存会创建父目录并原子替换文件。 |

CUDA 自动分帧将帧工作 buffer 的估算预算限制为 8 GiB，不另设固定帧数上限；每帧按 `4 × (源体素数 + 2 × 目标体素数)` bytes 保守计入输入及输出。全部帧满足预算时，一次全通道采样；较长序列自动分块。该预算还扣除当前 PyTorch allocated 的张量，并相对十进制 20 GB 留 512 MiB 余量；已有计划坐标也包含在 allocated 中。显式分帧大小仍检查 20 GB 预算，单帧无法满足时报告错误。该政策不依据其他进程的全卡空闲显存选块。CPU 默认全通道，也可显式分帧。

### 2.4 输出结构

`ApplyWarpResult` 包含：

- `image`：3D 输入输出 `(Xref,Yref,Zref)`；4D 输入输出 `(Xref,Yref,Zref,T)`，包括保留单帧 4D 的末维。空间 affine、qform/sform code 和 voxel size 来自 reference，4D 的 `pixdim[4]`/TR 来自 input；其余 header 字段及时间单位沿用 reference。
- `valid_mask`：`(Xref,Yref,Zref)` 的 NumPy bool 数组。warp 和 input 均在支持域内时为真；所有帧共享该空间掩膜。
- `qc`：设备、实际 TF32 开关、warp representation/intent/convention、插值方式、有效比例和输出 dtype。

无效体素仍按原规则乘以 0，因此保留原结果可能出现的负零符号位。显式整数输出采用向零截断，并裁剪到 dtype 范围。未指定类型时，浮点输入输出 float32/float64；整数输入通常根据输出动态范围小于 100 时改为 float32，否则保留原类型。超过 10 帧的整数输入沿用 FSL 原有范围规则，比较第一帧最大值与第二帧最小值；不会对每块重新选择 dtype。

### 2.5 坐标、文件格式和范围

FSL scaled-mm 根据 header voxel size 缩放存储轴；当 NIfTI affine 的行列式为正时，翻转并平移第一轴。warp 三分量使用这三个 scaled-mm 轴的毫米值，不是 world-RAS。

对 output voxel `x`，`Vref`/`Vin` 为 reference/input 的 voxel → scaled-mm 矩阵：

```text
q = Vref x
r = inverse(postmat) q
dense relative: s = r + sample(warp, r)
dense absolute: s = sample(warp, r)
intent 2007:    s = inverse(embedded_affine) r + sample(cubic_residual, r)
u = inverse(Vin) inverse(premat) s
output(x) = input(u)
```

从输入描述变换顺序为 `premat → warp → postmat`；重采样从输出反查输入。nearest 使用 FSL 正半整数向上取整规则。

| intent | 表示 | 支持情况 |
|---:|---|---|
| 0 或其他 | `(Xwarp,Ywarp,Zwarp,3)` dense field | 支持 relative/absolute；`auto` 使用 FSL 标准差判别。 |
| 2006 | FNIRT dense displacement | 按 FSL 规则始终 relative。 |
| 2007 | cubic B-spline coefficients | 从 qform offset 读取 field size、intent 读取 voxel size、pixdim 读取 knot spacing、sform 读取 embedded affine；完全由 PyTorch 展开。 |
| 2008、2009 | DCT、quadratic coefficients | 明确拒绝。 |

仅支持一个 premat 和一个 postmat，同一变换应用全部帧。输入影像的 sinc/spline 插值、supersampling、padding、`--mask`、`--usesqform` 和逐帧矩阵未实现。

SynthMorph 的 RAS pull displacement 先经 `fnit.synthmorph.convert_warp_to_fsl(warp, moving=..., fixed=...)` 转为 intent-2006，才能交给本模块。两个坐标约定的数组和 header 不直接互换；详见 [SynthMorph](../synthmorph/README.md)。

## 3. 命令行调用

```bash
# 输入为完整已校正 DWI；沿用同一 warp，不减帧。
fnit applywarp \
  --in corrected_dwi.nii.gz \
  --ref FMRIB58_FA_1mm.nii.gz \
  --warp dwi_to_template_coefficients.nii.gz \
  --interp trilinear \
  --datatype float \
  --device cuda:0 \
  --frame-chunk-size 32 \
  --out corrected_dwi_in_template.nii.gz
```

`fnit-applywarp` 使用相同参数。省略 `--frame-chunk-size` 使用自动政策。

| CLI 参数 | Python 对应或作用 |
|---|---|
| `--in`、`--ref`、`--warp` | `input`、`reference`、`warp`。 |
| `--premat`、`--postmat` | 对应矩阵文本文件。 |
| `--interp` | `trilinear`、`nearest`、`nn`。 |
| `--rel` / `--abs` | 显式 relative/absolute，二者互斥；省略为 auto。 |
| `--datatype` | `char`、`short`、`int`、`float`、`double`。 |
| `--device` | 默认 CPU；GPU 显式指定 `cuda:N`。 |
| `--frame-chunk-size` | 正整数帧数上限；省略时自动。 |
| `--out` | 输出路径。 |
| `--overwrite` | 允许覆盖已有输出；默认拒绝覆盖。 |

## 4. 原软件调用

上述操作对应 FSL：

```bash
applywarp \
  --in=corrected_dwi.nii.gz \
  --ref=FMRIB58_FA_1mm.nii.gz \
  --warp=dwi_to_template_coefficients.nii.gz \
  --interp=trilinear \
  --datatype=float \
  --out=corrected_dwi_in_template.nii.gz
```

FSL 没有 FNIT 的 `--device`/`--frame-chunk-size` 参数。输入、warp、premat/postmat、插值和输出类型需相同，才构成固定变换比较。官方用法见 [FNIRT User Guide：applywarp](https://fsl.fmrib.ox.ac.uk/fsl/docs/registration/fnirt/user_guide.html#applywarp)。

## 5. 最新验证、精度和运行时间

2026-10-02 已完成最终自动策略的完整 105 帧 DWI、3D FA，以及 FastVBM、volume、dMRI 三条 FNIRT pipeline 验证。原 FNIT 与新代码在同一设备上的输出逐位一致；全部记录见 [registration lossless 验收](../../validation/registration_lossless_20261002/README.md)。

### 最终默认策略：105 帧完整 DWI

输入为 `104×104×72×105`，输出为 `91×109×91×105`，固定同一官方 FNIRT coefficient。最终 auto 在本例选择全部 105 帧；全部 94,776,045 个输出值的位模式、完整 header、TR、有效掩膜和保存后重读均通过。

| 当前配置 | 冻结旧版 | 最新自动策略 | 计时范围 |
|---|---:|---:|---|
| API 中位数 / s | 0.886 | 0.813 | 已解码完整输入，两次；含坐标和 GPU/CPU 搬运，排除读写与校验。 |
| 路径输入 API / s | 3.700 | 3.838 | 单次，含读取、计算和回传，排除保存；本例未见稳定文件入口加速。 |
| peak allocated / GB | 1.157 | 0.777 | API 运行，十进制 GB。 |

独立 FSL `applywarp` 的模板脑内 23,990,715 个值：Pearson r=0.999999999997，MAE=0.001514，RMSE=0.004125，最大绝对误差=0.446289。shape、affine、dtype、TR 和空间/时间单位相同；未使用的 `pixdim[5:8]` 分别为 FSL 的 0 与 FNIT 的 1。跨软件仍有微小浮点差异，同一 FNIT 设备的无损门禁为逐位一致。

完整 CLI 另行实测，包含进程启动、读取和 gzip 保存：FNIT 单 GPU **18.43 s**，FSL CPU 8 线程 **500.62 s**，均为全 105 帧、同一输入与变换。这是各一次、共享服务器上的观测；不把 API 时间与原软件文件入口直接计算比值。[最终 API 报告](../../validation/registration_lossless_20261002/components_apply_final.public.json)、[完整 CLI 报告](../../validation/registration_lossless_20261002/cli_final.public.json)、[官方对照](../../validation/registration_lossless_20261002/official_warp.public.json)。

![105 帧 DWI 的首帧模板脑内官方输出、FNIT 输出和差值](../../validation/registration_lossless_20261002/figures/official_warp_001.png)

### 第一阶段配置记录：分帧选择

输入为 `104×104×72×105`，输出为 `91×109×91×105`，固定同一 FNIRT warp。各配置完整 warmup 输出逐位一致，包括正负零、header/TR 和有效掩膜；正式 API 使用已解码输入，排除加载、保存和校验，每配置两次计时。

| 第一阶段配置 | 正式 API 中位数 / s | peak allocated / GB |
|---|---:|---:|
| 冻结旧版全通道 | 0.8256 | 1.157 |
| 当时 auto：最多 64 帧、1 GiB | 0.9985 | 0.501 |
| 显式 128，上限截为全部 105 帧 | 0.7984 | 0.777 |

本例一次处理全部 105 帧比拆成 64 帧更快，并满足显存预算。因此自动政策改为按 8 GiB 工作预算和 resident 张量后的 20 GB 预算选块，取消固定 64 帧上限。表中数字保留为第一阶段显式配置记录；最终默认入口的完整 105 帧结果见上表。3D FA 的最终完整输出也逐位一致，API 约 0.051 s、peak allocated 1.969 GB；该 3D 单次观测未显示计算加速。阶段聚合及全部配置见 [component_tuning.public.json](../../validation/registration_lossless_20261002/component_tuning.public.json)。

本轮无损门禁使用冻结 FNIT `7473452` 与最新代码在**同一设备、同一输入和变换**上的对照：全部体素的位模式，包括正负零；完整 header/extensions、affine、shape、dtype、TR、有效掩膜；保存后重读。单独记录读取、API、坐标准备、采样、保存、CUDA allocated/reserved 与分帧参数。正式计时不使用 profiler；CUDA profiler 仅用于另外的诊断运行。复现工具为 [benchmark_warp_4d_lossless.py](../../tools/benchmark_warp_4d_lossless.py)，使用服务器私有 JSON 指定真实路径，不裁剪或减少帧。

### 2026-09-27：固定官方 FNIRT warp 的真实 3D FA

一例真实 UKB 格式 dMRI，输入 TBSS 预处理后 native FA、`FMRIB58_FA_1mm` 和 FSL intent-2007 coefficient，参照为 FSL 6.0.7.4 `applywarp`。当时 core SHA-256 为 `cf6de438f3ac1551804d38682ce3fbb11b8fb042ad881562ebc93aada80f2df5`。

| 指标 | 结果 |
|---|---:|
| 输出 shape / affine / dtype | 一致 |
| 非零并集体素 | 1,548,144 |
| Pearson r | 0.999999999994 |
| MAE / RMSE | 3.55e-7 / 5.87e-7 |
| 最大绝对误差 | 1.18e-5 |
| 有效比例 | 0.565360 |
| H100 peak allocated | 1.958 GB |
| FNIT 三次 wall，中位数 | 0.337 / 0.259 / 0.364 s，0.337 s |
| FSL 三次 wall，中位数 | 6.19 / 5.11 / 5.67 s，5.67 s |

FNIT 包含读取、展开、重采样和 CPU 回传，排除保存；FSL 包含读取、计算和保存。计时边界不同，不计算加速比。这是单例、连续 FA、三线性和固定 coefficient 的历史官方对照。

![真实 FA 在模板空间的官方与 FNIT 输出及差异](figures/applywarp_real_fa_comparison.png)

原始聚合记录见 [report.real.current.json](../../validation/applywarp/report.real.current.json)，复现入口见 [validate_real.py](../../validation/applywarp/validate_real.py)。另一对公开 T1w 的 intent-2006 固定 warp 对照见 [SynthMorph 转换验证](../synthmorph/README.md#fsl-warp-转换真实-t1w)。

## 6. 最近更新与 benchmark 记录

| 日期 | 更新 | 验证记录 |
|---|---|---|
| 2026-10-02，本轮 | 4D channel 分块、GPU 输入 buffer 复用、直接回传最终数组、nearest 索引缓存、原地乘法 mask；新增构造函数/函数入口/CLI 分帧参数。按第一阶段完整 DWI 的吞吐结果，自动政策改为 8 GiB 工作预算、取消固定 64 帧上限。 | [最终默认与三条 pipeline 验收](../../validation/registration_lossless_20261002/README.md)通过；保留[第一阶段配置实测](../../validation/registration_lossless_20261002/component_tuning.public.json)。 |
| 2026-10-02，前轮 | `ApplyWarpPlan` 复用 float64 几何和 float32 grid；跨图严格核验源网格及参考 header/extensions。 | 同一真实病例九图传播：原逐图调用→计划，热调用中位数 0.365734295→0.180841511 s；全部 voxel/header/affine/mask 及 TBSS 后处理一致。[九图报告](../../validation/dmri_pipeline/map_propagation_20261002.md)。 |
| 2026-09-27 | 固定官方 coefficient 的 3D FA 采样。 | 上节官方精度和时间。[聚合 JSON](../../validation/applywarp/report.real.current.json)。 |

九图计时包含计划创建、九图采样和 CPU 回传，排除输入读取、FA 预处理、保存及配准估计；它衡量固定 warp 传播，不代表 FNIRT 或整个 pipeline 的耗时。定向测试在 [tests/applywarp](../../tests/applywarp)，覆盖 3D/4D、dense/coefficient、nearest/linear、整数类型、完整 header、缓存失效和正负零。

## 7. 参考文献和原实现

- Andersson, Jenkinson & Smith. *Non-linear registration, aka spatial normalisation*. FMRIB Technical Report TR07JA2 (2007)，[原文](https://www.fmrib.ox.ac.uk/datasets/techrep/tr07ja2/tr07ja2.pdf)。
- [FSL FNIRT User Guide：applywarp](https://fsl.fmrib.ox.ac.uk/fsl/docs/registration/fnirt/user_guide.html#applywarp)。
- [官方 `applywarp.cc` 源码](https://git.fmrib.ox.ac.uk/fsl/fugue/-/blob/master/applywarp.cc)，位于 FSL `fugue` 仓库；[FNIRT 系数/场工具源码](https://git.fmrib.ox.ac.uk/fsl/fnirt)。
- [FNIT 当前实现](../../src/fnit/applywarp/core.py)。
