# TorchApplyWarp：变换场与 world 变换重采样

| 项目 | 内容 |
|---|---|
| 输入 | 3D/4D NIfTI、目标网格、可选FSL场和矩阵 |
| 输出 | 目标网格影像与有效覆盖/QC |
| 对应原软件 | FSL applywarp；world入口对应fMRI重采样 |
| Python / CLI | TorchApplyWarp / fnit applywarp、fnit-applywarp |
| CPU / GPU | CPU或CUDA；nibabel读写 |

## 1. 功能简介

`TorchApplyWarp` 读取 FSL dense warp 或 FNIRT cubic coefficient，将 3D 图像或完整 4D 序列重采样到参考网格。新增 `apply_world()` / `run_world()` 接受显式 RAS world 变换链，支持 fMRI 的样条插值与逐帧运动组合。几何与 CUDA 计算使用 PyTorch，普通 CPU 入口复用 PyTorch 采样；World 的 `fmriprep` / `grid-constant` CPU 路径使用项目已有 SciPy，以匹配原版查询。NIfTI 读写使用 nibabel，运行时不调用 FSL。

同一 warp 可应用于全部时间帧，也可先创建采样计划，再传播同网格上的 FA、MD、组织概率图或标签图。计划复用变换坐标、插值网格、有效掩膜和最近邻索引；4D 采样按 channel 分块，复用 GPU 输入 buffer，并将结果直接回传到最终 CPU 数组。分块改变数据流，不减少体素或时间帧。

CUDA 默认开启 TF32 全局开关；FSL 几何和 cubic coefficient 展开显式使用 float64，影像插值保留原 float32。输出 dtype 由输入类型和 `output_dtype` 决定；float64 输出仍使用 float32 影像插值。没有使用 float16/bfloat16，也没有新增依赖。

## 2. Python 调用

```python
from fnit.applywarp import TorchApplyWarp

input_image_path = "/data/diffusion_corrected.nii.gz"  # 个体空间完整3D/4D NIfTI
reference_image_path = "/data/MNI152_reference.nii.gz"  # 目标空间3D网格
forward_warp_path = "/data/diffusion_to_MNI_coeff.nii.gz"  # 定义在目标网格的pull场/系数
output_image_path = "/data/diffusion_in_MNI.nii.gz"  # 目标空间结果
resampler = TorchApplyWarp(device="cuda:0")  # PyTorch设备；默认自动选择
resampled_result = resampler.run(
    input=input_image_path, reference=reference_image_path,
    warp=forward_warp_path, output=output_image_path,
    interpolation="trilinear",  # 连续图；标签图用nearest
)
```

### 输入数据格式

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

以上限制和坐标规则属于 FSL `__call__()` / `run()` 入口。SynthMorph 的 RAS pull displacement 使用该入口时，先经 `fnit.synthmorph.convert_warp_to_fsl(warp, moving=..., fixed=...)` 转为 intent-2006。显式 world 入口直接接受下面的 `WorldTransformChain`；两个入口的场数组和矩阵约定不能混用。

### 输入参数

| 参数 | 必需 | 类型 | 默认值 | 含义 |
|---|---|---|---|---|
| `device` | 否 | str / torch.device / None | `None` | 计算设备；显式 CUDA 不可用时报错。 |
| `frame_chunk_size` | 否 | int / None | `None` | 4D 每批帧数；None 根据工作内存自动选择。 |
| `input` | 是 | 路径 / NIfTI | 无 | 待处理影像，网格与空间由 NIfTI affine 定义。 |
| `reference` | 是 | 路径 / NIfTI / None | 无 | 参考图；决定输出空间、shape、affine。 |
| `warp` | 否 | 路径 / NIfTI | `None` | FSL dense 场或 intent-2007 FNIRT 系数。 |
| `premat` | 否 | 路径 / (4,4)数组 / None | `None` | input→warp源空间的 FSL scaled-mm 矩阵。 |
| `postmat` | 否 | 路径 / (4,4)数组 / None | `None` | warp目标→最终reference的 FSL scaled-mm 矩阵。 |
| `interpolation` | 否 | str | `'trilinear'` | 最终插值方法；可选范围见本页输入格式。 |
| `warp_convention` | 否 | str | `'auto'` | auto/relative/absolute；系数由 intent 解释。 |
| `output_dtype` | 否 | str / None | `None` | 保存数据类型；None 遵循输入类型和接口默认规则。 |
| `output` | 是 | 路径 / None | 无 | 输出文件或前缀；父目录按接口创建。 |

### 输出

```text
/data/
└── diffusion_in_MNI.nii.gz
```



`ApplyWarpResult` 包含：

- `image`：3D 输入输出 `(Xref,Yref,Zref)`；4D 输入输出 `(Xref,Yref,Zref,T)`，包括保留单帧 4D 的末维。空间 affine、qform/sform code 和 voxel size 来自 reference，4D 的 `pixdim[4]`/TR 来自 input；其余 header 字段及时间单位沿用 reference。
- `valid_mask`：`(Xref,Yref,Zref)` 的 NumPy bool 数组。warp 和 input 均在支持域内时为真；所有帧共享该空间掩膜。
- `qc`：设备、实际 TF32 开关、warp representation/intent/convention、插值方式、有效比例和输出 dtype。

无效体素仍按原规则乘以 0，因此保留原结果可能出现的负零符号位。整数输出遵循原版 NEWIMAGE 的直接类型转换，向零截断，不做饱和裁剪；例如显式 `char` 会将大于 255 的有限 MRI 强度回绕到 uint8。未指定类型时，浮点输入输出 float32/float64；整数输入通常根据输出动态范围小于 100 时改为 float32，否则保留原类型。超过 10 帧的整数输入沿用 FSL 原有范围规则，比较第一帧最大值与第二帧最小值；不会对每块重新选择 dtype。

### 重复同网格调用和 world 变换

`prepare(input, reference, warp=..., premat=..., postmat=..., interpolation=..., warp_convention=...)` 返回可复用的 `ApplyWarpPlan`；同网格新图用 `plan.apply(input, output_dtype=...)`。计划核验源网格和参考头信息，不能跨shape/affine重用。

| 参数 | 结构、含义及默认值 |
|---|---|
| `input` | 3D 或 4D NIfTI 路径/图像；含单帧的 4D 仍输出 4D。 |
| `transformation.reference` | 3D NIfTI 路径/图像，定义输出 shape、affine 和空间 header。 |
| `reference_to_source_world` | 4×4 RAS mm pull affine，从参考世界坐标到源参考帧世界坐标。 |
| `pre_affine_pull_ras` | 可选 target-grid `(Xref,Yref,Zref,3)` 位移。先加位移，再应用 affine；保留输入场的 float64。 |
| `motion_pull_world` | 可选 `(T,4,4)` RAS mm 矩阵，每帧在 affine 后应用；省略表示所有帧用同一变换。 |
| `coordinate_precision` | 默认 `float64`；`fmriprep` 保留已实现的场查询/坐标舍入顺序。 |
| `interpolation` | 默认 `spline`；可选 `linear`、`nearest`。CPU `fmriprep` / `grid-constant` 的 nearest 按 SciPy 原版查询；其余路径保留 PyTorch 最近邻舍入。 |
| `boundary` | 默认 `grid-constant`：样条使用 12 体素零预填充及 float64 系数；`periodic`：空间周期系数使用 float32，再按原输入支持域裁剪。 |
| `output_mask` | 可选 3D 输出空间 NIfTI，值 >0.5 为有效；乘法保留可能的负零。 |
| `batch_size` | 默认 8，正整数帧上限。CUDA 按坐标、FFT 系数和查询工作区保守检查十进制 20 GB 预算，超限报错；显式减小此值后重试。 |
| `spatial_chunk_size` | 默认 262144，正整数空间查询上限；可减小以降低工作区显存。 |
| `output` | `run_world()` 的输出路径，自动创建父目录。 |

`apply_world()` 返回 `nib.Nifti1Image`，`run_world()` 保存该图像并返回 `Path`。输出总是 float32，空间 qform/sform 及 code、extensions 来自 reference，4D TR 和时间单位来自 input。时间轴不插值。逐帧运动变换分别应用于完整帧，不能提前折叠成一张静态场。

显式 world 链的参数和完整示例见[world 入口](../../src/fnit/applywarp/README.md)。`run_world(input, transformation, output, ...)` 写盘；`apply_world` 仅返回结果。world-RAS 毫米不能当作 FSL scaled-mm 分量。

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

volume 流程自动构造 `WorldTransformChain`。在 `fnit-fmri volume` 中选择 `--registration-backend fnirt`，其最终采样调用 `TorchApplyWarp.run_world()`；选择 `--registration-backend synthmorph` 则调用 `apply_transform()`。[完整 BIDS 命令和所有参数](../fmri/README.md#命令行调用)。

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

## 5. 最新精度和运行时间

CPU 完整进程包含启动、导入、全部输入读取、计算和保存；双方固定相同物理核和线程上限。普通入口参照为 FSL 6.0.7.4，World 参照为 fMRIPrep 25.2.4 的原生重采样函数。每行保留实际冻结版本与统计协议。

| 完整操作及来源 | 1 CPU：原版 / FNIT（s） | 8 CPU：原版 / FNIT（s） | 统计范围 |
|---|---:|---:|---|
| 公开全 T1，trilinear / float，[v23 补测](cpu_normalization_all23_official_20261004.public.json) | 2.493 / 4.545 | 1.996 / 3.052 | 各一次完整观察；小型操作仍慢于官方。 |
| 105 帧 DWI，原网格 premat，[v8 矩阵](benchmark_20261004.public.json) | 49.861 / 8.875 | 34.281 / 5.541 | warmup 后三组配对中位数；81,768,960 值。 |
| 同病例 105 帧 DWI→MNI 系数，[CPU25](mni_dwi_105_20261004.public.json) | 457.843 / 11.243 | 431.278 / 7.571 | 各一次完整观察；94,776,045 值。 |
| World 完整 490 帧，[v20 数值核](world_cpu_latest_20261004.public.json) | 301.475 / 241.778 | 56.297 / 46.831 | 各一次完整观察；442,288,210 值。 |

GPU 表比较起始 FNIT `1d31e7b` 与各冻结候选。完整 API 包含输入读取、计算、CPU 回传及保存，排除进程启动和入口导入，函数内惰性导入计入。普通两例和 MNI 例各完成两组 AB/BA，每进程 warmup 一次再测三次，共 16 次保存；World 为两版各一次完整 490 帧调用，无 warmup。

| 完整 H100 操作 | 起始 FNIT API（s） | 候选 API（s） | 两版 peak allocation（B） | 全输出门槛 |
|---|---:|---:|---:|---|
| 公开全 T1，GPU23 | 0.594 / 0.573 | 0.564 / 0.540 | 1,536,427,520 | 16 次的 10,223,616 值及完整 metadata 逐位一致。 |
| 105 帧原网格 premat，GPU23 | 3.930 / 3.764 | 3.912 / 3.908 | 719,979,008 | 16 次的 81,768,960 值及完整 metadata 逐位一致。 |
| 105 帧→MNI 系数，GPU26 | 5.836 / 6.107 | 5.498 / 5.395 | 777,303,040 | 16 次的 94,776,045 值及完整 metadata 逐位一致。 |
| World 完整 490 帧，GPU20 | 34.103 | 36.943 | 2,002,300,416 | 442,288,210 值及整份保存文件逐字节一致。 |

[GPU23 普通入口](../../validation/multimodal_cpu_20261004/gpu_final_v23_20261004.public.json)、[GPU26 MNI 传播](../../validation/multimodal_cpu_20261004/gpu_inv_mni_v26_20261004.public.json)与 [GPU20 World](../../validation/multimodal_cpu_20261004/gpu_world_v20_20261004.public.json)分别保存源码 SHA、协议和 GPU 负载。当前普通入口 core 与 GPU26 的 SHA 相同，World 数值核与 GPU20 相同；GPU23 保留当时的冻结来源，后续 prepared CPU 门控没有改变 CUDA 算式。GPU 均使用同一 H100 UUID、TF32 和十进制 20 GB allocation 上限；共享负载时钟不作稳定速度结论。

### 分步骤 benchmark

| 阶段 | FNIT | 原软件 |
|---|---|---|
| 读取、几何准备、采样、回传、保存分别计时 | 最新摘要未单列，见逐调用报告 | 未单列 |

源码绑定：本页核验 main `140c3739`；表内仍是2026-10-04的各冻结版本，输入包含1对公开T1、1例完整105帧DWI及1例490帧fMRI。CPU为Intel Xeon Gold 6418H，预算1/8线程；GPU为H100、TF32，float32图像与float64几何，20GB allocation上限。不同入口计时范围已在表前说明，不能混算。

![105 帧 DWI 的首帧模板脑内官方输出、FNIT 输出和差值](../../validation/registration_lossless_20261002/figures/official_warp_001.png)

脑图属于2026-10-02固定DWI官方对照；最新CPU/GPU表未新增同版本脑图。

## 6. 最近版本和 benchmark

| 日期 | 更新 | 验证记录 |
|---|---|---|
| 2026-10-04，GPU23/26/20 汇总 | 普通全 T1、原网格 premat、同病例 MNI 系数传播与 World 完整 490 帧分别保留实际来源。 | 本页最新表列完整 CPU 进程、GPU API 与显存；GPU 数值和完整 metadata 一致，共享时钟不作稳定速度结论。 |
| 2026-10-04，CPU25/GPU26补测 | 同病例真实EDDY105帧和最新TBSS系数的完整MNI2mm传播，另核对重复CPU FP64 prepared查询的静态坐标优化。 | [MNI完整官方对照与步骤](MNI_DWI_CPU_BENCHMARK_20261004.md)、[CPU坐标门槛](CPU_NORMALIZATION_20261004.md)；H10026的全部array/header/ext/affine与基线位级一致，peak相同。 |
| 2026-10-04，all_v8 | CPU 三线性布局、query batch 与坐标分配优化；修复整数 clip 与 FSL 直接 cast 不一致。 | [50 组真实 CPU 官方配对及 H100 回归](CPU_BENCHMARK_20261004.md)；10 组达到完整进程速度目标，公开 nearest 五种保存类型逐值一致。 |
| 2026-10-02，公共 world 入口 | 新增 `WorldTransformChain` 与 `apply_world()` / `run_world()`；共享成熟的 world 数值核，保留样条边界、逐帧运动与 metadata。 | [完整 490 帧与实际调用验证](../../validation/fmri/public_resamplers_20261002/README.md)。 |
| 2026-10-02，上一轮 | 4D channel 分块、GPU 输入 buffer 复用、直接回传最终数组、nearest 索引缓存、原地乘法 mask；新增构造函数/函数入口/CLI 分帧参数。按第一阶段完整 DWI 的吞吐结果，自动政策改为 8 GiB 工作预算、取消固定 64 帧上限。 | [最终默认与三条 pipeline 验收](../../validation/registration_lossless_20261002/README.md)通过；保留[第一阶段配置实测](../../validation/registration_lossless_20261002/component_tuning.public.json)。 |

## 7. 参考文献、原软件和资源

源码位置：[原软件 `fugue-2201.3/applywarp.cc`](../../src/fnit/_vendor_fsl/sources/fugue-2201.3/applywarp.cc)；[FNIT `applywarp/core.py`](../../src/fnit/applywarp/core.py)。固定 tag/commit、Git tree 及每文件 SHA-256 见[来源清单](../../src/fnit/_vendor_fsl/manifest.json)。

代码改写沿用 [FSL 6.0 非商业许可证](../../licenses/FSL-6.0.txt)，完整来源与再分发要求见[第三方声明](../../THIRD_PARTY_NOTICES.md)。

- Andersson, Jenkinson & Smith. *Non-linear registration, aka spatial normalisation*. FMRIB Technical Report TR07JA2 (2007)，[原文](https://www.fmrib.ox.ac.uk/datasets/techrep/tr07ja2/tr07ja2.pdf)。
- [FSL FNIRT User Guide：applywarp](https://fsl.fmrib.ox.ac.uk/fsl/docs/registration/fnirt/user_guide.html#applywarp)。
- [官方 `applywarp.cc` 源码](https://git.fmrib.ox.ac.uk/fsl/fugue/-/blob/master/applywarp.cc)，位于 FSL `fugue` 仓库；[FNIRT 系数/场工具源码](https://git.fmrib.ox.ac.uk/fsl/fnirt)。
- [FNIT 当前实现](../../src/fnit/applywarp/core.py)。

| 资源 | 用途 | 官方来源 | 大小 | SHA-256 | 是否允许 FNIT 再分发 |
|---|---|---|---|---|---|
| 本功能无模型权重；用户自备参考影像、掩膜或变换 | 定义目标网格/变换 | 各文件原作者 | 按实际文件 | 按实际文件 | 不随本功能发布用户数据 |

[完整历史说明与调试证据](../../validation/applywarp/readme_archive_20261005.md) · [返回主页](../../README.md)

<!-- 旧版文档锚点兼容 -->
<a id="1-功能简介"></a> <a id="2-python-调用输入和输出"></a> <a id="21-完整-4d-序列"></a> <a id="22-同网格多图复用"></a> <a id="23-参数"></a> <a id="24-输出结构"></a> <a id="25-坐标文件格式和范围"></a> <a id="26-fmri-的显式-world-变换链"></a> <a id="3-命令行调用"></a> <a id="4-原软件调用"></a> <a id="5-最新验证精度和运行时间"></a> <a id="最新完整-cpu-与-gpu-记录"></a> <a id="普通入口-cpu-功能矩阵与实现"></a> <a id="同病例完整105帧dwimni系数路径"></a> <a id="显式-world-入口完整-fmri-volume"></a> <a id="最终默认策略105-帧完整-dwi"></a> <a id="第一阶段配置记录分帧选择"></a> <a id="2026-09-27固定官方-fnirt-warp-的真实-3d-fa"></a> <a id="6-最近更新与-benchmark-记录"></a> <a id="7-参考文献和原实现"></a>
