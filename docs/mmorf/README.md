# TorchMMORF：多标量与扩散张量联合配准

| 项目 | 内容 |
|---|---|
| 输入 | 配对3D标量、六通道tensor与可选scaled-mm矩阵 |
| 输出 | 参考网格warp、Jacobian、重采样图和QC |
| 对应原软件 | FSL MMORF0.3.2 |
| Python / CLI | run_mmorf / fnit mmorf、fnit-mmorf |
| CPU / GPU | CPU或CUDA；自动线性调用FNIT FLIRT |

## 1. 功能简介

`run_mmorf` 按顺序接收一组或多组标量图配对，以及一组 diffusion tensor 配对，并估计一套共享的非线性形变。第一张 reference scalar 定义输出网格。缺少线性矩阵时，函数会先调用 FNIT 的 PyTorchFLIRT 计算矩阵；显式传入矩阵时直接使用。`fnit-mmorf` 与 dMRI pipeline 的 MMORF 分支都调用这个函数，运行时不启动 FSL executable。

CUDA 路径使用 float32 并默认允许 TF32，不使用 float16 或 bfloat16。默认五层计划与 MMORF 0.3.2 配置一致：控制点分辨率为 32、32、16、8、4 mm，平滑为 8、8、4、2、1 mm，每层 5 次更新。当前实现包含 world-mm cubic 控制格点、reference-axis mm warp、多组标量与张量代价、逐模态初始代价缩放、tensor reorientation、Bkk/SPRED 正则和 LBFGS 优化路径；数值结果仍未与官方逐体素等价。

## 2. Python 调用

```python
from fnit.mmorf import run_mmorf

moving_scalar_path = "/data/dti_FA.nii.gz"  # 个体3D FA
reference_scalar_path = "/data/template_FA.nii.gz"  # 参考FA，定义输出空间
moving_tensor_path = "/data/dti_tensor.nii.gz"  # 个体六通道tensor
reference_tensor_path = "/data/template_tensor.nii.gz"  # 参考六通道tensor
registration_output_directory = "/data/mmorf"  # 所有结果的目录
multimodal_result = run_mmorf(
    moving_scalar=moving_scalar_path, reference_scalar=reference_scalar_path,
    moving_tensor=moving_tensor_path, reference_tensor=reference_tensor_path,
    output_dir=registration_output_directory, device="cuda:0",
    auto_linear=True,  # 缺失矩阵由FNIT FLIRT估计
)
```

### 输入数据格式

标量是有限3D NIfTI/影像对象，可用按模态一一配对的列表。tensor末轴6，顺序Dxx,Dxy,Dxz,Dyy,Dyz,Dzz，单位与输入扩散率一致。所有矩阵为到第一张reference标量的FSL scaled-mm变换，不是NIfTI world affine。显式矩阵优先；auto_linear=False时缺项按identity，同网格但未对齐仍须传矩阵。

### 输入参数

| 参数 | 必需 | 类型 | 默认值 | 含义 |
|---|---|---|---|---|
| `moving_scalar` | 是 | 路径 / NIfTI / 列表 | 无 | 个体3D标量，按模态配对。 |
| `reference_scalar` | 是 | 路径 / NIfTI / 列表 | 无 | 对应3D模板，第一张决定输出空间。 |
| `moving_tensor` | 是 | 路径 / NIfTI | 无 | 个体(X,Y,Z,6)扩散张量。 |
| `reference_tensor` | 是 | 路径 / NIfTI | 无 | 模板(X,Y,Z,6)扩散张量。 |
| `output_dir` | 是 | 路径 | 无 | 本次调用的多文件输出目录。 |
| `moving_scalar_affine` | 否 | 矩阵 / 路径 / 列表 / None | `None` | 每张moving标量到第一张reference的scaled-mm矩阵。 |
| `reference_scalar_affine` | 否 | 矩阵 / 路径 / 列表 / None | `None` | 各reference到首张reference，首项须identity。 |
| `moving_tensor_affine` | 否 | 矩阵 / 路径 / None | `None` | 个体tensor到第一张reference矩阵。 |
| `reference_tensor_affine` | 否 | 矩阵 / 路径 / None | `None` | 模板tensor到第一张reference矩阵。 |
| `scalar_weights` | 否 | 数值列表 / None | `None` | 每组非负代价权重；None各组1/N。 |
| `auto_linear` | 否 | bool | `True` | 缺矩阵时调用FNIT FLIRT；False缺项按identity。 |
| `device` | 否 | str / torch.device / None | `None` | 计算设备；显式 CUDA 不可用时报错。 |
| `config` | 否 | MMORFConfig / None | `None` | 五层多模态配置对象；None 用下表默认值，Python 不接收 INI 路径。 |
| `overwrite` | 否 | bool | `False` | 是否覆盖已有结果，默认保护原文件。 |

### 配置参数

| 参数 | 必需 | 类型 | 默认值 | 含义 |
|---|---|---|---|---|
| `warp_resolution_mm` | 否 | tuple[float, ...] | `(32.0, 32.0, 16.0, 8.0, 4.0)` | 控制点分辨率，mm。 |
| `smoothing_mm` | 否 | tuple[float, ...] | `(8.0, 8.0, 4.0, 2.0, 1.0)` | 各层Gaussian平滑FWHM，mm。 |
| `regularization` | 否 | tuple[float, ...] | `(400000.0, 0.37, 0.31, 0.26, 0.22)` | 各层形变正则化权重。 |
| `iterations` | 否 | tuple[int, ...] | `(5, 5, 5, 5, 5)` | 每层 LBFGS 最大更新次数，须为正整数。 |
| `scalar_weight` | 否 | float | `1.0` | 总体标量项权重。 |
| `tensor_weight` | 否 | float | `1.0` | 总体张量项权重。 |
| `learning_rate` | 否 | float | `1.0` | LBFGS初始步长。 |

### 输出

```text
mmorf/
├── mmorf_warp.nii.gz
├── mmorf_jacobian.nii.gz
├── mmorf_warped_scalar.nii.gz
├── mmorf_warped_scalar_2.nii.gz
├── mmorf_warped_tensor.nii.gz
└── mmorf_report.json
```

| 文件 | shape | dtype | 定义 |
|---|---|---|---|
| `mmorf_warp.nii.gz` | `[Xref,Yref,Zref,3]` | float32 | relative pull displacement；三通道为沿 reference image axes 的毫米位移。 |
| `mmorf_jacobian.nii.gz` | `[Xref,Yref,Zref]` | float32 | `det(I + ∂u/∂x)`，只含 nonlinear warp。 |
| `mmorf_warped_scalar.nii.gz` | `[Xref,Yref,Zref]` | float32 | 第一张 moving scalar 的公共空间结果。 |
| `mmorf_warped_scalar_2.nii.gz` 等 | `[Xref,Yref,Zref]` | float32 | 第 2 张及之后的标量结果；单模态时没有。 |
| `mmorf_warped_tensor.nii.gz` | `[Xref,Yref,Zref,6]` | float32 | 六通道 moving tensor 的公共空间采样结果。 |
| `mmorf_report.json` | JSON | — | 各模态线性矩阵、配准方式、设备、五层 QC、耗时和显存。 |

MMORF warp 与 FNIRT/applywarp 的 coefficient/dense-warp 文件不是同一种坐标合同，不能互换。设 `u_axis(x)` 为保存的三通道值，`Rref` 为 reference affine 去除 voxel scaling 后的轴方向矩阵，则公共 world 位移为 `Rref · u_axis(x)`；pull 位置再通过 input affine 的逆映射到 moving voxel。这个定义已用 anisotropic 和 oblique affine 单元测试验证。

官方 MMORF 只直接输出 warp 和 Jacobian；debug 路径可保存 warped scalar。`mmorf_warped_tensor.nii.gz` 是 FNIT 的便利输出，本次官方对照没有 warped-tensor oracle，因此不声明它逐体素等价。

`apply_mmorf_warp(image, reference, warp, affine=..., device=..., interpolation="linear")`应用现有场；linear/nearest/cubic分别用于连续图/标签/样条。多图同网格可复用`prepare_mmorf_warp`，参数与示例见[组件页](../../src/fnit/mmorf/README.md)。

### 将固定MMORF场应用于参数图

```python
from fnit.mmorf import apply_mmorf_warp

native_parameter_path = "/data/dti_FA.nii.gz"  # 原个体网格参数图
reference_image_path = "/data/template_FA.nii.gz"  # 已有场的目标网格
mmorf_field_path = "/data/mmorf/mmorf_warp.nii.gz"  # 参考轴mm位移
input_to_reference_matrix_path = "/data/FA_to_MNI.mat"  # scaled-mm正向矩阵
registered_parameter_image = apply_mmorf_warp(
    image=native_parameter_path, reference=reference_image_path,
    warp=mmorf_field_path, affine=input_to_reference_matrix_path,
    device="cuda:0", interpolation="linear",
)
```

此函数返回nibabel影像，不自动写文件；保存需调用nibabel.save。warp必须与reference同shape/affine，不能直接传FNIRT系数。

| 参数 | 必需 | 类型 | 默认值 | 含义 |
|---|---|---|---|---|
| image / reference / warp | 是 | 路径或NIfTI | 无 | 原始图、参考网格与MMORF场 |
| affine | 否 | 矩阵/路径 | None | image→reference的scaled-mm矩阵；None为identity |
| device | 否 | 设备 | None | 默认可用CUDA否则CPU |
| interpolation | 否 | str | linear | linear / nearest / cubic |

## 3. 命令行调用

```bash
fnit-mmorf \
  --mov-scalar t1_brain.nii.gz \
  --ref-scalar MNI152_T1_1mm_brain.nii.gz \
  --mov-scalar dti_FA.nii.gz \
  --ref-scalar FSL_HCP1065_FA_1mm.nii.gz \
  --mov-tensor dti_tensor.nii.gz \
  --ref-tensor FSL_HCP1065_tensor_1mm.nii.gz \
  --aff-mov-scalar t1_to_MNI.mat \
  --aff-mov-scalar AUTO \
  --aff-mov-tensor FA_to_MNI.mat \
  --scalar-weight 0.5 \
  --scalar-weight 0.5 \
  -o mmorf \
  --device cuda:0
```

重复的 `--mov-scalar` 和 `--ref-scalar` 按出现顺序配对；上例依次为 T1 和 FA。`--aff-mov-scalar` 的第一个值是 T1 → T1 模板的已有矩阵，第二个 `AUTO` 让内部 PyTorchFLIRT 配准 FA → FA 模板；`--aff-mov-tensor` 是 tensor → 输出空间的已有矩阵。两次 `--scalar-weight` 分别设置两组标量代价。其他缺失矩阵默认自动计算；若第二张 reference scalar 不与首张同网格，可重复 `--aff-ref-scalar` 并用 `AUTO` 占位需要自动计算的位置。`--no-auto-linear` 会把缺失矩阵改为 identity。`-o` 指定目录，`--device` 指定设备；覆盖已有结果必须显式加 `--overwrite`。

## 4. 原软件调用

```bash
mmorf --config multimodal.ini
```

独立benchmark用相同输入和矩阵；[完整可复制INI配置](../../validation/mmorf/readme_archive_20261005.md#对应的官方-mmorf-032-调用)。`img_mov/ref_scalar`按FNIT列表顺序重复，aff_mov/ref_scalar对应各线性矩阵；`img_mov/ref_tensor`对应六通道张量，warp_out/jac_det_out对应FNIT输出。FNIT自动矩阵、设备和warped_tensor属于便利接口；当前LBFGS求解器与官方LM/MM不同，非逐体素等价。

## 5. 最新精度和运行时间

既有官方对照报告使用一例真实 T1w、DTI FA 和六通道 tensor，对应的 T1、FA 与 tensor 模板，计算两组标量加一组张量的共享 warp。T1 与 tensor 的既有线性矩阵保持固定；FA 标量的矩阵由内部 PyTorchFLIRT 12-DOF CorRatio 自动估计。两组标量权重均为 0.5。对照使用同一组图像和矩阵运行官方 MMORF 0.3.2。具体命令、完整 3D 指标及输入边界见[冻结报告](../../validation/mmorf/report.public.json)。

| 运行 | 时间 | 峰值分配显存 |
|---|---:|---:|
| FNIT：使用相同线性矩阵，五层非线性求解 | 63.22 s | 14.30 GB |
| FNIT：使用相同线性矩阵，含载入与写盘 | 79.78 s | 14.30 GB |
| FNIT：内部自动估计 FA 矩阵 | 210.27 s | 0.82 GB |
| FNIT：自动线性、非线性和写盘全程 | 287.12 s | 14.30 GB |
| 官方 MMORF：非线性求解 | 1161.31 s | 未提供进程 GPU 峰值 |
| 官方 MMORF：含写盘完整命令 | 1183.61 s | 未提供进程 GPU 峰值 |

FNIT 与官方的输出均可读取，warp 和 Jacobian 的 shape、affine、dtype 合同一致。以下 Pearson、MAE、RMSE 由同一参考脑掩膜上的完整 3D 图计算；FA 还要求两张输出都非零。两组 warped scalar 均用 FNIT cubic sampler 从各自 warp 生成。

| 输出 | Pearson r | MAE | RMSE |
|---|---:|---:|---:|
| warp 三通道，mm | 0.978064 | 0.275535 | 0.434509 |
| Jacobian | 0.970834 | 0.036512 | 0.061564 |
| warped T1，原图强度 | 0.958732 | 49.158114 | 86.690190 |
| warped FA | 0.983303 | 0.015765 | 0.029957 |

![真实 T1 和 FA 的官方与 FNIT 配准切片](figures/mmorf_fsl_comparison.png)

自动配准得到的 FA 矩阵与配对对照所用矩阵相同，自动和显式矩阵两次 FNIT warp 的最大绝对差为 0。共享 H100 GPU 运行前已有其他进程，FNIT 进程限制为整卡显存的 24%，前后全卡占用见机器报告。上述时间是本例实测，不能推断稳定加速比；各输出尚未达到逐体素数值等价。

本页核验main140c3739；上述正式报告来自已发布历史实现，未重新运行当前main。正式配准绑定 FNIT 0.16.0、FSL 6.0.7.4/MMORF 0.3.2；运行时 Python/PyTorch、CPU型号及线程数未记录。H100 PCIe GPU0共享、TF32、FP32图像，显存分配限额为整卡的24%。九图传播的最新组件记录见[2026-10-02验收](../../validation/dmri_pipeline/map_propagation_20261002.md)，不替代配准估计对照。

## 6. 最近版本和 benchmark

| 日期 | commit/version | 变化 | benchmark |
|---|---|---|---|
| 2026-10-02 | 九图传播版本 | 新增固定warp/affine采样计划复用 | [九图真实记录](../../validation/dmri_pipeline/map_propagation_20261002.md)，热调用0.517→0.270s，排除输入/保存 |
| 已发布联合配准版 | 源码SHA见报告 | 多标量+tensor、自动线性与显式矩阵入口 | [官方0.3.2对照](../../validation/mmorf/report.public.json)；未数值等价 |
| 原始实现 | 版本见历史报告 | 单标量/单tensor基础配准 | [完整历史](../../validation/mmorf/readme_archive_20261005.md) |

## 7. 参考文献、原软件和资源

源码位置：[官方 MMORF 0.3.2 固定提交](https://git.fmrib.ox.ac.uk/fsl/MMORF/-/tree/1c1c13b8368f05e1a79a6dafe919d6b61df36bd6)；[FNIT 求解与采样](../../src/fnit/mmorf/core.py)；[单次调用入口](../../src/fnit/mmorf/standalone.py)。代码改写遵守 [FSL 6.0 非商业许可](../../licenses/FSL-6.0.txt)。

- 参考文献：Lange et al., *MMORF—FSL’s MultiMOdal Registration Framework*, Imaging Neuroscience (2024), [doi:10.1162/imag_a_00100](https://doi.org/10.1162/imag_a_00100)。
- 原实现代码库：[FSL `MMORF`](https://git.fmrib.ox.ac.uk/fsl/MMORF)。

| 资源 | 用途 | 官方来源 | 大小 | SHA-256 | 是否允许 FNIT 再分发 |
|---|---|---|---|---|---|
| T1/FA/tensor模板 | 定义标量和张量目标空间 | [FSL MMORF官方资料](https://fsl.fmrib.ox.ac.uk/fsl/docs/registration/mmorf.html) | 按文件，本报告未集中记录 | 按文件，见实际输入报告 | 未证明文件级再分发许可；从原站获取 |

本功能不使用模型权重，发行包不包含官方MMORF源码/程序。

[完整历史说明与调试证据](../../validation/mmorf/readme_archive_20261005.md) · [返回主页](../../README.md)

<!-- 旧版文档锚点兼容 -->
<a id="输入"></a> <a id="python-单被试调用"></a> <a id="命令行单被试调用"></a> <a id="输出目录与坐标定义"></a> <a id="对应的官方-mmorf-032-调用"></a> <a id="源码对应与剩余差异"></a> <a id="真实数据-benchmark"></a> <a id="来源与许可"></a> <a id="2026-10-02同网格多指标图传播"></a> <a id="reference"></a>
