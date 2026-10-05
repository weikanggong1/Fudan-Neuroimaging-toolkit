# TorchTOPUP：相反相位编码b0畸变校正

| 项目 | 内容 |
|---|---|
| 输入 | 两帧相反PE b0及采集参数，或AP/PA原始目录 |
| 输出 | Hz场、系数、运动参数、校正b0和可选Jacobian |
| 对应原软件 | FSL TOPUP2203.2的b02b0.cnf路径 |
| Python / CLI | TorchTOPUP、run_ukb_topup / fnit topup、fnit-topup |
| CPU / GPU | CPU或CUDA；不调用FSL，无模型权重 |

## 1. 功能简介

TOPUP依据相反相位编码的两张b0估计共同磁场及运动，生成Hz场图和Jacobian调制后的校正图。公开接口一次处理一名受试者，采用b02b0.cnf完整九层计划，前五层联合LM、后四层SCG。

生产复用项目样条和求解器，不启动FSL。图像FP32，样条系数、几何和求解器FP64，CUDA默认TF32。2026-10-03修复原始b0选择器保存最后试探点评分的问题，改为返回优化器最终接受点的评分；场核心不变，记录见版本表。

## 2. Python 调用

```python
from fnit.topup import TorchTOPUP

opposite_b0_pair_path = "/data/B0_AP_PA.nii.gz"  # 两帧相反PE的原始b0
acquisition_parameters_path = "/data/acqparams.txt"  # 两行PE向量+读出时间秒
field_output_prefix = "/data/topup/fieldmap_out"  # 样条系数和运动参数根名
field_hz_output_path = "/data/topup/fieldmap_fout.nii.gz"  # Hz场图
corrected_b0_output_path = "/data/topup/fieldmap_iout.nii.gz"  # 两帧校正图
field_result = TorchTOPUP(device="cuda:0").run(
    imain=opposite_b0_pair_path, datain=acquisition_parameters_path,
    out=field_output_prefix, fout=field_hz_output_path,
    iout=corrected_b0_output_path,
)
```

### 输入数据格式

imain为有限(X,Y,Z,2)NIfTI，正好两帧、相反i/i-或j/j-；datain为两行四列文本，前三列单位PE向量，第四列TotalReadoutTime(秒)。两帧shape/affine必须相同。当前只支持b02b0.cnf九级配置，不支持k方向、多帧或其他正则模型。

### 输入参数

| 参数 | 必需 | 类型 | 默认值 | 含义 |
|---|---|---|---|---|
| `device` | 否 | str / torch.device / None | `None` | 计算设备；显式 CUDA 不可用时报错。 |
| `config` | 否 | TOPUPConfig / None | `None` | 构造 TorchTOPUP 的配置；None 使用下表默认值，Python 不接收配置文件名。 |
| `imain` | 是 | str / Path | 无 | 两帧相反 PE b0 的 NIfTI 文件，不直接接收内存影像。 |
| `datain` | 是 | str / Path | 无 | 两行四列：PE方向三分量和总读出时间(秒)。 |
| `out` | 是 | 路径 | 无 | 结果文件根名/前缀。 |
| `fout` | 否 | 路径 / None | `None` | Hz场图保存位置。 |
| `iout` | 否 | 路径 / None | `None` | reference网格上的重采样图。 |
| `jacout` | 否 | 路径 / None | `None` | 两张无量纲Jacobian的根名。 |
| `overwrite` | 否 | bool | `False` | 是否覆盖已有结果，默认保护原文件。 |

### 配置参数

| 参数 | 必需 | 类型 | 默认值 | 含义 |
|---|---|---|---|---|
| `warp_resolution_mm` | 否 | tuple[float, ...] | `(20, 16, 14, 12, 10, 6, 4, 4, 4)` | 控制点分辨率，mm。 |
| `subsampling` | 否 | tuple[int, ...] | `(2, 2, 2, 2, 2, 1, 1, 1, 1)` | 每层降采样倍数。 |
| `fwhm_mm` | 否 | tuple[float, ...] | `(8, 6, 4, 3, 3, 2, 1, 0, 0)` | 各轮/层平滑FWHM，mm。 |
| `maximum_iterations` | 否 | tuple[int, ...] | `(5, 5, 5, 5, 5, 10, 10, 20, 20)` | 每层最大迭代/接受步数。 |
| `regularization` | 否 | tuple[float, ...] | `(0.005, 0.001, 0.0001, 1.5e-05, 5e-06, 5e-07, 5e-08, 5e-10, 1e-11)` | 各层形变正则化权重。 |

旧optimizer_steps_per_iteration已移除；用maximum_iterations表达九层预算。Python可传TOPUPConfig，但CLI只接受b02b0.cnf。

### AP/PA原始输入准备

`--raw-dir` 需要以下单被试文件：

```text
/path/to/subject/raw/
├── AP.nii.gz
├── AP.bval
├── AP.json
├── PA.nii.gz
├── PA.bval
└── PA.json
```

NIfTI 第四维必须与对应 `.bval` 长度相同。JSON 需要 `PhaseEncodingDirection`，并提供 `TotalReadoutTime`，或提供可换算总读出时间的 `EffectiveEchoSpacing`。当前实现支持 `i/i-` 或 `j/j-`，AP 与 PA 必须方向相反、矩阵和几何一致。

UKB 准备步骤会在 AP、PA 中分别找出 `b<100 s/mm²` 的候选帧，对候选帧执行 GPU 6-DOF 刚体配准，以有效非零交集内的 Pearson 相关为目标，计算两两相关均值，并执行 UKB 的选择规则：第一帧得分不低于 0.98 时选择第一帧，否则选择最高分帧。2026-10-03 的本次修复源码在优化器最终接受的参数处计算相关分数。随后写出选中的两帧原始图像 `B0_AP_PA.nii.gz` 和两行 `acqparams.txt`。这条 PyTorch 选择器没有调用 FLIRT；在本页的一例真实数据中，它与官方 FLIRT/`fslcc` 流程都选择 AP 第 0 帧和 PA 第 0 帧，但相关分数并不相同。

`run_ukb_topup(raw_dir, output_dir, device=None, overwrite=False, pair_geometry="strict")`先选择b<100的AP/PA b0再估计场，返回(result,prepared)。输入/输出目录必需，device/overwrite可选；prepared记录原始零起始帧索引、候选分数与imain/datain路径。b0候选选择用FNIT刚体相关，不调用FLIRT；与官方选帧可以不同。

| 原始输入参数 | 必需 | 类型 | 默认值 | 含义 |
|---|---|---|---|---|
| raw_dir / output_dir | 是 | 路径 | 无 | AP/PA原目录与准备/校正输出 |
| device | 否 | 设备/None | None | 默认可用CUDA否则CPU |
| overwrite | 否 | bool | False | 覆盖已有输出 |
| pair_geometry | 否 | str | strict | strict核验几何；fslmerge-first使用AP头定义合并网格，另写pair_geometry.json |

pair_geometry的兼容模式只改变合并pair的头信息合同，不估计AP与PA之间的空间配准。CLI尚未暴露该参数；需要显式选择时使用Python。

### 输出

```text
topup/
├── fieldmap_out_fieldcoef.nii.gz
├── fieldmap_out_movpar.txt
├── fieldmap_fout.nii.gz
└── fieldmap_iout.nii.gz
# 请求jacout另生成 _01/_02.nii.gz
```

| FNIT 输出 | 原生 TOPUP 输出 | 形状与单位 |
|---|---|---|
| `fieldmap_out_fieldcoef.nii.gz` | 同名 | `[55,55,39]`（本例），float32，intent 2016；pixdim 是 knot spacing，qform offset 编码完整场尺寸。 |
| `fieldmap_out_movpar.txt` | 同名 | `[2,6]`；第一帧固定为零，反向帧沿相位编码轴的平移固定为零。 |
| `fieldmap_fout.nii.gz` | `--fout` | `[X,Y,Z]`，float32，Hz，intent 2018。 |
| `fieldmap_iout.nii.gz` | `--iout` | `[X,Y,Z,2]`，float32，输入几何。 |
| `fieldmap_jacout_01.nii.gz`、`_02` | `--jacout_01/02` | `[X,Y,Z]`，float32；保留 TOPUP 的 qform/sform-code 0 文件合同。 |

| Python 字段 | 内容 |
|---|---|
| `result.field_hz` | `--fout` 对应的 3D Hz 场图，NIfTI intent 2018。 |
| `result.corrected` | `--iout` 对应的 `[X,Y,Z,2]` Jacobian 调制校正图。 |
| `result.corrected_mean` | 两帧校正图的内存均值；原生 `topup --iout` 不单独写该文件。 |
| `result.jacobians` | `--jacout_01/02` 对应的两个 3D Jacobian。 |
| `result.coefficients` | `--out_fieldcoef` 对应的 cubic B-spline coefficient NIfTI，intent 2016。 |
| `result.movement_parameters` | `--out_movpar.txt` 对应的两行六列参数，平移单位 mm，旋转单位 rad。 |
| `result.qc` | 设备、精度、TF32、各层 SSD/正则化能、有效体素、LM/SCG 接受步及收敛状态、同步计算时间和 CUDA 峰值显存。 |

## 3. 命令行调用

```bash
fnit topup \
  --imain B0_AP_PA.nii.gz \
  --datain acqparams.txt \
  --config b02b0.cnf \
  --out fieldmap_out \
  --fout fieldmap_fout \
  --iout fieldmap_iout \
  --jacout fieldmap_jacout \
  --device cuda:0
```

| CLI 参数 | Python 参数 | 含义 |
|---|---|---|
| `--imain / --datain` | `imain / datain` | 两帧b0与采集参数 |
| `--out / --fout / --iout / --jacout` | `out / fout / iout / jacout` | 输出根名和文件 |
| `--config` | `config` | CLI只接受b02b0.cnf |
| `--raw-dir / --output-dir` | `raw_dir / output_dir` | 原始AP/PA模式 |
| `--device / --overwrite` | `device / overwrite` | 设备与覆盖 |

## 4. 原软件调用

```bash
topup \
  --imain=B0_AP_PA.nii.gz \
  --datain=acqparams.txt \
  --config=b02b0.cnf \
  --out=fieldmap_out \
  --fout=fieldmap_fout \
  --iout=fieldmap_iout \
  --jacout=fieldmap_jacout
```

原程序只用于隔离benchmark。imain/datain/out/fout/iout/jacout与Python同名参数逐项对应；config为本页九级b02b0.cnf。FNIT增加设备和覆盖选项，原始AP/PA模式的选择器不是独立topup命令。

## 5. 最新精度和运行时间

最新独立场估计的正式FSL对照为2026-10-02，同一例完整104×104×72×2输入，3次FNIT估计、2次原FSL估计；源码core.py SHA d6b9838c…、CUDA sampler ee19a764…，完整值见[报告](../../validation/topup/report.matched_20261002.public.json)。FSL6.0.7.4、PyTorch2.5.1、H100、8CPU线程，TF32、20GB限额；图像FP32/系数与求解器FP64。

| 输出与范围 | Pearson r | MAE | RMSE |
|---|---:|---:|---:|
| Hz场，固定信号区 | 0.9999998778 | 0.005255Hz | 0.010799Hz |
| 两帧校正图，固定官方脑区 | 0.9999965618 | 2.703056 | 12.849295 |
| Jacobian01/02，同脑区 | 0.9999965295 | 0.000223 | 0.000407 |

### 端到端与分步骤 benchmark

| 范围（中位数） | FNIT | FSL |
|---|---:|---:|
| 完整进程 | 10.82s，含额外统计/JSON | 211.10s，原程序 |
| API读取/计算/保存 | 7.094846s | 未单列 |
| 内部同步计算 | 6.368049s | 未单列 |
| CUDA peak allocated | 389,119,488B | CPU，GPU不适用 |

[GPU重复](../../validation/topup/repeats_20261002.public.json)和[原程序重复](../../validation/topup/reference_repeats_20261002.public.json)保留不同计时口径。2026-10-03接受点评分修复是独立10人选择器回归，选帧未改变；另2例同pair场诊断见[报告](../../validation/dmri_pipeline/public10_20261002/matched_topup_pair_20261003.public.json)，未重测最新main完整TOPUP/EDDY链。

最新场核心没有独立公开脑图；[旧L-BFGS图及来源](../../validation/topup/readme_archive_20261005.md#历史-l-bfgs-版与-fsl-6074-的真实数据对照)明确属于被替代版本。

## 6. 最近版本和 benchmark

| 日期 | 代码与 benchmark 范围 |
|---|---|
| 2026-10-03 | 修复 UKB b0 选择器返回最后试探点评分的问题，改为最终接受点评分；16 项 CPU 回归与真实十人选择器检查完成。30 个新 AP pair 分数与接受点诊断相同，十人索引不变；case02/08 与官方的不同选帧未消除。冻结 20 个整链仍为 bf339a0，独立组件结果不改标旧输出或时钟。 |
| 2026-10-02 | 补齐默认源图 regrid，修正目标函数支持区、周期平滑、图像插值精度及索引、层间场传递、FSL storage 约定，改为联合 LM/SCG；一例同输入真实场图信号区 RMSE 0.010799 Hz，固定参数脑内 iout RMSE 0.011135；该冻结源码三次 API 中位数 7.094846 s，104 项组合回归通过，独立估计非逐元素一致。 |
| FNIT 0.16.0 | 与 0.14.0 数值文件逐字节相同；保留当时一例真实 pair 的 L-BFGS 对照。 |
| FNIT 0.14.0 | 初版 AP/PA 路径、FSL 文件合同、单病例精度与三次共享节点计时；不声明数值等价。 |

## 7. 参考文献、原软件和资源

源码位置：[原软件 `topup-2203.2/topup.cpp`](../../src/fnit/_vendor_fsl/sources/topup-2203.2/topup.cpp)；[FNIT `topup/core.py`](../../src/fnit/topup/core.py)。固定 tag/commit、Git tree 及每文件 SHA-256 见[来源清单](../../src/fnit/_vendor_fsl/manifest.json)。

### 源码与许可

UKB AP/PA b0 准备顺序参考 UK Biobank brain imaging pipeline v1.5；场估计实现参照 FSL TOPUP `2203.2`（commit `3e2cb9104e834ce18c10e4b7edddbd500d0c459c`）、basisfield、miscmaths、newimage 和 warpfns。完整、未修改的上游 TOPUP 源码、每文件 SHA-256 和 Git tree 保存在 [`src/fnit/_vendor_fsl`](../../src/fnit/_vendor_fsl/README.md)。修改后的 PyTorch 源码与上游源码一同发布，受 [`FSL Software Licence 6.0`](../../licenses/FSL-6.0.txt) 的非商业条款约束。本项目不是 FSL 官方发布。

### Reference

- 参考文献：Andersson, Skare & Ashburner, *How to correct susceptibility distortions in spin-echo echo-planar images: application to diffusion tensor imaging*, NeuroImage (2003), [doi:10.1016/S1053-8119(03)00336-7](https://doi.org/10.1016/S1053-8119(03)00336-7)。
- 原实现代码库：[FSL `topup`](https://git.fmrib.ox.ac.uk/fsl/topup)。

| 资源 | 用途 | 官方来源 | 大小 | SHA-256 | 是否允许 FNIT 再分发 |
|---|---|---|---|---|---|
| 本功能无模型权重；用户自备参考影像、掩膜或变换 | 定义目标网格/变换 | 各文件原作者 | 按实际文件 | 按实际文件 | 不随本功能发布用户数据 |

[完整历史说明与调试证据](../../validation/topup/readme_archive_20261005.md) · [返回主页](../../README.md)

<!-- 旧版文档锚点兼容 -->
<a id="ukb-原始数据目录"></a> <a id="单被试命令行"></a> <a id="单被试-python-调用"></a> <a id="输出文件合同"></a> <a id="实现范围"></a> <a id="2026-10-03-b0-选择器接受点评分修复"></a> <a id="2026-10-03-两例相同-b0-对的独立诊断"></a> <a id="2026-10-02-数值修复与验收边界"></a> <a id="2026-10-02-新版同输入真实验收"></a> <a id="固定参数与首层梯度检查"></a> <a id="计时范围与报告"></a> <a id="历史发布验收状态"></a> <a id="历史-l-bfgs-版与-fsl-6074-的真实数据对照"></a> <a id="源码与许可"></a> <a id="reference"></a> <a id="最近更新"></a>
