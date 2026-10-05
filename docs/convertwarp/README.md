# TorchConvertWarp：组合矩阵与非线性场

| 项目 | 内容 |
|---|---|
| 输入 | 目标3D网格、FSL非线性场及前后矩阵，或MMORF场和FA矩阵 |
| 输出 | float32三通道相对位移或绝对pull坐标 |
| 对应原软件 | FSL convertwarp；MMORF转换为FNIT扩展 |
| Python / CLI | TorchConvertWarp / fnit convertwarp、fnit-convertwarp |
| CPU / GPU | CPU或CUDA；生产不调用FSL |

## 1. 功能简介

`TorchConvertWarp` 生成标准空间网格上的 FSL pull 位移场。它支持两类输入：FSL dense/FNIRT 系数场及可选的前后 FLIRT 矩阵；FNIT MMORF 的参考图像轴毫米场及其独立的 FA→MNI FLIRT 矩阵。前一类按 `premat → warp1 → postmat` 组合；后一类先把 MMORF 坐标约定换成 FSL scaled-mm，再把 affine 和非线性变换合为一张场。矩阵不能直接用 NIfTI 的 RAS affine 代替。运行时只使用本包代码、Nibabel、NumPy 和 PyTorch，不调用 FSL。

FNIT dMRI pipeline 的 **TBSS** 输出 `registration/dti_FA_to_MNI_warp.nii.gz` 已经含 FA→MNI 线性部分；将它交给 `warp1` 时**不要**再传 `dti_FA_to_MNI_affine.mat`。**MMORF** 输出 `registration/mmorf_warp.nii.gz` 则是另一种坐标约定，必须使用下述 `from_mmorf` / `--mmorf-warp` 和同目录的 `dti_FA_to_MNI_affine.mat`。这两个分支共用后续 [TorchInvWarp](../invwarp/README.md)。

## 2. Python 调用

```python
from fnit.convertwarp import TorchConvertWarp

reference_image_path = "/data/MNI152_reference.nii.gz"  # 3D目标网格
nonlinear_warp_path = "/data/T1_to_MNI_coeff.nii.gz"  # T1→MNI系数或dense pull场
input_to_T1_matrix_path = "/data/diffusion_to_T1.mat"  # diffusion→T1 scaled-mm矩阵
output_warp_path = "/data/diffusion_to_MNI_warp.nii.gz"  # 完整组合相对场
composed_warp_result = TorchConvertWarp(device="cuda:0").run(
    reference=reference_image_path, warp1=nonlinear_warp_path,
    premat=input_to_T1_matrix_path, output=output_warp_path,
    output_convention="relative",  # 保存FSL scaled-mm位移
)
```

### 输入数据格式

reference是3D网格；warp1是(X,Y,Z,3)dense場或intent-2007样条系数。矩阵须有限、可逆且最后一行[0,0,0,1]；场中NaN/Inf报错。dense auto遵循FSL启发式，建议显式声明。

### 输入参数

| 参数 | 必需 | 类型 | 默认值 | 含义 |
|---|---|---|---|---|
| `device` | 否 | str / torch.device / None | `'cpu'` | 计算设备；显式 CUDA 不可用时报错。 |
| `reference` | 是 | 路径 / NIfTI / None | 无 | 参考图；决定输出空间、shape、affine。 |
| `warp1` | 是 | 路径 / NIfTI | 无 | 一张非线性场；末轴3或 intent-2007 系数。 |
| `premat` | 否 | 路径 / (4,4)数组 / None | `None` | input→warp源空间的 FSL scaled-mm 矩阵。 |
| `postmat` | 否 | 路径 / (4,4)数组 / None | `None` | warp目标→最终reference的 FSL scaled-mm 矩阵。 |
| `warp_convention` | 否 | str | `'auto'` | auto/relative/absolute；系数由 intent 解释。 |
| `output_convention` | 否 | str | `'relative'` | relative位移或absolute pull坐标，单位 FSL scaled-mm。 |
| `output` | 是 | 路径 / None | 无 | 输出文件或前缀；父目录按接口创建。 |

### MMORF 输入

| 参数 | 必需 | 类型 | 默认值 | 含义 |
|---|---|---|---|---|
| `reference` | 是 | 路径 / NIfTI / None | 无 | 参考图；决定输出空间、shape、affine。 |
| `source` | 是 | 路径 / NIfTI | 无 | 产生线性矩阵的原始 FA 网格，用于 scaled-mm 转换。 |
| `mmorf_warp` | 是 | 路径 / NIfTI | 无 | reference网格的参考影像轴毫米位移。 |
| `affine` | 是 | 路径 / (4,4)数组 / None | 无 | source/input→reference的 FSL scaled-mm 矩阵。 |
| `output_convention` | 否 | str | `'relative'` | relative位移或absolute pull坐标，单位 FSL scaled-mm。 |
| `output` | 是 | 路径 / None | 无 | 输出文件或前缀；父目录按接口创建。 |

`run_mmorf`保存结果，`from_mmorf`只返回对象。`source`必须是产生affine的原FA网格；`mmorf_warp`须为reference.shape+(3,)且同affine的有限float32场，单位为参考影像轴mm。TBSS系数已含affine，不能重复传同一矩阵；MMORF模式必须显式提供FA→MNI矩阵。

### 输出

```text
/data/
└── diffusion_to_MNI_warp.nii.gz
```

返回WarpFieldResult：image为NIfTI；valid_fraction为warp查询落在输入场网格内的比例；qc记录约定和组合顺序。只计算用TorchConvertWarp(...)(...)，保存用result.save(output)。

标准dense输出float32、intent=0，shape=reference.shape+(3,)，affine/qform/sform沿用参考。relative保存位移，absolute保存pull坐标，均为FSL scaled-mm。MMORF转换的valid_fraction=1仅表示输入场覆盖参考网格。

### 转换已有 MMORF 结果

MMORF原场采用参考影像轴mm分量；必须连同产生该配准的原FA和矩阵转换。

```python
from fnit.convertwarp import TorchConvertWarp

native_fa_path = "/data/native/dti_FA.nii.gz"  # 产生FLIRT矩阵的原FA网格
standard_fa_path = "/data/standard/FA.nii.gz"  # MMORF参考网格
mmorf_field_path = "/data/registration/mmorf_warp.nii.gz"  # 参考轴mm场
fa_to_standard_matrix_path = "/data/registration/dti_FA_to_MNI_affine.mat"  # 原FA→参考矩阵
fsl_field_output_path = "/data/registration/mmorf_FSL_warp.nii.gz"  # FSL dense场
converted_mmorf_result = TorchConvertWarp(device="cuda:0").run_mmorf(
    reference=standard_fa_path, source=native_fa_path,
    mmorf_warp=mmorf_field_path, affine=fa_to_standard_matrix_path,
    output=fsl_field_output_path, output_convention="relative",
)
```

若BEDPOSTX mask与native_fa不在相同shape/affine，需先取得二者的配准关系；不能把这个场直接当作另一网格的变换。

### 选择输出约定

- 交给通常的FSL applywarp/invwarp链时保存relative场，并显式确认输入约定。
- 需要绝对采样位置时设output_convention="absolute"；不要仅修改NIfTI intent伪装转换。
- 前后矩阵只能各使用一次；TBSS系数内已包含其线性部分。
- 保存前可查看valid_fraction，但场有值不等于其指向坐标都在输入影像内。

## 3. 命令行调用

```bash
# --ref 指定 MNI 输出网格；--warp1 指定 T1→MNI 非线性场。
# --premat 指定 diffusion→T1 线性矩阵；--out 写出组合场。
fnit convertwarp --ref /data/MNI152_T1_2mm.nii.gz \
  --warp1 /data/T1_to_MNI_warp.nii.gz \
  --premat /data/diff_to_T1.mat \
  --out /data/diff_to_MNI_warp.nii.gz --relout --device cuda:0
```

独立入口 `fnit-convertwarp` 接受相同选项。`--absout` 改写为绝对 pull 坐标；`--rel` / `--abs` 可显式声明输入 dense warp 的约定；`--overwrite` 允许覆盖已有输出。输入系数场的 intent 仍决定其解释方式。

MMORF 模式必须同时提供 `--source` 和 `--premat`，不能提供 `--postmat`、`--rel` 或 `--abs`；标准 `--warp1` 模式不能提供 `--source`。参数互斥或缺失会报错。

| CLI 参数 | Python 参数 | 含义 |
|---|---|---|
| `--ref` | `reference` | 目标网格 |
| `--warp1` | `warp1` | FSL场 |
| `--premat / --postmat` | `premat / postmat` | 前后矩阵；MMORF模式premat对应affine |
| `--mmorf-warp / --source` | `mmorf_warp / source` | MMORF专用输入 |
| `--rel / --abs` | `warp_convention` | dense输入约定 |
| `--relout / --absout` | `output_convention` | 输出约定 |
| `--device / --overwrite` | `device / 写盘策略` | 设备与覆盖 |

## 4. 原软件调用

```bash
# 同一组 reference、warp1 与 premat；包括读取、组合和保存。
convertwarp --ref=/data/MNI152_T1_2mm.nii.gz \
  --warp1=/data/T1_to_MNI_warp.nii.gz \
  --premat=/data/diff_to_T1.mat \
  --out=/data/fsl_diff_to_MNI_warp.nii.gz --relout
```

| 功能 | FNIT | 官方对照方式 |
|---|---|---|
| intent-2006 dense 相对场 | 支持 | `convertwarp --warp1=...` |
| 无明确 intent 的 dense 相对/绝对场 | 支持；可显式指定或 `auto` 判断 | `--rel` / `--abs`，或原程序默认判断 |
| intent-2007 三次样条系数及其嵌入 affine | 支持 | `--warp1=...` 保留完整原始 NIfTI 头信息 |
| premat、postmat 和两者同时组合 | 支持 | `--premat=... --postmat=...` |
| 相对和绝对输出 | 支持 | `--relout` / `--absout` |
| reference 网格、保存、命令行入口 | 支持 | 读取同一 `--ref` 并输出完整场 |
| MMORF 参考影像轴 mm 到 FSL scaled-mm 转换 | FNIT 扩展 | 没有直接对应的原版 `convertwarp` 命令；另核对 MMORF 原重采样路径。 |
| warp2、midmat、shiftmap、Jacobian 计算/约束 | 不支持 | 不把这些原程序选项计入已验证范围。 |
| intent-2008 DCT、intent-2009 二次样条系数 | 不支持；明确报错 | 先在隔离的参考处理中用原程序转成 dense 场。 |

官方程序只用于独立 benchmark，FNIT 生产接口不启动这些命令。[FSL FNIRT 官方说明](https://fsl.fmrib.ox.ac.uk/fsl/docs/registration/fnirt/user_guide.html#convertwarp)解释了原软件的场组合选项。

### 原软件参数对应

| FNIT参数 | 原软件参数 |
|---|---|
| reference / warp1 | --ref / --warp1 |
| premat / postmat | --premat / --postmat |
| warp_convention | --rel / --abs |
| output_convention | --relout / --absout |
| output | --out |

MMORF专用输入没有原版convertwarp直接等价命令；先用FNIT适配器转换坐标。

## 5. 最新精度和运行时间

2026-10-04官方对照为FSL 6.0.7.4，CPU为Intel Xeon Gold 6418H、预算1/8线程。输入为1例真实完整系数及其矩阵分支；每预算warmup一次、三次完整读写配对，表列中位数，绑定v7；后续v28 prepared分支供InvWarp迭代，不把旧计时改标main。

| 完整系数转换 | 1 CPU：FSL / FNIT（s） | 8 CPU：FSL / FNIT（s） | 全场最大差（mm） |
|---|---:|---:|---:|
| 相对输出 | 3.2809 / 2.1275 | 3.3974 / 2.2781 | 9.53674e-6 |
| 绝对输出 | 3.0768 / 2.0702 | 3.2954 / 2.2388 | 1.52588e-5 |

H100同设备TF32、20GB上限，默认系数v5/v7完整API为0.0575/0.0661s，peak allocation均186,100,736B；API包括读写，不含启动/入口导入。分步骤记录：默认路径108个CUDA核、约0.88ms累计核时间；CPU各步骤和官方GPU峰值未记录。[正式完整报告](benchmark_20261004.public.json) · [全部功能/计时边界](CPU_BENCHMARK_20261004.md)。

![公开完整样本的FSL与FNIT场幅值及差异](assets/public_convertwarp_comparison.png)

[可编辑 SVG](assets/public_convertwarp_comparison.svg)、[公共来源 SHA 与完整精度](assets/public_figure_report.public.json)和[生成脚本](plot_public_comparison.py)随图保存。该公开示例的转换在 登录节点 单核完成，用于精度和可视化；CPU评测节点 速度数据见上节。

公开图绑定ds000114完整CC0 T1的独立转换；[图示数据与SHA](assets/public_figure_report.public.json)。

## 6. 最近版本和 benchmark

| 版本 | 更新和验证边界 |
|---|---|
| 2026-10-04 源 v28 | 供 InvWarp 迭代使用的 prepared FP64 系数路径以 fresh-base 两步加法减少临时数组；单次 ConvertWarp、CPU mixed dtype、AD 和 CUDA 原算式保留。CPU评测节点 完整逆场 1/8 核与 v26 文件/QC/停止规则一致，不能代替 ConvertWarp 官方计时。 |
| 2026-10-04 v7 严格边界版本 | 修复 postmat 越出 warp 网格后的 affine 外推及 float32 严格边界；44 项功能回归通过，包含独立 affine 拟合、极小偏移与实际 CUDA。CPU 1/8 两矩阵与极小 postmat 官方精度通过；完整支持模式的耗时、H100门和公开例子见本轮报告。 |
| 2026-10-04 v6 外场拟合版本 | 普通越界使用 float32 组合场的 best-fit affine；CPU 1/8 与 H100 对照完成。v7 进一步补齐极小位移的严格 voxel 边界。 |
| 2026-10-04 v3 CPU 冻结候选 | 复用 CPU 取样优化；跳过恒等矩阵全卷复制；MMORF absolute 省略无用网格；拒绝 MMORF NaN/Inf。CPU评测节点 真实官方对照发现上述外场问题，分区指标完整保留。 |
| 既有 main TBSS/MMORF 版本 | 同一真实 DWI 的两种场转换及重采样对照，指标见上节和机器报告；原计时边界不同。 |

## 7. 参考文献、原软件和资源

源码位置：[原软件 `fugue-2201.3/convertwarp.cc`](../../src/fnit/_vendor_fsl/sources/fugue-2201.3/convertwarp.cc)；[FNIT `convertwarp/core.py`](../../src/fnit/convertwarp/core.py)。固定 tag/commit、Git tree 及每文件 SHA-256 见[来源清单](../../src/fnit/_vendor_fsl/manifest.json)。

代码改写沿用 [FSL 6.0 非商业许可证](../../licenses/FSL-6.0.txt)，完整来源与再分发要求见[第三方声明](../../THIRD_PARTY_NOTICES.md)。

- 参考文献：Andersson, Jenkinson & Smith, *Non-linear registration, aka spatial normalisation*, FMRIB Technical Report TR07JA2 (2007), [原文](https://www.fmrib.ox.ac.uk/datasets/techrep/tr07ja2/tr07ja2.pdf)。 `convertwarp` 没有单独的方法论文。
- 原实现代码库：[FSL `fugue`（含 `convertwarp`）](https://git.fmrib.ox.ac.uk/fsl/fugue)。

| 资源 | 用途 | 官方来源 | 大小 | SHA-256 | 是否允许 FNIT 再分发 |
|---|---|---|---|---|---|
| 本功能无模型权重；用户自备参考影像、掩膜或变换 | 定义目标网格/变换 | 各文件原作者 | 按实际文件 | 按实际文件 | 不随本功能发布用户数据 |

[完整历史说明与调试证据](../../validation/convertwarp/readme_archive_20261005.md) · [返回主页](../../README.md)

<!-- 旧版文档锚点兼容 -->
<a id="python-单被试调用与输入输出"></a> <a id="参数"></a> <a id="直接使用-fnit-pipeline-的两类输出"></a> <a id="命令行调用"></a> <a id="原软件调用与支持范围"></a> <a id="当前-cpu-实现与真实数据对照"></a> <a id="最新独立-gpu-范围与场外修复成本"></a> <a id="公开脑图示例"></a> <a id="历史真实数据记录"></a> <a id="更新与-benchmark-记录"></a> <a id="参考文献与原软件代码库"></a>
