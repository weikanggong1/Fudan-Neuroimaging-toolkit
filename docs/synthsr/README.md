# SynthSR：从单幅扫描合成 1 mm T1w

[返回首页](../../README.md) · [完整旧文档与更早证据](../../validation/synthsr/readme_archive_20261005.md)

| 项目 | 内容 |
|---|---|
| 输入 | 单幅MRI或HU CT |
| 输出 | 1mm合成T1w；NIfTI/MGZ量化图或NPZ浮点图 |
| 对应原软件 | FreeSurfer mri_synthsr |
| Python / CLI | fnit.SynthSR；fnit synthsr |
| CPU / GPU | CPU/CUDA；重采样及保存包含CPU步骤 |

## 1. 功能简介

SynthSR 接受一幅 3D MRI 或 CT，输出标准对比度的 1 mm 等方 MP-RAGE 图像。它结合超分辨率与图像合成，并在合成时填补白质病灶。输入可以是 T1w、T2w、FLAIR 等对比度；原版不要求事先去颅骨、校正偏场或归一化强度。本包将原版的 TensorFlow 网络和推理流程改写为 PyTorch，推理时不调用 FreeSurfer。

默认使用通用v2，另有低场单输入v2和通用v1。模型按同一官方HDF5构建PyTorch网络，保持float32，CUDA默认TF32，不自动使用半精度。

<a id="python-输入与输出"></a>

## 2. Python 调用

```python
from fnit import SynthSR

super_resolution_model = SynthSR(
    weights=None,  # 权重：按 FNIT 配置顺序查找官方 HDF5
    device="cuda:0",  # 设备：第一张可见 CUDA GPU
    lowfield=False,  # 模型：不使用低场专用模型
    v1=False,  # 模型：使用默认 v2，而非 2021 年 v1
    threads=4,  # CPU 线程：用于预处理和后处理
)
super_resolution_result = super_resolution_model(
    image="case_FLAIR.nii.gz",  # 输入：单幅 3D MRI 或 CT
    ct=False,  # 输入类型：按 MRI 强度处理
    disable_flipping=False,  # 推理：保留左右翻转测试增强
    disable_sharpening=False,  # 后处理：保留末端锐化
)
super_resolution_result.image.save(path="case_synthsr.nii.gz")  # 输出路径：1 mm 合成 T1w
print(super_resolution_result.image.data.shape, super_resolution_result.image.affine)
```

### 输入数据格式

- `image`：`.nii`、`.nii.gz`、`.mgz`、`.npz`路径或nibabel SpatialImage。常规输入单幅3D `(X,Y,Z)`；末维不超过10的4D文件只取第一通道，不合成完整时间序列。
- NIfTI/MGZ需要有效affine与体素间距；不要求统一orientation、mask、去颅骨、偏场或强度归一化。MRI强度无固定单位；ct=True要求真实Hounsfield单位，截到0–80。
- NPZ输入需有`vol_data`数组；入口使用单位affine，不包含扫描几何。数组输入应先封装为nibabel SpatialImage；不直接接受ndarray。

### 模型构造

| 参数 | 必需 | 类型 | 默认值 | 含义 |
|---|---|---|---|---|
| `weights` | 否 | `str 或 Path 或 None` | `None` | 官方 checkpoint 文件或目录；省略时按显式配置、FNIT_WEIGHTS 和缓存查找 |
| `device` | 否 | `str` | `'cpu'` | 计算设备，cpu 或 cuda:N；编号遵循 CUDA_VISIBLE_DEVICES |
| `lowfield` | 否 | `bool` | `False` | 选择单输入低场 v2 模型；v1=True 时 v1 优先 |
| `v1` | 否 | `bool` | `False` | 选择 2021 年通用 v1 权重 |
| `threads` | 否 | `int 或 None` | `None` | PyTorch CPU 线程预算；None 保留当前值，CLI 默认另见下节 |

### 单次调用

| 参数 | 必需 | 类型 | 默认值 | 含义 |
|---|---|---|---|---|
| `image` | 是 | `str 或 Path 或 nibabel SpatialImage` | `—` | 输入影像，格式、维数与预处理约束见输入数据格式 |
| `ct` | 否 | `bool` | `False` | CT 按 Hounsfield 单位处理，截到 0–80 |
| `disable_flipping` | 否 | `bool` | `False` | 关闭左右翻转测试增强 |
| `disable_sharpening` | 否 | `bool` | `False` | 关闭输出末端锐化 |

### 输出

```text
case_synthsr.nii.gz  # 1mm网格、uint8、0–255
# 或case_synthsr.mgz：读入存储dtype为float32，值仍是量化0–255
# 或case_synthsr.npz：vol_data为锐化后、末端乘2前的浮点数值
```

返回`SynthSRResult.image`，其`data`是3D uint8数组，`float_data`是末端量化前浮点数组，`affine`为4×4 RAS坐标矩阵。输出为1mm网格、保留输入轴方向和物理空间，shape通常不同于输入；不回到原始体素尺寸。

### 输出字段与数值含义

| 字段 | 格式与用途 |
|---|---|
| `image.data` | `(X1,Y1,Z1)`、uint8，末端乘2并截到0–255后的合成强度 |
| `image.float_data` | 同shape的浮点数组，锐化后、乘2与量化前的数值 |
| `image.affine` | `(4,4)`、float64，1mm输出体素到RAS毫米的坐标映射 |
| `image.header` | NIfTI header对象；保存时dtype由输出规则决定 |

合成强度没有绝对物理单位；它不作为CT的HU输出。浮点NPZ与量化NIfTI具有不同数值定义，比较两者时应先按相同末端转换对齐。NPZ只保存`vol_data`，后续空间叠加需另保留`image.affine`。

`disable_flipping=True`使用单次网络预测；默认对原输入和左右翻转输入各预测一次。`disable_sharpening=True`关闭末端高斯反锐化，二者都会改变输出，benchmark需记录实际开关。

`.save(path)`按扩展写NIfTI、MGZ或NPZ；后者不乘2、不转uint8，须用`numpy.load(path)["vol_data"]`读。调用本身不自动保存，也不重新训练模型。

<a id="单例命令行"></a>

## 3. 命令行调用

```bash
fnit synthsr --i case_FLAIR.nii.gz --o case_synthsr.nii.gz --device cuda:0 --threads 4
```

### fnit synthsr

| CLI 参数 | Python 参数 / 输出 | 含义 |
|---|---|---|
| `--i` / `-i` | `image` | 输入影像路径 |
| `--o` / `-o` | `保存路径` | 输出影像或单例目录 |
| `--device` | `device` | 计算设备，cpu 或 cuda:N；编号遵循 CUDA_VISIBLE_DEVICES |
| `--cpu` | `device=cpu` | 覆盖device强制CPU |
| `--threads` | `threads` | PyTorch CPU 线程预算；None 保留当前值，CLI 默认另见下节 |
| `--ct` | `ct` | CT 按 Hounsfield 单位处理，截到 0–80 |
| `--lowfield` | `lowfield` | 选择单输入低场 v2 模型；v1=True 时 v1 优先 |
| `--v1` | `v1` | 选择 2021 年通用 v1 权重 |
| `--disable_sharpening` | `disable_sharpening` | 关闭输出末端锐化 |
| `--disable_flipping` | `disable_flipping` | 关闭左右翻转测试增强 |
| `--weights` / `--model` | `weights` | 官方 checkpoint 文件或目录；省略时按显式配置、FNIT_WEIGHTS 和缓存查找 |

CLI与Python默认差异：Python threads=None，CLI为1；--cpu覆盖--device。

## 4. 原软件调用

以下命令用于独立原软件参考环境；FNIT生产入口不执行它。

```bash
mri_synthsr --i case_FLAIR.nii.gz --o reference/case_synthsr.nii.gz --threads 4
```

| FNIT参数 / 产物 | 原软件参数 / 产物 |
|---|---|
| image / image.save | --i / --o |
| ct / lowfield / v1 | --ct / --lowfield / --v1 |
| disable_flipping / disable_sharpening | 同名参数 |
| weights / device=cpu / threads | --model / --cpu / --threads |

同为单输入模型，v1优先于lowfield。FNIT可显式选cuda:N；原版自动选可用TensorFlow GPU或--cpu。双输入mri_synthsr_hyperfine使用另一模型，此接口未覆盖。

<a id="原版指令与模型"></a>
<a id="2026-10-04相同-cpu-资源对照"></a>
<a id="默认完全物化-spatialimage-apiv5-冻结"></a>
<a id="2026-10-04cpu网络首差定位"></a>
<a id="2026-09-27既有-gpu-与-cpu-对照"></a>
<a id="该次公开-flair-示意图"></a>

## 5. 最新精度和运行时间

2026-10-04正式对照用两例公开原始T1及MRI/真实HU CT/64mT/EPI格式场景，原版FreeSurfer8.2.0-1、TensorFlow2.13.1。FNIT网络冻结`1d31e7baa`，共享几何`dc2fc052`；nodecw10同8物理核/8线程、默认T1 ABBA。CPUfloat32，完整CLI包含新进程、模型加载、全部计算和保存。

### 端到端 benchmark

| 指标 | FNIT | 原软件 | 差异 |
|---|---|---|---|
| 两病例CLI中位数再取病例中位数 | 44.374 s | 71.840 s | 量化输出差异见下行 |
| 全部NIfTI量化门 | 最差完全一致99.992227% | 独立官方输出 | max≤1、MAE≤1e-4通过 |
| 默认浮点NPZ门 | 未通过 | rtol=1e-5,atol=1e-3 | 具体超差数量/RMSE保留原报告 |

### 分步骤 benchmark

| 阶段 | FNIT | 原软件 |
|---|---|---|
| 单例两次CNN观察 | 13.827 + 14.092 s | 7.634 + 6.154 s |
| 单例预处理观察 | 约5.7 s | 约5.3 s |
| v5物化对象API / 保存 | 29.987 / 1.949 s | 保存参考复用，未重测 |

单例分步数据与两例CLI摘要分开；逐被试运行记录仅在详细报告中。
内部剖析和完整CLI边界不同。实际低场、EPI首通道及NPZ场景本轮慢于官方，不能将完整进程优势推广为网络加速。9个模型/参数与7个域/格式场景通过各自量化或EPI门，但默认浮点NPZ仍失败；预测回放定位差异在两框架FP32网络。H100 GPU旧新输出回归保留原TF32，峰值及实际负载见报告，不另造统一显存数值。[完整测试](../../validation/smri_cpu/strip_sr_20261004/README.md)、[模型参数](../../validation/smri_cpu/strip_sr_20261004/reports/synthsr_cpu_functions.public.json)、[默认浮点失败](../../validation/smri_cpu/strip_sr_20261004/reports/synthsr_default_float_gate.public.json)。

![真实T1官方与FNIT CPU量化合成图](figures/synthsr_cpu_comparison.png)

<a id="最近版本更新与-benchmark"></a>

## 6. 最近版本和 benchmark

| 日期 | commit / version | 变化 | benchmark |
|---|---|---|---|
| 2026-10-04 | 00fedf3544/v5 | 完全物化SpatialImage入口补测 | [数组、header和默认浮点失败保持](../../validation/smri_cpu/strip_sr_20261004/in_memory_20261004/README.md) |
| 2026-10-04 | 1d31e7baa + dc2fc052 | 同资源模型、参数、域及格式对照 | 9+7场景与实际网络阶段 |
| 较早版本 | 既有SynthSR报告SHA | 三份单输入权重及PyTorch推理 | [历史验证](../../validation/synthsr/README.md) |

每条记录保留真实冻结源码、输入与时间边界；逐例、debug/profiling和更早脑图见[完整归档](../../validation/synthsr/readme_archive_20261005.md)。文档整理不重跑MRI，不把执行成功或--help核验作为精度benchmark。

<a id="reference"></a>

## 7. 参考文献、原软件和资源

- 参考文献：Iglesias et al., *SynthSR: A public AI tool to turn heterogeneous clinical brain scans into high-resolution T1-weighted images for 3D morphometry*, Science Advances (2023), [doi:10.1126/sciadv.add3607](https://doi.org/10.1126/sciadv.add3607)。
- 原实现代码库：[FreeSurfer `mri_synthsr`](https://github.com/freesurfer/freesurfer/tree/dev/mri_synthsr)。

模型使用下列官方原始文件，Git/wheel不包含。固定[assets-v1 Release](https://github.com/weikanggong1/Fudan-Neuroimaging-toolkit/releases/tag/assets-v1)及公开asset-manifest与当前weights.py逐项大小/SHA记录一致；本轮未重新下载所有大文件。安装器先Release再原站；完整清单见[资源文件清单](../RESOURCE_MANIFEST.md)。

```bash
fnit-setup-weights --model synthsr --model synthsr-lowfield --model synthsr-v1 --dest /data/fnit-weights
fnit-setup-weights --model synthsr --model synthsr-lowfield --model synthsr-v1 --dest /data/fnit-weights --verify-only
```

离线环境按将使用的模型准备资源即可；默认仅需通用v2，lowfield和v1各使用另一份H5。下载及verify-only通过后显式传入模型文件或目录。`v1=True`会优先于`lowfield=True`，不能把两个模型叠加使用。

Python `threads=None`保留Torch当前值；负数使用CPU核心数。输入MRI至少应有非零强度范围，因为模型需要按全幅最小值和最大值归一化。

| 资源 | 用途 | 官方来源 | 大小 | SHA-256 | 是否允许 FNIT 再分发 |
|---|---|---|---|---|---|
| `synthsr_v20_230130.h5` | 官方推理权重 / 标签数组 | [原站](https://surfer.nmr.mgh.harvard.edu/pub/dist/freesurfer/repo/annex.git/annex/objects/f08/bc9/SHA256E-s106163752--a472f776e7b33b5ea6e10c801f55fee488f1477a208b3e6998dc1aec1d9c5f8b.h5/SHA256E-s106163752--a472f776e7b33b5ea6e10c801f55fee488f1477a208b3e6998dc1aec1d9c5f8b.h5) | 106,163,752 B | `a472f776e7b33b5ea6e10c801f55fee488f1477a208b3e6998dc1aec1d9c5f8b` | 允许；FreeSurfer许可，保留条款与归属 |
| `synthsr_lowfield_v20_230130.h5` | 官方推理权重 / 标签数组 | [原站](https://surfer.nmr.mgh.harvard.edu/pub/dist/freesurfer/repo/annex.git/annex/objects/de0/799/SHA256E-s106163752--a7c5ea91c94fe31f3c716252caae0d181629201bd884dc59af88ddfd75ed4b84.h5/SHA256E-s106163752--a7c5ea91c94fe31f3c716252caae0d181629201bd884dc59af88ddfd75ed4b84.h5) | 106,163,752 B | `a7c5ea91c94fe31f3c716252caae0d181629201bd884dc59af88ddfd75ed4b84` | 允许；FreeSurfer许可，保留条款与归属 |
| `synthsr_v10_210712.h5` | 官方推理权重 / 标签数组 | [原站](https://raw.githubusercontent.com/freesurfer/freesurfer/dev/mri_synthsr/synthsr_v10_210712.h5) | 53,075,984 B | `2fd59e96196388360eba95254fb6dfc9eb9eb8638018b590575e47e0a387f255` | 允许；FreeSurfer许可，保留条款与归属 |

本页列出的模型/数组共3个，265,403,488 B。原始文件许可及归属见[统一资源规则](../WEIGHTS.md#权重许可与归属)。模型推理从本地加载已准备资源。
