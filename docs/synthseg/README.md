# 独立 33 类 SynthSeg

[返回首页](../../README.md) · [完整旧文档与更早证据](../../validation/synthseg/readme_archive_20261005.md)

| 项目 | 内容 |
|---|---|
| 输入 | 单幅3D T1w，原生空间 |
| 输出 | 33类标签与32个前景结构软体积 |
| 对应原软件 | FreeSurfer mri_synthseg（普通2.0） |
| Python / CLI | fnit.SynthSeg；fnit synthseg |
| CPU / GPU | CPU/CUDA；预后处理及保存包含CPU步骤 |

## 1. 功能简介

`SynthSeg` 使用 FreeSurfer 8.2 的非 robust、非 parcellated SynthSeg 2.0 模型，从单幅 T1 生成 33 类结构标签和各结构软体积。它与 WMH-SynthSeg 是不同模型；本入口不输出 WMH 标签，也不生成皮层分区。推理使用 PyTorch，不需要安装 FreeSurfer、FSL 或 TensorFlow。

默认GPU卷积TF32；`cudnn_tf32=False`可用于已验证的recon-all局部例外。模型保持float32，不自动启用FP16/BF16。成熟子函数已修复CPU大卷积内存、endpoint网格和保存色表/几何的问题，真实对照见第5节。

<a id="python-api"></a>

## 2. Python 调用

```python
from fnit import SynthSeg

segmentation_model = SynthSeg(
    weights=None,  # 权重：按 FNIT 配置顺序查找四个官方模型文件
    device="cuda:0",  # 设备：第一张可见 CUDA GPU
    threads=4,  # CPU 线程：用于预处理和后处理
    cudnn_tf32=True,  # 独立入口默认开启 GPU 卷积 TF32；recon-all 显式使用 False
)
segmentation_result = segmentation_model(
    image="sub-01_T1w.nii.gz",  # 输入：单幅 3D T1w
    keep_geometry=False,  # 输出网格：保留预处理后的 RAS、约 1 mm 网格
    color_lut=None,  # 色表：不附加外部 FreeSurfer LUT
)
segmentation_result.segmentation.save(path="sub-01_synthseg.nii.gz")  # 输出路径：33 类硬分割图
segmentation_result.write_volumes_csv(
    source="sub-01_T1w.nii.gz",  # CSV 标识：原始输入文件名
    path="sub-01_synthseg.vol.csv",  # 输出：软体积 CSV 路径
)
print(segmentation_result.total_intracranial_mm3, segmentation_result.volumes_mm3)
print(segmentation_result.precision)  # 每次真实前向的 TF32、张量 dtype 与 autocast 设置
```

### 输入数据格式

- `image`：一幅3D `(X,Y,Z)` T1w，`.nii`、`.nii.gz`、`.mgz`路径或nibabel SpatialImage；不接受时间序列。强度无固定单位，不需预先去颅骨。
- 输入需有效affine和体素间距；内部重排RAS、重采样约1mm。`keep_geometry=True`输出回原始空间，False输出模型处理网格。
- `weights`：主H5与三份标签/名称/拓扑NPY必须同目录。`color_lut`是用户提供的`编号 名称 R G B T`文本，编号唯一、颜色合法；默认不读取外部LUT。

### 模型构造

| 参数 | 必需 | 类型 | 默认值 | 含义 |
|---|---|---|---|---|
| `weights` | 否 | `str 或 Path 或 None` | `None` | 官方 checkpoint 文件或目录；省略时按显式配置、FNIT_WEIGHTS 和缓存查找 |
| `device` | 否 | `str` | `'cpu'` | 计算设备，cpu 或 cuda:N；编号遵循 CUDA_VISIBLE_DEVICES |
| `threads` | 否 | `int 或 None` | `None` | PyTorch CPU 线程预算；None 保留当前值，CLI 默认另见下节 |
| `cudnn_tf32` | 否 | `bool 或 None` | `True` | 该模型前向的 cuDNN TF32：True 开启、False 关闭、None 沿用；恢复调用方设置 |

### 单次调用

| 参数 | 必需 | 类型 | 默认值 | 含义 |
|---|---|---|---|---|
| `image` | 是 | `str 或 Path 或 nibabel SpatialImage` | `—` | 输入影像，格式、维数与预处理约束见输入数据格式 |
| `keep_geometry` | 否 | `bool` | `False` | 标签以最近邻重采样回输入 shape 和 affine |
| `color_lut` | 否 | `str 或 Path 或 None` | `None` | 用户提供的 FreeSurfer 编号/名称/RGBA 文本色表；None 不附加 |

### 输出

```text
sub-01_synthseg.nii.gz  # int32硬分割
sub-01_synthseg.vol.csv # 32个前景结构及颅内容量，mm³
```

返回`SynthSegResult`：`segmentation`为nibabel兼容3D影像，qform/sform code为0/2；默认RAS、约1mm网格，keep_geometry=True时shape/affine与输入相同。33类含背景0，编号由官方标签NPY定义。`volumes_mm3`映射前景编号到软体积，`total_intracranial_mm3`为其和；不是硬标签体素计数。`label_names`提供名称，`near_tie_voxels`记录并列规则变化体素，`precision`记录实际前向精度。

### 返回字段的读取

| 字段 | 类型与用途 |
|---|---|
| `segmentation` | 3D影像；整数编码保留左右侧，不按33类顺序重新编号 |
| `volumes_mm3` | `dict[int, float]`，32个前景标签的软体积，mm³ |
| `total_intracranial_mm3` | `float`，上述前景软体积之和，mm³ |
| `label_names` | `dict[int, str]`，官方标签编号到名称的映射 |
| `near_tie_voxels` | `int`，接近并列后验的体素数，不是分割误差计数 |
| `precision` | `dict`，本次前向的设备、dtype、TF32与autocast记录 |

输入orientation与输出orientation不同并不表示配准到MNI。这里仅做网格重排和重采样，输出仍在同一扫描的物理空间；与其他影像叠加时使用affine。

影像用`.save()`；`write_volumes_csv(source, path)`两参均必需，分别为输入标识路径和CSV输出路径。单次调用不自动保存。

<a id="原版命令与参数对应"></a>
<a id="命令行"></a>

## 3. 命令行调用

```bash
fnit synthseg --i sub-01_T1w.nii.gz --o sub-01_synthseg.nii.gz \
  --csv-vols sub-01_synthseg.vol.csv --device cuda:0 --threads 4
```

### fnit synthseg

| CLI 参数 | Python 参数 / 输出 | 含义 |
|---|---|---|
| `--i` / `-i` | `image / t1（--parc）` | 输入影像路径 |
| `--o` / `-o` | `segmentation.save / combined.save（--parc）` | 硬标签影像保存路径 |
| `--csv-vols` / `--csv_vols` | `write_volumes_csv` | 写出软体积 CSV，mm³ |
| `--weights` | `weights` | 官方 checkpoint 文件或目录；省略时按显式配置、FNIT_WEIGHTS 和缓存查找 |
| `--device` | `device` | 计算设备，cpu 或 cuda:N；编号遵循 CUDA_VISIBLE_DEVICES |
| `--threads` | `SynthSeg.threads / CLI进程线程（--parc）` | PyTorch CPU 线程预算；None 保留当前值，CLI 默认另见下节 |
| `--keep-geometry` | `keep_geometry` | 标签以最近邻重采样回输入 shape 和 affine |
| `--color-lut` | `color_lut` | 用户提供的 FreeSurfer 编号/名称/RGBA 文本色表；None 不附加 |
| `--parc` | `选择 SynthSegPlus` | 选择普通 SynthSeg 2.0皮层分区 |
| `--fast` | `fast` | 普通 SynthSeg --parc 的快速路径；不代表 robust 模型 |
| `--parc-weights` | `parc_weights` | 官方皮层分区 H5 文件或目录 |
| `--parc-out` | `cortical_parcellation.save` | 另存仅皮层标签图 |

CLI与Python默认差异：Python threads=None，CLI为4；cudnn_tf32只在Python可调，CLI默认True。

## 4. 原软件调用

以下命令用于独立原软件参考环境；FNIT生产入口不执行它。

```bash
mri_synthseg --i sub-01_T1w.nii.gz --o reference/sub-01_synthseg.nii.gz \
  --vol reference/sub-01_synthseg.vol.csv --threads 4 --noaddctab
```

| FNIT参数 / 产物 | 原软件参数 / 产物 |
|---|---|
| image / segmentation.save | --i / --o |
| write_volumes_csv | --vol |
| keep_geometry | --keepgeom |
| weights / device=cpu / threads | --model / --cpu / --threads |
| color_lut | --addctab / --noaddctab（FNIT仅显式提供LUT时附加） |

本页为普通SynthSeg2.0非robust、非parcellated路径。目录批处理、QC、posterior、CT、Photo-SynthSeg及其他模型未覆盖；皮层分区用[下一入口](../synthseg_plus/README.md)。CLI的--parc相关选项仅为共享parser，普通调用不要启用。

<a id="下载模型"></a>
<a id="recon-all-集成发现的精度设置覆盖"></a>
<a id="与-freesurfer-82-的既有独立对照"></a>
<a id="2026-10-04-cpu-优化与回归"></a>

## 5. 最新精度和运行时间

2026-10-06 最新[C24 CPU列缓冲复用](CPU_COLUMNS_C24.md)在既有C72优化上完成同一公开T1、同八核的CPU ABBA与GPU AB：**API中位数85.499→80.621秒，缩短5.71%**。旧新完整分割、软体积CSV、全部header和gzip SHA相同；GPU实际设备、FP32/TF32作用域、输出和allocated/reserved相同，本人进程树采样最大15.177GB。新优化只覆盖已核验的权重和`[1,24,192,224,256]`层输入、CPU8及既有推理状态；其余模式沿用成熟路径，CUDA不导入该CPU模块。

冷进程中位数98.352→83.311秒，首臂有20.286秒未细分检查，其他臂约0.125秒，因此该15.29%观察不能当纯计算加速。CPU/GPU所有原始时钟、编译次数、资源和保存输出评分见[最新完整报告](../../validation/smri_cpu/seg_columns_c24_integration_20261006/README.md)。CPU对官方保持既有1体素差异、前景最低Dice0.999998569、CSV最大差0.8mm³；官方同例冷CLI55.046秒仍是独立计时，CPU速度目标未过。已有Conda编译与实际sdist三源码成员验证通过，完整wheel和全新Conda安装未测。

前一版[C72完整对照](../../validation/smri_cpu/seg_columns_integration_20261006/README.md)冷worker107.938→90.119秒（16.51%），API105.144→87.394秒；该组使用各自冻结版本与计时，不能与本组相乘推算累计加速。

![复用的公开T1分割图：新版本与该图的三个来源分割SHA相同，行序与原报告一致](../../validation/smri_cpu/seg_columns_integration_20261006/case02_current_cpu_labels.png)

此前2026-10-05 的 CPU decoder 拼接仅减少临时缓冲，卷积与后处理保持原数值。公开原始 T1 的 nodecw7 八核对照中，旧/新完整 worker 为 **115.886/112.952 s**，最大 RSS 为 **15.34/13.51 GB**；标签、体积 CSV 和几何旧新相同。与同节点官方仍有 **1 个不同体素**、最小 Dice **0.99999857**、CSV 最大差 **0.800 mm³**，均为已有误差。官方正常冷 CLI 为 **55.046 s**，首次异常的 373.087 s 单独保留；worker 还包含资源校验和报告写出，不把两种时钟混作精确加速比，CPU 速度目标仍未通过。

同一真实 decoder 中间输入的拼接操作中位数 **0.9723→0.5294 s**，逐位一致；这是局部时间。H100 默认 TF32 与显式 `cudnn_tf32=False` 两组完整旧新输出、CSV、几何、allocated/reserved 相同。默认 reserved 为 **14.615 GB**，False 为 **9.326 GB**；共享 GPU 的背景负载不支持稳定速度或进程树物理峰值结论。显式 False 单例标签与官方完全相同，CSV 仍有最大 0.20 mm³ 尾差，不能推广为所有输入等价。[最新完整协议、逐区指标和运行边界](../../validation/smri_cpu/seg_memory_20261005/README.md)。

2026-10-06 在同一公开 T1、八核预算上完成一次当前源码的分步骤观察：完整 API **106.377 s**，两次 CNN **81.716 s**，两次33通道平滑 **13.136 s**，预处理 **4.362 s**，后处理 **4.710 s**，并列判断 **1.074 s**，软体积 **0.342 s**。CNN 内卷积合计76.399 s，其中最后 decoder 的第一层占30.586 s；显式 slab 结果复制仅0.927 s。输出每个体素、完整 header、体积 CSV 和压缩文件 SHA 与已验收候选相同，保留对官方既有1个不同体素。带观察器的时间仅用于定位热点，未修改 CPU/GPU 算法，不替代对应版本的正式墙钟。该剖析据以定位高分辨率卷积和分组平滑；增加错误标签的 oneDNN 路线不采用。[完整分步骤报告](../../validation/smri_cpu/seg_cpu_profile_20261006/README.md)及[独立来源与计时复核](../../validation/smri_cpu/seg_cpu_profile_20261006/ROOT_REVIEW.json)。

后续对同一真实末端概率张量完成一次旧/新 ABBA 平滑回放：按原 slab 和 kernel 将11个通道并为独立 batch，三次实际比较的363,331,584个 FP32值全部逐位相同。峰值RSS最大值10.295→5.592 GB，降低45.68%；操作中位数6.963→6.978 s，没有速度收益。因此只保留独立候选，不替换默认实现，也不把局部内存下降记作完整T1的峰值下降。本轮只恢复一次已保存的 decoder 末端，未重复完整CNN或官方软件。[真实候选报告](../../validation/smri_cpu/seg_cpu_blur_trial_20261006/README.md)及[独立复核](../../validation/smri_cpu/seg_cpu_blur_trial_20261006/ROOT_REVIEW.json)。

高分辨率卷积的单一64MiB slab候选未通过目标Torch2.5.1合同：使用真实层权重的depth1/2输出逐位相同，depth7的37,128项中33,144项不同，最大绝对差4.292e−6。按预设门停止，没有运行真实MRI单层、ABBA或完整网络；也没有官方精度方向或速度结论。生产继续使用原分块，GPU未改。下一步研究保持原矩阵尺寸与偏置运算的工作区复用。[拒绝报告与静态方案](../../validation/smri_cpu/seg_cpu_conv_slab_trial_20261006/README.md)及[独立来源复核](../../validation/smri_cpu/seg_cpu_conv_slab_trial_20261006/ROOT_REVIEW.json)。

![本次公开真实T1的官方和FNIT CPU标签](../../validation/smri_cpu/seg_memory_20261005/case02_cpu_labels.png)

前版2026-10-04测试使用公开原始T1两例，nodecw10同8物理核/8线程。原版FreeSurfer8.2.0-1，FNIT冻结源码和每次文件SHA见[正式矩阵](../../validation/smri_cpu_20261004/t2_seg/README.md)；CPUfloat32、原后端分块，完整命令含权重加载、预后处理、标签/CSV保存。

### 端到端 benchmark

| 指标 | FNIT | 原软件 | 差异 |
|---|---|---|---|
| 所测单例重复CLI范围 | 133.71–171.51 s | 56.34–127.19 s | 两例各差1个标签体素；不是两例时间分布 |
| 两例CSV最大绝对差的最大值 | 0.80 mm³ | 独立官方输出 | 非逐值一致 |
| 所测进程RSS | 约15 GB | 本组未统一摘要 | 同输入FNIT旧版约94 GB |

### 分步骤 benchmark

| 阶段 | FNIT | 原软件 |
|---|---|---|
| 预处理网络输入 | 3幅原始T1数组SHA与官方同 | 独立官方输入 |
| 双向网络/后处理/保存 | 分项记录见原报告 | 未生成同边界统一阶段表 |

共享负载和缓存使两次时间波动，不据此宣称全面CPU提速。保留原后端分块对两例硬标签与修正baseline逐值同，CSV差0.04/0.30mm³通过既定容差；早期oneDNN切片使第二例2个标签改变，已撤回。保存几何GPU回归验证原T1shape/affine/pixdim和色表；本轮未重跑旧三例独立GPU精度矩阵，不能把2026-09-27的19,769MiB allocated当作最新全模式20GB验收。

![既有公开T1的官方与FNIT SynthSeg；冻结版本见原报告](figures/synthseg_comparison.png)

## 6. 最近版本和 benchmark

| 日期 | commit / version | 变化 | benchmark |
|---|---|---|---|
| 2026-10-06 | C24冻结源码 `2e503082` | 同SGEMM provider与原矩阵顺序复用down0 conv1列缓冲 | 完整CPU API缩短5.71%；六完整臂输出及GPU精度/内存保持，见[报告](../../validation/smri_cpu/seg_columns_c24_integration_20261006/README.md) |
| 2026-10-06 | 源码 `70ad6537`；调度 `41ede608`；文档 `f68ce251` | 普通33类受测CPU层复用原矩阵尺寸的列缓冲 | 完整ABBA冷worker缩短16.51%，CPU/GPU旧新分割、CSV和几何相同；官方仍1体素差异且CPU更慢，见[报告](../../validation/smri_cpu/seg_columns_integration_20261006/README.md) |
| 2026-10-06 | 独立报告 `84fbd765`；生产未变 | 普通33类CPU概率平滑的batch11候选 | 一份真实posterior的三次逐位门通过，RSS最大值降低45.68%，速度未改善，未接入默认路径，见[报告](../../validation/smri_cpu/seg_cpu_blur_trial_20261006/README.md) |
| 2026-10-06 | 独立报告 `10f3f913`；生产未变 | 单层64MiB slab候选的目标合同 | 首个跨slab合同出现FP32尾差，停止真实层；没有速度或官方精度结论，见[报告](../../validation/smri_cpu/seg_cpu_conv_slab_trial_20261006/README.md) |
| 2026-10-06 | `46eead65` 对应生产文件；报告 `0b6c9dbf` | 一次无 Module hooks 的当前33类 CPU 分步观察；生产未变 | 完整输出、CSV、header 与文件 SHA 相同；卷积与平滑为主要热点，官方速度门仍未通过，见[实测报告](../../validation/smri_cpu/seg_cpu_profile_20261006/README.md) |
| 2026-10-05 | `585bf181` 源码；`b1d46705` 完整报告 | CPU 单缓冲 nearest/skip 拼接，GPU 保留原路径 | 同节点 CPU 与 H100 完整旧新输出相同；官方 CPU 速度目标未通过，见[完整记录](../../validation/smri_cpu/seg_memory_20261005/README.md) |
| 2026-10-04 | 报告冻结v2及最终保存修复 | 原后端分块、endpoint及色表/几何 | [真实CPU与GPU回归](../../validation/smri_cpu_20261004/t2_seg/README.md) |
| 2026-10-01 | recon-all集成修复 | 前向作用域应用cuDNN精度，恢复调用方设置 | [同输入FP32报告](../recon_all/SYNTHSEG_PRECISION.md) |
| 2026-09-30 | 报告冻结CPU后端 | 避免oneDNN完整体积崩溃 | [阶段对照](../../validation/recon_all/python_gpu_port/synthseg_cpu_backend_20260930.json) |
| 2026-09-27 | 源码树39fa204a | 独立三例默认TF32对照 | [历史报告](../../validation/synthseg/report.public.json) |

每条记录保留真实冻结源码、输入与时间边界；逐例、debug/profiling和更早脑图见[完整归档](../../validation/synthseg/readme_archive_20261005.md)。文档整理不重跑MRI，不把执行成功或--help核验作为精度benchmark。

<a id="reference"></a>

## 7. 参考文献、原软件和资源

- 参考文献：Billot et al., *SynthSeg: Segmentation of brain MRI scans of any contrast and resolution without retraining*, Medical Image Analysis (2023), [doi:10.1016/j.media.2023.102789](https://doi.org/10.1016/j.media.2023.102789)。
- 原实现代码库：[FreeSurfer `mri_synthseg`](https://github.com/freesurfer/freesurfer/tree/dev/mri_synthseg)。

模型使用下列官方原始文件，Git/wheel不包含。固定[assets-v1 Release](https://github.com/weikanggong1/Fudan-Neuroimaging-toolkit/releases/tag/assets-v1)及公开asset-manifest与当前weights.py逐项大小/SHA记录一致；本轮未重新下载所有大文件。安装器先Release再原站；完整清单见[资源文件清单](../RESOURCE_MANIFEST.md)。

```bash
fnit-setup-weights --model synthseg --dest /data/fnit-weights
fnit-setup-weights --model synthseg --dest /data/fnit-weights --verify-only
```

离线运行时先在联网环境下载并完成verify-only，再把整个权重目录复制到计算环境；使用`weights="/data/fnit-weights"`显式指定。主H5与三份NPY需保留原文件名和同目录关系。构造模型只查本地资源，缺失时会提示安装命令。

`threads`为None时保留当前Torch线程数；正数设定预算，负数使用CPU核心数。GPU卷积设置由`cudnn_tf32`控制，并在前向完成后恢复调用方策略。

| 资源 | 用途 | 官方来源 | 大小 | SHA-256 | 是否允许 FNIT 再分发 |
|---|---|---|---|---|---|
| `synthseg_2.0.h5` | 官方推理权重 / 标签数组 | [原站](https://surfer.nmr.mgh.harvard.edu/pub/dist/freesurfer/repo/annex.git/annex/objects/bee/241/SHA256E-s53079152--f190bfd742f450ef3ca2c9df9ed4d2e0232b3a74471da5e51b7770bacdf80c3e.0.h5/SHA256E-s53079152--f190bfd742f450ef3ca2c9df9ed4d2e0232b3a74471da5e51b7770bacdf80c3e.0.h5) | 53,079,152 B | `f190bfd742f450ef3ca2c9df9ed4d2e0232b3a74471da5e51b7770bacdf80c3e` | 允许；FreeSurfer许可，保留条款与归属 |
| `synthseg_segmentation_labels_2.0.npy` | 官方推理权重 / 标签数组 | [原站](https://raw.githubusercontent.com/freesurfer/freesurfer/v8.2.0/mri_synthseg/synthseg_segmentation_labels_2.0.npy) | 348 B | `5ef25ec33fe917ac99f30b8f2185b2d77121136ee411b9c4970c0b59be615ed8` | 允许；FreeSurfer许可，保留条款与归属 |
| `synthseg_segmentation_names_2.0.npy` | 官方推理权重 / 标签数组 | [原站](https://raw.githubusercontent.com/freesurfer/freesurfer/v8.2.0/mri_synthseg/synthseg_segmentation_names_2.0.npy) | 7,168 B | `234eb6d514e10d6ebd748a8b30a1d12d9426fd874c607e37852406fae8f290fc` | 允许；FreeSurfer许可，保留条款与归属 |
| `synthseg_topological_classes_2.0.npy` | 官方推理权重 / 标签数组 | [原站](https://raw.githubusercontent.com/freesurfer/freesurfer/v8.2.0/mri_synthseg/synthseg_topological_classes_2.0.npy) | 348 B | `650b4b96834485c1e6d7421de4af74da80d861e6b2a39ef1164389bde3a5e14a` | 允许；FreeSurfer许可，保留条款与归属 |

本页列出的模型/数组共4个，53,087,016 B。原始文件许可及归属见[统一资源规则](../WEIGHTS.md#权重许可与归属)。模型推理从本地加载已准备资源。
