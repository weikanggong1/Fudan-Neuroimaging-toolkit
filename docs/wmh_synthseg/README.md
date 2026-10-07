# WMH-SynthSeg：脑结构与白质高信号分割

[返回首页](../../README.md) · [完整旧文档与更早证据](../../validation/wmh_synthseg/readme_archive_20261005.md)

| 项目 | 内容 |
|---|---|
| 输入 | 单幅3D T1w或FLAIR |
| 输出 | 脑结构/WMH标签、可选WMH概率和软体积 |
| 对应原软件 | FreeSurfer mri_WMHsynthseg |
| Python / CLI | fnit.WMHSynthSeg；fnit wmh-synthseg |
| CPU / GPU | CPU或CUDA；20GB配置显式crop=True，full未达到20GB |

## 1. 功能简介

WMH-SynthSeg 同时分割脑结构和白质高信号（WMH，FreeSurfer 标签 **77**），支持 T1w、FLAIR 等对比度。**FreeSurfer 原版已使用 PyTorch**。本包基于已验证的源码与官方 `WMH-SynthSeg_v10_231110.pth`，提供独立安装、单被试 Python 和单被试命令行调用；推理不调用 FreeSurfer 程序。

输入重排RAS并重采样1mm，crop=True先定位再裁脑周有限视野；裁剪可能移除视野边缘。输出使用处理后网格，通常不是输入shape。默认crop=False保留完整视野；当前GPU完整视野峰值仍超过20GB。

<a id="原版源码流程与输出几何"></a>

## 2. Python 调用

```python
from fnit import WMHSynthSeg

wmh_segmentation_model = WMHSynthSeg(
    weights=None,  # 权重：按 FNIT 配置顺序查找官方 checkpoint
    device="cuda:0",  # 设备：第一张可见 CUDA GPU
    threads=4,  # CPU 线程：用于预处理和后处理
)
wmh_segmentation_result = wmh_segmentation_model(
    image="case_FLAIR.nii.gz",  # 输入：单幅 3D FLAIR/T1w
    crop=True,  # 裁剪：先定位脑区，再在有限视野内正式推理
    save_lesion_probabilities=True,  # 输出：同时返回标签 77 的概率图
)
wmh_segmentation_result.segmentation.save(path="case_seg.nii.gz")  # 输出路径：脑结构与 WMH 硬分割
wmh_segmentation_result.lesion_probability.save(path="case_seg.lesion_probs.nii.gz")  # 输出路径：WMH 概率图
print(wmh_segmentation_result.volumes_mm3)
```

### 输入数据格式

- `image`：单幅3D `(X,Y,Z)` `.nii`、`.nii.gz`、`.mgz`路径或nibabel SpatialImage；T1w、FLAIR等对比度，强度无固定单位。4D均值/时间序列不在接口内。
- 有效affine、体素间距和orientation；无需先生成脑mask。内部按最大值归一化、RAS重排和1mm等方重采样。
- `crop=True`正式CNN视野不超过192×224×192，输出affine记录裁剪后物理坐标；不能把处理后概率图直接叠加原数组。

### 模型构造

| 参数 | 必需 | 类型 | 默认值 | 含义 |
|---|---|---|---|---|
| `weights` | 否 | `str 或 Path 或 None` | `None` | 官方 checkpoint 文件或目录；省略时按显式配置、FNIT_WEIGHTS 和缓存查找 |
| `device` | 否 | `str` | `'cpu'` | 计算设备，cpu 或 cuda:N；编号遵循 CUDA_VISIBLE_DEVICES |
| `threads` | 否 | `int 或 None` | `None` | PyTorch CPU 线程预算；None 保留当前值，CLI 默认另见下节 |

### 单次调用

| 参数 | 必需 | 类型 | 默认值 | 含义 |
|---|---|---|---|---|
| `image` | 是 | `str 或 Path 或 nibabel SpatialImage` | `—` | 输入影像，格式、维数与预处理约束见输入数据格式 |
| `crop` | 否 | `bool` | `False` | 两遍定位并限制正式推理视野；20 GB GPU 配置显式设 True |
| `save_lesion_probabilities` | 否 | `bool` | `False` | 另返回/保存 WMH 标签77的后验概率图 |

### 输出

```text
case_seg.nii.gz              # float32存储的33类整数标签
case_seg.lesion_probs.nii.gz # 请求时WMH后验，float32
case_volumes.csv             # CLI可选软体积表，mm³
```

返回`WMHResult`：`segmentation`与可选`lesion_probability`为nibabel兼容3D图，shape/affine为RAS/1mm处理网格，qform/sform code0/2，新的输出header不继承扫描XML扩展。标签77为WMH，其他编号是官方33类FreeSurfer编码；概率值0–1。

### 返回字段与保存行为

| 字段 | 类型 / 网格 | 含义 |
|---|---|---|
| `segmentation` | 3D float32影像，值为整数编码 | 33类硬标签；77为WMH |
| `lesion_probability` | 同shape/affine的float32影像或None | 标签77的后验概率，范围0–1；按参数请求 |
| `volumes_mm3` | `dict[int,float]` | 33个标签的后验求和，处理网格为1mm，单位mm³ |

以下仅从已经返回的对象读取结果，不增加MRI计算：

```python
import numpy as np

segmentation_image = wmh_segmentation_result.segmentation  # 已返回的处理网格影像
segmentation_data = np.asanyarray(segmentation_image.dataobj)  # 3D float32存储标签
wmh_mask = segmentation_data == 77  # 同shape的布尔WMH掩膜
wmh_soft_volume_mm3 = wmh_segmentation_result.volumes_mm3[77]  # 后验积分软体积
wmh_hard_volume_mm3 = float(wmh_mask.sum())  # 1mm网格上的硬标签体积，mm³
print(segmentation_image.affine, wmh_soft_volume_mm3, wmh_hard_volume_mm3)
```

软硬体积分别来自概率和赢家标签，不要求逐值相等。若需要在原T1网格使用mask，应根据两幅影像affine以最近邻重采样标签；概率图使用连续插值并保留概率含义。

`volumes_mm3`含背景0在内的软体积字典；非背景和为Intracranial-volume，不能用硬体素计数替代。Python只返回内存对象，需显式save；CSV由CLI生成。与原版一致，CSV第一列名Input-file但内容为输出分割路径。

<a id="本包单例-python-与命令行"></a>

## 3. 命令行调用

```bash
fnit wmh-synthseg --i case_FLAIR.nii.gz --o case_seg.nii.gz \
  --device cuda:0 --threads 4 --crop --save_lesion_probabilities --csv_vols case_volumes.csv
```

### fnit wmh-synthseg

| CLI 参数 | Python 参数 / 输出 | 含义 |
|---|---|---|
| `--i` / `-i` | `image` | 输入影像路径 |
| `--o` / `-o` | `segmentation.save` | 脑结构/WMH硬标签影像保存路径 |
| `--csv_vols` / `--csv-vols` | `CLI CSV写出` | 写出软体积 CSV，mm³ |
| `--device` | `device` | 计算设备，cpu 或 cuda:N；编号遵循 CUDA_VISIBLE_DEVICES |
| `--threads` | `threads` | PyTorch CPU 线程预算；None 保留当前值，CLI 默认另见下节 |
| `--crop` | `crop` | 两遍定位并限制正式推理视野；20 GB GPU 配置显式设 True |
| `--save_lesion_probabilities` / `--save-lesion-probabilities` | `save_lesion_probabilities` | 另返回/保存 WMH 标签77的后验概率图 |
| `--weights` | `weights` | 官方 checkpoint 文件或目录；省略时按显式配置、FNIT_WEIGHTS 和缓存查找 |

CLI与Python默认差异：Python threads=None，CLI为1；两者crop默认False，20GB示例显式True。

## 4. 原软件调用

以下命令用于独立原软件参考环境；FNIT生产入口不执行它。

```bash
mri_WMHsynthseg --i case_FLAIR.nii.gz --o reference/case_seg.nii.gz \
  --device cpu --threads 4 --crop --save_lesion_probabilities --csv_vols reference/case_volumes.csv
```

| FNIT参数 / 产物 | 原软件参数 / 产物 |
|---|---|
| image / segmentation.save | --i / --o |
| crop / save_lesion_probabilities | 同名开关 |
| volumes_mm3 | --csv_vols |
| device / threads | --device / --threads |
| weights | 官方固定WMH-SynthSeg_v10_231110.pth |

对应原版单文件3D路径；原版目录批量与4D均值分支不作为本接口覆盖。推理沿用官方PyTorch模型，未用33类SynthSeg2.0的H5替代WMH checkpoint。

<a id="2026-10-04生命周期修复与完整-gpu-裁剪回归"></a>
<a id="2026-10-04-cpu-对照"></a>
<a id="2026-09-27-历史验证"></a>
<a id="原版与当前实现示意图"></a>

## 5. 最新精度和运行时间

最新2026-10-04生命周期修复基于f1cbdab1，最终冻结source_v10，model/pipeline源码SHA见[完整报告](../../validation/smri_cpu/synth_fixes_20261004/README.md)。输入为公开CC0 ds003138 v1.0.1一例原始T1，原224×288×288，crop输出179×224×178。原版FreeSurfer8.2.0-1；CPU评测节点 Xeon Gold6418H同8物理核/8线程、CPUfloat32。H100 GPU为float32、原TF32策略、无autocast。

### 端到端 benchmark

| 指标 | FNIT | 原软件 / 旧实现 | 差异 |
|---|---|---|---|
| CPU crop完整进程两次 | 77.134 / 83.876 s | 172.064 / 107.942 s | 标签、WMH概率、33CSV数值、完整header/affine逐值同 |
| CPU采样RSS | 约16.778 GB | 约29.992 GB | 不同于GPUallocator峰值 |
| GPU crop allocated / reserved | 18.374 / 18.438 GB | 旧GPU32.634 /45.267 GB（不限额） | 三输出文件SHA同，57卷积kernel同 |
| GPU full allocated / reserved | 37.997 /43.136 GB | 旧GPU同范围数值参考 | full未通过20GB预算 |

### 分步骤 benchmark

| 阶段 | FNIT | 原软件 / 旧实现 |
|---|---|---|
| GPU crop API旧两次 / 新两次 | 新5.199 /5.375 s | 旧FNIT4.842 /5.364 s |
| GPU crop完整worker旧/新 | 新13.21 /13.18 s | 旧FNIT12.85 /13.32 s |
| 构造 / API / 保存 | 分项实测见原JSON | 没有同边界官方GPU阶段表 |

CPU时钟属于冻结v5算术路径；最终v10修改GPU缓存/模式调度，没有重跑官方CPU。GPU worker含导入、核验和报告，不是CLI专属时钟；旧参考超20GB，不能称同预算速度比较。默认full优化原型出现1个硬标签差已撤回。数据无人工WMH真值，上述衡量参考一致性。[CPU ABBA](../../validation/smri_cpu/synth_fixes_20261004/reports/wmh_cpu_node7_abba.public.json)、[最终GPU门](../../validation/smri_cpu/synth_fixes_20261004/reports/wmh_final_gpu_acceptance.public.json)。

![真实FLAIR官方与FNIT CPU标签；本图属已记录输出头修复矩阵](../../validation/smri_cpu_20261004/t2_seg/wmh_raw_flair_labels.png)

<a id="真实数据验证与更新记录"></a>

## 6. 最近版本和 benchmark

| 日期 | commit / version | 变化 | benchmark |
|---|---|---|---|
| 2026-10-04 | source_v10 | GPU crop缓冲生命周期与full原路径分派 | 20GB crop三文件SHA及实际kernel同，full超预算 |
| 2026-10-04 | t2_seg保存修复 | 新RAS网格输出header与XML扩展修复 | [FLAIR full/T1 crop逐值对照](../../validation/smri_cpu_20261004/t2_seg/wmh_output_header.public.json) |
| 2026-09-27 | WMH报告源码SHA | 独立公开三例完整对照 | [历史报告](../../validation/wmh/report.public.json) |

每条记录保留真实冻结源码、输入与时间边界；逐例、debug/profiling和更早脑图见[完整归档](../../validation/wmh_synthseg/readme_archive_20261005.md)。文档整理不重跑MRI，不把执行成功或--help核验作为精度benchmark。

<a id="reference"></a>

## 7. 参考文献、原软件和资源

- 参考文献：Laso et al., *Quantifying white matter hyperintensity and brain volumes in heterogeneous clinical and low-field portable MRI*, ISBI (2024), [doi:10.1109/ISBI56570.2024.10635502](https://doi.org/10.1109/ISBI56570.2024.10635502)。
- 原实现代码库：[FreeSurfer WMH-SynthSeg](https://github.com/freesurfer/freesurfer/tree/dev/mri_WMHsynthseg/WMHSynthSeg)。

模型使用下列官方原始文件，Git/wheel不包含。固定[assets-v1 Release](https://github.com/weikanggong1/Fudan-Neuroimaging-toolkit/releases/tag/assets-v1)及公开asset-manifest与当前weights.py逐项大小/SHA记录一致；本轮未重新下载所有大文件。安装器先Release再原站；完整清单见[资源文件清单](../RESOURCE_MANIFEST.md)。

```bash
fnit-setup-weights --model wmh-synthseg --dest /data/fnit-weights
fnit-setup-weights --model wmh-synthseg --dest /data/fnit-weights --verify-only
```

离线运行先下载此checkpoint并通过verify-only，再显式指定`weights="/data/fnit-weights"`。该模型不同于普通SynthSeg的H5权重；文件名、大小和SHA都应匹配下表。构造在CPU加载checkpoint后只转移一次设备，重复处理被试可以复用同一模型对象。

Python `threads=None`保留Torch线程设置；负数使用CPU核心数。输入需有正的最大强度以完成归一化。crop=True输出视野和affine会改变，批量汇总前分别保留每例几何。

| 资源 | 用途 | 官方来源 | 大小 | SHA-256 | 是否允许 FNIT 再分发 |
|---|---|---|---|---|---|
| `WMH-SynthSeg_v10_231110.pth` | 官方推理权重 / 标签数组 | [原站](https://ftp.nmr.mgh.harvard.edu/pub/dist/lcnpublic/dist/WMH-SynthSeg/WMH-SynthSeg_v10_231110.pth) | 790,531,383 B | `0ece39dd651357aa95222fc4d45fa32d00f11e763d2583cae3f869989ce35988` | 允许；FreeSurfer许可，保留条款与归属 |

本页列出的模型/数组共1个，790,531,383 B。原始文件许可及归属见[统一资源规则](../WEIGHTS.md#权重许可与归属)。模型推理从本地加载已准备资源。
