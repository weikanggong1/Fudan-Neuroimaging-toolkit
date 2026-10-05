# SynthSeg 体积分区的 CUDA 精度设置

## 1. 功能简介

`SynthSegPlus` 是普通 SynthSeg 2.0 加 `--parc`，包括 33 类分割、68 区皮层体积分区和软体积；原版 `--robust` SynthSeg+ 尚未实现。
本次修复首次调用时皮层模型构造覆盖全局 TF32 的问题。模型构造只声明策略；实际调用统一覆盖主分割网络、皮层网络及两者的高斯平滑，并在正常或异常退出时恢复调用者的 CUDA 开关。

```mermaid
flowchart LR
    A[原始3D T1] --> B[核对缓存声明策略]
    B --> C[进入精度作用域]
    C --> D[预处理与主分割]
    D --> E[普通集成或fast平滑]
    E --> F[皮层网络与平滑]
    F --> G[标签与软体积]
    G --> H[恢复调用者开关]
```

默认 `cudnn_tf32=True` 保留已有 CUDA 数学。CPU 不改变 CUDA 开关；不会开启、关闭或改变调用者的 autocast。没有新增依赖，也没有使用 float16/BF16。

## 2. Python 调用、输入与输出

```python
from fnit import SynthSegPlus

parcellation_model = SynthSegPlus(
    weights="/data/fnit-weights",       # 已校验的主模型和标签目录
    parc_weights="/data/fnit-weights",  # 已校验的皮层模型目录
    device="cuda:0",                   # 明确的目标设备
    cudnn_tf32=True,                    # 默认；两个网络和全部高斯平滑允许cuDNN TF32
)
parcellation_result = parcellation_model(
    t1="sub-01_T1w.nii.gz",  # 原始3D T1；不读取官方分割结果
    keep_geometry=False,    # 与原版默认RAS、约1mm输出网格对应
    fast=False,             # 普通主网络左右翻转集成；True改为原版fast路径
    min_pad=128,            # 各轴最小补零体素数
    volumes=True,           # 计算101列软体积，单位mm³
)
parcellation_result.combined.save("sub-01_parc.nii.gz")
parcellation_result.write_volumes_csv(
    source="sub-01_T1w.nii.gz",  # CSV输入标识
    path="sub-01_volumes.csv",   # 101个数值列及输入标识
)
actual_precision = parcellation_result.precision  # 标量诊断，不包含影像张量
```

所有原有参数、输入格式、图像结构和标签编号见[功能说明](README.md#2-python-调用)。新增参数如下。

| 参数/字段 | 默认/类型 | 含义 |
|---|---|---|
| `SynthSegPlus(..., cudnn_tf32=...)` | `True`；关键字参数，`bool或None` | 声明整条普通分区链的cuDNN TF32策略 |
| `SynthSegParc(..., cudnn_tf32=...)` | `True`；关键字参数，`bool或None` | 内部皮层网络及其高斯平滑的同一策略 |
| `run_synthseg_parc_t1(..., cudnn_tf32=...)` | `True`；关键字参数，`bool或None` | 内部完整分区链策略；使用缓存时必须匹配 |
| `SynthSegPlusResult.precision` | 末尾可选字段，默认`None` | 成功调用的整链、主网络、皮层网络和平滑的实际dtype、device、autocast及CUDA开关；原有位置参数顺序不变 |
| `SynthSegPlus.precision` | 初始或内部分区调用失败时为`None` | 已完成内部分区的相同记录；缓存提前拒绝时不会保留旧成功记录 |

| 声明 | CUDA实际前向 | 调用后 |
|---|---|---|
| `True` | cuDNN TF32开启；matmul TF32开启 | 恢复调用前两开关 |
| `False` | cuDNN TF32关闭；matmul TF32开启 | 恢复调用前两开关 |
| `None` | cuDNN继承每次调用前状态；matmul TF32开启 | 恢复调用前两开关 |
| 任一声明的CPU调用 | 不修改CUDA开关 | CUDA状态保持原样 |

`False` 是 **cuDNN** 策略，不是全部矩阵运算的“严格FP32”开关。以上操作仍使用FP32张量；TF32改变CUDA计算内部的乘法精度。显式CPU/CUDA设备语义保持原样；旧版允许的 `device=None` 自动选择也保留，预处理仍沿用其原设备参数。

缓存按声明的 `True/False/None` 区分，不能因某次有效开关碰巧相同而混用。改变模型实例的声明时须重新构造匹配的子模型；传给内部函数的缓存不匹配会在加载T1前抛出 `ValueError`。`None` 缓存则可在不同调用中继承不同的cuDNN状态。
两个CUDA开关是进程全局状态；不同精度调用应串行执行或使用独立进程，不应在同一进程多个线程中同时设置不同策略。

## 3. 命令行调用

```bash
fnit synthseg --parc --i sub-01_T1w.nii.gz --o sub-01_parc.nii.gz \
  --csv-vols sub-01_volumes.csv --weights /data/fnit-weights \
  --parc-weights /data/fnit-weights --device cuda:0 --threads 8
```

CLI 仍使用默认 `True`，没有新增CLI精度参数；声明 `False/None` 使用上面的Python接口。`--fast` 仍选择普通快速分区。CLI输出/线程/原始网格差异见[功能说明](README.md#3-命令行调用)。

## 4. 原软件调用

```bash
mri_synthseg --parc --i sub-01_T1w.nii.gz --o reference/sub-01_parc.nii.gz \
  --vol reference/sub-01_volumes.csv --cpu --threads 8 --noaddctab
# 快速分区再加 --fast。
```

这只用于隔离的原软件验证。FNIT生产调用不需要FreeSurfer或TensorFlow。内部精度作用域没有独立原软件命令，也不与 `--robust` 对应。

## 5. 最新精度和时间

本次验收范围、原始T1完整输出、实际CUDA前向及异常恢复记录见[完整报告](../../validation/smri_cpu/seg_tf32_20261005/README.md)。默认True的旧新比较与新增False/None对官方参考的差异分别报告；小网格真实特征诊断不作为完整影像benchmark。

## 6. 最近版本与 benchmark

| 阶段 | 变化 | 记录 |
|---|---|---|
| `46eead65` | 构造不修改全局精度；True/False/None覆盖普通/fast所有实际网络和平滑；缓存与退出恢复门 | [本次完整验收](../../validation/smri_cpu/seg_tf32_20261005/README.md) |
| `585bf181` | CPU最近邻和skip拼接减少缓冲；原卷积及GPU数学不变 | [CPU拼接验收](../../validation/smri_cpu/seg_memory_20261005/README.md) |
| large_pointwise冻结版 | 大体积CPU末层1×1×1分块，修复oneDNN崩溃 | [既有验收](../../validation/smri_cpu_20261004/t2_seg/README.md#9-大体积-plus-cpu-崩溃定位与修复) |

## 7. 原实现、文献与资源

- [FreeSurfer 8.2 `mri_synthseg`](https://github.com/freesurfer/freesurfer/tree/v8.2.0/mri_synthseg)。
- Billot et al., *Robust machine learning segmentation for large-scale analysis of heterogeneous clinical brain MRI datasets*, PNAS (2023), [doi:10.1073/pnas.2216399120](https://doi.org/10.1073/pnas.2216399120)。
- 权重/标签的来源、许可、大小和SHA-256见[功能资源表](README.md#7-参考文献原软件和资源)及[统一资源清单](../RESOURCE_MANIFEST.md)。本次从原已核验目录加载相同五份资源，未重新下载、改权重或加入Git。
