# SynthStrip：脑提取

[返回首页](../../README.md) · [完整旧文档与更早证据](../../validation/synthstrip/readme_archive_20261005.md)

| 项目 | 内容 |
|---|---|
| 输入 | 3D/4D MRI或CT，输入原生空间 |
| 输出 | 脑图、二值mask及有符号距离场 |
| 对应原软件 | FreeSurfer mri_synthstrip |
| Python / CLI | fnit.SynthStrip；fnit synthstrip |
| CPU / GPU | CPU或CUDA；几何处理和读写包含CPU步骤 |

## 1. 功能简介

SynthStrip 从脑影像预测有符号距离场，生成脑掩膜和去除背景的影像。官方模型已使用 PyTorch。本模块沿用其网络、权重和影像处理流程，提供可重复调用的 Python API；模型未重新训练。

参考版本为 **FreeSurfer 8.2.0**，build `freesurfer-linux-centos7_x86_64-8.2.0-20260314-d932c45`。分析对象是 `$FREESURFER_HOME/python/scripts/mri_synthstrip`，不是 `bin/` 下的包装脚本。脚本 SHA-256 为 `bbc2ff8f8779862039401b05d5cd6039fb4f3583e0032a793ac9adb3f4521590`，全部来源信息见 [provenance.json](../provenance.json)。

CUDA默认TF32；CPU oneDNN可用时用channels_last_3d，模型与输入保持float32。模型可重复使用，推理不启动FreeSurfer。

<a id="python-api"></a>
<a id="准备同一输入的两组脑图掩膜以及只使用一次的原配准场"></a>

## 2. Python 调用

```python
from pathlib import Path
from fnit import SynthStrip

output_directory = Path("results")  # 本次输出目录
output_directory.mkdir(parents=True, exist_ok=True)  # 准备保存目录
brain_extraction_model = SynthStrip(
    weights="/path/to/weights",  # 权重：官方 PT 文件或权重目录
    device="cuda:0",  # 设备：第一张可见 CUDA GPU
    no_csf=False,  # 模型：保留 CSF 的默认模型
    threads=4,  # CPU 线程：预处理与后处理使用 4 线程
)
brain_extraction_result = brain_extraction_model(
    image="subject_T1w.nii.gz",  # 输入：单幅 3D T1w，也可传 nibabel 空间影像
    border=1,  # 掩膜阈值：距离场小于 1 mm 视为脑内
    fill=0,  # 输出背景：脑掩膜外填 0
)
brain_extraction_result.image.save(path=output_directory / "subject_brain.nii.gz")  # 脑提取后的 T1w
brain_extraction_result.mask.save(path=output_directory / "subject_mask.nii.gz")  # 二值脑掩膜
brain_extraction_result.distance.save(path=output_directory / "subject_sdt.nii.gz")  # 有符号距离场
# 可继续 brain_extraction_model(image="another_T1w.nii.gz")，复用已加载模型。
```

### 输入数据格式

- `image`：`.nii`、`.nii.gz`、`.mgz`路径或 nibabel SpatialImage；3D `(X,Y,Z)` 或4D `(X,Y,Z,T)`，4D逐帧提取，保持帧顺序。MRI强度无固定单位，CT按原输入强度读入。
- 必须带有效4×4 affine和体素间距；不要求事先重排orientation、去颅骨或校正偏场。最终结果回到输入原生网格。
- 输入对象可含ArrayProxy或自有已解码ndarray；不再根据header对已解码数组重复缩放。距离场以mm计。

### 模型构造

| 参数 | 必需 | 类型 | 默认值 | 含义 |
|---|---|---|---|---|
| `weights` | 否 | `str 或 Path 或 None` | `None` | 官方 checkpoint 文件或目录；省略时按显式配置、FNIT_WEIGHTS 和缓存查找 |
| `device` | 否 | `str` | `'cpu'` | 计算设备，cpu 或 cuda:N；编号遵循 CUDA_VISIBLE_DEVICES |
| `no_csf` | 否 | `bool` | `False` | 自动选择排除 CSF 的模型；显式 PT 文件优先 |
| `threads` | 否 | `int 或 None` | `None` | PyTorch CPU 线程预算；None 保留当前值，CLI 默认另见下节 |
| `configure_precision` | 否 | `bool` | `True` | CUDA True 配置默认 TF32，False 保留调用方策略；CPU 不改 CUDA 状态 |

### 单次调用

| 参数 | 必需 | 类型 | 默认值 | 含义 |
|---|---|---|---|---|
| `image` | 是 | `str 或 Path 或 nibabel SpatialImage` | `—` | 输入影像，格式、维数与预处理约束见输入数据格式 |
| `border` | 否 | `float` | `1` | 有符号距离场阈值，单位 mm；小于此值视为脑内 |
| `fill` | 否 | `float 或 None` | `None` | 视野外或掩膜外填充值；SynthStrip None 为 min(input.min(),0) |
| `precision_report` | 否 | `list[dict] 或 None` | `None` | 可选列表，追加实际前向设备、dtype、TF32 和 autocast 记录 |

### 输出

```text
results/
├── subject_brain.nii.gz  # 掩膜外按fill填充；保留输入dtype
├── subject_mask.nii.gz   # uint8；0背景、1脑内
└── subject_sdt.nii.gz    # float32有符号距离，mm
```

Python返回`StripResult(image, mask, distance)`，三者是nibabel兼容影像。3D/4D shape、affine和orientation与输入一致，空间仍为原生空间。Python需显式`.save()`，调用本身不创建结果目录；mask先由`distance < border`阈值生成，再保留最大连通分量并填孔。

<a id="命令行"></a>

## 3. 命令行调用

```bash
fnit synthstrip -i subject_T1w.nii.gz -o results/subject_brain.nii.gz \
  -m results/subject_mask.nii.gz -d results/subject_sdt.nii.gz --device cuda:0
```

### fnit synthstrip

| CLI 参数 | Python 参数 / 输出 | 含义 |
|---|---|---|
| `-i` / `--image` | `image` | 输入影像，格式、维数与预处理约束见输入数据格式 |
| `-o` / `--out` | `result.image.save` | 脑图保存路径 |
| `-m` / `--mask` | `result.mask.save` | 二值脑掩膜保存路径 |
| `-d` / `--sdt` | `result.distance.save` | SDT保存路径，mm |
| `--weights` | `weights` | 官方 checkpoint 文件或目录；省略时按显式配置、FNIT_WEIGHTS 和缓存查找 |
| `--device` | `device` | 计算设备，cpu 或 cuda:N；编号遵循 CUDA_VISIBLE_DEVICES |
| `--no-csf` | `no_csf` | 自动选择排除 CSF 的模型；显式 PT 文件优先 |
| `-b` / `--border` | `border` | 有符号距离场阈值，单位 mm；小于此值视为脑内 |
| `-f` / `--fill` | `fill` | 视野外或掩膜外填充值；SynthStrip None 为 min(input.min(),0) |
| `-j` / `--threads` | `threads` | PyTorch CPU 线程预算；None 保留当前值，CLI 默认另见下节 |

CLI与Python默认差异：Python threads=None，CLI为4；GPU均需显式device。

## 4. 原软件调用

以下命令用于独立原软件参考环境；FNIT生产入口不执行它。

```bash
mri_synthstrip -i subject_T1w.nii.gz -o reference/subject_brain.nii.gz \
  -m reference/subject_mask.nii.gz -d reference/subject_sdt.nii.gz -b 1
```

| FNIT参数 / 产物 | 原软件参数 / 产物 |
|---|---|
| image / result.image / mask / distance | -i / -o / -m / -d |
| border / fill | -b / -f |
| no_csf / weights | --no-csf / --model |
| device=cpu / threads | --cpu / --threads |

沿用同一官方网络与模型；支持逐帧4D调用。FNIT显式选择cuda:N，原脚本主要使用cpu开关和默认GPU选择。原软件仅在独立对照环境运行。

<a id="单次调用与结果"></a>
<a id="源码组织"></a>
<a id="网络"></a>
<a id="每帧处理"></a>
<a id="功能差异与验证"></a>
<a id="2026-10-04cpu-对照与原始-nifti-几何问题"></a>
<a id="默认完全物化-spatialimage-apiv5-冻结"></a>
<a id="既有-gpu-完整三维对照cfb7beee"></a>
<a id="跨进程卷积选择修复"></a>
<a id="模板空间脑提取示例"></a>
<a id="用本包-cpu-采样两组结果保存模板-png匿名统计和私有中间图"></a>
<a id="测试与复现"></a>
<a id="检查-synthstrip-几何公开接口和默认-tf32-设置"></a>
<a id="该验证脚本使用-fnit-推理不从生产接口启动-freesurfer"></a>

## 5. 最新精度和运行时间

2026-10-04正式CPU CLI测试用公开ds003138 v1.0.1两例原始T1及9个附加场景，原版为FreeSurfer 8.2.0-1。冻结模块`6b7aeafd`与共享几何`dc2fc052`，nodecw10相同8物理核/8线程，默认T1按ABBA各两次。CPU张量float32；完整CLI含启动、模型加载、读写。报告保留逐文件源码SHA、权重、RSS和负载，不改标为本次文档整理的main。

### 端到端 benchmark

| 指标 | FNIT | 原软件 | 差异 |
|---|---|---|---|
| 两病例CLI中位数再取病例中位数 | 14.464 s | 47.380 s | brain/mask逐值同，SDT容差通过 |
| 11场景、26次运行 | mask Dice=1 | 独立官方输出 | SDT最大全场景绝对差4.3392e-5 mm |

### 分步骤 benchmark

| 阶段 | FNIT | 原软件 |
|---|---|---|
| 单例网络剖析 | 6.029 s | 8.209 s |
| 单例观察API | 9.510 s | 未记录相同API边界 |
| 默认物化对象v5：API / 保存 | 6.606 / 2.898 s | 保存参考复用；未重测 |

单例分步数据与两例CLI摘要分开；逐被试运行记录仅在详细报告中。
分步剖析含包装与哈希开销，不能与CLI中位数拼成新加速比。同几何H100旧新GPU回归三数组逐值同，allocated/reserved为6.503/10.800 GB。GPU TF32相对官方CPU仍差59个mask体素，Dice0.99998962；CPU与GPU结论分别保留。[完整协议与各场景](../../validation/smri_cpu/strip_sr_20261004/README.md)、[CLI报告](../../validation/smri_cpu/strip_sr_20261004/reports/synthstrip_cpu_cli.public.json)、[物化API报告](../../validation/smri_cpu/strip_sr_20261004/in_memory_20261004/README.md)。

![真实T1的官方、旧几何差异与修复后CPU掩膜](figures/synthstrip_cpu_header_alignment.png)

<a id="冻结版本的几何与原程序对照"></a>
<a id="最近版本更新与-benchmark"></a>

## 6. 最近版本和 benchmark

| 日期 | commit / version | 变化 | benchmark |
|---|---|---|---|
| 2026-10-04 | 00fedf3544/v5 | 默认完全物化SpatialImage API | 三输出与既有FNIT参考SHA同；[报告](../../validation/smri_cpu/strip_sr_20261004/in_memory_20261004/README.md) |
| 2026-10-04 | 6b7aeafd + dc2fc052 | 保留header间距、CPU布局及CUDA状态 | 11场景CPU CLI和GPU回归，见上节 |
| 2026-10-02 | 独立固定b0控制 | dMRI调用标准权重 | [实际输入与报告](../../validation/dmri_pipeline/synthstrip_fixed_input_20261002.public.json) |
| 较早版本 | 44364a8 | 固定跨进程卷积算法选择 | [真实T1重复性](../../validation/synthstrip/cudnn_repeatability.public.json) |

每条记录保留真实冻结源码、输入与时间边界；逐例、debug/profiling和更早脑图见[完整归档](../../validation/synthstrip/readme_archive_20261005.md)。文档整理不重跑MRI，不把执行成功或--help核验作为精度benchmark。

<a id="权重与-cpugpu-分工"></a>
<a id="在装有官方-surfa-的验证环境中对同一真实输入和权重比较几何及回采样"></a>
<a id="reference"></a>

## 7. 参考文献、原软件和资源

- 参考文献：Hoopes et al., *SynthStrip: Skull-Stripping for Any Brain Image*, NeuroImage (2022), [doi:10.1016/j.neuroimage.2022.119474](https://doi.org/10.1016/j.neuroimage.2022.119474)。
- 原实现代码库：[FreeSurfer `mri_synthstrip`](https://github.com/freesurfer/freesurfer/tree/dev/mri_synthstrip)。

模型使用下列官方原始文件，Git/wheel不包含。固定[assets-v1 Release](https://github.com/weikanggong1/Fudan-Neuroimaging-toolkit/releases/tag/assets-v1)及公开asset-manifest与当前weights.py逐项大小/SHA记录一致；本轮未重新下载所有大文件。安装器先Release再原站；完整清单见[资源文件清单](../RESOURCE_MANIFEST.md)。

```bash
fnit-setup-weights --model synthstrip --model synthstrip-nocsf --dest /data/fnit-weights
fnit-setup-weights --model synthstrip --model synthstrip-nocsf --dest /data/fnit-weights --verify-only
```

| 资源 | 用途 | 官方来源 | 大小 | SHA-256 | 是否允许 FNIT 再分发 |
|---|---|---|---|---|---|
| `synthstrip.1.pt` | 官方推理权重 / 标签数组 | [原站](https://surfer.nmr.mgh.harvard.edu/docs/synthstrip/requirements/synthstrip.1.pt) | 30,851,709 B | `37417f802196186441aae3e7f385d94f8a98c64a88acaeaa2723af995c653e33` | 允许；CC BY 4.0，保留归属 |
| `synthstrip.nocsf.1.pt` | 官方推理权重 / 标签数组 | [原站](https://surfer.nmr.mgh.harvard.edu/docs/synthstrip/requirements/synthstrip.nocsf.1.pt) | 30,851,709 B | `62bf01137c45b5f0cc04d59dbaed5b9ac138b3f25b766c062a7c1a0d696ecb28` | 允许；CC BY 4.0，保留归属 |

本页列出的模型/数组共2个，61,703,418 B。原始文件许可及归属见[统一资源规则](../WEIGHTS.md#权重许可与归属)。模型推理从本地加载已准备资源。
