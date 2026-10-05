# SynthSeg+ 体积分区

[返回首页](../../README.md) · [完整旧文档与更早证据](../../validation/synthseg_plus/readme_archive_20261005.md) · [CUDA精度声明](precision.md)

| 项目 | 内容 |
|---|---|
| 输入 | 单幅3D T1w |
| 输出 | 33类主分割、68区皮层及合并标签、可选软体积 |
| 对应原软件 | FreeSurfer mri_synthseg --parc（普通2.0） |
| Python / CLI | fnit.SynthSegPlus；fnit synthseg --parc |
| CPU / GPU | CPU/CUDA；读写和部分后处理在CPU |

## 1. 功能简介

`SynthSegPlus` 在已有 PyTorch SynthSeg 2.0 的 33 类分割后运行官方 69 通道皮层分区网络，输出左右半球 68 个体积脑区标签。推理只依赖 FNIT 的 Python/Conda 环境和官方权重；对照测试才使用 FreeSurfer 8.2。这里的分区是体素标签，不是 recon-all 的表面 DKT 分区，因而不提供表面面积或厚度。
[原版 `mri_synthseg` 源码](https://github.com/freesurfer/freesurfer/blob/v8.2.0/mri_synthseg/mri_synthseg)中的 `--parc` 网络输入、皮层背景重置和 `--vol` 后验求和，是这里的对应步骤。

FNIT 的类名 `SynthSegPlus` 对应普通 SynthSeg 2.0 **加 `--parc`**。原论文中的 SynthSeg+ 是鲁棒分割方法，对应原版 `--robust`，使用不同模型和后处理；FNIT 当前没有实现这条模式。`fast=True` 对应普通 `--parc --fast`，不能代替 `--robust`。

模型重复使用时复用主分割及皮层分区权重。CPU大体积末层投影已按深度分块以修复oneDNN崩溃；其他网络与GPU策略保持原样。

<a id="python-用法"></a>

## 2. Python 调用

```python
from fnit import SynthSegPlus

parcellation_model = SynthSegPlus(
    weights="/absolute/path/weights",       # 主网络及三个主分割 npy 所在目录
    parc_weights="/absolute/path/weights",  # 皮层分区 H5 所在目录
    device="cuda:0",                         # 计算设备；CPU 可写 "cpu"
    cudnn_tf32=True,                         # 默认cuDNN TF32；False关闭，None逐次继承
)
parcellation_result = parcellation_model(
    t1="sub-01_T1w.nii.gz",  # 输入：一幅 3D T1 NIfTI/MGZ 路径
    keep_geometry=False,    # 输出：使用原版默认的 RAS、约 1 mm 推理网格
    fast=False,             # 使用原版普通 --parc 的左右翻转集成
    min_pad=128,            # 网络输入各轴的最小补零尺寸，单位为体素
    volumes=True,           # 计算 --vol 软体积；默认 False，关闭时推理较快
)
parcellation_result.segmentation.save("sub-01_33class.nii.gz")          # 33 类主分割
parcellation_result.cortical_parcellation.save("sub-01_cortex.nii.gz")  # 仅皮层的 68 区标签
parcellation_result.combined.save("sub-01_synthseg_plus.nii.gz")        # 主分割与皮层分区合并
parcellation_result.write_volumes_csv(
    source="sub-01_T1w.nii.gz",  # CSV 第一列的被试名由此路径生成
    path="sub-01_volumes.csv",  # 输出：101 个数值列的软体积表，单位 mm³
)
left_precentral_mask = parcellation_result.mask("ctx-lh-precentral")         # 布尔掩膜；shape 与 combined 相同
```

### 输入数据格式

- `t1`：单幅3D `(X,Y,Z)` T1w路径，`.nii`、`.nii.gz`、`.mgz`；原生空间、有效affine和体素间距，强度无固定单位，无需先去颅骨。
- 内部生成RAS、约1mm网格；**Python默认keep_geometry=True，CLI默认False**。示例显式False用于原版默认输出对照。
- `weights`为主模型与三份NPY所在目录，`parc_weights`为皮层模型H5或目录。两者可省略用FNIT已配置缓存；不在推理中联网。

### 模型构造

| 参数 | 必需 | 类型 | 默认值 | 含义 |
|---|---|---|---|---|
| `weights` | 否 | `str 或 Path 或 None` | `None` | 官方 checkpoint 文件或目录；省略时按显式配置、FNIT_WEIGHTS 和缓存查找 |
| `parc_weights` | 否 | `str 或 Path 或 None` | `None` | 官方皮层分区 H5 文件或目录 |
| `device` | 否 | `str 或 torch.device` | `'cpu'` | 计算设备，cpu 或 cuda:N；编号遵循 CUDA_VISIBLE_DEVICES |
| `cudnn_tf32` | 否 | `bool 或 None`，关键字参数 | `True` | 两个CUDA网络及全部平滑的cuDNN TF32；False关闭，None继承每次调用前cuDNN；作用域内matmul TF32仍True，退出恢复两开关 |

### 单次调用

| 参数 | 必需 | 类型 | 默认值 | 含义 |
|---|---|---|---|---|
| `t1` | 是 | `str 或 Path` | `—` | 一幅原始 3D T1w；格式见输入数据格式 |
| `keep_geometry` | 否 | `bool` | `True` | 标签以最近邻重采样回输入 shape 和 affine |
| `fast` | 否 | `bool` | `False` | 普通 SynthSeg --parc 的快速路径；不代表 robust 模型 |
| `min_pad` | 否 | `int` | `128` | 各轴网络输入的最小补零尺寸，单位体素 |
| `volumes` | 否 | `bool` | `False` | 计算软体积并允许写出 CSV；False 不计算 |

### 输出

```text
sub-01_33class.nii.gz      # 33类主分割
sub-01_cortex.nii.gz       # 仅皮层的左右68区
sub-01_synthseg_plus.nii.gz # 主分割+皮层合并
sub-01_volumes.csv         # volumes=True时101个数值列，mm³
```

`SynthSegPlusResult`中`segmentation`、`cortical_parcellation`和`combined`均为int32的nibabel兼容3D图，qform/sform code0/2。keep_geometry=True时shape/affine与输入同；False时为处理后的RAS约1mm网格。combined用官方皮层编号替换主分割3/42，标签名在`label_names`；`mask(编号或名称)`返回同shape布尔数组。

### 结果辅助方法

| 方法 / 参数 | 必需 | 类型 / 默认值 | 含义 |
|---|---|---|---|
| `mask(label)`的`label` | 是 | `int 或 str` | combined标签编号或唯一名称；返回同shape布尔数组，名称不匹配抛KeyError |
| `write_volumes_csv(source,path)`的`source` | 是 | `str 或 Path` | CSV被试名来源，取文件名并去掉`.nii.gz` |
| `write_volumes_csv(source,path)`的`path` | 是 | `str 或 Path` | CSV保存路径；自动创建父目录 |

`volumes_mm3`在`volumes=False`时为None，在True时按主分割32项及皮层68项保存软体积。CSV另有total intracranial，共101个数值列；皮层体积在各半球皮层概率内分配，不是硬标签体素数。

`precision`是结果末尾默认None的可选字段，记录本次实际网络/平滑的dtype、device、autocast和CUDA开关；模型实例`.precision`在缓存拒绝或内部分区调用失败时为None。CPU不写CUDA状态，CUDA调用正常/异常退出均恢复原开关。缓存按声明True/False/None匹配，不匹配在读T1前报错；None缓存可逐次继承。完整语义及串行/独立进程要求见[精度说明](precision.md)。

`volumes=True`才计算101列软体积，定义与原--vol相同；`write_volumes_csv(source,path)`两参必需，不可在volumes=False后调用。Python需显式保存；CLI默认只保存combined，--parc-out另存仅皮层图。

<a id="命令行与原版"></a>

## 3. 命令行调用

```bash
fnit synthseg --parc --i sub-01_T1w.nii.gz --o sub-01_synthseg_plus.nii.gz \
  --csv-vols sub-01_volumes.csv --device cuda:0 --threads 4
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

CLI与Python默认差异：Python keep_geometry=True，CLI默认False；CLI threads=4，Python构造不含threads；只有--csv-vols才计算volumes。CLI仍使用默认cudnn_tf32=True，没有新增精度参数；False/None用Python声明。

## 4. 原软件调用

以下命令用于独立原软件参考环境；FNIT生产入口不执行它。

```bash
mri_synthseg --parc --i sub-01_T1w.nii.gz --o reference/sub-01_synthseg_plus.nii.gz \
  --vol reference/sub-01_volumes.csv --threads 4 --noaddctab
```

| FNIT参数 / 产物 | 原软件参数 / 产物 |
|---|---|
| SynthSegPlus / fast | --parc / --fast |
| keep_geometry | --keepgeom |
| volumes / write_volumes_csv | --vol |
| weights / parc_weights | 主分割 / 皮层H5 |
| device=cpu / threads | --cpu / --threads |

当前为普通2.0加皮层体积分区，未实现原论文SynthSeg+的robust路径；不提供表面厚度/面积或QC。--parc与--color-lut组合当前明确拒绝。CLI设置线程，Python此类构造没有threads参数。

<a id="真实数据对照"></a>
<a id="2026-10-04-cpu-功能与-gpu-回归"></a>
<a id="此前公开-t1w-单例"></a>
<a id="大体积-t1-的-cpu-原生崩溃修复"></a>

## 5. 最新精度和运行时间

2026-10-06 修复首次调用的lazy皮层模型构造覆盖全局TF32的问题。构造不写开关，普通/fast的两个网络及全部高斯平滑统一使用声明True/False/None，并在正常和异常退出时恢复调用者状态。默认True的原始T1 **CPU4arm、H1004arm**三张图、完整影像头及101列软体积均与旧版逐值相同，连压缩NIfTI和CSV文件SHA也相同；GPU allocated/reserved旧新相同。新增False/None另有四个完整GPUarm，并单独核验继承和恢复。[七节报告及逐区/逐列数值](../../validation/smri_cpu/seg_tf32_20261005/README.md)。

对同输入官方，CPU默认普通/fast仍差 **5/1体素**、CSV最大 **0.657/0.220mm³**；GPU默认仍差 **328/498体素**、CSV最大 **118.280/121.600mm³**，没有改变默认数学。GPU新增False普通硬标签0差、Dice1、CSV最大0.20mm³；fast差2体素、最小Dice0.99997630、CSV最大0.346mm³。None继承关闭状态逐值同False。这是单例可选政策效果，不代表所有输入或全部官方数值完全等价。

同节点八核默认CPU旧/新API普通 **49.898/51.018s**、fast **35.474/35.347s**，冷worker外层 **52.680/53.577s**及 **38.009/38.160s**。本次每功能一组，普通新API约慢2.2%、fast约快0.36%，不作提速声明。既有同节点官方冷CLI **48.296/33.784s**；双方保存输出/校验边界不同且已分别列出，CPU总速度目标仍未通过。

H100默认普通/fast旧/新API **10.068/9.732s**及 **7.097/6.937s**，allocated均12.568GB、reserved18.207/18.900GB旧新相同。本进程树driver采样最大18.772/19.464GB，小于20GB；最大采样间隔0.786s，不能称未采样绝对物理峰值。整GPU同时约66–67GB共享背景，墙钟不作稳定加速倍率。False/None普通/fast reserved15.162/15.590GB。此前CPU单缓冲拼接及大体积CPU末层投影防崩保留，[上一阶段验收](../../validation/smri_cpu/seg_memory_20261005/README.md)。

前版大T1崩溃修复用公开ds000114完整真实T1，网络张量224×288×288，nodecw10同8物理核/8线程，CPUfloat32；原版FreeSurfer8.2.0-1。普通--parc的冷进程ABBA含加载、计算、CSV和合并图保存。源码SHA与逐区结果绑定[正式记录](../../validation/smri_cpu_20261004/t2_seg/large_pointwise.public.json)。

### 端到端 benchmark

| 指标 | FNIT | 原软件 | 差异 |
|---|---|---|---|
| 普通--parc完整CLI两次 | 102.13 / 106.16 s | 325.90 / 118.17 s | 官方首轮有文件读取等待 |
| 合并标签不同体素 / 最小Dice | 5 / 0.99980350 | 独立官方输出 | 不是逐值一致 |
| 101列软体积最大差 | 0.683 mm³ | 独立官方输出 | 候选两次数字/几何逐值同 |
| GPU完整输入allocated峰值 | 12.57 GB | 未测相同预算 | 旧新普通/fast数组与CSV相同 |

### 分步骤 benchmark

| 阶段 | FNIT | 原软件 |
|---|---|---|
| 主分割 / 皮层网络 / 后处理 | 原报告含逐层崩溃定位；未保存统一双方阶段计时 | 未记录同边界阶段表 |
| GPU普通完整worker旧/新 | 15.03 / 12.53 s | 非原软件对照 |
| GPUfast完整worker旧/新 | 10.04 / 10.03 s | 非原软件对照 |

只分块估算填充超过2³¹−1 B的CPUFP32 1×1×1投影，两个已有病例普通/fast以及最终GPU配对硬标签、CSV和几何均同；未将较早全层切片变慢候选接入。共享负载不同，单组时间不能作为稳定提速。[协议和所有状态](../../validation/smri_cpu_20261004/t2_seg/README.md#9-大体积-plus-cpu-崩溃定位与修复)。

![真实大T1官方和FNIT CPU标签及差异位置](../../validation/smri_cpu_20261004/t2_seg/large_pointwise_labels.png)

## 6. 最近版本和 benchmark

| 日期 | commit / version | 变化 | benchmark |
|---|---|---|---|
| 2026-10-06 | `46eead65` 源码 | 普通/fast统一cuDNN策略，修复lazy构造覆盖开关；True默认不变，False/None新增 | 12完整arm及真实特征/异常门；见[本次报告](../../validation/smri_cpu/seg_tf32_20261005/README.md) |
| 2026-10-05 | `585bf181` 源码；`b1d46705` 完整报告 | CPU 单缓冲拼接；原卷积、GPU 数学与精度政策不变 | 普通/fast 完整旧新输出相同，fast ABBA 近似持平；同节点官方仍较快，见[最新记录](../../validation/smri_cpu/seg_memory_20261005/README.md) |
| 2026-10-04 | large_pointwise冻结源码 | 最小末层CPU投影分块，修复SIGSEGV | 上节ABBA及CPU/GPU配对 |
| 2026-10-04 | t2_seg v2 | 共享endpoint、CPU连通域与保存几何 | [矩阵与保存合同](../../validation/smri_cpu_20261004/t2_seg/README.md) |
| 较早单例 | synthseg_plus报告源码SHA | 普通parc及101列软体积 | [历史完整CPU/GPU对照](../../validation/synthseg_plus/README.md) |

每条记录保留真实冻结源码、输入与时间边界；逐例、debug/profiling和更早脑图见[完整归档](../../validation/synthseg_plus/readme_archive_20261005.md)。文档整理不重跑MRI，不把执行成功或--help核验作为精度benchmark。

<a id="安装与权重"></a>
<a id="reference"></a>

## 7. 参考文献、原软件和资源

- 参考文献：Billot et al., *Robust machine learning segmentation for large-scale analysis of heterogeneous clinical brain MRI datasets*, PNAS (2023), [doi:10.1073/pnas.2216399120](https://doi.org/10.1073/pnas.2216399120)。
- 原实现代码库：[FreeSurfer `mri_synthseg --parc`](https://github.com/freesurfer/freesurfer/tree/dev/mri_synthseg)。

模型使用下列官方原始文件，Git/wheel不包含。固定[assets-v1 Release](https://github.com/weikanggong1/Fudan-Neuroimaging-toolkit/releases/tag/assets-v1)及公开asset-manifest与当前weights.py逐项大小/SHA记录一致；本轮未重新下载所有大文件。安装器先Release再原站；完整清单见[资源文件清单](../RESOURCE_MANIFEST.md)。

```bash
fnit-setup-weights --model synthseg-plus --dest /data/fnit-weights
fnit-setup-weights --model synthseg-plus --dest /data/fnit-weights --verify-only
```

| 资源 | 用途 | 官方来源 | 大小 | SHA-256 | 是否允许 FNIT 再分发 |
|---|---|---|---|---|---|
| `synthseg_2.0.h5` | 官方推理权重 / 标签数组 | [原站](https://surfer.nmr.mgh.harvard.edu/pub/dist/freesurfer/repo/annex.git/annex/objects/bee/241/SHA256E-s53079152--f190bfd742f450ef3ca2c9df9ed4d2e0232b3a74471da5e51b7770bacdf80c3e.0.h5/SHA256E-s53079152--f190bfd742f450ef3ca2c9df9ed4d2e0232b3a74471da5e51b7770bacdf80c3e.0.h5) | 53,079,152 B | `f190bfd742f450ef3ca2c9df9ed4d2e0232b3a74471da5e51b7770bacdf80c3e` | 允许；FreeSurfer许可，保留条款与归属 |
| `synthseg_parc_2.0.h5` | 官方推理权重 / 标签数组 | [原站](https://surfer.nmr.mgh.harvard.edu/pub/dist/freesurfer/repo/annex.git/annex/objects/c04/403/SHA256E-s53090840--83bb1de76fb6f173c6dacacd433f81209fc6abb1dbc179a930ec06ecabbeb684.0.h5/SHA256E-s53090840--83bb1de76fb6f173c6dacacd433f81209fc6abb1dbc179a930ec06ecabbeb684.0.h5) | 53,090,840 B | `83bb1de76fb6f173c6dacacd433f81209fc6abb1dbc179a930ec06ecabbeb684` | 允许；FreeSurfer许可，保留条款与归属 |
| `synthseg_segmentation_labels_2.0.npy` | 官方推理权重 / 标签数组 | [原站](https://raw.githubusercontent.com/freesurfer/freesurfer/v8.2.0/mri_synthseg/synthseg_segmentation_labels_2.0.npy) | 348 B | `5ef25ec33fe917ac99f30b8f2185b2d77121136ee411b9c4970c0b59be615ed8` | 允许；FreeSurfer许可，保留条款与归属 |
| `synthseg_segmentation_names_2.0.npy` | 官方推理权重 / 标签数组 | [原站](https://raw.githubusercontent.com/freesurfer/freesurfer/v8.2.0/mri_synthseg/synthseg_segmentation_names_2.0.npy) | 7,168 B | `234eb6d514e10d6ebd748a8b30a1d12d9426fd874c607e37852406fae8f290fc` | 允许；FreeSurfer许可，保留条款与归属 |
| `synthseg_topological_classes_2.0.npy` | 官方推理权重 / 标签数组 | [原站](https://raw.githubusercontent.com/freesurfer/freesurfer/v8.2.0/mri_synthseg/synthseg_topological_classes_2.0.npy) | 348 B | `650b4b96834485c1e6d7421de4af74da80d861e6b2a39ef1164389bde3a5e14a` | 允许；FreeSurfer许可，保留条款与归属 |

本页列出的模型/数组共5个，106,177,856 B。原始文件许可及归属见[统一资源规则](../WEIGHTS.md#权重许可与归属)。模型推理从本地加载已准备资源。

皮层权重的Release清单official_url仍写annex p0/0f，当前源码使用c04/403；大小与SHA一致。此页采用当前源码原站地址，清单URL差异不当作模型字节变化。
