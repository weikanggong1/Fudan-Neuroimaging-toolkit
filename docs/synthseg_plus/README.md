# SynthSeg+ 体积分区

[返回首页](../../README.md) · [33 类 SynthSeg](../synthseg/README.md) · [权重](../WEIGHTS.md)

`SynthSegPlus` 在已有 PyTorch SynthSeg 2.0 的 33 类分割后运行官方 69 通道皮层分区网络，输出左右半球 68 个体积脑区标签。推理只依赖 FNIT 的 Python/Conda 环境和官方权重；对照测试才使用 FreeSurfer 8.2。这里的分区是体素标签，不是 recon-all 的表面 DKT 分区，因而不提供表面面积或厚度。

## 安装与权重

按首页 `environment.yml` 创建环境，然后运行：

```bash
fnit-setup-weights --model synthseg-plus --dest /absolute/path/weights
```

这一组包括 SynthSeg 2.0 主网络、皮层分区网络、主分割标签/名称/拓扑数组。下载器检查文件大小和 SHA-256，推理过程不联网。皮层分区标签顺序与 FreeSurfer 8.2 `synthseg_parcellation_labels.npy` 的 69 项一致。

## Python 用法

```python
from fnit import SynthSegPlus

model = SynthSegPlus(
    weights="/absolute/path/weights",       # 主网络及三个主分割 npy 所在目录
    parc_weights="/absolute/path/weights",  # 皮层分区 H5 所在目录
    device="cuda:0",                         # 计算设备；CPU 可写 "cpu"
)
result = model(
    t1="sub-01_T1w.nii.gz",  # 输入：一幅 3D T1 NIfTI/MGZ 路径
    keep_geometry=True,     # 输出：最近邻重采样至原 T1 shape 和 affine
    fast=False,             # 使用完整左右翻转集成及后处理
    min_pad=128,            # 网络输入各轴的最小补零尺寸
)
result.segmentation.save("sub-01_33class.nii.gz")
result.cortical_parcellation.save("sub-01_cortex.nii.gz")
result.combined.save("sub-01_synthseg_plus.nii.gz")
left_precentral = result.mask("ctx-lh-precentral")
```

输入 `t1` 为单幅三维结构像路径。`weights`、`parc_weights` 可省略，按 `FNIT_WEIGHTS`、已配置目录和默认缓存查找。`fast=True` 采用较短的后处理路径；要对照原版普通 `--parc`，保持 `False`。`keep_geometry=False` 返回预处理后的 RAS、约 1 mm 网格，与命令行默认值相同。

返回对象的 `segmentation` 为 33 类解剖标签，`cortical_parcellation` 仅在皮层为 68 个脑区编号，`combined` 在 33 类标签上用脑区编号替换左右皮层 3/42。三者均为 `FNITNifti1Image`，标签 dtype 为 `int32`；`label_names` 为 `{编号: 名称}`，`mask(编号或名称)` 返回与 `combined` 同 shape 的布尔数组。本功能没有 `--vol` 软体积 CSV；硬标签体积可由体素数乘仿射行列式计算，但不能冒充原版的软体积。

## 命令行与原版

```bash
fnit synthseg --i sub-01_T1w.nii.gz --o sub-01_synthseg_plus.nii.gz \
  --parc --parc-out sub-01_cortex.nii.gz --keep-geometry \
  --weights /absolute/path/weights --parc-weights /absolute/path/weights \
  --device cuda:0

mri_synthseg --i sub-01_T1w.nii.gz --o sub-01_official_parc.nii.gz \
  --parc --threads 4
```

`--i` 是输入 T1，`--o` 是合并后的标签图，`--parc-out` 是可选的皮层单独标签图，`--keep-geometry` 要求输出使用输入体素网格。`--weights` 指主网络目录，`--parc-weights` 指皮层网络目录，`--device` 指 PyTorch 设备。原版 `mri_synthseg --parc` 只输出合并标签图；FNIT 的三个返回图用于分别检查主分割和分区。

## 真实数据对照

见[本轮验证记录](../../validation/synthseg_plus/README.md)。只有同一 T1、同一网格上逐标签核对后，才能把耗时差称为等价分区的速度差。GPU 使用 TF32；未使用 float16 或 bfloat16。
