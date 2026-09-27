# TorchFAST：单通道 T1 脑组织分割

[返回首页](../../README.md) · [代码](../../src/fnit/fast/) · [验证](../../validation/fast/README.md)

TorchFAST 对**已去脑、单帧 T1w** 估计 CSF、GM、WM 三种组织的分数体积，并输出组织标签、乘性偏置场和校正图。分割算法使用 PyTorch；影像读写使用仓库的 NiBabel 图像类，不调用已安装的 FSL 或 Surfa。该功能不需要模型权重。

## Python 调用

```python
from fnit.fast import TorchFAST

model = TorchFAST(
    device="cuda:0",             # PyTorch 设备；没有 GPU 时用 "cpu"
    threads=1,                   # CPU 线程数；CUDA 路径仍用它控制 CPU 前后处理
    init_iterations=15,          # 初始混合高斯更新次数
    bias_iterations=4,           # 联合分割与偏置场更新次数
    fixed_iterations=4,          # 偏置场固定后的分割更新次数
    bias_fwhm_mm=20.0,           # 偏置场平滑尺度，单位 mm
    init_mrf=0.02,               # 初始组织空间权重
    mrf=0.1,                     # 最终组织空间权重
    mixel_mrf=0.3,               # 混合组织类型的空间权重
    pve_steps=100,               # 最终 PVE 比例网格数
    mean_field_iterations=5,     # 每次 HMRF 同步更新次数
    pve_chunk_size=8,            # PVE 计算的切片块厚度
)
result = model(
    image="T1_brain.nii.gz",    # 必需：已去脑、单帧 T1w 的 NIfTI/MGZ 路径或内存影像
    mask="brain_mask.nii.gz",   # 可选：与 T1 相同尺寸和仿射的二值脑掩模
)
result.pve_gm.save(path="T1_brain_pve_1.nii.gz")  # 保存 GM 分数体积
```

`image` 的轴顺序与文件中的体素轴一致。`mask` 未提供时，使用 `image > 0`；提供时，使用 `(image > 0) & (mask > 0)`。掩模不重采样；尺寸或 voxel-to-world 仿射不同会报错。文件路径、仓库 `Volume`、NiBabel 影像和旧代码已持有的 Surfa 内存体都可作为输入；**加载路径和调用 TorchFAST 本身不导入 Surfa**。旧 Surfa 内存体输入仍返回同类内存体，便于已有调用迁移。路径、NiBabel 及仓库 `Volume` 输入返回仓库 `Volume`；其 `.data`、`.affine`、`.shape` 和 `.save(path)` 可直接使用。

`model(...)` 返回 `FASTResult`。八张影像的尺寸、体素方向、世界坐标与输入一致；NIfTI 路径输入的空间编码和单位等文件头字段在 NIfTI 输出中保留。前六张标签或概率图的脑外值为 0，偏置场脑外值为 1。

| 输出字段 | 数据类型 | 含义 |
|---|---|---|
| `pve_csf`、`pve_gm`、`pve_wm` | float32 影像 | CSF、GM、WM 分数体积，取值 0–1；脑内三者之和为 1 |
| `hard_segmentation` | int32 影像 | PVE 前的 HMRF 分类；1=CSF、2=GM、3=WM |
| `pve_segmentation` | int32 影像 | 最大 PVE 对应的组织标签；1/2/3 同上 |
| `mixel_type` | int32 影像 | 0–2 为纯 CSF/GM/WM，3–5 为 CSF-GM、CSF-WM、GM-WM |
| `bias_field` | float32 影像 | 原图中的乘性偏置场 |
| `restored` | float32 影像 | 脑内 `image / bias_field`，脑外为 0 |
| `tissue_means` | 3 个浮点数 | 校正后 CSF、GM、WM 的强度均值 |
| `tissue_variances` | 3 个浮点数 | 对应的强度方差 |

## 命令行

下例中 `-i` 是已去脑的单帧 T1w，`-o` 是输出文件名前缀；`--mask` 是可选同网格脑掩模，`--device` 选 PyTorch 设备，`--threads` 控制 CPU 线程数，`-b/-B` 分别保存偏置场和校正图：

```bash
fnit fast \
  -i T1_brain.nii.gz \
  -o results/T1_brain \
  --mask brain_mask.nii.gz \
  --device cuda:0 \
  --threads 1 -b -B
```

默认产生六张压缩 NIfTI；`-b/-B` 各增加一张：

| 文件名后缀 | 内容 |
|---|---|
| `_pve_0.nii.gz`、`_pve_1.nii.gz`、`_pve_2.nii.gz` | CSF、GM、WM 分数体积 |
| `_seg.nii.gz` | PVE 前的硬分类 |
| `_pveseg.nii.gz` | 最大 PVE 分类 |
| `_mixeltype.nii.gz` | 纯/混合组织类型 |
| `_bias.nii.gz` | `-b` 时输出乘性偏置场 |
| `_restore.nii.gz` | `-B` 时输出校正图 |

`-W/-I/-O/-l/-f/-H/-R` 分别对应 Python 的 `init_iterations/bias_iterations/fixed_iterations/bias_fwhm_mm/init_mrf/mrf/mixel_mrf`；`--pve-steps` 对应 `pve_steps`。`-N` 关闭偏置场更新，仍运行默认四次 HMRF 外循环。`--overwrite` 允许覆盖已有结果；默认遇到已有目标文件即报错。CLI 暂不开放 `mean_field_iterations` 与 `pve_chunk_size`，需用 Python API 修改。

原生 FSL FAST 的同输入命令为：

```bash
fast -n 3 -t 1 -b -B -o results/T1_brain T1_brain.nii.gz
```

本实现固定为三组织、单通道 T1、无先验。FSL 的 T2/PD、多通道、任意类别数、外部 priors、手工初始均值和 `--nopve` 路径没有实现。

## 算法与精度边界

输入先按 FAST 的规则处理负值，再在 `log(T1 + 1)` 域初始化三类混合高斯。HMRF-EM 更新组织均值、方差、后验概率及物理尺度平滑的乘性偏置场；最后在校正后的线性强度中估计纯组织和双组织混合体素的 PVE。FSL 的逐体素原地更新与本实现的同步卷积更新不同，因此**与 FSL 的 PVE 不逐体素一致**。这一算法差异与本次移除 Surfa 的影像包装层是两件事。

此前 10 例的 FSL FAST 对比是**旧 Surfa 读写包装层下的历史算法基准**：GM PVE Pearson、0.5 Dice 和体积比中位数为 0.98488、0.99232、0.99059；完整 GPU 命令时间中位数 12.43 s，原生 FSL CPU `fast -b` 为 305.80 s。该记录不能直接作为本次新包装层的 GPU 性能或 FSL 数值复验。原始对比口径和公开图像见[历史记录](../../validation/fast/README.md)。新旧包装层使用真实同一 T1 的逐体素配对见[本次验证](../../validation/fast/no_surfa_20260928/README.md)。

FAST4 2111.3 的未改动源码保存在 [`upstream_fast4/`](../../src/fnit/fast/upstream_fast4/) 供许可与溯源，不会被编译或导入。它与本改写受 [FSL 6.0 许可](../../licenses/FSL-6.0.txt)约束，仅供非商业使用。

## 直接张量接口

```python
import nibabel as nib
import numpy as np
import torch
from fnit.fast import FASTConfig, segment_t1

input_image = nib.load("T1_brain.nii.gz")  # 已去脑、单帧 T1w
t1_tensor = torch.from_numpy(             # 必需：按 (X, Y, Z) 排列的 T1 张量
    np.asanyarray(input_image.dataobj).astype(np.float32)
)
mask_tensor = None                        # 可选：同尺寸布尔脑掩模；None 表示 T1 > 0
voxel_size = input_image.header.get_zooms()[:3]  # 三个体素轴的物理尺寸，单位 mm
config = FASTConfig()                     # 算法迭代、平滑和 PVE 参数，使用默认值
result = segment_t1(
    t1_tensor=t1_tensor,
    mask_tensor=mask_tensor,
    voxel_size=voxel_size,
    config=config,
)
gm_tensor = result.pve[1]                 # 输出 GM 分数体积，形状 (X, Y, Z)
```

该接口只返回张量，不读取文件、不创建世界坐标信息；常规影像调用使用 `TorchFAST`。
