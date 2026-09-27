# WMH-SynthSeg：脑结构与白质高信号分割

[返回首页](../../README.md) · [官方说明](https://surfer.nmr.mgh.harvard.edu/fswiki/WMH-SynthSeg) · [官方源码](https://github.com/freesurfer/freesurfer/tree/dev/mri_WMHsynthseg/WMHSynthSeg)

WMH-SynthSeg 同时分割脑结构和白质高信号（WMH，FreeSurfer 标签 **77**），支持 T1w、FLAIR 等对比度。**FreeSurfer 原版已使用 PyTorch**。本包基于已验证的源码与官方 `WMH-SynthSeg_v10_231110.pth`，提供独立安装、单被试 Python 和单被试命令行调用；推理不调用 FreeSurfer 程序，也不需要导入 Surfa。

## 原版源码流程与输出几何

参考源为 gpucw1 安装的 FreeSurfer 8.2.0-1：`bin/mri_WMHsynthseg` 启动 `python/packages/WMHSynthSeg/inference.py`，模型定义来自同目录的 `unet3d`。输入是单幅 3D `.nii`、`.nii.gz` 或 `.mgz`。模型把影像重排到 RAS 方向，用最大值归一化，再重采样到 **1 mm 等方体素**。网络是五层 3D U-Net，输入单通道、输出 39 通道；前 33 通道用于标签后验概率。它还对第一空间轴翻转后的输入再预测一次，交换左右标签通道后平均两个概率图。

`crop=False` 时在完整重采样视野上推理，网络输入各轴填充到 32 的倍数。`crop=True` 时先在 192×224×192 的区域进行定位，再围绕估计的脑中心裁出不超过该大小的区域用于正式预测。原版官网指出 GPU 运行需要 `--crop`；裁剪可能移除视野边缘。**输出是模型处理后的 RAS/1 mm 网格，通常不等于输入网格**。分割体素值来自 33 个 FreeSurfer 标签；另可保存标签 77 的浮点概率图。CSV 体积来自各标签后验概率在 1 mm 网格上的求和，单位 mm³，因此通常不等于硬分割中各标签的体素计数。

原版 CLI 的单例指令：

```bash
mri_WMHsynthseg --i case_FLAIR.nii.gz --o case_seg.nii.gz \
  --device cpu --threads 8 --crop \
  --save_lesion_probabilities --csv_vols case_volumes.csv
```

`--i` 是唯一输入影像；`--o` 是硬分割图；`--crop` 启用上述两遍定位/裁剪；`--save_lesion_probabilities` 另写 `case_seg.lesion_probs.nii.gz`；`--csv_vols` 汇总各结构体积；`--device` 与 `--threads` 控制 PyTorch。本页只对应原版的单文件输入和单文件输出。原版 CSV 的第一列表头为 `Input-file`，实际写入的是**输出分割路径**。原版 3D 流程经过验证；安装源码中的 4D 均值分支存在参数错误，不在本包支持范围内。

## 本包单例 Python 与命令行

官方权重无需随 GitHub 仓库下载；用专属脚本从 FreeSurfer 官方地址获取并自动记录位置：

```bash
python tools/setup_weights.py --model wmh-synthseg
```

Python 模型构造一次即可复用；输入可为 3D `.nii`、`.nii.gz`、`.mgz` 路径。已有 `surfa.Volume` 内存对象仍可传入；WMH-SynthSeg 推理本身不会导入 Surfa，仓库其他模块的依赖将在各自迁移后处理。返回 `WMHResult`。`crop` 和 `save_lesion_probabilities` 默认均为 `False`，下面显式开启二者以适应 GPU 并保存概率图。

```python
from fnit import WMHSynthSeg

weights_path = None                  # 权重位置；None 从已配置权重目录查找
image_path = "case_FLAIR.nii.gz"      # 输入：单幅 3D FLAIR
model = WMHSynthSeg(weights=weights_path, device="cuda:0", threads=4)
result = model(image=image_path, crop=True, save_lesion_probabilities=True)
result.segmentation.save(path="case_seg.nii.gz")  # 输出：33 类标签图
result.lesion_probability.save(path="case_seg.lesion_probs.nii.gz")  # 输出：WMH 概率图
print(result.volumes_mm3)             # 输出：标签编号到软体积 mm³ 的字典
```

| 参数或字段 | 类型、含义 | 原版对应 |
|---|---|---|
| `WMHSynthSeg(weights=None, device="cpu", threads=None)` | 模型构造时加载一次官方 `.pth`；`weights` 可为文件或目录；`device` 选 CPU/GPU | `--device`、`--threads`；原版从 `$FREESURFER_HOME/models` 加载固定文件 |
| `model(image, crop=False, save_lesion_probabilities=False)` | 单幅 3D 影像；`crop=True` 两遍定位并限制推理区域 | `--i`、`--crop`、`--save_lesion_probabilities` |
| `result.segmentation` | 仓库内 `Volume`，33 类整数值标签；WMH 为 77 | `--o` 指定的图像 |
| `result.lesion_probability` | 请求时为仓库内 `Volume`，否则为 `None`；体素值为 WMH 后验概率 | `--save_lesion_probabilities` 额外写出的 `.lesion_probs` 图 |
| `result.volumes_mm3` | `{标签编号: 软体积}` 字典，单位 mm³，包括背景 0 | `--csv_vols` 的各标签体积列；CSV 不输出背景列 |

两幅图像结果都使用原版同样的处理后网格，原图的坐标变换保留在输出仿射矩阵中；不保证与输入数组同形状。仓库内 `Volume` 提供 `.data`、`.affine`、`.shape`、`.geom.vox2world.matrix`、`.geom.voxsize` 和 `.save(path)`。调用者通过 `.save(path)` 写出影像；Python 字典若需 CSV，可用下面的 CLI 直接生成与原版列名相同的表。原版 `Intracranial-volume` 列是非背景软体积之和，`Input-file` 列实际记录**输出分割路径**。

对应的新命令为：

```bash
fnit wmh-synthseg --i case_FLAIR.nii.gz --o case_seg.nii.gz \
  --device cuda:0 --threads 4 --crop \
  --save_lesion_probabilities --csv_vols case_volumes.csv
```

这条命令的 `--i` 读取单幅 FLAIR，`--o` 写标签图；`--crop` 将正式预测限制在脑周围的区域；`--save_lesion_probabilities` 额外写 `case_seg.lesion_probs.nii.gz`；`--csv_vols` 写软体积 CSV；`--device` 指定 GPU，`--threads` 指定 Torch 的 CPU 线程。本包 CLI 每次处理单幅影像并创建输出父目录。

## 验证边界

本次验收固定一份真实 FLAIR、同一官方权重、CPU 和 `--crop`，比较标签、病灶概率、输出几何、软体积和存盘字节；并用已保存的 FreeSurfer CPU 输出核对新版 CLI。原版在 gpucw1 的 FreeSurfer Python 环境运行，新旧仓库 API 在 headcw 的同一 Conda 环境配对运行，时间不作为同硬件官方加速比。公开病例没有人工 WMH 标注，本报告不评价病灶检测准确率。

移除 Surfa 后，使用同一份真实 `sub-04` FLAIR 复核了路径输入、Python API 和 CLI：新旧版本的标签、概率、仿射、软体积相同，NIfTI/MGZ 输出逐字节一致；新 CLI 的标签、概率、已保存仿射和 CSV 数值列也与已保存的官方 CPU 输出一致。实测时间、输入哈希、命令和未测模式见[无 Surfa 迁移记录](../../validation/wmh_no_surfa_20260928/README.md)。此前 12 例基准针对旧版 Surfa 输出路径，保留为[历史验证](../../validation/wmh/README.md)，不代替本次改写的多例验收。

### 原版与本包示意图

下图使用仓库公开的 `sub-04` FLAIR。左列为输入，中、右列分别叠加 FreeSurfer
原版和旧版仓库输出的标签 77（红色）；两次运行均使用官方权重、CUDA 和 `--crop`。当前无 Surfa 版本的数值结论以上述新验证为准。

![公开 FLAIR、FreeSurfer WMH-SynthSeg 与本包 WMH-SynthSeg](../figures/wmh_synthseg_comparison.png)

该图只显示同一物理位置的二维切面。完整三维标签、病灶概率、软体积及仿射矩阵
用于数值比较；生成命令和公开数据来源见[图示记录](../figures/README.md)和
[FLAIR 示例](../../examples/WMH.md)。
