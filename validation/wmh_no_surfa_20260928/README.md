# WMH-SynthSeg：移除 Surfa 后的真实 FLAIR 核对

[返回功能说明](../../docs/wmh_synthseg/README.md)

本次只替换 WMH-SynthSeg 的影像结果封装与存盘，以及内存输入的 Surfa 类型判断。方向重排、1 mm 重采样、网络和概率/软体积计算没有改动。新结果用仓库内 `Volume` 保存；单例 CLI 在解析 WMH 参数后直接运行该模块，不导入其他影像流水线。

## 输入、命令与输出

输入为 [OpenNeuro ds003592](https://openneuro.org/datasets/ds003592) 的 `sub-04_ses-1_FLAIR.nii.gz` 原始 3D FLAIR，SHA-256 为 `49de51a875ade818485c239d37632dce4aafc042f7bf241188460ff9d989c622`。使用官方 `WMH-SynthSeg_v10_231110.pth`，SHA-256 为 `0ece39dd651357aa95222fc4d45fa32d00f11e763d2583cae3f869989ce35988`。本次选 `crop=True`、CPU 4 线程，并输出病灶概率。

旧版参考使用本仓库改写前的 Surfa 路径（提交 `1239a34`）；新版使用本次改写，两者在 headcw 的同一 Conda 环境中顺序运行。两版 Python API 均调用 `WMHSynthSeg(weights=权重文件, device="cpu", threads=4)(image=输入文件, crop=True, save_lesion_probabilities=True)`，并将标签图和概率图分别保存为 NIfTI 与 MGZ。新版 CLI 另用相同输入执行：

```bash
fnit wmh-synthseg --i sub-04_FLAIR.nii.gz --o sub-04_seg.nii.gz \
  --weights WMH-SynthSeg_v10_231110.pth --device cpu --threads 4 --crop \
  --save_lesion_probabilities --csv_vols sub-04_volumes.csv
```

原版 FreeSurfer 的对应命令为：

```bash
mri_WMHsynthseg --i sub-04_FLAIR.nii.gz --o sub-04_seg.nii.gz \
  --device cpu --threads 8 --crop \
  --save_lesion_probabilities --csv_vols sub-04_volumes.csv
```

`--i` 是唯一 3D 输入；`--o` 产生 33 类标签（WMH 为 77）；`--save_lesion_probabilities` 产生同网格的 WMH 后验概率图；`--csv_vols` 产生各类后验概率之和，单位 mm³。两张图为模型处理后的 RAS/1 mm 网格，尺寸 `192×216×126`，保存数据类型均为 `float32`。API 返回的 `WMHResult` 包含 `segmentation`、可选 `lesion_probability` 和 `{标签编号: 软体积}` 字典。图像对象提供 `.data`、`.affine`、`.geom.vox2world.matrix`、`.geom.voxsize`、`.save(path)`。

## 实测结果

| 检查项 | 新版对旧版 Surfa 路径 | 新版 CLI 对已保存的官方 CPU 输出 |
|---|---:|---:|
| 标签不一致体素 | 0 / 5,225,472 | 0 / 5,225,472 |
| WMH 概率不一致体素 | 0 / 5,225,472 | 0 / 5,225,472 |
| 未写盘仿射/体素尺寸最大绝对差 | 0 mm / 0 mm | 与官方文件头比较口径不同 |
| 已保存 NIfTI 仿射最大绝对差 | 0 mm | 0 mm |
| 33 类软体积最大绝对差 | 0 mm³ | CSV 数值列文本全部相同 |
| NIfTI 标签与概率文件 | 各自 SHA-256 相同 | 数值内容相同 |
| MGZ 标签与概率文件 | 各自 SHA-256 相同 | 本次未与官方 MGZ 比较 |

新旧 API 返回的未写盘仿射矩阵逐元素相同。原版 FreeSurfer 已保存 NIfTI 与新版 CLI 的仿射矩阵也逐元素相同；若将 API 的双精度内存仿射直接与官方 NIfTI 头的单精度仿射比较，最大差为 `2.12×10⁻⁶ mm`，这是写盘精度的比较口径不同。新版 CLI 在拦截 `surfa` 导入的进程中完成，结束时 `sys.modules` 中无 Surfa 模块。

| 实际运行 | 墙钟秒数 | 条件 |
|---|---:|---|
| 旧版 Surfa API | 110.72 | headcw，同环境，CPU 4 线程；加载、推理、保存四张图 |
| 新版 API | 85.52 | headcw，同环境，CPU 4 线程；同一范围 |
| 新版 CLI | 101.11 | headcw，CPU 4 线程；保存两张 NIfTI 和 CSV |
| 官方 CPU CLI（已有 12 例记录中的同例） | 90.54 | gpucw1，CPU 8 线程，FreeSurfer 自带 Python |

旧版与新版 API 的加载/推理/四张图保存分别为 `0.93/108.32/1.47` 秒与 `0.85/83.17/1.51` 秒。这些是各一次顺序运行；节点、线程、保存文件数与 Python 环境并未在四臂之间统一。新旧 API 虽然同环境且范围相同，也只有各一次，不能据此判定稳定提速。当前改写的速度目标是消除 Surfa 依赖；网络仍占主要时间。

已测输入为原始 3D `.nii.gz` 路径，`crop=True`，CPU，保存 NIfTI/MGZ；CLI 还测了 CSV 和禁止 Surfa 导入。尚未在本次迁移中重测 `.nii`、`.mgz` 输入、内存体对象、`crop=False`、`save_lesion_probabilities=False`、CUDA、批量接口或其余 11 例。此前 12 例结果记录在[旧版完整基准](../wmh/README.md)，不能直接当作本次迁移的 12 例验收。没有人工 WMH 真值，本报告只核对实现一致性。

可机读数值见 [report.json](report.json)。影像与 checkpoint 不提交仓库。
