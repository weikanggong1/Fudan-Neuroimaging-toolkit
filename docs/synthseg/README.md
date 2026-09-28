# 独立 33 类 SynthSeg

[返回首页](../../README.md) · [源码目录](../../src/fnit/synthseg_parc/) · [权重](../WEIGHTS.md) · [当前验证报告](../../validation/synthseg/report.public.json)

`SynthSeg` 使用 FreeSurfer 8.2 的非 robust、非 parcellated SynthSeg 2.0 模型，从单幅 T1 生成 33 类结构标签和各结构软体积。它与 WMH-SynthSeg 是不同模型；本入口不输出 WMH 标签，也不生成皮层分区。推理使用 PyTorch，不需要安装 FreeSurfer、FSL 或 TensorFlow。

## 下载模型

在仓库根目录运行，或安装后用 `fnit-setup-weights` 代替 `python tools/setup_weights.py`。配置脚本下载固定的 `.h5` 和三个 `.npy` 文件，逐个核对大小与 SHA-256，然后记录权重目录。推理本身不联网。

```bash
python tools/setup_weights.py --model synthseg --dest /path/to/weights
```

这个组仅需四个文件，总计 53,087,016 字节，不需要下载 790 MB 的 WMH-SynthSeg 权重。已有模型时可用 `--verify-only` 检查。完整文件名、官方地址及哈希见[权重说明](../WEIGHTS.md)。

## Python API

```python
from fnit import SynthSeg

model = SynthSeg(
    weights=None,  # 权重：按 FNIT 配置顺序查找四个官方模型文件
    device="cuda:0",  # 设备：第一张可见 CUDA GPU
    threads=4,  # CPU 线程：用于预处理和后处理
)
result = model(
    image="sub-01_T1w.nii.gz",  # 输入：单幅 3D T1w
    keep_geometry=False,  # 输出网格：保留预处理后的 RAS、约 1 mm 网格
    color_lut=None,  # 色表：不附加外部 FreeSurfer LUT
)
result.segmentation.save(path="sub-01_synthseg.nii.gz")  # 输出路径：33 类硬分割图
result.write_volumes_csv(
    source="sub-01_T1w.nii.gz",  # CSV 标识：原始输入文件名
    path="sub-01_synthseg.vol.csv",  # 输出：软体积 CSV 路径
)
print(result.total_intracranial_mm3, result.volumes_mm3)
```

`SynthSeg(weights=None, device="cpu", threads=None)` 在构造时加载一次模型；每次调用接收一幅图像。`weights` 可传包含四个文件的目录，或直接传 `synthseg_2.0.h5` 路径；三个 `.npy` 必须与这份 `.h5` 位于同一目录。省略 `weights` 时先查 `FNIT_WEIGHTS`、配置脚本记录的目录，再查默认缓存。输入是单幅 3D `.nii`、`.nii.gz`、`.mgz` T1 路径或 `nibabel.spatialimages.SpatialImage`。

`result.segmentation` 是 `FNITNifti1Image`（`nibabel.Nifti1Image` 子类），默认位于 SynthSeg 预处理后的 RAS 方向、约 1 mm 网格。标签编号和存储类型均为 `int32`；NIfTI qform code 为 0，sform code 为 2。`model(image, keep_geometry=True)` 会将标签以最近邻法重采样到输入网格，并保持相同 dtype 与 form-code 契约。`color_lut="/path/to/FreeSurferColorLUT.txt"` 可选地在返回图像的 `extra["color_lut"]` 中记录色表路径；默认不读取 FreeSurfer 文件。

`result.volumes_mm3` 是 `{前景标签编号: 软体积}`，`result.total_intracranial_mm3` 是所有前景软体积之和，后验概率先恢复到输入方向，再按原版 NumPy float32 顺序求和并保留三位小数；`result.label_names` 对应 32 个前景结构名。CSV 列顺序、总量和近似并列标签规则与本仓库 GPU recon-all 的 `mri_synthseg` 入口相同。`result.near_tie_voxels` 记录近似并列规则相对于普通 `argmax` 更改的体素数。

## 原版命令与参数对应

本模块对应 FreeSurfer 8.2 `mri_synthseg` 的单幅、SynthSeg 2.0、非 robust、非
parcellated 路径。原版单例命令为：

```bash
mri_synthseg --i sub-01_T1w.nii.gz --o sub-01_synthseg.nii.gz \
  --vol sub-01_synthseg.vol.csv --threads 4
```

本包执行同一输入输出角色的命令为：

```bash
fnit synthseg --i sub-01_T1w.nii.gz --o sub-01_synthseg.nii.gz \
  --csv-vols sub-01_synthseg.vol.csv --device cuda:0 --threads 4
```

| 本包参数 | 原版参数 | 输入或输出 |
|---|---|---|
| `--i` | `--i` | 一幅 3D T1；本包接受 `.nii`、`.nii.gz` 或 `.mgz` 文件 |
| `--o` | `--o` | 33 类硬分割图；默认位于预处理后的 RAS、约 1 mm 网格 |
| `--csv-vols` | `--vol` | 前景结构软体积和 total intracranial volume，单位 mm³ |
| `--weights` | `--model` | 官方 `synthseg_2.0.h5`；本包还从同目录读取三份标签、名称和拓扑 `.npy` |
| `--device cpu` | `--cpu` | 强制 CPU；本包也可用 `--device cuda:N` 选择具体 GPU |
| `--threads` | `--threads` | PyTorch CPU 线程数；本包 CLI 默认 4，原版默认 1 |
| `--keep-geometry` | `--keepgeom` | 以最近邻法把硬标签重采样回输入网格 |
| `--color-lut` | `--addctab` / `--noaddctab` | 本包只在显式提供 FreeSurfer LUT 时附加色表；原版默认附加色表 |

原版的目录输入、robust、QC、posterior、CT、Photo-SynthSeg 和其他模型路径没有在此 33 类接口中实现。皮层分区请使用同一 CLI 的 `--parc` 或 [SynthSeg+ Python API](../synthseg_plus/README.md)。

## 命令行

```bash
fnit synthseg --i sub-01_T1w.nii.gz --o sub-01_synthseg.nii.gz \
  --csv-vols sub-01_synthseg.vol.csv --device cuda:0 --threads 4
```

可选参数为 `--weights /path/to/weights`、`--keep-geometry` 和 `--color-lut /path/to/FreeSurferColorLUT.txt`。命令行和 Python 每次均处理一幅图像。独立入口不依赖 recon-all 的原生运行包或个人 license。

## 与 FreeSurfer 8.2 的当前对照

2026-09-27 在三幅仓库公开、去面容 T1w 上重跑当前源码和 FreeSurfer 8.2.0-1 `mri_synthseg --noaddctab`。候选推理没有调用 FreeSurfer。该次 33 类推理所用源码树 SHA-256 为 `39fa204aea7674ad7c6e09652d0f8750dd2872b1b78799812ab0d71b5b6c8972`。

本轮同时核对文件头：默认输出和 `keep_geometry=True` 均写 `int32`，qform code 为 0，sform code 为 2，与原版相同。一幅真实 T1w 的 `keep_geometry=True` 输出与输入 shape、affine 完全一致。 针对性测试为 `2 passed`；WMH-SynthSeg、SynthSR 和 TorchFAST 的最小跨模块回归为 `28 passed, 4 skipped`。

| 指标 | 三例结果 |
|---|---:|
| shape / 数值 affine / dtype / qform / sform | 3/3 一致 |
| 最低逐体素标签一致率 | 0.99998817 |
| 全部前景标签最低 Dice | 0.99902629 |
| CSV 最大绝对误差 | 104.06 mm³ |
| FNIT H100 完整命令 | 中位数 9.48 s [8.84–10.13] |

FNIT CPU 单例为 55.57 s；与原版相比只有 1 个标签体素不同，CSV 最大差 0.24 mm³。原版三例 CPU 参考任务所在时段有重叠，中位数 304.23 s，因此不计算稳定加速倍数。GPU 单例 Torch 峰值 allocated 19,769 MiB、reserved 22,820 MiB。

完整逐例标签 Dice、CSV、shape/affine/dtype、输出哈希、实际命令和边界见[验证页](../../validation/synthseg/README.md)与[机器报告](../../validation/synthseg/report.public.json)。三例没有人工结构分割真值，这些数值只衡量对参考实现的复现程度。

下图使用本轮公开 `sub-02` 重跑。中间两列显示同网格标签，最后一列标出不同体素；该例逐体素一致率为 0.99998881。

![公开 T1w、FreeSurfer SynthSeg 与当前 FNIT SynthSeg](figures/synthseg_comparison.png)
