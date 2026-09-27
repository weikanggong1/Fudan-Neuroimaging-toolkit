# SynthSR：从单幅扫描合成 1 mm T1w

[返回首页](../../README.md) · [FreeSurfer 使用说明](https://surfer.nmr.mgh.harvard.edu/fswiki/SynthSR) · [原版源码](https://github.com/freesurfer/freesurfer/blob/dev/mri_synthsr/mri_synthsr) · [权重配置](../WEIGHTS.md)

SynthSR 接受一幅 3D MRI 或 CT，输出标准对比度的 1 mm 等方 MP-RAGE 图像。它结合超分辨率与图像合成，并在合成时填补白质病灶。输入可以是 T1w、T2w、FLAIR 等对比度；原版不要求事先去颅骨、校正偏场或归一化强度。本包将原版的 TensorFlow 网络和推理流程改写为 PyTorch，推理时不调用 FreeSurfer。

## 原版指令与模型

单幅 FLAIR 的原版调用如下。`--i` 读取扫描，`--o` 写合成 T1w，`--threads` 指定 CPU 线程数；安装了可用的 TensorFlow GPU 时，原版可自动使用 GPU，`--cpu` 则强制使用 CPU。

```bash
mri_synthsr --i case_FLAIR.nii.gz --o case_synthsr.nii.gz --threads 4
```

原版还接受 `--ct`（将以 Hounsfield 单位保存的 CT 截到 0–80）、`--lowfield`（低场单输入模型）、`--v1`（2021 年模型）、`--disable_flipping`（关闭翻转测试增强）、`--disable_sharpening`（关闭末端锐化）和 `--model /path/to/model.h5`（指定权重）。同时给出 `--v1` 和 `--lowfield` 时，原版先选择 `--v1`。独立的双输入 `mri_synthsr_hyperfine --t1 ... --t2 ...` 使用另一模型，不属于这里的单输入接口。

| 选择 | 官方权重 | 本包配置命令 |
|---|---|---|
| 默认，2023 年通用 v2 | `synthsr_v20_230130.h5` | `python tools/setup_weights.py --model synthsr` |
| 低场单输入 v2 | `synthsr_lowfield_v20_230130.h5` | `python tools/setup_weights.py --model synthsr-lowfield` |
| 通用 v1 | `synthsr_v10_210712.h5` | `python tools/setup_weights.py --model synthsr-v1` |

这些权重来自 [FreeSurfer 官方源码目录](https://github.com/freesurfer/freesurfer/tree/dev/mri_synthsr)及其 git-annex 存储。配置脚本下载后核对文件大小和 SHA-256，并记录权重目录；仓库和 wheel 不包含权重。推理只按显式路径、`FNIT_WEIGHTS`、配置目录和用户缓存查找，不会自动读取 `$FREESURFER_HOME/models/`，运行时也不会联网。下载链接、校验值和查找顺序见[权重说明](../WEIGHTS.md)。

## Python 输入与输出

```python
from fnit import SynthSR

sr = SynthSR(
    weights=None,  # 权重：按 FNIT 配置顺序查找官方 HDF5
    device="cuda:0",  # 设备：第一张可见 CUDA GPU
    lowfield=False,  # 模型：不使用低场专用模型
    v1=False,  # 模型：使用默认 v2，而非 2021 年 v1
    threads=4,  # CPU 线程：用于预处理和后处理
)
result = sr(
    image="case_FLAIR.nii.gz",  # 输入：单幅 3D MRI 或 CT
    ct=False,  # 输入类型：按 MRI 强度处理
    disable_flipping=False,  # 推理：保留左右翻转测试增强
    disable_sharpening=False,  # 后处理：保留末端锐化
)
result.image.save(path="case_synthsr.nii.gz")  # 输出路径：1 mm 合成 T1w
print(result.image.data.shape, result.image.affine)
```

构造 `SynthSR` 时加载一次权重；每次调用接收一幅影像。`device="cpu"` 是 Python 默认值；要用 GPU，显式指定 `"cuda:0"` 等设备。`weights` 可以是 `.h5` 文件或包含所选官方文件的目录；省略时按[权重配置](../WEIGHTS.md)自动查找。`threads` 控制 PyTorch CPU 线程，省略时保留当前设置。

| 本包接口 | 输入或返回值 | 原版对应 |
|---|---|---|
| `SynthSR(weights=None, device="cpu", lowfield=False, v1=False, threads=None)` | 选择设备、权重和单输入模型；`v1=True` 优先于 `lowfield=True` | `--model`、`--cpu`、`--lowfield`、`--v1`、`--threads` |
| `sr(image, ct=False, disable_flipping=False, disable_sharpening=False)` | `image` 是单幅 `.nii`、`.nii.gz`、`.mgz`、`.npz` 路径或 `nibabel.spatialimages.SpatialImage` | `--i`、`--ct`、`--disable_flipping`、`--disable_sharpening` |
| `result.image.data` | 3D `numpy.ndarray`，`uint8`，是 NIfTI/MGZ 写盘前的量化数值 | `--o` 输出的体素数组 |
| `result.image.affine` | 4×4 RAS 仿射矩阵，描述 1 mm 输出网格 | `--o` 输出的几何信息 |
| `result.image.save(path)` | 写 `.nii`、`.nii.gz`、`.mgz` 或 `.npz` | `--o` 指定输出路径 |

原版先按输入 affine 重采样到 1 mm，再将体素轴对齐 RAS，居中补到 32 的倍数。网络是单通道、五层 3D U-Net。默认对左右翻转后的图像再推理一次并平均；预测截到 0–128，裁掉填充，再做锐化：原图加上原图与高斯模糊图之差，高斯标准差为 1.5 体素。最后转回输入的**轴方向**。因此输出与输入共享物理空间，但尺寸通常不同：它是 1 mm 网格，并未重采样回原始体素尺寸。写盘前乘以 2、截到 0–255、转换为 `uint8`。

`.npz` 是原版的特例：`save()` 将锐化后的浮点数组写在 `vol_data` 字段，**不执行 NIfTI/MGZ 的末端乘 2 和 `uint8` 转换**。因此 `result.image.data` 对应 NIfTI/MGZ 的量化数值；读 `.npz` 应使用 `numpy.load(path)["vol_data"]`，两者的数值尺度不同。原版及本包均通过 `nibabel.save(Nifti1Image(...))` 写 `.mgz`；用 nibabel 重新读入时，该格式报告的存储 dtype 为 `float32`，体素数值仍为量化后的 0–255。原版对常规输入使用单幅 3D 数据；当文件带多个通道时只取第一个通道。

## 单例命令行

```bash
fnit synthsr --i case_FLAIR.nii.gz --o case_synthsr.nii.gz \
  --device cuda:0 --threads 4
```

这条命令读取 `case_FLAIR.nii.gz`，在 `cuda:0` 上用通用 v2 模型合成图像，并把 1 mm `uint8` T1w 写到 `case_synthsr.nii.gz`。本包的 `--device` 可选具体 GPU，原版没有对应的 GPU 编号参数；原版自动选择可用的 TensorFlow GPU。`--cpu` 可覆盖 `--device` 强制 CPU；`--weights` 或同义的 `--model` 指定本地权重。其余模型和处理开关与上表同名。单幅输入的 `--o` 也可指定目录，输出文件名自动加 `_synthsr`。

## 当前源码对照验证

2026-09-27 用当前默认 TF32 和 Nibabel I/O 重跑 12 幅真实临床 T1w。候选推理没有调用 FreeSurfer；同一病例的 FreeSurfer 8.2.0-1 CPU/CUDA 输出作为固定参考。当前 SynthSR 源码树 SHA-256 为 `7b5bc19e1afa806fe8698ea70b6358bacaab23b19877d20d21d2e7c5f3560543`。

12/12 例的 shape、`uint8` dtype 和数值 affine 均一致，affine 最大差为 0。当前 GPU 对原版 CPU 的平均 MAE 为 0.02314 灰度级、平均 NRMSE 为 0.001580，最低完全相同体素比例为 96.3994%，最大差为 9。当前 CPU 单例对原版 CPU 的完全相同体素比例为 99.99497%，最大差为 1。默认 TF32 会改变更多靠近量化边界的体素，因此当前 GPU 结果不能写成逐体素等价。

| 运行 | 完整单例命令时间 |
|---|---:|
| FreeSurfer 原版 CPU，固定同批 12 例参考 | 中位数 103.60 s |
| FreeSurfer 原版 CUDA，固定同批 12 例参考 | 中位数 53.27 s |
| FNIT 当前源码 H100 GPU，12 例重跑 | 中位数 14.45 s [13.05–16.88] |
| FNIT 当前源码 CPU，1 例 | 31.36 s |

原版和当前候选来自不同运行时段，表中不计算稳定加速倍数。独占 GPU 单例的 Torch 峰值 allocated 13,780 MiB、reserved 19,074 MiB。逐例数值、命令、输出契约与源码哈希见[验证页](../../validation/synthsr/README.md)和[机器报告](../../validation/synthsr/report.public.json)。

### 当前公开 FLAIR 示意图

下图使用仓库公开 `sub-04` FLAIR，并用当前默认 TF32 重新生成 FNIT 一列。临床 12 例统计与这幅公开示意图不是同一数据集。公开图两幅输出的 shape、`uint8` 和 affine 一致，完全相同体素比例为 97.8558%，MAE 为 0.02145 灰度级，最大差为 2。

![公开 FLAIR、FreeSurfer SynthSR 与当前 FNIT SynthSR](figures/synthsr_flair_comparison.png)
