# SynthStrip：脑提取

[返回首页](../../README.md) · [源码目录](../../src/fnit/synthstrip/) · [权重](../WEIGHTS.md)

SynthStrip 从脑影像预测有符号距离场，生成脑掩膜和去除背景的影像。官方模型已使用 PyTorch。本模块沿用其网络、权重和影像处理流程，提供可重复调用的 Python API；模型未重新训练。

参考版本为 **FreeSurfer 8.2.0**，build `freesurfer-linux-centos7_x86_64-8.2.0-20260314-d932c45`。分析对象是 `$FREESURFER_HOME/python/scripts/mri_synthstrip`，不是 `bin/` 下的包装脚本。脚本 SHA-256 为 `bbc2ff8f8779862039401b05d5cd6039fb4f3583e0032a793ac9adb3f4521590`，全部来源信息见 [provenance.json](../provenance.json)。

## Python API

```python
from pathlib import Path
from fnit import SynthStrip

out = Path("results")
out.mkdir(exist_ok=True)
extract = SynthStrip(
    weights="/path/to/weights",  # 权重：官方 PT 文件或权重目录
    device="cuda:0",  # 设备：第一张可见 CUDA GPU
    no_csf=False,  # 模型：保留 CSF 的默认模型
    threads=4,  # CPU 线程：预处理与后处理使用 4 线程
)
result = extract(
    image="subject_T1w.nii.gz",  # 输入：单幅 3D T1w，也可传 nibabel 空间影像
    border=1,  # 掩膜阈值：距离场小于 1 mm 视为脑内
    fill=0,  # 输出背景：脑掩膜外填 0
)
result.image.save(path=out / "subject_brain.nii.gz")  # 输出路径：脑提取后的 T1w
result.mask.save(path=out / "subject_mask.nii.gz")  # 输出路径：二值脑掩膜
result.distance.save(path=out / "subject_sdt.nii.gz")  # 输出路径：有符号距离场
# 可继续 extract(image="another_T1w.nii.gz")，复用已加载模型。
```

`from fnit.synthstrip import SynthStrip, StripResult` 是等价的功能模块入口。

### 模型构造

`SynthStrip(weights=None, device="cpu", no_csf=False, threads=None)`：

| 参数 | 含义 |
|---|---|
| `weights` | 官方 PT 文件或所在目录；省略时使用[统一查找顺序](../ARCHITECTURE.md#公开-python-api) |
| `device` | `"cpu"` 或 `"cuda:N"`；CUDA 编号遵循 `CUDA_VISIBLE_DEVICES` |
| `no_csf` | 为 `True` 时使用排除 CSF 的官方权重 |
| `threads` | 当前进程的 Torch 线程数；`None` 保留当前值 |

模型进入 eval 模式并使用 float32 张量；CUDA 构造默认允许 TF32 matmul 和 cuDNN 内核，不使用 float16 或 bfloat16。重复使用实例可避免重复加载权重。

### 单次调用与结果

`extract(image, border=1, fill=None) -> StripResult`：

| 参数或字段 | 含义 |
|---|---|
| `image` 输入 | 文件路径或 `nibabel.spatialimages.SpatialImage`；支持 3D 和逐帧处理的 4D |
| `border` | SDT 阈值，单位 mm，默认 1 |
| `fill` | 掩膜外的强度；省略时为 `min(image.min(), 0)` |
| `result.image` | 掩膜外已填充的影像，保留原网格和几何 |
| `result.mask` | 二值脑掩膜 |
| `result.distance` | 有符号距离场，单位 mm |

三个返回字段均为 `FNITNifti1Image`（`nibabel.Nifti1Image` 子类），可用 `.save(path)` 保存；也可直接传给 `nibabel.save`。调用不会修改输入对象。直接使用 Python 保存时，由调用者准备输出父目录。

## 命令行

```bash
fnit synthstrip -i subject_T1w.nii.gz \
  -o results/subject_brain.nii.gz -m results/subject_mask.nii.gz \
  -d results/subject_sdt.nii.gz --weights /path/to/weights --device cuda:0
```

对应的 FreeSurfer 原版指令为：

```bash
mri_synthstrip -i subject_T1w.nii.gz \
  -o results/subject_brain.nii.gz -m results/subject_mask.nii.gz \
  -d results/subject_sdt.nii.gz
```

两条命令的 `-o`、`-m`、`-d` 分别保存脑图、掩膜和毫米单位的距离场；Python 返回字段依次为 `result.image`、`result.mask`、`result.distance`。原版 `-g` 选择可见 GPU，统一 CLI 用 `--device cuda:N` 指定设备；原版 `-t` 和 `--model` 分别对应统一 CLI 的 `-j` 与 `--weights`。

| CLI 参数 | 对应 API / 行为 |
|---|---|
| `-i`, `--image` | 输入影像，必需 |
| `-o`, `--out` | `result.image` |
| `-m`, `--mask` | `result.mask` |
| `-d`, `--sdt` | `result.distance` |
| `--weights` | PT 文件或目录 |
| `--device` | 默认 `cpu` |
| `--no-csf` | `no_csf=True` |
| `-b`, `--border` | 默认 1 mm |
| `-f`, `--fill` | 可选背景强度 |
| `-j`, `--threads` | Torch 线程数，统一 CLI 默认 4 |

至少指定一个输出。统一 CLI 会创建输出父目录。

## 权重与 CPU/GPU 分工

| 文件 | 用途 |
|---|---|
| `synthstrip.1.pt` | 默认脑提取 |
| `synthstrip.nocsf.1.pt` | `no_csf=True` |

通过 `checkpoint["model_state_dict"]` 严格加载，无权重转换或精度压缩。下载、许可和 SHA-256 见 [WEIGHTS.md](../WEIGHTS.md)。推理使用本地权重，不调用 FreeSurfer 命令。

U-Net、1 mm 最近邻重采样和 SDT 回采样在所选 PyTorch 设备执行。nibabel 负责影像读写；离散轴重排、裁剪、归一化在 CPU 完成，SDT 扩展和连通域使用 SciPy。4D 输入逐帧处理，每次网络推理一个 frame。

## 源码组织

| 文件 | 责任 |
|---|---|
| [model.py](../../src/fnit/synthstrip/model.py) | `ConvBlock`、`StripModel`，保留官方网络和参数名 |
| [pipeline.py](../../src/fnit/synthstrip/pipeline.py) | `SynthStrip`、`StripResult` 和 `extend_sdt` |
| [__init__.py](../../src/fnit/synthstrip/__init__.py) | 功能公开导出 |

## 网络

- 输入为 `[N, 1, X, Y, Z]` float32，默认 API 每次推理一个 frame。
- 3D U-Net 共 7 个分辨率层级、6 次 `MaxPool3d(2)`。
- 编码器每级两次 `Conv3d(kernel=3, stride=1, padding=1)`；通道依次为
  16、32、64、64、64、64。
- 解码器每级两次卷积，然后最近邻 2 倍上采样，与相应编码器特征在通道维拼接。
  最后保留两层 16 通道卷积，再输出一个通道。
- 中间激活均为 `LeakyReLU(0.2)`；最后一层不加激活。没有 BatchNorm 或 Dropout。
- 默认网络共 **2,566,561 个可训练参数**、27 个卷积层；预测的是有符号距离，
  不直接输出 softmax 分割。保留原网络可选 `return_mask=True` 架构入口，官方 SDT
  权重和高级 API 使用默认 `False`。
- `encoder.*`、`decoder.*`、`remaining.*` 参数名称与官方权重一致。

## 每帧处理

1. 先通过离散轴重排转为 LIA，再按 NIfTI `pixdim` 决定 1 mm 网格尺寸，使用最近邻重采样和 float32 数据。保留官方 `shape / 2` 定义的视野中心；网格和中间 affine 使用 float64，采样坐标使用逐项 float32 运算，不受 TF32 矩阵乘法影响。
2. 按强度 `> 0` 的包围盒裁剪；各轴尺寸向上取 64 的倍数，再限制在 192–320；居中裁剪或补零。奇数个体素需裁掉时，多裁的一个体素位于高索引端。
   此上限是官方行为，大视野超过 320 mm 的部分可能被裁掉。
3. 减去全图最小值，除以第 99 百分位数，限制到 `[0,1]`。这里的百分位数包含零值。
4. U-Net 预测窄带 signed distance transform (SDT)。`border` 大于窄带范围时，按官方
   `extend_sdt` 重算外部距离；负值内部预测保持不变。
5. SDT 线性重采样回原始体素空间，视野外填 100。以 `SDT < border` 得到 mask，
   保留最大连通域并填洞。默认 `border=1` mm；`--no-csf` 切换另一份官方权重。
6. 原影像 mask 外填 `min(image.min(), 0)`，也可显式提供 `fill`。4D 输入逐帧执行，
   再恢复 frame 维；原数据几何、体素格和有效区强度保留。

算法沿用官方输入范围：非有限强度、纯常数图像等没有增设修复规则。若归一化分母
为零，不能将所得输出视为有效提取结果。

## 功能差异与验证

官方脚本集成参数解析和执行流程；本包将同一网络和影像处理拆成可导入接口。生产运行使用 nibabel、PyTorch 和 SciPy，不调用 FreeSurfer，也不依赖 Surfa。统一 CLI 为 `fnit synthstrip`。

本次在 gpucw1 的 H100 上，以一例真实受试者的原始 SBRef 和存档 T1 作固定输入控制，使用同一官方 `synthstrip.1.pt`。存档 T1 的转换保留体素和 affine，但尚未确认是扫描仪直接输出的原始 T1。候选 `pipeline.py` SHA-256 为 `41304bafc412bcc914d76a6cbbd29550173df4419c1aa3a54d31d95739bda95d`。官方几何参照使用 Surfa 0.6.3；独立脑掩膜参照来自未修改的 FreeSurfer 8.2 `mri_synthstrip`。

修复了通用 `conform` 与官方视野中心定义、NIfTI `pixdim` 网格尺寸以及边界采样的差异。两幅真实输入的 1 mm LIA 数组和网络归一化输入均逐元素相同。

| 当前真实输入控制 | SBRef | T1 |
|---|---:|---:|
| 1 mm LIA 数组是否逐元素一致 | 是 | 是 |
| 1 mm LIA affine 最大差异（mm） | 0 | 0 |
| 归一化网络输入是否逐元素一致 | 是 | 是 |
| 同一网络预测，经两种实现回采样后 SDT RMSE（mm） | 0 | 2.91 × 10⁻⁹ |
| 同一网络预测，经两种实现回采样后 mask 是否逐元素一致 | 是 | 是 |
| 与原程序独立推理所得 mask Dice | 0.999995 | 0.999991 |
| 与原程序独立推理所得 mask 不同体素数 | 1 | 25 |
| 控制运行时间（s） | 7.61 | 4.83 |

上表时间从模型构造前开始，到网络预测、双实现回采样和 mask 比较后停止；不含 Python 启动、输入 conform/归一化或最终写盘。它是共享 GPU 上的控制计时，不是双方完整 CLI 的速度比较。候选保留默认 TF32；重复控制中 SBRef 有 1–2 个、T1 有 19–25 个阈值附近体素不同。同一次预测的回采样 mask 始终一致，独立 GPU 推理的二值输出仍存在微小数值差异。

完整输入、权重和源码 SHA-256、环境以及重复结果见 [机器可读报告](../../validation/fmri/synthstrip_geometry_control.public.json)。体积流程的模板空间脑图见 [fMRI 完整对照](../../validation/fmri/matched_native.md)。原始头部图像留在服务器。

## 测试与复现

几何回归测试覆盖视野中心、原 header 体素尺寸、正值包围盒、奇数裁剪、最近邻半体素及线性采样末端边界。本次以下测试共 28 项通过：

```bash
# 检查 SynthStrip 几何、公开接口和默认 TF32 设置。
python -m pytest tests/synthstrip tests/test_tf32_defaults.py

# 在装有官方 Surfa 的验证环境中，对同一真实输入和权重比较几何及回采样。
# 该验证脚本使用 FNIT 推理，不从生产接口启动 FreeSurfer。
python validation/fmri/compare_synthstrip_geometry.py \
  --manifest /path/to/private_manifest.json \
  --private-output /path/to/private_controls \
  --report /path/to/public_summary.json \
  --device cuda:1
```

私有清单格式如下。`weights` 指向官方权重文件；`original_program` 用于记录原脚本的 SHA-256；每个 `input` 为待验证影像，`original_mask` 为事先运行未修改的官方程序得到的脑掩膜。`--private-output` 保存控制生成的 mask；`--report` 只写匿名汇总和哈希。

```json
{
  "weights": "/path/to/synthstrip.1.pt",
  "original_program": "/path/to/freesurfer/python/scripts/mri_synthstrip",
  "cases": {
    "sbref": {
      "input": "/path/to/SBRef.nii.gz",
      "original_mask": "/path/to/original_SBRef_mask.nii.gz"
    },
    "t1w": {
      "input": "/path/to/T1w.nii.gz",
      "original_mask": "/path/to/original_T1w_mask.nii.gz"
    }
  }
}
```

## Reference

- 参考文献：Hoopes et al., *SynthStrip: Skull-Stripping for Any Brain Image*, NeuroImage (2022), [doi:10.1016/j.neuroimage.2022.119474](https://doi.org/10.1016/j.neuroimage.2022.119474)。
- 原实现代码库：[FreeSurfer `mri_synthstrip`](https://github.com/freesurfer/freesurfer/tree/dev/mri_synthstrip)。
