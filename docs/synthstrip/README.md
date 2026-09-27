# SynthStrip：脑提取

[返回首页](../../README.md) · [源码目录](../../src/fnit/synthstrip/) · [权重](../WEIGHTS.md)

SynthStrip 从脑影像预测有符号距离场，生成脑掩膜和去除背景的影像。官方模型已使用 PyTorch。本模块沿用其网络、权重和影像处理流程，提供可重复调用的 Python API；模型未重新训练。

参考版本为 **FreeSurfer 8.2.0**，build `freesurfer-linux-centos7_x86_64-8.2.0-20260314-d932c45`。分析对象是 `$FREESURFER_HOME/python/scripts/mri_synthstrip`，不是 `bin/` 下的包装脚本。脚本 SHA-256 为 `bbc2ff8f8779862039401b05d5cd6039fb4f3583e0032a793ac9adb3f4521590`，全部来源信息见 [provenance.json](../provenance.json)。

## Python API

```python
from pathlib import Path
from fnit import SynthStrip

output_dir = Path("results")
output_dir.mkdir(exist_ok=True)
extract = SynthStrip(
    weights="/path/to/weights",  # 输入：synthstrip.1.pt 文件或其目录
    device="cuda:0",           # 输入：运行设备，也可写 "cpu"
    no_csf=False,              # 输入：是否改用排除脑脊液的权重
    threads=4,                 # 输入：PyTorch CPU 线程数
)
result = extract(
    image="subject_T1w.nii.gz",  # 输入：3D/4D T1 影像路径或内存影像
    border=1,                   # 输入：距离阈值，单位 mm
    fill=0,                     # 输入：掩膜外的强度
)
result.image.save(output_dir / "subject_brain.nii.gz")     # 输出：去颅骨影像
result.mask.save(output_dir / "subject_mask.nii.gz")       # 输出：二值脑掩膜
result.distance.save(output_dir / "subject_sdt.nii.gz")    # 输出：有符号距离图，mm
# 可继续以 image="another_T1w.nii.gz" 调用 extract，复用已加载模型。
```

`from fnit.synthstrip import SynthStrip, StripResult` 是等价的功能模块入口。

### 模型构造

`SynthStrip(weights=None, device="cpu", no_csf=False, threads=None)`：

| 参数 | 含义 |
|---|---|
| `weights` | 官方 PT 文件或所在目录；省略时使用[统一查找顺序](../ARCHITECTURE.md#公开-api-与兼容性) |
| `device` | `"cpu"` 或 `"cuda:N"`；CUDA 编号遵循 `CUDA_VISIBLE_DEVICES` |
| `no_csf` | 为 `True` 时使用排除 CSF 的官方权重 |
| `threads` | 当前进程的 Torch 线程数；`None` 保留当前值 |

模型进入 eval 模式并使用 float32 张量；CUDA 构造默认允许 TF32 matmul 和 cuDNN 内核，不使用 float16 或 bfloat16。重复使用实例可避免重复加载权重。

### 单次调用与结果

`extract(image, border=1, fill=None) -> StripResult`：

| 参数或字段 | 含义 |
|---|---|
| `image` 输入 | `.nii`、`.nii.gz`、`.mgh`、`.mgz` 路径，或 `nibabel` 影像对象；支持 3D 和逐帧处理的 4D。已有 `surfa.Volume` 内存对象也可传入，供旧调用方过渡 |
| `border` | SDT 阈值，单位 mm，默认 1 |
| `fill` | 掩膜外的强度；省略时为 `min(image.min(), 0)` |
| `result.image` | 掩膜外已填充的影像，保留原网格和几何 |
| `result.mask` | 二值脑掩膜 |
| `result.distance` | 有符号距离场，单位 mm |

从路径或 `nibabel` 对象调用时，三个字段均为仓库内的 `Volume`，具有 `.data`、`.affine`、`.shape` 和 `.save(path)`，可保存 NIfTI、MGH、MGZ。传入已有 `surfa.Volume` 时，三个字段仍返回同类对象以兼容旧调用。调用不会修改输入对象；直接使用 Python 保存时，由调用者准备输出父目录。

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

U-Net 在所选设备执行。影像读写使用 `nibabel`，LIA 几何、裁剪和网格形状由仓库内代码计算；最近邻与线性采样、距离扩展及连通域在 CPU 上调用 NumPy/SciPy。完整推理无需导入 Surfa。单例 4D 输入逐帧处理，每次网络推理一个 frame。GPU 可加快网络部分，完整进程耗时还取决于预后处理和 I/O。

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

1. 使用原影像几何将输入变换至 LIA 方向、1 mm 各向同性、float32，使用最近邻重采样。
2. 按非零包围盒裁剪；各轴尺寸向上取 64 的倍数，再限制在 192–320；居中裁剪或补零。
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

官方脚本集成了参数解析和执行流程；本包允许导入并缓存模型，且文件几何由仓库内代码与 `nibabel` 处理。日志、版本和帮助格式由本包维护。统一 CLI 名称为 `fnit synthstrip`，不会替换系统 `mri_synthstrip`。

当前无 Surfa 实现在同一份真实 T1、同一台 headcw 上与 FreeSurfer 8.2 官方 CLI 比较：`orig.mgz` 的脑图、掩膜、距离图各 **16,777,216 / 16,777,216** 个体素一致；实际参数 `border=1`、`fill=0`、CPU 4 线程。单次完整 CLI 墙钟时间和进程峰值内存、`border=8`、`no_csf=True`、斜切输入与 4D 功能检查见[迁移验证](../../validation/synthstrip_no_surfa_20260927/README.md)。这些耗时只代表该次执行条件，不能据此宣称普遍加速。

以下保留此前使用同一份去面容的公开 `sub-02` T1w 输入和官方权重生成的脑图；当前无 Surfa 版本的数值结论以上述新验证为准。两行分别为轴位和冠状位；三个面板使用同一切面及灰度范围。图片的制作步骤与完整影像比较见[图示记录](../figures/README.md)。

![公开 T1w 输入、FreeSurfer 脑图与本包脑图](../figures/synthstrip_comparison.png)

| 检查 | 记录 |
|---|---|
| 无 Surfa 实现、同一真实 T1、官方 CLI 与新 CLI/API | [迁移验证](../../validation/synthstrip_no_surfa_20260927/README.md) |

测试源码在 [tests/synthstrip/](../../tests/synthstrip/)；原版对照工具在 [tools/validate_synthstrip.py](../../tools/validate_synthstrip.py)：

```bash
python -m pytest tests/synthstrip tests/test_public_api.py
# 只有与原版比较时才需要 FreeSurfer。
module load freesurfer
python tools/validate_synthstrip.py --image /path/to/test_T1w.nii.gz \
  --out-dir validation/synthstrip --device cuda --reference-device cpu
```

该工具从指定原脚本 AST 提取网络类，用相同权重比较参数名、参数量和随机 `64³` 输入的预测，再执行完整影像流程；正式流程仍使用官方最小 `192³` 网格。参考安装自带的 Torch 是 CPU build，因此 GPU 原版参考使用未修改官方脚本与 CUDA Python 环境，报告明确标为 `official_source_cuda`。

这些检查衡量与原版的数值一致性。真实病例没有人工脑掩膜真值，无法从中得出临床提取准确率。大视野裁切和常数输入的处理见上文。

原方法：Hoopes et al., *SynthStrip: Skull-Stripping for Any Brain Image*, NeuroImage (2022), [doi:10.1016/j.neuroimage.2022.119474](https://doi.org/10.1016/j.neuroimage.2022.119474)。
