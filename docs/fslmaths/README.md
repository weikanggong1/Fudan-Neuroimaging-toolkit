# fslmaths：常用影像运算

| 项目 | 内容 |
|---|---|
| 输入 | 3D/4D NIfTI与有序操作列表 |
| 输出 | 保存影像的Path及同空间NIfTI |
| 对应原软件 | FSL fslmaths常用选项 |
| Python / CLI | run_fslmaths / fnit fslmaths、fnit-fslmaths |
| CPU / GPU | PyTorch CPU/CUDA；nibabel读写 |

## 1. 功能简介

`fnit-fslmaths` 用 PyTorch 执行一组常用的 FSL `fslmaths` 运算。输入和输出为 NIfTI；运行时不调用 FSL。选项按命令行出现的顺序作用于当前图像。本页只描述已经实现的选项，其余选项会明确报错。

## 2. Python 调用

```python
from fnit.fslmaths import run_fslmaths

input_path = "/data/sub-01/T1w_brain.nii.gz"  # 输入：3D T1w 脑图
operations = [
    "-thr", "100",  # 保留强度 >= 100 的体素，其余置零
    "-bin",         # 把正值置 1，非正值置 0
]
output_path = "/data/sub-01/T1w_mask.nii.gz"  # 输出：同网格 3D 二值 NIfTI
device = "cuda:0"  # 计算设备；无 GPU 时改为 "cpu"
input_dtype = "float"  # 内部 float32 计算
output_dtype = "char"  # 输出 uint8 掩膜

saved_path = run_fslmaths(
    input_path=input_path,
    operations=operations,
    output_path=output_path,
    device=device,
    input_dtype=input_dtype,
    output_dtype=output_dtype,
)  # 返回：实际写入的 pathlib.Path
```

### 输入数据格式

3D/4D NIfTI；前三维X/Y/Z，第四维时间或体积序列。图像操作数必须同shape/affine，3D沿第四维广播；函数不自动配准。运算顺序严格按operations字符串次序。

### 输入参数

| 参数 | 必需 | 类型 | 默认值 | 含义 |
|---|---|---|---|---|
| `input_path` | 是 | 路径 | 无 | 3D 或 4D NIfTI 路径。 |
| `operations` | 是 | str序列 | 无 | 按FSL顺序的操作符和参数，影像操作数须同空间网格。 |
| `output_path` | 是 | 路径 | 无 | 输出NIfTI；无扩展名补.nii.gz。 |
| `device` | 否 | str / torch.device / None | `'cpu'` | 计算设备；显式 CUDA 不可用时报错。 |
| `input_dtype` | 否 | str / None | `None` | float/double；None为原图float64则double，否则float。 |
| `output_dtype` | 否 | str / None | `None` | 保存数据类型；None 遵循输入类型和接口默认规则。 |

### 输出

```text
/data/sub-01/
└── T1w_mask.nii.gz
```

返回值是实际写出的 `pathlib.Path`。文件内容是一张 NIfTI，保留输入的空间仿射、qform、sform 与体素尺寸。时间归约后的 4D 图像成为 3D；X/Y/Z 归约的对应空间轴长度变为 1；`-roi` 保持图像原尺寸，把区域外置零。3D 图像参与 4D 二元运算时沿第四维广播。图像操作数必须具有相同空间尺寸和仿射，程序不会自动重采样。

物理单位随运算变化：掩膜无量纲，-s/-kernel物理尺度mm，-bptf为帧数，反三角函数rad。空间affine/qform/sform/pixdim保留，T归约为3D，X/Y/Z归约相应轴长1；-roi保持原shape将区外置零。

### 多步运算示例

对已经同空间的参数图顺序应用脑mask、缩放和阈值：

```python
from fnit.fslmaths import run_fslmaths

parameter_image_path = "/data/dti_FA.nii.gz"  # 同个体空间3D FA
brain_mask_path = "/data/nodif_brain_mask.nii.gz"  # 同shape/affine的脑mask
masked_parameter_output_path = "/data/FA_scaled.nii.gz"  # 仍在原个体空间
ordered_operations = [
    "-mas", brain_mask_path,  # 先仅保留脑内
    "-mul", "1000",         # 再改变数值尺度
    "-thr", "200",          # 最后按缩放后数值阈值化
]
saved_parameter_path = run_fslmaths(
    input_path=parameter_image_path, operations=ordered_operations,
    output_path=masked_parameter_output_path, device="cuda:0",
    input_dtype="float", output_dtype="float",
)
```

-mas使用mask>0处；它不自动重采样mask。阈值200作用于乘1000后的FA，所以对应原FA≥0.2；改顺序会改变结果。

### 输出数据类型

| output_dtype | NIfTI存盘类型 | 用途 |
|---|---|---|
| char | uint8 | 0/1掩膜；先-bin避免非二值截断 |
| short | int16 | 需要整数的有符号图 |
| int | int32 | 整数图/label |
| float | float32 | 常规连续图 |
| double | float64 | 保留double输出 |
| input | 与原输入相同 | 需评估整数转换造成的强度损失 |

整数输出先按FSL规则四舍五入；输入整数的-dt input尚未实现。内部dtype与输出dtype是两个参数，改变输出类型不会恢复此前计算中已经损失的精度。

### 处理完整时间序列

-bptf的sigma单位为volume数，需要调用者按TR把希望的秒数换算为帧数。高通结果去均值；不应将-bptf输出均值当作原BOLD均值。

T归约在第四维完成；3D图像的第四维不存在时，应先确认操作含义。空间X/Y/Z归约保持轴位置但该轴长1，仍保留物理affine。

-roi输入依次是xmin/xsize/ymin/ysize/zmin/zsize/tmin/tsize；size=-1表示从起点到末端。它清零区外而非生成裁剪后的新网格。

### 失败行为

输入与图像操作数网格不匹配、未知操作或偶数boxv边长会报错。设备参数指定不可用CUDA时，不按GPU成功记录；需改为明确cpu调用。

## 3. 命令行调用

```bash
fnit-fslmaths --device cuda:0 -dt float /data/sub-01/T1w_brain.nii.gz -thr 100 -bin /data/sub-01/T1w_mask.nii.gz -odt char
```

--device须放最前；-dt→input_dtype、-odt→output_dtype，输入/输出位置→input_path/output_path，中间选项→operations。统一fnit fslmaths接受同样顺序。

| 选项 | 输入参数与输出规则 | 官方命令片段 |
|---|---|---|
| `-add`、`-sub`、`-mul`、`-div` | 后接数值或同网格图像，依次进行加、减、乘、除。图像分母为零时输出零；数值分母为零时保持原值。 | `fslmaths in -mul 2 out` |
| `-mas` | 后接同网格掩膜，仅保留掩膜大于零处。 | `fslmaths in -mas mask out` |
| `-max`、`-min` | 后接数值或图像，逐体素取较大或较小值。 | `fslmaths in -max 0 out` |
| `-thr`、`-uthr` | 后接阈值；分别保留大于等于、或小于等于阈值的值。 | `fslmaths in -thr 100 out` |
| `-abs`、`-sqr`、`-sqrt`、`-recip`、`-pow` | 绝对值、平方、平方根、倒数、指定幂；`-pow` 后接指数。负数的 `-sqrt` 输出零，零的 `-recip` 保持零。 | `fslmaths in -sqr out` |
| `-exp`、`-log`、`-sin`、`-cos`、`-tan`、`-asin`、`-acos`、`-atan` | 逐体素函数。`-log` 只变换正数，其余值保持原样。反三角函数使用弧度。 | `fslmaths in -log out` |
| `-bin`、`-binv` | 分别输出正值掩膜、非正值掩膜。 | `fslmaths in -bin out` |
| `-nan`、`-nanm` | 非有限值置零；或输出非有限值的 0/1 掩膜。 | `fslmaths in -nan out` |
| `-inm`、`-ing` | 后接目标均值。按正值体素计算均值；`-inm` 逐 3D 体积缩放，`-ing` 在整张 4D 图像上使用一个缩放因子。 | `fslmaths in -inm 100 out` |
| `-kernel 3D`、`2D`、`box`、`boxv`、`boxv3`、`gauss`、`sphere` | 设置后续滤波核。`box`、`gauss`、`sphere` 的尺寸单位为 mm；`boxv` 和 `boxv3` 为奇数体素尺寸。默认核是 3×3×3。 | `fslmaths in -kernel boxv 3 -fmean out` |
| `-fmean`、`-fmeanu`、`-s` | 加权均值滤波、无边界重归一化的均值滤波、以 mm 为单位的高斯平滑。`-s` 后接高斯 sigma。 | `fslmaths in -s 2 out` |
| `-dilF`、`-eroF`、`-dilM`、`-ero`、`-fmedian` | 核内最大、最小；仅把零体素替换为非零邻居均值；遇到零邻居时把非零体素置零；核内中位数。形态学和中位数目前使用 box 核。 | `fslmaths in -dilM out` |
| `-Tmean/std/max/maxn/min/median/perc/ar1` | 沿第四维归约。`maxn` 是最大值索引，从零开始；`perc` 后接 0–100 的百分比。将 `T` 换为 `X`、`Y`、`Z` 可沿对应空间轴归约。 | `fslmaths series -Tmean mean` |
| `-roi` | 后接 `xmin xsize ymin ysize zmin zsize tmin tsize` 共八个整数；尺寸为 `-1` 表示从起点到该维末端。 | `fslmaths in -roi 4 32 3 40 2 48 0 -1 out` |
| `-bptf` | 后接高通 sigma、低通 sigma，单位都是体积数；小于等于零跳过对应滤波。高通输出去均值序列。 | `fslmaths series -bptf 50 -1 filtered` |
| `-range` | 把 NIfTI 显示范围写为当前数据的实际最小/最大值。 | `fslmaths in -range out` |

尚未实现的官方选项包括百分比稳健阈值 `-thrp/-thrP/-uthrp/-uthrP`、`-rem`、`-fillh/-fillh26`、`-dilD/-dilall`、`-subsamp2`、`-tensor_decomp`、TFCE、随机噪声、非参数统计、ROC 和 `-kernel file`。这些选项不会被静默忽略。当前不接受偶数边长的 boxv 核。

## 4. 原软件调用

```bash
fslmaths -dt float /data/sub-01/T1w_brain.nii.gz -thr 100 -bin /data/sub-01/T1w_mask.nii.gz -odt char
```

只在独立benchmark环境调用原程序。原命令输入、operations、-dt/-odt映射如上；未实现选项会报错。

## 5. 最新精度和运行时间

[逐项结果与输入 SHA-256](../../validation/fslmaths/real_data_20260929.json)记录了真实 T1w 脑图（176×256×256）和真实 DTI tensor（104×104×72×6）上的完整图像对照、两张图像的 64³ 和 48³×6 连续体素块，以及真实 BOLD 时间序列的 16³×490 空间块。参考软件为 FSL 6.0.7.4；FNIT 使用 PyTorch 2.5.1 和 Nibabel 5.4.2。FSL 在 登录节点 CPU 运行，FNIT 在 GPU评测节点 H100 GPU 0 运行。GPU 测前已有其他任务，显存占用 56,019 MiB/81,559 MiB、利用率 98%；测试进程限制为总显存的 20%。计时包括读写和压缩，不能据此推断空闲 GPU 或整套流程的加速比。

| 真实图像与运算 | FSL 耗时 | FNIT GPU 耗时 | 平均绝对差 | 最大绝对差 |
|---|---:|---:|---:|---:|
| 完整 T1：`-thr 100 -bin` | 3.72 s | 1.45 s | 0 | 0 |
| 完整 tensor：`-Tmean` | 0.27 s | 0.24 s | 6.94×10⁻¹² | 2.33×10⁻¹⁰ |
| 完整 T1：`-s 2` | 5.92 s | 6.73 s | 3.00×10⁻⁵ | 8.43×10⁻⁴ |
| 完整 T1：`-kernel boxv 3 -fmean` | 3.33 s | 1.50 s | 4.85×10⁻⁶ | 2.92×10⁻⁴ |
| 完整 tensor：`-bptf 2 -1` | 0.98 s | 0.77 s | 1.52×10⁻¹² | 2.33×10⁻¹⁰ |
| 真实 BOLD 16³×490：`-bptf 20 -1` | 1.68 s | 0.92 s | 2.79×10⁻⁶ | 2.44×10⁻⁴ |
| 真实 BOLD 16³×490：`-bptf 20 2` | 0.75 s | 0.74 s | 1.91×10⁻⁶ | 1.22×10⁻⁴ |

完整图像测试的 FNIT CUDA 峰值分配不超过 0.430 GiB；48 组裁剪块对照包含每个选项组合的耗时、误差和峰值分配。高斯平滑的差异主要来自卷积累加顺序和浮点舍入。空间仿射、体素尺寸、qform/sform 和输出形状在上述完整图像对照中一致。

同一真实 tensor 裁剪块及其 3D 时间中位图还用于三次混合维度检验：4D 加 3D、3D 加 4D、4D 用 3D 图像作掩膜。三组输出均为 48×48×48×6，与 FSL 逐体素一致。

本页源码自首次功能提交45b5b22d（2026-09-29）之后未变；正式报告仍按该日期读取。数据为1例完整T1、1例tensor和1例BOLD空间块，裁剪块不作为完整MRI流程速度。CPU型号和线程未记录。

### 分步骤 benchmark

| 阶段 | FNIT | FSL |
|---|---|---|
| 输入/运算/保存分别计时 | 未记录 | 未记录 |

尚无本功能公开脑图，精度依据完整体素报告；未用模拟图补充。

## 6. 最近版本和 benchmark

| 日期 | commit/version | 变化 | benchmark |
|---|---|---|---|
| 2026-09-29 | 45b5b22d | 首次常用运算/CLI/类型合同实现 | [真实完整影像和48组裁剪记录](../../validation/fslmaths/real_data_20260929.json) |

当前只有这一项实质发布记录；不虚构额外版本。

## 7. 参考文献、原软件和资源

源码位置：[原软件 `avwutils-2209.8/fslmaths.cc`](../../src/fnit/_vendor_fsl/sources/avwutils-2209.8/fslmaths.cc)；[FNIT `fslmaths/core.py`](../../src/fnit/fslmaths/core.py)。固定 tag/commit、Git tree 及每文件 SHA-256 见[来源清单](../../src/fnit/_vendor_fsl/manifest.json)。

代码改写沿用 [FSL 6.0 非商业许可证](../../licenses/FSL-6.0.txt)，完整来源与再分发要求见[第三方声明](../../THIRD_PARTY_NOTICES.md)。

- 参考文献：Smith et al., *Advances in functional and structural MR image analysis and implementation as FSL*, NeuroImage (2004), [doi:10.1016/j.neuroimage.2004.07.051](https://doi.org/10.1016/j.neuroimage.2004.07.051)。 `fslmaths` 没有单独的方法论文。
- 原实现代码库：[FSL `avwutils`（含 `fslmaths`）](https://git.fmrib.ox.ac.uk/fsl/avwutils)。

| 资源 | 用途 | 官方来源 | 大小 | SHA-256 | 是否允许 FNIT 再分发 |
|---|---|---|---|---|---|
| 本功能无模型权重；用户自备参考影像、掩膜或变换 | 定义目标网格/变换 | 各文件原作者 | 按实际文件 | 按实际文件 | 不随本功能发布用户数据 |

[完整历史说明与调试证据](../../validation/fslmaths/readme_archive_20261005.md) · [返回主页](../../README.md)

<!-- 旧版文档锚点兼容 -->
<a id="安装与输入输出"></a> <a id="已实现的选项"></a> <a id="真实影像对照"></a> <a id="reference"></a>
