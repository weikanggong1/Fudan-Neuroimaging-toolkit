<!-- 原正文完整归档；基线 140c3739ac6c7a6826bf9421202ef59bec7ffe67；只修复迁移后的相对链接。 -->

# fslmaths 常用运算

`fnit-fslmaths` 用 PyTorch 执行一组常用的 FSL `fslmaths` 运算。输入和输出为 NIfTI；运行时不调用 FSL。选项按命令行出现的顺序作用于当前图像。本页只描述已经实现的选项，其余选项会明确报错。

## 安装与输入输出

主页的 `environment.yml` 已包含 PyTorch、NumPy 和 Nibabel，安装 FNIT 后可使用 `fnit-fslmaths` 或 `fnit fslmaths`。不需要另外下载模型。

公开 Python 函数是 `fnit.fslmaths.run_fslmaths(input_path, operations, output_path, *, device="cpu", input_dtype=None, output_dtype=None)`。

| 参数 | 输入含义 |
|---|---|
| `input_path` | 3D 或 4D NIfTI 路径。前三维是 X、Y、Z，第四维是时间或体积序列。 |
| `operations` | 字符串序列；每个选项及其参数按 FSL 命令行次序写入。图像操作数是另一张同空间网格的 NIfTI，数值操作数写成字符串。 |
| `output_path` | 输出 `.nii` 或 `.nii.gz` 路径；无扩展名时补 `.nii.gz`。 |
| `device` | `cpu` 或 `cuda:0` 等 PyTorch 设备。CPU 为默认值。CUDA 路径允许 TF32，不自动使用 float16/bfloat16。 |
| `input_dtype` | 计算精度：`float` 或 `double`。默认与 FSL 相同：float64 原图用 double，其余用 float。整数图像的 `-dt input` 尚未实现。 |
| `output_dtype` | 输出 NIfTI 类型：`char`、`short`、`int`、`float`、`double` 或 `input`。默认 float；原图为 float64 时默认 double。整数输出在转换前按 FSL 规则四舍五入。 |

返回值是实际写出的 `pathlib.Path`。文件内容是一张 NIfTI，保留输入的空间仿射、qform、sform 与体素尺寸。时间归约后的 4D 图像成为 3D；X/Y/Z 归约的对应空间轴长度变为 1；`-roi` 保持图像原尺寸，把区域外置零。3D 图像参与 4D 二元运算时沿第四维广播。图像操作数必须具有相同空间尺寸和仿射，程序不会自动重采样。

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

等价的官方命令：

```bash
fslmaths -dt float /data/sub-01/T1w_brain.nii.gz -thr 100 -bin /data/sub-01/T1w_mask.nii.gz -odt char
```

FNIT 命令行把 `--device` 放在最前面，其余参数保留 FSL 顺序：

```bash
fnit-fslmaths --device cuda:0 -dt float /data/sub-01/T1w_brain.nii.gz -thr 100 -bin /data/sub-01/T1w_mask.nii.gz -odt char
```

## 已实现的选项

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

## 真实影像对照

[逐项结果与输入 SHA-256](real_data_20260929.json)记录了真实 T1w 脑图（176×256×256）和真实 DTI tensor（104×104×72×6）上的完整图像对照、两张图像的 64³ 和 48³×6 连续体素块，以及真实 BOLD 时间序列的 16³×490 空间块。参考软件为 FSL 6.0.7.4；FNIT 使用 PyTorch 2.5.1 和 Nibabel 5.4.2。FSL 在 登录节点 CPU 运行，FNIT 在 GPU评测节点 H100 GPU 0 运行。GPU 测前已有其他任务，显存占用 56,019 MiB/81,559 MiB、利用率 98%；测试进程限制为总显存的 20%。计时包括读写和压缩，不能据此推断空闲 GPU 或整套流程的加速比。

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

## Reference

- 参考文献：Smith et al., *Advances in functional and structural MR image analysis and implementation as FSL*, NeuroImage (2004), [doi:10.1016/j.neuroimage.2004.07.051](https://doi.org/10.1016/j.neuroimage.2004.07.051)。 `fslmaths` 没有单独的方法论文。
- 原实现代码库：[FSL `avwutils`（含 `fslmaths`）](https://git.fmrib.ox.ac.uk/fsl/avwutils)。
