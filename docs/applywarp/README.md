# TorchApplyWarp

`TorchApplyWarp` 是 FSL `applywarp` 已验证子集的 PyTorch 实现。运行时不调用
FSL，可在 CPU 或 CUDA 上读取 FSL dense warp 以及 FNIRT cubic coefficient
文件，并把 3D/4D 输入重采样到 reference 网格。

CUDA 运行保持 TF32 matrix/cuDNN 全局开关；FSL 坐标矩阵、坐标网格和 cubic
coefficient 展开显式使用 float64，避免 TF32 改变重采样坐标。影像插值张量与输出
使用所声明的 float32/float64 dtype，不自动使用 float16 或 bfloat16。实际全局开关
记录在 `result.qc["tf32"]`。

## Python 调用

```python
from fnit.applywarp import TorchApplyWarp

result = TorchApplyWarp("cuda:0")(
    input="subject_GM.nii.gz",
    reference="GM_template.nii.gz",
    warp="subject_to_template_coef.nii.gz",
    premat=None,
    postmat=None,
    interpolation="trilinear",
    warp_convention="auto",
    output_dtype="float",
)
result.save("subject_GM_to_template.nii.gz")
```

`result.image` 是 reference 网格上的 NIfTI image，`result.valid_mask` 标记 warp
和输入采样都有效的体素，`result.qc` 记录实际使用的 intent、warp convention、
插值和 dtype。需要一步写盘时使用：

```python
TorchApplyWarp("cuda:0").run(
    "subject_GM.nii.gz",
    "GM_template.nii.gz",
    "subject_GM_to_template.nii.gz",
    warp="subject_to_template_coef.nii.gz",
    interpolation="trilinear",
)
```

函数形式从子包导入，避免与 `fnit.applywarp` 模块同名：

```python
from fnit.applywarp import applywarp

result = applywarp(
    "subject_GM.nii.gz",
    "GM_template.nii.gz",
    warp="subject_to_template_warp.nii.gz",
    device="cuda:0",
)
```

## 命令行调用

以下两行执行同一操作：

```bash
fnit-applywarp \
  --in subject_GM.nii.gz \
  --ref GM_template.nii.gz \
  --warp subject_to_template_coef.nii.gz \
  --interp trilinear \
  --datatype float \
  --device cuda:0 \
  --out subject_GM_to_template.nii.gz

fnit applywarp \
  --in subject_GM.nii.gz \
  --ref GM_template.nii.gz \
  --warp subject_to_template_coef.nii.gz \
  --interp trilinear \
  --datatype float \
  --device cuda:0 \
  --out subject_GM_to_template.nii.gz
```

这对应 FSL：

```bash
applywarp \
  --in=subject_GM.nii.gz \
  --ref=GM_template.nii.gz \
  --warp=subject_to_template_coef.nii.gz \
  --interp=trilinear \
  --datatype=float \
  --out=subject_GM_to_template.nii.gz
```

| Python | 本包 CLI | FSL `applywarp` | 含义 |
|---|---|---|---|
| `input` | `--in` | `--in` | 待重采样的 3D image 或 4D series |
| `reference` | `--ref` | `--ref` | 决定输出 shape、voxel size、qform 和 sform |
| `warp` | `--warp` | `--warp` | FSL dense field 或 intent 2007 coefficient；省略时只应用矩阵 |
| `premat` | `--premat` | `--premat` | input → warp source 的一个 4×4 FLIRT scaled-mm 矩阵 |
| `postmat` | `--postmat` | `--postmat` | warp reference → output reference 的一个 4×4 FLIRT scaled-mm 矩阵 |
| `interpolation` | `--interp` | `--interp` | 输入 image 的 `trilinear` 或 `nearest`/`nn` 插值 |
| `warp_convention` | `--rel` / `--abs` | `--rel` / `--abs` | 只控制无专用 intent 的 dense field；`auto` 使用 FSL 的标准差判别 |
| `output_dtype` | `--datatype` | `--datatype` | `char`、`short`、`int`、`float`、`double`；省略时遵循 FSL 默认规则 |
| `device` | `--device` | 无 | `cpu`、`cuda` 或 `cuda:N` |
| 无 | `--overwrite` | FSL 默认覆盖 | 允许 CLI 覆盖已有输出 |

CLI 当前只接收一个 4×4 `premat` 和一个 4×4 `postmat`。FSL 支持的逐帧堆叠
矩阵不在本实现范围内。

## 坐标和变换方向

FSL scaled-mm 不是 NIfTI world-RAS。它按 header voxel size 缩放体素轴；当
voxel-to-world affine 的行列式为正时，还会翻转并平移第一存储轴。dense field
的三个分量是这组三个 scaled-mm 轴上的毫米值。

对 output/reference voxel `x`，设 `Vref` 和 `Vin` 分别为 reference 与 input
的 voxel → FSL scaled-mm 矩阵。本实现执行以下 pull mapping：

```text
q = Vref x
r = inverse(postmat) q

dense relative: s = r + sample(warp, r)
dense absolute: s = sample(warp, r)
intent 2007:    s = inverse(embedded_affine) r + sample(cubic_residual, r)

u = inverse(Vin) inverse(premat) s
output(x) = input(u)
```

因此，从输入到输出描述线性步骤时，顺序是 `premat → warp → postmat`；真正
重采样时从输出点反查输入点，按上式使用逆矩阵。warp field 本身始终用
trilinear 插值；`--interp` 选择的是最后对输入 image 的插值。nearest 使用
FSL 对正坐标的 half-up 规则。

SynthMorph/Surfa 的 `disp_ras` warp 同样可表示 target/output → source/input 的
pull mapping，但其位移分量属于 world-RAS，并携带 source/target geometry。
FSL warp 的分量属于 FSL scaled-mm，FNIRT coefficient 还把 residual、knot
spacing 和 affine 分开保存。两者方向角色相近，数组和 header 不能直接互换。
本模块不读取 FreeSurfer `.m3z`、`.mgz` warp 或 `surfa.Warp`；转换时必须同时
使用 source 和 target geometry 完成 RAS/FSL 坐标换算。

## 支持的 warp 文件

| NIfTI intent | FSL 表示 | 状态 | 解释 |
|---:|---|---|---|
| 0 或其他 | 4D dense field，最后一维为 3 | 支持 | `--rel`、`--abs` 或 FSL `auto` 判别 |
| 2006 | FNIRT dense displacement | 支持 | 按 FSL 规定始终视为 relative，忽略 `--abs` |
| 2007 | cubic B-spline coefficients | 支持 | 解码 qform offset field size、intent voxel size、pixdim knot spacing、raw coefficient order 和 sform embedded affine |
| 2008 | DCT coefficients | 拒绝 | 抛出 `NotImplementedError`，不做近似 |
| 2009 | quadratic B-spline coefficients | 拒绝 | 抛出 `NotImplementedError`，不做 cubic 替代 |

intent 2007 的运行时解码完全在 PyTorch 中完成，不调用 `fnirtfileutils`。
`fnirtfileutils` 只用于外部验证。

## 输出约定和实现边界

输出 shape、affine、qform/sform code 和空间 voxel size 取自 reference；4D 输入
的时间间隔取自 input。显式整数 dtype 使用 FSL 的向零截断。未指定 dtype 时，
float/double 输入保持 dtype；整数输入在输出动态范围小于 100 时写 float32，
否则保持输入 dtype。

已实现的 FSL 子集包括 reference-grid 输出、dense relative/absolute、intent
2006、intent 2007、一个 premat、一个 postmat、trilinear/nearest 和 FSL dtype
规则。以下 FSL 选项没有实现：`sinc`、输入 image 的 cubic `spline` 插值、
supersampling、`--paddingsize`、`--mask`、`--usesqform` 和逐帧矩阵。CLI 会拒绝
这些选项，代码不会把它们降级成已支持模式。

## 与 FSL 6.0.7.4 的验证

解析验证脚本为
[`tools/validate_applywarp_fsl.py`](../../tools/validate_applywarp_fsl.py)：

```bash
module load fsl/6.0.7.4
PYTHONPATH=src python tools/validate_applywarp_fsl.py \
  --work-dir /tmp/applywarp-parity \
  --device cuda:0 \
  --json-out validation/applywarp/report.json
```

脚本用坐标 ramp 检查 dense 与 cubic coefficient、trilinear 与 nearest、含/不含
premat 和 postmat，并用 `fnirtfileutils --out`、`--withaff` 分别检查 residual
展开和 embedded affine。它还检查 reference header、显式 short dtype，并在
91×109×91 的 2 mm synthetic fixture 上记录 FSL CPU、PyTorch CPU 和 PyTorch
GPU 的端到端 NIfTI 读写时间。当前机器实测结果见
[`validation/applywarp/report.json`](../../validation/applywarp/report.json)。

CPU 和 H100 路径各自完成 11 项输出检查，均为 11/11 通过。下表中的误差由
H100 完整三维输出逐体素计算；nearest、显式 short dtype、reference header 均与
FSL 完全一致。trilinear 的最大误差没有超过预设的 `5e-4`，cubic coefficient
展开与 `fnirtfileutils` 的最大误差没有超过 `1e-6`。

| FSL 对照范围 | 检查数 | 最大绝对误差范围 | 结果 |
|---|---:|---:|---:|
| dense / cubic，trilinear，含或不含矩阵 | 4 | 0.000366–0.000488 | 4/4 通过 |
| dense / cubic，nearest，含或不含矩阵 | 4 | 0 | 4/4 逐元素一致 |
| reference header 与显式 short dtype | 1 | 0 | 完全一致 |
| cubic residual / embedded affine 展开 | 2 | 3.73e-9–1.78e-8 | 2/2 通过 |

计时为同一个 91×109×91、2 mm、dense-relative、trilinear synthetic fixture，
包含 NIfTI 读取、变换和写盘；先预热，再取三次运行的中位数。FSL 在 CPU 上运行，
本包分别在 CPU 和 H100 GPU 上运行。

| 实现 | 设备 | 完整调用中位数 |
|---|---|---:|
| FSL 6.0.7.4 `applywarp` | CPU | 0.418 s |
| `TorchApplyWarp` | CPU | 0.384 s |
| `TorchApplyWarp` | H100 GPU | 0.221 s |

下图显示同一 synthetic 输入经过同一 dense relative warp 后的 FSL 与本包 GPU
输出。两行分别为轴位和冠状位；FSL 与本包面板使用同一色阶，差值面板单独使用
误差色阶。图片用于展示误差位置，判断标准仍是上面的完整三维比较。

![FSL applywarp 与 TorchApplyWarp 的 synthetic 输出比较](figures/applywarp_fsl_comparison.png)

GPU 三次计时与 11 项输出记录见
[`report.json`](../../validation/applywarp/report.json)，独立 CPU 11 项对照见
[`report.cpu.json`](../../validation/applywarp/report.cpu.json)。该验证覆盖本页列出的
已支持子集，不能外推到尚未实现的 sinc、输入图像 cubic spline、supersampling
或逐帧矩阵。

单元测试和可选 FSL 外部测试位于
[`tests/applywarp`](../../tests/applywarp)。
