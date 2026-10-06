# FNIRT 系数文件单位与输出对照范围

## 1. 功能简介

修复 `make_fsl_coefficient_image` 和 `save_fsl_coefficients` 的系数文件单位。原函数使用 nibabel 默认的未知单位；NEWIMAGE 的写出逻辑明确设置毫米和秒，NIfTI `xyzt_units=10`。本次补齐该字段，系数值、网格、qform、sform、优化器和 GPU 计算保持原实现。

## 2. Python 调用、输入与输出

```python
from pathlib import Path
from fnit.fnirt.io import load_fsl_coefficients, save_fsl_coefficients

# 从自己已完成配准的系数文件读取数值和格式，不重新估计形变。
coefficient_input_path = Path("inputs/completed_warp_coefficients.nii.gz")
loaded_coefficients = load_fsl_coefficients(coefficient_input_path)
coefficient_output_path = Path("outputs/warp_coefficients.nii.gz")
reference_field_shape = loaded_coefficients.field_shape  # 目标网格尺寸
reference_voxel_sizes = loaded_coefficients.field_voxel_sizes  # 毫米
spline_knot_spacing = loaded_coefficients.knot_spacing  # 目标体素单位
forward_flirt_affine = loaded_coefficients.affine_forward  # 原 FLIRT 矩阵

save_fsl_coefficients(
    filename=coefficient_output_path,
    coefficients=loaded_coefficients.coefficients,
    field_shape=reference_field_shape,
    field_voxel_sizes=reference_voxel_sizes,
    knot_spacing=spline_knot_spacing,
    affine_forward=forward_flirt_affine,
)
```

| 参数 | 意义与格式 |
|---|---|
| `filename` | 输出 NIfTI 路径；父目录自动创建。 |
| `coefficients` | 有限数值数组，形状 `[Cx, Cy, Cz, 3]`；写出为 Float32。控制网格必须与目标尺寸和 knot spacing 对应。 |
| `field_shape` | 目标形变场的三个正整数尺寸。 |
| `field_voxel_sizes` | 目标网格的三个正数体素尺寸，毫米。 |
| `knot_spacing` | B-spline 的三个正整数间距，目标体素单位。 |
| `affine_forward` | 原输入到目标的有限、可逆、齐次 `4×4` FLIRT 矩阵，不是 scanner RAS 矩阵。 |

`save_fsl_coefficients` 返回写出的路径；`make_fsl_coefficient_image` 参数相同但不含 `filename`，返回 nibabel NIfTI 对象。qform offsets 保存目标场尺寸，pixdim 保存 knot spacing，sform 保存 FLIRT 初始矩阵，intent 2007 的三个参数保存目标体素尺寸。这些是 FSL 系数格式的约定。

## 3. 命令行调用

系数写出是内部步骤，没有独立 CLI。完整入口见 [FNIRT 调用说明](../../docs/fnirt/README.md)。已有配准的 `--cout` 输出自动采用修正后的单位。

## 4. 原软件调用

```bash
fnirt --in=moving.nii.gz --ref=reference.nii.gz \
  --aff=initial_flirt.mat --cout=warp_coefficients.nii.gz
```

本函数对应 `FnirtFileWriter::common_coef_construction` 的系数保存步骤；单位由 NEWIMAGE 的 NIfTI writer 设置。

## 5. 已有真实结果核对与测试

同一真实 GM 配准的既有保存系数文件中，候选的 `xyzt_units=0`，官方为 10；两方系数网格为 `21×24×21×3`，pixdim、qform code、sform code 相同。只读取既有文件头并核对身份，没有重新配准、生成候选脑图或测量新耗时。

修复后，已有官方系数写出 oracle 测试和保存／重读测试 **2/2 通过**，本地测试总时间 3.99 秒；这是测试时间。修复只涉及主机端一个头字段，不是新 CPU／GPU 配准 benchmark。完整真实像素验收仍未通过，旧脑图和完整耗时见 [CPU benchmark](../../docs/fnirt/CPU_BENCHMARK.md)。

### moved 与调制图的参考来源

完整 GM 诊断原官方命令使用 `--cout/--fout/--jout`，没有 `--iout`。其中 moved 参考由后续 `applywarp` 生成，调制参考由 `fslmaths -mul` 生成；候选则由 FNIRT 内部采样和乘法生成终点。原记录的相对 L2 **5.41%／6.54%** 是这两条终点链的差异，包含后续采样，不能单独归因于 FNIRT 参数估计。

真实输入的时间间隔为 2.4 秒，目标模板为 1 秒。官方 applywarp moved／调制图保留 2.4 秒，候选沿用目标头为 1 秒；这不是同一个 FNIRT `--iout` 保存步骤的对照。本次不据此改动 FNIRT 的时间字段。Jacobian 和系数各自仍按同类输出比较。系数单位修复不解决已记录的像素误差。

## 6. 最近更新

| 日期 | 记录 |
|---|---|
| 2026-10-07 | 补齐系数 mm／sec 单位；加强已有官方写出 oracle 的单位断言；2 项 I/O 测试通过；核清已有 moved／调制输出的工具来源。没有重新运行配准。 |

## 7. 原实现与参考文献

- [FSL warpfns](https://git.fmrib.ox.ac.uk/fsl/warpfns)：`fnirt_file_writer.cpp`，仓库固定参考版本 2203.0。
- [FSL newimage](https://git.fmrib.ox.ac.uk/fsl/newimage)：`newimageio.h`，固定参考版本 2203.11；单位逻辑在 NIfTI writer 中。
- Andersson, Jenkinson & Smith, *Non-linear registration, aka spatial normalisation*, FMRIB Technical Report TR07JA2 (2007)。
- 原源码版本与许可见 [固定参考清单](../../src/fnit/_vendor_fsl/README.md)。项目运行时使用 nibabel 写出，不调用上述原程序。
