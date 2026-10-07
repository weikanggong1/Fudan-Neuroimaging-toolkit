# robust-register：CPU 保序质心候选的真实对照

## 1. 功能简介

`centroid_serial.py` 按固定 FreeSurfer 源码定义的顺序计算强度加权质心：Z、Y、X 三层循环，X 最快；Float32 体素转 Double 后逐次累加，最后除以总强度。Numba 关闭 fastmath、并行归约和缓存。

真实准备图像中，当前 PyTorch 质心与独立 SDK 参考存在约 `2e-13` 的尾差。两种实现采用不同的 Double 加法分组。本候选保留串行累加顺序，两幅图像的 6 个 Double 值均与保存的 SDK 参考逐位相同。

这是独立验证函数，尚未接入默认配准。本报告对应的旧初始化使用原 PyTorch 质心，M0 最大绝对差为 `4.121147867408581e-13`；后续[完整初始化对照](../centroid_m0_cpu_prefix_20261006/README.md)已接入该保序内核，两图及全部54个Double初始化值逐位匹配SDK参考。正式完整 robust-register 验收仍为 **17/20**。SDK 参考由固定源码局部构建并捕获，不能称为安装版可执行文件的运行轨迹。

## 2. Python 调用、输入与输出

函数：`centroid_serial(values, width, height, depth, outside)`。

| 参数 | 格式与意义 |
| --- | --- |
| `values` | 一维、连续、只读 Float32 数组，长度为 `width * height * depth`。顺序为 X 最快、随后 Y、Z；对应 XYZ 三维数组的 `ravel(order="F")`。 |
| `width` | XYZ 网格的 X 维大小，正整数。 |
| `height` | XYZ 网格的 Y 维大小，正整数。 |
| `depth` | XYZ 网格的 Z 维大小，正整数。 |
| `outside` | Double 背景值。原条件为 `abs(value - outside) < outside / 255.0` 时将值置零。本次两图实际为正零。 |

返回 `(x, y, z)` 三个 Double，使用**从 1 开始的体素坐标**。它不返回 scanner RAS 或毫米坐标，也不读取或变更影像 affine。函数无默认参数。调用者须确认数组长度、Float32 类型、有限值以及正的总强度；本次只验收保存的两幅真实准备图像，质心输出均有限；未单独记录全数组有限性或总强度数值。未知输入和生产异常语义尚待独立验证。

下面示例读取已经准备好的 Float32 XYZ 图像。它不包含配准所需的转换、重采样和裁剪；原始 T1 不能直接当作本次准备态输入。

```python
from pathlib import Path
import importlib.util
import nibabel as nib
import numpy as np
from numba import types

# 从仓库 validation 目录显式加载候选，不替换 FNIT 默认后端。
candidate_file = Path(
    "validation/robust_register/centroid_serial_cpu_probe_20261006/centroid_serial.py"
).resolve()
module_spec = importlib.util.spec_from_file_location("centroid_cpu_probe", candidate_file)
candidate_module = importlib.util.module_from_spec(module_spec)
module_spec.loader.exec_module(candidate_module)

# 用户提供已完成准备、满足有限值和正质量条件的 Float32 XYZ 图像。
prepared_image_file = Path("prepared_float32.nii.gz")
prepared_image = nib.load(prepared_image_file)
prepared_values_xyz = np.asanyarray(prepared_image.dataobj)
assert prepared_values_xyz.ndim == 3
assert prepared_values_xyz.dtype == np.dtype("float32")
width, height, depth = prepared_values_xyz.shape
prepared_values_flat = np.array(
    prepared_values_xyz.ravel(order="F"), dtype=np.float32, copy=True
)
prepared_values_flat.setflags(write=False)
outside_value = np.float64(0.0)

# 冷编译单列；与本次真实对照一致，使用只读 Float32 数组和 Double 背景。
centroid_signature = (
    types.Array(types.float32, 1, "C", readonly=True),
    types.int64, types.int64, types.int64, types.float64,
)
candidate_module.centroid_serial.compile(centroid_signature)
centroid_xyz_one_based = candidate_module.centroid_serial(
    values=prepared_values_flat,
    width=width,
    height=height,
    depth=depth,
    outside=outside_value,
)
print(centroid_xyz_one_based)
```

NumPy 和 Numba 已包含在项目 Conda 环境中，本次未新增依赖。公开函数除模块说明外，其完整 AST 与实际测试的候选相同；实际测试源码 SHA-256 见 [benchmark_summary.json](benchmark_summary.json)。

## 3. 命令行调用

此内部质心候选没有独立生产 CLI。默认 `fnit.robust_register` 接口保持原样。本次通过私密受控入口读取此前保存的两幅准备数组；未执行完整注册、GPU、QR、Schur 或新的 M0 计算。

## 4. 原软件调用与参考边界

对应内部函数是 `CostFunctions::centroid`，由 `mri_robust_register` 的准备与初始化调用，原软件也没有独立质心命令。完整配准的原软件命令形式为：

```bash
# 仅说明原软件隔离对照的调用形式；本次没有重新执行完整命令。
mri_robust_register --mov moving.mgz --dst fixed.mgz --lta output.lta --sat 50
```

本次复用同一真实刚体 case 已保存的独立 SDK 准备图像和 6 个 Double 质心值；没有重新运行原生准备。参考绑定 FreeSurfer 源码提交 `d932c45b7941662ea380a05efef580568b98d41a`。SDK、安装版 FreeSurfer 和 FNIT 默认实现的身份分别保留。

## 5. 真实精度、耗时与可视化

两幅图像均为 `39 × 45 × 61`、107,055 个 Float32 体素，X 最快。CPU 总线程预算为 8，质心内核单线程；地址空间上限为 20,000,000,000 字节。

| 准备图像 | 原 PyTorch 质心最大绝对差 | 保序候选最大绝对差 | Double 逐位对照 | 单次内核耗时 |
| --- | ---: | ---: | --- | ---: |
| source | `1.9184653865522705e-13` | 0 | 3/3 相同 | 0.283333 ms |
| target | `2.2026824808563106e-13` | 0 | 3/3 相同 | 0.227768 ms |

原 PyTorch 数值也来自已保存结果，本次未重新归约。实际候选运行前的第三方导入为 **0.298878 s**，冷签名编译为 **0.391598 s**；两幅图各执行一次。数值结果在运行库观察门前持久保存。没有独立原生质心的时间测量，因此不报告相对于原生的速度比，也不把这些内核时间写成完整配准耗时。

当前结果是两个质心点的数值对照，未重采样或生成新脑图；不产生新的配准可视化。完整刚体/仿射的已有结果与脑图范围见 [既有单例报告](../rigid_affine_20261006/README.md#5-真实精度时间与可视化)。后续完整 M0 接入已完成同例初始化对照，见[新报告](../centroid_m0_cpu_prefix_20261006/README.md)；完整配准正式门验证仍待完成。

## 6. 更新与 benchmark 记录

| 记录 | 实际状态 |
| --- | --- |
| centroid v1 | 数值调用后观察器按库名称拒绝既有 Conda 的间接 MKL 依赖，结果未保存；没有精度结论。原失败保留。 |
| centroid v2 | 原候选数学字节不变；两图各一次，6 个 Double 真值已保存并逐位相同。观察器遗漏冷编译时加载的 14 个既有环境依赖，原 `actual_DSO_exact=false`、原组失败。 |
| metadata v3 | 仅核查保存的 91 个实际库身份、474 个资源绑定以及 6 个整数位模式，独立元数据 seal 通过。新导入/JIT/内核/准备/原生/M0/GPU 调用全部为 0；原 v1/v2 失败原件不改。 |
| CPU 初始化接入 | 只在规划中。拟对已证明条件的 CPU Float32 准备态使用保序质心；GPU 及未知输入保留原函数。尚未执行该前缀，更未替换默认。 |

公开聚合记录分别保存 `original_v2_actual_DSO_exact=false`、`original_v2_group_passed=false` 和 `metadata_seal_passed=true`。库已加载的证据与静态 LLVM 无 BLAS/LAPACK/MKL 调用名称的证据单列；它们不构成动态 BLAS 调用轨迹。

## 7. 参考文献、原代码与许可

- [固定源版本的 CostFunctions.cpp](https://github.com/freesurfer/freesurfer/blob/d932c45b7941662ea380a05efef580568b98d41a/mri_robust_register/CostFunctions.cpp#L1308)：原始串行质心循环和 Double 类型。
- [FreeSurfer robust-register 文档](https://surfer.nmr.mgh.harvard.edu/fswiki/mri_robust_register)：原命令与算法。
- Reuter M, Rosas HD, Fischl B. Highly accurate inverse consistent registration: a robust approach. *NeuroImage* (2010). [doi:10.1016/j.neuroimage.2010.07.020](https://doi.org/10.1016/j.neuroimage.2010.07.020)。
- [FreeSurfer 软件许可](https://surfer.nmr.mgh.harvard.edu/fswiki/FreeSurferSoftwareLicense)。此目录仅发布自有 Python 候选和去私密的聚合报告，未发布 SDK 源码、可执行文件、影像或私密运行收据。
