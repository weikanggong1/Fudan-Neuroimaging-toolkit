# MMORF 模块

[完整用法](../../../docs/mmorf/README.md) · [真实数据验证](../../../validation/mmorf/README.md)

`run_mmorf` 是公开的单被试函数。标量输入可以是一张 NIfTI，也可以是按模态一一配对的列表；此外输入一组 FSL 六通道张量。第一张 reference scalar 决定输出网格。没有提供线性矩阵时，内部 PyTorchFLIRT 先估计各模态到该网格的 FSL scaled-mm 矩阵。六通道张量先计算 FA，再用于线性估计。

```python
from fnit import run_mmorf

result = run_mmorf(
    moving_scalar=["t1_brain.nii.gz", "dti_FA.nii.gz"],  # 输入：个体 T1 与 FA
    reference_scalar=["MNI152_T1_1mm_brain.nii.gz", "FSL_HCP1065_FA_1mm.nii.gz"],  # 输入：按同一顺序配对的两个模板
    moving_tensor="dti_tensor.nii.gz",  # 输入：个体 [X,Y,Z,6] 张量
    reference_tensor="FSL_HCP1065_tensor_1mm.nii.gz",  # 输入：模板 [X,Y,Z,6] 张量
    output_dir="mmorf",  # 输出：本受试者的注册结果目录
    moving_scalar_affine=[None, None],  # 两组 moving scalar 均自动估计线性矩阵
    reference_scalar_affine=[None, None],  # 首张模板定义空间；第二张同网格时使用 identity
    moving_tensor_affine=None,  # 由 moving tensor 的 FA 自动估计矩阵
    reference_tensor_affine=None,  # reference tensor 同网格时使用 identity
    scalar_weights=[0.5, 0.5],  # 两组标量代价各占 0.5
    auto_linear=True,  # 缺失矩阵调用 PyTorchFLIRT
    device="cuda:0",  # 运行设备
    config=None,  # 默认五层 MMORF 配置
    overwrite=False,  # 已有输出时不覆盖
)
```

`result.warp`、`result.jacobian`、`result.warped_tensor` 是 NIfTI；`result.warped_scalars` 按输入顺序保存所有标量输出，`result.warped_scalar` 是首张图的兼容属性。目录包含 warp、Jacobian、首张标量、张量、JSON 报告，以及第二张以后标量的 `mmorf_warped_scalar_2.nii.gz` 等文件。报告记录全部线性矩阵、耗时与显存。warp 是 reference image axes 毫米单位的 pull displacement，不能交给 FNIRT `applywarp`。

`TorchMMORF` 是不写盘的低层接口；`apply_mmorf_warp` 应用已有 warp。非线性目标包含逐模态初始代价缩放、symmetric scalar/tensor cost、finite-strain tensor reorientation 和 Bkk/SPRED。优化器使用 LBFGS strong-Wolfe，官方使用 sparse Gauss–Newton 与 Levenberg；报告保留 `mmorf_numerically_equivalent=false`。完整输入输出、官方命令、真实数据精度和时间见上方链接。

## 复用 MMORF 九图传播的坐标（2026-10-02）

`prepare_mmorf_warp()` 将固定的 MMORF reference-axis 毫米场和 FSL affine 转为一次采样计划。`MMORFWarpPlan.apply()` 对每张图独立调用原 shape 的 sampler；默认 linear 插值、零边界填充、`align_corners=True` 和 float32 输出保持原算法。保持原混合精度：矩阵与位移转换使用 float64，基础网格、最终采样坐标及图像使用 float32；reference axes 极分解也保留原 float32 计算。

```python
import nibabel as nib
import numpy as np
from fnit.mmorf import prepare_mmorf_warp

native_fractional_anisotropy = nib.load("dti_FA.nii.gz")  # 定义 native dMRI 网格
native_mean_diffusivity = nib.load("dti_MD.nii.gz")  # 同网格 MD 指标图
standard_t1_template = nib.load("MNI152_T1_1mm_brain.nii.gz")  # 输出参考图
shared_mmorf_warp = nib.load("mmorf_warp.nii.gz")  # 固定三通道 reference-axis mm pull field
native_to_standard_affine = np.loadtxt("dti_FA_to_MNI_affine.mat")  # FSL input→reference 矩阵
sampling_plan = prepare_mmorf_warp(
    image=native_fractional_anisotropy,  # 只用于确定源图空间几何
    reference=standard_t1_template,  # 输出网格和 header
    warp=shared_mmorf_warp,  # 本次计划捕获的形变
    affine=native_to_standard_affine,  # 与独立 apply_mmorf_warp 相同的 FSL affine
    device="cuda:0",  # CPU 或 CUDA 设备
    interpolation="linear",  # 可选 nearest/nn 或 cubic；在计划创建时固定
)
standard_mean_diffusivity = sampling_plan.apply(
    image=native_mean_diffusivity,  # 独立采样这一张图
    reference=standard_t1_template,  # 可省略；提供时精确核验几何和完整 header
)
nib.save(standard_mean_diffusivity, "MD.nii.gz")  # 输出同 reference 网格的 float32 NIfTI
```

`prepare_mmorf_warp` 与 `apply_mmorf_warp` 接受相同输入参数，返回可复用计划。`apply()` 接受 3D/4D NIfTI及可选 reference，返回原 float32 NIfTI。输入 shape、affine、header pixdim、FSL scaled-mm 坐标必须完全相同；参考图还核验其类型、完整 header 和 extensions。不同 native 网格分别准备；变更 warp、affine 或插值时创建新计划。reference axes 极分解、warp H2D、坐标转换和归一化 grid 可复用，图像强度与 cubic prefilter 仍逐图计算。

原 `apply_mmorf_warp` API 和独立 MMORF 命令行保持兼容；它也沿用相同计划/采样代码，但单次调用不会跨图持久缓存。复用不会改变 MMORF 非线性估计或官方等价结论。

真实九图验收于 2026-10-02 在 H100 完成：固定 MMORF warp 与 affine，输出 `182×218×182`，CPU 线程 8，PyTorch CUDA 分配限额 20,000,000,000 bytes。三轮每轮交替先后顺序，全部九图的解码体素、完整 header 和 affine 逐值一致。首轮含冷启动，原逐图调用/采样计划为 `0.645693060/0.224990048 s`；后两轮热调用中位数为 `0.517138056/0.270091293 s`，观察到约 **1.91 倍**速度。计时包含计划准备、九图独立采样与输出回传，输入加载和输出写盘在计时外；结果适用于固定形变的传播阶段，原混合精度保持不变。

复现脚本与完整三轮见[九图验收](../../../validation/dmri_pipeline/map_propagation_20261002.md)，整体进展见[主报告](../../../validation/dmri_pipeline/lossless_20261002.md)。
