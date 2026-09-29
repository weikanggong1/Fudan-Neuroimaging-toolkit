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
