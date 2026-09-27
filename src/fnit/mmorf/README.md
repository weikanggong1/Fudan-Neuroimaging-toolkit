# MMORF 模块

[完整说明](../../../docs/mmorf/README.md) · [真实数据验证](../../../validation/mmorf/README.md)

公开的单被试函数是 `run_mmorf`；`fnit-mmorf` 和 dMRI pipeline 的 MMORF 分支都调用它。四幅输入依次为 moving scalar、reference scalar、FSL 六通道 moving tensor 和 reference tensor。三个可选 affine 都采用 input → `reference_scalar` 的 FSL scaled-mm 合同。

```python
from fnit import run_mmorf

result = run_mmorf(
    "t1_brain.nii.gz",
    "MNI152_T1_1mm_brain.nii.gz",
    "dti_tensor.nii.gz",
    "FSL_HCP1065_tensor_1mm.nii.gz",
    moving_scalar_affine="t1_to_MNI.mat",
    moving_tensor_affine="FA_to_MNI.mat",
    output_dir="mmorf",
    device="cuda:0",
)
```

输出为 `mmorf_warp.nii.gz`、`mmorf_jacobian.nii.gz`、`mmorf_warped_scalar.nii.gz`、`mmorf_warped_tensor.nii.gz` 和 `mmorf_report.json`。warp 位于 reference grid，三通道是沿 reference image axes 的毫米 pull displacement；它不能交给 FNIRT `applywarp`。

```bash
fnit-mmorf \
  --mov-scalar t1_brain.nii.gz \
  --ref-scalar MNI152_T1_1mm_brain.nii.gz \
  --mov-tensor dti_tensor.nii.gz \
  --ref-tensor FSL_HCP1065_tensor_1mm.nii.gz \
  --aff-mov-scalar t1_to_MNI.mat \
  --aff-mov-tensor FA_to_MNI.mat \
  -o mmorf --device cuda:0
```

`TorchMMORF` 是不写盘的低层接口；`apply_mmorf_warp` 应用已有 warp。当前实现包含 cubic control lattice、官方 robust sample grid、symmetric scalar/tensor cost、finite-strain tensor reorientation、Bkk/SPRED 和 LBFGS strong-Wolfe。官方使用 sparse full/diagonal Gauss–Newton 与 Levenberg 更新，因此 `mmorf_numerically_equivalent=false`。输入输出、官方配置、算法差异、实际精度、时间和示意图见完整说明。
