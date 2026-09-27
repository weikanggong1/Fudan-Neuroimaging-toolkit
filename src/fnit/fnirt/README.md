# TorchFNIRT 源码目录

TorchFNIRT 是 FNIT 的 PyTorch FNIRT 核心。独立接口对应 FSL
`GM_2_MNI152GM_2mm.cnf`；`TorchTBSS` 使用同一核心执行 UK Biobank Oxford
三阶段 FA schedule。完整输入输出、坐标定义、原软件命令、真实数据 benchmark 和
当前数值边界见 [`docs/fnirt/README.md`](../../../docs/fnirt/README.md)。

## 单被试命令行

```bash
python -m fnit.fnirt \
  --in subject_GM.nii.gz \
  --ref template_GM.nii.gz \
  --aff subject_GM_to_template_GM.mat \
  --cout subject_GM_to_template_GM_warp.nii.gz \
  --iout subject_GM_to_template_GM.nii.gz \
  --jout subject_GM_JAC_nl.nii.gz \
  --refmask MNI152_T1_2mm_brain_mask_dil.nii.gz \
  --config GM_2_MNI152GM_2mm.cnf \
  --device cuda:0
```

## Python

```python
from fnit.fnirt.standalone import run_fnirt

result = run_fnirt(
    input="subject_GM.nii.gz",  # 输入：个体 3D GM 概率图
    reference="template_GM.nii.gz",  # 输入：GM 模板及输出网格
    affine="subject_GM_to_template_GM.mat",  # 输入：input -> reference 的 FSL scaled-mm 矩阵
    cout="subject_GM_to_template_GM_warp.nii.gz",  # 输出：intent-2007 coefficient NIfTI
    iout="subject_GM_to_template_GM.nii.gz",  # 输出：warped GM
    jout="subject_GM_JAC_nl.nii.gz",  # 输出：nonlinear Jacobian determinant
    refmask="MNI152_T1_2mm_brain_mask_dil.nii.gz",  # 输入：reference-grid 二值 mask
    config="GM_2_MNI152GM_2mm.cnf",  # 配置：官方 GM schedule
    device="cuda:0",  # 运行设备
    overwrite=False,  # 不覆盖已有文件
)
```

输入 affine 是 input → reference 的 FSL scaled-mm 4×4 矩阵。`cout` 是 FSL
intent-2007 cubic coefficient NIfTI；`iout` 和 `jout` 位于 reference 网格，`jout`
只含 nonlinear determinant。返回值为 `TorchFNIRTResult`，包含这些输出、完整 pull
Jacobian、pull transform 和 QC。

CUDA 默认允许 TF32；图像和写出的 coefficient NIfTI 使用 float32，优化器内部 state 和主计算使用 float64，不使用 float16/bfloat16。当前实现匹配 FSL 的 implicit-mask 阈值、建立顺序、`volume<char>` warped-mask 截断以及 newimage 的有效 FOV 边界。

当前 `registration.py` 已用 1 例真实 FA 完成 matched-input 验证：TorchFNIRT 与 FSL 6.0.7.4 固定相同 FA、FMRIB58_FA_1mm、FSL affine 和 Oxford 三阶段配置。coefficient、warped FA、nonlinear Jacobian、含 affine Jacobian 的 Pearson r 分别为 `0.999893`、`0.999203`、`0.999064`、`0.994737`；shape、affine、float32 与 coefficient intent-2007 合同通过。误差超过浮点舍入，因此数值等价仍判定失败。H100 同步优化核心为 `16.933 s`、进程外部 wall 为 `25.16 s`、peak CUDA allocation 为 `3.598 GB`；FSL 三阶段 CPU wall 合计 `1265.14 s`。两侧运行均未隔离，不发布加速比。完整报告、输入边界、时间限制和图见 [`docs/fnirt/README.md`](../../../docs/fnirt/README.md)。
