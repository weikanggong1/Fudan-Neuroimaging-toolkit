# TorchFNIRT

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
    "subject_GM.nii.gz",
    "template_GM.nii.gz",
    "subject_GM_to_template_GM.mat",
    cout="subject_GM_to_template_GM_warp.nii.gz",
    iout="subject_GM_to_template_GM.nii.gz",
    jout="subject_GM_JAC_nl.nii.gz",
    refmask="MNI152_T1_2mm_brain_mask_dil.nii.gz",
    device="cuda:0",
)
```

输入 affine 是 input → reference 的 FSL scaled-mm 4×4 矩阵。`cout` 是 FSL
intent-2007 cubic coefficient NIfTI；`iout` 和 `jout` 位于 reference 网格，`jout`
只含 nonlinear determinant。返回值为 `TorchFNIRTResult`，包含这些输出、完整 pull
Jacobian、pull transform 和 QC。

CUDA 默认允许 TF32；图像和写出的 coefficient NIfTI 使用 float32，优化器内部 state 和主计算使用
float64，不使用 float16/bfloat16。当前实现匹配 FSL 的 implicit-mask 阈值、建立顺序、
`volume<char>` warped-mask 截断以及 newimage 的有效 FOV 边界。一个真实 TBSS 病例中，
默认 TorchFLIRT 路径的九图 standard `r=0.995086–0.998886`，official-affine 隔离路径为
`0.997483–0.999523`；完整 stage-1 coefficient `r=0.999832`。本例未触发 topology
projection，验证病例数为一，因此不声明跨输入逐体素数值等价。完整证据见
[`docs/fnirt/README.md`](../../../docs/fnirt/README.md)。
