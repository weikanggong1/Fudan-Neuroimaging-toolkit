# TorchFNIRT 源码目录

TorchFNIRT 是 FNIT 的 PyTorch FNIRT 核心。`TorchFNIRT()` 与独立命令默认
使用 FSL 不带配置文件的参数；`gm`、`t1`、`tbss` 是专用预设。fMRI volume
默认使用 `t1`，dMRI/TBSS 默认使用 `tbss`，FastVBM 的 GM 配准使用 `gm`。
参数表、输出结构、原软件命令及真实数据对照见
[`docs/fnirt/README.md`](../../../docs/fnirt/README.md)。

## 单被试命令行

```bash
# --in：个体 GM；--ref：模板 GM 与输出网格。
# --aff：input→reference 的 FSL scaled-mm 初始矩阵。
# --cout：形变系数图；--iout：变形 GM；--jout：非线性 Jacobian。
# --refmask：模板网格上的显式 mask；--config：选择 GM 预设。
# --device：PyTorch 计算设备。
python -m fnit.fnirt \
  --in subject_GM.nii.gz \
  --ref template_GM.nii.gz \
  --aff subject_GM_to_template_GM.mat \
  --cout subject_GM_to_template_GM_warp.nii.gz \
  --iout subject_GM_to_template_GM.nii.gz \
  --jout subject_GM_JAC_nl.nii.gz \
  --refmask MNI152_T1_2mm_brain_mask_dil.nii.gz \
  --config gm \
  --device cuda:0
```

不传 `--config` 时使用 FSL 无配置默认值，且不要求 `--refmask`。
T1w 输入改用 `--config t1` 并提供 T1 与 MNI 脑模板及掩膜；TBSS 可用
`--config tbss`。覆盖参数的用法见[配置说明](../../../docs/fnirt/README.md#配置与覆盖规则)。

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
    config="gm",  # 配置：官方 GM schedule；省略时使用 FSL 无配置默认值
    execution="optimized",  # GPU 执行优化；reference 保留原 dense 算子供复核
    device="cuda:0",  # 运行设备
    overwrite=False,  # 不覆盖已有文件
)
```

输入 affine 是 input → reference 的 FSL scaled-mm 4×4 矩阵。`cout` 是 FSL
intent-2007 cubic coefficient NIfTI；`iout` 和 `jout` 位于 reference 网格，`jout`
只含 nonlinear determinant。返回值为 `TorchFNIRTResult`，包含这些输出、完整 pull
Jacobian、pull transform 和 QC。

CUDA 默认允许 TF32；图像和写出的 coefficient NIfTI 使用 float32，优化器内部 state 和主计算使用 float64，不使用 float16/bfloat16。当前实现匹配 FSL 的 implicit-mask 阈值、建立顺序、`volume<char>` warped-mask 截断以及 newimage 的有效 FOV 边界。

当前 T1 六级预设的 FSL 配对、冷/热调用、CUDA profile 和 reference/optimized 数值差异统一记录在[功能页](../../../docs/fnirt/README.md#真实数据验证)与[配准报告](../../../validation/fmri/registration_gpu.current.public.json)。FA/TBSS 另有对应版本的验证，不能用 T1 测量替代。
