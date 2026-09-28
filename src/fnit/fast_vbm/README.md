# FastVBM 模块

[完整说明](../../../docs/fast_vbm/README.md) · [验证状态](../../../validation/fast_vbm/README.md)

`fnit.fast_vbm.FastVBM` 是单被试 raw T1w 到 modulated GM 的 Python API：

```text
raw T1w
  -> SynthStrip 或显式 input-grid brain mask
  -> TorchFAST 三组织 PVE + bias correction
  -> TorchFLIRT 12-DOF correlation ratio
  -> PyTorch SynthMorph deform 或 PyTorch TorchFNIRT GM config
  -> FSL relative pull conversion + GPU TorchApplyWarp
  -> nonlinear-only Jacobian
  -> warped GM * Jacobian
```

## Python

```python
from fnit import FastVBM

pipeline = FastVBM(
    device="cuda:0",  # 计算设备；可改为 "cpu"
    threads=4,  # 读写和部分预后处理使用的 CPU 线程数
    registration_backend="fnirt",  # 非线性后端；可改为 "synthmorph"
)

result = pipeline.run(
    image="subject_T1w.nii.gz",  # 输入：单幅原始 3D T1w
    template="template_GM.nii.gz",  # 输入：fixed GM template 和输出网格
    output_dir="results/sub-01",  # 输出：本被试结果目录
    brain_mask=None,  # 可选输入：None 时运行 SynthStrip
    reference_mask="MNI152_T1_2mm_brain_mask_dil.nii.gz",  # 输入：template 网格 mask
    overwrite=False,  # 是否覆盖已有输出
)
```

`image` 和 `template` 可传路径或 `nibabel.spatialimages.SpatialImage`。`brain_mask` 必须与 T1w 同网格，提供时跳过 SynthStrip；`reference_mask` 必须与 template 同网格。`run()` 保存结果并返回 `FastVBMResult`。只需内存结果时调用：

```python
result = pipeline(
    image="subject_T1w.nii.gz",  # 输入：单幅原始 3D T1w
    template="template_GM.nii.gz",  # 输入：fixed GM template
    brain_mask=None,  # 可选输入：None 时运行 SynthStrip
    reference_mask="MNI152_T1_2mm_brain_mask_dil.nii.gz",  # 输入：template 网格 mask
)
```

## 命令行

```bash
fnit fast-vbm \
  -i subject_T1w.nii.gz \
  --template template_GM.nii.gz \
  -o results/sub-01 \
  --registration-backend fnirt \
  --reference-mask MNI152_T1_2mm_brain_mask_dil.nii.gz \
  --device cuda:0 \
  --threads 4
```

`-i` 是单帧 raw T1w；`--template` 决定模板空间输出网格；`-o` 是单个受试者的输出目录；`--registration-backend` 选择 `fnirt` 或 `synthmorph`。完整参数见 `fnit fast-vbm --help`。

## 后端与 mask

两后端共用 TorchFAST GM、TorchFLIRT、坐标转换、TorchApplyWarp、Jacobian 和 modulation。SynthMorph 运行官方 deform 网络，不消费 reference mask。TorchFNIRT 使用 cubic B-spline GM schedule：

- 与 FSL GM 配置一致，关闭 implicit reference 和 input mask；零值不自动排除；
- 显式 `reference_mask` 在默认 GM schedule 的最后一级使用。

省略 `reference_mask` 时，FastVBM 使用 `template > 0` 的派生 mask。复现 FSL/UKB 时应传入原运行实际使用的 mask。完整链尚未达到数值等价。

## 输出

`FastVBMResult` 包含 `brain`、`brain_mask`、`fast`、`registration`、`pve_gm`、`warped_gm`、`jacobian`、`modulated_gm`、`settings` 和 `timing_sec`。`run()` 写出：

| 键 | 文件名 | 网格 |
|---|---|---|
| `brain` | `T1_brain.nii.gz` | 输入 T1 |
| `brain_mask` | `brain_mask.nii.gz` | 输入 T1 |
| `pve_csf` | `T1_brain_pve_0.nii.gz` | 输入 T1 |
| `pve_gm` | `T1_brain_pve_1.nii.gz` | 输入 T1 |
| `pve_wm` | `T1_brain_pve_2.nii.gz` | 输入 T1 |
| `hard_segmentation` | `T1_brain_seg.nii.gz` | 输入 T1 |
| `pve_segmentation` | `T1_brain_pveseg.nii.gz` | 输入 T1 |
| `mixel_type` | `T1_brain_mixeltype.nii.gz` | 输入 T1 |
| `bias_field` | `T1_brain_bias.nii.gz` | 输入 T1 |
| `restored` | `T1_brain_restore.nii.gz` | 输入 T1 |
| `warped_gm` | `T1_GM_to_template_GM.nii.gz` | GM template |
| `jacobian` | `T1_GM_JAC_nl.nii.gz` | GM template |
| `modulated_gm` | `T1_GM_to_template_GM_mod.nii.gz` | GM template |

另写 `fast_vbm_report.json`。`jacobian` 排除 FLIRT affine determinant，`modulated_gm = warped_gm * jacobian`。

## UKB/FSL 对应命令

```bash
fsl_reg T1_brain_pve_1.nii.gz template_GM.nii.gz \
  T1_GM_to_template_GM -fnirt \
  "--config=GM_2_MNI152GM_2mm.cnf --jout=T1_GM_JAC_nl"

fslmaths T1_GM_to_template_GM -mul T1_GM_JAC_nl \
  T1_GM_to_template_GM_mod -odt float
```

UKB 命令从已有 FSL FAST GM 开始；FastVBM 从 raw T1w 开始。文件角色、template grid 和 modulation 公式对应，脑提取、GM estimation 和配准优化器不是同一数值实现。

## 当前验证状态

单例固定 FSL GM 输入的掩膜诊断及本版原始 T1w 完整链复测见[验证页](../../../validation/fast_vbm/README.md)。旧配置下的 FNIRT 数值报告不适用于本版；当前源码尚未完成多例配对 benchmark。
