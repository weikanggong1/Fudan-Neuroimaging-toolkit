# TorchFNIRT：非线性配准

[返回首页](../../README.md) · [dMRI/TBSS pipeline](../dmri_pipeline/README.md)

`TorchFNIRT` 是 FNIT 对 FSL FNIRT 非线性配准核心的 PyTorch 实现。独立命令对应
FSL 6.0.7.4 的 `GM_2_MNI152GM_2mm.cnf` 灰质路径；dMRI pipeline 使用同一核心执行
UK Biobank 的 `oxford_s1.cnf` → `oxford_s2.cnf` → `oxford_s3.cnf` 三阶段 FA
配准。运行时不启动 FSL executable。

CUDA 默认允许 TF32 matmul 和 cuDNN，但图像和写出的 coefficient NIfTI 为 float32，优化器内部 coefficient state 和
主计算为 float64；不使用 float16 或 bfloat16。实际设置写入
`result.qc["tf32"]`。

## 独立命令行：灰质 FNIRT

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

| 参数 | 输入或输出 | 作用 |
|---|---|---|
| `--in` | 输入 | 单张 3D moving 灰质概率图。 |
| `--ref` | 输入 | 单张 3D fixed 灰质模板；决定 `iout`/`jout` 的网格。 |
| `--aff` | 输入，可省略 | 4×4 FSL scaled-mm 矩阵，方向为 input → reference；省略时使用 FSL scaled-mm identity。 |
| `--refmask` | 输入，必需 | reference 网格上的非空二值 mask；必须显式提供，运行时不会读取 `$FSLDIR`。 |
| `--config` | 输入 | 只接受未经修改的 `GM_2_MNI152GM_2mm.cnf` 或该名称。 |
| `--cout` | 输出 | FSL intent-2007 cubic B-spline coefficient NIfTI；它不是 dense warp。 |
| `--iout` | 输出，可省略 | 原始 input 经 affine 和 nonlinear warp 后的 reference-grid 图像。 |
| `--jout` | 输出，可省略 | nonlinear-only Jacobian determinant，不含 FLIRT affine determinant。 |
| `--device` | 运行选项 | `cpu`、`cuda` 或 `cuda:N`；省略时优先 CUDA。 |
| `--overwrite` | 运行选项 | 允许替换已有输出；默认保护已有文件。 |

`cout` 总会写出。省略 `--cout` 时，输入 `subject_GM.nii.gz` 生成
`subject_GM_warpcoef.nii.gz`。不带扩展名的输出遵循 `FSLOUTPUTTYPE=NIFTI` 或
`NIFTI_GZ`。三个输出路径必须不同，也不能覆盖 input、reference、affine 或 mask。

等价的原软件调用形式是：

```bash
fnirt \
  --in=subject_GM.nii.gz \
  --ref=template_GM.nii.gz \
  --aff=subject_GM_to_template_GM.mat \
  --cout=subject_GM_to_template_GM_warp.nii.gz \
  --iout=subject_GM_to_template_GM.nii.gz \
  --jout=subject_GM_JAC_nl.nii.gz \
  --refmask=MNI152_T1_2mm_brain_mask_dil.nii.gz \
  --config=GM_2_MNI152GM_2mm.cnf
```

两条命令的参数角色和输出文件类型一致；`--device`、`--overwrite` 是 FNIT 选项。
当前验证尚未达到逐体素数值等价，因此这里的“等价调用”只表示接口与文件契约对应。

## Python：灰质 FNIRT

```python
from fnit.fnirt.standalone import run_fnirt

result = run_fnirt(
    input="subject_GM.nii.gz",  # 输入：待配准的 3D 个体 GM 概率图
    reference="template_GM.nii.gz",  # 输入：fixed GM 模板及输出网格
    affine="subject_GM_to_template_GM.mat",  # 输入：input -> reference 的 FSL scaled-mm 4x4 矩阵
    cout="subject_GM_to_template_GM_warp.nii.gz",  # 输出：intent-2007 B-spline coefficient NIfTI
    iout="subject_GM_to_template_GM.nii.gz",  # 输出：reference-grid warped GM
    jout="subject_GM_JAC_nl.nii.gz",  # 输出：仅 nonlinear warp 的 Jacobian determinant
    refmask="MNI152_T1_2mm_brain_mask_dil.nii.gz",  # 输入：reference-grid 二值 mask
    config="GM_2_MNI152GM_2mm.cnf",  # 配置：只支持该官方 GM 配置
    device="cuda:0",  # 运行设备：第一张 CUDA GPU
    overwrite=False,  # 写盘策略：不覆盖已有文件
)
```

`result` 是 `TorchFNIRTResult`：

| 属性 | shape/类型 | 含义 |
|---|---|---|
| `coefficient_image` | intent-2007 NIfTI | 与 `cout` 相同，可交给 FNIT `TorchApplyWarp`。 |
| `coefficients` | `[Cx,Cy,Cz,3]` NumPy array | cubic residual coefficient，不是 reference-grid dense displacement。 |
| `moved` | reference-grid `FNITNifti1Image` | 与 `iout` 相同。 |
| `nonlinear_jacobian` | reference-grid `FNITNifti1Image` | 与 `jout` 相同，排除 affine determinant。 |
| `full_pull_jacobian` | reference-grid `FNITNifti1Image` | 包含 affine 的完整 pull Jacobian；FSL `fnirt --jout` 不写这一项。 |
| `pull_transform` | `DenseWarp`（`nibabel.Nifti1Image` 子类） | reference → input 的 world-RAS 毫米位移场，保存 source/target geometry。 |
| `qc` | `dict` | schedule、mask、优化、拓扑、设备、TF32 和数值验证状态。 |

## coefficient 和坐标定义

FLIRT `.mat` 不是 NIfTI world affine。设 `Vin`、`Vref` 是 input/reference voxel
到 FSL scaled-mm 的矩阵，`Win`、`Wref` 是 voxel-to-world affine，`A` 是
`--aff`，则 FNIT 传给配准器的 input → reference world-RAS affine 为：

```text
Wref · inverse(Vref) · A · Vin · inverse(Win)
```

`cout` 的 sform 保存 forward FLIRT affine；qform offset 保存 dense reference field
size；pixdim 保存 knot spacing；intent parameters 保存 dense field voxel size。展开系数
得到 residual `d` 后，FSL scaled-mm pull 关系为：

```text
x_input = inverse(A) · x_reference + d(x_reference)
```

因此 coefficient array 不能当作 `[X,Y,Z,3]` dense warp 使用。`jout` 定义为
`det(I + ∂d_nonlinear/∂x_reference)`。

## 当前实现与 FSL 对应关系

| FSL 行为 | 当前实现 |
|---|---|
| SPM-like mean normalization、float32 image scaling | 相同顺序实现 |
| implicit input/reference zero mask | 使用 `1e-16` 判零；reference mask 在归一化前建立，input mask 在 float32 归一化后建立 |
| warped input mask | 按 FSL `volume<char>` 语义，三线性插值后先截断为 char，再执行 `>0.5` |
| 有效 FOV 边界 | 使用 newimage 的 `1e-8` tolerance、原始 floor index 和 zero-padded neighbor |
| input masked smoothing 与 reference zero-padded Gaussian smoothing | 已实现 |
| cubic B-spline field、bending regularization、SSD-weighted lambda | 已实现 |
| LM stages | matrix-free analytic `JᵀJ` + FP64 PCG |
| `minmet=scg` stages | 按 MISCMATHS `sccngr` 更新顺序实现 |
| `inwarp`/`intin` process handoff | 在内存中执行 float32 coefficient/header、10 位 intensity 交接 |
| `FullResKsp` 与 `ZoomField` | 按每个进程最后 subsampling 计算 full-grid spacing |
| coefficient NIfTI | 输出 FSL intent 2007，shape、spacing、sform 契约已验证 |
| `ForceJacobianRange` | 已移植；外部逐步 topology oracle 仍未通过 |

独立 GM schedule 为 `subsampling=4,2,1,1`、`maxiter=5,5,10,5`、input FWHM
`6,4,2,2 mm`、reference FWHM `4,2,0,0 mm`、bending lambda
`150,75,50,30`、10 mm warp resolution，四层使用 LM。dMRI/TBSS schedule 见
[dMRI 页面](../dmri_pipeline/README.md#ukb-tbss-对应关系)。

## 当前真实数据验证

2026 年 9 月 28 日在 gpucw1 上完成 1 例去标识化真实 UKB 格式 FA 的 matched-input
验证。FSL 6.0.7.4 和当前 TorchFNIRT 使用完全相同的 preprocessed FA、
`FMRIB58_FA_1mm`、FSL scaled-mm affine、implicit zero mask 和
`oxford_s1/s2/s3.cnf` 参数。`dti_FA_mask` 只用于前一步 FLIRT 加权，两个 FNIRT
实现都不接收它。候选源码快照 tar SHA-256 为
`f7547d0a39ddd9fb6ba70deb720f229ecedc6385fa72d457efb2ded78b6c173d`，
`registration.py` 为
`a63ed0b09e43a5af4bf63b2f583e710b1d0fc73aac548a326c552334a741cd83`。
报告记录的包入口 `__init__.py` 为 `cc9aa4…`，当前 0.16.0 为 `be1cab…`；差异包括
版本号、独立的 fMRI 懒加载分支（含 13 个 surface API），以及主动撤下五个内部实现名称。归一化前两项并从新旧源码
中过滤这五个名称后，保留 API 的新旧 AST SHA-256 均为 `fd295c…`；被撤下名称不主张
API 兼容，也不影响 `fnit.fnirt` 的直接调用路径。完整 hash、AST 指纹和 `fresh=false`
边界见[包入口源码等价证明](../../validation/runtime_dependencies/package_entry_source_equivalence.public.json)。
FSL 参考在本轮重新执行三个进程；新旧官方 coefficient 和 warped FA 逐体素完全相同。

TorchFNIRT 在一个 Python 进程内执行相同的六层 schedule 和三次 process handoff。
比较范围为 coefficient 全数组、warped FA 两图非零并集，以及模板非零区内的两类
Jacobian：

| 输出 | 合同 | Pearson r | MAE | RMSE | 最大绝对误差 |
|---|---|---:|---:|---:|---:|
| cubic coefficient | shape/affine/float32/intent-2007 通过 | 0.999893 | 0.018804 | 0.038019 | 1.342859 |
| warped FA (`iout`) | shape/affine/float32 通过 | 0.999203 | 0.003795 | 0.006697 | 0.217295 |
| nonlinear Jacobian (`jout`) | shape/affine/float32 通过 | 0.999064 | 0.010260 | 0.016942 | 0.422846 |
| 含 affine 的完整 Jacobian | shape/affine/float32 通过 | 0.994737 | 0.037611 | 0.050840 | 0.613320 |

四类输出的文件合同均通过，但误差明显大于单纯浮点舍入；报告据此保留
`numerical_equivalence_passed=false`。这表示当前实现不能宣称与 FSL 逐体素数值等价，
同时也说明此前 raw-to-standard 结果不能只用上游输入分叉解释。普通 API 返回的
`qc["equivalence_status"]` 仍写“external numerical gate not passed”：该字段表示一次普通
调用不会自行启动外部 FSL oracle；本次独立报告已经执行外部 gate，结论仍为未达到数值等价。

| 运行 | 实测时间 | 内存 |
|---|---:|---:|
| TorchFNIRT / H100，同步优化核心 | 16.933 s | peak CUDA allocation 3.598 GB |
| TorchFNIRT / H100，进程外部 wall | 25.16 s | max CPU RSS 1,091,028 KiB |
| FSL stage 1 / CPU | 95.20 s | max RSS 794,696 KiB |
| FSL stage 2 / CPU | 795.08 s | max RSS 1,161,748 KiB |
| FSL stage 3 / CPU | 374.86 s | max RSS 1,180,244 KiB |
| FSL 三阶段合计 | 1265.14 s | 上表最大值 |
| FSL 三阶段加 full-affine Jacobian utility | 1273.20 s | max RSS 1,180,244 KiB |

Torch 启动时物理 GPU 0 利用率为 0%，但同卡常驻进程已占 48,310 MiB，余
32,697 MiB。FSL 启动时主机 load average 为 94.86/89.93/88.62。两个计时都不是隔离
benchmark，而且候选为单进程、FSL 为三进程，因此不发布加速比；机器报告保留观测 wall
比值供复核。

![同一真实 FA、模板和 affine 的 FSL 与 TorchFNIRT 对照](figures/fnirt_real_current.png)

完整 3D 指标、输入和配置 SHA-256、12 个源码 SHA-256、命令、环境、计时和显存见
[`report.real.current.json`](../../validation/fnirt/report.real.current.json)。候选执行脚本为
[`run_current_matched.py`](../../validation/fnirt/run_current_matched.py)，官方参考脚本为
[`run_official_matched.sh`](../../validation/fnirt/run_official_matched.sh)，报告生成脚本为
[`validate_real_current.py`](../../validation/fnirt/validate_real_current.py)。仓库不保存原始 FA
或受试者标识。

同一最终源码还完成了从 raw AP/PA 开始的
[TBSS 端到端单例](../../validation/dmri_pipeline/tbss_e2e.real.current.json)。九张 standard
与九张 skeleton 图的网格和 dtype 合同通过，但上游 native 参数图已经分叉，整条流程数值
等价失败；该 444.93 s wall time 受同卡 100% 训练任务影响，也不用于加速比。

## 支持范围与许可

独立 CLI 只支持一个 3D input/reference、可选 4×4 FLIRT affine、binary reference
mask 和官方 `GM_2_MNI152GM_2mm.cnf`。它不接受任意 FNIRT config、DCT basis、
quadratic spline、局部 intensity model 或通用 `--inwarp`。TBSS 的 Oxford schedule
只由 `TorchTBSS` 内部调用。

实现依据 FSL FNIRT 2203.0、basisfield 2203.1、miscmaths 2203.2、newimage
2203.11 和 warpfns 2203.0，受
[`FSL Software Licence, Release 6.0`](../../licenses/FSL-6.0.txt) 约束，仅用于该许可
允许的非商业用途。本项目不是官方 FSL 发布。上游版本、commit 和文件哈希见
[`_vendor_fsl`](../../src/fnit/_vendor_fsl/README.md)。
