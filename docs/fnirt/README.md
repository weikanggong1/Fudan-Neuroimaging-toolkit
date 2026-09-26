# TorchFNIRT

`TorchFNIRT` 是 FSL 6.0.7.4 中 FNIRT 灰质配准路径的 PyTorch 实现。本页面的
独立接口只实现 `GM_2_MNI152GM_2mm.cnf`：输入灰质概率图、灰质模板、可选 FLIRT
仿射矩阵和 binary reference mask，输出 cubic B-spline 系数、配准后的输入图以及
非线性 Jacobian。运行时不调用 FSL。

CUDA 运行默认允许 TF32 matmul 和 cuDNN 内核，同时保留实现声明的
float32/float64 张量；不使用 float16 或 bfloat16。实际开关写入
`result.qc["tf32"]`。

## 命令行调用

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

每一行的作用如下：

| 参数 | 作用 |
|---|---|
| `python -m fnit.fnirt` | 启动本包的单被试 PyTorch FNIRT，不调用外部 `fnirt` executable。 |
| `--in` | moving/input 灰质概率图；配准要把它变换到 `--ref`。 |
| `--ref` | fixed/reference 灰质模板；决定 `--iout` 和 `--jout` 的 shape、voxel size 和空间网格。 |
| `--aff` | 可选 FLIRT `.mat`；方向是 input → reference，坐标是 FSL scaled-mm。代码先转换成带 source/target geometry 的 world-RAS affine，再传给 `TorchFNIRT`。省略时使用 FSL scaled-mm identity。 |
| `--cout` | cubic B-spline residual coefficient 文件。它是 FSL intent `2007`，不是 dense warp。省略时与 FSL 一样，从 input 文件名生成 `<input>_warpcoef`。 |
| `--iout` | 原始 `--in` 经仿射和非线性变换后，在 reference 网格上的图像。优化阶段的平滑图不会写到这里。 |
| `--jout` | 非线性位移的 Jacobian determinant；定义及是否包含 affine 与 FSL `fnirt::SaveJacobian` 对应，不包含 FLIRT affine determinant。体素值差异见验证结果。 |
| `--refmask` | reference 网格上的 binary `0/1` mask。GM 配置只在最后一级按 `applyrefmask=0,0,0,1` 使用它。省略时仅在 `FSLDIR/data/standard/` 可找到官方 mask 的情况下自动使用。 |
| `--config` | 只接受官方、未修改的 `GM_2_MNI152GM_2mm.cnf` 或该名称。其他配置会直接报错。 |
| `--device` | `cpu`、`cuda` 或 `cuda:N`；CLI 和 Python API 均默认优先使用可用 CUDA。 |
| `--overwrite` | 允许替换已有输出。未给出时，任何一个输出已存在都会在模型运行前报错。 |

`cout` 始终生成；省略 `--cout` 时，`subject_GM.nii.gz` 对应默认文件
`subject_GM_warpcoef.nii.gz`。输出 root 可以不带扩展名：`FSLOUTPUTTYPE=NIFTI`
补 `.nii`，`NIFTI_GZ` 或未设置时补 `.nii.gz`；其他 `FSLOUTPUTTYPE` 会明确报错。
三个输出必须使用不同路径，也不能覆盖 input、reference、affine 或 mask。输出先
写入同目录临时文件。未指定 `--overwrite` 时以原子 hard-link 创建最终文件，若
并行任务发生路径冲突，本次已创建的输出会回滚，已有文件不会被覆盖。

对应的 FSL 命令是：

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

两条命令的参数角色相同。额外的 `--device` 和 `--overwrite` 是本包选项。

## Python 调用

```python
from fnit.fnirt.standalone import run_fnirt

result = run_fnirt(
    "subject_GM.nii.gz",                 # input/moving GM
    "template_GM.nii.gz",                # reference/fixed GM
    "subject_GM_to_template_GM.mat",     # FSL scaled-mm input -> reference
    cout="subject_GM_to_template_GM_warp.nii.gz",
    iout="subject_GM_to_template_GM.nii.gz",
    jout="subject_GM_JAC_nl.nii.gz",
    refmask="MNI152_T1_2mm_brain_mask_dil.nii.gz",
    config="GM_2_MNI152GM_2mm.cnf",
    device="cuda:0",
    overwrite=False,
)
```

`affine` 可以传 `None`，含义与 FSL 省略 `--aff` 相同，即使用 scaled-mm identity。
`device=None` 时与 CLI 一样自动选择 CUDA，否则使用 CPU。若 `input` 是文件路径，
省略 `cout` 会自动生成同目录的 `<input>_warpcoef`；若 `input` 是内存中的
`surfa.Volume`，必须显式给出 `cout`，因为没有文件名可用于派生输出路径。

返回值是 `TorchFNIRTResult`。主要内存结果为：

| 属性 | 含义 |
|---|---|
| `result.coefficient_image` | 与 `--cout` 相同的 intent-2007 NIfTI image。 |
| `result.coefficients` | `[Cx, Cy, Cz, 3]` coefficient array。 |
| `result.moved` | 与 `--iout` 相同的 reference-grid `surfa.Volume`。 |
| `result.nonlinear_jacobian` | 与 `--jout` 相同的非线性 Jacobian。 |
| `result.full_pull_jacobian` | 包含 affine 的完整 pull Jacobian；FSL `fnirt --jout` 不写这一项。 |
| `result.pull_transform` | reference → input 的 fixed-grid world-RAS pull warp。它不能作为 FSL scaled-mm warp array 直接使用。 |
| `result.qc` | 优化层级、拓扑约束和当前数值验证状态。 |

## 坐标和 coefficient 文件

FLIRT `.mat` 不是 NIfTI world affine。设 `Vin`、`Vref` 是 input/reference voxel
到 FSL scaled-mm 的矩阵，`Win`、`Wref` 是对应 voxel-to-world affine，`A` 是
`--aff`，则传给 `TorchFNIRT` 的 input → reference world-RAS affine 为：

```text
Wref · inverse(Vref) · A · Vin · inverse(Win)
```

`--cout` 使用 FSL cubic coefficient intent `2007`。它的 sform 保存 input →
reference FLIRT affine；qform offset 保存 dense reference field size；pixdim 保存
knot spacing；intent parameters 保存 dense field voxel size。设 `A` 为 sform 中的
forward FLIRT affine，`x_ref`、`x_in` 为 FSL scaled-mm 坐标，展开系数得到的 residual
field 为 `d`，则 coefficient 文件定义的 pull 关系是：

```text
x_in = inverse(A) · x_ref + d(x_ref)
```

因此 residual 的定义域是 reference grid，分量位于 input scaled-mm；不能把
coefficient array 当作 `[X,Y,Z,3]` dense displacement。
本包的 `TorchApplyWarp` 可以直接读取该文件。

`--jout` 是
`det(I + ∂d_nonlinear / ∂x_reference)`。它排除 affine determinant，对应 FSL
FNIRT `SaveJacobian` 和 UKB VBM 非线性 modulation 使用的 nonlinear-only 量。

## 固定配置

内置参数对应 FSL 6.0.7.4 的官方配置：

| 项目 | 值 |
|---|---|
| subsampling | `4,2,1,1` |
| maximum iterations | `5,5,10,5` |
| input/reference FWHM, mm | `6,4,2,2` / `4,2,0,0` |
| bending regularization | `150,75,50,30`，按当前 SSD 加权 |
| intensity model | `global_linear`，前三层估计，最后一层固定 |
| warp resolution | `10,10,10 mm` |
| reference mask schedule | `0,0,0,1` |
| implicit input/reference masking | 关闭 |
| prescribed Jacobian range | `0.2,5` |

若 `--config` 指向一个实际文件，文件名和 SHA-256 都必须与官方
`GM_2_MNI152GM_2mm.cnf` 一致。这样不会把修改过的配置静默当成已实现配置。

FSL 的 `ForceJacobianRange` 不保证最终范围严格落在 `0.2–5`。达到最大尝试次数后
若仍有少量体素越界，原生 FNIRT 会打印 warning 并继续写出结果。本包默认采用同一
语义，并在 `result.qc["levels"][...]["topology_projection"]` 保存实际范围与
`succeeded` 状态。直接构造 `TorchFNIRT(strict_topology=True)` 可把该 warning
提升为 `RuntimeError`，用于诊断；独立 CLI 和 `run_fnirt()` 使用 FSL 默认语义。

## 实现边界

当前接口只接收单个 3D input/reference、一个可选 4×4 FLIRT affine 和一个
binary reference mask。不支持 `T1_2_MNI152_2mm.cnf`、FA 配置、自定义 schedule、
`--inwarp`、`--intin`、`--inmask`、`--fout`、`--refout`、局部 intensity model、
DCT basis 或 quadratic spline。CLI 遇到这些参数会报告未识别参数；Python API
遇到其他 config 会抛出 `NotImplementedError`，不会退化为近似实现。

## 许可和验证门

本实现根据 FSL FNIRT 2203.0、basisfield 2203.1、miscmaths 2203.2、newimage
2203.11 和 warpfns 2203.0 修改，受
[`FSL Software Licence, Release 6.0`](../../licenses/FSL-6.0.txt) 约束，仅限
该许可允许的非商业用途。本项目不是官方 FSL 发布。未修改的上游源码和版本、
commit、文件哈希保存在
[`_vendor_fsl`](../../src/fnit/_vendor_fsl/README.md)。

intent-2007 header、ZoomField 和 bending energy 使用 FSL 生成的 oracle fixture；
B-spline expansion 与 Jacobian 有解析/组件测试。topology 路径目前覆盖 identity 和
无需投影的 affine 回归，仓库中的 C++ topology oracle 尚未纳入自动门，因此不据此声称
逐元素复现。0.9 的真实 GM 外部门使用相同 input、reference、
FLIRT matrix 和官方 reference mask，比较 coefficient-expanded residual、`iout`、
`jout` 和 modulated GM。文件合同 10/10 一致，数值结果高度相关，但误差不只来自
末位舍入。因此 `TorchFNIRT.qc["fsl_fnirt_numerically_equivalent"]` 保持 `False`。

### 当前 FSL 6.0.7.4 十例 matched-input 对照

10 例真实 T1w-derived FSL FAST GM 均使用官方
`GM_2_MNI152GM_2mm.cnf`、同一 UKB GM template、同一 FSL FLIRT matrix 和同一
`MNI152_T1_2mm_brain_mask_dil`。下表是完整 coefficient grid 或 reference grid
上的十例中位数；最大误差一列也是逐例 maximum absolute error 的中位数。

| 输出 | Pearson r | MAE | RMSE | 最大绝对误差 |
|---|---:|---:|---:|---:|
| cubic coefficients | 0.999089 | 0.033959 | 0.074783 | 2.106445 |
| expanded nonlinear residual | 0.999508 | 0.028455 | 0.051817 | 0.980345 |
| warped GM (`iout`) | 0.998695 | 0.003101 | 0.013682 | 0.642930 |
| nonlinear Jacobian (`jout`) | 0.999267 | 0.003395 | 0.007417 | 0.240333 |
| modulated GM | 0.998507 | 0.003759 | 0.017655 | 1.186854 |

五类输出的 shape、affine、voxel size、dtype、qform、sform 和 NIfTI intent 均为
10/10 一致。共享节点上的 FSL CPU `fnirt` 中位数为 855.583 秒
[IQR 822.899–910.276]，TorchFNIRT H100 同步墙钟中位数为 1244.896 秒
[1054.397–1337.331]；逐例 `FSL/Torch` 时间比中位数为 0.686。该执行未隔离节点
负载，说明本次实现的实际时间，不能解释为硬件加速倍数。完整分布和计时边界见
[`fnirt_fsl_10case.v0.9.public.json`](../../validation/fast_vbm/fnirt_fsl_10case.v0.9.public.json)。

下图使用同一批 matched-input 结果。两行分别为十例平均的轴位和冠状位；前三列
比较 warped GM，后三列比较 nonlinear Jacobian。FSL 与 TorchFNIRT 面板在各自
输出类型内使用相同色阶，差值面板单独使用误差色阶。图像用于查看误差位置，
上表的数值由完整三维数据计算。

![FSL FNIRT 与 TorchFNIRT 的十例 matched-input 平均输出](figures/fnirt_fsl_comparison.png)

最终发布源码只移动了 FNIRT 使用的三个坐标函数及其 import；归一化 AST 和其余
FNIRT Python 源码的逐字节核验见
[`source_equivalence.v0.9.public.json`](../../validation/fast_vbm/source_equivalence.v0.9.public.json)。该记录是源码继承证明，
不是新的数值运行，也不改变 `fsl_fnirt_numerically_equivalent=false`。

最早可定位的分叉出现在第三次 coefficient update 的截断 PCG：FSL 组装稀疏
Hessian 并按固定顺序累加，TorchFNIRT 使用 matrix-free FP64 `einsum`。两者都在
相对残差 `1e-3` 处停止，因此不同的归约顺序会使后续 Krylov 和 LM 轨迹分开。
归一化、LM 阻尼、边界计算、topology projection 和后续更新顺序还会继续传播该
差异；现有实验不能把最终误差分解到某一个步骤。GPU topology projection 为保留
FSL 的更新顺序采用串行 kernel，也是当前 Torch 路径没有快于 FSL CPU 的原因之一。
