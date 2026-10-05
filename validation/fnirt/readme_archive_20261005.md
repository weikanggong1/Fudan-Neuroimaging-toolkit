<!-- 原正文完整归档；基线 140c3739ac6c7a6826bf9421202ef59bec7ffe67；只修复迁移后的相对链接。 -->

# TorchFNIRT：非线性配准

[返回首页](../../README.md) · [dMRI/TBSS pipeline](../../docs/dmri_pipeline/README.md)

[CPU 同线程官方 benchmark：功能覆盖与复现](../../docs/fnirt/CPU_BENCHMARK.md)

## 1. 功能简介

`TorchFNIRT` 是 FNIT 的 PyTorch FNIRT 非线性配准核心。直接调用且不传配置时，
使用 FSL `fnirt` **不带 `--config`** 的参数默认值；GM、T1w 和 UKB TBSS
各有独立预设。FNIT 配准运行时不启动 FSL。

CUDA 默认允许 TF32 matmul 和 cuDNN，但图像和写出的 coefficient NIfTI 为 float32，优化器内部 coefficient state 和
主计算为 float64；不使用 float16 或 bfloat16。实际设置写入
`result.qc["tf32"]`。

## 2. Python 调用、输入输出与参数

### 选择预设

| 名称 | Python 配置 | 独立命令 `--config` | 流程默认入口 |
|---|---|---|---|
| `default` | `FNIRTConfig()` | 省略 `--config`，或写 `default` | 直接 `TorchFNIRT()`、`run_fnirt()` |
| `gm` | `GMFNIRTConfig()` | `gm` 或未修改的 `GM_2_MNI152GM_2mm.cnf` | FastVBM 灰质配准 |
| `t1` | `T1FNIRTConfig()` | `t1` 或未修改的 `T1_2_MNI152_2mm.cnf` | fMRI volume 的 T1→MNI |
| `tbss` | `TBSSFNIRTConfig()` | `tbss` | dMRI pipeline 的 UKB 三阶段 FA 配准 |

`default` 的各级参数已与 FSL 帮助及无配置运行日志核对：`subsamp=4,2,1,1`，
`miter=5,5,5,5`，`infwhm=6,4,2,2`，`reffwhm=4,2,0,0`，
`lambda=120,60,30,30`，`estint=1,1,1,0`，`applyrefmask=1,1,1,1`。
其余默认值为 `imprefm=1`、`impinm=1`、`warpres=10,10,10`、
`ssqlambda=1`、`jacrange=0.01,100`、`intmod=global_non_linear_with_bias`、
`intorder=5`、`biasres=50,50,50`、`biaslambda=10000`；固定使用三次 B 样条、
bending energy 正则化和线性重采样。[FSL 官方 FNIRT 参数说明](https://fsl.fmrib.ox.ac.uk/fsl/docs/registration/fnirt/user_guide.html)
解释各选项；参数匹配不代表形变和强度优化逐步数值等价。

fMRI 的默认值仍是 `t1`，TBSS 的默认值仍是 `tbss`；两条流程不会因为直接
`TorchFNIRT()` 改用 FSL 无配置默认值而换掉自己的配准 schedule。FastVBM 的默认
GM 配置与 `GMFNIRTConfig()` 一致，包括关闭两个隐式零值掩膜，详见
[FastVBM 掩膜说明](../../docs/fast_vbm/README.md#fnirt-gm-掩膜)。

### 配置与覆盖规则

Python 用 `dataclasses.replace` 更改单项参数，并把配置对象传给 `TorchFNIRT`、
`run_fnirt`、fMRI 的 `fnirt_config` 或 dMRI 的 `fnirt_config`。函数也接受上述四个
预设名称；fMRI 和 TBSS 未指定时分别选 `t1`、`tbss`。命令行覆盖选中的预设，
只接受已实现的 FNIRT 选项：

| 命令行选项 | `FNIRTConfig` 字段 | 输入规则 |
|---|---|---|
| `--subsamp`、`--miter` | `subsampling`、`maximum_iterations` | 各级逗号分隔整数 |
| `--infwhm`、`--reffwhm`、`--lambda` | `input_fwhm_mm`、`reference_fwhm_mm`、`regularization` | 各级逗号分隔数值；FWHM 单位 mm |
| `--estint`、`--applyrefmask` | `estimate_intensity`、`apply_reference_mask` | 各级逗号分隔的 0/1 |
| `--warpres`、`--biasres` | `warp_resolution_mm`、`bias_resolution_mm` | `x,y,z`，单位 mm；TBSS 的 `--warpres` 覆盖全部级别 |
| `--jacrange` | `jacobian_range` | 最小值、最大值 |
| `--intmod`、`--intorder` | `intensity_model`、`intensity_order` | 当前支持 `global_linear` 或 `global_non_linear_with_bias`；后者的 `intorder` 为系数个数 2–5，最高次数为 `intorder−1` |
| `--biaslambda`、`--ssqlambda` | `bias_regularization`、`ssd_weighted_lambda` | 正则化数值、0/1 |
| `--imprefm`、`--impinm`、`--minmet` | `implicit_reference_mask`、`implicit_input_mask`、`minimization_methods` | 0/1、`lm` 或 `scg`；优化器可写各级列表 |

各级列表必须与所选预设的级数相同；更改级数时，需要同时给出对应的全部列表。
`--ssqlambda=0` 还须显式给出 `--lambda`，因为 FSL 的默认 λ 随该开关变化。
Python 配置另可设置 `process_stages` 和逐级 `warp_resolution_schedule_mm`。
未实现的 FSL 选项不会被默默忽略；独立命令不读取 `$FSLDIR` 中的 mask。

### Python：GM 专用预设

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
    config="gm",  # 配置：官方 GM 预设；也可传 GMFNIRTConfig() 或原配置名称
    execution="optimized",  # 保留同一 schedule、精度和停止准则的 GPU 执行优化
    device="cuda:0",  # 运行设备：第一张 CUDA GPU
    overwrite=False,  # 写盘策略：不覆盖已有文件
)
```

### Python 与命令行：T1w 专用预设

`T1FNIRTConfig` 采用官方 `T1_2_MNI152_2mm.cnf` 的六级采样、平滑、
形变正则化、`intorder=5` 的全局强度映射和 50 mm 偏置场设置。这里的
`intorder=5` 按 FSL 语义表示 5 个多项式系数，即常数项至四次项。
以下是直接调用核心的例子；
fMRI volume 的完整入口见[T1→MNI 用法](../../docs/fmri/normalization.md)。

```python
import nibabel as nib
from fnit.flirt import TorchFLIRT
from fnit.fnirt import T1FNIRTConfig, TorchFNIRT

moving = nib.load("/absolute/path/T1_brain.nii.gz")  # 输入：同被试 3D 去脑 T1
fixed = nib.load("/absolute/path/MNI152_T1_2mm_brain.nii.gz")  # 输入：3D MNI 脑模板，决定输出网格
mask = nib.load("/absolute/path/MNI152_T1_2mm_brain_mask.nii.gz")  # 输入：模板同网格二值脑掩膜
linear = TorchFLIRT(device="cuda:0")(moving=moving, fixed=fixed)  # 初始 12 自由度 T1→MNI 配准
result = TorchFNIRT(device="cuda:0", config=T1FNIRTConfig(), execution="optimized")(
    moving=moving,  # 待变形的 T1 NIfTI
    fixed=fixed,  # 固定的 MNI NIfTI 与输出网格
    moving_to_fixed=linear.moving_to_fixed_world,  # 输入：T1→MNI 的 4×4 RAS-world 初始矩阵
    reference_mask=mask,  # 输入：只在模板网格的脑内估计配准
)
result.moved.save("/absolute/path/T1_in_MNI.nii.gz")  # 输出：MNI 网格重采样 T1
nib.save(result.coefficient_image, "/absolute/path/T1_to_MNI_coeff.nii.gz")  # 输出：FSL intent-2007 系数图
```

`result` 与上表的 `TorchFNIRTResult` 结构相同；`pull_transform` 是 MNI→T1
的 world-RAS 位移，可与 EPI→T1 BBR 合成，一次插值输出 BOLD。`qc["levels"]`
记录各级强度多项式、偏置场范围和偏置场弯曲能。前五级把形变、强度多项式和
三次 B 样条偏置场放进同一个 LM/PCG 法方程求解，末级固定强度参数，只优化形变。
最终 `iout` 使用与系数文件重采样相同的坐标计算路径。当前实现仍与 FSL 的
优化轨迹存在数值差异，见下文真实数据对照。
### 输出结构与坐标

`run_fnirt` 和 `TorchFNIRT` 返回 `TorchFNIRTResult`：

| 属性 | shape/类型 | 含义 |
|---|---|---|
| `coefficient_image` | intent-2007 NIfTI | 与 `cout` 相同，可交给 FNIT `TorchApplyWarp`。 |
| `coefficients` | `[Cx,Cy,Cz,3]` NumPy array | cubic residual coefficient，不是 reference-grid dense displacement。 |
| `moved` | reference-grid `FNITNifti1Image` | 与 `iout` 相同。 |
| `nonlinear_jacobian` | reference-grid `FNITNifti1Image` | 与 `jout` 相同，排除 affine determinant。 |
| `full_pull_jacobian` | reference-grid `FNITNifti1Image` | 包含 affine 的完整 pull Jacobian；FSL `fnirt --jout` 不写这一项。 |
| `pull_transform` | `DenseWarp`（`nibabel.Nifti1Image` 子类） | reference → input 的 world-RAS 毫米位移场，保存 source/target geometry。 |
| `qc` | `dict` | schedule、mask、优化、拓扑、设备、TF32 和数值验证状态。 |

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

### 算法对应与数值边界

| FSL 行为 | 当前实现 |
|---|---|
| SPM-like mean normalization、float32 image scaling | 相同顺序实现 |
| implicit input/reference zero mask | 使用 `1e-16` 判零；reference mask 在归一化前建立，input mask 在 float32 归一化后建立 |
| warped input mask | 按 FSL `volume<char>` 语义，三线性插值后先截断为 char，再执行 `>0.5` |
| 有效 FOV 边界 | 使用 newimage 的 `1e-8` tolerance、原始 floor index 和 zero-padded neighbor |
| input masked smoothing 与 reference zero-padded Gaussian smoothing | 已实现 |
| cubic B-spline field、bending regularization、SSD-weighted lambda | 已实现 |
| LM stages | matrix-free analytic `JᵀJ` + FP64 PCG |
| T1 强度映射 | 5 系数全局多项式、B 样条乘性偏置场与形变联合进入 LM 法方程；强度关闭的末级固定上一层参数 |
| `minmet=scg` stages | 按 MISCMATHS `sccngr` 更新顺序实现 |
| `inwarp`/`intin` process handoff | 在内存中执行 float32 coefficient/header、10 位 intensity 交接 |
| `FullResKsp` 与 `ZoomField` | 按每个进程最后 subsampling 计算 full-grid spacing |
| coefficient NIfTI | 输出 FSL intent 2007，shape、spacing、sform 契约已验证 |
| `ForceJacobianRange` | 已移植；外部逐步 topology oracle 仍未通过 |

独立 GM schedule 为 `subsampling=4,2,1,1`、`maxiter=5,5,10,5`、input FWHM
`6,4,2,2 mm`、reference FWHM `4,2,0,0 mm`、bending lambda
`150,75,50,30`、10 mm warp resolution，四层使用 LM。dMRI/TBSS schedule 见
[dMRI 页面](../../docs/dmri_pipeline/README.md#ukb-tbss-对应关系)。

### GPU 执行方式

`TorchFNIRT(..., execution="optimized")`、`run_fnirt(..., execution="optimized")` 与 CLI `--execution optimized` 默认启用：

- 每个方向一个 Gaussian kernel，保留官方零填充、核生成和每个偏移赋值回 float32 的舍入顺序。未安装 Triton 时自动使用原张量平滑。
- PCG 每轮合并分母与残差标量的 GPU→CPU 传输；每轮仍检查相同停止条件，无效分母不接受试算更新，不跳过迭代检查。
- 缓存分辨率层内的 affine 网格、固定张量类型转换和 bending diagonal。
- bending Hessian 在系数空间计算 `BᵀB` 的三个方向 Gram 乘积，避免每轮展开六张全网格导数场；energy 保留原 dense 算法及求和顺序。

本轮继续减少重复计算：只在需要导数时计算 trilinear 梯度；T1 联合 linearization 复用已算好的强度多项式和 bias；每次联合 PCG matvec 只展开一次 FP64 形变增量，图像线性项仍按原路径转换为 FP32。上述三个改动保留原浮点运算和求和顺序，按冻结 optimized 基线逐位验证。

`execution="reference"` 保留逐偏移平滑、逐标量 PCG 主机判断和 dense bending Hessian，用于检查执行改动。两条路径都采用修正后的 header `pixdim`、相同 cubic spline、SSD、强度模型、LM/SCG、PCG 容差、Jacobian 约束和配置。Gram 改变 FP64 求和顺序，不能称为逐 bit 相同；固定 FSL basisfield oracle 和真实 T1 配对用于验收。

图像与插值为 float32；系数、法方程、Gram、PCG、强度模型及关键归约保留 float64。不使用 FP16/BF16。GPU 默认允许 TF32，不降低 FP64 算子的精度。上述优化使用主页环境已有的 PyTorch/Triton，无新增编译依赖，也不启动 FSL。对应原软件没有 `--execution` 或 `--device` 选项。

### CPU 执行方式

`device="cpu", execution="optimized"` 使用设备对应的保序平滑、弯曲能量、角点筛选和法方程缓冲。空间法方程在每次线性化时固定权重布局，PCG 中按原 FP64 运算顺序处理八个体素；尾部、非连续布局和非有限值块使用原标量算式。函数遵循调用方的 CPU 线程预算，不自行修改全局线程数。没有新增运行依赖，CPU helper 使用主页环境已有的 Numba；CUDA、需要梯度的 CPU 张量和 `execution="reference"` 保留各自路径。布局内存、逐位门槛、真实精度及速度见 [CPU 专页](../../docs/fnirt/CPU_BENCHMARK.md)。

最新 v27 的 CPU 采样合并图像值、FOV 判定与所需梯度，SCG 梯度跳过未使用的代价计算；原插值舍入、真实 cost、有效 lambda 和 CUDA 路径保持原方式。当前组合源 v28 保留该注册器和采样 helper 的相同 SHA。[完整真实 SCG 阶段检查](../../docs/fnirt/CPU_BENCHMARK.md#v27-的完整-scg-阶段检查)记录了输出与迭代轨迹的逐位一致性；[完整功能报告](../../docs/fnirt/assets/cpu-functional-v27-node8-20261004.public.json)包含 13 项预设/参数分支在 1/8 线程预算的 26 项完整输出检查及八项新官方配对。

## 3. 命令行调用

### 命令行：无配置默认值

```bash
# --in：待变形的单张 3D 影像；--ref：目标 3D 影像和输出网格。
# --aff：input→reference 的 FSL scaled-mm 初始矩阵；省略时使用单位矩阵。
# --config：default 使用 FSL 不带配置文件的参数默认值。
# --cout：保存 FSL intent-2007 系数；--iout：保存目标网格上的变形图像。
# --device：选择 PyTorch 设备；省略时优先 CUDA。
python -m fnit.fnirt \
  --in moving.nii.gz --ref reference.nii.gz \
  --aff moving_to_reference.mat --config default \
  --cout warp_coeff.nii.gz --iout moving_in_reference.nii.gz \
  --device cuda:0
```

### 命令行：GM 专用预设

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
| `--in` | 输入 | 单张 3D moving 灰质概率图或 T1 影像，与 `--config` 对应。 |
| `--ref` | 输入 | 单张 3D fixed 灰质或 T1 模板；决定 `iout`/`jout` 的网格。 |
| `--aff` | 输入，可省略 | 4×4 FSL scaled-mm 矩阵，方向为 input → reference；省略时使用 FSL scaled-mm identity。 |
| `--refmask` | 输入，GM/T1 必需 | reference 网格上的非空二值 mask；default/TBSS 可省略。 |
| `--config` | 输入 | `default`、`gm`、`t1`、`tbss`，或未修改的官方 GM/T1 `.cnf`；省略时使用 `default`。 |
| `--cout` | 输出 | FSL intent-2007 cubic B-spline coefficient NIfTI；它不是 dense warp。 |
| `--iout` | 输出，可省略 | 原始 input 经 affine 和 nonlinear warp 后的 reference-grid 图像。 |
| `--jout` | 输出，可省略 | nonlinear-only Jacobian determinant，不含 FLIRT affine determinant。 |
| `--device` | 运行选项 | `cpu`、`cuda` 或 `cuda:N`；省略时优先 CUDA。 |
| `--execution` | 运行选项 | `optimized`（默认）启用设备对应的缓冲复用和算子优化；`reference` 使用原 dense 算子与逐标量 PCG 执行方式，见下文。 |
| `--overwrite` | 运行选项 | 允许替换已有输出；默认保护已有文件。 |

`cout` 总会写出。省略 `--cout` 时，输入 `subject_GM.nii.gz` 生成
`subject_GM_warpcoef.nii.gz`。不带扩展名的输出遵循 `FSLOUTPUTTYPE=NIFTI` 或
`NIFTI_GZ`。三个输出路径必须不同，也不能覆盖 input、reference、affine 或 mask。

### T1w 六级预设

T1w 可用上述 Python API、fMRI volume 入口，或独立命令行：

```bash
# --in：待配准的单张去脑 T1。
# --ref：MNI 脑模板，同时决定输出图像网格。
# --aff：T1→MNI 的 FSL scaled-mm 初始矩阵。
# --refmask：与模板同网格的二值脑掩膜。
# --config：选择 T1w 六级配准和强度模型。
# --cout：写出 FSL intent-2007 形变系数。
# --iout：写出模板网格上的 T1 图像。
# --device：指定 PyTorch 计算设备。
python -m fnit.fnirt \
  --in T1_brain.nii.gz \
  --ref MNI152_T1_2mm_brain.nii.gz \
  --aff T1_to_MNI_affine.mat \
  --refmask MNI152_T1_2mm_brain_mask.nii.gz \
  --config T1_2_MNI152_2mm.cnf \
  --cout T1_to_MNI_coeff.nii.gz \
  --iout T1_in_MNI.nii.gz \
  --device cuda:0
```

`--aff` 是 FSL scaled-mm 的 T1→MNI 初始矩阵；`--refmask` 与 `--ref` 同网格；
`--cout` 写 intent-2007 系数，`--iout` 写模板网格的变形 T1。其余选项见上表。

## 4. 原软件调用

同一输入的原版 FSL 调用不传 `--config`：

```bash
# --in、--ref、--aff 分别是 moving、reference 和 input→reference 初始矩阵。
# --cout 写系数图，--iout 写 reference 网格上的 warped input。
fnirt --in=moving.nii.gz --ref=reference.nii.gz \
  --aff=moving_to_reference.mat \
  --cout=warp_coeff.nii.gz --iout=moving_in_reference.nii.gz
```

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

两条命令的参数角色和输出文件类型一致；`--device`、`--execution`、`--overwrite` 是 FNIT 选项。
当前验证尚未达到逐体素数值等价，因此这里的“等价调用”只表示接口与文件契约对应。

同输入的原软件命令是：

```bash
flirt -in T1_brain.nii.gz -ref MNI152_T1_2mm_brain.nii.gz \
  -dof 12 -omat T1_to_MNI_affine.mat
fnirt --in=T1_brain.nii.gz --ref=MNI152_T1_2mm_brain.nii.gz \
  --aff=T1_to_MNI_affine.mat --refmask=MNI152_T1_2mm_brain_mask.nii.gz \
  --config=T1_2_MNI152_2mm.cnf --cout=T1_to_MNI_coeff.nii.gz \
  --iout=T1_in_MNI.nii.gz
```

## 5. 最新真实数据精度、耗时与脑图

### 2026-10-04：最新 v27 的同 CPU 预算官方对照

本轮补充 CPU 执行缓冲优化，完整 default/T1/GM/TBSS schedule 和受支持参数分支按同输入官方配对验收；
精度与端到端时间见 [CPU benchmark 的范围和复现](../../docs/fnirt/CPU_BENCHMARK.md)。官方与 FNIT 的优化轨迹仍有差异，
不能把配对参数相同写成逐体素等价。CPU 修改不替换既有 CUDA 分支；GPU 回归独立核对。
公开脑图补充使用已核验 CC0 许可、逐文件 SHA 的 [ds000114 T1 示例](../../examples/README.md)，
共享官方 FLIRT 初始矩阵、完整 FNIRT 默认四级及三个相同输出。最新 v27 在已验收的严格 FP64 法方程 SIMD 上增加 CPU 三线性采样融合与 SCG 未使用代价的跳过，组合源 v28 保留其相同文件 SHA。八项主要功能分别完成新的官方配对，包含导入、读取、完整求解和全部声明输出保存；[主要配对报告](../../docs/fnirt/assets/cpu-primary-v27-node8-20261004.public.json)记录完整数值与源码 SHA。[全部 26 项完整输出检查](../../docs/fnirt/assets/cpu-functional-v27-node8-20261004.public.json)也已完成，覆盖 13 项预设/参数分支的 1/8 线程预算，原网格、全部层数和迭代预算保持不变；非主要分支沿用已核验的官方精度参照，不复制旧时钟为新配对。T1 单线程的[同节点旧/新诊断](../../docs/fnirt/CPU_BENCHMARK.md#t1-单线程的同节点旧新诊断)也已完成。私有病例只公开聚合指标和文件 SHA。

| 完整预设 | 线程上限 | 原版完整进程 | FNIT 完整进程 | FNIT 已导入 API（含读写） | 脑掩膜内 iout Pearson r | Jacobian MAE |
|---|---:|---:|---:|---:|---:|---:|
| default | 1 | 181.04 s | 149.12 s | 146.56 s | 0.956660 | 0.050896 |
| default | 8 | 172.21 s | 57.68 s | 55.67 s | 0.955977 | 0.051218 |
| T1 六级 | 1 | 224.02 s | 293.71 s | 291.49 s | 0.999078 | 0.008707 |
| T1 六级 | 8 | 221.80 s | 169.21 s | 166.93 s | 0.999243 | 0.008490 |
| TBSS 六级/三阶段 | 1 | 836.24 s | 795.22 s | 793.07 s | 0.999704 | 0.011844 |
| TBSS 六级/三阶段 | 8 | 831.94 s | 400.19 s | 398.29 s | 0.999032 | 0.019836 |
| GM 四级/末级 SCG | 1 | 548.11 s | 143.57 s | 141.42 s | 0.970540 | 0.029834 |
| GM 四级/末级 SCG | 8 | 556.22 s | 63.28 s | 60.73 s | 0.973537 | 0.028395 |

这些是共享 CPU评测节点 上各一次完整观察，七项快于原版，T1 单线程未达到速度目标；8 是相同线程上限，原版的实际 CPU 使用量另行记录。每个预算的完整三图/四图文件、数值、header 和 affine 与该预算旧 normal SIMD v2 输出一致；旧完整输出 adapter 未保存 QC/停止记录，不宣称这两项逐位核对完成。原有 FSL 求解轨迹差异仍存在。旧 CPU评测节点 的[normal SIMD v2 配对](../../docs/fnirt/assets/cpu-default-t1-normal-simd-v2-20261004.public.json)保留历史来源，不与本节点时间构成优化比。

新算子保留九次乘法与加法的顺序、每步 FP64 舍入和原标量尾部，不使用 FMA、fastmath 或低精度。完整 default/T1 的[独立热点门槛](../../docs/fnirt/assets/cpu-normal-simd-v2-gate-20261004.public.json)核对了全部输出文件 SHA、停止条件和调用次数；热点钟与官方完整进程时间分别报告。

完整六层 T1 的独立 CPU95 [热点报告](../../docs/fnirt/assets/cpu-profile-t1-v20-20261004.public.json)包含四图 SHA 与正式 CPU 输出核对。其 7,247 次空间法方程乘积累计 158.905 s，完整 instrumented API 为 362.182 s，峰值 RSS 1,371,236 KiB；函数钟互相包含，不用于相加或替代官方配对时间。具体范围见[六层 T1 诊断](../../docs/fnirt/CPU_BENCHMARK.md#六层-t1-的独立热点诊断)。

![公开 T1 的完整 FNIRT warped 图与原版差异](../../docs/fnirt/assets/cpu-public-v27-node8/fnirt_warped.png)

![同一公开 T1 的非线性 Jacobian 与原版差异](../../docs/fnirt/assets/cpu-public-v27-node8/fnirt_jacobian.png)

图读取同一公开 CC0 输入在 CPU评测节点 的 v27 CPU8 和新原版配对保存输出，三个完整文件与该预算旧输出逐位相同。显示 MNI 脑掩膜内真实结果；双方共用标尺，差异显示有分位裁剪，完整误差见数值报告。[绘图位置、范围和来源 SHA](../../docs/fnirt/assets/cpu-public-v27-node8/figures.public.json) · [warped PDF](../../docs/fnirt/assets/cpu-public-v27-node8/fnirt_warped.pdf) · [Jacobian PDF](../../docs/fnirt/assets/cpu-public-v27-node8/fnirt_jacobian.pdf)。旧 v17 公开图保留为历史产物，私有病例没有新增脑图。

[最新组合源 v28 的 H100 完整回归](../multimodal_cpu_20261004/gpu_cpu_final_v28_ready_retry_20261004.public.json)使用相同的 v27 注册器与采样 helper，完整运行六级 TBSS 和三阶段交接，没有降低网格或迭代预算。四个进程各一次完整 warmup 与三次完整 API 调用，共 16 次保存；系数、重采样、非线性及完整 pull Jacobian 的全部 float32 位模式、header/extensions 和 affine 与基线相同。

| H100 完整 API，含读写 | 测试冻结源 | 优化前 FNIT，两组中位数 | 当前 FNIT，两组中位数 | 两版对应位置的峰值 allocation |
|---|---|---:|---:|---|
| default 四级，三输出，16 次保存 | v27；注册器/helper 与 v28 相同 | 16.945 / 16.702 s | 16.737 / 16.868 s | warmup 1,181,705,216 B；每个 measured repeat 1,181,704,704 B |
| TBSS 六级/三阶段，四输出，16 次保存 | v28 | 14.866 / 13.365 s | 14.854 / 14.102 s | warmup 3,564,715,520 B；每个 measured repeat 3,565,562,368 B |

GPU 表对照本轮优化前 FNIT，CPU 表对照原版 FSL，两者的参照和计时范围分别保留。对应调用位置的 allocation 两版完全相同；各例的 warmup 与 measured repeat 分列。GPU 为共享 H100 观测，默认 TF32、进程上限 20 GB；短时波动不用于推断稳定加速比。

上表 default 的[完整 v27 报告](../../docs/fnirt/assets/cuda-final-v27-20261004.public.json)保存 16 次三输出的逐位和 metadata 校验。较早 [v27 TBSS 的 12 次有效调用](../../docs/fnirt/assets/cuda-tbss-partial-v27-20261004.public.json)也逐位相同，但最后基线进程在 CUDA setup 失败，原 16 次协议未完成；其失败和 partial 范围保留，不计作算法耗时。新 v28 完整运行另有独立目录与协议，不覆盖该失败事实。历史 v23 见[原报告](../multimodal_cpu_20261004/gpu_final_v23_20261004.public.json)。

本轮以 `7473452` 的现有 optimized 实现为固定基线，实际运行 FastVBM、
fMRI volume 和 dMRI/TBSS 三条完整流程。所有 FNIRT 层级、优化精度、
PCG 停止判断和输出公式保持一致。最新逐位验收、分步骤耗时、显存、
原软件对照与模板空间脑图统一见[2026-10-02 验证](../registration_lossless_20261002/README.md)。

### 2026-10-02：同输入无损复用

固定同一真实已处理 T1、初始 FSL affine、MNI 模板和完整六级配置，基线/候选各运行 cold、两个 warm 和独立 profile。20 对已保存的系数、图像、pull、两种 Jacobian 的所有位模式、完整 header 和 SHA-256 均相同；四对完整 QC 文件也相同。

| 指标 | 基线 | 本轮候选 |
|---|---:|---:|
| cold API | 27.101 s | 26.516 s |
| 两个 warm 的中位数 | 24.813 s | 24.084 s |
| 正式峰值 allocated / reserved | 1.178 / 1.491 GB | 1.183 / 1.458 GB |
| profile：coefficient expand 次数 | 20,819 | 14,147 |
| profile：PCG matvec 次数 | 7,266 | 7,266 |

warm 观测降低 2.94%；图像相对独立 FSL 参照的脑内 r 仍为 0.9977178843。FSL 2026-10-01 命令为 217.558 s，与 FNIT 函数的读写边界不同。profile 会增加同步和事件开销，其总时间不用作速度表；数据、逐次时间、完整门禁与计数见[本轮固定输入报告](../registration_lossless_20261002/fixed_fnirt.public.json)。FastVBM、volume、dMRI 的最新端到端结果见[统一验证页](../registration_lossless_20261002/README.md)。

### T1w 专用预设：当前 GPU 修复版

以下为 **2026-10-01 历史记录**，对应优化前的发布，不作为本轮整链计时。

2026-10-01 固定同一例真实已处理的去颅骨 T1、MNI152 2 mm 脑模板、脑掩膜及 FSL 初始仿射，重新运行 FSL 6.0.7.22 六级 T1 配置。T1 的更早预处理来源未知，不把它视为扫描仪原始 T1。官方实际 FNIRT 二进制子进程退出 0；包装器返回 255 单独保留。新官方输出与固定参照的系数、图像及 header 逐位一致。独立 `applywarp` 对三个 T1 RAS world 坐标图采样，构成完整 pull 的对照。

| 同输入耗时 / 显存 | 修改前 `7952b33` | 当前 reference | 当前 optimized |
|---|---:|---:|---:|
| 首次 / 热调用 | 69.350 / 71.501 s | 79.003 / 74.543 s | 32.595 / 30.422 s |
| 热调用峰值 allocated / reserved | 1.091 / 1.474 GB | 1.178 / 1.476 GB | 1.178 / 1.491 GB |

FSL CPU FNIRT 为 217.558 s，包含命令输入读写、启动及 exec/exit 追踪。FNIT 函数计时包含 ArrayProxy 读入/解压与 CPU 结果转换，排除持久输出写盘、精度计算和 CUDA 上下文初始化。首次调用未清空 Triton 磁盘缓存。两侧负载均未隔离，仅报告观测时间。

| 与 FSL 的精度 | 修改前 / 当前 reference | 当前 optimized |
|---|---:|---:|
| MNI 脑内 warped T1 Pearson r | 0.99784173 | 0.99771788 |
| warped T1 MAE / RMSE，原强度单位 | 4.97974 / 16.37197 | 4.87159 / 16.83218 |
| 支持区 Dice | 0.99924899 | 0.99922162 |
| 完整 pull mean / median / p95 | 0.08663 / 0.05322 / 0.23351 mm | 0.08642 / 0.05176 / 0.23294 mm |
| coefficient Pearson r | 0.99774652 | 0.99763395 |
| coefficient MAE / RMSE | 0.04222 / 0.12553 | 0.04316 / 0.12872 |
| nonlinear Jacobian Pearson r | 0.99800749 | 0.99812312 |
| nonlinear Jacobian MAE / RMSE | 0.00746 / 0.01567 | 0.00715 / 0.01522 |
| 完整 pull Jacobian min / max；非正占比 | 0.03564 / 1.80007；0 | 0.03243 / 1.78385；0 |

r/强度误差在 MNI 脑掩膜内；支持区使用正值第 99 百分位的 5% 阈值。完整 pull 在模板掩膜及双方有效支持区的 228,458 个体素比较；coefficient 比较全系数数组，nonlinear Jacobian 比较脑内。两个路径的影像 shape/affine/pixdim、系数 intent-2007、knot pixdim、dense-grid intent 参数、qoffset、保存的 FSL 初始 affine 及 q/sform code 均通过合同核验。

本例当前 reference 复现修改前已有图像、系数、pull 与完整 Jacobian。optimized 的图像 r 下降 0.000124、Dice 下降 0.000027，MAE 和 pull median/p95 改善，图像 RMSE 与 coefficient 误差略增。不能称为所有指标不退化或逐位等价。仅换回 dense bending normal、保留新平滑/PCG/缓存的消融中，五类输出及逐级 PCG 记录与 reference **逐位一致**，确认差异来自 Gram 的 FP64 求和顺序。optimized 与 reference 的最大 pull 分量差为 0.62815 mm；优化轨迹变化超过输出末位舍入。需要原数值轨迹时选择 `execution="reference"`。

| 完整 CUDA profile | 修改前 | 当前 optimized |
|---|---:|---:|
| kernels | 2,126,760 | 1,553,433 |
| `cudaStreamSynchronize` | 22,224 | 8,016 |
| H2D / D2H 事件 | 21 / 22,203 | 72 / 7,944 |
| CUDA 张量转 Python float / bool | 14,765 / 7,222 | 415 / 47 |
| cost evaluations | 100 | 100 |
| PCG matvec | 7,175 | 7,266 |

同一热调用的嵌套 CPU wall：PCG 68.430→27.933 s，其中 matvec 32.477→21.345 s；联合 linearization 0.908→0.743 s，topology 检查 0.046→0.038 s。Gaussian dispatch wall 为 0.044→0.127 s，未表现为该局部时钟更快；新路径减少逐偏移 launch，全流程主要收益来自 PCG 同步与弯曲算子。以上阶段钟无额外 GPU fence，不能互相直接相加，也不是独立 kernel 时间。完整算子时钟与 CUDA 事件见报告。

六级 LM 尝试/接受次数一致，PCG 每轮停止判断仍保留，没有减少 miter、搜索或采样级别。GPU 内仍有 PCG 主机分支和 spline expansion/adjoint kernels；本次未把整个优化器改成完全无同步 CUDA 算子。整张共享 GPU 的平均利用率为 98.4%→99.1%，含其他作业，不能归为本进程。所有冷/热/profile 重复产物各自逐位一致。

源码哈希、FSL 退出/产物核验、profile、消融和坐标指标见 [当前配准报告](../fmri/registration_gpu.current.public.json)。整条 490 帧 volume/surface 的既有运行见[全流程页](../fmri/README.md)；该历史测量不替代 2026-10-02 的三条完整流程验证。

### 其他预设的独立验证

无配置默认值与 Oxford TBSS/FA 使用不同的 schedule 和输入；各自最近一次真实对照、原始命令、图像及源码哈希保留在[FNIRT 验证页](README.md)。这些原软件对照绑定 2026-09-28/29 的候选；本轮 FNIT 前后比较另见最新验证。当前 T1 结果不能替代其数值验收，也没有建立跨预设的逐体素 FSL 等价。

## 6. 最近版本与 benchmark 记录

| 日期 | 更新与验收 |
|---|---|
| 2026-10-04，最新 v27/v28 | CPU float32 采样的值、FOV 和梯度融合；SCG 梯度跳过未使用的能量/cost。83 项采样专项、33 项 SCG 专项和[完整真实末阶段轨迹](../../docs/fnirt/assets/cpu-scg-cost-skip-stage3-node8-20261004.public.json)通过；CPU评测节点 [26 项完整功能输出检查](../../docs/fnirt/assets/cpu-functional-v27-node8-20261004.public.json)及[八项新官方单次配对](../../docs/fnirt/assets/cpu-primary-v27-node8-20261004.public.json)完成，T1 单线程未达速度目标。H100 default 和完整 TBSS 各 16 次保存逐位相同、对应位置 allocation 不增加。 |
| 2026-10-04，normal SIMD v2 | FP64 空间法方程按原运算顺序执行八点 SIMD；105 项局部回归通过，完整 default/T1 的输出 SHA、停止条件和 PCG 计数保持一致。CPU评测节点 的四项完整官方单次配对见[历史报告](../../docs/fnirt/assets/cpu-default-t1-normal-simd-v2-20261004.public.json)。 |
| 2026-10-04 | CPU Gaussian 合并 offset 循环，bending 保留 dense 展开/原 sum 并融合逐元素乘方，FP64 法方程缓冲/固定 weight 布局复用，Jacobian limiter 保序筛选角点。216 项函数专项通过；v17/v19/v20 在完整公开 default 的 1/8 预算三输出 SHA 一致。最新 CPU 官方观测与 GPU 门槛见[专页](../../docs/fnirt/CPU_BENCHMARK.md)。 |
| 2026-10-02 | 跳过无用插值梯度、T1 强度映射复用、联合 PCG 增量场复用；三种预设在完整 pipeline 中实际执行，固定输入逐位比较，见[最新报告](../registration_lossless_20261002/README.md) |
| 2026-10-01 | T1 六级 joint intensity/bias、GPU smoothing/PCG/Gram 优化与同输入 FSL 对照，见[历史报告](../fmri/registration_gpu.current.public.json) |
| 2026-09-28/29 | 无配置、GM、TBSS 预设与 FSL 独立验证，见[FNIRT 验证索引](README.md) |

## 7. 参考文献、原实现与许可

### 支持范围与许可

独立 CLI 支持一个 3D input/reference、可选 4×4 FLIRT affine、显式 binary reference
mask，以及上表四种预设和已实现参数的覆盖。传入官方 GM/T1 配置文件路径时，
先核对文件 SHA-256；不支持任意 `.cnf`、DCT basis、quadratic spline、局部
intensity model 或通用 `--inwarp`。TBSS 预设在直接配准时可选，dMRI pipeline
仍负责 weighted FLIRT 与九张参数图传播。

实现依据 FSL FNIRT 2203.0、basisfield 2203.1、miscmaths 2203.2、newimage
2203.11 和 warpfns 2203.0，受
[`FSL Software Licence, Release 6.0`](../../licenses/FSL-6.0.txt) 约束，仅用于该许可
允许的非商业用途。本项目不是官方 FSL 发布。上游版本、commit 和文件哈希见
[`_vendor_fsl`](../../src/fnit/_vendor_fsl/README.md)。

### 参考文献

- 参考文献：Andersson, Jenkinson & Smith, *Non-linear registration, aka spatial normalisation*, FMRIB Technical Report TR07JA2 (2007), [原文](https://www.fmrib.ox.ac.uk/datasets/techrep/tr07ja2/tr07ja2.pdf)。
- 原实现代码库：[FSL `fnirt`](https://git.fmrib.ox.ac.uk/fsl/fnirt)。
