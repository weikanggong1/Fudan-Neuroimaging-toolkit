# 单被试 dMRI 参数图流程

[返回首页](../../README.md) · [PyTorch MMORF](../mmorf/README.md)

本流程实现以下固定链：

~~~mermaid
flowchart LR
  A[UKB AP 或 BIDS dwi + optional reverse PE] --> B{reverse PE present?}
  B -->|yes| C[TorchTOPUP]
  B -->|no| D[zero susceptibility field]
  C --> E[TorchEDDY]
  D --> E
  E --> F[TorchDTIFIT b≈1000]
  E --> G[TorchAMICONODDI all shells: AMICO or classic]
  F --> H{registration backend}
  G --> H
  H -->|tbss| I[weighted TorchFLIRT + three-stage TorchFNIRT]
  H -->|mmorf| J[SynthStrip T1 + two TorchFLIRT + run_mmorf]
  I --> K[9 maps on selected MNI template grid]
  J --> K
~~~

两条分支输出相同的九个参数名：FA、MD、L1、L2、L3、MO、ICVF、OD、ISOVF。输入的 FA、T1 和 tensor 模板采用同一 MNI 网格时，两条分支的标准空间图也有相同的 shape、affine 和 float32 dtype；TBSS/FNIRT 与 MMORF 仍会估计不同的非线性形变。TBSS 分支另输出九张 UKB 风格 skeleton-mask 图。

| 输入 | Python 入口 | 命令行入口 | 配准选择 |
|---|---|---|---|
| 原始 BIDS | [`run_bids()`](#python原始-bids-单被试) | [`--bids-root`](#原始-bids-输入) | 无 T1w 用 `tbss`；有 T1w 可用 `mmorf` |
| UKB 命名的 AP/PA 文件 | [`run()`](#pythonukb-命名输入) | [`--raw-dir`](#ukb-命名输入-1) | `tbss` 或另传 T1w 使用 `mmorf` |

## 输入目录

`run_bids()` 和 `--bids-root` 直接读取单被试原始 BIDS 目录。反向相位编码图既可以是本 session 或 subject 层级的 `fmap/*_epi.nii.gz`，也可以是同一 session 的 `dwi/*_dwi.nii.gz`；没有反向图时跳过 TOPUP。`anat/*_T1w.nii.gz` 只供 MMORF 分支使用。

~~~text
bids/
├── dataset_description.json
└── sub-01/
    └── ses-01/
        ├── dwi/
        │   ├── sub-01_ses-01_dir-AP_run-01_dwi.nii.gz
        │   ├── sub-01_ses-01_dir-AP_run-01_dwi.bval
        │   ├── sub-01_ses-01_dir-AP_run-01_dwi.bvec
        │   └── sub-01_ses-01_dir-AP_run-01_dwi.json
        ├── fmap/
        │   ├── sub-01_ses-01_dir-PA_epi.nii.gz
        │   └── sub-01_ses-01_dir-PA_epi.json
        └── anat/
            └── sub-01_ses-01_T1w.nii.gz
~~~

| BIDS 输入 | 用途 | 缺少时的行为 |
|---|---|---|
| `dwi/*_dwi.nii[.gz]`、`.bval`、`.bvec`、JSON | 主 DWI，供 EDDY、DTIFIT、NODDI 使用 | 报错；多条主 DWI 需用 `run`/`acquisition`/`direction` 选一条 |
| 反向 `fmap/*_epi.nii[.gz]` 或 `dwi/*_dwi.nii[.gz]` 及 JSON | 提取反向 b0，供 TOPUP 估计畸变场 | 跳过 TOPUP，EDDY 使用零 susceptibility field |
| `anat/*_T1w.nii[.gz]` | 只供 MMORF 的 SynthStrip 和 T1 配准使用 | TBSS 照常运行；MMORF 报错 |

被选 DWI 必须是 4D，`.bval` 和 `.bvec` 的列数必须等于体积数；梯度文件和 JSON 可按 BIDS 继承规则放在上级目录。运行 EDDY 需要 DWI 的 `PhaseEncodingDirection` 以及 `TotalReadoutTime` 或 `EffectiveEchoSpacing`。反向图的 JSON 也需要这两项，且相位编码方向必须与 DWI 相反；若有 `B0FieldSource`／`B0FieldIdentifier` 或 `IntendedFor`，则按关联选择。多个 DWI 或多个可用反向图不能无依据地任选一个，需指定 run、acquisition、direction 或补充 fieldmap 关联。这些是本流程运行所需的采集信息；BIDS 允许某些 DWI 缺少上述推荐字段，但缺少时不能计算 EDDY 的采集参数。

原来的 `run()` 和 `--raw-dir` 继续读取 UKB 命名：

~~~text
raw/
├── AP.nii.gz
├── AP.bval
├── AP.bvec
├── AP.json
├── PA.nii.gz   # optional
├── PA.bval     # optional
└── PA.json     # optional
~~~

AP 四个文件必须存在。PA 的 image、bval 和 JSON 同时存在时才运行 TOPUP；缺少全部 PA 文件时进入 AP-only 路径，PA 文件只出现一部分时立即报错，避免误把不完整采集当成 AP-only。AP-only 分支仍运行 EDDY，但 susceptibility field 固定为零，并在报告中记录 topup_applied=false。

TBSS 分支需要 FMRIB58_FA_1mm 和 FMRIB58_FA-skeleton_1mm。MMORF 分支还需要 subject T1w、官方 SynthStrip 权重、MNI152_T1_1mm_brain 和与其同网格的 FSL_HCP1065_tensor_1mm。三个标准模板必须有相同 shape 和 affine。

## Python：原始 BIDS 单被试

无 T1w 时使用 TBSS：

~~~python
from fnit import DMRIPipeline

result = DMRIPipeline(
    device="cuda:0",  # 所有计算使用这一 GPU；默认允许 TF32
    registration_backend="tbss",  # 不依赖 T1w，使用 FA 的 FLIRT/FNIRT 配准
    fnirt_config="tbss",  # UKB Oxford 三阶段配置
    synthstrip_weights=None,  # TBSS 不使用 T1 脑提取
    dti_shell=1000,  # DTI 拟合目标 b 值，单位 s/mm²
    dti_tolerance=100,  # DTI shell 容差，单位 s/mm²
    bvec_source="rotated",  # DTI/NODDI 使用 EDDY 旋转后的梯度
    noddi_fit_method="amico",  # 默认 AMICO 字典拟合；"classic" 为连续 Watson 拟合
).run_bids(
    bids_root="/data/bids",  # 含 dataset_description.json 的原始 BIDS 根目录
    output_dir="/data/results/sub-01_tbss",  # 单被试 FNIT 输出根目录
    subject="01",  # BIDS sub 标签；也可写 "sub-01"
    session="01",  # BIDS ses 标签；没有 session 层时写 None
    run="01",  # 选择 DWI run；只有一条 DWI 时可写 None
    acquisition=None,  # 可按 acq 标签进一步选择 DWI
    direction="AP",  # 选择主 DWI 的 dir 标签，不根据文件名推断实际 PE 符号
    fa_template="FMRIB58_FA_1mm.nii.gz",  # 标准 FA 图及输出网格
    fa_skeleton="FMRIB58_FA-skeleton_1mm.nii.gz",  # TBSS skeleton 模板
    t1=None,  # TBSS 不使用 T1w，即使 anat/ 中存在 T1w
    t1_template=None,  # TBSS 不使用 T1 模板
    tensor_template=None,  # TBSS 不使用 tensor 模板
    overwrite=False,  # 拒绝覆盖已有输出
)
~~~

有 T1w 且需要 T1/tensor 联合配准时，选择 MMORF。`t1=None` 表示从同一 subject/session 的 `anat/` 自动选取唯一 T1w；若有多张 T1w，显式传 `t1` 文件路径。

~~~python
from fnit import DMRIPipeline

result = DMRIPipeline(
    device="cuda:0",  # 所有计算使用这一 GPU；默认允许 TF32
    registration_backend="mmorf",  # 使用 T1w 与 DTI tensor 联合估计形变
    fnirt_config=None,  # MMORF 不使用 FNIRT
    synthstrip_weights="/weights/synthstrip.1.pt",  # 官方 SynthStrip 权重文件
    dti_shell=1000,  # DTI 拟合目标 b 值，单位 s/mm²
    dti_tolerance=100,  # DTI shell 容差，单位 s/mm²
    bvec_source="rotated",  # DTI/NODDI 使用 EDDY 旋转后的梯度
    noddi_fit_method="amico",  # 保持与本页 raw-to-standard 对照相同的 NODDI 路径
).run_bids(
    bids_root="/data/bids",  # 原始 BIDS 根目录
    output_dir="/data/results/sub-01_mmorf",  # 单被试 FNIT 输出根目录
    subject="01",  # BIDS sub 标签
    session="01",  # BIDS ses 标签；无 session 时写 None
    run="01",  # 主 DWI 的 run 标签
    acquisition=None,  # 可选 acq 标签
    direction="AP",  # 主 DWI 的 dir 标签
    fa_template="FMRIB58_FA_1mm.nii.gz",  # FA 仿射初始化模板
    fa_skeleton=None,  # MMORF 不输出 TBSS skeleton
    t1=None,  # 自动读取唯一的同被试 T1w；也可显式给路径
    t1_template="MNI152_T1_1mm_brain.nii.gz",  # T1 标准模板及输出网格
    tensor_template="FSL_HCP1065_tensor_1mm.nii.gz",  # 同网格六通道 tensor 模板
    overwrite=False,  # 拒绝覆盖已有输出
)
~~~

两种调用均返回 `DMRIPipelineResult`：`native_maps` 和 `standard_maps` 是以 FA、MD、L1、L2、L3、MO、ICVF、OD、ISOVF 为键的 `nibabel.Nifti1Image` 字典；`registration_backend` 是实际分支名，`output_dir` 是结果根目录，`qc` 含各阶段运行记录。MMORF 找不到 T1w 时直接报错；TBSS 不要求 T1w。

## Python：UKB 命名输入

以下两个示例调用 `run(raw_dir=...)`，读取 `AP.*` 和可选 `PA.*`；它们与上面的 `run_bids()` 共用同一计算链。

### TBSS（不使用 T1w）

~~~python
from fnit import DMRIPipeline

result = DMRIPipeline(
    device="cuda:0",  # 运行设备：CUDA float32，允许 TF32
    registration_backend="tbss",  # 配准分支：UKB weighted FLIRT + TorchFNIRT
    fnirt_config="tbss",  # FNIRT 预设；省略时仍为 UKB Oxford 三阶段 TBSS
    synthstrip_weights=None,  # TBSS 分支不使用 SynthStrip 权重
    dti_shell=1000,  # DTIFIT 使用的目标 b-value，单位 s/mm²
    dti_tolerance=100,  # 纳入 DTI shell 的 b-value 容差
    bvec_source="rotated",  # DTIFIT/NODDI 梯度：EDDY 旋转后；对照试验可选 "raw"
    noddi_fit_method="classic",  # 连续 Watson 非线性拟合；默认 "amico"
).run(
    raw_dir="raw",  # 输入：AP.* 及可选 PA.* 的单被试目录
    output_dir="subject_tbss",  # 输出：该受试者的唯一结果根目录
    fa_template="FMRIB58_FA_1mm.nii.gz",  # 输入：标准 FA 模板及输出网格
    fa_skeleton="FMRIB58_FA-skeleton_1mm.nii.gz",  # 输入：UKB skeleton mask 模板
    t1=None,  # TBSS 分支不使用 T1w
    t1_template=None,  # TBSS 分支不使用 T1 模板
    tensor_template=None,  # TBSS 分支不使用 tensor 模板
    overwrite=False,  # 写盘策略：不覆盖已有结果
)
~~~

构造器选择设备和 registration_backend；`fnirt_config` 可传 `TBSSFNIRTConfig()`
或 `dataclasses.replace` 后的配置对象。命令行的 `--fnirt-preset` 默认在 TBSS 分支
选择 `tbss`。run 的前两个参数是原始采集目录和输出目录；fa_template 定义 1 mm MNI grid，fa_skeleton 用于 UKB 的 2000 阈值 skeleton mask。返回的 result.native_maps 与 result.standard_maps 都使用同一组九个键。

### MMORF（使用 T1w）

~~~python
from fnit import DMRIPipeline

result = DMRIPipeline(
    device="cuda:0",  # 运行设备：CUDA float32，允许 TF32
    registration_backend="mmorf",  # 配准分支：T1 + tensor 联合 TorchMMORF
    synthstrip_weights="/weights/synthstrip.1.pt",  # 输入：官方 SynthStrip 权重
    dti_shell=1000,  # DTIFIT 使用的目标 b-value，单位 s/mm²
    dti_tolerance=100,  # 纳入 DTI shell 的 b-value 容差
    bvec_source="rotated",  # DTIFIT/NODDI 梯度：EDDY 旋转后；对照试验可选 "raw"
    noddi_fit_method="classic",  # 连续 Watson 非线性拟合；默认 "amico"
).run(
    raw_dir="raw",  # 输入：AP.* 及可选 PA.* 的单被试目录
    output_dir="subject_mmorf",  # 输出：该受试者的唯一结果根目录
    fa_template="FMRIB58_FA_1mm.nii.gz",  # 输入：FA 仿射初始化模板
    fa_skeleton=None,  # MMORF 分支不生成 TBSS skeleton 图
    t1="T1w.nii.gz",  # 输入：个体原始 T1w
    t1_template="MNI152_T1_1mm_brain.nii.gz",  # 输入：T1 模板及输出网格
    tensor_template="FSL_HCP1065_tensor_1mm.nii.gz",  # 输入：公共空间六通道 tensor
    overwrite=False,  # 写盘策略：不覆盖已有结果
)
~~~

该分支先用 SynthStrip 生成 t1_brain，再分别计算 T1→MNI T1 和 FA→FMRIB58 的 12-DOF FLIRT 矩阵。随后它调用公开的 [`run_mmorf`](../mmorf/README.md) 函数，用两份仿射初始化 T1 scalar pair 与 DTI tensor pair，并只估计一个共享 warp；九张 dMRI 参数图全部使用 FA/tensor affine 与同一 warp 重采样。独立函数、命令行和 pipeline 因而共用同一实现和五文件 MMORF 输出。SynthStrip 权重可通过 fnit-setup-weights --model synthstrip 下载。

## 命令行：单被试

### 原始 BIDS 输入

BIDS 无 T1w（TBSS）：

~~~bash
fnit-dmri-pipeline --bids-root /data/bids --subject 01 --session 01 \
  --run 01 --direction AP -o /data/results/sub-01_tbss \
  --registration-backend tbss --fnirt-preset tbss \
  --fa-template FMRIB58_FA_1mm.nii.gz \
  --fa-skeleton FMRIB58_FA-skeleton_1mm.nii.gz \
  --bvec-source rotated --noddi-fit-method amico --device cuda:0
~~~

BIDS 有 T1w（MMORF）：

~~~bash
fnit-dmri-pipeline --bids-root /data/bids --subject 01 --session 01 \
  --run 01 --direction AP -o /data/results/sub-01_mmorf \
  --registration-backend mmorf --fa-template FMRIB58_FA_1mm.nii.gz \
  --t1-template MNI152_T1_1mm_brain.nii.gz \
  --tensor-template FSL_HCP1065_tensor_1mm.nii.gz \
  --synthstrip-weights /weights/synthstrip.1.pt \
  --bvec-source rotated --noddi-fit-method amico --device cuda:0
~~~

`--bids-root` 是原始 BIDS 根目录；`--subject` 选择一名被试，`--session`、`--run`、`--acquisition` 和 `--direction` 在有多次采集时限定主 DWI。第二条命令省略 `--t1`，表示从该被试的 `anat/` 读取唯一 T1w；如果有多张，用 `--t1` 明确指定。`--noddi-fit-method amico` 使用默认字典拟合，改为 `classic` 时使用连续 Watson 拟合。`-o` 保存九张 native 和 standard 参数图及报告。BIDS 图像与梯度不会被改写：`bids_input/` 中保留源文件链接、合并后的 JSON 和 `bids_selection.json`；输出九图沿用下面的 FNIT 目录结构，尚未命名为 BIDS Derivatives。

下面带 T1w 的完整实测将第二条命令中的 FA、T1、tensor 三个 1 mm 模板换为同一 MNI 网格的 2 mm 版本；重采样方法和模板哈希见[验收报告](../../validation/dmri_pipeline/bids_mmorf_2mm.real.current.json)。九图的输出 shape 由模板决定。

### UKB 命名输入

TBSS：

~~~bash
fnit-dmri-pipeline \
  --raw-dir raw \
  -o subject_tbss \
  --registration-backend tbss \
  --fnirt-preset tbss \
  --fa-template FMRIB58_FA_1mm.nii.gz \
  --fa-skeleton FMRIB58_FA-skeleton_1mm.nii.gz \
  --bvec-source rotated \
  --noddi-fit-method classic \
  --device cuda:0
~~~

MMORF：

~~~bash
fnit-dmri-pipeline \
  --raw-dir raw \
  -o subject_mmorf \
  --registration-backend mmorf \
  --fa-template FMRIB58_FA_1mm.nii.gz \
  --t1 T1w.nii.gz \
  --t1-template MNI152_T1_1mm_brain.nii.gz \
  --tensor-template FSL_HCP1065_tensor_1mm.nii.gz \
  --synthstrip-weights /weights/synthstrip.1.pt \
  --bvec-source rotated \
  --noddi-fit-method classic \
  --device cuda:0
~~~

--raw-dir 选择单个 subject；-o 是该 subject 的唯一输出根；--registration-backend 选择非线性标准化分支。其余行分别提供该分支需要的模板、T1 和权重。--device 控制所有 PyTorch 步骤使用同一设备；每次调用只处理这一名被试。--bvec-source rotated 使 DTIFIT 和 NODDI 使用 EDDY 旋转后的梯度；改为 raw 时，两者都使用原始 AP.bvec。`--noddi-fit-method classic` 将 NODDI 阶段切换到[连续 Watson 拟合](../amico_noddi/README.md)；默认 `amico` 保留原数值路径。报告的 `noddi_fit_method` 与 `noddi.fit_method` 记录实际选择。

本页下方既有 raw-to-standard 整链对照使用默认 `amico`；上面的 `classic` 示例是新增的选择方式，单独的集成结果见“经典 NODDI 接入验证”。

## 输出契约

~~~text
subject/
├── bids_input/                    # 仅 run_bids；源图链接、元数据和选择清单
├── topup/                         # only meaningful when PA is present
├── eddy/data.nii.gz
├── eddy/data.eddy_rotated_bvecs
├── native/
│   ├── dti_FA.nii.gz
│   ├── dti_MD.nii.gz
│   ├── dti_L1.nii.gz
│   ├── dti_L2.nii.gz
│   ├── dti_L3.nii.gz
│   ├── dti_MO.nii.gz
│   ├── dti_tensor.nii.gz
│   ├── NODDI_ICVF.nii.gz
│   ├── NODDI_OD.nii.gz
│   └── NODDI_ISOVF.nii.gz
├── registration/
│   ├── dti_FA_to_MNI_warp.nii.gz     # TBSS only；FNIRT 系数已含 affine
│   ├── dti_FA_to_MNI_affine.mat      # TBSS/MMORF；MMORF 转 FSL 场时使用
│   ├── mmorf_warp.nii.gz            # MMORF only；参考图像轴 mm 场
│   ├── standard/{FA,MD,L1,L2,L3,MO,ICVF,OD,ISOVF}.nii.gz
│   └── skeleton/{...}.nii.gz      # TBSS only
└── dmri_pipeline_report.json
~~~

两条分支的 `registration/standard/` 有相同的九个文件名、float32 dtype 和参数定义；输出网格由传入的模板决定。report 记录是否使用 TOPUP、各阶段耗时、设备、TF32、每个子函数的 QC 和非等价边界。

概率追踪若传入 MNI 掩膜，可把上述单被试输出根作为 `TorchProbtrackX.run(dmri_pipeline_dir=...)` 的输入。它按 TBSS/MMORF 分支生成 FSL dense 组合场，再在 BEDPOSTX diffusion 网格求逆并最近邻映射掩膜；详见[ProbtrackX 的 MNI 掩膜用法](../probtrackx/README.md)。

## UKB TBSS 对应关系

| 本包步骤 | UKB v1.5 / FSL 命令 |
|---|---|
| FA 上限、support erosion、端层归零 | fslmaths dti_FA -min 1 -ero -roi 1 X 1 Y 1 Z 0 1 |
| FLIRT 权重 | fslmaths mask -dilD -dilD -sub 1 -abs -add mask |
| affine | flirt -ref FMRIB58_FA_1mm -in dti_FA -inweight dti_FA_mask -omat affine.mat |
| nonlinear stage 1–3 | 三次 fnirt，依次使用 oxford_s1.cnf、s2.cnf、s3.cnf |
| 九图传播 | applywarp --rel -r FMRIB58_FA_1mm -w dti_FA_to_MNI_warp |
| skeleton mask | FMRIB58_FA-skeleton_1mm ≥ 2000，再乘以有效 FA mask |

包内三份 Oxford 配置与 UKB ancillary archive 的 SHA-256 完全一致。`TBSSConfig` 使用相同的六层 subsampling、FWHM、lambda、iteration 和 intensity schedule，并明确保留三个 process stage：stage 1 为四层 LM，stage 2/3 分别为 50/25 次 SCG。当前实现也包括 implicit input/reference zero mask、input mask-normalized smoothing，以及 `inwarp/intin` 的 float32 coefficient/header 和 10 位 intensity 交接。交接在一个 Python 进程内完成，不启动三个 FSL executable。

control grid 按 FSL `FullResKsp` 计算：stage 1 的 10 mm `warpres` 在最终 `subsamp=2` 后对应 full-grid 5 mm spacing；stage 2/3 使用 2 mm spacing，最终 coefficient grid 为 `[94,112,94,3]`。验证页的既有 UKB 真实单例覆盖 FLIRT、FNIRT、applywarp 与 `tbss.py`。该例的输出网格合同通过，但逐体素数值等价未通过，因此报告仍记录 `ukb_numerically_equivalent=false`。单被试 UKB 脚本使用官方 skeleton mask 相乘，并不执行经典多被试 TBSS 的跨被试最大投影；本包复现的是该单被试行为。

## MMORF 分支对应关系

MMORF 分支和 TBSS 分支共用 TOPUP、EDDY、DTIFIT、NODDI、九图命名及标准 grid。它用两次 TorchFLIRT 保持相同的 input→reference FSL affine contract，只把 FNIRT 非线性注册换成 [PyTorch MMORF](../mmorf/README.md) 的 T1 scalar + DTI tensor 联合目标。warp 位于 reference grid，三通道保存沿 reference image axes 的毫米位移，因此由 `apply_mmorf_warp` 应用，不能交给 FNIRT 的 `TorchApplyWarp`。

## 真实数据验证与原软件对照

下面几组结果的输入和计时边界不同，完整方法、逐图误差及源码范围见[验证页](../../validation/dmri_pipeline/README.md)。

| 验证 | 已完成的输出 | 结果与范围 |
|---|---|---|
| [原始 BIDS、无 T1w、TBSS](../../validation/dmri_pipeline/bids_tbss.real.current.json) | 九张 native、九张标准空间和九张 skeleton 图 | 真实 AP/PA 105 volume DWI；各组 shape/affine 一致、float32、有限值；共享 H100 完整命令 1649.82 s。 |
| [原始 BIDS、带 T1w、MMORF 2 mm](../../validation/dmri_pipeline/bids_mmorf_2mm.real.current.json) | 九张 native、九张标准空间图和 MMORF 形变场 | 同一真实 AP/PA+T1w 采集；standard 九图同 91×109×91 网格、float32、有限值；完整命令 1829.25 s。 |
| [原始 BIDS、带 T1w、MMORF 1 mm](../../validation/dmri_pipeline/bids_mmorf.shared_gpu_oom.json) | 九张 native 图、T1 脑图、两份 FLIRT 矩阵 | 36:20.19 后在 MMORF 非线性阶段因共享 GPU 显存不足退出；标准空间九图未完成。 |
| [既有 UKB TBSS 与 FSL 对照](../../validation/dmri_pipeline/tbss_e2e.real.current.json) | 九张 native、standard、skeleton 图 | 相对匹配 FSL EDDY 的下游对照：native r=0.978640–0.999882、standard r=0.940917–0.988332、skeleton r=0.949470–0.991235。 |
| [经典 NODDI 接入](../../validation/dmri_pipeline/pipeline_classic_real.public.json) | 九图接口与真实体素拟合 | 24 个真实脑内体素的阶段集成测试；使用 identity 配准，不是 raw-to-MNI 整链。 |

上述 FSL EDDY 对照固定 TOPUP 场、掩膜和后续 FNIT 步骤，只替换 EDDY 的校正图。另一个[原版 FSL TBSS 参考](../../validation/dmri_pipeline/README.md#既有-ukb-tbss-与-fsl-对照)从官方 UKB native 参数图开始，其 standard 九图 r=0.358268–0.699381；它与本包 raw AP/PA 起步的输入不同，不能据此计算整链加速比或把差异单独归给 FNIRT。FSL 对照的九图相关性不是本次 BIDS 运行重新测得的结果。

## 参考文献与原实现

- 参考文献：Alfaro-Almagro et al., *Image processing and Quality Control for the first 10,000 brain imaging datasets from UK Biobank*, NeuroImage (2018), [论文](https://discovery.ucl.ac.uk/id/eprint/10039942/)。
- 参考文献：Smith et al., *Tract-based spatial statistics: Voxelwise analysis of multi-subject diffusion data*, NeuroImage (2006), [doi:10.1016/j.neuroimage.2006.02.024](https://doi.org/10.1016/j.neuroimage.2006.02.024)。
- 原实现代码库：[UK Biobank pipeline v1.5](https://git.fmrib.ox.ac.uk/falmagro/uk_biobank_pipeline_v_1.5)；[FSL `tbss`](https://git.fmrib.ox.ac.uk/fsl/tbss)；[FSL `MMORF`（可选配准分支）](https://git.fmrib.ox.ac.uk/fsl/MMORF)。
- 经典 NODDI 参考：Zhang et al., *NODDI: Practical in vivo neurite orientation dispersion and density imaging of the human brain*, NeuroImage (2012), [doi:10.1016/j.neuroimage.2012.03.072](https://doi.org/10.1016/j.neuroimage.2012.03.072)；[NODDI Matlab Toolbox](https://www.nitrc.org/projects/noddi_toolbox)。
- 输入格式：[BIDS 1.11.1 MRI/DWI 与反向相位编码 EPI 规范](https://bids-specification.readthedocs.io/en/stable/modality-specific-files/magnetic-resonance-imaging-data.html)。
