# 单被试 dMRI 参数图 pipeline

[返回首页](../../README.md) · [PyTorch MMORF](../mmorf/README.md)

本流程实现以下固定链：

~~~mermaid
flowchart LR
  A[AP dMRI + optional PA] --> B{PA complete?}
  B -->|yes| C[TorchTOPUP]
  B -->|no| D[zero susceptibility field]
  C --> E[TorchEDDY]
  D --> E
  E --> F[TorchDTIFIT b≈1000]
  E --> G[TorchAMICO-NODDI all shells]
  F --> H{registration backend}
  G --> H
  H -->|tbss| I[weighted TorchFLIRT + three-stage TorchFNIRT]
  H -->|mmorf| J[SynthStrip T1 + two TorchFLIRT + run_mmorf]
  I --> K[9 maps on FMRIB58/MNI152 1 mm grid]
  J --> K
~~~

两条分支输出完全相同的九个标准空间参数名：FA、MD、L1、L2、L3、MO、ICVF、OD、ISOVF。这里的“一致”指文件集合、shape、affine、dtype 和参数含义一致；它不表示 TBSS/FNIRT 与 MMORF 会估计相同的非线性形变。TBSS 分支另输出九张 UKB 风格 skeleton-mask 图。

## 输入目录

raw_dir 使用 UKB 命名：

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

## Python：TBSS 分支

~~~python
from fnit import DMRIPipeline

result = DMRIPipeline(
    device="cuda:0",  # 运行设备：CUDA float32，允许 TF32
    registration_backend="tbss",  # 配准分支：UKB weighted FLIRT + TorchFNIRT
    synthstrip_weights=None,  # TBSS 分支不使用 SynthStrip 权重
    dti_shell=1000,  # DTIFIT 使用的目标 b-value，单位 s/mm²
    dti_tolerance=100,  # 纳入 DTI shell 的 b-value 容差
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

构造器选择设备和 registration_backend。run 的前两个参数是原始采集目录和输出目录；fa_template 定义 1 mm MNI grid，fa_skeleton 用于 UKB 的 2000 阈值 skeleton mask。返回的 result.native_maps 与 result.standard_maps 都使用同一组九个键。

## Python：MMORF 分支

~~~python
from fnit import DMRIPipeline

result = DMRIPipeline(
    device="cuda:0",  # 运行设备：CUDA float32，允许 TF32
    registration_backend="mmorf",  # 配准分支：T1 + tensor 联合 TorchMMORF
    synthstrip_weights="/weights/synthstrip.1.pt",  # 输入：官方 SynthStrip 权重
    dti_shell=1000,  # DTIFIT 使用的目标 b-value，单位 s/mm²
    dti_tolerance=100,  # 纳入 DTI shell 的 b-value 容差
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

TBSS：

~~~bash
fnit-dmri-pipeline \
  --raw-dir raw \
  -o subject_tbss \
  --registration-backend tbss \
  --fa-template FMRIB58_FA_1mm.nii.gz \
  --fa-skeleton FMRIB58_FA-skeleton_1mm.nii.gz \
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
  --device cuda:0
~~~

--raw-dir 选择单个 subject；-o 是该 subject 的唯一输出根；--registration-backend 只改变非线性标准化分支。其余行分别提供该分支需要的模板、T1 和权重。--device 控制所有 PyTorch 步骤使用同一设备；每次调用只处理这一名被试。

## 输出契约

~~~text
subject/
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
│   ├── standard/{FA,MD,L1,L2,L3,MO,ICVF,OD,ISOVF}.nii.gz
│   └── skeleton/{...}.nii.gz      # TBSS only
└── dmri_pipeline_report.json
~~~

两条分支的 registration/standard 中九个文件名、MNI grid、float32 dtype 和参数定义相同。report 记录是否使用 TOPUP、各阶段耗时、设备、TF32、每个子函数的 QC 和非等价边界。

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

control grid 按 FSL `FullResKsp` 计算：stage 1 的 10 mm `warpres` 在最终 `subsamp=2` 后对应 full-grid 5 mm spacing，输出 `[39,46,39,3]`；stage 2/3 使用 2 mm spacing，输出 `[94,112,94,3]`。shape、spacing 和 sform 已与官方文件匹配。当前真实病例已通过逐 LM oracle；由于后续数值归约仍有小差异、topology 分支未触发且病例数为一，`ukb_numerically_equivalent=false` 仍保留。单被试 UKB 脚本使用官方 skeleton mask 相乘，并不执行经典多被试 TBSS 的跨被试最大投影；本包复现的是该单被试行为。

## MMORF 分支对应关系

MMORF 分支和 TBSS 分支共用 TOPUP、EDDY、DTIFIT、NODDI、九图命名及标准 grid。它用两次 TorchFLIRT 保持相同的 input→reference FSL affine contract，只把 FNIRT 非线性注册换成 [PyTorch MMORF](../mmorf/README.md) 的 T1 scalar + DTI tensor 联合目标。warp 位于 reference grid，三通道保存沿 reference image axes 的毫米位移，因此由 `apply_mmorf_warp` 应用，不能交给 FNIRT 的 `TorchApplyWarp`。

## 验证边界

当前发布只列出由现行源码生成的证据：

- TOPUP、EDDY、DTIFIT 和 AMICO-NODDI 的真实数据结果分别位于各自的[验证目录](../../validation/README.md)。
- TBSS 注册从同一组九张 native 参数图开始，比较官方 weighted FLIRT、三阶段 FNIRT、九图传播和 skeleton multiplication；结果见 [`tbss_diagnosis.public.json`](../../validation/dmri_pipeline/tbss_diagnosis.public.json)。
- MMORF 的独立真实数据对照固定 T1、DTI tensor、模板和 FLIRT 初始化，结果见 [`validation/mmorf`](../../validation/mmorf/README.md)。

当前版本尚无从原始 AP/PA 到九张 standard 图的 fresh 端到端配对 benchmark。
以下 TBSS 表只衡量 registration 及其后续传播；MMORF 表与图只衡量独立注册函数。

### 当前 TBSS 真实数据 benchmark

对照使用一例去标识、真实 UKB 格式 dMRI，并从同一组九张 native 参数图开始。A 是
用户实际调用的 TorchFLIRT + TorchFNIRT 路径；B 固定官方 FLIRT matrix，用于隔离
nonlinear registration。

| 实现 | affine | 观测时间 | peak CUDA allocation |
|---|---|---:|---:|
| FSL 6.0.7.4 / UKB CPU | FSL FLIRT | 976.88 s external wall | 不适用 |
| FNIT A / H100 | 本次重新运行 TorchFLIRT | 106.186 s Python pipeline；121.48 s external wall | 3.607 GB |
| FNIT B / H100 | 固定官方 FLIRT matrix | 18.884 s Python pipeline | 3.608 GB |

B 排除了 affine 优化；共享节点负载没有隔离，因此不报告“等价加速比”。A/B 的九张
standard 和 skeleton 图均通过 shape、affine、dtype 合同：

| map | A standard r | A skeleton r | B standard r | B skeleton r |
|---|---:|---:|---:|---:|
| FA | 0.998886 | 0.999463 | 0.999523 | 0.999679 |
| MD | 0.997129 | 0.997087 | 0.998712 | 0.994744 |
| L1 | 0.996619 | 0.997320 | 0.998438 | 0.994806 |
| L2 | 0.997222 | 0.997592 | 0.998738 | 0.995896 |
| L3 | 0.997640 | 0.997887 | 0.998937 | 0.996788 |
| MO | 0.996725 | 0.998206 | 0.998262 | 0.998889 |
| ICVF | 0.995086 | 0.996789 | 0.997483 | 0.994425 |
| OD | 0.997079 | 0.998257 | 0.998542 | 0.997927 |
| ISOVF | 0.997986 | 0.997748 | 0.999087 | 0.997208 |

A 的 standard/skeleton FA MAE 为 `0.003129/0.003111`；B 为
`0.002418/0.002422`。完整 MAE、RMSE、grid contract、Jacobian 和 source hash 见
[`tbss_diagnosis.public.json`](../../validation/dmri_pipeline/tbss_diagnosis.public.json)。

### FNIRT 差异诊断

真实 level-1 LM oracle 的 initial cost/mask 为 Torch `639.821722/2598`、FSL
`639.822/2598`。十次候选的接受序列均为
`A,A,A,R,R,R,R,A,R,A`，最终为 Torch `478.784565/2508`、FSL
`478.784/2508`。完整 stage 1 coefficient `r=0.999832`，warped FA
`r=0.999004`；GPU 对应值为 `0.999879/0.999148`。

关键修复来自 FSL 的 input-mask 数据类型：`general_transform` 把三线性插值结果写入
`volume<char>`，因此边界小数在 `Mask()>0.5` 之前已截断。当前实现还匹配
newimage `1e-8` valid-FOV tolerance、zero-padded boundary neighbor、implicit-zero
`1e-16` 和 input-mask-after-normalization 顺序。详细逐 attempt 表见
[TorchFNIRT 页面](../fnirt/README.md#stage-1-lm-oracle-与修复原因)。

后续层的 sparse BFMatrix 与 matrix-free FP64 reduction 仍有小数值分叉。topology
projection 未在本例触发，且当前只有一个真实病例，所以不声明全局逐体素等价。定向
测试为 `63 passed`。

下图显示同一次 current A 运行的官方 FA、FNIT FA 和绝对差值。数值指标来自完整 3D
NIfTI，不由切片估计。

![FSL/UKB 与 FNIT TorchTBSS 的真实 FA 对照](../fnirt/figures/fnirt_fsl_comparison.png)

公开仓库不包含该病例的原始 dMRI、native 参数图或 subject 标识。重现脚本和去标识
汇总报告位于 [`validation/dmri_pipeline`](../../validation/dmri_pipeline/README.md)。
