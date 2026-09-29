# 单被试 dMRI 参数图流程

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
    fnirt_config="tbss",  # FNIRT 预设；省略时仍为 UKB Oxford 三阶段 TBSS
    synthstrip_weights=None,  # TBSS 分支不使用 SynthStrip 权重
    dti_shell=1000,  # DTIFIT 使用的目标 b-value，单位 s/mm²
    dti_tolerance=100,  # 纳入 DTI shell 的 b-value 容差
    bvec_source="rotated",  # DTIFIT/NODDI 梯度：EDDY 旋转后；对照试验可选 "raw"
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

## Python：MMORF 分支

~~~python
from fnit import DMRIPipeline

result = DMRIPipeline(
    device="cuda:0",  # 运行设备：CUDA float32，允许 TF32
    registration_backend="mmorf",  # 配准分支：T1 + tensor 联合 TorchMMORF
    synthstrip_weights="/weights/synthstrip.1.pt",  # 输入：官方 SynthStrip 权重
    dti_shell=1000,  # DTIFIT 使用的目标 b-value，单位 s/mm²
    dti_tolerance=100,  # 纳入 DTI shell 的 b-value 容差
    bvec_source="rotated",  # DTIFIT/NODDI 梯度：EDDY 旋转后；对照试验可选 "raw"
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
  --fnirt-preset tbss \
  --fa-template FMRIB58_FA_1mm.nii.gz \
  --fa-skeleton FMRIB58_FA-skeleton_1mm.nii.gz \
  --bvec-source rotated \
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
  --device cuda:0
~~~

--raw-dir 选择单个 subject；-o 是该 subject 的唯一输出根；--registration-backend 只改变非线性标准化分支。其余行分别提供该分支需要的模板、T1 和权重。--device 控制所有 PyTorch 步骤使用同一设备；每次调用只处理这一名被试。--bvec-source rotated 使 DTIFIT 和 NODDI 使用 EDDY 旋转后的梯度；改为 raw 时，两者都使用原始 AP.bvec。EDDY 的图像校正及其自身保存的旋转梯度文件照常运行，报告中的 bvec_source 记录实际拟合输入。

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
│   ├── dti_FA_to_MNI_warp.nii.gz     # TBSS only；FNIRT 系数已含 affine
│   ├── dti_FA_to_MNI_affine.mat      # TBSS/MMORF；MMORF 转 FSL 场时使用
│   ├── mmorf_warp.nii.gz            # MMORF only；参考图像轴 mm 场
│   ├── standard/{FA,MD,L1,L2,L3,MO,ICVF,OD,ISOVF}.nii.gz
│   └── skeleton/{...}.nii.gz      # TBSS only
└── dmri_pipeline_report.json
~~~

两条分支的 registration/standard 中九个文件名、MNI grid、float32 dtype 和参数定义相同。report 记录是否使用 TOPUP、各阶段耗时、设备、TF32、每个子函数的 QC 和非等价边界。

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

control grid 按 FSL `FullResKsp` 计算：stage 1 的 10 mm `warpres` 在最终 `subsamp=2` 后对应 full-grid 5 mm spacing；stage 2/3 使用 2 mm spacing，最终 coefficient grid 为 `[94,112,94,3]`。下述真实单例覆盖 FLIRT、FNIRT、applywarp 与 `tbss.py`。该例的输出网格合同通过，但逐体素数值等价未通过，因此报告仍记录 `ukb_numerically_equivalent=false`。单被试 UKB 脚本使用官方 skeleton mask 相乘，并不执行经典多被试 TBSS 的跨被试最大投影；本包复现的是该单被试行为。

## MMORF 分支对应关系

MMORF 分支和 TBSS 分支共用 TOPUP、EDDY、DTIFIT、NODDI、九图命名及标准 grid。它用两次 TorchFLIRT 保持相同的 input→reference FSL affine contract，只把 FNIRT 非线性注册换成 [PyTorch MMORF](../mmorf/README.md) 的 T1 scalar + DTI tensor 联合目标。warp 位于 reference grid，三通道保存沿 reference image axes 的毫米位移，因此由 `apply_mmorf_warp` 应用，不能交给 FNIRT 的 `TorchApplyWarp`。

## 当前真实数据验证

2026 年 9 月 29 日在 gpucw1 H100 用一例真实 UKB 格式 AP/PA dMRI，从原始图像运行 TOPUP、新版 TorchEDDY、DTIFIT、NODDI 和两条配准分支。TBSS 使用与既有报告哈希一致的 FMRIB58 FA/skeleton 模板；MMORF 另使用同一病例的 T1w、MNI T1/tensor 模板及官方 SynthStrip 权重。两条分支分别实际运行，不混用旧 EDDY 结果。数值比较使用非零体素并集。

此后 TorchEDDY 只将样条权重改为无布尔索引计算；固定种子的完整八轮校正图与改动前文件 SHA-256 相同，参数和其余数值输出也逐值相同。本页九图精度比较来自改动前的整链实测，两条分支尚未用改动后源码重跑。下列整链阶段耗时也只代表那次运行；当前 EDDY 单独运行时间见 [EDDY 验证页](../../validation/eddy/README.md)。

### TBSS：九张标准图和 skeleton 图

“匹配 FSL EDDY”一列固定同一 TOPUP 场与脑掩膜，使用 `eddy_cuda10.2` 输出，再调用相同的 FNIT DTIFIT、NODDI 和 TBSS 代码。它隔离 EDDY 输入造成的差异。“原版 FSL TBSS”一列使用先前从官方 UKB native 参数图开始，经 FSL weighted FLIRT、三阶段 FNIRT、applywarp 生成的标准图和 skeleton 图；FA 文件哈希分别为 `ce420c…`、`ad2d24…`。后者与本流程从 raw AP/PA 起步的范围不同。

| 图 | 匹配 FSL EDDY：native r | 匹配 FSL EDDY：standard r | 原版 FSL TBSS：standard r | 匹配 FSL EDDY：skeleton r | 原版 FSL TBSS：skeleton r |
|---|---:|---:|---:|---:|---:|
| FA | 0.999224 | 0.988332 | 0.667522 | 0.991235 | 0.517921 |
| MD | 0.999882 | 0.981015 | 0.649976 | 0.954968 | 0.489054 |
| L1 | 0.999797 | 0.979714 | 0.591519 | 0.952986 | 0.426730 |
| L2 | 0.999823 | 0.981538 | 0.661351 | 0.961240 | 0.553910 |
| L3 | 0.999819 | 0.981895 | 0.699381 | 0.966816 | 0.596406 |
| MO | 0.978640 | 0.940917 | 0.511447 | 0.968428 | 0.640657 |
| ICVF | 0.986409 | 0.977894 | 0.358268 | 0.949470 | 0.098878 |
| OD | 0.993978 | 0.976456 | 0.632424 | 0.974554 | 0.651267 |
| ISOVF | 0.999170 | 0.978553 | 0.690850 | 0.962090 | 0.599029 |

九图相对匹配 FSL EDDY 的 native r 为 0.978640–0.999882，standard r 为 0.940917–0.988332，skeleton r 为 0.949470–0.991235。相对原版 FSL TBSS 的 standard r 为 0.358268–0.699381，skeleton r 为 0.098878–0.651267；修正 EDDY 并未消除官方整链差距。两套比较的 shape 和 affine 均一致。原版参考使用不同的上游参数图，因此这些低相关不能单独归咎于 FNIRT。

| 阶段 | 本次耗时 |
|---|---:|
| TOPUP 与 EDDY 输入准备 | 64.82 s |
| 新版 TorchEDDY | 673.42 s |
| DTIFIT | 17.18 s |
| AMICO-NODDI | 28.25 s |
| TBSS 配准、九图传播和 skeleton | 58.02 s |
| 完整进程 wall | 14:06.89 |

EDDY 的进程内 CUDA allocation 峰值为 4.54 GiB；全流程最大组件峰值来自 NODDI，为 12.87 GiB。EDDY 对同输入 FSL GPU 的 4D r=0.999727、MAE=21.46，离群切片重合 14/15，预设阈值通过；独立测评见 [EDDY 页面](../eddy/README.md)。FSL TBSS 参考的时间只覆盖 prepared native 图后的配准，不能与上述 raw-to-standard 时间计算加速比。逐图 MAE、RMSE、网格合同和完整阶段记录见[当前 TBSS 报告](../../validation/dmri_pipeline/tbss_e2e.real.current.json)。

### MMORF：T1 与 tensor 联合配准

当前 MMORF 模块已用真实 T1w、FA 双标量与 tensor 完成和官方实现的配对对照，见 [MMORF 验证](../mmorf/README.md)。此前完整 dMRI pipeline 的 MMORF 分支使用旧求解器，旧版整链指标已移除。当前源码的 raw-to-standard MMORF 整链仍需重新验证。
