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

control grid 按 FSL `FullResKsp` 计算：stage 1 的 10 mm `warpres` 在最终 `subsamp=2` 后对应 full-grid 5 mm spacing；stage 2/3 使用 2 mm spacing，最终 coefficient grid 为 `[94,112,94,3]`。当前真实单例已经覆盖迁移后的 FLIRT、FNIRT、applywarp 与 `tbss.py`，源码哈希和结果见下文。该例的输出网格合同通过，但逐体素数值等价未通过，因此报告仍记录 `ukb_numerically_equivalent=false`。单被试 UKB 脚本使用官方 skeleton mask 相乘，并不执行经典多被试 TBSS 的跨被试最大投影；本包复现的是该单被试行为。

## MMORF 分支对应关系

MMORF 分支和 TBSS 分支共用 TOPUP、EDDY、DTIFIT、NODDI、九图命名及标准 grid。它用两次 TorchFLIRT 保持相同的 input→reference FSL affine contract，只把 FNIRT 非线性注册换成 [PyTorch MMORF](../mmorf/README.md) 的 T1 scalar + DTI tensor 联合目标。warp 位于 reference grid，三通道保存沿 reference image axes 的毫米位移，因此由 `apply_mmorf_warp` 应用，不能交给 FNIRT 的 `TorchApplyWarp`。

## 当前真实数据验证

以下数值运行使用源码快照 tar `f7547d0a39ddd9fb6ba70deb720f229ecedc6385fa72d457efb2ded78b6c173d`，其中 `flirt/core.py` 为 `552856…`；当前文件为 `ce375d…`。继承链分两段：[第一段](../../validation/runtime_dependencies/flirt_qc_source_equivalence.public.json)只清理 runtime QC，[第二段](../../validation/runtime_dependencies/flirt_profile_source_equivalence.public.json)证明本流程使用的 12-DOF/corratio 数值路径未变。报告保留原测量 hash，并在新增 chain 对象中明确 `fresh=false`。这不是 current-hash 完整真实数据重跑，也不覆盖已改变的 6-DOF/normmi 路径。
两份报告记录的 `__init__.py`/`cli.py` 为 `cc9aa4…`/`2d5e3d…`，当前 0.16.0 为
`be1cab…`/`c92a3f…`。归一化版本号、移除独立的 fMRI 懒加载分支（含 13 个 surface API），并从新旧源码中
过滤五个主动撤下的内部实现名称后，保留 API 的新旧 AST SHA-256 均为 `fd295c…`。
这五个名称不主张 API 兼容；`dmri-pipeline` 的动态 parser、最终 `parse_args` 和 handler
分发 AST 未变。完整入口证明见
[包入口源码等价证明](../../validation/runtime_dependencies/package_entry_source_equivalence.public.json)。

两份 dMRI source manifest 均未记录 SynthMorph，TBSS 和 MMORF 路径也都不执行 SynthMorph。机器报告以结构化字段记录 `executed=false` 和 `attestation_applicable=false`，因此这里不引用 SynthMorph linear 证明作数值继承。

### MMORF：raw AP/PA 到九张标准图

2026 年 9 月 28 日在 gpucw1 的 NVIDIA H100 PCIe 上，用上述数值运行冻结快照跑完 1 例去标识化真实 UKB 格式 AP/PA dMRI 和配对 T1w。输入从 `raw/AP.*`、`raw/PA.*` 开始，实际执行 TOPUP、EDDY、DTIFIT、AMICO-NODDI、SynthStrip、两次 TorchFLIRT、TorchMMORF 和九张图的 warp propagation。源码快照 tar SHA-256 为 `f7547d0a39ddd9fb6ba70deb720f229ecedc6385fa72d457efb2ded78b6c173d`；`dmri_pipeline/pipeline.py` 的 SHA-256 为 `7ed08495a70065f46e8cd6a8402053ec9e5d6a764e5ce8ec37dc961fc369a347`，`mmorf/core.py` 为 `a89092d2ef561b53049400361a646b9cc8f94300bcb7b3eb14b37ed1e4a18466`；机器报告逐项记录 50 个调用链源码哈希。

参考图由既有 UKB native DTI/NODDI 参数图和 FSL MMORF 0.3.2 warp 生成。该参考传播沿用了先前匹配验证中固定的 FNIT affine，因此不是“全部阶段均由官方软件重跑”的 raw-to-standard reference。下面的数值用于检查当前端到端输出和暴露差异，不能作为完全匹配的官方验收。

| 参数图 | native r | standard r | standard MAE | standard RMSE |
|---|---:|---:|---:|---:|
| FA | 0.586157 | 0.559845 | 0.099869 | 0.151588 |
| MD | 0.547537 | 0.563159 | 3.677e-4 | 5.874e-4 |
| L1 | 0.436994 | 0.535731 | 4.042e-4 | 6.454e-4 |
| L2 | 0.567251 | 0.563184 | 3.769e-4 | 5.930e-4 |
| L3 | 0.634022 | 0.585998 | 3.622e-4 | 5.631e-4 |
| MO | 0.413579 | 0.329829 | 0.370717 | 0.481165 |
| ICVF | -0.002070 | 0.372548 | 0.146745 | 0.223835 |
| OD | 0.506862 | 0.540236 | 0.154255 | 0.220781 |
| ISOVF | 0.608481 | 0.528075 | 0.177561 | 0.269235 |

九张 standard 图的 shape、affine 和 float32 dtype 均与 reference 一致；文件集合和网格合同通过。所有连续值指标都未达到逐体素数值等价，`numerical_equivalence_passed=false`。native 图在配准前已经存在明显差异，尤其 ICVF，因此标准空间差异不能只归因于 MMORF。当前 end-to-end 结果也不能继承组件报告中的 AMICO 数值等价结论。

| 阶段 | 当前 H100 时间 |
|---|---:|
| TOPUP 和 EDDY 准备 | 136.625 s |
| EDDY | 49.675 s |
| DTIFIT | 17.502 s |
| AMICO-NODDI | 45.753 s |
| SynthStrip、两次 FLIRT、MMORF 和九图传播 | 308.496 s |
| Python pipeline 内部总计 | 558.100 s |
| 外部进程 wall | 563.06 s |

最大组件级 CUDA allocation 为 13.669 GB，来自 AMICO-NODDI；MMORF 为 12.320 GB，EDDY 为 3.480 GB，DTIFIT 为 1.313 GB。外部进程最大 CPU RSS 为 2,776,112 KiB。启动时同卡已有其他训练，占用 22,246.8 MiB 且利用率为 100%，所以 563.06 s 只证明完整运行，不作为隔离性能值。FSL MMORF 的既有外部记录为 947.62 s，但只含 MMORF registration；两者范围和负载均不同，不计算加速比。

![当前 raw-to-standard MMORF 的真实 FA 对照](figures/dmri_mmorf_fa_real.png)

机器报告见 [`mmorf_e2e.real.current.json`](../../validation/dmri_pipeline/mmorf_e2e.real.current.json)，复现和比较脚本见 [`validate_mmorf_e2e.py`](../../validation/dmri_pipeline/validate_mmorf_e2e.py)。仓库不保存原始 dMRI、T1w 或受试者标识，只发布输入文件 SHA-256。

### TBSS：raw AP/PA 到九张标准图和 skeleton 图

2026 年 9 月 28 日在 gpucw1 的 NVIDIA H100 PCIe 上，用上述数值运行冻结快照跑完同一例去标识化真实 UKB 格式 AP/PA dMRI。执行命令为：

~~~bash
python -m fnit.cli dmri-pipeline \
  --raw-dir <RAW_DIR> \
  --output-dir <OUTPUT_DIR> \
  --registration-backend tbss \
  --fa-template <FMRIB58_FA_1mm.nii.gz> \
  --fa-skeleton <FMRIB58_FA-skeleton_1mm.nii.gz> \
  --device cuda
~~~

`--raw-dir` 指向一名被试的 AP/PA 原始采集；`--output-dir` 是该被试的独立输出目录；`--registration-backend tbss` 选择 weighted FLIRT、三阶段 TorchFNIRT、九图传播和 skeleton multiplication；`--fa-template` 同时定义配准目标与 1 mm 输出网格；`--fa-skeleton` 提供 UKB 单被试 skeleton mask；`--device cuda` 在 H100 上执行 PyTorch 阶段。实测源码快照 tar SHA-256 为 `f7547d0a39ddd9fb6ba70deb720f229ecedc6385fa72d457efb2ded78b6c173d`；`pipeline.py`、`tbss.py`、`flirt/core.py` 和 `fnirt/registration.py` 分别为 `7ed08495a70065f46e8cd6a8402053ec9e5d6a764e5ce8ec37dc961fc369a347`、`93abf99c86dcc688959b34eb1a8e8b0bd8bec6a561b7332a4782f2338f937bfc`、`5528560292d3ba72778fda619abd907ab4cd666b80a8f18b95538ee4b8ca89b2` 和 `a63ed0b09e43a5af4bf63b2f583e710b1d0fc73aac548a326c552334a741cd83`。机器报告逐项记录 52 个调用链源码哈希。

native reference 是既有官方 UKB DTI/NODDI 参数图。standard 和 skeleton reference 从这套 prepared official maps 开始，实际运行 FSL 6.0.7.4 weighted FLIRT、三次 FNIRT、applywarp 和 skeleton multiplication。候选则从 raw AP/PA 开始，因此这是当前完整链的真实回归，也是上游差异和配准差异的合并结果。

| 参数图 | native r | standard r | standard MAE | skeleton r | skeleton MAE |
|---|---:|---:|---:|---:|---:|
| FA | 0.586157 | 0.694611 | 0.087692 | 0.561165 | 0.123220 |
| MD | 0.547537 | 0.600453 | 3.624e-4 | 0.447009 | 2.192e-4 |
| L1 | 0.436994 | 0.535743 | 4.229e-4 | 0.413255 | 2.896e-4 |
| L2 | 0.567251 | 0.613368 | 3.619e-4 | 0.504887 | 2.166e-4 |
| L3 | 0.634022 | 0.658698 | 3.274e-4 | 0.540063 | 1.942e-4 |
| MO | 0.413579 | 0.464303 | 0.320516 | 0.622653 | 0.285623 |
| ICVF | -0.002070 | 0.313822 | 0.165453 | 0.107073 | 0.142074 |
| OD | 0.506862 | 0.592098 | 0.148483 | 0.605502 | 0.098723 |
| ISOVF | 0.608481 | 0.624363 | 0.154927 | 0.525343 | 0.090987 |

九张 standard 图和九张 skeleton 图的 shape、affine、float32 dtype 均与 reference 一致；文件集合和网格合同通过。连续值没有达到逐体素数值等价，`numerical_equivalence_passed=false`。native 图在配准前的 Pearson r 已为 -0.002070–0.634022，其中 ICVF 几乎不相关，所以不能把 standard 或 skeleton 的全部差异归因于 FLIRT/FNIRT。

| 阶段 | 当前 H100 时间 |
|---|---:|
| TOPUP 和 EDDY 准备 | 102.208 s |
| EDDY | 54.489 s |
| DTIFIT | 19.137 s |
| AMICO-NODDI | 50.678 s |
| weighted FLIRT、三阶段 FNIRT、九图传播和 skeleton | 213.520 s |
| Python pipeline 内部总计 | 440.092 s |
| 外部进程 wall | 444.93 s |

最大组件级 CUDA allocation 为 13.664 GB，来自 AMICO-NODDI；TBSS 配准为 3.632 GB，EDDY 为 3.475 GB，DTIFIT 为 1.313 GB。外部进程最大 CPU RSS 为 2,510,476 KiB。启动时同卡已有其他训练，占用 20,869 MiB 且利用率为 100%，所以 444.93 s 只证明完整运行，不作为隔离性能值。FSL 参考的外部 wall 为 976.88 s，但它从 prepared FA 和九张参数图开始；计时边界和负载不同，不计算加速比。候选保存 float32 影像；DTIFIT、NODDI 和 FNIRT 在文档所列求解步骤中使用 float64，CUDA matmul 和 cuDNN 允许 TF32，没有使用 float16 或 bfloat16。

![当前 raw-to-standard TBSS 的真实 FA 对照](figures/dmri_tbss_fa_real.png)

机器报告见 [`tbss_e2e.real.current.json`](../../validation/dmri_pipeline/tbss_e2e.real.current.json)，比较和作图脚本见 [`validate_tbss_e2e.py`](../../validation/dmri_pipeline/validate_tbss_e2e.py)。报告保存九张 native、standard 和 skeleton 图的逐图指标、输入与 reference SHA-256、完整调用链源码 SHA-256、阶段时间和显存。仓库不保存原始 dMRI 或受试者标识。
