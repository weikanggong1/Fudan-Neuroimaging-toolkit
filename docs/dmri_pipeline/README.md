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
  H -->|tbss| I[weighted TorchFLIRT + combined six-level TorchFNIRT]
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
    device="cuda:0",                 # CUDA float32，允许 TF32
    registration_backend="tbss",     # UKB FA registration branch
).run(
    "raw",                            # AP.* and optional PA.*
    "subject_tbss",                   # one-subject output root
    fa_template="FMRIB58_FA_1mm.nii.gz",
    fa_skeleton="FMRIB58_FA-skeleton_1mm.nii.gz",
)
~~~

构造器选择设备和 registration_backend。run 的前两个参数是原始采集目录和输出目录；fa_template 定义 1 mm MNI grid，fa_skeleton 用于 UKB 的 2000 阈值 skeleton mask。返回的 result.native_maps 与 result.standard_maps 都使用同一组九个键。

## Python：MMORF 分支

~~~python
from fnit import DMRIPipeline

result = DMRIPipeline(
    device="cuda:0",
    registration_backend="mmorf",
    synthstrip_weights="/weights/synthstrip.1.pt",
).run(
    "raw",
    "subject_mmorf",
    fa_template="FMRIB58_FA_1mm.nii.gz",
    t1="T1w.nii.gz",
    t1_template="MNI152_T1_1mm_brain.nii.gz",
    tensor_template="FSL_HCP1065_tensor_1mm.nii.gz",
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

包内三份 oxford 配置与 UKB ancillary archive 的 SHA-256 完全一致。`TBSSConfig` 选取相同六层 subsampling、FWHM、lambda、intensity-estimation 和 10→2 mm control-resolution 数值，但把官方三个进程合并为一次连续调用。当前实现还没有复现 implicit zero masks、mask-normalized smoothing、`inwarp/intin` 的阶段交接和 stage 2/3 SCG。因此 FNIRT 结果不是逐值等价；具体隔离证据见下方诊断。单被试 UKB 脚本使用官方 skeleton mask 相乘，并不执行经典多被试 TBSS 的跨被试最大投影；本包复现的是这一行为。

## MMORF 分支对应关系

MMORF 分支和 TBSS 分支共用 TOPUP、EDDY、DTIFIT、NODDI、九图命名及标准 grid。它用两次 TorchFLIRT 保持相同的 input→reference FSL affine contract，只把 FNIRT 非线性注册换成 [PyTorch MMORF](../mmorf/README.md) 的 T1 scalar + DTI tensor 联合目标。warp 是 MMORF reference-voxel displacement，因此由 apply_mmorf_warp 应用，不能交给 FNIRT 的 TorchApplyWarp。

## 验证

真实 UKB 格式单被试的 FA 前处理图和 FLIRT input-weight 图与 UKB/FSL 命令逐体素完全相同：差异体素 0，最大绝对误差 0。完整 MMORF 分支写出的九张标准图均为 `(182, 218, 182)`、float32、有限且非空，并与 FMRIB58/MNI152 1 mm affine 一致。

从原始 AP/PA 到 MMORF 九图的 H100 wall time 为 372.28 s，内部阶段计时如下：

| stage | seconds |
|---|---:|
| TOPUP + EDDY preparation | 98.55 |
| EDDY | 48.88 |
| DTIFIT | 17.25 |
| AMICO-NODDI | 36.40 |
| SynthStrip + two FLIRT + MMORF + nine-map propagation | 165.33 |
| total internal / full command | 366.41 / 372.28 |

上述完整运行发生在最终显存分块修改之前，AMICO-NODDI 的 PyTorch peak allocation 为 17.06 GB。最终代码把 500 个 LUT direction 按最多 400 个分块；相同输入复测的 peak allocation 为 13.63 GB，`nvidia-smi` 进程占用约 16.5 GiB，wall time 49.19 s。五类输出与未分块结果逐元素完全相同，最大绝对误差 0。计时均来自共享 H100 节点。

端到端 native map 与现有 UKB/FSL 输出使用相同 shape 和 affine。下表同时给出包含任一实现非零体素的 union support，以及双方均非零的 common support；二者差距反映 EDDY 和脑 mask 的支持域差异。

| map | union-support r | common-support r |
|---|---:|---:|
| FA | 0.586157 | 0.876134 |
| MD | 0.547537 | 0.898341 |
| L1 | 0.436994 | 0.886474 |
| L2 | 0.567251 | 0.897319 |
| L3 | 0.634022 | 0.900734 |
| MO | 0.413579 | 0.466099 |
| ICVF | -0.002070 | 0.628843 |
| OD | 0.506862 | 0.757693 |
| ISOVF | 0.608481 | 0.853708 |

### TBSS 时间边界与瓶颈

原始 AP/PA 到全部标准图和 skeleton 图的 FNIT wall time 为 2054.62 s。内部计时 2048.56 s 的分解如下：

| stage | seconds | internal time |
|---|---:|---:|
| TOPUP + EDDY preparation | 95.82 | 4.68% |
| EDDY | 47.49 | 2.32% |
| DTIFIT | 16.58 | 0.81% |
| AMICO-NODDI | 37.63 | 1.84% |
| TBSS registration + nine-map propagation | 1851.05 | 90.36% |

官方 976.88 s 从已经准备好的九张 native maps 开始，不能与 2054.62 s 构成端到端加速比。把两种实现固定到同一组官方 native maps 后，计时边界才一致：

| 实现 | 相同输入和输出范围 | wall time |
|---|---|---:|
| FSL/UKB | weighted FLIRT + three FNIRT stages + nine maps + skeleton | 976.88 s |
| FNIT | weighted TorchFLIRT + TorchFNIRT + nine maps + skeleton | 982.77 s |

同输入时 FNIT 只慢 5.89 s，即 0.60%；因此没有证据支持“PyTorch TBSS 一般慢约两倍”。原始端到端运行变慢，是该次 FNIT native FA 使非线性优化产生更多超出 Jacobian 范围的形变，继而反复进入拓扑修复。

| 运行 | `constrain_topology` 已计时内循环 | ForceJacobianRange calls / inner iterations | 占 TorchTBSS |
|---|---:|---:|---:|
| FNIT raw 端到端输入 | 1187.59 s | 24 / 122 | 64.45% |
| 相同官方 native maps | 169.98 s | 1 / 5 | 17.52% |

raw 运行的第 3–6 层 topology 时间依次为 155.97、221.28、384.44 和 425.90 s；第 3、4 层各重试十次后仍未达到要求的 Jacobian 范围。当前 Triton limiter 为保持 FSL 的原位 `z/y/x/corner` 更新顺序，只启动一个 program 和一个 warp 串行遍历全部 cube。这个实现让 H100 的大部分计算单元空闲。1187.59 s 还没有包含每轮的 B-spline 重拟合和 Jacobian 重算，因此是 topology 开销的下界。

九张真实参数图逐张调用 `TorchApplyWarp` 仅用 2.895 s；堆为一次 4D 调用用 0.990 s，九图逐元素相同，最大绝对误差 0。匹配输入运行的 PCG 迭代反而更多，25,328 次对 raw 运行的 20,343 次，却更快。这两项隔离共同排除了九图传播和 PCG 数量作为约两倍时间差的主因。从迭代结构推断，PCG 是 topology 之外优先 profile 的计算热点：94% 左右的迭代发生在全分辨率层，系数和 B-spline 运算使用 float64，所以 TF32 不会加速这条主路径。当前没有单独的 PCG wall-time profile，因此不把它定量排序为第二大耗时。

### TBSS 不完全一致的原因

以下隔离实验都使用同一真实病例。它们把输出差异定位到 nonlinear warp 的估计，而不是输出格式或 warp 应用：

| 隔离项 | 真实数据结果 | 判断 |
|---|---|---|
| FA preprocessing 与 FLIRT weight | 对 FSL 差异体素 0，最大误差 0 | 已排除 |
| 固定官方 nonlinear residual，只换 TorchFLIRT affine | 九图 common-support r = 0.999671–0.999910 | affine 的直接重采样效应很小；对 nonlinear 优化轨迹的间接影响尚未单独消融 |
| 固定官方 FNIRT warp，TorchApplyWarp 对 FSL applywarp | 九图 r 均大于 0.99999999997；全图最大误差 3.53e-5 | warp 方向、坐标和插值已对齐 |
| 固定官方 warp，再执行 skeleton | mask 差异体素 0；九图 r 均大于 0.999999999985 | skeleton threshold 与相乘已排除 |

固定官方 warp 后，applywarp 与 skeleton 都达到数值容差，因此主要剩余差异位于 TorchFNIRT 的 warp 估计。以下是依据配置、源码和运行 QC 确定的优先核查项；尚未逐项实现后做 one-change ablation，顺序不是各项误差贡献的定量排名：

1. Oxford 三份配置都启用 implicit input/reference zero mask。FSL 在 masked smoothing 和代价函数中使用这些隐式 mask；当前实现没有相同的 zero-mask 与 mask-normalized smoothing。stage 2/3 配置中的 `applyrefmask/applyinmask=1` 只有在命令传入显式 mask 时才生效，本次官方命令没有传入，不能把这两个 dormant flags 当作实测差异来源。真实 moving FA 只有 253,960/778,752 个非零体素，reference 只有 1,489,274/7,221,032 个非零体素，而当前 full-resolution cost mask 达到约 4.00–4.41 million voxels，因此缺少隐式 mask 会改变 SSD、梯度、有效正则权重和优化轨迹。
2. 官方 stage 2/3 使用 `--minmet=scg`；当前六层全部使用 Gauss–Newton/LM + matrix-free PCG。官方三个独立 `fnirt` 进程还通过 float32 `--inwarp` 和文本 `--intin` 交接，当前实现则在一次调用中保持连续 float64 状态，并用 dense expand/refit 完成 10→2 mm 换基。
3. 即使在 stage 1 的 LM 部分，FNIT 的 matrix-free FP64 Hessian/PCG、求和顺序和 LM 接受规则也不同于 FSL 的 assembled sparse Hessian。此前逐更新对照最早在第三次 coefficient update 分叉。
4. topology projection 尚未通过 FSL oracle。raw 运行最终 full-pull Jacobian minimum 为 -0.018944，低于要求的 0.01；投影失败既改变最终 warp，也造成上述额外时间。
5. raw 端到端输入还叠加 TOPUP、EDDY、脑 mask、DTIFIT 和 NODDI 的 native-map 差异。固定九张 native maps 后，standard r 从 0.313–0.754 提高到 0.520–0.880，但仍未达到数值等价，进一步说明主要剩余误差位于 nonlinear registration。

后续修复应先补齐 implicit masks、masked smoothing、三个 stage 的状态交接和 stage 2/3 SCG，再用 FSL `ForceJacobianRange` oracle 逐轮校验 topology。这样同时处理数值轨迹和反复 topology 的速度代价。之后才适合缓存不变的 bending diagonal、减少 PCG 的 CUDA scalar 同步，并评估保持原位顺序的 C++ topology loop。九图 4D 合并只能节省约 1.9 s，不会改变主要瓶颈。

完整去标识 profile、计时边界和隔离指标见 [`tbss_diagnosis.public.json`](../../validation/dmri_pipeline/tbss_diagnosis.public.json)。

| map | 端到端 standard r | 端到端 skeleton r | 同 native standard r | 同 native skeleton r |
|---|---:|---:|---:|---:|
| FA | 0.753871 | 0.694463 | 0.879642 | 0.893747 |
| MD | 0.493614 | 0.406312 | 0.655100 | 0.561907 |
| L1 | 0.457490 | 0.456469 | 0.591632 | 0.653572 |
| L2 | 0.503689 | 0.442469 | 0.670972 | 0.636938 |
| L3 | 0.530033 | 0.433466 | 0.704715 | 0.623788 |
| MO | 0.313156 | 0.494300 | 0.586139 | 0.746085 |
| ICVF | 0.470968 | 0.386101 | 0.520143 | 0.755608 |
| OD | 0.534512 | 0.563077 | 0.666650 | 0.767782 |
| ISOVF | 0.469608 | 0.393052 | 0.712972 | 0.630840 |

“同 native”列让两种实现使用完全相同的九张 native maps，因此主要反映 weighted FLIRT、FNIRT 和 applywarp 的差异。FA 的同输入相关为 0.879642/0.893747，其余图仍有明显差异。最终 full-pull Jacobian minimum 为 -0.01894，没有通过预设 0.01 下界；当前 TorchFNIRT 不应被描述为与 FSL 只差浮点舍入。

DTIFIT 和 AMICO-NODDI 在各自相同输入的独立功能 benchmark 中更接近参考实现；本表从原始数据开始，因此也包含 TOPUP、EDDY 和 mask 差异。TBSS、MMORF 配准对照、九图完整误差和运行记录见[公开报告](../../validation/dmri_pipeline/report.public.json)及[MMORF 报告](../../validation/mmorf/report.public.json)。

下图使用合成 diffusion maps 展示两条分支共享的九文件标准空间命名和显示范围；真实数据指标来自完整 3D NIfTI，不由该图计算。

![Nine-map standard-space output example](example.png)

公开仓库不包含该真实病例的原图或 subject 标识。示意图使用无身份信息的合成输入；真实验证只发布汇总统计。
