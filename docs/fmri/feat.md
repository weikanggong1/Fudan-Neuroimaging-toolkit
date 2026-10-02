# FEAT 核心：运动、掩膜、强度缩放和高通

## 功能简介与流程图

`run_feat_core` 从一个原始 BIDS run 生成 PICA 前的 `filtered_func_data.nii.gz`。它调用本包 `TorchMCFLIRT` 估计每帧到 SBRef 的刚体运动，使用三次 B 样条重采样 BOLD，求 EPI 均值和脑掩膜，按整段掩膜内第 50 百分位缩放到 10000，再做高斯加权局部直线高通。缺少 SBRef 时以 BOLD 中间帧为参考。没有场图/GDC warp 时直接使用 TorchMCFLIRT 的 Constant 样条和原输出类型转换；提供共享形变时，将运动矩阵与形变组合后只采样一次。场图估计不包含在本函数内。

脑提取默认使用 PyTorch SynthStrip 对重采样后的 EPI 均值求掩膜。若有同网格现成掩膜，传 `brain_mask`；`brain_extraction="otsu"` 保留无需权重的独立 EPI 掩膜算法。默认与官方 FEAT 的 BET/强度阈值掩膜定义不同，因此整链对照须分别记录掩膜差异和后续数值差异。完整 `fMRIVolume_pipeline` 会先对 SBRef（缺失时 BOLD 中间帧）运行 SynthStrip，再把该掩膜通过 `brain_mask` 显式交给本函数；本页独立调用的 `brain_mask=None` 则对运动校正后的 EPI 均值运行 SynthStrip。

```mermaid
flowchart LR
    BIDS["单 run BIDS 与参考图"] --> MOTION["TorchMCFLIRT：运动矩阵与单次重采样"]
    MOTION --> MASK["提供的掩膜，或 EPI 均值 SynthStrip"]
    MASK --> SCALE["掩膜内 p50 缩放至 10000"]
    SCALE --> HIGH["高斯局部直线高通，保留均值"]
    HIGH --> OUT["filtered BOLD、均值、掩膜与运动参数"]
```

## Python 调用、输入输出与参数

输入必须包括 `dataset_description.json`、3D T1w、4D BOLD 和 BOLD JSON 的 `TaskName`、`RepetitionTime`。T1w 在此函数只检查，不进入配准；完整流程由 [总入口](README.md)处理。`locate_bids_inputs` 返回 `BIDSInputs.bold`、`t1w_images`、`sbref`、`fieldmaps`、`tr`、`bold_metadata` 和所用 JSON 路径。多 run 的 `session/run/acquisition/direction/reconstruction/echo` 必须指定到唯一一份 BOLD。识别到关联场图但未提供形变时会报错。

```python
from fnit import run_feat_core

feat = run_feat_core(
    bids_root="/absolute/path/bids",              # 原始 BIDS 根目录
    output_dir="/absolute/path/sub-0001.feat",   # FEAT 核心输出目录
    subject="0001",                              # BIDS sub 标签，不含 sub-
    session=None,                                 # ses 标签；无 session 时为 None
    task="rest",                                 # task 标签
    run=None,                                    # run 标签；多个 run 时指定
    acquisition=None,                            # acq 标签；多个候选时指定
    direction=None,                              # dir 标签；多个候选时指定
    reconstruction=None,                         # rec 标签；多个候选时指定
    echo=None,                                   # echo 标签；多个候选时指定
    brain_mask=None,                             # 已对齐 EPI 参考的 3D 掩膜；None 则运行默认脑提取
    brain_extraction="synthstrip",               # 默认 PyTorch SynthStrip；otsu 为无权重备选算法
    synthstrip_weights="/absolute/path/synthstrip.1.pt",  # 脑提取权重；已部署缓存时填 None
    spatial_warp=None,                           # 可选 FSL dense/FNIRT warp；None 时只做运动重采样
    postmat=None,                                # warp 参考空间→最终参考网格的 FLIRT 矩阵
    highpass_cutoff_seconds=100.0,               # 高通截止周期，秒
    device="cuda:0",                             # PyTorch 设备；None 自动选择
    batch_size=8,                                # 带共享 warp 时每批帧数；motion-only 不使用
    motion_iterations=(1, 1, 1),                 # 8/4/4 mm 各一次 Brent 坐标优化轮回
    overwrite=False,                             # 已存在最终输出时是否覆盖
)
print(feat.filtered_func_data)                   # X×Y×Z×T float32，原生 EPI 网格
print(feat.motion_matrices)                      # 每帧的 4×4 FLIRT 矩阵目录
print(feat.intensity_factor)                     # 整段强度乘数
```

| 文件或返回值 | 含义 |
|---|---|
| `example_func.nii.gz` | 3D SBRef，或 BOLD 中间帧；运动参考网格。 |
| `mc/prefiltered_func_data_mcf.mat/MAT_0000` 等 | 逐帧输入→参考的 4×4 FLIRT scaled-mm 矩阵，从 0 编号。 |
| `mc/prefiltered_func_data_mcf.par` | T×6 文本，旋转弧度和平移毫米。 |
| `prefiltered_func_data_unwarp_mean.nii.gz` | 运动与可选形变单次插值后的 3D 均值。无 warp 时也保留该文件名。 |
| `mask.nii.gz` | 同 EPI 参考网格的 3D uint8 掩膜。 |
| `filtered_func_data.nii.gz` | 掩膜、整段强度缩放及高通后的 4D float32 BOLD，保留 TR。 |
| `mean_func.nii.gz` | 上一文件的 3D 时间均值。 |
| `FeatCoreResult.unwarp_applied` | 有无使用 `spatial_warp` 的布尔值；不是自动估计场图的标志。 |

## 命令行调用

`run_feat_core` 的独立入口目前是上述 Python 函数。运动校正的独立命令行见 [`fnit mcflirt`](../mcflirt/README.md)；完整 BIDS→MNI 体积命令行为 [`fnit-fmri volume`](README.md)，它包含本页 FEAT 核心步骤。CLI 的各项参数与完整示例见[体积命令行](README.md#命令行调用)。

## 原软件调用

| FNIT 函数 | 输入、输出 | 相应 FSL 命令 |
|---|---|---|
| `TorchMCFLIRT.run` | 4D BOLD、3D SBRef；返回 T×6 原 `.par` 参数、T×4×4 FLIRT 矩阵及可选校正图。 | `mcflirt -in BOLD -reffile SBREF -out mcf -mats -plots -spline_final`；默认 8/4/4 mm、原 NCC、相邻帧初值、Brent 优化。 |
| `apply_motion_warp` | BOLD、参考、逐帧 FLIRT 矩阵、可选共享 warp/postmat；返回参考网格 4D NIfTI。独立调用默认三线性；带共享 warp 的 FEAT 使用周期边界 CPU 三次 B 样条，无 warp 的 FEAT 直接使用 TorchMCFLIRT 最终采样。 | MCFLIRT `-spline_final`；若同时给定空间 warp，对应 FEAT 的 `applywarp --premat=... --warp=... --postmat=... --interp=spline`。 |
| `epi_brain_mask` | 3D EPI 均值→uint8 掩膜，仅在选择 `otsu` 时使用。 | FEAT 的 BET 加后续强度阈值、时间交集与扩张步骤；算法不同。 |
| `grand_mean_scale` | 4D 数组和同网格掩膜→缩放 4D 数组及乘数。 | `fslstats input -k mask -p 50`，随后 `fslmaths input -mul factor output`。 |
| `gaussian_highpass` | 4D 数组和体积单位 sigma→同形状 4D 数组。 | `fslmaths input -bptf sigma -1 -add tempMean output`。 |
| `scale_nifti`、`highpass_nifti` | 对应路径版；写 NIfTI，返回因子或路径。 | 上述 `fslmaths` 文件命令。 |

以下独立子函数示例以 BIDS run 含 SBRef 为例；若缺少 SBRef，直接调用 `run_feat_core` 会自动使用 BOLD 中间帧。所有 `/absolute/path/` 路径均需换成实际文件。

```python
import nibabel as nib
import numpy as np
from fnit import locate_bids_inputs, TorchMCFLIRT
from fnit.fmri.mask import epi_brain_mask
from fnit.feat.temporal import grand_mean_scale, gaussian_highpass, scale_nifti, highpass_nifti

inputs = locate_bids_inputs(
    bids_root="/absolute/path/bids",    # 原始 BIDS 数据集根目录
    subject="0001",                    # sub 标签，不含 sub-
    session=None,                      # ses 标签；无会话实体用 None
    task="rest",                       # task 标签
    run=None,                          # run 标签；多个 run 时填写具体值
    acquisition=None,                  # acq 标签；有多个候选时填写
    reconstruction=None,               # rec 标签；有多个候选时填写
    direction=None,                    # dir 标签；有多个候选时填写
    echo=None,                         # echo 标签；有多个候选时填写
)
# inputs.bold 是 4D 路径；inputs.t1w_images 是全部 T1w 路径；inputs.sbref 可为 None；
# inputs.fieldmaps 是关联场图路径元组；inputs.tr 为秒；bold_metadata 和 bold_sidecars 给出来源。
assert inputs.sbref is not None, "本段独立子函数示例要求 SBRef；缺失时请调用 run_feat_core"

motion = TorchMCFLIRT(device="cuda:0").run(
    input_bold=inputs.bold,              # 4D BOLD 路径，X×Y×Z×T
    reference=inputs.sbref,              # 同网格 3D SBRef
    stage_iterations=(1, 1, 1),          # 8/4/4 mm 各一次坐标优化轮回
    interpolation="spline",             # 本例匹配原 MCFLIRT -spline_final
    resample=True,                       # 同时返回校正后的 NIfTI
)
aligned = motion.corrected                # 原 uint16 输入会得到 int32 校正图
# motion.parameters 是 T×6 原 .par 参数；motion.matrices 是 T×4×4 FLIRT 矩阵。
mean_epi = nib.Nifti1Image(
    np.asarray(aligned.dataobj).mean(axis=3), aligned.affine
)
epi_mask = epi_brain_mask(
    reference=mean_epi,                 # 运动校正后 3D EPI 时间均值
    dilation=2,                        # 二值脑区向外扩张的体素层数
)
scaled, factor = grand_mean_scale(
    data=np.asarray(aligned.dataobj),  # 4D BOLD 数组，维度 X×Y×Z×T
    mask=np.asarray(epi_mask.dataobj), # 同网格 3D 二值脑掩膜
    target_median=10000.0,             # 掩膜内全部时空值的目标中位强度
)
filtered = gaussian_highpass(
    data=scaled,                        # 已缩放的 4D BOLD 数组
    sigma_volumes=100.0 / (2 * inputs.tr), # 100 秒截止周期换算成体积单位
    device="cuda:0",                   # 计算设备；没有 GPU 时填写 cpu
    voxel_chunk=8192,                  # 一批计算的体素数
    preserve_mean=True,                # 高通后加回各体素原时间均值
)
factor_from_file = scale_nifti(
    input_path="/absolute/path/aligned.nii.gz", # 待缩放 4D NIfTI
    mask_path="/absolute/path/mask.nii.gz",     # 同网格 3D 掩膜
    output_path="/absolute/path/scaled.nii.gz", # 缩放后的 4D 文件
    target_median=10000.0,                     # 目标中位强度
)
filtered_path = highpass_nifti(
    input_path="/absolute/path/scaled.nii.gz",   # 待高通的 4D 文件
    output_path="/absolute/path/filtered.nii.gz", # 高通后的 4D 文件
    cutoff_seconds=100.0,                      # 高通截止周期，秒
    tr_seconds=inputs.tr,                      # TR，秒；None 时读 NIfTI header
    device="cuda:0",                          # 计算设备；没有 GPU 时填写 cpu
    voxel_chunk=8192,                         # 一批计算的体素数
    preserve_mean=True,                       # 是否加回逐体素时间均值
)
```

`MCFLIRTResult.parameters` 是 T×6 原 MCFLIRT 参数，`matrices` 是 T×4×4 输入→参考 FLIRT 矩阵；`corrected` 只在 `resample=True` 时返回。以上独立演示显式使用 Otsu 掩膜；完整 volume 默认使用 SynthStrip。`apply_motion_warp` 与 `epi_brain_mask` 返回 NiBabel 影像；`grand_mean_scale` 返回 4D 数组与乘数，`gaussian_highpass` 返回 4D 数组。

## 最新真实数据精度、耗时与脑图

### 当前完整 volume 的 FEAT 阶段

2026-10-02 公共入口版从同一原始 BIDS 完整处理 **490 帧**，新目录、解剖缓存关闭、额外 WM/CSF/运动回归关闭。在共享 H100、8 CPU 线程上，FNIRT/SynthMorph 完整 volume 中的 FEAT 分别观测 **88.85 / 64.91 s**，包括运动、采样、缩放、高通和阶段保存；SBRef 脑提取另计。对同后端冻结 FNIT 的完整科学输出逐位相同。完整时间、配置和脑图见[volume 实测](README.md#latest-real-benchmark)，运动脑图见 [MCFLIRT](../mcflirt/README.md)。

### 原软件同输入控制

下表绑定 `1eb9c417` 的真实 **88×88×64×490** BOLD/SBRef，TR 0.735 s，原参照使用相同 SynthStrip 掩膜；它检验运动、缩放与高通，不是本次新运行。

| 控制范围 | 对原软件的真实结果 | 原计时范围 |
|---|---|---|
| 完整 GPU 运动估计与采样 | 时间 r mean/median **0.99957825 / 0.99983382**；脑内 pull RMS mean **0.00881 mm**，双方 int32。 | 当时完整 FEAT **973.12 s**，未单独分离运动函数。 |
| 完整 490 帧 CPU 运动估计 | pull RMS mean **0.00906 mm**，max **0.02603 mm**。 | **387.60 s**不含采样/写盘；原 MCFLIRT **326.10 s**包含二者。 |
| 固定原矩阵文本的 8 帧样条 | 时间 r mean **0.999998725**，RMSE **0.28163**；整数值 **93.54%**相同。 | CPU 采样 **1.49 s**；未取得未量化的原内存矩阵。 |
| 固定运动输出的完整强度缩放 | 因子同为 **1.4359563469270533**，全部值相同。 | 独立单步控制。 |
| 固定原输入的完整高通 | 时间 r median **0.999999999999144**，RMSE **0.000505985**。 | GPU **7.642 s**含传输与返回，排除读入/哈希。 |

完整输出和源码见[运动报告](../../validation/mcflirt/full490_gpu.public.json)、[高通控制](../../validation/fmri/matched_highpass_control.public.json)。float32 代价差异仍会影响与 FSL 的平坦最优点；当前矩阵缓存/CUDA graph 保持旧 FNIT 数值，最新配对耗时见[MCFLIRT](../mcflirt/README.md)。

`cfb7beee` 的旧完整 volume 重新提取 EPI 掩膜时有 1 个边界体素差异，因此不能把固定掩膜高通的一致结论写为该独立整链全值一致，原结果见[该轮对照](../../validation/fmri/mcflirt_optimization.md)。公开 `spatial_warp/postmat` 分支仍可单独使用；默认 raw-BIDS volume 不调用它，完整官方同输入验证尚未完成，其周期样条也不能描述为 FSL Constant 的严格复现，见[采样审计](../../validation/fmri/resampling_audit_20261002/README.md)。

## 最近版本与 benchmark 记录

| 版本/日期 | 变化与报告 |
|---|---|
| 2026-10-02 文档整理 | 保留当前参数、原软件误差及共享 warp 的公开功能；历史瓶颈长表移至专属报告。 |
| `81f1bb3` | volume 完整 490 帧公开入口版，FEAT **88.85 / 64.91 s**；完整科学输出对同后端旧 FNIT 相同，见[该轮报告](../../validation/fmri/public_resamplers_20261002/README.md)。 |
| 独立 MCFLIRT 精确缓存/CUDA graph | 不改变 cost 次数或矩阵/运动输出，当前真数据配对见[MCFLIRT 更新](../mcflirt/README.md)。 |
| `cfb7beee` | 当时完整 volume FEAT **177.787 s**、API **707.287 s**，额外混杂回归开启；见[原阶段记录](../../validation/fmri/mcflirt_optimization_api.public.json)。 |
| 2026-10-01 FEAT 覆盖修复 | 长 run 被短 run 覆盖时，拟合成功后清理本函数旧 `MAT_数字`；真实 8→2 帧检查仅保留当前两份矩阵，参数/影像与新目录相同，用户附加文件保留。见[验证](../../validation/fmri/organization_20261001.public.json)。 |

早期 **728.28 s** 的分步骤瓶颈、单独运动和完整 pipeline 计时不混用，完整旧 profile 保留在[原报告](../../validation/fmri/feat_profile.public.json)。motion-only 最终采样逐帧执行，`batch_size` 控制公开共享 warp 路径；调大它不会改变 MCFLIRT 的 motion-only 执行。

## 参考文献与原实现

- Jenkinson et al., *Improved Optimization for the Robust and Accurate Linear Registration and Motion Correction of Brain Images*, NeuroImage 2002，[doi:10.1006/nimg.2002.1132](https://doi.org/10.1006/nimg.2002.1132)。
- [FSL MCFLIRT 使用与算法说明](https://pages.fmrib.ox.ac.uk/docs-881397/registration/mcflirt.html)；[原实现代码库](https://git.fmrib.ox.ac.uk/fsl/mcflirt)。
