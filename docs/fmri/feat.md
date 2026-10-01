# FEAT 核心：运动、掩膜、强度缩放和高通

`run_feat_core` 从一个原始 BIDS run 生成 PICA 前的 `filtered_func_data.nii.gz`。它调用本包 `TorchMCFLIRT` 估计每帧到 SBRef 的刚体运动，使用三次 B 样条重采样 BOLD，求 EPI 均值和脑掩膜，按整段掩膜内第 50 百分位缩放到 10000，再做高斯加权局部直线高通。缺少 SBRef 时以 BOLD 中间帧为参考。没有场图/GDC warp 时直接使用 TorchMCFLIRT 的 Constant 样条和原输出类型转换；提供共享形变时，将运动矩阵与形变组合后只采样一次。场图估计不包含在本函数内。

脑提取默认使用 PyTorch SynthStrip 对重采样后的 EPI 均值求掩膜。若有同网格现成掩膜，传 `brain_mask`；`brain_extraction="otsu"` 保留无需权重的独立 EPI 掩膜算法。默认与官方 FEAT 的 BET/强度阈值掩膜定义不同，因此整链对照须分别记录掩膜差异和后续数值差异。完整 `fMRIVolume_pipeline` 会先对 SBRef（缺失时 BOLD 中间帧）运行 SynthStrip，再把该掩膜通过 `brain_mask` 显式交给本函数；本页独立调用的 `brain_mask=None` 则对运动校正后的 EPI 均值运行 SynthStrip。

## BIDS 输入与调用

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
    batch_size=8,                                # 每批重采样的 BOLD 帧数
    motion_iterations=(1, 1, 1),                 # 8/4/4 mm 各一次 Brent 坐标优化轮回
    overwrite=False,                             # 已存在最终输出时是否覆盖
)
print(feat.filtered_func_data)                   # X×Y×Z×T float32，原生 EPI 网格
print(feat.motion_matrices)                      # 每帧的 4×4 FLIRT 矩阵目录
print(feat.intensity_factor)                     # 整段强度乘数
```

`run_feat_core` 的独立入口目前是上述 Python 函数。运动校正的独立命令行见 [`fnit mcflirt`](../mcflirt/README.md)；完整 BIDS→MNI 体积命令行为 [`fnit-fmri volume`](README.md)，它包含本页 FEAT 核心步骤。

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

## 子函数和官方命令

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

## 当前真实数据对照

2026-10-01，冻结源码 `1eb9c417` 使用同一例真实 88×88×64×490 BOLD 和 SBRef，TR 为 0.735 秒，完整处理所有帧。独立原参照采用 SynthStrip 掩膜，因此这个同步骤对照没有混入默认 BET 掩膜差异。

| 控制范围 | 与原软件比较 | 耗时边界 |
|---|---|---|
| 完整 GPU 运动估计与最终采样 | 逐体素时间 r 均值 0.99957825、中位数 0.99983382；脑内 pull RMS 均值 0.00881 mm；双方 int32 | 完整 FEAT 含运动、缩放、高通及阶段输出为 973.12 s，未独立分离运动函数耗时。 |
| 完整 490 帧 CPU 运动估计 | pull RMS 均值 0.00906 mm、最大逐帧 RMS 0.02603 mm | 387.60 s，包含读取和参数估计，不含最终采样与写盘；原 MCFLIRT 的 326.10 s 包含二者。 |
| 固定原矩阵文本的 8 帧样条采样 | 时间 r 均值 0.999998725、RMSE 0.28163，脑内 93.54% 整数值相同 | CPU 1.49 s，只测采样；原内存矩阵未取得，文本量化也可能影响整数边界。 |
| 固定原运动输出的完整缩放 | 乘数同为 1.4359563469270533，全部解码值相同 | 单步控制，排除运动估计。 |
| 固定原输入的完整高通 | 时间 r 中位数 0.999999999999144；RMSE 0.000505985 | GPU 调用 7.642 s，含数据传输及返回，排除读取与哈希；共享 GPU。 |

运动参数、样条实现和脑图见 [TorchMCFLIRT 专属页](../mcflirt/README.md)及[冻结 GPU 报告](../../validation/mcflirt/full490_gpu.public.json)。固定输入的缩放、高通见[独立控制](../../validation/fmri/matched_highpass_control.public.json)。运动实现保留相邻帧初始化、原 NCC 累加行为、8/4/4 mm 网格、Brent 容差和原整数转换。相对原 FSL，float32 代价差异仍会影响平坦最优点附近的矩阵；本次融合优化则逐值保留冻结 FNIT 的运动输出。

完整 FEAT 与后续去噪、MNI 输出的当前结果见[全流程对照](../../validation/fmri/matched_native.md)。带 GDC/B0 的共享 warp 路径在本轮未做完整对照；其空间方向与采样合同保持原有实现。

## FEAT 耗时来源

优化后的独立 [`TorchMCFLIRT`](../mcflirt/README.md) 在完整 490 帧上用时 **159.27 s**，包含输入解压、估计、样条采样和输出类型转换，不含写盘；其中最终样条采样为 24.83 s。写出的矩阵和参数文本、运动校正 int32 图以及给定同一掩膜后的高通 float32 图，均与冻结 FNIT 结果相同。统计和实际源文件哈希见[当前优化报告](../../validation/mcflirt/gpu_optimization.public.json)。该验证由独立运动函数及固定掩膜的后续子函数组成，其 202.91 s 合计还包含额外私有文件写盘，不是完整 `run_feat_core` 的入口计时。

优化复用 TorchFLIRT 的精确 float32 CUDA 运算，将坐标、八邻点采样、边界降权和参考读取合并到一个 kernel。方向判断在 CPU 完成，每次 cost 只读回最终代价。各帧的质心跨三阶段复用，完整 float32 输入在显存预算允许时缓存。NCC 归约、Brent 搜索、相邻帧初值、样条边界和整数截断沿用原有实现。

以下是优化前的[完整 490 帧分段计时](../../validation/fmri/feat_profile.public.json)，用于说明瓶颈来源。它使用同一真实 BOLD、SBRef 和已有 FNIT EPI 掩膜，输出与冻结版本逐值相同。FEAT 总耗时为 728.28 s，以下项目互不重叠：

| 项目 | 耗时（s） | 包含内容 |
|---|---:|---|
| 运动估计、准备与输出类型转换 | 678.37 | 输入读取、参考网格、逐帧优化、参数转换和 int32 截断；占本次总时间 93.1% |
| 最终运动重采样 | 30.74 | 490 帧样条采样及传回 CPU；占 4.2% |
| 全局强度缩放 | 1.06 | 掩膜内第 50 百分位及全段乘法 |
| 高通 | 1.91 | 时间投影、数据传输和返回 float32 |
| `filtered_func_data.nii.gz` 保存 | 13.51 | NIfTI 写入和 gzip 压缩 |
| 其他准备、掩膜和小文件输出 | 2.69 | 其余 CPU 操作、均值影像和运动文本等 |

旧路径三个阶段各拟合 490 帧，共 1470 次帧/阶段拟合、45972 次 cost。cost 累计 625.08 s，包含在优化器的 652.15 s 内，不能相加。8 mm 和 4 mm 参考网格分别只有 12844 和 102752 个体素，但采样准备由多个小 GPU 运算完成，仅统计归约经 `torch.compile` 编译；每次 cost 至少读回三个方向和最终代价。此次优化减少这部分调度和同步，完整搜索仍调用 45972 次 cost。

最终样条继续逐帧执行空间轴递推、坐标累加和 64 邻点加权，再传回 CPU。motion-only 分支不使用 `batch_size`，调大该参数不会改变这一执行方式；估计和最终采样仍分别读取完整 BOLD。它在优化前 profile 中占 4.2%，本次保持该采样实现。

各次测量使用共享 H100，负载、缓存状态和验证输出集合不同。旧 profile 的 728.28 s、冻结 volume 的 973.12 s 和独立优化计时分别记录，不用相减推算全流程时间。复测参考图还须匹配 `pixdim`：FEAT 保存并重读的 `example_func.nii.gz` 可能与原 SBRef 有微小头信息舍入差异，即使像素和 affine 相同。

## 参考文献与原实现

- Jenkinson et al., *Improved Optimization for the Robust and Accurate Linear Registration and Motion Correction of Brain Images*, NeuroImage 2002，[doi:10.1006/nimg.2002.1132](https://doi.org/10.1006/nimg.2002.1132)。
- [FSL MCFLIRT 使用与算法说明](https://pages.fmrib.ox.ac.uk/docs-881397/registration/mcflirt.html)；[原实现代码库](https://git.fmrib.ox.ac.uk/fsl/mcflirt)。
