# FEAT 核心：运动、掩膜、强度缩放和高通

`run_feat_core` 从一个原始 BIDS run 生成 PICA/FIX 前的 `filtered_func_data.nii.gz`。它估计每帧到 SBRef 的刚体运动，使用三次 B 样条在一次插值中重采样 BOLD，求 EPI 均值和脑掩膜，按整段掩膜内第 50 百分位缩放到 10000，再做高斯加权局部直线高通。缺少 SBRef 时以 BOLD 中间帧为参考。没有场图/GDC warp 时只做运动重采样；不会自行制造畸变场。

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
    motion_iterations=(35, 25, 15),              # 运动配准三级迭代次数
    overwrite=False,                             # 已存在最终输出时是否覆盖
)
print(feat.filtered_func_data)                   # X×Y×Z×T float32，原生 EPI 网格
print(feat.motion_matrices)                      # 每帧的 4×4 FLIRT 矩阵目录
print(feat.intensity_factor)                     # 整段强度乘数
```

同一函数的命令行：

```bash
fnit-fmri feat --bids-root /absolute/path/bids --subject 0001 \
  --output-dir /absolute/path/sub-0001.feat --device cuda:0
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

## 子函数和官方命令

| FNIT 函数 | 输入、输出 | 相应 FSL 命令 |
|---|---|---|
| `estimate_motion` | 4D BOLD、3D SBRef、可选 mask；返回 T×6 参数和 T×4×4 FLIRT 矩阵。 | `mcflirt -in BOLD -reffile SBREF -out mcf -mats -plots -spline_final`。优化轨迹和最终插值核不同。 |
| `apply_motion_warp` | BOLD、参考、逐帧 FLIRT 矩阵、可选共享 warp/postmat；返回参考网格 4D NIfTI。独立调用默认三线性，`interpolation="spline"` 用 CPU 三次 B 样条；FEAT 核心默认选样条。 | MCFLIRT `-spline_final`；若同时给定空间 warp，对应 FEAT 的 `applywarp --premat=... --warp=... --postmat=... --interp=spline`。 |
| `epi_brain_mask` | 3D EPI 均值→uint8 掩膜，仅在选择 `otsu` 时使用。 | FEAT 的 BET 加后续强度阈值、时间交集与扩张步骤；算法不同。 |
| `grand_mean_scale` | 4D 数组和同网格掩膜→缩放 4D 数组及乘数。 | `fslstats input -k mask -p 50`，随后 `fslmaths input -mul factor output`。 |
| `gaussian_highpass` | 4D 数组和体积单位 sigma→同形状 4D 数组。 | `fslmaths input -bptf sigma -1 -add tempMean output`。 |
| `scale_nifti`、`highpass_nifti` | 对应路径版；写 NIfTI，返回因子或路径。 | 上述 `fslmaths` 文件命令。 |

以下独立子函数示例以 BIDS run 含 SBRef 为例；若缺少 SBRef，直接调用 `run_feat_core` 会自动使用 BOLD 中间帧。所有 `/absolute/path/` 路径均需换成实际文件。

```python
import nibabel as nib
import numpy as np
from fnit import locate_bids_inputs
from fnit.fmri.motion import estimate_motion
from fnit.fmri.spatial import apply_motion_warp
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

motion = estimate_motion(
    input_bold=inputs.bold,             # 4D 原始 BOLD 路径或 NiBabel 对象
    reference=inputs.sbref,             # 同网格 3D SBRef 路径或 NiBabel 对象
    mask=None,                          # 可选同网格 3D 优化掩膜；None 自动估计
    device=None,                        # None 自动选 GPU/CPU，也可指定 cuda:0 或 cpu
    batch_size=16,                      # 一次估计的 BOLD 帧数
    iterations=(35, 25, 15),           # 8 mm、4 mm、原分辨率三层迭代次数
    resample=False,                    # 是否同时返回已重采样的 BOLD
)
aligned = apply_motion_warp(
    input_bold=inputs.bold,             # 待重采样的 4D BOLD
    reference=inputs.sbref,             # 最终 3D 参考网格
    motion_matrices=motion.fsl_matrices, # 每帧输入→参考的 T×4×4 FLIRT 矩阵
    warp=None,                          # 可选共享空间形变；None 只做运动校正
    postmat=None,                       # 可选 warp 参考→最终参考的 4×4 FLIRT 矩阵
    warp_convention="auto",             # dense warp 位移方向；按 FSL intent 自动识别
    interpolation="spline",             # 三次 B 样条，CPU 实现；linear 使用 GPU 三线性
    batch_size=16,                      # 一批重采样的帧数
    device=None,                        # None 自动选 GPU/CPU
)
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

`MotionResult.parameters` 是 T×6 本地刚体参数，`fsl_matrices` 是 T×4×4 输入→参考 FLIRT 矩阵；`corrected` 只在 `resample=True` 时返回。`apply_motion_warp` 与 `epi_brain_mask` 返回 NiBabel 影像；`grand_mean_scale` 返回 4D 数组与乘数，`gaussian_highpass` 返回 4D 数组。

## 真实数据精度与耗时

下列子函数对照使用同一例 88×88×64×490、TR 0.735 秒的 UKB BOLD/SBRef。官方 FSL 只用于独立基准；FNIT 运行时不调用 FSL。除注明的 8 帧或裁剪块外，均用真实完整影像。

| 对照 | FNIT | FSL | 精度和范围 |
|---|---:|---:|---|
| 490 帧运动估计；FNIT 优化区使用官方 mask | 仅拟合 21.64 秒，GPU 峰值 1.04 GB | MCFLIRT 完整命令 397.54 秒 | 计时范围不同；逐帧平移差中位 0.328 mm、95% 位 0.620 mm；旋转差中位 0.208°。 |
| 固定官方 8 帧 MCFLIRT 矩阵，原始 BOLD→SBRef 三线性插值（独立函数 `linear` 选项） | 含保存中位 0.948 秒 | 8 次 `applywarp --premat` 合计中位 2.148 秒 | 全体素 MAE 0.1066、RMSE 0.1436、r=0.9999999990；最大差 26.66，位于边界。 |
| 同一真实 BOLD 的 8 帧和同一 mask，整段中位数缩放 | 含压缩保存中位 0.353 秒 | `fslstats`＋`fslmaths` 1.163 秒 | 乘数 1.5420054353 vs 1.5420054173；4D float32 输出逐体素相同。 |
| 真实 BOLD 的 16³×490 裁剪块高通 | CUDA 1.02 秒，峰值 0.052 GB | `fslmaths -bptf 68.0272108844 -1` 2.04 秒 | MAE 2.47×10⁻⁶，RMSE 7.06×10⁻⁶，最大误差 2.44×10⁻⁴；比较时 FNIT 关闭加回均值以匹配单条命令。 |

固定官方运动矩阵的另一次真实 8 帧测试中，`interpolation="spline"` 的 CPU 样条计算为 1.548 秒，相对 FSL `-spline_final` 的脑区 MAE 2.776、RMSE 30.18；三线性 MAE 约 100。FNIT 自估矩阵加样条对官方结果的 MAE 仍为 83.63，说明运动矩阵仍是剩余误差来源。该 8 帧耗时只测重采样，不能与含优化的整次 MCFLIRT 时间直接比较。

整例 FEAT 核心与跳过 GDC/B0 的官方流程对照见 [验证页](../../validation/fmri/README.md)。官方 FEAT 掩膜的实际步骤和本函数默认 SynthStrip 的差异须一起解释，不把上述单步接近误写为整链逐体素等价。

## 运动估计与 MCFLIRT 的差异

上述 490 帧比较的官方命令是 `mcflirt -in BOLD.nii.gz -reffile SBREF.nii.gz -out prefiltered_func_data_mcf -mats -plots -spline_final`。`-mats` 输出每帧到 SBRef 的 FSL scaled-mm 矩阵，`-plots` 输出每帧六列旋转和平移参数，`-spline_final` 指定最终影像重采样。FNIT 的 `estimate_motion(input_bold=..., reference=..., mask=..., resample=False)` 只估计矩阵，没有在 21.64 秒内重采样和写出 4D 图像；FSL 的 397.54 秒包含这些工作。因此这两个时间不能当作同范围加速比。

两者的搜索过程也不同。MCFLIRT 默认代价为 `normcorr`，先在 8 mm 优化，再在 4 mm 优化两次，并使用相邻时间帧的结果作为后续帧初值。FNIT 用带掩膜的标准化相关和 Adam，按 8 mm、4 mm、原分辨率三级优化，每帧从零初值独立开始。本次 FNIT 测试还显式提供了官方脑掩膜，而对应 MCFLIRT 命令没有掩膜输入。矩阵坐标和文件结构能互通，但这不是 MCFLIRT 求解器的源码级复现。

同一例真实 490 帧的相对矩阵，平移差中位数／95% 位为 0.328／0.620 mm，旋转差为 0.208／0.287°。详见[逐项记录](../../validation/fmri/mcflirt_difference.public.json)。本次 FLIRT 只修改角度采样和粗网格插值；`motion.py` 的 SHA-256 与该 490 帧测试时的源码相同，因此 FLIRT 的修改不会改变此处的 MCFLIRT 结果。

## 参考文献与原实现

- Jenkinson et al., *Improved Optimization for the Robust and Accurate Linear Registration and Motion Correction of Brain Images*, NeuroImage 2002，[doi:10.1006/nimg.2002.1132](https://doi.org/10.1006/nimg.2002.1132)。
- [FSL MCFLIRT 使用与算法说明](https://pages.fmrib.ox.ac.uk/docs-881397/registration/mcflirt.html)；[原实现代码库](https://git.fmrib.ox.ac.uk/fsl/mcflirt)。
