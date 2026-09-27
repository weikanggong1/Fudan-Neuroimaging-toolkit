# 静息态 fMRI：BIDS 输入、FEAT 核心与 ICA-AROMA

本模块从一例 BIDS BOLD 序列读取影像，估计逐帧刚体运动，在一次重采样中应用运动矩阵和可选空间形变，随后生成脑掩膜、做整段强度缩放和 100 秒高通滤波。输出沿用 FEAT 在 ICA/FIX 前的主要文件名。ICA-AROMA、WM/CSF/motion 回归是后续独立步骤，不改变前一步的 `filtered_func_data.nii.gz`。

当前实现**尚不能复现 UKB 完整的 FIX 前数值结果**：指定的 ZIP 没有原始 B0 场图及幅度图；按本次要求跳过 GDC；EPI→T1 的 BBR 与 B0 形变估计尚未实现。已有的 FSL warp 可通过 `spatial_warp` 传入，但这不等于从 BIDS 场图自行估计。例中 T1 是已去颅骨的 `T1_brain.nii.gz`，不是原始 T1w。真实数据逐项误差与速度见下文。

## 安装与输入

主页的 `environment.yml` 已包含 PyTorch、NiBabel、NumPy、SciPy；该模块无模型权重。安装后可用 `fnit-fmri --help`。GPU 默认启用 TF32；影像输出是 float32，高通投影内部用 float64 累加，以保留小幅 BOLD 变化。测试峰值显存低于 2 GB。

输入是 BIDS 原始目录，不是 UKB ZIP 本身。至少包含一份 4D `sub-<ID>_task-rest_bold.nii.gz`、可继承的 BOLD JSON（`TaskName`、`RepetitionTime`，单位秒）以及同被试的 3D `sub-<ID>_T1w.nii.gz`。同 run 的 `*_sbref.nii.gz` 可选；缺失时使用 BOLD 中间帧。若有多次采集，传 `run`、`session`、`direction` 等实体，不会任意选第一份。BIDS 场图只在 `IntendedFor` 或 `B0FieldSource` 明确指向该 BOLD 时被识别；识别到了场图但未传已估计的 `spatial_warp`，运行会报错。

```text
bids_root/
├── dataset_description.json
└── sub-0001/
    ├── anat/sub-0001_T1w.nii.gz
    └── func/
        ├── sub-0001_task-rest_bold.nii.gz
        ├── sub-0001_task-rest_bold.json
        └── sub-0001_task-rest_sbref.nii.gz   # 可选
```

`locate_bids_inputs` 返回 `BIDSInputs`：BOLD 绝对路径、同被试 T1w 路径列表 `t1w_images`、SBRef、关联场图的绝对路径，TR、合并后的 BOLD 元数据及所用 JSON 列表。同层存在多张 T1w 时全部返回；当前 FEAT 核心不使用 T1w 做配准。检查 3D/4D 维度、NIfTI 与 JSON 的 TR，以及多文件歧义。输入约束遵循 [BIDS MRI 规范](https://bids-specification.readthedocs.io/en/v1.11.0/modality-specific-files/magnetic-resonance-imaging-data.html)和 [fMRIPrep 的 BIDS 输入要求](https://fmriprep.org/en/latest/usage.html)。

## FEAT 核心：一次调用与文件结构

```python
from fnit import run_feat_core

result = run_feat_core(
    bids_root="/absolute/path/bids",       # BIDS 数据集根目录
    output_dir="/absolute/path/sub-0001.feat",  # 本次运行输出目录
    subject="0001",                       # BIDS 的 sub 标签，不含 sub-
    session=None,                         # ses 标签；没有 session 时用 None
    task="rest",                          # BOLD 文件中的 task 标签
    run=None,                             # run 标签；多 run 时必须明确填写
    acquisition=None,                     # acq 标签；多 acq 时必须明确填写
    direction=None,                       # dir 标签；如 AP/PA 同时存在则明确选择
    reconstruction=None,                  # rec 标签；多重建版本时明确选择
    echo=None,                            # echo 标签；多 echo 时必须明确填写
    brain_mask=None,                      # 可选 3D BOLD 网格脑掩膜；None 时由 EPI 估计
    spatial_warp=None,                    # 可选 FSL dense/FNIRT warp；None 时只做运动重采样
    postmat=None,                         # 可选 warp 参考空间到最终 BOLD 空间的 FLIRT 矩阵
    highpass_cutoff_seconds=100.0,        # UKB 此例的高通截止周期，单位秒
    device=None,                          # PyTorch 设备；None 自动选 CUDA 或 CPU
    batch_size=32,                        # 每批重采样的 3D 帧数
    motion_iterations=(35, 25, 15),      # 8 mm、4 mm、原分辨率三层运动优化步数
    overwrite=False,                     # 已有最终文件时是否允许覆盖
)
print(result.filtered_func_data)          # ICA/FIX 前的 4D BOLD
print(result.intensity_factor)            # 整段强度缩放因子
```

命令行只处理一个 BIDS run：

```bash
fnit-fmri feat --bids-root /absolute/path/bids --subject 0001 \
  --output-dir /absolute/path/sub-0001.feat --device cuda:0
```

| 输出 | 内容 |
|---|---|
| `example_func.nii.gz` | 3D SBRef；缺失时为 BOLD 中间帧，保持原参考网格。 |
| `mc/prefiltered_func_data_mcf.mat/MAT_0000` 等 | 每帧一张 4×4、输入帧→参考帧的 FLIRT scaled-mm 矩阵。编号从 0 开始。 |
| `mc/prefiltered_func_data_mcf.par` | T×6 文本；前三列为 x/y/z 旋转弧度，后三列为参考强度重心处的 x/y/z 平移毫米。 |
| `prefiltered_func_data_unwarp_mean.nii.gz` | 运动及可选 warp 单次重采样后的 3D 时间均值，用于生成掩膜。没有 warp 时文件名仍保留以便定位同一处理节点。 |
| `mask.nii.gz` | 3D uint8 EPI 掩膜；默认使用 Otsu 阈值、最大连通域、填洞与两层扩张。算法不同于 BET。 |
| `filtered_func_data.nii.gz` | 掩膜内整段缩放并高通后的 4D float32 BOLD；高通后加回逐体素时间均值。 |
| `mean_func.nii.gz` | 最终 4D BOLD 的 3D 时间均值。 |

T1w 由 BIDS 入口检查并返回；目前核心函数**不**执行 T1 配准。`spatial_warp` 为已有变换的应用入口，不负责估计 B0 或 GDC。省略 `postmat` 时，warp 的参考网格必须和 `example_func` 网格一致；网格不同需提供 warp 参考空间到最终参考空间的 FLIRT 矩阵。没有 `spatial_warp` 时，`result.unwarp_applied` 为 `False`。

## 独立子函数与原软件命令

| FNIT 函数 | 输入与返回 | 原软件相应步骤 |
|---|---|---|
| `estimate_motion` | 4D BOLD、同网格 3D SBRef、可选 3D mask；返回 T×6 本地参数、T×4×4 FLIRT 矩阵及可选重采样影像。 | `mcflirt -in BOLD -reffile SBREF -out mcf -mats -plots -spline_final`；优化和最终插值并非逐值相同。 |
| `apply_motion_warp` | 每帧一张输入→参考 FLIRT 矩阵；可选共享 warp 和 postmat；返回参考网格 4D NIfTI。只做一次三线性重采样。 | FEAT 用 `applywarp --premat=mc.cat --warp=... --postmat=... --interp=spline`；当前插值核不同。 |
| `epi_brain_mask` | 3D EPI → 同网格 uint8 mask。 | `bet mean_func ...` 及 FEAT 后续阈值、扩张；当前独立实现并非 BET。 |
| `grand_mean_scale` | 4D 数组、3D mask → float32 数组和一个缩放因子；用全部时间点、全部 mask 体素的第 50 百分位。 | `fslstats input -k mask -p 50` 后 `fslmaths input -mul factor`。 |
| `gaussian_highpass` | 4D 数组 → 同形状 4D float32；高斯加权局部直线拟合，`sigma_volumes=cutoff_seconds/(2*TR)`。 | `fslmaths input -bptf sigma -1 -add tempMean output`。 |
| `scale_nifti`、`highpass_nifti` | 对应的路径输入、NIfTI 文件输出；后者读取 header TR 或显式 `tr_seconds`。 | 上述 `fslmaths` 文件命令。 |

### FEAT 子函数的具名调用

以下路径分别指同一 run 的原始 BOLD、SBRef 和已对齐的掩膜。若无现成掩膜，可从重采样后的 EPI 时间均值调用 `epi_brain_mask`。代码中独立显示每个参数；`device=None` 自动选 GPU/CPU，文件路径均须由调用方替换。

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

`MotionResult.parameters` 为 T×6 本地刚体参数；`fsl_matrices` 为 T×4×4 输入→参考矩阵；`corrected` 仅在 `resample=True` 时为 4D NIfTI；`reference` 为 3D NIfTI。`apply_motion_warp` 与 `epi_brain_mask` 各返回一个 NiBabel NIfTI 对象。`grand_mean_scale` 返回 4D float32 数组和缩放因子；`gaussian_highpass` 返回 4D float32 数组。文件包装器分别返回缩放因子和输出路径。`mean_epi` 为调用者临时构造的 3D 数组，并非额外的包输出。`inputs.sbref=None` 时可改用 BOLD 中间帧作为 3D `reference`，与高层 `run_feat_core` 一致。

这例 UKB 的真实 FEAT 日志依次运行 MCFLIRT、运动与 B0 的合并重采样、mask、`-mul 1.39695737495`、`-bptf 68.0272108844 -1 -add tempMean`、MELODIC；不进行 slice timing、空间平滑或低通。`68.0272108844 = 100/(2×0.735)`。完整设置以每例 `design.fsf` 为准；通用 FEAT 处理顺序可参见 [FEAT 用户指南](https://fsl.fmrib.ox.ac.uk/fsl/docs/task_fmri/feat/user_guide.html)、[MCFLIRT 说明](https://fsl.fmrib.ox.ac.uk/fsl/docs/registration/mcflirt.html)和 [FEAT 输出格式](https://fsl.fmrib.ox.ac.uk/fsl/docs/task_fmri/feat/feat_output.html)。

## ICA-AROMA 与可选混杂回归

ICA-AROMA 阶段读取已完成的 `filtered_func_data.nii.gz`、同一次运动参数和同网格的 CSF、脑边缘、脑外掩膜。BIDS 原始影像本身不含后三张掩膜；调用方需提供与 BOLD 网格已对齐的文件。`n_components` 必须显式填写；FNIT 使用 PCA + spatial FastICA，输出的阈值图以 `|Z|` 截断，**不是** MELODIC 混合模型图。因此官方分类阈值被复现，但不能把自产 IC 身份或最终降噪结果称为与官方 ICA-AROMA 完全一致。

```python
from fnit import run_aroma_pipeline

aroma = run_aroma_pipeline(
    filtered_func_data="/absolute/path/sub-0001.feat/filtered_func_data.nii.gz",  # FIX 前 4D BOLD
    brain_mask="/absolute/path/sub-0001.feat/mask.nii.gz",                      # BOLD 网格脑掩膜
    motion_parameters="/absolute/path/sub-0001.feat/mc/prefiltered_func_data_mcf.par",  # T×6 运动参数
    csf_mask="/absolute/path/csf_in_bold.nii.gz",             # BOLD 网格 CSF 掩膜
    edge_mask="/absolute/path/edge_in_bold.nii.gz",           # BOLD 网格脑边缘掩膜
    outside_mask="/absolute/path/outside_in_bold.nii.gz",     # BOLD 网格脑外掩膜
    output_dir="/absolute/path/sub-0001.aroma",              # ICA 与降噪输出目录
    n_components=106,                    # 要估计的 IC 数；不自动仿造 MELODIC -d 0
    tr=0.735,                            # 重复时间，秒；None 则读取 NIfTI header
    mode="nonaggr",                      # 非强制降噪；也可选 aggr
    device="cuda:0",                     # PyTorch 设备
    n_splits=1000,                        # 运动相关特征的 90% 帧重复抽样次数
    random_state=0,                       # ICA 初始化与抽样随机种子
    ica_max_iter=500,                     # FastICA 最大迭代次数
    wm_mask="/absolute/path/wm_in_bold.nii.gz",               # 可选 BOLD 网格 WM 掩膜
    regress_csf=True,                     # AROMA 后是否再回归 CSF 均值信号
    regress_motion=True,                  # AROMA 后是否再回归运动信号
    motion_model=24,                      # 6、12 或 24 个运动回归项
    bandpass=(0.01, 0.1),                 # 可选 Hz 带通；None 为不做带通
    global_signal=False,                  # 是否额外回归全脑均值
)
print(aroma.denoised_bold)                # ICA-AROMA 输出
print(aroma.confounds_cleaned_bold)      # 额外 WM/CSF/motion 回归输出
```

高层返回 `AromaResult`：`ica` 是上述 `ICAResult`；`features`、`noise_components`、`denoised_bold`、`confounds_cleaned_bold` 是对应文件路径，最后一项在未启用额外回归时为 `None`。

| 输出 | 内容 |
|---|---|
| `ica/ica_components_z.nii.gz` | X×Y×Z×K 的原生空间 IC Z 图。 |
| `ica/ica_components_abs_z_thresholded.nii.gz` | 同形状 `|Z|≥z_threshold` 的近似阈值图；未通过 MELODIC 混合模型。 |
| `ica/ica_mixing.tsv`、`ica/ica_frequency_power.tsv` | T×K mixing 与 ⌊T/2⌋×K 频谱功率。 |
| `aroma_features.tsv` | K 行；从 1 开始的 IC 编号及四项分类特征。 |
| `aroma_noise_components.txt` | 每行一个从 1 开始的噪声 IC 编号。 |
| `filtered_func_data_aroma.nii.gz` | AROMA 回归后的 X×Y×Z×T float32 BOLD。 |
| `filtered_func_data_aroma_confounds.nii.gz` | 可选 WM/CSF/motion/带通联合投影后的同形状 float32 BOLD；未启用时不生成。 |

### ICA、分类和混杂回归的独立调用

三张分类掩膜需与阈值 IC 图同网格；用官方 MELODIC 的 IC 图核对分类器时，应使用官方 MNI 2 mm 掩膜。下例自产 IC 位于 BOLD 原生空间，只能配原生空间掩膜，不能据此宣称与官方成分身份相同。

```python
from fnit import decompose_spatial_ica, classify_aroma, denoise_aroma, clean_confounds, motion_regressors

ica = decompose_spatial_ica(
    input_bold="/absolute/path/filtered_func_data.nii.gz", # ICA 前 4D BOLD
    brain_mask="/absolute/path/mask.nii.gz",               # 同网格 3D 脑掩膜
    output_dir="/absolute/path/ica",                       # IC 图、mixing、频谱输出目录
    n_components=106,                                      # 显式 IC 个数；需小于时间点数
    device=None,                                           # None 自动选 GPU/CPU
    voxel_batch_size=8192,                                 # ICA 方差估计的体素分块
    max_iter=500,                                           # ICA 最大迭代次数
    tolerance=1e-3,                                        # ICA 相邻迭代的收敛阈值
    random_state=0,                                        # ICA 初始化随机种子
    z_threshold=2.3,                                       # 原生 |Z| 阈值，非 MELODIC 混合模型
)
features = classify_aroma(
    thresholded_ic_maps=ica.thresholded_maps,              # X×Y×Z×K 阈值 IC 图
    mixing=ica.mixing,                                     # T×K 成分时间序列
    ftmix=ica.frequency_power,                             # ⌊T/2⌋×K 非负频谱功率
    motion="/absolute/path/mc/prefiltered_func_data_mcf.par", # T×6 运动参数
    csf_mask="/absolute/path/csf_in_bold.nii.gz",         # IC 图网格的 CSF 掩膜
    edge_mask="/absolute/path/edge_in_bold.nii.gz",       # IC 图网格的脑边缘掩膜
    outside_mask="/absolute/path/outside_in_bold.nii.gz", # IC 图网格的脑外掩膜
    tr=0.735,                                              # 重复时间，秒
    n_splits=1000,                                         # 90% 帧抽样次数
    random_state=0,                                        # 抽样随机种子
)
denoised_path = denoise_aroma(
    input_bold="/absolute/path/filtered_func_data.nii.gz", # 待降噪的 4D BOLD
    mixing=ica.mixing,                                     # 同一次 ICA 的 T×K mixing
    noise_indices=features["noise_indices"],              # 从 0 开始的噪声 IC 索引
    output_bold="/absolute/path/filtered_func_data_aroma.nii.gz", # 降噪输出路径
    mode="nonaggr",                                        # nonaggr 部分回归；aggr 全回归
    device=None,                                           # None 自动选 GPU/CPU
    chunk_size=4096,                                       # 一批回归的体素数
)
motion24 = motion_regressors(
    motion="/absolute/path/mc/prefiltered_func_data_mcf.par", # T×6 运动参数
    model=24,                                              # 返回 6、12 或 24 列
)
cleaned_path = clean_confounds(
    input_bold=denoised_path,                              # AROMA 后的 4D BOLD
    output_bold="/absolute/path/filtered_func_data_aroma_confounds.nii.gz", # 输出文件
    wm_mask="/absolute/path/wm_in_bold.nii.gz",            # 同网格 WM 掩膜；None 不回归
    csf_mask="/absolute/path/csf_in_bold.nii.gz",          # 同网格 CSF 掩膜；None 不回归
    brain_mask="/absolute/path/mask.nii.gz",              # 全脑掩膜；全脑回归时必需
    motion="/absolute/path/mc/prefiltered_func_data_mcf.par", # T×6 运动参数；None 不回归
    motion_model=24,                                       # 6、12 或 24 项运动模型
    bandpass=(0.01, 0.1),                                  # Hz；None 则不带通
    tr=0.735,                                              # TR，秒；None 时读 NIfTI header
    global_signal=False,                                   # 是否加入全脑均值回归
    device=None,                                           # None 自动选 GPU/CPU
    chunk_size=4096,                                       # 一批投影的体素数
)
```

`ICAResult` 包含原始 Z 图 `component_maps`、阈值图 `thresholded_maps`、T×K 的 `mixing` 路径、⌊T/2⌋×K 的 `frequency_power` 路径，以及成分数、有效体素数、迭代数、是否收敛、最终正交化变化量、PCA 解释方差比例。`classify_aroma` 返回四组每 IC 特征和 `noise_indices`，均为数组，索引从 0 开始。`denoise_aroma`、`clean_confounds` 返回 4D NIfTI 文件路径；`motion_regressors` 返回 T×6/12/24 的 NumPy 数组。`clean_confounds` 的 WM/CSF/全脑掩膜仅用于提取均值信号，并不自动生成组织分割。

用于原软件对照的命令如下；它们只在私有基准中运行，不由 FNIT 调用。`-d 106` 与示例 `n_components=106` 对应，官方 UKB 实际运行使用 `-d 0` 自动定阶。

```bash
melodic -i filtered_func_data.nii.gz -o melodic_ref \
  --nobet --bgthreshold=3 --tr=0.735 -d 106 --Ostats --mmthresh=0.5
python ICA_AROMA.py -in filtered_func_data.nii.gz -out aroma_ref \
  -mc mc/prefiltered_func_data_mcf.par -affmat reg/example_func2highres.mat \
  -warp reg/highres2standard_warp.nii.gz -mni MNI152_T1_2mm_brain.nii.gz -den nonaggr
fsl_regfilt -i filtered_func_data.nii.gz -d melodic_ref/melodic_mix \
  -f 2,5,9 -o filtered_func_data_aroma_ref.nii.gz
3dTproject -input filtered_func_data_aroma_ref.nii.gz \
  -ort confounds.1D -polort 2 -passband 0.01 0.1 \
  -prefix filtered_func_data_aroma_confounds_ref.nii.gz
```

`fsl_regfilt -f` 使用从 1 开始的 IC 编号；`confounds.1D` 是对应 WM/CSF/motion 列，不由这四条命令自动生成。

`decompose_spatial_ica` 单独返回 `ica_components_z.nii.gz`（X×Y×Z×K）、`ica_components_abs_z_thresholded.nii.gz`、`ica_mixing.tsv`（T×K）、`ica_frequency_power.tsv`（⌊T/2⌋×K）及收敛标志。`classify_aroma` 接受同一次 ICA 的阈值图、mixing、频谱、运动 T×6 和三张同网格掩膜；返回最大运动相关、边缘比例、高频比例、CSF 比例和**从 0 开始**的噪声 IC 索引。高层输出 `aroma_noise_components.txt` 按官方阅读习惯改为**从 1 开始**的编号。`denoise_aroma` 的 `nonaggr` 以全部 IC 拟合，只减去噪声 IC 的部分贡献；`aggr` 只拟合噪声 IC。

`clean_confounds` 可独立调用。WM、CSF 和可选全脑信号取 mask 内均值。6 项运动模型只用当前帧；12 项增加当前帧的一阶差分；24 项用当前、前一帧及双方平方。带通与回归联合投影，默认去除常数和二阶趋势，输出残差均值为 0。原 MATLAB 脚本的第一个输出没有全脑回归，第二个输出实际加入全脑回归；FNIT 用明确的 `global_signal` 参数区分。参考 [原 MATLAB 函数](https://github.com/weikanggong/Resting-state-fMRI-preprocessing/blob/master/g_regressWmCsf_and_filter.m)、[AFNI 3dTproject](https://afni.nimh.nih.gov/pub/dist/doc/program_help/3dTproject.html)和 [ICA-AROMA 源码](https://github.com/maartenmennes/ICA-AROMA/blob/master/ICA_AROMA_functions.py)。

## 真实数据对照

全部测试在 gpucw1 的同一例 UKB BOLD（88×88×64×490，TR 0.735 秒）完成，输入与输出只保存在服务器私有目录；没有把影像或逐体素值放入仓库。FSL 命令仅用作独立基准，FNIT 函数运行时不调用 FSL/AFNI。

| 对照项目 | FNIT | 原软件 | 数值检查 |
|---|---:|---:|---|
| 490 帧运动估计，同一 BOLD/SBRef；FNIT 另用官方 mask 限定优化区域 | 21.64 秒；峰值 GPU 1.04 GB | MCFLIRT 397.54 秒 | 逐帧 FLIRT 矩阵差：平移中位 0.328 mm、95% 位 0.620 mm、最大 0.733 mm；旋转中位 0.208°。不能称数值等价。 |
| 固定官方 MCFLIRT 的 8 帧矩阵，原始 BOLD→SBRef 三线性重采样 | 含文件保存中位 0.948 秒 | 8 次 FSL `applywarp --premat` 中位总 2.148 秒 | 全体素 MAE 0.1066、RMSE 0.1436、r=0.9999999990、最大差 26.66 强度单位（边界）。同输入同矩阵，仅比较重采样。 |
| 脑掩膜，FNIT 未掩膜的运动校正 3D 均值 | Otsu＋扩张 0.0468 秒；107902 体素 | 同输入 BET `-m -n -f 0.05` 0.900 秒 | 对 ZIP 官方 mask 的 Dice：FNIT 0.9482、BET 0.9578。官方 mask 来自含 B0 校正的另一输入，Dice 不能解释为同输入算法的逐值差。 |
| 全局均值缩放，真实原始 BOLD 的 8 帧与同一官方 mask | 含压缩保存中位 0.353 秒 | `fslstats -k -p 50`＋`fslmaths -mul` 1.163 秒 | 因子 1.5420054353 vs 1.5420054173；4D float32 输出逐体素完全一致。 |
| 高通，真实 BOLD 的 16³×490 裁剪块，同一输入 | CUDA 1.02 秒；峰值 GPU 0.052 GB | `fslmaths -bptf 68.0272108844 -1` 2.04 秒 | MAE 2.47×10⁻⁶、RMSE 7.06×10⁻⁶、最大误差 2.44×10⁻⁴；比较时 FNIT 关闭“加回原均值”以与单条 FSL 命令一致。 |
| BIDS 输入到 `filtered_func_data.nii.gz`，无场图/GDC | 65.19 秒；峰值 GPU 1.97 GB | 此输入缺少原始场图，不能同配置重跑 UKB FEAT | 结果 88×88×64×490、TR 0.735 秒、全部有限；默认 mask 对官方 mask Dice 0.948。其 `mean_func` 在两张 mask 交集内对官方 r=0.936、MAE=1374.5 强度单位，尚未达到整链等价。 |
| ICA 分解，真实原始 BOLD 490 帧、K=106、官方掩膜固定 | CUDA 25.71 秒；峰值分配 0.301 GB；57 次迭代收敛 | MELODIC 核心文件在 127.20 秒时全部生成、RSS 峰值约 1.92 GB，但进程退出 255，不能当作成功的整例计时 | PCA 解释方差 0.830172 vs 0.830406；匈牙利匹配并消除符号后，IC 时间序列相关中位 0.9458，空间图相关中位 0.8930。组件并未完全一致；官方时间仅是核心输出检查点。 |
| ICA-AROMA 运动相关特征，真实 490×106 MELODIC mixing | 6.28 秒 | 官方函数 6.17 秒 | 同随机种子、1000 次 90% 帧抽样，MAE 4.67×10⁻¹⁷；高频比例逐项一致。 |
| ICA-AROMA 空间特征与分类，同一份官方 106 个阈值 IC 图、已对齐 MNI 2 mm 掩膜 | 空间特征 2.33 秒 | 官方空间特征 201.02 秒 | edge fraction MAE 3.51×10⁻⁷，CSF fraction MAE 2.32×10⁻⁸；106/106 个 IC 的噪声判定一致。此项固定了官方 ICA 输入，不代表自产 ICA 图等价。 |
| ICA-AROMA 非积极与积极回归，真实 BOLD 20³×490 裁剪、官方 106 列 mixing、测试用第 1–3 个噪声 IC | CUDA 含 I/O 分别 2.83 / 1.27 秒 | FSL `fsl_regfilt` 分别 2.02 / 1.85 秒 | 两种模式输出均逐体素 float32 完全一致。测试索引并非真实分类结果。 |
| 完整 `run_aroma_pipeline`，同一例 FNIT FEAT 输出 88×88×64×490，K=106，启用 WM/CSF/motion 24 项及 0.01–0.1 Hz 带通 | CUDA 102.39 秒；峰值 PyTorch 分配显存 0.280 GB；ICA 在 100 次迭代收敛 | 无同输入官方整链对照 | 两份 4D float32 输出均可读取、全部有限、TR 0.735 秒。四张组织/边缘掩膜为基于强度与形态的计算测试掩膜，并非解剖分割；该运行只验收调用和输出结构，不评价真实噪声分类。 |
| 可选 WM/CSF/motion 与带通联合投影，真实 BOLD 的 8000 体素×490 帧裁剪 | CUDA 含 I/O 1.83 秒；峰值 GPU 0.144 GB | 同输入 NumPy float64 独立投影计算 0.095 秒，不含 I/O | MAE 1.26×10⁻⁶，最大误差 1.10×10⁻⁴。WM/CSF 为计算测试掩膜，并非组织标注；gpucw1 无 AFNI，尚无 `3dTproject` 实测对照。 |

本机部分 FSL 启动器返回非零状态，但输出完整可读；高通子进程返回 0、两次输出 SHA-256 相同。缩放与重采样按实际输出比较，仍保留退出状态异常。MELODIC 核心进程也返回 255，因此不把该时间称为成功的完整运行。表中计时为完整命令墙钟或注明的核心文件检查点。MCFLIRT 不接受同一张 mask 作为优化输入；运动估计的加速不包括相同插值核下的逐体素等价；FEAT 完整计时与官方 `filtered_func_data.nii.gz` 在此 ZIP 中均不可得。
