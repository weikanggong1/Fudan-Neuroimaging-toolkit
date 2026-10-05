# fMRI 时间处理与采样参考网格的 CPU 官方对照

## 1. 功能与处理流程

本页比较三个可独立调用的功能：FEAT 全运行强度缩放和 Gaussian 高通、可选 Fourier slice timing，以及按原生 BOLD 分辨率生成 T1w 采样参考网格。影像读写使用 nibabel；高通与 slice timing 使用 PyTorch。参考网格沿用已有 nibabel/NumPy 几何实现。FNIT 运行时不调用参考软件。

```mermaid
flowchart LR
    A[完整 BOLD 与同网格脑掩膜] --> B[全运行中位数缩放]
    B --> C[Gaussian 高通与可选均值保留]
    D[原始 BOLD 与 SliceTiming] --> E[显式开启 Fourier slice timing]
    F[T1w 与同网格 FOV 掩膜] --> G[BOLD 分辨率的 RAS 采样参考]
    H[BOLD 体素尺寸] --> G
    classDef default fill:#fff,stroke:#000,color:#000;
    linkStyle default stroke:#000;
```

完整 volume pipeline 默认关闭 slice timing。本页开启它来核对可选功能；缩放、高通和参考网格分别使用固定中间输入，避免把上游配准差异混入子函数精度。

## 2. Python 调用、输入与输出

```python
import json
from pathlib import Path
from fnit.feat.temporal import scale_nifti, highpass_nifti
from fnit.fmri.slice_timing import slice_timing_correct
from fnit.fmri.sampling_reference import native_bold_sampling_reference

motion_corrected_bold_path = Path("/absolute/path/motion_corrected.nii.gz")
brain_mask_path = Path("/absolute/path/brain_mask.nii.gz")
scaled_bold_path = Path("/absolute/path/output/scaled.nii.gz")
filtered_bold_path = Path("/absolute/path/output/highpass.nii.gz")

intensity_factor = scale_nifti(
    input_path=motion_corrected_bold_path,  # 完整 X×Y×Z×T 时序
    mask_path=brain_mask_path,             # 同网格 3D 脑掩膜
    output_path=scaled_bold_path,
    target_median=10000.0,                 # 全部掩膜内时点合并后的第 50 百分位目标
)
highpass_nifti(
    input_path=scaled_bold_path,
    output_path=filtered_bold_path,
    cutoff_seconds=100.0,                 # FSL sigma = cutoff / (2×TR)
    tr_seconds=0.735,                     # 秒；None 从 NIfTI 及时间单位读取
    device="cpu",                        # cpu 或 cuda:0
    voxel_chunk=8192,                     # 每批空间体素数
    preserve_mean=True,                   # True 恢复原均值；False 输出去均值残差
)

raw_bold_path = Path("/absolute/path/raw_bold.nii.gz")
bold_metadata_path = Path("/absolute/path/raw_bold.json")
bold_metadata = json.loads(bold_metadata_path.read_text())
slice_corrected_path, slice_timing_details = slice_timing_correct(
    source=raw_bold_path,
    output="/absolute/path/output/stc.nii.gz",
    metadata=bold_metadata,               # RepetitionTime、SliceTiming、SliceEncodingDirection
    reference_fraction=0.5,              # 切片最早到最晚采集范围内的相对位置，0..1
    ignore=0,                            # 初始帧数；这些帧保持原值且不参与校正
    device="cpu",
    voxel_batch_size=4096,               # 每批处理体素数
)

sampling_reference_path = native_bold_sampling_reference(
    fixed_image="/absolute/path/t1_brain.nii.gz",   # 3D T1w
    moving_image=raw_bold_path,                      # 提供 BOLD 分辨率与方向
    fov_mask="/absolute/path/t1_fov_mask.nii.gz",    # 与 T1w 同网格的非空 3D FOV 掩膜
    output="/absolute/path/output/sampling_reference.nii.gz",
)
```

| 输入／参数 | 格式与意义 |
|---|---|
| 缩放的 `input_path`、`mask_path` | 完整 4D BOLD 和同仿射、同前三维的 3D 掩膜。掩膜值 >0.5 的所有体素、所有时点共同确定百分位；偶数样本按 FSL 的上中位数规则取值。 |
| 高通的 `cutoff_seconds`、`tr_seconds` | 正的秒数；二者确定 Gaussian sigma。高通保留完整帧数，没有裁短时间窗口。 |
| `device`、`voxel_chunk`／`voxel_batch_size` | 计算设备和空间分批大小。输出为 float32；高通矩阵乘法沿用已有 float64 累加，避免约 10,000 单位基线掩盖小 BOLD 残差。 |
| STC `metadata` | `RepetitionTime` 为正秒数；`SliceTiming` 为每片的有限采集时间；`SliceEncodingDirection` 可为 i/j/k 或其反向。真实数据本轮覆盖 k 轴。 |
| `reference_fraction`、`ignore` | 目标为最早时刻加 fraction×采集范围，按毫秒舍入；ignore≥0，忽略后至少留五帧。本轮另测 ignore=4、fraction=0。 |
| 参考网格 `fixed_image`、`moving_image`、`fov_mask` | T1w 的三维数据、BOLD 的体素尺寸，以及 T1w 网格的 FOV 掩膜。网格转换为 RAS 方向，BOLD 尺寸取三位小数，FOV 包围盒外扩两体素。 |

缩放返回一个乘法因子并写出同源网格的 float32 4D NIfTI；高通写出同网格、同 TR 的 float32 4D NIfTI。STC 返回输出路径和时序元数据，ignored 帧保持原值。采样参考是重采样后的 3D T1w，qform 和 sform code 均为 2；它不估计 T1w–BOLD 配准矩阵。

## 3. 命令行调用

上述四个独立 helper 使用 Python API。完整 pipeline 的对应设置为：

```bash
fnit-fmri volume --bids-root /absolute/path/bids \
  --derivatives-root /absolute/path/derivatives --subject EXAMPLE \
  --synthstrip-weights /absolute/path/synthstrip.pt \
  --mni-template /absolute/path/mni_template.nii.gz \
  --mni-brain-mask /absolute/path/mni_mask.nii.gz \
  --registration-backend fnirt --fnirt-preset t1 \
  --device cpu --highpass-cutoff-seconds 100 --ignore-slice-timing
```

完整权重、配准和输出参数见 [volume 用法](README.md)。只有需要 STC 时才把 `--ignore-slice-timing` 替换为 `--slice-timing --slice-time-reference 0.5`。

## 4. 原软件调用

```bash
# 原版 FEAT 缩放；P50 与 FNIT 使用相同完整时序及脑掩膜。
P50=$(fslstats motion_corrected.nii.gz -k brain_mask.nii.gz -p 50)
SCALE=$(awk -v value="$P50" 'BEGIN {printf "%.17g", 10000/value}')
fslmaths motion_corrected.nii.gz -mul "$SCALE" scaled.nii.gz

# TR=0.735 s、cutoff=100 s；sigma=68.0272108844 volumes。
fslmaths scaled.nii.gz -Tmean mean.nii.gz
fslmaths scaled.nii.gz -bptf 68.0272108844 -1 highpass_zero_mean.nii.gz
fslmaths highpass_zero_mean.nii.gz -add mean.nii.gz highpass.nii.gz

# timings.1D 含实际每片采集秒数；tzero 从采集范围和 reference_fraction 求出。
# 完整数值控制以预先核对的 float32 输入为参考；原始 int16 协议另行保留。
3dTshift -Fourier -TR 2.1s -tzero "$SLICE_TARGET_SECONDS" -ignore 0 \
  -tpattern @timings.1D -prefix stc.nii.gz raw_bold_float32.nii.gz
```

原版 `GenerateSamplingReference` 使用同一个 fixed T1w、moving BOLD 和 fov_mask，详见 [隔离原版调用](../../validation/fmri/e2e_latest/make_original_sampling_reference.py)。FSL、AFNI、NiWorkflows 和 fMRIPrep 只在验证环境中运行。

AFNI `3dTshift` 没有 `-float` 选项，原程序沿用输入的存储类型。本轮原始 BOLD 为 int16；此前直接执行原程序时，float32 FNIT 输出与 int16 参考的差值包含最后的整数取整。浮点控制在计时外将完整原始值无损保存为 float32，重新核对所有体素、网格、TR 和 SHA，再运行原程序。该控制用于比较算法数值；fMRIPrep 25.2.4 的 TShift workflow 本身没有显式指定浮点类型。[原 AFNI 帮助](https://afni.nimh.nih.gov/pub/dist/doc/program_help/3dTshift.html)、[原选项解析源码](https://github.com/afni/afni/blob/master/src/3dTshift.c)。

## 5. 完整真实数据精度、时间与脑图

固定基线为 `cc9402734faeba93b3a13c29932fa1392eaccf62`。时间处理使用完整 `88×88×64×490` 真实运动校正 BOLD；STC 使用公开数据的完整 `64×64×42×180` 原始 BOLD；参考网格使用完整真实 T1w、BOLD 几何与 WM FOV 掩膜，输出 `55×72×60`。WM 掩膜在这里限定参考网格范围，不代表生产 T1w 脑掩膜提取的 benchmark。

CPU 为 Intel Xeon Gold 6418H；CPU1／8 分别绑定一个／八个相同物理核心，同组串行运行。以下为首次完整 API 墙钟，包含正常读写及输出保存；导入、SHA 校验和后续数值分析另计。FNIT warm 是同一进程的第二次完整调用；原软件各次是新进程。各一次时间是观测值。

| 完整处理 | FNIT CPU1／8 | 原版 CPU1／8 | FNIT warm CPU1／8 |
|---|---:|---:|---:|
| 缩放＋高通，保留均值 | 99.611／87.378 s | 300.469／294.973 s | 96.751／89.924 s |
| 缩放＋高通，不恢复均值 | 109.694／98.432 s | 293.906／283.869 s | 未重复 |
| 原生 BOLD 分辨率 T1w 参考 | 5.094／5.200 s | 12.549／14.628 s | 5.159／4.701 s |

缩放和高通的时间来自连续执行两步的完整 API，未单独测量两步墙钟，不能从总时间推算单步速度。逐步数值比较覆盖全部网格、全部 490 帧；两种线程预算的结果相同：

| 子函数／设置 | 全网格 RMSE | 最大绝对差 | 平均体素时间相关 |
|---|---:|---:|---:|
| 全运行缩放，两种高通设置 | 0 | 0 | 1 |
| 高通，保留均值 | 0.000232276 | 0.00390625 | 0.999999999999706 |
| 高通，不恢复均值 | 0.00000847004 | 0.001953125 | 0.9999999999999999 |
| 3D T1w 采样参考 | 0 | 0 | 不适用 |

两边输出均为 float32、有限值、同一预期网格。采样参考的数据逐值相同，qform/sform 矩阵与仿射一致，双方 code 均为 2。完整来源、两种线程预算和逐图指标见 [13 项最终 helper 报告](../../validation/fmri_cpu_20261004/task05_volume/helper_float32_protocol_20261005.public.json)。本轮没有修改这些 helper 的生产源码；此固定输入对照独立于最新完整 volume pipeline 的验收。

### 可选 slice timing：存储协议与浮点控制

旧的原始 int16 参考保留在 [首轮完整聚合报告](../../validation/fmri_cpu_20261004/task05_volume/helper_int16_protocol_20261005.public.json)：默认 CPU1／8 全网格 RMSE 均为 **0.283332**，最大绝对差 **0.500244**；ignore=4、reference_fraction=0 的 RMSE 为 **0.274697**。这是不同输出存储类型的比较，不能直接据此判断 Fourier 校正算法错误。忽略的前四帧在 FNIT 中逐值保持原样。

完整 float32 输入转换已核对：所有原始数值逐值相同，尺寸、仿射、TR 和时间单位不变；一次转换耗时 **3.941 s**，不计入后续原版 STC API。同一新核心组上的 3 次原版和 6 次 FNIT first/warm 完整调用均已完成，并经过独立的全部 30,965,760 个值核对：

| STC 设置 | 线程 | FNIT first／warm API | AFNI API | 全网格 RMSE | 最大绝对差 |
|---|---:|---:|---:|---:|---:|
| fraction=0.5、ignore=0 | 1 | 6.811／6.673 s | 10.165 s | 0.0000192785 | 0.000732422 |
| fraction=0.5、ignore=0 | 8 | 6.312／6.417 s | 9.890 s | 0.0000192785 | 0.000732422 |
| fraction=0、ignore=4 | 8 | 6.608／6.431 s | 10.071 s | 0.0000189300 | 0.000854492 |

三个设置的平均体素时间相关均超过 **0.99999999999985**。ignore=4 的前四帧在两边均逐值保留；新核心组的 FNIT 输出与先前三项输出的全部数据、影像头网格和二进制 NIfTI 头均相同。新官方输出是 float32，支持算法的高精度数值一致性；它不复现原始 int16 输出的取整。AFNI API 含本轮实际容器启动、正常读取、计算与保存；这些是各一次完整观察，不是纯内核时间。所有 first/warm 时钟和完整新、旧结果见 [最终聚合报告](../../validation/fmri_cpu_20261004/task05_volume/helper_float32_protocol_20261005.public.json)。

以下只引用仓库中此前已公开的 MNI BOLD 对照图，说明体积输出的可视化方式；不是本轮 helper 的输入、精度或计时证据。本轮新产生的个体图像未公开。

![此前公开的 MNI preproc 体积对照](../../validation/fmri/e2e_latest/figures/preproc_mni_fnit_original.png)

## 6. 最近更新与 benchmark 记录

- 2026-10-04—05：完成两种高通设置的完整 CPU1／8 官方对照，输出缩放值完全一致，高通残差见上表；完整采样参考逐值一致。全部比较使用原始完整尺寸。
- 2026-10-05：发现首轮 AFNI STC 保存为 int16，而 FNIT 保存 float32。保留三项原协议结果；完成无损完整 float32 输入的三项原版控制、同核 FNIT first/warm 运行和 13 项独立全网格分析。float32 STC RMSE 约 1.9×10⁻⁵；旧新 FNIT 数据和影像头相同。未修改 helper 生产实现和默认 STC 设置。
- 最新 main 的 volume 默认使用 robust BOLD reference；此前完整 volume 的 first/cache-call 时间仅作 [执行基线](../../validation/fmri_cpu_20261004/task05_volume/baseline_execution_20261005.public.json)，不作为当前整链精度结论。本页三个 helper 的固定输入对照与整链验证分开记录。

## 7. 原实现与参考文献

- [FSL utilities](https://fsl.fmrib.ox.ac.uk/fsl/docs/utilities/fslutils.html)、[FSL FEAT](https://fsl.fmrib.ox.ac.uk/fsl/docs/task_fmri/feat/index.html)；Smith SM et al. Advances in functional and structural MR image analysis and implementation as FSL. NeuroImage, 2004。
- [AFNI 3dTshift](https://afni.nimh.nih.gov/pub/dist/doc/program_help/3dTshift.html)、[AFNI 原代码库](https://github.com/afni/afni)；Cox RW. AFNI: Software for analysis and visualization of functional magnetic resonance neuroimages. Computers and Biomedical Research, 1996。
- [NiWorkflows 1.14.4 原实现](https://github.com/nipreps/niworkflows)、[Nilearn 0.11.1 原实现](https://github.com/nilearn/nilearn)；来源与许可见 [第三方记录](../../THIRD_PARTY_NOTICES.md)。
- [fMRIPrep 原实现](https://github.com/nipreps/fmriprep)、[25.2.4 文档](https://fmriprep.org/en/25.2.4/)；Esteban O et al. fMRIPrep: a robust preprocessing pipeline for functional MRI. Nature Methods, 2019。
