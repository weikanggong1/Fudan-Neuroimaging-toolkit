# 已处理 UKB volume/surface 的 MS-HBM 官方 release 对照

这是 2026-10-01 的下游网络验证，推断源码为 `09a0313`，上游 BOLD 为 `3f8b756` 的完整 FNIT API 运行。`core.py` 与固定 HCP_40 prior 沿用已验证的 CBIG 适配；新代码负责体积投影、标签回写及网络/CIFTI 文件输出。

## 参照来源与匹配

- Surface：用户提供的 UKB 官方 release，`surf_fMRI/CIFTIs/bb.rfMRI.MNI.MSMAll.dtseries.nii`，490×91,282，TR 0.735 s。参照文件是官方处理结果，未用 FNIT 的球面/投影结果替代。
- Volume：同一次扫描 UKB rfMRI ZIP 的 `rfMRI.ica/filtered_func_data_clean.nii.gz`，使用同 ZIP 的 `reg/example_func2standard_warp.nii.gz`，经原版 FSL 6.0.7.22 `applywarp` 生成 91×109×91×490 的 MNI152 2 mm BOLD。
- 官方 release 的左右白质表面与匹配 FreeSurfer 存档的 scanner-RAS 几何逐点核对，顶点数为 120,035/122,950，最大坐标差 0.0000267 mm。两份时序的帧数、TR 与起始时间相同。
- 官方 volume 在 996 个抽样皮层下灰质坐标做 2 mm FWHM 平滑后，与官方 CIFTI 的时间相关性中位数为 0.9999745、均值 0.9438834；不同 atlas dilation/masking 的边界仍有差异。这一检查进一步支持使用的是匹配扫描，不将解剖或帧数相同单独当作时序来源证明。
- 官方 CIFTI provenance 对应 Workbench 1.4.2、MSMAll 及 2 mm FWHM 表面平滑；原软件发布文件名对应 [UKB `bb_surf_clean`](https://git.fmrib.ox.ac.uk/falmagro/UK_biobank_pipeline_v_1/-/blob/master/bb_surf_pipeline/bb_surf_clean) 与 [UKB surface 发布字段](https://biobank.ctsu.ox.ac.uk/ukb/field.cgi?id=32136)。无法从这些文件确定整个发布 pipeline 的源码 commit，不将当前 upstream commit 当作发布版本。

原数据、逐顶点/体素数组和原始 provenance 路径留在服务器。仓库只保存匿名指标、哈希与公开模板几何上的对照 PNG。

## 模型与统计定义

两边固定 HCP_40 fsLR32k 17-network prior、`w=200`、`c=50`。单 run 分为前后各 245 帧。无 censor；不重排、不截短时间轴。Volume 在双侧 CBIG fsLR32k 群体 MNI 中层表面三线性采样，两边使用同一坐标；MS-HBM 推断后以相同 cortical mask 和 3 mm 最近顶点距离回写标签。

Surface Dice 在完整 59,412 个皮层顶点计算。Volume 另报告投影表面 Dice 与最终 MNI 标签 Dice；最终标签一致率只在两边非零标签的并集计算，避免背景提高一致率。FC 用**固定官方参照标签**提取两边 17 个网络的平均时序，比较 136 条上三角非对角边。没有把分别变化的分区混入时序差异。

| 指标 | Volume | Surface |
|---|---:|---:|
| fsLR32k 标签一致率 | 0.837474 | 0.724365 |
| fsLR32k 平均 Dice | 0.836288 | 0.717516 |
| MNI 标签一致率（非零并集） | 0.835782 | — |
| MNI 标签平均 Dice | 0.833613 | — |
| 有效时间相关性顶点数 | 58,928 | 59384 |
| 逐顶点时间 r 均值 / 中位数 | 0.325510 / 0.317377 | 0.269508 / 0.246646 |
| 17 个固定网络时间 r 均值 | 0.570771 | 0.580207 |
| FC 上三角 r | 0.845574 | 0.818932 |
| FC MAE | 0.276401 | 0.260353 |
| FC 最大绝对差 | 0.755338 | 0.666002 |

MNI 候选非零标签 82,986 体素，参照 83,840 体素；并集为 83,840。

| HCP_40 网络编号 | Volume 最终 MNI Dice | Surface Dice |
|---|---:|---:|
| 01 | 0.776614 | 0.670677 |
| 02 | 0.780746 | 0.679342 |
| 03 | 0.896685 | 0.804361 |
| 04 | 0.772001 | 0.676369 |
| 05 | 0.800135 | 0.698024 |
| 06 | 0.743413 | 0.597798 |
| 07 | 0.846688 | 0.748647 |
| 08 | 0.783144 | 0.673931 |
| 09 | 0.824410 | 0.729894 |
| 10 | 0.937379 | 0.893134 |
| 11 | 0.913277 | 0.752785 |
| 12 | 0.832454 | 0.702709 |
| 13 | 0.829060 | 0.630639 |
| 14 | 0.888322 | 0.733360 |
| 15 | 0.794135 | 0.616119 |
| 16 | 0.938815 | 0.815501 |
| 17 | 0.814139 | 0.774476 |

## 时间与实现检查

| 操作 | 进程 wall time | 最大 RSS |
|---|---:|---:|
| FNIT CIFTI → MS-HBM/文件输出 | 78.75 s | 2,633,408 KiB |
| 官方 MSMAll CIFTI → 相同 MS-HBM/文件输出 | 73.60 s | 2,635,040 KiB |
| FNIT MNI → 投影/MS-HBM/标签回写/文件输出 | 83.03 s | 3,925,836 KiB |
| 官方 FIX MNI → 相同投影/MS-HBM/标签回写/文件输出 | 96.54 s | 3,925,688 KiB |
| 官方 FIX native → MNI，原版 FSL applywarp | 558.38 s | 5,408,076 KiB |

线程设定为 `OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8`，CUDA 使用共享 gpucw1 H100。MS-HBM 推断仍在 CPU；float32 BOLD、TF32 开启，不用 fp16。表面模型耗时不同主要由输入和收敛过程不同引起，不是不同实现的加速比。官方上游 pipeline 的处理时间没有记录，不推算其完整时间。

FSL 本机在写完输出后返回 255。生成文件已通过 gzip CRC、91×109×91×490、TR、affine 与有限值核查；接受该输出作条件参照，退出状态保留在 [`reference_matching.public.json`](reference_matching.public.json)。不能只凭存在文件认定命令成功。

真实前八帧的独立 SciPy 三线性坐标对照：475,296 个值，r 0.9999999999928、MAE 0.0003133、最大绝对差 0.0151367；GPU 采样 0.6678 s，峰值分配显存 0.07553 GiB，仅适用于八帧采样。完整 Python API 用时 87.2879 s，峰值分配/保留显存 0.17050/0.29883 GiB，运行在 Intel Xeon Gold 6430 + H100 PCIe。Python 与 CLI 的表面及体积标签差异均为 0；新增 CIFTI 和两份网络 TSV 的结构/有限值检查也通过。完整显存及与 CLI 的逐标签核查见 [`volume_api.public.json`](volume_api.public.json)。`tests/test_mshbm.py` 和 `tests/test_mshbm_volume.py` 的 8 项 CPU/CUDA 检查在指定 Conda 环境通过，耗时 5.48 s。

## 复现

先按[功能页](../../docs/mshbm/README.md)安装 Conda 和投影资源。下列路径应指向同一被试、同一次扫描。官方软件只在隔离的**参照生成**中调用；FNIT 运行接口不依赖它。

```bash
# 由 UKB 官方 FIX 清理图及已发布 warp 生成 MNI 对照；不是重新估计配准。
export FSLDIR=/path/to/fsl
export FSLOUTPUTTYPE=NIFTI_GZ
export LD_LIBRARY_PATH="$FSLDIR/lib:${LD_LIBRARY_PATH:-}"
"$FSLDIR/bin/applywarp" \
  --in=official/filtered_func_data_clean.nii.gz \
  --ref="$FSLDIR/data/standard/MNI152_T1_2mm.nii.gz" \
  --warp=official/example_func2standard_warp.nii.gz \
  --mask="$FSLDIR/data/standard/MNI152_T1_2mm_brain_mask.nii.gz" \
  --interp=spline --out=official/ukb_fix_mni2mm.nii.gz

# 分别给 FNIT 和官方时序做同一 MS-HBM；下面以官方 surface 为例。
OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 \
fnit-mshbm --timeseries official/surf_fMRI/CIFTIs/bb.rfMRI.MNI.MSMAll.dtseries.nii \
  --output-dir out/official-surface --w 200 --c 50

# 比较两个已完成的 surface 分区及固定参照标签下的 FC。
python validation/mshbm/compare_processed.py --kind surface \
  --candidate fnit/clean_bold.dtseries.nii \
  --reference official/surf_fMRI/CIFTIs/bb.rfMRI.MNI.MSMAll.dtseries.nii \
  --candidate-labels out/fnit-surface/labels_fslr32k_64984.npy \
  --reference-labels out/official-surface/labels_fslr32k_64984.npy \
  --report out/surface_comparison.public.json
```

Volume 用同一脚本的 `--kind volume`，提供两份 MNI `.nii.gz` 和对应标签 NPY，再加 `--left-surface`、`--right-surface`、`--device cuda:0`。标签 NPY 的目录需同时有 `labels_mni.nii.gz`，脚本比较最终体积标签。各参数对应的数值和结构见功能页。

- [`benchmark_volume.py`](benchmark_volume.py)：调用单被试 Python API，记录总时间/峰值显存，并逐标签检查与 CLI 结果一致。
- [`validate_projection.py`](validate_projection.py)：同一真实 BOLD 前八帧，对照独立 SciPy 插值。
- [`render_processed.py`](render_processed.py)：在服务器生成 template-space/标准表面对照 PNG，不转移影像数组。
- 数值报告：[surface](surface_release_comparison.public.json)、[volume](volume_release_comparison.public.json)、[投影](projection_oracle.public.json)、[数据匹配](reference_matching.public.json)、[输出合同](output_contracts.public.json)、[Python API](volume_api.public.json)。

## 结论范围

MS-HBM 本身沿用已验证的核心算法；本次差异主要衡量上游处理数据不同后的网络结果。官方为 FIX、GDC/B0 与 MSMAll，FNIT 运行采用 ICA-AROMA/混杂回归、无 GDC/B0 与 MSMSulc；官方还做了表面 2 mm FWHM 平滑。当前数据对照没有逐项固定这些步骤，不能将差异分摊到单一算法。共享空间先验会使标签图较时序更相似，本例没有达到官方数值等价。
