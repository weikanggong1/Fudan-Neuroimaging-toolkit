# UKBConnectome_pipeline：从 BIDS DWI/T1 到结构连接矩阵

[返回首页](../../README.md) · [源码](../../src/fnit/connectome/) · [逐阶段验证](../../validation/connectome/ds004666/README.md)

输入原始 BIDS DWI、可选反向相位编码图像和配对 T1w，流程依次运行 PyTorch TOPUP、PyTorch EDDY、官方 FreeSurfer `recon-all`、响应估计、MSMT-CSD、ACT/iFOD2 追踪、SIFT2 和 atlas 端点赋值。已有完整 FreeSurfer subject 或已校正 DWI 时跳过相应阶段。多个 atlas 共用一次追踪和 SIFT2，各输出 count、SIFT2 FBC、mean length、mean FA 四张矩阵。

## 1. 功能和流程

以同一次扫描的 BIDS 元数据确定 DWI、梯度和可选反向相位编码图像；已提供校正 DWI 与旋转 bvec 时直接使用。T1w 的结构重建采用已完成的官方 `recon-all` subject，缺失时才运行官方命令。FNIT 从这些输入计算 DWI 模型、5TT/GMWMI、配准和追踪；先完成**一次全脑追踪与 SIFT2**，再让所选的一套或多套 atlas 分别给同一批流线端点赋值。atlas 的节点编号以各自的 `nodes.tsv` 为准。

```mermaid
flowchart TD
    B["BIDS DWI、bval/bvec、JSON"] --> SEL["选定受试者、session 和 run"]
    SEL --> CORR{"已提供校正 DWI 与旋转 bvec？"}
    CORR -- 是 --> DWI["校正 DWI 与旋转梯度"]
    CORR -- 否 --> REV{"有配对的反向相位编码图像？"}
    REV -- 有 --> TOP["PyTorch TOPUP"] --> EDDY["PyTorch EDDY"]
    REV -- 无 --> EDDY
    EDDY --> DWI

    T1["BIDS T1w 或已完成的 subject"] --> FS{"recon-all 已完成？"}
    FS -- 否 --> RECON["官方 FreeSurfer recon-all"] --> ANAT["读取分割、表面与注释"]
    FS -- 是 --> ANAT

    DWI --> B0["mean b0、脑掩膜"] --> REG["TorchFLIRT：DWI 到 T1"]
    DWI --> FOD["Dhollander → MSMT-CSD → mtnormalise"]
    DWI --> FA["张量拟合 → FA"]
    ANAT --> TISSUE["5TT 与 GMWMI"]
    ANAT --> ATL["构建所选 1..N 套 atlas"]
    REG --> TISSUE_ALIGN["将 5TT/GMWMI 对齐 DWI 世界"]
    TISSUE --> TISSUE_ALIGN
    REG --> ATLAS_DWI["每套 atlas 映射至 DWI"]
    ATL --> ATLAS_DWI

    FOD --> TRACK["一次 iFOD2 + ACT 追踪"]
    TISSUE_ALIGN --> TRACK
    TRACK --> SIFT["一次 SIFT2 估计"]
    FOD --> SIFT
    TISSUE_ALIGN --> SIFT
    TRACK --> METRIC["逐流线长度与平均 FA"]
    FA --> METRIC
    TRACK --> MATRIX["每套 atlas 分别做端点赋值与矩阵汇总"]
    SIFT --> MATRIX
    METRIC --> MATRIX
    ATLAS_DWI --> MATRIX
    MATRIX --> OUT["每套 atlas：nodes.tsv、count、FBC、mean length、mean FA"]
    classDef default fill:#ffffff,stroke:#000000,color:#000000;
```

Tian 路径还需将 MNI 模板标签映射到个体 T1：默认使用 FNIT SynthMorph，提供 `--tian-fnirt-coeff` 时使用已有变形系数；这一分支只影响相应 atlas 的构建。图中官方 `recon-all` 是用户已许可的结构像前置程序，其他计算由 FNIT 完成；Glasser 模板目前另有下文说明的 Workbench 依赖。

## 2. Python 调用、输入与输出

```python
from pathlib import Path
from fnit.connectome import UKBConnectome_pipeline

bids_root = Path("/data/study_bids")                  # 标准 BIDS DWI、梯度和 JSON
output_directory = Path("/data/derivatives/fnit_sc")  # 可恢复的预处理文件
freesurfer_subject = Path("/data/freesurfer/sub-01")  # 已完成的 recon-all subject

pipeline = UKBConnectome_pipeline(device="cuda:0")
result = pipeline.run_bids(
    bids_root=bids_root,
    output_dir=output_directory,
    subject="01",
    freesurfer_subject_dir=freesurfer_subject,
    atlas=("fs-aparc", "fs-aparc-a2009s"),            # 两套 atlas 共用一次追踪
    n_seeds=10000,                                    # 播种尝试数
    seed=0,
)
aparc_result = result.atlas_results["fs-aparc"]
count_matrix = aparc_result.matrices["count"]         # K×K int64
nodes = aparc_result.nodes                            # 行列与 nodes.tsv 对应
```

Python 返回内存中的 `ConnectomeResult`，`run_bids` 保存可恢复的预处理图像；CLI 另将矩阵和节点表写盘。已校正 DWI 可同时传入 `corrected_dwi` 和 `rotated_bvecs`；显式影像接口为 `pipeline(dwi=..., bvals=..., bvecs=..., freesurfer_subject_dir=..., atlas=..., n_seeds=...)`。

- **DWI**：4D NIfTI `[X,Y,Z,N]`；bval 为 N 值，bvec 为 `3×N` 或 `N×3`，校正入口须用 EDDY 旋转后的梯度。原始 BIDS 侧车需定义相位编码方向和读出时间。
- **结构输入**：配对 T1w 或完整 FreeSurfer subject。核心解剖链读取 `mri/brain.mgz`、`mri/aparc+aseg.mgz`；表面 atlas 另读取 ribbon、white/pial、sphere.reg 和相应 annot。所有输入须来自同一受试者。
- **模板**：按所选 atlas 提供 Tian/Schaefer 模板、fsaverage 与 MNI T1。逐文件许可与 SHA-256 见[资源说明](atlas-assets.md)。
- **矩阵**：每个 `atlas_results[name].matrices` 含 `count`、`sift2_fbc`、`mean_length`、`mean_fa`；后三项为 float32。矩阵对称，长度单位 mm，mean length/FA 按 SIFT2 权重求均值，零连接边为零。
- **节点**：`nodes`、`region_labels` 固定矩阵行列顺序，缺失节点保留零行列。首个 atlas 同时对应 `result.matrices`、`result.atlas`。
- **共享结果**：`wm_sh` 为 `[X,Y,Z,45]` FOD；`fa`、`brain_mask` 位于 DWI 网格；`five_tissue`、`gmwmi` 保留 T1 网格，其 affine 映射到 DWI 世界空间。`tractogram` 含路径、端点和长度；`sift2_weights` 是按流线顺序的 float64 向量。

### 输出目录

```text
OUTPUT_DIR/
  dataset_description.json       # BIDS derivative 数据集说明
  run_state.json                 # 完整运行的输入与参数记录
  preproc/raw/                  # BIDS 选片的 AP/PA 入口
  preproc/topup/                # 有反向图像时出现
  preproc/eddy/data.nii.gz      # 校正 DWI
  preproc/eddy/data.eddy_rotated_bvecs
  freesurfer/sub-01/            # 自动运行 recon-all 时出现
  five_tissue_dwi_world.nii.gz  # cGM/sGM/WM/CSF/path，T1 网格
  gmwmi_dwi_world.nii.gz
  fa_dwi.nii.gz
  brain_mask_dwi.nii.gz
  dwi_to_t1_world.csv           # 4×4 DWI RAS mm → T1 RAS mm
  atlases/fs-aparc/
    atlas_dwi.nii.gz            # 0 背景、1..K 连续节点
    nodes.tsv                   # index/original_label/hemisphere/name
    region_labels.csv
    connectome_count.csv        # K×K int64，流线条数
    connectome_sift2_fbc.csv    # K×K，SIFT2 权重和
    connectome_mean_length.csv  # K×K，SIFT2 加权平均长度，mm
    connectome_mean_fa.csv      # K×K，SIFT2 加权平均 FA
  atlases/<other-atlas>/...      # 每个 atlas 各一套
```

旧显式 `--dwi/--bvals/--bvecs` 单 atlas 入口保留平铺输出；BIDS 或多 atlas 入口按 `atlases/<name>/` 分开。矩阵行列以同目录的 `nodes.tsv` 为准。零连接边的均值为零。

### 参数

| 参数 | 输入和默认值 |
|---|---|
| `--bids-root`、`--subject` | 原始 BIDS 根目录与受试者编号；BIDS 入口必选，不能与显式 DWI 三件套混用。 |
| `--session`、`--run`、`--acquisition`、`--direction` | 可选 BIDS 实体，候选不唯一时用于精确选片。 |
| `--t1`、`--freesurfer-subject-dir` | 可选 T1w 路径和已完成 recon-all subject；无 subject 时使用 BIDS T1w 启动官方重建。 |
| `--corrected-dwi`、`--rotated-bvecs` | BIDS 模式下已有校正 4D 图像及逐卷旋转梯度，必须成对给出。 |
| `--dwi`、`--bvals`、`--bvecs` | 旧显式入口：已校正 4D NIfTI、每卷 b 值、eddy 旋转后的 3×N/N×3 FSL 梯度，须同时提供。 |
| `--atlas` | 一个或多个上述名称；默认 `fs-aparc`，顺序决定 Python 返回结果的主 atlas。 |
| `--atlas-templates-dir`、`--fsaverage-dir` | Tian/Schaefer/Glasser 模板目录；Schaefer、Glasser 另需 fsaverage subject。 |
| `--download-atlases` | 可选；只下载所选 Tian/Schaefer 文件，按固定清单校验大小和 SHA-256；Glasser 暂不支持镜像下载。 |
| `--mni-template`、`--synthmorph-weights`、`--tian-fnirt-coeff` | Tian 同网格 MNI T1；可选 SynthMorph 权重目录；已有 FNIRT 前向 coefficient 可替代 SynthMorph。 |
| `--brain-mask`、`--response-mask`、`--fod-mask`、`--normalise-mask`、`--fa-map` | 可选同 DWI 网格掩膜/FA；省略时 FNIT 计算。BET 内部临时转 LAS 后将掩膜映回原网格。 |
| `--shell-bvals`、`--dwi-to-t1-world` | 可选 shell 中心序列和 4×4 DWI→T1 RAS-mm 矩阵；省略则估计 shell 并运行 TorchFLIRT。 |
| `--n-seeds`、`--seed` | 播种尝试数必选；随机种子默认 0。PyTorch 与 MRtrix 相同数值 seed 不生成同一流线。 |
| `--device`、`--compile-arc` | 设备默认 `cuda:0`；可选编译 iFOD2 CUDA 核。CUDA 默认 TF32，不自动用半精度。 |
| `--output-dir`、`--overwrite` | 结果目录必选；后者强制重算和覆盖。 |

## 3. 命令行调用

```bash
conda env create -f environment.yml
conda activate fnit

BIDS_ROOT=/data/study_bids                   # 原始 BIDS 根目录，含 dataset_description.json
OUTPUT_DIR=/data/derivatives/fnit_connectome # 本次结果与可恢复的预处理目录
FREESURFER_SUBJECT=/data/freesurfer/sub-01   # 已完成的官方 recon-all subject；可省略

fnit UKBConnectome_pipeline \
  --bids-root "$BIDS_ROOT" --subject 01 \
  --freesurfer-subject-dir "$FREESURFER_SUBJECT" \
  --atlas fs-aparc --n-seeds 10000 --seed 0 \
  --device cuda:0 --output-dir "$OUTPUT_DIR"
```

省略 `--freesurfer-subject-dir` 时，命令读取 BIDS `anat/*_T1w.nii[.gz]` 并调用用户安装的官方 `recon-all -sd OUTPUT_DIR/freesurfer -s sub-01 -i T1w -all`。不完整 subject 用不带 `-i` 的 `-all` 续跑。用户需自行安装并许可 FreeSurfer。FNIT 后续计算仅读取其图像和表面。

多个 DWI run/session 时用 `--session`、`--run`、`--acquisition`、`--direction` 明确选片。DWI 需要 `.bval`、`.bvec`、JSON 中的 `PhaseEncodingDirection` 以及 `TotalReadoutTime` 或 `EffectiveEchoSpacing`。有反向相位编码 EPI/DWI 时通过 `B0FieldSource`/`B0FieldIdentifier` 或 `IntendedFor` 配对；没有反向图像时跳过 TOPUP，EDDY 无场图运行。侧车支持 BIDS 继承规则，约定依据 [BIDS MRI 规范](https://bids-specification.readthedocs.io/en/stable/modality-specific-files/magnetic-resonance-imaging-data.html)。

### 多 atlas 示例

只用已完成的 FreeSurfer subject 时，可直接选择两套原生全脑 atlas；`fs-aparc-a2009s` 另外读取 `mri/aparc.a2009s+aseg.mgz` 与双半球 `label/*.aparc.a2009s.annot`，不需要 MNI 模板或配准权重：

```bash
BIDS_ROOT=/data/study_bids                         # 原始 BIDS 根目录
OUTPUT_DIR=/data/derivatives/fnit_connectome       # 输出和续跑目录
FREESURFER_SUBJECT=/data/freesurfer/sub-01          # 含两个 aparc 分割的 recon-all subject
fnit UKBConnectome_pipeline \
  --bids-root "$BIDS_ROOT" --subject 01 \
  --freesurfer-subject-dir "$FREESURFER_SUBJECT" \
  --atlas fs-aparc fs-aparc-a2009s \
  --n-seeds 10000 --device cuda:0 --output-dir "$OUTPUT_DIR"
```

该体积由官方 [`mri_aparc2aseg --s SUBJECT --annot aparc.a2009s --o aparc.a2009s+aseg.mgz`](https://surfer.nmr.mgh.harvard.edu/fswiki/mri_aparc2aseg) 产生；FNIT 根据受试者注释名称和 [FreeSurferColorLUT 的 11100/12100 系列原始编号](https://github.com/freesurfer/freesurfer/blob/dev/distribution/FreeSurferColorLUT.txt)转成 `1..K`，再使用与 `fs-aparc` 相同的 16 个皮层下/小脑节点。参考文献：[Destrieux 等，*NeuroImage* 2010](https://doi.org/10.1016/j.neuroimage.2010.06.010)。

使用 Tian/Schaefer 模板的多 atlas 示例：

```bash
BIDS_ROOT=/data/study_bids                         # 同一组 DWI 与 T1w
OUTPUT_DIR=/data/derivatives/fnit_connectome       # 结果目录
FREESURFER_SUBJECT=/data/freesurfer/sub-01          # 官方 recon-all 结果
ATLAS_TEMPLATES=/data/atlas/ukb                     # Tian S1/S4 和 Schaefer 注释
FSAVERAGE_SUBJECT=/opt/freesurfer/subjects/fsaverage # fsaverage sphere.reg
MNI_TEMPLATE=/data/templates/MNI152_T1_2mm_brain.nii.gz # Tian 同网格 MNI T1

fnit UKBConnectome_pipeline \
  --bids-root "$BIDS_ROOT" --subject 01 \
  --freesurfer-subject-dir "$FREESURFER_SUBJECT" \
  --atlas fs-aparc schaefer200+tian-s1 schaefer500+tian-s4 \
  --atlas-templates-dir "$ATLAS_TEMPLATES" \
  --fsaverage-dir "$FSAVERAGE_SUBJECT" --mni-template "$MNI_TEMPLATE" \
  --n-seeds 10000 --seed 0 --device cuda:0 \
  --output-dir "$OUTPUT_DIR"
```

可选名称：`fs-aparc`、`fs-aparc-a2009s`、`aparc+tian-s1`、`aparc.a2009s+tian-s1`、`glasser+tian-s1`、`glasser+tian-s4`、`schaefer200+tian-s1`、`schaefer500+tian-s4`、`schaefer1000+tian-s4`。前两套从个体 FreeSurfer 分割直接生成；`fs-aparc-a2009s` 由本人的 `.annot` 定义名称，按左皮层、右皮层、16 个皮层下与小脑节点连续编号，行列以 `nodes.tsv` 为准。Tian 默认用 FNIT PyTorch SynthMorph；已有原 UKB 的 T1→MNI FNIRT coefficient 时可以 `--tian-fnirt-coeff` 替换。Glasser 的 32k→164k 标签重采样目前仍依赖 Connectome Workbench `wb_command`，故该 atlas 还不满足仅官方 recon-all 为外部运行时程序的约束。

资源许可、模板结构与按需下载见[atlas 资源说明](atlas-assets.md)。固定 FNIT Release `assets-v1` 暂无这组 atlas；有明确许可的 Tian/Schaefer 文件已镜像于仓库，命令可加 `--download-atlases` 自动获取并校验。fsaverage 和 MNI T1 仍由用户提供。

### 跳过与续跑

| 阶段 | 判定 | 主要输出 |
|---|---|---|
| BIDS 选片 | 输入路径、大小、修改时间和元数据与记录相同 | `preproc/raw/AP.*`、可选 `PA.*`、`bids_selection.json` |
| TOPUP | 状态记录匹配且场系数、校正 b0、采集参数完整；无反向图像时不运行 | `preproc/topup/fieldmap_out_*` |
| EDDY | 状态记录匹配且校正 DWI、旋转 bvec 完整 | `preproc/eddy/data.nii.gz`、`data.eddy_rotated_bvecs` |
| recon-all | 显式提供完整 subject，或 `recon-all.done`、脑图、分割及双半球表面存在 | `freesurfer/sub-01/` |
| connectome | BIDS 模式下运行记录匹配且所有矩阵完整 | `atlases/<name>/` 和共用图像 |

状态记录用文件大小、mtime 和参数核对 FNIT 自己产生的中间结果，不是内容哈希。外部替换了文件但保留原大小/mtime 时加 `--overwrite`。在其他软件中已完成 TOPUP/EDDY 时，同时给 `--corrected-dwi` 和 `--rotated-bvecs`；原始 BIDS bval 仍定义每卷 b 值。`--overwrite` 强制重算并覆盖同名结果。

## 4. 原软件调用

以下命令用于独立 MRtrix 对照，输入须与 FNIT 使用同一 FOD、5TT、GMWMI、FA 与 atlas；完整前处理和七模板命令在[逐阶段验证](../../validation/connectome/ds004666/README.md)中。

```bash
MRTRIX_RNG_SEED=0 tckgen wm_fod_norm.mif tracks.tck \
  -algorithm iFOD2 -act five_tissue.mif -seed_gmwmi gmwmi.mif \
  -seeds 100000 -select 0 -maxlength 250 -cutoff 0.1 -samples 3 -power 0.5
tcksift2 tracks.tck wm_fod_norm.mif weights.txt -act five_tissue.mif
tcksample tracks.tck fa.mif mean_fa.txt -precise -stat_tck mean
tck2connectome tracks.tck atlas.mif count.csv -symmetric -assignment_radial_search 4
tck2connectome tracks.tck atlas.mif fbc.csv -symmetric -assignment_radial_search 4 \
  -tck_weights_in weights.txt
```

`mean_length` 和 `mean_fa` 的完整官方命令及固定 TCK 数值比较见[SIFT2/FA/矩阵验证](../../validation/connectome/ds004666/README.md#与官方流程逐项对照)。

## 5. 精度、运行时间与脑图

| 真实输入对照 | 已观察结果 | 证据 |
|---|---|---|
| 最新无损组件优化，真实 ds004666 | 2,000 次播种两版路径及相关量逐值相同；轨迹整理 313.93→4.69 ms。三 atlas 构建/矩阵阶段 10.07→5.63 s；固定真实 TCK 七套四矩阵全部逐值相同 | [范围、profile、时间和显存](../../validation/connectome/ds004666/lossless_20261002/README.md) |
| 原始 UKB AP/PA BIDS 全链 | 100 次播种的流程检查成功；84 节点四矩阵完整，3,570 s、PyTorch 峰值 4.542 GiB；同输入重跑 4.06 s 且矩阵哈希不变 | [私有输入的公开汇总](../../validation/connectome/ukb_bids_e2e_20260930.md) |
| 校正 UKB DWI + 两套原生 FreeSurfer atlas | 一次运行得到 84/164 节点各四矩阵，6,159 s、PyTorch 峰值 2.687 GiB；续跑 3 s 且八矩阵哈希不变。独立重算的配准变换和首次运行不同，矩阵未逐值一致 | [同一受试者的公开汇总](../../validation/connectome/ukb_bids_e2e_20260930.md) |
| AP/PA TOPUP | UKB 一例校正 4D r=0.9944；未逐体素等价 | [TOPUP 报告](../../validation/topup/report.public.json) |
| EDDY | UKB 一例对 FSL GPU 的脑内 4D r=0.999738，FNIT 10:38.75、参考 10:21.19；计时边界不同 | [EDDY 报告](../../validation/eddy/README.md) |
| 固定同一 100k TCK/权重/atlas | 七套 count 逐元素相同；FBC 最大绝对误差 ≤5.07e-5 | [七 atlas 矩阵和脑图](../../validation/connectome/ds004666/seven_atlas_100k_20260929.md) |
| 独立 100k 追踪 | 部分 count/support 落入 MRtrix 自身三次重复范围；长度、8 mm 端点和 TDI 未全面进入 | [三次对照和脑图](../../validation/connectome/ds004666/tracking_100k_three_seed_20260929.md) |

最终随机追踪验收采用**MRtrix 自身重复范围**：双方固定同一输入，各运行多个种子，比较接受率、长度分布、端点、TDI 及每套 atlas 的四张矩阵；FNIT 落入参考范围即可，无需比官方自身更稳定。固定轨迹矩阵精度已高，独立追踪仍有指标未达成；BIDS 编排的加入不等于原 UKB 全链数值一致。100 万及 1,000 万播种仍需实测性能。

最新优化自动生效，保持原数值与 RNG 操作。实测 Torch 已分配/预留峰值最高为 2.544/2.938 GB；该范围固定已有配准，尚未重新测量原始 BIDS 整链总时间。完整精确 FA 只从 109.81 降至 106.82 ms，收益很小；追踪的 SH、组织采样与圆弧概率仍是后续主要优化对象。

![配对 T1、校正前后 b0 与 atlas](figures/ds004666_t1_raw_vs_topup_eddy_atlas.png)

### BEDPOSTX + ProbtrackX2 能否作为完整对照？

**可以作为同一输入的 FSL-FDT 独立流程，不能代替原 UKB-connectomics 的 MRtrix 数值基准。**独立流程可共用同一份校正 DWI、旋转梯度、脑掩膜、DWI 空间的 atlas ROI 及 `nodes.tsv` 顺序：`BEDPOSTX` 估计逐体素纤维方向后验，`ProbtrackX2 --network` 从每个 ROI 播种并输出 `fdt_network_matrix`。官方定义中，第 *i* 行第 *j* 列是从 ROI *i* 发出的样本到达 ROI *j* 的计数，因此通常是有向矩阵；[BEDPOSTX](https://fsl.fmrib.ox.ac.uk/fsl/docs/diffusion/bedpostx.html)和[ProbtrackX](https://fsl.fmrib.ox.ac.uk/fsl/docs/diffusion/probtrackx.html)文档给出输入与矩阵语义。

独立 FSL benchmark 的命令形式如下。先将同一张 `atlas_dwi.nii.gz` 按 `nodes.tsv` 顺序拆成各节点的二值 NIfTI，并逐行写入 `roi_list.txt`；`BEDPOSTX_INPUT` 目录放校正后的 `data.nii.gz`、`bvals`、旋转后的 `bvecs` 和 `nodif_brain_mask.nii.gz`。这条全脑串联目前是**待验证的对照设计**，不是 `UKBConnectome_pipeline` 已运行的分支。

```bash
BEDPOSTX_INPUT=/data/fsl_reference/sub-01             # 同一校正 DWI、梯度和脑掩膜
ROI_LIST=/data/fsl_reference/sub-01/roi_list.txt       # 每行一个 DWI 空间二值 ROI，顺序同 nodes.tsv
FSL_NETWORK_DIR=/data/fsl_reference/sub-01/network     # 官方网络矩阵输出目录

bedpostx "$BEDPOSTX_INPUT" -n 3 -model 2 -w 1 -b 1000 -j 1250 -s 25
probtrackx2 -s "${BEDPOSTX_INPUT}.bedpostX/merged" \
  -m "${BEDPOSTX_INPUT}.bedpostX/nodif_brain_mask.nii.gz" \
  -x "$ROI_LIST" --network --dir="$FSL_NETWORK_DIR" --forcedir \
  -P 5000 -S 2000 --steplength=0.5
```

| 比较对象 | 原 UKB / 本流程 | FSL-FDT 独立流程 | 可比性 |
|---|---|---|---|
| 局部方向模型 | Dhollander 响应、MSMT-CSD、归一化 WM FOD | BEDPOSTX 纤维方向后验 | 同 DWI 可分别验证模型输出；参数不是同一个量。 |
| 播种与追踪 | GMWMI 播种，iFOD2 + 5TT/ACT，生成一次全脑流线集 | 各 ROI 体素播种，从后验抽方向并传播 | 比较空间覆盖、长度和重复性；播种分布与解剖约束不同。 |
| 边权 | 端点配对的流线条数与 SIFT2 权重和 | 种子 ROI 到目标 ROI 的样本命中数 | 矩阵可按相同节点对齐后比较支持和秩；原始值、方向性和单位不同。 |
| 其他矩阵 | SIFT2 加权 mean length、mean FA | 可另算路径长度；默认 `--network` 不产生这两张同定义矩阵 | 不能把 FSL 原始 network 矩阵当成四张 UKB 矩阵的逐值参考。 |

因此，若“完全对照”指**从同一 BIDS 输入独立得到 ROI×ROI 结果**，FSL 路线可行；若指**复现原 [UKB 追踪脚本](https://github.com/sina-mansour/UKB-connectomics/blob/main/scripts/bash/probabilistic_tractography_native_space.sh)的四种边权和端点定义**，答案是否定的。给 FSL 计数做归一化或对称化后，可以研究跨方法的一致性，但不会变成 [MRtrix `tcksift2`](https://mrtrix.readthedocs.io/en/latest/reference/commands/tcksift2.html) 加 [`tck2connectome`](https://mrtrix.readthedocs.io/en/latest/reference/commands/tck2connectome.html) 的结果。FSL `matrix1/2/3` 也各有种子或目标体素定义，不应改名充当原 UKB 的端点矩阵。

FNIT 已有独立的 [TorchBEDPOSTX](../bedpostx/README.md) 和 [TorchProbtrackX](../probtrackx/README.md)；后者支持 DWI 网格体积 ROI 的 `regions` 网络模式。现有真实 DWI 证据只覆盖 BEDPOSTX 的小范围体素检查，以及**固定 FSL BEDPOSTX 后验**时 ProbtrackX 的五区网络：计数模式的网络密度图 r 为 0.9290（CPU）/0.9436（GPU），非零支持 Dice 为 0.5516/0.5303，原始 ROI 矩阵 MAE 为 0.60/0.76；见[现有 FSL 比较与图](../probtrackx/README.md#与-fsl-的真实-dwi-benchmark)。这没有验证 TorchBEDPOSTX→TorchProbtrackX 的独立全脑串联，更不能证明它与 MRtrix 流程相同。当前 TorchProbtrackX 也未覆盖官方表面播种等全部模式。

若增加这条**独立验证分支**，应依次固定校正 DWI/梯度/掩膜、分辨率和 ROI 顺序；先比较同输入 BEDPOSTX 后验，再固定同一份后验比较 ProbtrackX2 的 `fdt_network_matrix`，最后比较两套独立串联流程。每阶段都记录真实数据墙钟、显存、ROI 矩阵误差及重复运行的支持范围。FSL 原程序只在独立 benchmark 环境运行，不接入 `UKBConnectome_pipeline` 的 FNIT 运行时。

## 6. 最近更新与 benchmark

| 日期 | 更新与证据 |
|---|---|
| 2026-10-02 | 批量整理轨迹、原点打包、多 atlas 复用；真实逐值一致性和计时见[无损优化报告](../../validation/connectome/ds004666/lossless_20261002/README.md) |
| 2026-09-30 | 原始 BIDS、TOPUP/EDDY 自动跳过、单/多 atlas；[真实 UKB 流程与续跑](../../validation/connectome/ukb_bids_e2e_20260930.md) |
| 2026-09-29 | iFOD2/ACT、可选编译核、100k 三种子和七模板对照；[验证索引](../../validation/connectome/ds004666/README.md) |

## 7. 参考文献与原实现

参考步骤对应 [`topup`](https://fsl.fmrib.ox.ac.uk/fsl/docs/diffusion/topup/)、[`eddy_cuda`](https://fsl.fmrib.ox.ac.uk/fsl/docs/diffusion/eddy/)、[`recon-all -i T1w -s SUBJECT -all`](https://surfer.nmr.mgh.harvard.edu/fswiki/recon-all)、[`dwi2response`/`dwi2fod`/`mtnormalise`](https://mrtrix.readthedocs.io/en/latest/dwi_preprocessing/response_function_estimation.html)、[`tckgen -algorithm iFOD2 -act -seed_gmwmi`](https://mrtrix.readthedocs.io/en/latest/reference/commands/tckgen.html)、[`tcksift2`](https://mrtrix.readthedocs.io/en/latest/reference/commands/tcksift2.html)、[`tck2connectome`](https://mrtrix.readthedocs.io/en/latest/reference/commands/tck2connectome.html)。原版脚本见 [UKB-connectomics](https://github.com/sina-mansour/UKB-connectomics)。参考文献：[ACT](https://doi.org/10.1016/j.neuroimage.2012.06.005)、[SIFT2](https://doi.org/10.1016/j.neuroimage.2015.06.092)、[MRtrix3](https://doi.org/10.1016/j.neuroimage.2019.116137)、[Tian atlas](https://doi.org/10.1038/s41593-020-00711-6)。
