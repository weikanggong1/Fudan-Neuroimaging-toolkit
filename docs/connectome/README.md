# UKBConnectome_pipeline：从 BIDS DWI/T1 到结构连接矩阵

[返回首页](../../README.md) · [源码](../../src/fnit/connectome/) · [逐阶段验证](../../validation/connectome/ds004666/README.md)

输入原始 BIDS DWI、可选反向相位编码图像和配对 T1w，流程依次运行 PyTorch TOPUP、PyTorch EDDY、官方 FreeSurfer `recon-all`、响应估计、MSMT-CSD、ACT/iFOD2 追踪、SIFT2 和 atlas 端点赋值。已有完整 FreeSurfer subject 或已校正 DWI 时跳过相应阶段。多个 atlas 共用一次追踪和 SIFT2，各输出 count、SIFT2 FBC、mean length、mean FA 四张矩阵。

## BIDS 用法

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

可选名称：`fs-aparc`、`aparc+tian-s1`、`aparc.a2009s+tian-s1`、`glasser+tian-s1`、`glasser+tian-s4`、`schaefer200+tian-s1`、`schaefer500+tian-s4`、`schaefer1000+tian-s4`。`fs-aparc` 从个体 `aparc+aseg.mgz` 生成，无需外部 atlas。Tian 默认用 FNIT PyTorch SynthMorph；已有原 UKB 的 T1→MNI FNIRT coefficient 时可以 `--tian-fnirt-coeff` 替换。Glasser 的 32k→164k 标签重采样目前仍依赖 Connectome Workbench `wb_command`，故该 atlas 还不满足仅官方 recon-all 为外部运行时程序的约束。

资源许可、模板结构与按需下载见[atlas 资源说明](atlas-assets.md)。固定 FNIT Release `assets-v1` 暂无这组 atlas；有明确许可的 Tian/Schaefer 文件已镜像于仓库，命令可加 `--download-atlases` 自动获取并校验。fsaverage 和 MNI T1 仍由用户提供。

## 跳过与续跑

| 阶段 | 判定 | 主要输出 |
|---|---|---|
| BIDS 选片 | 输入路径、大小、修改时间和元数据与记录相同 | `preproc/raw/AP.*`、可选 `PA.*`、`bids_selection.json` |
| TOPUP | 状态记录匹配且场系数、校正 b0、采集参数完整；无反向图像时不运行 | `preproc/topup/fieldmap_out_*` |
| EDDY | 状态记录匹配且校正 DWI、旋转 bvec 完整 | `preproc/eddy/data.nii.gz`、`data.eddy_rotated_bvecs` |
| recon-all | 显式提供完整 subject，或 `recon-all.done`、脑图、分割及双半球表面存在 | `freesurfer/sub-01/` |
| connectome | BIDS 模式下运行记录匹配且所有矩阵完整 | `atlases/<name>/` 和共用图像 |

状态记录用文件大小、mtime 和参数核对 FNIT 自己产生的中间结果，不是内容哈希。外部替换了文件但保留原大小/mtime 时加 `--overwrite`。在其他软件中已完成 TOPUP/EDDY 时，同时给 `--corrected-dwi` 和 `--rotated-bvecs`；原始 BIDS bval 仍定义每卷 b 值。`--overwrite` 强制重算并覆盖同名结果。

## 输出结构

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

## 参数

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

Python 入口 `UKBConnectome_pipeline(device="cuda:0").run_bids(bids_root, output_dir, subject="01", n_seeds=10000, atlas=("fs-aparc", "aparc+tian-s1"), freesurfer_subject_dir=...)` 返回 `ConnectomeResult`。`result.atlas_results[name]` 分别含 `matrices`、`nodes`、`region_labels`、`atlas` 和仿射；FOD、追踪、SIFT2 与 FA 仅计算一次。Python 方法保存可恢复的预处理图像；矩阵写盘使用 CLI。

## 官方对照和当前结论

参考步骤对应 [`topup`](https://fsl.fmrib.ox.ac.uk/fsl/docs/diffusion/topup/)、[`eddy_cuda`](https://fsl.fmrib.ox.ac.uk/fsl/docs/diffusion/eddy/)、[`recon-all -i T1w -s SUBJECT -all`](https://surfer.nmr.mgh.harvard.edu/fswiki/recon-all)、[`dwi2response`/`dwi2fod`/`mtnormalise`](https://mrtrix.readthedocs.io/en/latest/dwi_preprocessing/response_function_estimation.html)、[`tckgen -algorithm iFOD2 -act -seed_gmwmi`](https://mrtrix.readthedocs.io/en/latest/reference/commands/tckgen.html)、[`tcksift2`](https://mrtrix.readthedocs.io/en/latest/reference/commands/tcksift2.html)、[`tck2connectome`](https://mrtrix.readthedocs.io/en/latest/reference/commands/tck2connectome.html)。原版脚本见 [UKB-connectomics](https://github.com/sina-mansour/UKB-connectomics)。参考文献：[ACT](https://doi.org/10.1016/j.neuroimage.2012.06.005)、[SIFT2](https://doi.org/10.1016/j.neuroimage.2015.06.092)、[MRtrix3](https://doi.org/10.1016/j.neuroimage.2019.116137)、[Tian atlas](https://doi.org/10.1038/s41593-020-00711-6)。

| 真实输入对照 | 已观察结果 | 证据 |
|---|---|---|
| AP/PA TOPUP | UKB 一例校正 4D r=0.9944；未逐体素等价 | [TOPUP 报告](../../validation/topup/report.public.json) |
| EDDY | UKB 一例对 FSL GPU 的脑内 4D r=0.999738，FNIT 10:38.75、参考 10:21.19；计时边界不同 | [EDDY 报告](../../validation/eddy/README.md) |
| 固定同一 100k TCK/权重/atlas | 七套 count 逐元素相同；FBC 最大绝对误差 ≤5.07e-5 | [七 atlas 矩阵和脑图](../../validation/connectome/ds004666/seven_atlas_100k_20260929.md) |
| 独立 100k 追踪 | 部分 count/support 落入 MRtrix 自身三次重复范围；长度、8 mm 端点和 TDI 未全面进入 | [三次对照和脑图](../../validation/connectome/ds004666/tracking_100k_three_seed_20260929.md) |

最终随机追踪验收采用**MRtrix 自身重复范围**：双方固定同一输入，各运行多个种子，比较接受率、长度分布、端点、TDI 及每套 atlas 的四张矩阵；FNIT 落入参考范围即可，无需比官方自身更稳定。固定轨迹矩阵精度已高，独立追踪仍有指标未达成；BIDS 编排的加入不等于原 UKB 全链数值一致。100 万及 1,000 万播种仍需实测性能。
