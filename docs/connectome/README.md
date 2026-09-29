# 单被试 dMRI region × region connectome

[返回首页](../../README.md) · [源码](../../src/fnit/connectome/) · [ds004666 验证](../../validation/connectome/ds004666/README.md)

**工作流边界：** `fs-aparc` 是 FNIT 的原生 84 节点输出，不等于原 UKB-connectomics 的七套「皮层 atlas + Tian」组合。现已接入七套图谱的生成入口；默认 5TT 使用 FreeSurfer 分割，不加入 FIRST。原 UKB 的 FIRST、FNIRT 标签与 1,000 万次播种整链仍属于兼容性验收。

`UKBConnectome` 从**已完成畸变、运动和涡流校正**的 DWI、配套 bval 和 eddy 旋转后的 bvec 起步，结合配对 T1 已完成的**官方 FreeSurfer recon-all** subject 目录、DWI 脑掩膜与自动生成的 84 节点 atlas，返回 count、SIFT2 FBC、加权 mean length 和加权 mean FA 四张矩阵。响应、FOD、强度归一化、5TT/GMWMI、追踪、SIFT2、FA 采样与矩阵赋值由 PyTorch 在指定 CPU/CUDA 设备执行。CUDA 默认允许 TF32；图像主要为 float32，条件数敏感的求解和几何步骤使用 float64，不自动转为 float16/bfloat16。

**当前状态：** `recon-all` 目录可自动生成 84 节点 `fs-aparc` 或原 UKB 七套皮层+Tian 图谱；其中 216 节点 Schaefer200+Tian S1 已通过公开真实 DWI 的[一键四矩阵产物检查](../../validation/connectome/ds004666/atlas_synthmorph_20260929.md)。追踪已改为 iFOD2 校准拒绝采样，固定真实 FOD 的 80 条圆弧和 52 条连续两弧与官方概率最大误差分别为 `1.58×10⁻⁶` 和 `1.49×10⁻⁶`；真实 5TT 的 12,600 个 ACT 状态点与官方无分歧。10,000 固定位置的三次独立追踪与三次官方运行完成 3×3 四矩阵比较；共同边 mean FA 误差有 6/9 对落入官方自身重复范围，count 相对 L1 有 4/9 对落入，仍未达到全指标匹配。[当前追踪报告](../../validation/connectome/ds004666/ifod2_rejection_20260929.md)列出精度、时间、显存和脑图。

进一步的[公开真实输入 100k 比较](../../validation/connectome/ds004666/tracking_100k_matrices_20260929.md)包含三次 MRtrix 与一次 FNIT 的独立追踪、SIFT2、FA、四矩阵和轨迹密度。count 相对 L1 的跨软件比较有 2/3 进入官方自身重复范围；长度 KS、端点分布和 8 mm TDI 仍略超出。FNIT 100k 本次追踪耗时 1,443.91 s、后处理 36.53 s、全链 Torch 峰值 2.473 GiB；共享 H100 的负载与早前纯追踪计时不同。100 万及 1,000 万次播种尚未实测。

可选 `compile_arc=True` / `--compile-arc` 只编译 CUDA iFOD2 圆弧概率核。相同真实输入的[100k 实测](../../validation/connectome/ds004666/tracking_compile_20260929.md)中，首次编译计入后追踪 782.49 s、保留 27,353 条、全链 Torch 峰值 2.468 GiB。未编译的另一次追踪 1,443.91 s、保留 27,401 条；两次共享 GPU 负载不同，不能将墙钟比值视为稳定加速比。长度、端点和 TDI 的独立群体差异仍在官方重复范围外。

[七套原 UKB atlas 的同轨迹与独立轨迹矩阵对照](../../validation/connectome/ds004666/seven_atlas_100k_20260929.md)现已覆盖 84–1054 节点、四种矩阵和真实连接图。同一 27,616 条官方 TCK、权重、长度、FA 和七张 DWI atlas 下，七个 count 矩阵均逐元素一致，FBC/平均长度/平均 FA 最大绝对误差分别不超过 `5.07e−5`、`5.05e−4 mm`、`5.58e−7`。FNIT 自身 100k 轨迹与三次官方随机范围比较仍有未进入的矩阵指标；图谱 Tian 使用 SynthMorph，不能等同原 UKB 的 FNIRT 标签。这七套固定轨迹及独立轨迹 `.npz`、节点表、时间、显存和图已入库；除 Schaefer200+Tian S1 外，各套还没有分别完成正式 CLI 从 DWI 的一键运行。

多 atlas 的函数级验证已覆盖 Schaefer200、500、1000 的 fsaverage→native 表面与 T1 ribbon 体积投影；500/1000 在同一真实 T1 上与原脚本均逐体素一致，Tian S4 合并后 554/1054 个节点在 DWI 网格均有体素，见[扩展图谱报告和脑图](../../validation/connectome/ds004666/atlas_schaefer_multi_20260929.md)。原生 aparc/a2009s 皮层体积也与原版逐体素一致；与 Tian S1 合并后生成 84/164 节点，见[原生图谱实测](../../validation/connectome/ds004666/atlas_native_aparc_20260929.md)。Glasser 皮层体积与原版逐体素一致；与 Tian S1/S4 合并后为 376/414 节点，其中各有两个极小的皮层节点在 DWI 降采样后无体素，见[Glasser 图谱实测与脑图](../../validation/connectome/ds004666/atlas_glasser_20260929.md)。Tian S1/S4 改用 FNIT PyTorch SynthMorph 时，标签与官方 SynthMorph 的前景 Dice 均为 0.981；固定官方形变后，最近邻标签逐体素一致，见[配准实测](../../validation/connectome/ds004666/atlas_synthmorph_20260929.md)。七套原 UKB 图谱均已接入 CLI；仅 Schaefer200+Tian S1 已完成公开真实 DWI 的四矩阵一键产物检查。

## 输入准备与安装

原 [UKB-connectomics 追踪脚本](https://github.com/sina-mansour/UKB-connectomics/blob/main/scripts/bash/probabilistic_tractography_native_space.sh) 读取 UKB 已校正的 `data_ud.nii.gz`、`bvecs` 和 `bvals`；它不从原始 BIDS DWI 执行 TOPUP/eddy。本接口也不在单次 connectome 调用内执行 TOPUP、eddy、去噪或 Gibbs 校正；LAS 网格 DWI 的 mean b0 可在调用内用 PyTorch BET 去脑。公开 ds004666 的校正输入由 AP/PA b0 运行 FSL TOPUP，再用 `eddy_cuda10.2` 生成校正 DWI 与旋转后的 bvec；元数据缺少实测总读出时间，实验两步均使用**假定 0.05 s**。[预处理命令、QC 与哈希](../../validation/connectome/ds004666/corrected_input_provenance.public.json)和[校正参考命令](../../validation/connectome/ds004666/corrected_mrtrix_commands.public.txt)保留完整条件。原始 DWI 不满足本接口的输入约定。

T1 解剖输入须先在包外运行**官方 FreeSurfer recon-all**，提供同次 T1 的 skull-stripped brain 图与 `aparc+aseg.mgz`。包内近似 Python recon-all 不参与当前 connectome 验证，也不在本接口自动执行。`fs-aparc` 模式由 `aparc+aseg.mgz` 自动生成连续编号 atlas；自定义 atlas 可继续使用旧显式输入。DWI 为 LAS 体素顺序时，省略 `brain_mask` 会依照原 UKB 的 `bet mean_b0 -m -R -f 0.2 -g -0.05` 在 PyTorch 中生成脑掩膜；其他方向的 DWI 须提供同网格二值 `brain_mask`。默认不会自动运行 SynthSeg；`fs-aparc` atlas 在包内生成，不下载外部图谱；比较 MRtrix 时应固定同一脑掩膜、atlas、梯度、配准变换和参考响应掩膜。FreeSurfer、FSL 与 MRtrix 是**输入准备和软件对照工具**，不是 PyTorch 核心调用的内部可执行程序。

从仓库根目录安装 GPU 验证环境：

```bash
conda env create -f environment.yml
conda activate fnit
python -c "import torch, fnit; print(torch.__version__, torch.cuda.is_available())"
fnit connectome --help
```

CPU 小规模检查可使用 `--device cpu`。官方 FreeSurfer 的运行与许可由使用者在包外管理；本仓库不提供 recon-all 可执行程序、`license.txt`、UKB 原始受控数据或大体积 T1/DWI。

## Python API

```python
from fnit import UKBConnectome

result = UKBConnectome(device="cuda:0")(
    dwi="derivatives/dwi/sub-01_desc-preproc_dwi.nii.gz",  # 输入：校正后的 4D DWI
    bvals="derivatives/dwi/sub-01_desc-preproc_dwi.bval",  # 输入：每卷 b 值
    bvecs="derivatives/dwi/sub-01_desc-eddyRotated_dwi.bvec",  # 输入：旋转后的方向
    freesurfer_subject_dir="freesurfer/sub-01",  # 输入：已完成的 recon-all 目录
    atlas="fs-aparc",  # 输入：84 节点 Desikan atlas；生成 DWI 网格标签
    brain_mask=None,        # 输入：None 时从 LAS DWI mean b0 运行包内 BET
    n_seeds=10_000,         # 输入：尝试的流线种子数
    shell_bvals=None,       # 输入：None 时从 b 值聚类 shell
    response_mask=None,     # 输入：None 时使用 DWI 默认响应掩膜
    fod_mask=None,          # 输入：None 时对 BET 掩膜膨胀两次
    normalise_mask=None,    # 输入：None 时对 BET 掩膜侵蚀两次
    fa_map=None,            # 输入：None 时从 DWI 拟合 FA
    dwi_to_t1_world=None,   # 输入：None 时运行 TorchFLIRT 6DOF/normmi
    seed=0,                # 输入：PyTorch 随机种子
    compile_arc=False,     # 输入：True 时首次编译 CUDA 圆弧概率核；默认直接执行
)
count = result.matrices["count"]  # 输出：nodes 定义行列的 84×84 计数矩阵
```

| 参数 | 输入约定 |
|---|---|
| `dwi`、`bvals`、`bvecs` | 已校正 float32 DWI NIfTI `[X,Y,Z,N]`；逐 volume 对应的 bval、eddy-rotated FSL bvec |
| `freesurfer_subject_dir`、`atlas` | 推荐入口：已完成的同被试 `recon-all` subject 目录，含 `mri/brain.mgz` 与 `mri/aparc+aseg.mgz`；默认 `atlas="fs-aparc"` 映射为 1–84，另有原 UKB 七套皮层+Tian 图谱。输出 atlas 重采样到 DWI 体素网格。也可显式提供 `t1_brain`、`t1_segmentation`、`atlas_dwi` 三项以运行既有参考条件 |
| `brain_mask` | 可选 DWI 网格二值 BET 掩膜；LAS DWI 省略时，用双精度累加 mean b0、MRtrix 写头体素尺寸规则及 PyTorch BET 自动生成；非 LAS DWI 须提供掩膜 |
| `n_seeds`、`seed` | 播种尝试数必选；PyTorch 随机数序列与 MRtrix 不同。原脚本 10,000,000 次尝试的耗时和内存未在此验证 |
| `compile_arc` | 可选 bool，默认 `False`；`True` 时需要 CUDA，首次运行包含 PyTorch Inductor 编译时间；输出仍为同结构的 `Tractogram`。数值舍入会影响随机接受边界，详情见[真实 100k 对照](../../validation/connectome/ds004666/tracking_compile_20260929.md) |
| `shell_bvals` | 可选；固定 MRtrix response 文件中的 shell 标签。省略时从 bval 聚类 |
| `response_mask`、`fod_mask`、`normalise_mask` | 可选；固定参考阶段掩膜；默认响应掩膜由 DWI 的 `dwi2mask_legacy` 计算；FOD/归一化仍分别使用 `brain_mask` 的六邻域两轮膨胀/侵蚀 |
| `fa_map` | 可选；同网格已计算 FA（如 UKB `dti_FA`）；省略时从 DWI 拟合 MRtrix 风格 tensor FA |
| `dwi_to_t1_world` | 可选 `4×4` DWI RAS-mm → T1 RAS-mm 矩阵；省略时运行包内 6-DOF/normmi TorchFLIRT。FSL scaled-mm `.mat` 不能直接传入 |

`ConnectomeResult.matrices` 的四张对称 `K×K` 矩阵以 `region_labels` 映射行列；未分配流线不计入，自连接保留：


| 键 | 结构与单位 |
|---|---|
| `count` | int64，双端被分配的流线条数 |
| `sift2_fbc` | float32，逐流线 SIFT2 权重之和 |
| `mean_length` | float32，SIFT2 加权的边均值，mm |
| `mean_fa` | float32，SIFT2 加权的边均值，无量纲 |

自动 DWI→T1 配准向 TorchFLIRT 传入 NiBabel 图像，不导入 Surfa。公开样本的[固定轨迹配准敏感性实验](../../validation/connectome/ds004666/registration_sensitivity_20260929.md)表明，FSL 与 TorchFLIRT 两份变换生成的 84 节点 DWI atlas 相差 2,205 / 778,752 个体素；同一批流线只更换 atlas 时，count 矩阵上三角相关 0.9986、相对 L1 0.0157。各自重跑追踪时流线群体也会改变，应以分离实验解释最终矩阵差异。

其余结果字段包括 `nodes`（84 个 `index/original_label/hemisphere/name` 行）、`atlas`/`atlas_affine`、`five_tissue`/`five_tissue_affine`、`gmwmi`、`wm_sh`、`fa`、`brain_mask`、`tractogram`、`sift2_weights`、`dwi_affine` 与 `dwi_to_t1_world`。归一化 WM FOD 为 float32 `[X,Y,Z,45]`；5TT 是 cGM/sGM/WM/CSF/path 顺序的 float32 `[A,B,C,5]`，GMWMI 为同一 T1 网格 `[A,B,C]`，其 affine 映射到 DWI RAS 世界毫米。`tractogram.paths` 按流线顺序保存各 `[Pi,3]` 世界毫米坐标，`endpoints` 为 `[T,2,3]`，`lengths_mm` 和精确采样 `mean_fa` 为 `[T]`；`sift2_weights` 是同序 float64 `[T]`。

## 命令行与输出

```bash
# --dwi：已校正的四维 DWI；--bvals：逐体积 b 值。
# --bvecs：eddy 旋转后的梯度方向。
# --freesurfer-subject-dir：已完成 recon-all 的 subject 目录。
# --atlas：由 aparc+aseg 自动生成的 fs-aparc 84 节点 atlas。
# --output-dir：结果目录。
# --device：PyTorch 设备；--n-seeds：播种次数；--seed：随机种子。
# --compile-arc：可选；CUDA 上首次编译 iFOD2 圆弧概率核，省略时默认关闭。
fnit connectome \
  --dwi derivatives/dwi/sub-01_desc-preproc_dwi.nii.gz \
  --bvals derivatives/dwi/sub-01_desc-preproc_dwi.bval \
  --bvecs derivatives/dwi/sub-01_desc-eddyRotated_dwi.bvec \
  --freesurfer-subject-dir freesurfer/sub-01 \
  --atlas fs-aparc \
  --output-dir derivatives/fnit_connectome/sub-01 \
  --device cuda:0 --n-seeds 10000 --seed 0
```

Schaefer200 与 Tian S1 的一键入口使用同一套 DWI、追踪和矩阵计算，增加原始 UKB 模板及 SynthMorph 配准输入：

```bash
DWI=derivatives/dwi/sub-01_desc-preproc_dwi.nii.gz # 已完成 TOPUP/eddy 的四维 DWI
BVALS=derivatives/dwi/sub-01_desc-preproc_dwi.bval # 与 DWI 体积逐项对应的 b 值
BVECS=derivatives/dwi/sub-01_desc-eddyRotated_dwi.bvec # eddy 旋转后的方向
SUBJECT=freesurfer/sub-01 # 已完成 recon-all 的受试者目录
ATLAS_TEMPLATES=data/UKB-connectomics/atlases # 原 UKB Schaefer 双半球注释、Tian S1 和标签名
FSAVERAGE=data/fsaverage # 含双半球 sphere.reg 的 fsaverage 目录
MNI_TEMPLATE=data/MNI152_T1_2mm.nii.gz # 与 Tian S1 同体素网格的 MNI T1
WEIGHTS=data/fnit-weights # fnit-setup-weights --model synthmorph-joint 获取的权重目录
OUTPUT=derivatives/fnit_connectome/sub-01_schaefer200_tian_s1 # 输出目录

fnit connectome \
  --dwi "$DWI" --bvals "$BVALS" --bvecs "$BVECS" \
  --freesurfer-subject-dir "$SUBJECT" \
  --atlas schaefer200+tian-s1 \
  --atlas-templates-dir "$ATLAS_TEMPLATES" \
  --fsaverage-dir "$FSAVERAGE" \
  --mni-template "$MNI_TEMPLATE" \
  --synthmorph-weights "$WEIGHTS" \
  --n-seeds 10000 --seed 0 --device cuda:0 \
  --output-dir "$OUTPUT"
```

`--atlas` 指定 200 个 Schaefer 皮层节点与 16 个 Tian S1 节点；`--n-seeds` 是 GMWMI 播种尝试次数，`--seed` 固定 FNIT 随机序列，`--device` 选择推理与追踪设备。`--output-dir` 中四矩阵均为 216×216，`nodes.tsv` 逐行说明节点编号、来源标签、侧别和名称。当前完整命令仅在公开真实数据上做接口和结果结构验收；随机追踪的数值一致性以独立多种子报告为准。

同一命令可将 `--atlas` 改为 `schaefer500+tian-s4` 或 `schaefer1000+tian-s4`。两者分别读取原 UKB Schaefer500/1000 双半球注释、Tian S4 NIfTI 与 54 行名称表；`--atlas-templates-dir` 必须包含这些文件。对应 `nodes.tsv` 和四张矩阵分别应有 554 或 1054 行/列。两种扩展图谱已完成真实 T1 体积与 DWI 标签网格检查，尚未分别完成扩散至四矩阵的一键实测。[输入、输出、原版命令、时间与体素精度](../../validation/connectome/ds004666/atlas_schaefer_multi_20260929.md)单列说明。

`--atlas aparc+tian-s1` 和 `--atlas aparc.a2009s+tian-s1` 直接读取 recon-all 的原生双半球 `.annot`，无需 `--fsaverage-dir`；仍需 `--atlas-templates-dir` 提供 Tian S1 图像与名称表，以及 `--mni-template` 或 `--tian-fnirt-coeff`。对应节点数分别为 84 和 164。真实 T1 的原版皮层体积 XOR 均为 0，两种 DWI 图谱的所有节点均有体素；完整四矩阵尚未分别一键实测，详见[原生图谱报告](../../validation/connectome/ds004666/atlas_native_aparc_20260929.md)。

`--atlas glasser+tian-s1` 或 `--atlas glasser+tian-s4` 读取 `--atlas-templates-dir` 中的原 UKB Glasser 32k dlabel、相邻 `surfaces` 目录的 fsLR/fsaverage 球面和 Tian S1/S4 图像及名称，再使用 `--fsaverage-dir` 的双半球 `sphere.reg`。环境中的 `wb_command` 完成 32k→164k 标签重采样，余下表面和 T1 投影由 PyTorch 完成。其余参数与上面的 Schaefer 一键命令相同；`--mni-template` 使用 SynthMorph T1→MNI 路线，或以 `--tian-fnirt-coeff` 固定原流程变换。输出应为 376×376 或 414×414 四矩阵及同数目 `nodes.tsv`；公开数据已有真实 T1/DWI 图谱验证，尚未分别完成四矩阵一键实测。函数用法、参考命令、时间、逐体素精度和切片见[Glasser 报告](../../validation/connectome/ds004666/atlas_glasser_20260929.md)。

上面的 SynthMorph 路线与原 UKB 使用的 FNIRT 是两种配准方法。同一 UKB T1 的 Tian S1 对照中，前景 Dice 为 0.906，不能声称与原 FNIRT atlas 一致。已有该 T1 的 FNIRT `--cout` 前向 coefficient 时，可把 `--mni-template "$MNI_TEMPLATE" --synthmorph-weights "$WEIGHTS"` 两项替换为 `--tian-fnirt-coeff "$FNIRT_COEFF"`；`FNIRT_COEFF` 是与 `SUBJECT` 对应的 T1→MNI coefficient NIfTI。FNIT 自己计算逆场和最近邻标签，不在正式路径调用 FSL。两个参数路径只能选其一。函数及配对精度、时间见[图谱实测报告](../../validation/connectome/ds004666/atlas_synthmorph_20260929.md)。

输出是 `connectome_count.csv`、`connectome_sift2_fbc.csv`、`connectome_mean_length.csv`、`connectome_mean_fa.csv`，以及 `atlas_dwi.nii.gz`、`five_tissue_dwi_world.nii.gz`、`gmwmi_dwi_world.nii.gz`、`fa_dwi.nii.gz`、`brain_mask_dwi.nii.gz`、`region_labels.csv`、`nodes.tsv`、`dwi_to_t1_world.csv`。矩阵 CSV 无表头；第 i 行对应 `nodes.tsv` 中 index 为 i 的节点。显式自定义 atlas 模式保留 `region_labels.csv`，不产生 `nodes.tsv`。5TT/GMWMI NIfTI 保留 T1 分割网格，affine 表示已映射到 DWI 世界坐标；文件名中的 `dwi_world` 不表示重采样到了 DWI 体素网格。默认拒绝覆盖现有输出，`--overwrite` 可重跑但始终拒绝覆盖输入文件。`--brain-mask`、`--shell-bvals`、`--response-mask`、`--fod-mask`、`--normalise-mask`、`--fa-map` 和 `--dwi-to-t1-world` 可固定对应的参考条件。

## 阶段函数的张量约定

所有影像张量使用所选 CPU/CUDA 设备；FOD 和 FA 与 DWI 同网格，5TT/GMWMI 可保留独立 T1 网格。仿射表示 voxel center → RAS 世界毫米，梯度方向遵循 MRtrix 导出的梯度坐标约定。

| 函数 | 主要输入 → 输出 |
|---|---|
| `mean_bzero` / `mrtrix_roundtrip_voxel_size` / `bet_mask` | 4D DWI 与 b 值 → float32 mean b0；原始体素尺寸 → MRtrix 往返尺寸；mean b0 与尺寸 → bool 脑掩膜。参数、输出与原版命令见 [BET 专页](BET_B0_OPERATORS.md) |
| `dwi2mask_legacy` | float32 DWI `[X,Y,Z,N]`、对应 b 值 `[N]`、shell 均值 `[S]` → 同设备 bool `[X,Y,Z]`；原版默认响应掩膜，详见[实测报告](../../validation/connectome/default_dwi_mask_stage_20260927.md) |
| `estimate_mrtrix_dhollander` | 原始幅值 float32 DWI `[X,Y,Z,N]`、MRtrix 梯度 `[N,4]`、shell `[S]`、bool 掩膜 `[X,Y,Z]` → shell 标签、float64 WM `[S,6]`、GM/CSF `[S,1]` 响应及组织掩膜/FA 诊断 |
| `fit_mrtrix_msmt_csd` | 同一 DWI/梯度/响应与 bool FOD 掩膜 → float32 WM SH `[X,Y,Z,45]`、GM/CSF `[X,Y,Z]`；默认 WM lmax=8 |
| `normalise_mrtrix_three_tissue` | 三组织原始 FOD、bool 掩膜、DWI affine → `MTNormaliseResult`：归一化三组织、bias field、接受掩膜及组织平衡系数 |
| `FreeSurferSubject` / `fs_aparc_atlas` | 完成的 recon-all 目录 → `brain.mgz`、`aparc+aseg.mgz` 路径；整数分割 `[A,B,C]` → 连续节点标签 `[A,B,C]` 与 84 行 `ConnectomeNode`；[同输入实测](../../validation/connectome/fs_aparc84_subject_dir_20260928.md) |
| `schaefer_to_t1` / `native_annotation_to_t1` / `glasser_to_t1` / `synthmorph_tian_to_t1` / `fnirt_tian_to_t1` / `combine_cortical_tian` | fsaverage Schaefer 注释、recon-all 原生 aparc 注释或 Glasser fsLR dlabel 与球面 → 皮层 T1 标签及节点表；T1 脑图、MNI 模板和 Tian 标签 → SynthMorph 原生标签及可复用形变；已给定 FNIRT 前向 coefficient → 原流程原生标签；同网格皮层/Tian 标签与名称 → 连续整数合并 atlas；各参数、原版命令及[配准实测](../../validation/connectome/ds004666/atlas_synthmorph_20260929.md)、[原生图谱实测](../../validation/connectome/ds004666/atlas_native_aparc_20260929.md)、[Glasser 实测](../../validation/connectome/ds004666/atlas_glasser_20260929.md) |
| `freesurfer_five_tissue` / `gmwmi_from_five_tissue` | 官方 FreeSurfer 整数标签 `[A,B,C]` → float32 5TT `[A,B,C,5]` → GMWMI `[A,B,C]` |
| `sample_gmwmi_seeds` | 同网格 float32 GMWMI `[A,B,C]`、5TT `[A,B,C,5]`、体素→RAS 毫米仿射 `[4,4]`、返回位置数、PyTorch generator → 同设备 float32 `[N,3]` 世界毫米位置；[真实 100k 播种分布、命令、时间和图](../../validation/connectome/ds004666/gmwmi_seed_100k_20260929.md) |
| `tracking_sh_precomputed` | 非零 float32 方向 `[...,3]`、偶数 `lmax` → 同设备 float32 球谐函数值 `[...,C]`；参数、等价原版运算及真实单弧基准见[专项报告](../../validation/connectome/ds004666/ifod2_single_arc_20260929.md) |
| `probabilistic_tractography` | 归一化 WM SH/affine、5TT/affine、GMWMI、播种次数，以及可选 5TT 头文件体素尺寸 `five_tissue_spacing_mm=(sx,sy,sz)` → `Tractogram` 的世界毫米流线、端点、长度和已接受种子；每步最多 1,000 次校准拒绝采样、ACT 整数深度状态；可选 `compile_arc=True` 编译 CUDA 圆弧核，默认关闭。详见[同输入 10k 精度](../../validation/connectome/ds004666/ifod2_rejection_20260929.md)、[100k 四矩阵和轨迹分布](../../validation/connectome/ds004666/tracking_100k_matrices_20260929.md)与[编译核精度、时间和图](../../validation/connectome/ds004666/tracking_compile_20260929.md) |
| `estimate_sift2_weights` | 同序流线、WM SH/affine、5TT/affine、`step_size_mm` → float64 逐流线权重 `[T]` |
| `sample_streamline_mean_precise` | 同序流线、float32 FA `[X,Y,Z]`、DWI affine → float32 沿轨迹均值 `[T]` |
| `build_connectomes` | 端点 `[T,2,3]`、整数 atlas/affine、逐轨权重/长度/FA、可选 `node_count` → 四张对称 `[K,K]` 矩阵；`node_count` 保留没有体素的末尾节点零行列 |

## 原 UKB 脚本、ds004666 适配参考与当前实现

| 阶段 | 原 UKB 脚本 | 公开 ds004666 固定输入 MRtrix 参考 | 当前 PyTorch |
|---|---|---|---|
| 输入/掩膜 | UKB `data_ud` + 旋转梯度；BET b0 掩膜；Dhollander 不传 `-mask` 时由 DWI 自建掩膜 | FSL TOPUP/EDDY 校正 AP-DWI；已有阶段对照使用固定 SynthSeg 脑掩膜 | 调用方提供校正 DWI、旋转梯度；LAS DWI 默认运行 PyTorch BET，也可固定外部 BET 掩膜；默认响应掩膜从 DWI 计算，FOD/归一化使用 BET 掩膜的六邻域两轮膨胀/侵蚀 |
| 响应与 FOD | `dwi2response dhollander` → `dwi2fod msmt_csd` → `mtnormalise` | 同序；默认 WM `lmax=8`；固定参考掩膜 | `estimate_mrtrix_dhollander`、`fit_mrtrix_msmt_csd`、`normalise_mrtrix_three_tissue`；响应/FOD 的 float64 约束求解输出 float32 SH |
| T1/5TT/配准 | FreeSurfer 7.1 + FIRST；`5ttgen freesurfer -first -nocrop -sgm_amyg_hipp`、`5tt2gmwmi`；6-DOF/normmi FLIRT | 官方 FreeSurfer 8.2 `recon-all` 外置；安装的 MRtrix 3.0.3 无 `-first`，用 `5ttgen freesurfer -nocrop -sgm_amyg_hipp`；校正 b0 重新 FLIRT | 读取官方 `aparc+aseg.mgz` 构建 5TT/GMWMI；可固定变换或运行 TorchFLIRT 6-DOF/normmi；不重做 recon-all/FIRST |
| 流线/SIFT2 | `tckgen -seed_gmwmi -act -seeds N -select 0 -maxlength 250 -cutoff 0.1 -samples 3 -power 0.5`；`tcksift2 -act` | 同参数，10,000 次播种；固定 FOD/5TT/同 atlas 的阶段验证 | 连续初始方向、校准拒绝采样和 ACT 逐点状态已有独立 PyTorch 实现；`arc_proposals=16` 是每轮 GPU 并行数，单弧仍最多试 1,000 次。SIFT2、FA 和矩阵按现有函数执行；随机流线群体的矩阵误差仍见当前报告 |
| atlas/矩阵 | 皮层 parcellation + Tian 亚皮层图；`tck2connectome -symmetric -assignment_radial_search 4`；原脚本另采样 MD、MO、S0、NODDI 等 | 固定相同 20 区 SynthSeg 示例 atlas；只比较四张矩阵 | `fs-aparc` 可从 recon-all 目录自动构建 84 节点 atlas；四张矩阵，长度与精确沿程 FA；不提供原脚本所有扩展指标 |

[原追踪脚本](https://github.com/sina-mansour/UKB-connectomics/blob/main/scripts/bash/probabilistic_tractography_native_space.sh)、[原矩阵脚本](https://github.com/sina-mansour/UKB-connectomics/blob/main/scripts/bash/map_structural_connectivity.sh)、[校正参考命令](../../validation/connectome/ds004666/corrected_mrtrix_commands.public.txt)和[FreeSurfer 适配命令](../../validation/connectome/ds004666/corrected_mrtrix_fs5tt_act_adapted/commands.public.txt)给出实际调用。公开样本的[七套图谱固定/独立 100k 矩阵对照](../../validation/connectome/ds004666/seven_atlas_100k_20260929.md)已完成；原 UKB 的 FIRST、FNIRT Tian 标签和 1,000 万次播种整链尚未逐项复跑。

以下是对应阶段的参考命令骨架；实际 ds004666 参数、线程数、路径和校正条件以[校正参考命令清单](../../validation/connectome/ds004666/corrected_mrtrix_commands.public.txt)与[FreeSurfer 适配清单](../../validation/connectome/ds004666/corrected_mrtrix_fs5tt_act_adapted/commands.public.txt)为准：

```bash
mrconvert corrected_dwi.nii.gz corrected.mif -fslgrad eddy_rotated.bvec dwi.bval
dwi2response dhollander corrected.mif wm.txt gm.txt csf.txt
maskfilter brain_mask.mif dilate fod_mask.mif -npass 2
dwi2fod msmt_csd corrected.mif wm.txt wm_fod.mif gm.txt gm.mif csf.txt csf.mif -mask fod_mask.mif
maskfilter brain_mask.mif erode norm_mask.mif -npass 2
mtnormalise wm_fod.mif wm_norm.mif gm.mif gm_norm.mif csf.mif csf_norm.mif -mask norm_mask.mif
5ttgen freesurfer aparc+aseg.mgz 5tt_t1.mif -nocrop -sgm_amyg_hipp
5tt2gmwmi 5tt_t1.mif gmwmi_t1.mif
flirt -in b0_brain.nii.gz -ref t1_brain.nii.gz -cost normmi -dof 6 -omat diff2struct_fsl.txt
tckgen -algorithm iFOD2 -seed_gmwmi gmwmi_dwi.mif -act 5tt_dwi.mif -seeds 10000 -select 0 -maxlength 250 -cutoff 0.1 -samples 3 -power 0.5 wm_norm.mif tracks.tck
tcksift2 tracks.tck wm_norm.mif weights.txt -act 5tt_dwi.mif
tcksample -precise -stat_tck mean tracks.tck fa.mif mean_fa.txt
tck2connectome -symmetric -assignment_radial_search 4 tracks.tck atlas_dwi.nii.gz count.csv
```

最后三张矩阵分别在 `tck2connectome` 中增加 `-tck_weights_in weights.txt`；mean length/FA 还分别用 `-scale_file lengths.txt` 或 `-scale_file mean_fa.txt` 及 `-stat_edge mean`。`lengths.txt` 来自 `tckstats -dump`。固定真实流线的精确赋值报告检查了这些规则。

## 固定输入阶段证据

各行使用同一 ds004666 图像或同一官方中间输出，但**不是同一次端到端运行**。PyTorch 已载入张量的核心时间与 MRtrix/FSL 独立进程墙钟（含启动和 I/O）边界不同；不能把行间时间相加为整链加速比。

| 固定输入阶段 | 当前配对数值结果 | 时间与报告 |
|---|---|---|
| mean b0 与 `bet -R` | 公开 ds004666 及匹配 UKB 各 778,752 个 mean b0 值逐值一致，两个最终掩膜均 XOR 0；CPU/GPU 均核验 | 公开原版 MRtrix/FSL 0.65/9.46 s；FNIT H100 mean/BET 0.076/7.769 s、峰值 0.354 GiB；[参数、同输入报告与脑图](BET_B0_OPERATORS.md) |
| 默认响应掩膜 `dwi2mask legacy` | 相同真实 UKB DWI 掩膜 XOR 0 / 778,752；公开 ds004666 另测 XOR 0 / 778,752 | 原版 13.45 / 17.50 s，FNIT CPU 完整进程 4.73 / 5.33 s；[同输入报告与脑图](../../validation/connectome/default_dwi_mask_stage_20260927.md) |
| 原 UKB 默认 Dhollander 响应 | 11 个选择掩膜 XOR 0；WM/GM/CSF 响应系数最大误差 6.28e−11 / 2.55e−10 / 8.73e−11 | 原版完整命令 27.04 s；当前版 FNIT CPU 8 线程核心 3.197 s；[同输入报告](../../validation/connectome/original_ukb_dhollander_stage_20260927.md) |
| 原 UKB 全脑 MSMT-CSD | 212,831 个掩膜体素；WM 9,577,395 个系数最大误差 2.98e−8，GM/CSF 最大误差 2.84e−14 / 9.31e−10；全部小于 1e−7 | H100 求解 269.457 s、峰值已分配 1.354 GiB；MRtrix CPU 完整命令 107.04 s；[同输入报告](../../validation/connectome/original_ukb_fod_stage_20260927.md) |
| 原 UKB 掩膜形态学与 mtnormalise | 官方 BET 掩膜两次膨胀／侵蚀 XOR 均 0；WM/GM/CSF 归一化全图最大误差 1.19e−7 / 5.96e−8 / 2.98e−8 | H100 核心 0.650 s、0.461 GiB；MRtrix CPU 完整命令 2.11 s；[同输入报告](../../validation/connectome/original_ukb_mtnormalise_stage_20260927.md) |
| 六邻域掩膜、官方 FreeSurfer 5TT/GMWMI | 两轮膨胀/侵蚀各 XOR 0；256³×5 5TT 和 GMWMI 逐值 0 不一致 | [maskfilter](../../validation/connectome/ds004666/maskfilter_real.public.json)、[解剖/配准](../../validation/connectome/ds004666/ANATOMY_STAGE_20260927.md)；FreeSurfer recon-all 在包外耗时 1.645 h |
| Dhollander 响应（公开 ds004666） | 11/11 选择掩膜 XOR 0；WM/GM/CSF 最大误差 3.89e−9 / 9.06e−10 / 1.24e−10 | 当前版 CPU 核心 3.748 s；MRtrix 命令 17.01 s；[报告](../../validation/connectome/ds004666/response_fod_stage_20260927.md) |
| 全脑原始 MSMT-CSD（公开 ds004666） | 208,522 体素、9,383,490 个 WM SH 值，MAE 1.98e−12，最大 1.21e−6；尚未 mtnormalise | 当前版 H100 核心 267.419 s、batch_size=4096、峰值 1.253 GiB；MRtrix CPU 命令 185.51 s；[报告](../../validation/connectome/ds004666/response_fod_full.public.json) |
| 三组织 mtnormalise | 固定官方原始 FOD，WM SH 7,967,115 元素 MAE 4.79e−10、最大 5.96e−8 | H100 核心 0.568 s；MRtrix 命令 2.41 s；[报告与图](../../validation/connectome/ds004666/mtnormalise_stage_20260927.md) |
| GMWMI 播种与 iFOD2/ACT 追踪 | 100k GMWMI 位置的 8 mm 三维直方图相关 FNIT 对官方 0.92689、官方两次 0.92458；真实 FOD 校准方向 7/7，拒绝乘数绝对误差 `1.96×10⁻⁵`；80 条固定圆弧正／零概率判定一致；ACT 12,600 个逐点状态无分歧。10,000 固定种子的 FNIT 三次接受 2,713/2,727/2,691 条，官方 TCK 三次为 2,758/2,728/2,727 条；四矩阵 3×3 比较尚有差异 | 播种 H100 核心 2.64 s、Torch 峰值 0.461 GiB；官方 Seedtest 1.26–1.36 s。[播种报告与图](../../validation/connectome/ds004666/gmwmi_seed_100k_20260929.md)；追踪 H100 核心 66.25/69.96/72.20 s、峰值约 2.176 GiB，[报告](../../validation/connectome/ds004666/ifod2_rejection_20260929.md) |
| SIFT2 固定 FOD/5TT/官方 TCK | 处理掩膜 285/778,752 体素不同，非零支持 Dice=1；fixel 总数两臂均 243,822；全 fixel TDI r=0.9999999371；固定 2,758 条轨迹最终权重 r=0.999999903、MAE 3.44e−5 | [FMLS/处理掩膜图](../../validation/connectome/ds004666/sift2_fmls_stage_20260927.md)、[轨迹映射图](../../validation/connectome/ds004666/sift2_mapping_stage.md)、[优化器报告](../../validation/connectome/ds004666/sift2_optimizer_fmls_exact.public.json) |
| 固定 2,758 条轨迹精确 FA 均值 | 原 MIF 几何逐轨 r=0.9999999949、MAE 5.58e−7，最大 0.0005499 | CPU 首轮 0.124 s、后续中位 0.102 s；MRtrix 命令 0.03 s；[报告与图](../../validation/connectome/ds004666/tcksample_precise_stage.md) |
| 固定真实轨迹的四矩阵赋值 | 2,915 条双端分配一致；count 400/400 元素完全一致，FBC/长度/FA 最大误差 8.99e−6 / 5.63e−6 mm / 2.96e−8 | [赋值报告](../../validation/connectome/ds004666_fsl_act_real_tracks_assignment_report.json) |

[84 节点 atlas、固定 TCK 与新入口 10,000 次播种报告](../../validation/connectome/fs_aparc84_subject_dir_20260928.md)给出同一公开 DWI 和配对 UKB 的新增验证。

![相同 FreeSurfer T1 的官方与 FNIT 84 节点 atlas](../../validation/connectome/ds004666/figures/fs_aparc84_comparison.png)

![校正输入的响应与 FOD 阶段切片](../../validation/connectome/ds004666/response_fod_example.png)

![相同 FOD 的 SIFT2 处理掩膜与 fixel 数](../../validation/connectome/ds004666/sift2_fmls_proc_mask_comparison.png)

![当前真实 FOD 拒绝概率和 20 节点矩阵](../../validation/connectome/ds004666/ifod2_rejection_20260929/rejection_and_connectome.png)

![当前三次 FNIT 与三次官方运行的四矩阵误差](../../validation/connectome/ds004666/ifod2_rejection_20260929/rejection_act_3x3.png)

当前 10,000 固定位置的四矩阵 3×3 比较：count 上三角相对 L1 跨软件 `0.230–0.293`，官方自身 `0.228–0.258`；共同边 mean FA 归一化误差跨软件 `0.064–0.104`，官方自身 `0.078–0.101`。count 有 4/9、mean FA 有 6/9 对进入官方重复范围；其余配对和支持率仍有差异。

## 复跑阶段与整链比较

[公开 ds004666 清单](../../validation/connectome/ds004666/download_manifest.tsv)记录 OpenNeuro T1、AP/PA DWI 地址、大小和 SHA-256。完整中间数据体积较大，NIfTI、TCK、官方 FreeSurfer 输出及原始 SIFT2 调试文件留在验证机器；Git 保留脚本、矩阵 CSV、报告和图像。按各阶段报告准备配对图像后运行对应脚本的 `--help`：

| 阶段 | 脚本入口 |
|---|---|
| 响应、FOD、掩膜 | [`benchmark_connectome_response_fod_dhollander.py`](../../tools/benchmark_connectome_response_fod_dhollander.py)、[`benchmark_connectome_response_fod.py`](../../tools/benchmark_connectome_response_fod.py)、[`benchmark_connectome_maskfilter.py`](../../tools/benchmark_connectome_maskfilter.py) |
| mtnormalise、解剖、追踪 | [`benchmark_connectome_mtnormalise.py`](../../tools/benchmark_connectome_mtnormalise.py)、[`benchmark_connectome_anatomy.py`](../../tools/benchmark_connectome_anatomy.py)、[`benchmark_connectome_tracking_act.py`](../../tools/benchmark_connectome_tracking_act.py) |
| iFOD2 单弧球谐函数 | [`benchmark_connectome_ifod2_single_arc.py`](../../tools/benchmark_connectome_ifod2_single_arc.py)；输入/输出结构及官方源码观测程序见[单弧报告](../../validation/connectome/ds004666/ifod2_single_arc_20260929.md) |
| iFOD2 初始方向 | [`benchmark_connectome_ifod2_initial_directions.py`](../../tools/benchmark_connectome_ifod2_initial_directions.py)；官方观测程序、输入/输出、时间及真实 FOD 分布见[初始方向报告](../../validation/connectome/ds004666/ifod2_initial_direction_20260929.md) |
| ACT 种子检查 | [`benchmark_connectome_ifod2_act_seed.py`](../../tools/benchmark_connectome_ifod2_act_seed.py)；同一真实 5TT、10,000 个冻结种子与官方 C++ 观测程序的数值、耗时见[ACT 报告](../../validation/connectome/ds004666/ifod2_act_seed_20260929.md) |
| iFOD2 校准拒绝采样与 ACT 逐点状态 | [`benchmark_connectome_ifod2_calibration.py`](../../tools/benchmark_connectome_ifod2_calibration.py)、[`benchmark_connectome_ifod2_rejection.py`](../../tools/benchmark_connectome_ifod2_rejection.py)、[`benchmark_connectome_act_state.py`](../../tools/benchmark_connectome_act_state.py)、[`benchmark_connectome_ifod2_rejection_matrices.py`](../../tools/benchmark_connectome_ifod2_rejection_matrices.py)；当前数值、时间、3×3 四矩阵与图见[当前报告](../../validation/connectome/ds004666/ifod2_rejection_20260929.md) |
| SIFT2、FA、矩阵 | [`benchmark_sift2_processing_mask.py`](../../tools/benchmark_sift2_processing_mask.py)、[`benchmark_sift2_fixels.py`](../../tools/benchmark_sift2_fixels.py)、[`benchmark_connectome_sift2_mapping.py`](../../tools/benchmark_connectome_sift2_mapping.py)、[`benchmark_connectome_sift2_optimizer.py`](../../tools/benchmark_connectome_sift2_optimizer.py)、[`benchmark_connectome_tcksample_precise.py`](../../tools/benchmark_connectome_tcksample_precise.py)、[`compare_connectome_matrices.py`](../../tools/compare_connectome_matrices.py) |

整链脚本 [`benchmark_connectome_end_to_end.py`](../../tools/benchmark_connectome_end_to_end.py)要求固定同一校正 DWI、旋转梯度、官方 aparc+aseg、T1 brain、atlas、脑掩膜、DWI→T1 RAS-mm 变换与参考矩阵目录；可再传响应/FOD/归一化掩膜、FA 和 shell 标签。参考目录接受原始 MRtrix 的 `count.csv` 等文件名，也接受公开归档的 `connectome_count.csv` 等文件名。脚本写出 `candidate_*.csv`、轨迹 TCK、逐轨权重/长度/FA 和带输入 SHA-256、边指标、时间、Torch 峰值显存的 `report.json`。固定真实 FOD/5TT 的当前追踪 3×3 比较见[报告](../../validation/connectome/ds004666/ifod2_rejection_20260929.md)。

```bash
python tools/benchmark_connectome_end_to_end.py --help
python tools/benchmark_connectome_end_to_end.py \
  --dwi corrected_dwi.nii.gz --bvals corrected_dwi.bval \
  --bvecs eddy_rotated.bvec --t1-brain t1_brain.nii.gz \
  --aparc-aseg aparc+aseg.mgz --atlas-dwi atlas_20_dwi.nii.gz \
  --brain-mask brain_mask_dwi.nii.gz --transform dwi_to_t1_world.txt \
  --reference-dir reference_matrices --output-dir candidate_seed0 \
  --n-seeds 10000 --seed 0 --device cuda:0
```

四张矩阵的比较固定脑区顺序，按严格上三角同时报告全边与共同非零边指标；长度/FA 的共同边指标用于区分数值偏差与连接支持差异。固定真实轨迹的赋值一致性不能代替独立追踪后的矩阵一致性。

## 参考文献与原实现

- Tournier JD 等，*MRtrix3: A fast, flexible and open software framework for medical image processing and visualisation*，NeuroImage 202:116137，2019。[论文](https://pubmed.ncbi.nlm.nih.gov/31473352/)
- Smith RE 等，*Anatomically-constrained tractography: improved diffusion MRI streamlines tractography through effective use of anatomical information*，NeuroImage 62:1924–1938，2012。[论文](https://pubmed.ncbi.nlm.nih.gov/22705374/)
- [原 UKB-connectomics 代码库](https://github.com/sina-mansour/UKB-connectomics)、[本次对照的 MRtrix3 源码提交](https://github.com/MRtrix3/mrtrix3/tree/eeab681d3e0cb004cf1d1d31579d3892197ef5b6)、[PyTorch 代码库](https://github.com/pytorch/pytorch)。
