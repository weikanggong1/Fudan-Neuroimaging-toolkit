# 静息态 fMRI：MNI 体积到 fsLR32k 表面

这一页对应 UK Biobank（UKB）表面流程中 **MNI 体积时间序列到 fsLR32k 灰质时间序列**的步骤。输入是已经清理并配准到 MNI152 2 mm 的 4D BOLD，以及同一被试事先生成的 FreeSurfer 白质面、软脑膜面、球面和 `wmparc.mgz`。FNIT 读取这些结构文件，不运行 FreeSurfer 可执行程序。当前实现把 FreeSurfer 球面配准投到 fsLR，使用 Connectome Workbench 做 ribbon 采样、表面重采样、2 mm FWHM 平滑和 CIFTI 组装。没有这些结构文件时，现有 BIDS 体积流程不会凭原始 T1 自动生成表面。

官方 `bb_surf` 在这段处理之后还执行 T2_FLAIR 偏置校正、髓鞘图、MSMSulc、MSMAll、DeDrift、FIX 重应用、双回归和网络 IDP。当前页面的输出采用 **FS sphere 注册**；尚无 MSMSulc/MSMAll/DeDrift，也不把 ICA-AROMA 称作 FIX。因球面注册与清理方式不同，当前 CIFTI 与官方最终 `bb.rfMRI.MSMAll_smooth_2.dtseries.nii` 不具备逐点数值等价性。官方步骤见 [UKB v1.5 `bb_surf`](https://git.fmrib.ox.ac.uk/falmagro/uk_biobank_pipeline_v_1.5/-/blob/0e39a7f7eb76b55437942bfa3073512506b6c8fa/bb_surf_pipeline/bb_surf) 和 [`bb_hcp_surf_mni`](https://git.fmrib.ox.ac.uk/falmagro/uk_biobank_pipeline_v_1.5/-/blob/0e39a7f7eb76b55437942bfa3073512506b6c8fa/bb_surf_pipeline/bb_hcp_surf_mni)。

## 安装与公开模板

主页的 Conda 环境包含 Python 依赖与 `connectome-workbench-cli`。HCP 的公开模板单独下载，固定于 [HCPpipelines v4.7.0](https://github.com/Washington-University/HCPpipelines/tree/f8cac6892f88bdf889d644711ff038198eb81533) 的 commit `f8cac6892f88bdf889d644711ff038198eb81533`，逐文件核对 SHA-256；输出目录同时保存该版本的 `LICENSE.md`。原始站点不可达时，安装器转到同一 commit 的 jsDelivr 地址，校验规则不变。

```bash
python tools/setup_fmri_surface_assets.py \
  --output-dir /absolute/path/hcp_surface_assets
```

`--output-dir` 是保存 HCP 公开球面、atlas ROI、`Atlas_ROIs.2.nii.gz`、`Avgwmparc.nii.gz` 和 FreeSurfer 标签表的**绝对目录**。安装器保留 `global/templates/` 和 `global/config/` 相对路径；再次运行会检查已有文件的 SHA-256。该目录不含被试影像、FreeSurfer 结构结果、HCP group ICA d25/d50 图或 UKB 专用 DeDrift 球面。完整官方 MSMAll 还需要后两类资源；HCP group ICA 数据受 ConnectomeDB 注册与数据使用条款约束，不随 FNIT 安装。

`run_surface_projection` 调用 Workbench 时，若环境中没有 `OMP_NUM_THREADS`，默认给其子进程设置 `min(可见 CPU 数, 8)`；已设置时沿用用户的值。例如 `export OMP_NUM_THREADS=4` 可将本次投影限制为 4 线程。结构准备和 ribbon QC 的 Workbench 命令不由这一投影函数管理。

## 输入

| 输入 | 格式、空间和作用 |
|---|---|
| `clean_mni` | 4D NIfTI，已清理 BOLD；MNI152 2 mm 网格，头文件须记录正数 TR。可用体积流程的 `filtered_func_data_clean_MNI152_2mm.nii.gz`。 |
| `mni_reference` | 3D MNI152 2 mm 参考 NIfTI。`clean_mni`、`goodvoxels`、`subject_rois`、`atlas_rois` 必须与其尺寸和 affine 相同。 |
| `subject_dir` | 已生成的单被试 FreeSurfer 目录。必需 `mri/orig.mgz`、`mri/wmparc.mgz`，以及双侧 `white`、`pial`、`sphere.reg`、`thickness`。`mri/orig/001.mgz` 不能替代 `mri/orig.mgz`，因为表面顶点需按 conformed T1 的 tkRAS 转为 scanner RAS。 |
| `pull_ras` | 体积流程输出的 `reg/MNI152_2mm_to_T1_pull_ras.nii.gz`；MNI 网格上 3 分量 RAS 毫米位移，表示 MNI 世界坐标到 T1 世界坐标的非线性 pull。 |
| `initial_t1_to_mni_world` | 4×4 NumPy 数组，T1 scanner RAS 到 MNI scanner RAS 的初始**世界坐标**仿射；用 `volume_result.t1_to_mni.moving_to_fixed_world`。不能直接把 FLIRT scaled-mm `.mat` 文件读作这个参数。 |
| `hcp_assets_dir` | 上述安装器保存的模板根目录。 |
| `wb_command` | Connectome Workbench 可执行文件名或绝对路径；用于球面操作、距离场、ribbon 投影、表面重采样和 CIFTI。 |

准备函数将 native `white/pial` 从 FreeSurfer tkRAS 转为 T1 scanner RAS，然后逐顶点求解 MNI 非线性 pull 的反变换，生成 MNI scanner RAS 下的 white、pial、midthickness。球面采用现有 `sphere.reg` 到 fsLR 的 FS sphere 对应关系；原生皮层 ROI 从 `abs(thickness)>0` 生成，Workbench 依次执行 `-metric-fill-holes` 和 `-metric-remove-islands`，然后把 164k fsLR atlas ROI 沿注册球面反投到原生网格（`BARYCENTRIC -largest`），与个人 ROI 做并集。`wmparc.mgz` 按同一 pull 最近邻重采样到 MNI 网格，再用 HCP 标签表提取被试皮层下 ROI。本例结构文件缺少官方使用的 `mri/brain.finalsurfs.mgz`；FNIT 以 `orig.mgz` 的 tkRAS 坐标变换定位表面，尚未与官方的 CRAS 坐标逐顶点核对。`make_ribbon_goodvoxels` 用 Workbench 距离场生成皮层 ribbon，并用 BOLD 时间变异系数和局部平滑趋势计算 goodvoxels；时间标准差采用 N−1 样本方差，空间阈值统计只使用 ribbon 内的非零体素且采用 N−1 样本方差，对应 FSL `-Tstd` 和 `fslstats -S`。可传入固定 goodvoxels 掩膜以做同输入比较。

## 子函数接口

下表列出六个公开表面函数的全部参数。`output_dir` 均为本次运行的输出目录；输入影像与结构文件的空间约定见上表。右栏指它在 [UKB `bb_hcp_surf_mni`](https://git.fmrib.ox.ac.uk/falmagro/uk_biobank_pipeline_v_1.5/-/blob/0e39a7f7eb76b55437942bfa3073512506b6c8fa/bb_surf_pipeline/bb_hcp_surf_mni) 中对应的处理位置，不表示不同形变算法逐值相同。

| 函数 | 输入参数 | 返回值和磁盘输出 | 官方对照命令 |
|---|---|---|---|
| `invert_mni_to_t1_pull` | `t1_world_points`：T1 scanner RAS 中的 N×3 毫米坐标；`pull_ras`：MNI→T1 三分量位移；`initial_t1_to_mni_world`：T1→MNI 4×4 世界仿射。可选 `device="cpu"`、`tolerance_mm=0.05`、`max_iterations=80`、`chunk_size=32768`。 | `(mni_world_points, max_inverse_residual_mm)`；N×3 MNI scanner RAS 坐标与最大反解残差；无磁盘文件。 | `invwarp` 后的 `wb_command -surface-apply-warpfield ... -fnirt` 所处的表面形变步骤；FNIT 直接反解自身保存的 pull。 |
| `prepare_mni_surface_geometry` | `subject_dir`：现成 FreeSurfer 结构目录；`pull_ras`、`initial_t1_to_mni_world` 同上；`output_dir`。可选 `device="cpu"`、`tolerance_mm=0.05`、`overwrite=False`。 | `MNISurfaceResult.left/right`：各含 `white`、`pial`、`midthickness` 的 native GIFTI 路径、`vertex_count`、`max_inverse_residual_mm`；文件写入 `output_dir`。 | `mri_info --cras`、`wb_command -surface-apply-affine`、`-surface-apply-warpfield`；当前以 `orig.mgz` 的 tkRAS 变换取代官方 CRAS 输入。 |
| `make_ribbon_goodvoxels` | `clean_bold`：MNI 2 mm 4D BOLD；`reference`：同网格 3D 图；`left_white`、`left_pial`、`right_white`、`right_pial`：MNI scanner RAS GIFTI；`output_dir`。可选 `wb_command="wb_command"`、`neighborhood_sigma_mm=5.0`、`threshold_factor=0.5`。 | `SurfaceQCResult.ribbon`、`.goodvoxels`、`.report`：两个 3D NIfTI 与统计 JSON，位于 `output_dir`。 | `wb_command -create-signed-distance-volume`；`fslmaths -Tmean/-Tstd/-bin/-s/-dilD/-thr` 和 `fslstats -M/-S`。 |
| `prepare_fs_sphere_projection_inputs` | `subject_dir`、`pull_ras`、`initial_t1_to_mni_world`；`mni_reference`：同网格 3D MNI 2 mm；`hcp_assets_dir`：公开 HCP 资产根；`output_dir`。可选 `wb_command="wb_command"`、`device="cpu"`、`overwrite=False`。 | `SurfacePreparationResult.left/right`：各为八条 GIFTI 路径组成的 `SurfaceHemisphere`；`.subject_rois` 为 `ROIs.2.nii.gz`、`.atlas_rois` 为 HCP 标准标签、`.inverse_residual_mm` 为双侧反解误差。其他中间文件见输出树。 | `-surface-sphere-project-unproject`、`-metric-fill-holes/-metric-remove-islands/-metric-resample`、`-volume-label-import`；对应官方 `bb_hcp_surf_mni` 的 FS sphere 和 ROI 段。 |
| `run_surface_projection` | `clean_mni`：4D BOLD；`mni_reference`、`goodvoxels`、`subject_rois`、`atlas_rois`：同一 2 mm 网格 NIfTI；`left`、`right`：已准备的 `SurfaceHemisphere`；`output_dir`。可选 `wb_command="wb_command"`、`overwrite=False`。 | `SurfaceProjectionResult.dtseries`、`.left_metric`、`.right_metric`、`.subcortical_volume`、`.coverage_report`：CIFTI、双侧 GIFTI、4D NIfTI、覆盖 JSON；`.timing_seconds` 为每条 Workbench 命令耗时。 | `-volume-to-surface-mapping -ribbon-constrained`、`-metric-resample ADAP_BARY_AREA`、`-metric-smoothing`、`-cifti-resample`、`-cifti-create-dense-timeseries`。 |
| `run_surface_from_mni` | `clean_mni`、`mni_reference`；`inputs=SurfacePipelineInputs(left, right, subject_rois, atlas_rois, wb_command, goodvoxels)`；`output_dir`。可选 `overwrite=False`。`goodvoxels=None` 表示现场计算。 | `SurfacePipelineResult.projection`：上述投影结果；`.qc`：现场计算时的 `SurfaceQCResult`，传入固定掩膜时为 `None`。在 `output_dir/qc/` 与 `projection/` 写文件。 | 组合上述 goodvoxels 与 Workbench 投影步骤；官方对应 `bb_hcp_surf_mni` 的 ribbon→dense CIFTI 段。 |

单独生成 goodvoxels 时，四张表面必须已经在 MNI scanner RAS，`reference` 与 `clean_bold` 的前三维尺寸和 affine 一致：

```python
from fnit.fmri.surface_qc import make_ribbon_goodvoxels

qc = make_ribbon_goodvoxels(
    clean_bold="/absolute/path/filtered_func_data_clean_MNI152_2mm.nii.gz",  # 4D MNI 2 mm BOLD
    reference="/absolute/path/MNI152_T1_2mm.nii.gz",  # 同网格 3D 参考
    left_white="/absolute/path/lh.white.MNI.native.surf.gii",  # 左侧白质面
    left_pial="/absolute/path/lh.pial.MNI.native.surf.gii",  # 左侧软脑膜面
    right_white="/absolute/path/rh.white.MNI.native.surf.gii",  # 右侧白质面
    right_pial="/absolute/path/rh.pial.MNI.native.surf.gii",  # 右侧软脑膜面
    output_dir="/absolute/path/qc",  # ribbon、goodvoxels 与 JSON 的输出目录
    wb_command="/absolute/path/bin/wb_command",  # Connectome Workbench 程序
    neighborhood_sigma_mm=5.0,  # 变异系数邻域平滑标准差，单位 mm
    threshold_factor=0.5,  # 上限为 ribbon 内均值加 0.5 倍标准差
)
print(qc.ribbon, qc.goodvoxels, qc.report)  # 3D ribbon、3D goodvoxels、标量报告
```

## 从原始 BIDS 一次运行

已有外部生成的 FreeSurfer 结构目录时，可让高层体积入口接续表面准备与投影。以下仅列必要路径和本例选择的参数；其余体积处理选项及默认值见[体积流程](README.md)。这里的 `surface_subject_dir` 是必需的外部结构输入，原始 BIDS T1w 本身不会在 FNIT 中自动生成 white、pial 和 sphere.reg。

```python
from fnit import run_fmri_pipeline

result = run_fmri_pipeline(
    bids_root="/absolute/path/bids",  # 原始 BIDS 数据集根目录，含 BOLD、JSON、T1w
    output_dir="/absolute/path/sub-0001_fmri",  # 本次运行的体积和 surface/ 输出根目录
    subject="0001",  # BIDS 被试标签，不含 sub- 前缀
    mni_template="/absolute/path/MNI152_T1_2mm.nii.gz",  # 3D MNI152 2 mm 参考影像
    mni_brain_mask="/absolute/path/MNI152_T1_2mm_brain_mask.nii.gz",  # 与模板同网格的脑掩膜
    task="rest",  # BIDS task 实体，选择对应 BOLD run
    registration_backend="synthmorph",  # T1→MNI 形变后端；也可用 fnirt
    surface_subject_dir="/absolute/path/subject/FreeSurfer",  # 已生成的同被试结构表面目录
    surface_assets_dir="/absolute/path/hcp_surface_assets",  # SHA-256 校验后的 HCP 模板目录
    wb_command="/absolute/path/bin/wb_command",  # Connectome Workbench 可执行文件
    synthstrip_weights=None,  # 已安装在 FNIT 缓存的 SynthStrip 权重
    synthmorph_weights=None,  # 已安装在 FNIT 缓存的 SynthMorph 权重
    device="cuda:0",  # PyTorch 设备；无 GPU 时可改 cpu
)
print(result.clean_mni)  # MNI152 2 mm 清理后 4D BOLD
print(result.surface.projection.dtseries)  # fsLR32k 皮层与 2 mm 皮层下 CIFTI
```

## Python 调用

先按[体积流程](README.md)运行 `run_fmri_pipeline(...)`，保留其 Python 返回值 `volume_result`。下例从该返回值继续；每个路径改为本机绝对路径。`prepare_fs_sphere_projection_inputs` 和 `run_surface_from_mni` 均一次处理一名被试。

```python
from fnit.fmri.surface_prepare import prepare_fs_sphere_projection_inputs
from fnit.fmri.surface_pipeline import SurfacePipelineInputs, run_surface_from_mni

prepared = prepare_fs_sphere_projection_inputs(
    subject_dir="/absolute/path/subject/FreeSurfer",  # 同被试结构目录，内含 mri/orig.mgz、wmparc.mgz 和双侧 surf
    pull_ras=volume_result.t1_to_mni.pull_ras,  # 体积流程返回的 MNI→T1 非线性 RAS pull
    initial_t1_to_mni_world=volume_result.t1_to_mni.moving_to_fixed_world,  # T1→MNI 世界坐标初始 4×4 仿射
    mni_reference="/absolute/path/MNI152_T1_2mm.nii.gz",  # 与 clean_mni 完全同网格的 3D 参考
    hcp_assets_dir="/absolute/path/hcp_surface_assets",  # 模板安装器的输出根目录
    output_dir="/absolute/path/sub-0001_surface_prepare",  # 表面和 ROI 准备文件的输出目录
    wb_command="/absolute/path/bin/wb_command",  # Connectome Workbench 可执行文件
    device="cuda:0",  # 非线性表面顶点反变换与 wmparc 重采样的 PyTorch 设备
    overwrite=False,  # 已有 native white 文件时是否覆盖
)

surface_inputs = SurfacePipelineInputs(
    left=prepared.left,  # 左侧 MNI native white/pial/midthickness、注册球面和 fsLR32k ROI
    right=prepared.right,  # 右侧对应文件；两侧均为 SurfaceHemisphere
    subject_rois=prepared.subject_rois,  # 同网格被试皮层下标签，通常为 ROIs.2.nii.gz
    atlas_rois=prepared.atlas_rois,  # HCP Atlas_ROIs.2.nii.gz，标准皮层下标签
    wb_command="/absolute/path/bin/wb_command",  # Workbench 可执行文件
    goodvoxels=None,  # None 时按当前 BOLD 与 white/pial 计算；也可传 3D 掩膜绝对路径
)

surface_result = run_surface_from_mni(
    clean_mni=volume_result.clean_mni,  # 体积流程清理后的 4D MNI152 2 mm BOLD
    mni_reference="/absolute/path/MNI152_T1_2mm.nii.gz",  # 与 BOLD、ROI 同网格
    inputs=surface_inputs,  # 上面显式列出的左右表面、ROI、Workbench 与 goodvoxels
    output_dir="/absolute/path/sub-0001_surface",  # qc/ 和 projection/ 的父目录
    overwrite=False,  # 已存在最终 dtseries 时是否覆盖
)
print(surface_result.projection.dtseries)  # fsLR32k 双侧皮层加 2 mm 皮层下的 CIFTI 时间序列
print(surface_result.qc.goodvoxels)  # 自动估计时的 3D goodvoxels；手工提供掩膜时 qc 为 None
```

若已有同空间的 `SurfaceHemisphere` 文件，可用低层函数单独重跑投影。下例固定刚生成的 goodvoxels，适合同输入 Workbench 对照；它不会重新生成掩膜、注册球面或结构 ROI。

```python
from fnit.fmri.surface import run_surface_projection

fixed_mask_result = run_surface_projection(
    clean_mni=volume_result.clean_mni,  # 4D MNI152 2 mm 清理后 BOLD
    mni_reference="/absolute/path/MNI152_T1_2mm.nii.gz",  # 同网格 3D 模板
    goodvoxels=surface_result.qc.goodvoxels,  # 已生成的 3D 采样掩膜
    subject_rois=prepared.subject_rois,  # 被试 MNI 2 mm 皮层下标签
    atlas_rois=prepared.atlas_rois,  # HCP 标准 2 mm 皮层下标签
    left=prepared.left,  # 左侧 MNI native 与 fsLR32k 表面、球面、ROI 路径
    right=prepared.right,  # 右侧对应路径
    output_dir="/absolute/path/sub-0001_surface_fixed_mask",  # 新的投影输出目录
    wb_command="/absolute/path/bin/wb_command",  # Connectome Workbench
    overwrite=False,  # 最终 CIFTI 已存在时是否覆盖
)
print(fixed_mask_result.dtseries)  # 固定掩膜的 CIFTI 结果
```

## 输出结构

```text
sub-0001_surface_prepare/
├── native/lh.white.MNI.native.surf.gii           # MNI scanner RAS，原生顶点顺序
├── native/lh.pial.MNI.native.surf.gii            # 左侧 pial；右侧有对应 rh.*
├── native/lh.midthickness.MNI.native.surf.gii    # white/pial 中点面
├── L.sphere.FS_to_fsLR.native.surf.gii           # FS sphere 到 fsLR 的注册球面
├── L.roi.thickness.native.shape.gii              # abs(thickness)>0 原始 ROI
├── L.roi.filled.native.shape.gii                 # Workbench 填孔后的 ROI
├── L.roi.individual.native.shape.gii             # 再去除孤岛的个人 ROI
├── L.atlasroi.native_projected.shape.gii        # 164k fsLR atlas ROI 反投到原生球面
├── L.roi.native.shape.gii                        # 去孤岛的个人 ROI 与 atlas ROI 并集；右侧同理
├── L.midthickness.32k_fsLR.surf.gii              # fsLR32k 表面；右侧有对应 R.*
├── wmparc.MNI152_2mm.nii.gz                      # 最近邻重采样的整数分区
└── ROIs.2.nii.gz                                 # 被试皮层下标签

sub-0001_surface/
├── qc/ribbon_only.nii.gz                         # 双侧 white 与 pial 之间的 3D ribbon
├── qc/goodvoxels.nii.gz                          # ribbon 质量筛选后的 3D 采样掩膜
├── qc/goodvoxels_report.json                     # ribbon/goodvoxels 数量、阈值及分段耗时
└── projection/
    ├── L.32k_registered_sphere_s2.func.gii      # 左侧顶点×时间，2 mm FWHM
    ├── R.32k_registered_sphere_s2.func.gii      # 右侧顶点×时间，2 mm FWHM
    ├── subcortical_MNI_s2.nii.gz                 # 标准皮层下结构，X×Y×Z×T
    ├── clean_MNI_Atlas_registered_sphere_s2.dtseries.nii  # 时间×灰质单元
    ├── dense_coverage_report.json                # 按结构统计随时间变化的灰质单元
    └── work/                                      # Workbench 中间文件
```

`SurfaceProjectionResult` 返回最终 `dtseries`、左右 `func.gii`、皮层下 4D NIfTI、`coverage_report` 路径及各 Workbench 阶段的墙钟秒数 `timing_seconds`。覆盖报告列出时间帧、TR、总灰质单元数、随时间变化的单元数及各脑结构的对应计数。`SurfacePipelineResult.qc` 返回 ribbon、goodvoxels 和 JSON 路径；传入现成 `goodvoxels` 时为 `None`。CIFTI 中每一帧的时间步长取自输入 BOLD 的 TR。最终灰质单元数量取决于标准皮层 ROI 和皮层下标签，不能只按两个完整 32,492 顶点相加。

## 官方参照命令与差异

下列命令用于官方软件比较。UKB `bb_surf_setup` 根据研究目录准备环境变量；`bb_hcp_surf_mni` 包含 FNIT 当前覆盖的球面/投影步骤，也包含目前未覆盖的结构、偏置和 QC 文件生成。它使用 FreeSurfer、FSL、Workbench 和 MSM，并要求官方 `bb_ext_tools`、FSL 环境及完整的上游 UKB 文件；仅安装 FNIT Conda 环境不能执行这两条官方命令。新增的独立表面阶段读取已有结构文件，使用 PyTorch、nibabel、SciPy 与 Workbench；从原始 BIDS 运行的体积 T1→MNI 后端仍需 Surfa。

```bash
export BB_BIN_DIR=/absolute/path/uk_biobank_pipeline_v_1.5
subj=SUBJECT_ID
study=/absolute/path/ukb_study
source "$BB_BIN_DIR/bb_surf_pipeline/bb_surf_setup" "$subj" "$study"
"$BB_BIN_DIR/bb_surf_pipeline/bb_hcp_surf_mni"
```

对应的官方 Workbench 核心命令示例；`registered_sphere` 换成当前 FS sphere 结果即可按同一注册输入比较投影阶段：

```bash
wb_command -volume-to-surface-mapping clean_mni.nii.gz midthickness.native.surf.gii native.func.gii \
  -ribbon-constrained white.native.surf.gii pial.native.surf.gii -volume-roi goodvoxels.nii.gz
wb_command -metric-resample native.func.gii registered_sphere.native.surf.gii sphere.32k_fs_LR.surf.gii \
  ADAP_BARY_AREA atlas.func.gii -area-surfs midthickness.native.surf.gii midthickness.32k_fs_LR.surf.gii \
  -current-roi roi.native.shape.gii
```

官方完整 `bb_surf` 仍要求 T2_FLAIR、场图处理的先验结果、MSMSulc/MSMAll 配准模板、UKB DeDrift 球面及 FIX 相关输出。指定 UKB 原始 rfMRI ZIP 没有原始 B0 场图，体积流程已按现有数据走无 GDC/B0 路径；因此即使上述投影在同输入下吻合，也不能据此宣称整条 UKB 表面流程等价。

## 真实数据对照

基准固定为同一例 UKB BOLD（490 帧、TR 0.735 秒）、同一份 MNI 2 mm clean BOLD、同一套 MNI white/pial/注册球面/ROI，以及同一个 Connectome Workbench 2.1.0（commit `f724d200fc8a43cd966de6c87ed7d1912c6e42d4`，OpenMP YES）。Workbench 命令对照核对的是相同输入和二进制下的编排与数值；原 UKB 完整发行环境所用 Workbench 版本未在本例确认。对 goodvoxels，应先固定 ribbon，比较 FNIT 与官方 `fslmaths`/`fslstats` 的二值 Dice、不同体素数及墙钟时间；对投影，再固定 goodvoxels 和球面，比较双侧顶点时间序列的 Pearson r、MAE、最大绝对误差、CIFTI 时间轴与皮层下标签。最后单列由 FS sphere 与官方 MSMAll 注册造成的差异。

| 比较 | 输出一致性 | FNIT 耗时 | 对照耗时 | 解释 |
|---|---|---:|---:|---|
| FS sphere/结构准备 | MNI pull 逐顶点反解最大残差：左 0.04972 mm、右 0.04892 mm；皮层 ROI 顶点：左 114,540、右 117,129；被试皮层下 ROI 31,091 体素 | 23.19 秒 | 无配对记录 | 原 UKB T1→MNI warp 与 ROI 不在现有数据中，尚无同输入官方精度或耗时对照 |
| MNI `wmparc`/皮层下 ROI | 独立重算的 FSL warp 与 FNIT pull：前景 Dice 0.9083、逐标签一致率 0.8183、19 标签平均 Dice 0.8657；固定 FNIT pull 的独立 SciPy 最近邻重采样与 FNIT 标签差异 0 体素 | 1.429 秒（GPU＋Workbench） | 2.856 秒（FSL warp＋Workbench） | 两条路线使用不同 T1→MNI warp，仅作描述性比较；`applywarp` 返回 255，但产物完整解码并通过 shape、finite、identity 核查 |
| 同 ribbon 的 goodvoxels | FNIT 178,558、FSL 178,548 个体素；Dice 0.999149、差异 304 体素 | 10.83 / 11.31 秒 | 58.41 / 73.22 秒 | 同一 490 帧 BOLD、同一 87,160 体素 ribbon；两次时间均从固定 ribbon 后的统计步骤计，节点负载有波动 |
| 同 goodvoxels/球面的左右皮层 `func.gii` | 各 32,492 顶点×8 帧；Pearson r=1、MAE=0、最大绝对误差=0，逐值相同 | 23.72 秒 | 23.87 秒 | 同一 Workbench 2.1.0、同一输入、`OMP_NUM_THREADS=8`；分别累加双侧 14 条皮层命令的耗时 |
| 皮层下 4D 与 CIFTI | 皮层下 91×109×91×8、CIFTI 8×91,282；两者 Pearson r≈1、MAE=0、最大绝对误差=0；BrainModelAxis 相同，TR 0.735 秒 | 8.28 秒 | 8.44 秒 | 同一二进制显式执行皮层下及最终 dense 命令；命令阶段总计 FNIT 32.01 秒、显式对照 32.31 秒 |
| BIDS→MNI→表面整链 | 490×91,282 CIFTI，21 个结构的全部灰质单元随时间变化；峰值 GPU reserved 17.58 GB | 外部 wall 1,341.10 秒；其中表面准备 14.67 秒、QC＋投影 807.91 秒 | 无配对完整 UKB 记录 | 整链使用 SynthMorph、ICA-AROMA、FS sphere；不能与官方 MSMAll/FIX 最终结果逐点比较 |
| FS sphere 与官方 MSMAll 最终 CIFTI | 注册方法不同，暂不作逐点等价声明 | — | — | MSMAll/DeDrift 未实现 |

goodvoxels 使用 FSL 6.0.7.22 的原始 `fslmaths`/`fslstats` 命令对照。两轮 FSL 命令均返回 **255**；每个影像产物均通过 gzip 解码、91×109×91 尺寸与 finite 核查，最终掩膜数量和 Dice 两轮一致，故表中保留其数值并如实记录退出状态。FNIT 最终 QC 源码 SHA-256 为 `68045f83e0bdf755e0fbbcdd9113fc1e0f3f5364284e623dcf1fe48f4e19b666`；固定 ribbon 后统计两轮为 10.83/11.31 秒，ribbon 本身另耗 15.38/14.53 秒。FSL 两轮为 58.41/73.22 秒；不把单次时间换算成稳定加速倍数。FNIT 阈值为 1.1719721296，FSL 为 1.1719725；当前仍有 304 个边界体素差异，需继续定位 FSL 平滑与浮点实现细节。真实数据没有负均值、负归一化系数或恰等于阈值的体素；最终 `-bin` 和 `< Upper` 语义修订后，旧/新 FNIT 掩膜逐体素相同。另一轮完整 BIDS 整链的 clean MNI 不同，其 goodvoxels 为 178,740；最终 QC 源码在同一整链输入上重算的阈值与掩膜逐值相同。

8 帧投影对照将固定输入分别交给 FNIT 编排函数和显式 Workbench 命令；两条路线都调用同一个 Workbench 2.1.0，因此逐值相同验证的是**命令编排和文件组织**。一次默认 128 线程的 8 帧 `-cifti-resample` 在运行超过 226 秒后仍未结束，已只中止该测试子进程；同输入的 8 线程命令完成于 6.09 秒。默认线程的 226 秒是删失下界，且当时存在另一条整链任务，不能用它计算正式加速倍数。最终投影源码 SHA-256 为 `724b782f5d265386dda9fac06b6e8be7685c3d748bdb92ef44a13722f4669fc2`。490 帧整链启动时的投影源码是较早版本，脚本显式设置了 `OMP_NUM_THREADS=8`，与最终源码的有效线程数和 Workbench 参数相同；QC `< Upper` 修改后对整链同输入重算，掩膜差异 0 体素。这是针对修改范围的复核，不代表最终源码又从 BIDS 完整重跑了一遍。可机器读取的标量结果见[表面验证摘要](../../validation/fmri/surface_summary.json)。
