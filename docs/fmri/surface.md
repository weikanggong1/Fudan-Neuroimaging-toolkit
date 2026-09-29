# 静息态 fMRI：清理后体积影像到 fsLR32k 时间序列

本页目标是把**已经完成混杂回归的 MNI152 2 mm BOLD**转换成可分析的双侧 fsLR32k 皮层与 2 mm 皮层下 CIFTI 时间序列。输入还包括同一被试已经生成的 FreeSurfer white、pial、sphere、sphere.reg、sulc、thickness 和 wmparc。FNIT 读取这些结构文件，不运行 FreeSurfer 可执行程序。默认由 PyTorch 根据 sulc 估计个体 fsLR 注册球面，再用 Connectome Workbench 完成 ribbon 采样、表面重采样、2 mm FWHM 平滑和 CIFTI 组装。**不需要 FLAIR、髓鞘图、MSMAll 或 DeDrift。**

参考 [fMRIPrep 的 HCP Grayordinates 步骤](https://fmriprep.org/en/stable/workflows.html#hcp-grayordinates)，FNIT 也按“个体表面 ribbon 采样→填补采样空洞→沿注册球面重采样到 fsLR→组合皮层下时间序列”处理。fMRIPrep 启用 MSM 时使用 MSMSulc 球面，见[其输出说明](https://fmriprep.org/en/stable/outputs.html)；FNIT 的球面优化器是独立 PyTorch 实现，**不等同于官方 MSM 算法或其逐点结果**。FNIT 从已配准的 MNI BOLD 开始，而 fMRIPrep 从 T1w 空间的 BOLD 投影，因此两条路径的 CIFTI 数值也不要求相同。UKB 最终 `bb.rfMRI.MSMAll_smooth_2.dtseries.nii` 另含 MSMAll、DeDrift 和 FIX；本页输出不冒用该名称。[UKB 官方表面脚本](https://git.fmrib.ox.ac.uk/falmagro/uk_biobank_pipeline_v_1.5/-/tree/master/bb_surf_pipeline)

## 安装与公开模板

主页的 Conda 环境包含 Python 依赖与 `connectome-workbench-cli`。HCP 的公开模板单独下载，固定于 [HCPpipelines v4.7.0](https://github.com/Washington-University/HCPpipelines/tree/f8cac6892f88bdf889d644711ff038198eb81533) 的 commit `f8cac6892f88bdf889d644711ff038198eb81533`，逐文件核对 SHA-256；输出目录同时保存该版本的 `LICENSE.md`。原始站点不可达时，安装器转到同一 commit 的 jsDelivr 地址，校验规则不变。

```bash
python tools/setup_fmri_surface_assets.py \
  --output-dir /absolute/path/hcp_surface_assets
```

`--output-dir` 是保存 HCP 公开模板的**绝对目录**。默认清单包含 fsLR32k/164k 球面、双侧参考 sulc、ROI 和皮层下标签表，已足够运行本页流程；再次运行会检查 SHA-256。安装器仍有 `--msmall` 选项供额外实验下载 MSMAll 模板，但本页的时间序列无需该选项，也无需 UKB DeDrift 文件。

`run_surface_projection` 调用 Workbench 时，若环境中没有 `OMP_NUM_THREADS`，默认给其子进程设置 `min(可见 CPU 数, 8)`；已设置时沿用用户的值。例如 `export OMP_NUM_THREADS=4` 可将本次投影限制为 4 线程。结构准备和 ribbon QC 的 Workbench 命令不由这一投影函数管理。

## 输入

| 输入 | 格式、空间和作用 |
|---|---|
| `clean_mni` | 4D NIfTI，已清理 BOLD；MNI152 2 mm 网格，头文件须记录正数 TR。可用体积流程的 `filtered_func_data_clean_MNI152_2mm.nii.gz`。 |
| `mni_reference` | 3D MNI152 2 mm 参考 NIfTI。`clean_mni`、`goodvoxels`、`subject_rois`、`atlas_rois` 必须与其尺寸和 affine 相同。 |
| `subject_dir` | 已生成的单被试 FreeSurfer 目录。必需 `mri/orig.mgz`、`mri/orig/001.mgz`、`mri/wmparc.mgz`，以及双侧 `white`、`pial`、`sphere`、`sphere.reg`、`sulc`、`thickness`。`mri/orig/001.mgz` 不能替代 `mri/orig.mgz`，因为表面顶点需按 conformed T1 的 tkRAS 转为 scanner RAS。 |
| `pull_ras` | 体积流程输出的 `reg/MNI152_2mm_to_T1_pull_ras.nii.gz`；MNI 网格上 3 分量 RAS 毫米位移，表示 MNI 世界坐标到 T1 世界坐标的非线性 pull。 |
| `initial_t1_to_mni_world` | 4×4 NumPy 数组，T1 scanner RAS 到 MNI scanner RAS 的初始**世界坐标**仿射；用 `volume_result.t1_to_mni.moving_to_fixed_world`。不能直接把 FLIRT scaled-mm `.mat` 文件读作这个参数。 |
| `hcp_assets_dir` | 上述安装器保存的模板根目录。 |
| `wb_command` | Connectome Workbench 可执行文件名或绝对路径；用于球面操作、距离场、ribbon 投影、表面重采样和 CIFTI。 |

UKB `T1_20263/raw_data` 存放的是每人一个 ZIP，内部 `FreeSurfer/` 才是 recon-all 结果。已完成体积处理时，可直接向下面的独立入口传入该 ZIP；无需手工解压。FNIT 只读取上述结构文件，不调用 FreeSurfer 程序。入口先核对 `orig/001.mgz` 与体积流程 T1 的尺寸和仿射，防止将另一人的表面映射到当前 BOLD。

准备函数将 native `white/pial` 从 FreeSurfer tkRAS 转为 T1 scanner RAS，然后逐顶点求解 MNI 非线性 pull 的反变换，生成 MNI scanner RAS 下的 white、pial、midthickness。球面采用现有 `sphere.reg` 到 fsLR 的 FS sphere 对应关系；原生皮层 ROI 从 `abs(thickness)>0` 生成，Workbench 依次执行 `-metric-fill-holes` 和 `-metric-remove-islands`，然后把 164k fsLR atlas ROI 沿注册球面反投到原生网格（`BARYCENTRIC -largest`），与个人 ROI 做并集。`wmparc.mgz` 按同一 pull 最近邻重采样到 MNI 网格，再用 HCP 标签表提取被试皮层下 ROI。本例 recon-all ZIP 内有 `mri/brain.finalsurfs.mgz`。它与 `orig.mgz` 的空间仿射相同；FNIT 的 tkRAS→scanner RAS 坐标与官方 CRAS 平移后的双侧 white/pial 表面逐顶点核对，最大距离为 0.0000314 mm。后续 MNI 非线性形变仍与官方 FNIRT warp 不同。`make_ribbon_goodvoxels` 用 Workbench 距离场生成皮层 ribbon，并用 BOLD 时间变异系数和局部平滑趋势计算 goodvoxels；时间标准差采用 N−1 样本方差，空间阈值统计只使用 ribbon 内的非零体素且采用 N−1 样本方差，对应 FSL `-Tstd` 和 `fslstats -S`。可传入固定 goodvoxels 掩膜以做同输入比较。

## 子函数接口

下表列出已有表面函数的参数。`output_dir` 均为本次运行的输出目录；输入影像与结构文件的空间约定见上表。右栏指它在 [UKB `bb_hcp_surf_mni`](https://git.fmrib.ox.ac.uk/falmagro/uk_biobank_pipeline_v_1.5/-/blob/0e39a7f7eb76b55437942bfa3073512506b6c8fa/bb_surf_pipeline/bb_hcp_surf_mni) 中对应的处理位置，不表示不同形变算法逐值相同。

| 函数 | 输入参数 | 返回值和磁盘输出 | 官方对照命令 |
|---|---|---|---|
| `invert_mni_to_t1_pull` | `t1_world_points`：T1 scanner RAS 中的 N×3 毫米坐标；`pull_ras`：MNI→T1 三分量位移；`initial_t1_to_mni_world`：T1→MNI 4×4 世界仿射。可选 `device="cpu"`、`tolerance_mm=0.05`、`max_iterations=80`、`chunk_size=32768`。 | `(mni_world_points, max_inverse_residual_mm)`；N×3 MNI scanner RAS 坐标与最大反解残差；无磁盘文件。 | `invwarp` 后的 `wb_command -surface-apply-warpfield ... -fnirt` 所处的表面形变步骤；FNIT 直接反解自身保存的 pull。 |
| `prepare_mni_surface_geometry` | `subject_dir`：现成 FreeSurfer 结构目录；`pull_ras`、`initial_t1_to_mni_world` 同上；`output_dir`。可选 `device="cpu"`、`tolerance_mm=0.05`、`overwrite=False`。 | `MNISurfaceResult.left/right`：各含 `white`、`pial`、`midthickness` 的 native GIFTI 路径、`vertex_count`、`max_inverse_residual_mm`；文件写入 `output_dir`。 | `mri_info --cras`、`wb_command -surface-apply-affine`、`-surface-apply-warpfield`；本例 `orig.mgz` 的 tkRAS→scanner RAS 与官方 CRAS 逐点核对。 |
| `make_ribbon_goodvoxels` | `clean_bold`：MNI 2 mm 4D BOLD；`reference`：同网格 3D 图；`left_white`、`left_pial`、`right_white`、`right_pial`：MNI scanner RAS GIFTI；`output_dir`。可选 `wb_command="wb_command"`、`neighborhood_sigma_mm=5.0`、`threshold_factor=0.5`。 | `SurfaceQCResult.ribbon`、`.goodvoxels`、`.report`：两个 3D NIfTI 与统计 JSON，位于 `output_dir`。 | `wb_command -create-signed-distance-volume`；`fslmaths -Tmean/-Tstd/-bin/-s/-dilD/-thr` 和 `fslstats -M/-S`。 |
| `prepare_fs_sphere_projection_inputs` | `subject_dir`、`pull_ras`、`initial_t1_to_mni_world`；`mni_reference`：同网格 3D MNI 2 mm；`hcp_assets_dir`：公开 HCP 资产根；`output_dir`。可选 `wb_command="wb_command"`、`device="cpu"`、`overwrite=False`。 | `SurfacePreparationResult.left/right`：各为八条 GIFTI 路径组成的 `SurfaceHemisphere`；`.subject_rois` 为 `ROIs.2.nii.gz`、`.atlas_rois` 为 HCP 标准标签、`.inverse_residual_mm` 为双侧反解误差。其他中间文件见输出树。 | `-surface-sphere-project-unproject`、`-metric-fill-holes/-metric-remove-islands/-metric-resample`、`-volume-label-import`；对应官方 `bb_hcp_surf_mni` 的 FS sphere 和 ROI 段。 |
| `prepare_msmsulc_inputs` | `subject_dir`：现成 recon-all 目录，须有双侧 `sphere` 和 `sulc`；`initial_spheres`：左、右 FS→fsLR 原生拓扑球面；`hcp_assets_dir`、`output_dir`；可选 `wb_command="wb_command"`。 | 字典 `L`、`R` 各含 `native_sphere`、`rotated_sphere`、`native_sulc`、`reference_sphere`、`reference_sulc`、`affine` 六个路径；本函数只准备输入。 | HCP `MSMSulc.sh` 的 `mris_convert`、`-metric-math`、`-surface-affine-regression`、`-surface-apply-affine`、`-surface-modify-sphere`。 |
| `run_msmsulc` | `inputs`：上一步返回的双侧字典；`output_dir`：注册球面目录；可选 `device="cuda:0"`。 | 返回 `L`、`R` 的原生顶点顺序球面 GIFTI 路径；目录内另有 `registration_report.json` 标量报告。 | `msm --inmesh ... --refmesh ... --indata ... --refdata ... --conf MSMSulcStrainFinalconf --out ...`；FNIT 优化目标与官方不同。 |
| `run_surface_projection` | `clean_mni`：4D BOLD；`mni_reference`、`goodvoxels`、`subject_rois`、`atlas_rois`：同一 2 mm 网格 NIfTI；`left`、`right`：已准备的 `SurfaceHemisphere`；`output_dir`。可选 `wb_command="wb_command"`、`overwrite=False`。 | `SurfaceProjectionResult.dtseries`、`.left_metric`、`.right_metric`、`.subcortical_volume`、`.coverage_report`：CIFTI、双侧 GIFTI、4D NIfTI、覆盖 JSON；`.timing_seconds` 为每条 Workbench 命令耗时。 | `-volume-to-surface-mapping -ribbon-constrained`、`-metric-resample ADAP_BARY_AREA`、`-metric-smoothing`、`-cifti-resample`、`-cifti-create-dense-timeseries`。 |
| `run_surface_from_mni` | `clean_mni`、`mni_reference`；`inputs=SurfacePipelineInputs(left, right, subject_rois, atlas_rois, wb_command, goodvoxels)`；`output_dir`。可选 `overwrite=False`。`goodvoxels=None` 表示现场计算。 | `SurfacePipelineResult.projection`：上述投影结果；`.qc`：现场计算时的 `SurfaceQCResult`，传入固定掩膜时为 `None`。在 `output_dir/qc/` 与 `projection/` 写文件。 | 组合上述 goodvoxels 与 Workbench 投影步骤；官方对应 `bb_hcp_surf_mni` 的 ribbon→dense CIFTI 段。 |
| `run_surface_from_volume` | `volume_dir`：FNIT 已完成 WM/CSF/motion 回归的体积输出目录；`recon_all`：同被试 UKB T1 ZIP 或已解压的 `FreeSurfer/`；`hcp_assets_dir`：公开模板根；`output_dir`。`registration="msmsulc"` 默认以 PyTorch 根据 sulc 估计球面；`registration="fs"` 用原有 FS 对应关系。可选 `wb_command="wb_command"`、`device="cpu"`、`overwrite=False`；`registered_spheres` 可传已有左右球面覆盖内部估计，`dedrift_spheres` 仅随外部球面使用。 | `SurfacePipelineResult`；在 `prepared/`、`msmsulc_inputs/`、`msmsulc/`、`registered/`、`qc/`、`projection/` 写中间球面、ROI、goodvoxels 和 fsLR32k CIFTI。`msmsulc/registration_report.json` 记录球面优化耗时、显存和折叠三角形。ZIP 中的临时结构文件运行结束后删除。 | 参考 fMRIPrep 的 MSMSulc→ribbon→fsLR 灰质时间序列顺序；Workbench 投影命令与下述 `run_surface_projection` 一致，PyTorch 球面优化不等于官方 `msm`。 |

### T1/FLAIR 联合偏置校正

`correct_surface_structural_bias` 是 UKB `bb_surf_BFC` 中构建髓鞘图前的结构像校正。`t1w` 是完整 T1w，`t2_flair` 是配准到同一体素网格的完整 FLAIR，`t1_brain` 是同一网格的去颅骨 T1w，用其非零体素定义脑掩膜；三者必须是 3D RAS 或 LAS 图像。`sigma_mm` 是偏置场两次平滑的标准差，单位毫米，默认 5。`output_dir` 下返回 `bias.nii.gz`、`T1w_restored.nii.gz`、`T1w_restored_brain.nii.gz`、`T2_FLAIR_restored.nii.gz`、`T2_FLAIR_restored_brain.nii.gz` 五张 NIfTI。`*_brain` 文件仅保留脑内体素；这些文件用于后续髓鞘图，**当前高层 fMRI 入口尚未自动调用本函数**。

```python
from fnit.fmri.surface_bias import correct_surface_structural_bias

bfc = correct_surface_structural_bias(
    t1w="/absolute/path/sub-01_T1w.nii.gz",  # 完整 T1w，含颅外组织
    t2_flair="/absolute/path/sub-01_FLAIR_in_T1w.nii.gz",  # 与 T1w 同网格的完整 FLAIR
    t1_brain="/absolute/path/sub-01_desc-brain_T1w.nii.gz",  # 非零体素定义脑掩膜
    output_dir="/absolute/path/sub-01/surface_bias",  # 五张校正结果的保存目录
    sigma_mm=5.0,  # 高斯平滑标准差，单位毫米；UKB 设置为 5
)
print(bfc.bias, bfc.t1_restored, bfc.t1_brain_restored)
print(bfc.t2_restored, bfc.t2_brain_restored)
```

官方对照为 [`bb_surf_BFC`](https://git.fmrib.ox.ac.uk/falmagro/uk_biobank_pipeline_v_1.5/-/blob/0e39a7f7eb76b55437942bfa3073512506b6c8fa/bb_surf_pipeline/bb_surf_BFC)：`fslmaths T1w -mul T2_FLAIR -abs -sqrt`、`-mas T1_brain`、`-div <脑内均值>`、`-s 5`、`-thr <均值−0.5×标准差> -bin -ero`、`wb_command -volume-remove-islands`、`-dilall -s 5`，最后 T1w/FLAIR 分别除以偏置场。FNIT 只用 nibabel、SciPy、Numba 计算；固定 FSL 的同一输入和掩膜时，放射学方向上的 26 邻域均值扩展与本例 `-dilall` 结果逐值相同。

### 髓鞘图

`create_surface_myelin_maps` 接收上述 `t1_restored`、`t2_restored` 两张 T1 网格的完整结构像，`subject_dir` 指向同被试已完成的 recon-all 目录。`left` 和 `right` 是 `SurfaceHemisphere`，使用其中的 `registered_sphere` 与 `native_roi`；白质面、软脑膜面和厚度从 `subject_dir` 读取，转换到 T1 scanner RAS。`hcp_assets_dir` 指向安装器的 HCP 模板根，`output_dir` 保存 T1/FLAIR 比值图、双侧 T1 空间表面、ribbon、原生髓鞘图、参考图校正结果及 fsLR32k 髓鞘图。返回字典的 `L`、`R` 各含 `native_map`、`native_map_corrected`、`atlas_map_corrected`、`native_ribbon` 四条路径；前两张 GIFTI 为原生顶点数，`atlas_map_corrected` 为 32,492 顶点，ribbon 为 T1 网格 3D NIfTI。要用于正式 MSMAll 特征，应把 `left/right.registered_sphere` 换成已经估计的 MSMSulc 球面；当前 FS sphere 只能检验函数本身。

```python
from fnit.fmri.surface_myelin import create_surface_myelin_maps

myelin = create_surface_myelin_maps(
    t1_restored=bfc.t1_restored,  # 偏置校正后的完整 T1w NIfTI
    t2_restored=bfc.t2_restored,  # 同一体素网格的完整 FLAIR NIfTI
    subject_dir="/absolute/path/subject/FreeSurfer",  # 同被试 recon-all 目录，含 orig.mgz、orig/001.mgz、双侧 white/pial/thickness
    left=prepared.left,  # 左侧 SurfaceHemisphere；registered_sphere 应为 MSMSulc 结果
    right=prepared.right,  # 右侧 SurfaceHemisphere；registered_sphere 应为 MSMSulc 结果
    hcp_assets_dir="/absolute/path/hcp_surface_assets",  # fsLR 球面、ROI 和 Conte69 参考髓鞘图
    output_dir="/absolute/path/surface_myelin",  # 原生和 32k 髓鞘图输出目录
    wb_command="/absolute/path/bin/wb_command",  # Connectome Workbench 可执行文件
)
print(myelin["L"].native_map_corrected, myelin["R"].atlas_map_corrected)
```

对应的官方命令见 [`bb_create_ribbon`](https://git.fmrib.ox.ac.uk/falmagro/uk_biobank_pipeline_v_1.5/-/blob/0e39a7f7eb76b55437942bfa3073512506b6c8fa/bb_surf_pipeline/bb_create_ribbon) 和 [`bb_create_myelin_maps`](https://git.fmrib.ox.ac.uk/falmagro/uk_biobank_pipeline_v_1.5/-/blob/0e39a7f7eb76b55437942bfa3073512506b6c8fa/bb_surf_pipeline/bb_create_myelin_maps)：`wb_command -create-signed-distance-volume` 建 ribbon；`-volume-math "clamp((T1w / T2w), 0, 100)"` 建比值图；`-volume-to-surface-mapping ... -myelin-style ribbon thickness 2.1233` 建原生图；随后 `-metric-smoothing`、`-metric-resample ADAP_BARY_AREA`、`-metric-dilate`、`-metric-mask` 和 `-metric-math` 做参考图校正。FNIT 的 FSL 掩膜操作由 nibabel/NumPy 取代，其余表面操作仍调用允许使用的 Workbench。

### MSMSulc 输入与球面重投影

`prepare_msmsulc_inputs` 的 `subject_dir` 必须是已解压的 recon-all 目录，另外需要双侧 `surf/lh.sphere`、`rh.sphere`、`lh.sulc` 和 `rh.sulc`。`initial_spheres` 按左、右顺序传 FS→fsLR 原生拓扑球面；它们只用于求球面仿射初始化。`hcp_assets_dir` 是安装器的模板根；`output_dir` 是本次 GIFTI 与 `.mat` 的输出目录。返回字典的 `L`、`R` 各有 `native_sphere`、`rotated_sphere`、`native_sulc`、`reference_sphere`、`reference_sulc`、`affine` 六条路径。`native_sulc` 与官方转换后的 sulc 符号一致；随后将这个字典传给 `run_msmsulc` 才执行脑沟配准。

```python
from fnit.fmri.surface_registration import prepare_msmsulc_inputs

msm_inputs = prepare_msmsulc_inputs(
    subject_dir="/absolute/path/subject/FreeSurfer",  # 已解压的同被试 recon-all 目录，含双侧 sphere、sulc
    initial_spheres=(
        "/absolute/path/prepared/L.sphere.FS_to_fsLR.native.surf.gii",  # 左侧 FS→fsLR 初始球面
        "/absolute/path/prepared/R.sphere.FS_to_fsLR.native.surf.gii",  # 右侧 FS→fsLR 初始球面
    ),
    hcp_assets_dir="/absolute/path/hcp_surface_assets",  # HCP 参考球面与双侧 sulc 所在根目录
    output_dir="/absolute/path/msmsulc_inputs",  # 左右原生 sulc、旋转球面和仿射矩阵输出目录
    wb_command="/absolute/path/bin/wb_command",  # Connectome Workbench 可执行文件
)
print(msm_inputs["L"].rotated_sphere)  # 左侧用于后续配准的 100 mm 球面
print(msm_inputs["R"].native_sulc)  # 右侧取反后的原生顶点 sulc 指标
```

`run_msmsulc` 读取 `msm_inputs` 的双侧 100 mm 旋转球面、原生 sulc、fsLR164k 参考球面和参考 sulc。它把两侧 sulc 分别归一化，从粗到细进行四阶段球面优化，并限制局部边长和三角形翻折。`device` 是 PyTorch 设备；CUDA 运行默认启用 TF32。返回字典的 `L`、`R` 是**与原生 white/pial 同顶点顺序**的注册球面 GIFTI，输出目录另有 `registration_report.json`，记录各侧顶点/三角形数、耗时、峰值显存、每阶段误差和翻折数。它不是官方 MSM 优化器，不能把高 sulc 相关误认为球面逐点等价。

```python
from fnit import run_msmsulc

sulc_spheres = run_msmsulc(
    inputs=msm_inputs,  # prepare_msmsulc_inputs 返回的左右双侧输入
    output_dir="/absolute/path/msmsulc",  # 注册球面 GIFTI 与标量报告的输出目录
    device="cuda:0",  # PyTorch 优化设备；CUDA 默认允许 TF32
)
print(sulc_spheres["L"], sulc_spheres["R"])  # 原生顶点顺序的左右注册球面路径
```

官方对照需在**独立验证环境**以相同原生/参考球面和 sulc 调用 [HCP `MSMSulc.sh`](https://github.com/Washington-University/HCPpipelines/blob/master/global/scripts/MSMSulc.sh)，其核心为 `msm --inmesh <旋转球面> --refmesh <fsLR164k球面> --indata <原生sulc> --refdata <参考sulc> --conf MSMSulcStrainFinalconf --out <前缀>`；FNIT 产品运行时不调用 `msm`。下方真实数据表同时给出个体球面的精度与运行时间。

`apply_dedrift` 接收**已经估计好的**左右个体 MSMAll 球面 `registered_spheres`，以及同一 fsLR164k 拓扑的左右群体 DeDrift 球面 `dedrift_spheres`。`hcp_assets_dir` 提供未变形的 fsLR164k 参考球面，`output_dir` 保存两张原生拓扑的 `L/R.sphere.MSMAll_DeDrift.native.surf.gii`。要进入 UKB 最终空间，`dedrift_spheres` 必须是 UKB 的 `DeDriftMSMAllUKB` 文件；安装器提供的 HCP `DeDriftingGroup` 只能用于 HCP 对照。

```python
from fnit.fmri.surface_registration import apply_dedrift

dedrifted = apply_dedrift(
    registered_spheres=(
        "/absolute/path/L.sphere.MSMAll.native.surf.gii",  # 个体左侧 MSMAll 注册结果
        "/absolute/path/R.sphere.MSMAll.native.surf.gii",  # 个体右侧 MSMAll 注册结果
    ),
    dedrift_spheres=(
        "/absolute/path/DeDriftMSMAllUKB.L.sphere.DeDriftMSMAllUKB.164k_fs_LR.surf.gii",  # UKB 左侧群体变换
        "/absolute/path/DeDriftMSMAllUKB.R.sphere.DeDriftMSMAllUKB.164k_fs_LR.surf.gii",  # UKB 右侧群体变换
    ),
    hcp_assets_dir="/absolute/path/hcp_surface_assets",  # fsLR164k 未变形参考球面所在目录
    output_dir="/absolute/path/dedrift",  # 两张 DeDrift 后原生拓扑 GIFTI 的输出目录
    wb_command="/absolute/path/bin/wb_command",  # Connectome Workbench 可执行文件
)
print(dedrifted["L"], dedrifted["R"])  # 双侧 DeDrift 后注册球面路径
```

`prepare_registered_projection` 接收一侧的现有 `SurfaceHemisphere`、该侧仅由厚度生成的 `individual_roi`、新球面 `registered_sphere`、fsLR164k 的 `reference_sphere_164k` 与 `reference_roi_164k`，重算原生 ROI 和 32k midthickness。`output_dir` 保存 `atlasroi.native_projected.shape.gii`、`roi.native.shape.gii`、`midthickness.32k_fsLR.surf.gii`；返回的新 `SurfaceHemisphere` 可传给 `run_surface_from_mni`。左右半球需分别调用，随后才能投影清理后的 volume BOLD。

```python
from fnit.fmri.surface_registration import prepare_registered_projection

left_final = prepare_registered_projection(
    hemisphere=prepared.left,  # prepare_fs_sphere_projection_inputs 返回的左侧表面与 32k 模板路径
    individual_roi="/absolute/path/prepared/L.roi.individual.native.shape.gii",  # 左侧厚度有效顶点 ROI
    registered_sphere=sulc_spheres["L"],  # 上面 PyTorch 脑沟配准得到的左侧原生球面
    reference_sphere_164k="/absolute/path/hcp_surface_assets/global/templates/standard_mesh_atlases/fsaverage.L_LR.spherical_std.164k_fs_LR.surf.gii",  # fsLR164k 参考球面
    reference_roi_164k="/absolute/path/hcp_surface_assets/global/templates/standard_mesh_atlases/L.atlasroi.164k_fs_LR.shape.gii",  # fsLR164k 皮层 ROI
    output_dir="/absolute/path/registered/L",  # 新的左侧 ROI 与 32k midthickness 输出目录
    wb_command="/absolute/path/bin/wb_command",  # Connectome Workbench 可执行文件
)
print(left_final.registered_sphere, left_final.native_roi, left_final.atlas_midthickness)
```

官方对照分别是 `mris_convert -c lh.sulc lh.white L.sulc.shape.gii` 后 `wb_command -metric-math "var * -1"`、`wb_command -surface-affine-regression` / `-surface-apply-affine` / `-surface-modify-sphere`，以及 [`bb_dedrift_resample`](https://git.fmrib.ox.ac.uk/falmagro/uk_biobank_pipeline_v_1.5/-/blob/0e39a7f7eb76b55437942bfa3073512506b6c8fa/bb_surf_pipeline/bb_dedrift_resample) 的 `wb_command -surface-sphere-project-unproject subject.MSMAll.native.surf.gii fsLR.164k.surf.gii DeDriftMSMAllUKB.164k.surf.gii output.native.surf.gii`。原生 ROI 与 midthickness 则对应官方 `-metric-resample ... BARYCENTRIC -largest`、`-metric-math`、`-surface-resample ... BARYCENTRIC`。FNIT 运行只调用 Workbench；FreeSurfer 转换命令只用于下面的对照测试。

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

## 从已完成的体积回归输出运行

先用 `run_fmri_pipeline(..., regress_wm=True, regress_csf=True, regress_motion=True)` 生成 `filtered_func_data_clean_MNI152_2mm.nii.gz`。`run_surface_from_volume` 从同一个体积目录读取这幅 4D 图、`T1_brain.nii.gz`、MNI 参考、T1→MNI FLIRT 矩阵和 MNI→T1 pull；它要求 `pipeline_report.json` 确认至少一项 WM、CSF 或 motion 回归已启用。请传与该体积 T1 对应的同被试 T1 ZIP，不要只传 `raw_data` 总目录。

```python
from fnit import run_surface_from_volume

result = run_surface_from_volume(
    volume_dir="/absolute/path/sub-EXAMPLE_volume",  # FNIT 体积流程的完整输出目录；内含回归后的 MNI BOLD
    recon_all="/absolute/path/T1_20263/raw_data/SUBJECT_ID_20263_2_0.zip",  # 同被试的 recon-all ZIP
    hcp_assets_dir="/absolute/path/hcp_surface_assets",  # setup_fmri_surface_assets.py 下载并校验的模板根目录
    output_dir="/absolute/path/sub-EXAMPLE_surface",  # prepared/、qc/、projection/ 的父目录
    wb_command="/absolute/path/bin/wb_command",  # Connectome Workbench 可执行文件
    device="cuda:0",  # 表面顶点的非线性反变换和 wmparc 重采样所用 PyTorch 设备
    registration="msmsulc",  # 用 PyTorch 脑沟配准生成左右球面；不需要 FLAIR
    overwrite=False,  # 目标表面文件已存在时停止，避免混合两次运行
)
print(result.projection.dtseries)  # fsLR32k 双侧皮层及 2 mm 皮层下时间序列，形状为时间×灰质单元
print(result.qc.goodvoxels)  # MNI 2 mm 3D 采样掩膜
```

同一功能的命令行形式：

```bash
fnit-fmri surface \
  --volume-dir /absolute/path/sub-EXAMPLE_volume \
  --recon-all /absolute/path/T1_20263/raw_data/SUBJECT_ID_20263_2_0.zip \
  --surface-assets-dir /absolute/path/hcp_surface_assets \
  --output-dir /absolute/path/sub-EXAMPLE_surface \
  --wb-command /absolute/path/bin/wb_command \
  --device cuda:0 \
  --registration msmsulc
```

`--volume-dir`、`--recon-all`、`--surface-assets-dir`、`--output-dir`、`--wb-command`、`--device` 与上面的同名 Python 参数含义相同；`--registration` 默认是 `msmsulc`，传 `fs` 才使用原始 FS 球面作对照。加 `--overwrite` 才允许覆盖已有结果。输出是基于独立 PyTorch 脑沟配准的 fsLR32k time series，不对应 UKB 最终 MSMAll 文件。

默认运行的主要输出如下。`result.projection.dtseries` 指向 CIFTI；`result.qc.goodvoxels` 指向投影用掩膜。皮层 GIFTI 每侧有 32,492 个顶点，CIFTI 只保留 atlas ROI 内的顶点，另含 2 mm 皮层下灰质单元。

```text
output_dir/
├── msmsulc/L.sphere.sulc_registered.native.surf.gii
├── msmsulc/R.sphere.sulc_registered.native.surf.gii
├── msmsulc/registration_report.json
├── registered/L/roi.native.shape.gii
├── registered/R/roi.native.shape.gii
├── qc/goodvoxels.nii.gz
├── projection/L.32k_registered_sphere_s2.func.gii
├── projection/R.32k_registered_sphere_s2.func.gii
├── projection/subcortical_MNI_s2.nii.gz
└── projection/clean_MNI_Atlas_registered_sphere_s2.dtseries.nii
```

已有**独立估计完成**的双侧 MSMAll 原生球面与 UKB 群体 DeDrift 文件时，Python 入口可重新投影同一份清理后的 volume BOLD；本接口不估计 MSMAll 球面。下面是可选的外部球面入口，不属于本页默认 MSMSulc 流程。

```python
from fnit import run_surface_from_volume

registered_result = run_surface_from_volume(
    volume_dir="/absolute/path/sub-EXAMPLE_volume",  # 已完成混杂回归的 FNIT 体积输出目录
    recon_all="/absolute/path/T1_20263/raw_data/SUBJECT_ID_20263_2_0.zip",  # 与 volume T1 同被试的 recon-all ZIP
    hcp_assets_dir="/absolute/path/hcp_surface_assets",  # fsLR32k/164k 公共模板根目录
    output_dir="/absolute/path/sub-EXAMPLE_surface_msmall",  # 新 ROI、DeDrift 和 CIFTI 输出父目录
    wb_command="/absolute/path/bin/wb_command",  # Connectome Workbench 可执行文件
    device="cuda:0",  # 表面非线性坐标变换和 wmparc 重采样设备
    overwrite=False,  # 目标表面文件存在时停止
    registered_spheres=(
        "/absolute/path/L.sphere.MSMAll.native.surf.gii",  # 左侧原生顶点顺序的 MSMAll 球面
        "/absolute/path/R.sphere.MSMAll.native.surf.gii",  # 右侧原生顶点顺序的 MSMAll 球面
    ),
    dedrift_spheres=(
        "/absolute/path/DeDriftMSMAllUKB.L.sphere.DeDriftMSMAllUKB.164k_fs_LR.surf.gii",  # UKB 左侧 164k 群体变换
        "/absolute/path/DeDriftMSMAllUKB.R.sphere.DeDriftMSMAllUKB.164k_fs_LR.surf.gii",  # UKB 右侧 164k 群体变换
    ),
)
print(registered_result.projection.dtseries)  # 490 帧示例对应时间×91,282 灰质单元
```

## 从原始 BIDS 生成体积输入

先按[体积流程](README.md)从原始 BIDS 运行 `run_fmri_pipeline`，启用需要的 WM/CSF/motion 回归；保存其 `output_dir`，再将这个目录作为本页 `run_surface_from_volume(volume_dir=..., registration="msmsulc")` 的输入。已有 recon-all 结果是表面投影的必需结构输入；只有 BIDS T1w 时，FNIT 不会自动生成 white、pial 或 sphere。

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

官方完整 `bb_surf` 仍要求 T2_FLAIR 偏置校正、个体 MSMSulc/MSMAll 球面配准及 DeDrift；官方最终清理还使用 FIX。本页的目标输入是 FNIT 已做混杂回归的 volume BOLD，因而不重复 FIX。`--msmall` 已提供公开群体模板和 HCP 配置，但当前尚未生成个体 MSMAll 球面。指定 UKB 原始 rfMRI ZIP 没有原始 B0 场图，体积流程已按现有数据走无 GDC/B0 路径；即使上述投影在同输入下吻合，也不能据此宣称整条 UKB 表面流程等价。

## 真实数据对照

基准固定为同一例 UKB BOLD（490 帧、TR 0.735 秒）、同一份 MNI 2 mm clean BOLD、同一套 MNI white/pial/ROI，以及 Connectome Workbench 2.1.0（commit `f724d200fc8a43cd966de6c87ed7d1912c6e42d4`，OpenMP YES）。先在相同原生 sulc 和 fsLR 参考图上比较 FNIT 与官方 MSM 的个体球面，再保持 clean BOLD、goodvoxels、ROI 和 Workbench 命令相同，仅替换注册球面，对照前 8 帧 CIFTI。同球面的 Workbench 对照则检验编排与文件组织；原 UKB 完整发行环境所用 Workbench 版本未在本例确认。

| 比较 | 输出一致性 | FNIT 耗时 | 对照耗时 | 解释 |
|---|---|---:|---:|---|
| FS sphere/结构准备 | MNI pull 逐顶点反解最大残差：左 0.04972 mm、右 0.04892 mm；皮层 ROI 顶点：左 114,540、右 117,129；被试皮层下 ROI 31,091 体素 | 23.19 秒 | 无配对记录 | 原 UKB T1→MNI warp 与 ROI 不在现有数据中，尚无同输入官方精度或耗时对照 |
| tkRAS→scanner RAS / 官方 CRAS | 同一 recon-all ZIP 的 `orig.mgz` 与 `brain.finalsurfs.mgz` 仿射完全相同；双侧 white/pial 共 485,970 顶点，四张表面最大顶点距离 0.0000314 mm | NumPy 读取和转换四张表面合计约 0.14 秒，无 GIFTI 写盘 | `mris_convert`＋Workbench 四张合计约 7.22 秒，含 GIFTI 写盘 | 两侧计时范围不同，不计算加速比；这里核对的是 T1 scanner RAS，未核对不同的 MNI 非线性 warp |
| T1/FLAIR 联合偏置校正 | 真实 T1、FLAIR 和同网格脑掩膜；固定 FSL 中间输入后 `-dilall` 结果逐体素相同；脑内最终偏置场 r=0.999999951、MAE=0.00000172，恢复后 T1/FLAIR 脑内 MAE=0.00113/0.000610 强度单位 | 10.08 秒，含首次 Numba 编译与五张压缩 NIfTI 写盘 | FSL 6.0.7.22＋Workbench 57.81 秒，含 16 条命令与写盘 | 同一节点、同一输入；全视野偏置场 MAE=0.000145，残差主要来自平滑边界实现；T1/FLAIR 来自 recon-all 存档，不等同于 UKB 最初的结构预处理输入 |
| T1/FLAIR 原生髓鞘图 | 同一真实 T1/FLAIR 和双侧表面；比值图差异 0 体素、双侧 ribbon 差异 0 体素、双侧原生髓鞘图 MAE/最大误差均为 0 | 双侧髓鞘图与 32k 参考图校正全程 101.90 秒，`OMP_NUM_THREADS=8` | 官方 `-volume-math`、FSL ribbon 掩膜和双侧 `-myelin-style` 子步骤 26.32 秒 | 两侧计时范围不同，不计算加速比；32k 参考图校正尚无整链官方输出对照；本次使用 FS sphere 测试函数，未声称 MSMAll 等价 |
| MSMSulc 输入准备 | 左 120,035、右 122,950 个真实被试顶点；`native_sulc` 与官方 `mris_convert` 后取反逐点相同，双侧 MAE/最大误差均为 0；旋转球面半径均值 100 mm | 双侧完整输入准备 3.71 秒 | 官方 `mris_convert` 左 0.97、右 0.65 秒；后续 Workbench 命令未计时 | 时间范围不同，不计算加速比；本行不含 MSM 优化器 |
| 独立 PyTorch 脑沟注册 / 官方 MSM | 同一真实 sulc 和参考球面，最近邻参考 sulc 的左/右相关：初始 0.775/0.762、FNIT 0.884/0.876、官方 MSM 0.838/0.829；FNIT 对官方球面中位顶点距离 1.993/2.277 mm，第 95 百分位 4.669/5.427 mm；双侧折叠三角形均为 0 | 左 15.66、右 15.32 秒；峰值 GPU reserved 0.239/0.254 GB；脚本双侧含读取和对照墙钟 37.86 秒 | 官方 MSM 左 46 分 39.45 秒、右 46 分 42.17 秒 | FNIT 实现了脑沟驱动球面配准，但目标函数与官方 MSM 不同，不能据此宣称球面逐点等价或完整流程加速 |
| 回归后 volume→MSMSulc→fsLR32k | 490×91,282 CIFTI、TR 0.735 秒；91,282 个灰质单元均有时间变化；双侧球面无翻折 | 整链 757.14 秒，含 recon-all ZIP 解压、双侧注册、goodvoxels 和投影；其中 Workbench 投影命令累计 673.37 秒；本次球面注册左 15.90、右 14.82 秒，GPU peak reserved 0.281/0.296 GB | 无配对完整官方 UKB 计时 | 使用同一例真实回归后 BOLD，输出可供 fsLR32k 分析；完整 UKB MSMAll/FIX 输出不在本次范围 |
| FNIT/官方 MSM 球面对应的 CIFTI | 同一真实 BOLD 前 8 帧、同一 white/pial、goodvoxels、ROI 和 Workbench 2.1.0，仅球面不同；左皮层 r=0.796、MAE=34.78，右皮层 r=0.779、MAE=40.28（原影像强度单位）；所有皮层下结构逐值相同 | 以上整链计时只对应 FNIT 的 490 帧输出 | 官方球面 8 帧投影 41.40 秒，含提取 8 帧；与 490 帧 FNIT 计时不可比较 | 球面差异会传递到皮层时间序列，当前不能声称官方 MSM 数值等价；8 帧比较不能替代完整 490 帧精度评估 |
| DeDrift 球面组合内核 | 双侧 120,035/122,950 顶点；与独立 Workbench 球面组合命令逐点相同，双侧 MAE/最大误差均为 0 | 双侧 5.95 秒 | 独立 Workbench 左 2.77、右 2.94 秒 | 固定真实个体 FS 球面测试组合命令，群体变换采用公开 HCP DeDrift；尚非 UKB MSMAll + UKB DeDrift 结果 |
| 新注册球面的投影准备 | 用同一 FS 球面重建时，左右原生 ROI 差异均为 0 顶点，左右 32k midthickness 最大坐标差均为 0 mm | 左 2.25、右 2.30 秒 | 显式 Workbench 左 2.20、右 2.19 秒 | 复核换球面后必须重建的 ROI 与 midthickness 编排；两条路线调用同一 Workbench |
| 外部球面接入整链 | 同一真实 490 帧回归后 BOLD、同一 FS 球面；重建新 ROI 后得到 490×91,282 CIFTI，与既有 FS 投影逐值相同，MAE/最大误差均为 0 | 727.84 秒，含 ZIP 解压、结构准备、QC 与 490 帧投影 | 既有 FNIT 同输入 FS 投影文件，未保存配对整链计时 | 此行检验传入球面的接口，不是 MSMAll 或 UKB DeDrift 的输出一致性 benchmark |
| MNI `wmparc`/皮层下 ROI | 独立重算的 FSL warp 与 FNIT pull：前景 Dice 0.9083、逐标签一致率 0.8183、19 标签平均 Dice 0.8657；固定 FNIT pull 的独立 SciPy 最近邻重采样与 FNIT 标签差异 0 体素 | 1.429 秒（GPU＋Workbench） | 2.856 秒（FSL warp＋Workbench） | 两条路线使用不同 T1→MNI warp，仅作描述性比较；`applywarp` 返回 255，但产物完整解码并通过 shape、finite、identity 核查 |
| 同 ribbon 的 goodvoxels | FNIT 178,558、FSL 178,548 个体素；Dice 0.999149、差异 304 体素 | 10.83 / 11.31 秒 | 58.41 / 73.22 秒 | 同一 490 帧 BOLD、同一 87,160 体素 ribbon；两次时间均从固定 ribbon 后的统计步骤计，节点负载有波动 |
| 同 goodvoxels/球面的左右皮层 `func.gii` | 各 32,492 顶点×8 帧；Pearson r=1、MAE=0、最大绝对误差=0，逐值相同 | 23.72 秒 | 23.87 秒 | 同一 Workbench 2.1.0、同一输入、`OMP_NUM_THREADS=8`；分别累加双侧 14 条皮层命令的耗时 |
| 皮层下 4D 与 CIFTI | 皮层下 91×109×91×8、CIFTI 8×91,282；两者 Pearson r≈1、MAE=0、最大绝对误差=0；BrainModelAxis 相同，TR 0.735 秒 | 8.28 秒 | 8.44 秒 | 同一二进制显式执行皮层下及最终 dense 命令；命令阶段总计 FNIT 32.01 秒、显式对照 32.31 秒 |
| BIDS→MNI→表面整链 | 490×91,282 CIFTI，21 个结构的全部灰质单元随时间变化；峰值 GPU reserved 17.58 GB | 外部 wall 1,341.10 秒；其中表面准备 14.67 秒、QC＋投影 807.91 秒 | 无配对完整 UKB 记录 | 整链使用 SynthMorph、ICA-AROMA、FS sphere；不能与官方 MSMAll/FIX 最终结果逐点比较 |

goodvoxels 使用 FSL 6.0.7.22 的原始 `fslmaths`/`fslstats` 命令对照。两轮 FSL 命令均返回 **255**；每个影像产物均通过 gzip 解码、91×109×91 尺寸与 finite 核查，最终掩膜数量和 Dice 两轮一致，故表中保留其数值并如实记录退出状态。FNIT 最终 QC 源码 SHA-256 为 `68045f83e0bdf755e0fbbcdd9113fc1e0f3f5364284e623dcf1fe48f4e19b666`；固定 ribbon 后统计两轮为 10.83/11.31 秒，ribbon 本身另耗 15.38/14.53 秒。FSL 两轮为 58.41/73.22 秒；不把单次时间换算成稳定加速倍数。FNIT 阈值为 1.1719721296，FSL 为 1.1719725；当前仍有 304 个边界体素差异，需继续定位 FSL 平滑与浮点实现细节。真实数据没有负均值、负归一化系数或恰等于阈值的体素；最终 `-bin` 和 `< Upper` 语义修订后，旧/新 FNIT 掩膜逐体素相同。另一轮完整 BIDS 整链的 clean MNI 不同，其 goodvoxels 为 178,740；最终 QC 源码在同一整链输入上重算的阈值与掩膜逐值相同。

8 帧投影对照将固定输入分别交给 FNIT 编排函数和显式 Workbench 命令；两条路线都调用同一个 Workbench 2.1.0，因此逐值相同验证的是**命令编排和文件组织**。一次默认 128 线程的 8 帧 `-cifti-resample` 在运行超过 226 秒后仍未结束，已只中止该测试子进程；同输入的 8 线程命令完成于 6.09 秒。默认线程的 226 秒是删失下界，且当时存在另一条整链任务，不能用它计算正式加速倍数。最终投影源码 SHA-256 为 `724b782f5d265386dda9fac06b6e8be7685c3d748bdb92ef44a13722f4669fc2`。490 帧整链启动时的投影源码是较早版本，脚本显式设置了 `OMP_NUM_THREADS=8`，与最终源码的有效线程数和 Workbench 参数相同；QC `< Upper` 修改后对整链同输入重算，掩膜差异 0 体素。这是针对修改范围的复核，不代表最终源码又从 BIDS 完整重跑了一遍。既有 FS 球面结果见[表面验证摘要](../../validation/fmri/surface_summary.json)；本次 MSMSulc 路径的源码哈希、球面误差及 CIFTI 对照见[MSMSulc 标量摘要](../../validation/fmri/msmsulc_summary.json)。
