# 静息态 fMRI：MNI 体积到 fsLR32k 表面

这一页对应 UK Biobank（UKB）表面流程中 **MNI 体积时间序列到 fsLR32k 灰质时间序列**的步骤。输入是已经清理并配准到 MNI152 2 mm 的 4D BOLD，以及同一被试事先生成的 FreeSurfer 白质面、软脑膜面、球面和 `wmparc.mgz`。FNIT 读取这些结构文件，不运行 FreeSurfer 可执行程序。当前实现把 FreeSurfer 球面配准投到 fsLR，使用 Connectome Workbench 做 ribbon 采样、表面重采样、2 mm FWHM 平滑和 CIFTI 组装。没有这些结构文件时，现有 BIDS 体积流程不会凭原始 T1 自动生成表面。

官方 `bb_surf` 在这段处理之后还执行 T2_FLAIR 偏置校正、髓鞘图、MSMSulc、MSMAll、DeDrift、FIX 重应用、双回归和网络 IDP。当前高层入口的输出仍采用 **FS sphere 注册**。新增子函数完成结构像偏置校正、髓鞘图、MSMSulc 输入准备、已有球面与 DeDrift 的组合，以及新球面的投影 ROI 重建；尚未估计 MSMSulc/MSMAll 球面，也不把 ICA-AROMA 称作 FIX。因球面注册与清理方式不同，当前 CIFTI 与官方最终 `bb.rfMRI.MSMAll_smooth_2.dtseries.nii` 不具备逐点数值等价性。官方步骤见 [UKB v1.5 `bb_surf`](https://git.fmrib.ox.ac.uk/falmagro/uk_biobank_pipeline_v_1.5/-/blob/0e39a7f7eb76b55437942bfa3073512506b6c8fa/bb_surf_pipeline/bb_surf) 和 [`bb_hcp_surf_mni`](https://git.fmrib.ox.ac.uk/falmagro/uk_biobank_pipeline_v_1.5/-/blob/0e39a7f7eb76b55437942bfa3073512506b6c8fa/bb_surf_pipeline/bb_hcp_surf_mni)。

## 安装与公开模板

主页的 Conda 环境包含 Python 依赖与 `connectome-workbench-cli`。HCP 的公开模板单独下载，固定于 [HCPpipelines v4.7.0](https://github.com/Washington-University/HCPpipelines/tree/f8cac6892f88bdf889d644711ff038198eb81533) 的 commit `f8cac6892f88bdf889d644711ff038198eb81533`，逐文件核对 SHA-256；输出目录同时保存该版本的 `LICENSE.md`。原始站点不可达时，安装器转到同一 commit 的 jsDelivr 地址，校验规则不变。

```bash
python tools/setup_fmri_surface_assets.py \
  --output-dir /absolute/path/hcp_surface_assets \
  --msmall
```

`--output-dir` 是保存 HCP 公开模板的**绝对目录**。安装器保留 `global/templates/`、`global/config/` 和 `MSMConfig/` 相对路径；再次运行会检查已有文件的 SHA-256。`--msmall` 增加 MSMSulc/MSMAll 配置、40 维群体 RSN 及权重、髓鞘和拓扑参考图，以及 **HCP** 的 DeDrift 球面。核对来源为上述固定 commit 的 [`global/templates/MSMAll`](https://github.com/Washington-University/HCPpipelines/tree/f8cac6892f88bdf889d644711ff038198eb81533/global/templates/MSMAll) 与 [`MSMConfig`](https://github.com/Washington-University/HCPpipelines/tree/f8cac6892f88bdf889d644711ff038198eb81533/MSMConfig)。[UKB `bb_surf_setup`](https://git.fmrib.ox.ac.uk/falmagro/uk_biobank_pipeline_v_1.5/-/blob/0e39a7f7eb76b55437942bfa3073512506b6c8fa/bb_surf_pipeline/bb_surf_setup) 指定的是另外两张 `DeDriftMSMAllUKB.L/R...surf.gii`；当前安装器没有它们，因此公开 HCP DeDrift 不能作为 UKB 最终空间的等价替代。这些群体模板也不含被试 MSMAll 配准结果；安装完成不会自动把当前 FS sphere 投影变成 MSMAll。省略 `--msmall` 时只安装原有投影模板。

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

UKB `T1_20263/raw_data` 存放的是每人一个 ZIP，内部 `FreeSurfer/` 才是 recon-all 结果。已完成体积处理时，可直接向下面的独立入口传入该 ZIP；无需手工解压。FNIT 只读取所需的 `orig.mgz`、`orig/001.mgz`、`wmparc.mgz` 和双侧 white、pial、sphere.reg、thickness，不调用 FreeSurfer 程序。入口先核对 `orig/001.mgz` 与体积流程 T1 的尺寸和仿射，防止将另一人的表面映射到当前 BOLD。

准备函数将 native `white/pial` 从 FreeSurfer tkRAS 转为 T1 scanner RAS，然后逐顶点求解 MNI 非线性 pull 的反变换，生成 MNI scanner RAS 下的 white、pial、midthickness。球面采用现有 `sphere.reg` 到 fsLR 的 FS sphere 对应关系；原生皮层 ROI 从 `abs(thickness)>0` 生成，Workbench 依次执行 `-metric-fill-holes` 和 `-metric-remove-islands`，然后把 164k fsLR atlas ROI 沿注册球面反投到原生网格（`BARYCENTRIC -largest`），与个人 ROI 做并集。`wmparc.mgz` 按同一 pull 最近邻重采样到 MNI 网格，再用 HCP 标签表提取被试皮层下 ROI。本例 recon-all ZIP 内有 `mri/brain.finalsurfs.mgz`。它与 `orig.mgz` 的空间仿射相同；FNIT 的 tkRAS→scanner RAS 坐标与官方 CRAS 平移后的双侧 white/pial 表面逐顶点核对，最大距离为 0.0000314 mm。后续 MNI 非线性形变仍与官方 FNIRT warp 不同。`make_ribbon_goodvoxels` 用 Workbench 距离场生成皮层 ribbon，并用 BOLD 时间变异系数和局部平滑趋势计算 goodvoxels；时间标准差采用 N−1 样本方差，空间阈值统计只使用 ribbon 内的非零体素且采用 N−1 样本方差，对应 FSL `-Tstd` 和 `fslstats -S`。可传入固定 goodvoxels 掩膜以做同输入比较。

## 子函数接口

下表列出已有表面函数的参数。`output_dir` 均为本次运行的输出目录；输入影像与结构文件的空间约定见上表。右栏指它在 [UKB `bb_hcp_surf_mni`](https://git.fmrib.ox.ac.uk/falmagro/uk_biobank_pipeline_v_1.5/-/blob/0e39a7f7eb76b55437942bfa3073512506b6c8fa/bb_surf_pipeline/bb_hcp_surf_mni) 中对应的处理位置，不表示不同形变算法逐值相同。

| 函数 | 输入参数 | 返回值和磁盘输出 | 官方对照命令 |
|---|---|---|---|
| `invert_mni_to_t1_pull` | `t1_world_points`：T1 scanner RAS 中的 N×3 毫米坐标；`pull_ras`：MNI→T1 三分量位移；`initial_t1_to_mni_world`：T1→MNI 4×4 世界仿射。可选 `device="cpu"`、`tolerance_mm=0.05`、`max_iterations=80`、`chunk_size=32768`。 | `(mni_world_points, max_inverse_residual_mm)`；N×3 MNI scanner RAS 坐标与最大反解残差；无磁盘文件。 | `invwarp` 后的 `wb_command -surface-apply-warpfield ... -fnirt` 所处的表面形变步骤；FNIT 直接反解自身保存的 pull。 |
| `prepare_mni_surface_geometry` | `subject_dir`：现成 FreeSurfer 结构目录；`pull_ras`、`initial_t1_to_mni_world` 同上；`output_dir`。可选 `device="cpu"`、`tolerance_mm=0.05`、`overwrite=False`。 | `MNISurfaceResult.left/right`：各含 `white`、`pial`、`midthickness` 的 native GIFTI 路径、`vertex_count`、`max_inverse_residual_mm`；文件写入 `output_dir`。 | `mri_info --cras`、`wb_command -surface-apply-affine`、`-surface-apply-warpfield`；本例 `orig.mgz` 的 tkRAS→scanner RAS 与官方 CRAS 逐点核对。 |
| `make_ribbon_goodvoxels` | `clean_bold`：MNI 2 mm 4D BOLD；`reference`：同网格 3D 图；`left_white`、`left_pial`、`right_white`、`right_pial`：MNI scanner RAS GIFTI；`output_dir`。可选 `wb_command="wb_command"`、`neighborhood_sigma_mm=5.0`、`threshold_factor=0.5`。 | `SurfaceQCResult.ribbon`、`.goodvoxels`、`.report`：两个 3D NIfTI 与统计 JSON，位于 `output_dir`。 | `wb_command -create-signed-distance-volume`；`fslmaths -Tmean/-Tstd/-bin/-s/-dilD/-thr` 和 `fslstats -M/-S`。 |
| `prepare_fs_sphere_projection_inputs` | `subject_dir`、`pull_ras`、`initial_t1_to_mni_world`；`mni_reference`：同网格 3D MNI 2 mm；`hcp_assets_dir`：公开 HCP 资产根；`output_dir`。可选 `wb_command="wb_command"`、`device="cpu"`、`overwrite=False`。 | `SurfacePreparationResult.left/right`：各为八条 GIFTI 路径组成的 `SurfaceHemisphere`；`.subject_rois` 为 `ROIs.2.nii.gz`、`.atlas_rois` 为 HCP 标准标签、`.inverse_residual_mm` 为双侧反解误差。其他中间文件见输出树。 | `-surface-sphere-project-unproject`、`-metric-fill-holes/-metric-remove-islands/-metric-resample`、`-volume-label-import`；对应官方 `bb_hcp_surf_mni` 的 FS sphere 和 ROI 段。 |
| `run_surface_projection` | `clean_mni`：4D BOLD；`mni_reference`、`goodvoxels`、`subject_rois`、`atlas_rois`：同一 2 mm 网格 NIfTI；`left`、`right`：已准备的 `SurfaceHemisphere`；`output_dir`。可选 `wb_command="wb_command"`、`overwrite=False`。 | `SurfaceProjectionResult.dtseries`、`.left_metric`、`.right_metric`、`.subcortical_volume`、`.coverage_report`：CIFTI、双侧 GIFTI、4D NIfTI、覆盖 JSON；`.timing_seconds` 为每条 Workbench 命令耗时。 | `-volume-to-surface-mapping -ribbon-constrained`、`-metric-resample ADAP_BARY_AREA`、`-metric-smoothing`、`-cifti-resample`、`-cifti-create-dense-timeseries`。 |
| `run_surface_from_mni` | `clean_mni`、`mni_reference`；`inputs=SurfacePipelineInputs(left, right, subject_rois, atlas_rois, wb_command, goodvoxels)`；`output_dir`。可选 `overwrite=False`。`goodvoxels=None` 表示现场计算。 | `SurfacePipelineResult.projection`：上述投影结果；`.qc`：现场计算时的 `SurfaceQCResult`，传入固定掩膜时为 `None`。在 `output_dir/qc/` 与 `projection/` 写文件。 | 组合上述 goodvoxels 与 Workbench 投影步骤；官方对应 `bb_hcp_surf_mni` 的 ribbon→dense CIFTI 段。 |
| `run_surface_from_volume` | `volume_dir`：FNIT 已完成 WM/CSF/motion 回归的体积输出目录；`recon_all`：同被试 UKB T1 ZIP 或已解压的 `FreeSurfer/`；`hcp_assets_dir`：公开模板根；`output_dir`。可选 `wb_command="wb_command"`、`device="cpu"`、`overwrite=False`。 | `SurfacePipelineResult`；在 `output_dir/prepared/`、`qc/`、`projection/` 依次写结构准备、goodvoxels 和 fsLR32k CIFTI。ZIP 中的临时结构文件运行结束后删除。 | 读取 volume 已保存的 T1→MNI 矩阵和 pull，组合现有 FS sphere、ribbon、Workbench 投影；对应官方 initial CIFTI 段。 |

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

以下三个子函数用于核对球面配准接口，**不会计算个体 MSMSulc 或 MSMAll 注册**。`prepare_msmsulc_inputs` 的 `subject_dir` 必须是已解压的 recon-all 目录，另外需要双侧 `surf/lh.sphere`、`rh.sphere`、`lh.sulc` 和 `rh.sulc`。`initial_spheres` 按左、右顺序传 FS→fsLR 原生拓扑球面；它们只用于求球面仿射初始化。`hcp_assets_dir` 是安装器的模板根；`output_dir` 是本次 GIFTI 与 `.mat` 的输出目录。返回字典的 `L`、`R` 各有 `native_sphere`、`rotated_sphere`、`native_sulc`、`reference_sphere`、`reference_sulc`、`affine` 六条路径。`native_sulc` 与官方转换后的 sulc 符号一致。

```python
from fnit.fmri.surface_registration import prepare_msmsulc_inputs

msm_inputs = prepare_msmsulc_inputs(
    subject_dir="/absolute/path/subject/FreeSurfer",  # 已解压的同被试 recon-all 目录，含双侧 sphere、sulc
    initial_spheres=(
        "/absolute/path/prepared/L.sphere.FS_to_fsLR.native.surf.gii",  # 左侧 FS→fsLR 初始球面
        "/absolute/path/prepared/R.sphere.FS_to_fsLR.native.surf.gii",  # 右侧 FS→fsLR 初始球面
    ),
    hcp_assets_dir="/absolute/path/hcp_surface_assets",  # HCP 模板与 MSMSulc 配置根目录
    output_dir="/absolute/path/msmsulc_inputs",  # 左右原生 sulc、旋转球面和仿射矩阵输出目录
    wb_command="/absolute/path/bin/wb_command",  # Connectome Workbench 可执行文件
)
print(msm_inputs["L"].rotated_sphere)  # 左侧用于后续配准的 100 mm 球面
print(msm_inputs["R"].native_sulc)  # 右侧取反后的原生顶点 sulc 指标
```

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
    registered_sphere=dedrifted["L"],  # 个体 MSMAll 与 UKB DeDrift 组合后的左侧球面
    reference_sphere_164k="/absolute/path/hcp_surface_assets/global/templates/standard_mesh_atlases/fsaverage.L_LR.spherical_std.164k_fs_LR.surf.gii",  # fsLR164k 参考球面
    reference_roi_164k="/absolute/path/hcp_surface_assets/global/templates/standard_mesh_atlases/L.atlasroi.164k_fs_LR.shape.gii",  # fsLR164k 皮层 ROI
    output_dir="/absolute/path/final_projection/L",  # 新的左侧 ROI 与 32k midthickness 输出目录
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
  --device cuda:0
```

`--volume-dir`、`--recon-all`、`--surface-assets-dir`、`--output-dir`、`--wb-command`、`--device` 与上面的同名 Python 参数含义相同；加 `--overwrite` 才允许覆盖已有结果。这里得到的是基于 FreeSurfer `sphere.reg` 的 fsLR32k time series，尚未执行 MSMSulc 或 MSMAll，不对应 UKB 最终 MSMAll 文件。

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
    regress_wm=True,  # 使用体积 BOLD 网格中的白质均值做混杂回归
    regress_csf=True,  # 使用脑脊液均值做混杂回归
    regress_motion=True,  # 使用运动参数做混杂回归
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

官方完整 `bb_surf` 仍要求 T2_FLAIR 偏置校正、个体 MSMSulc/MSMAll 球面配准及 DeDrift；官方最终清理还使用 FIX。本页的目标输入是 FNIT 已做混杂回归的 volume BOLD，因而不重复 FIX。`--msmall` 已提供公开群体模板和 HCP 配置，但当前尚未生成个体 MSMAll 球面。指定 UKB 原始 rfMRI ZIP 没有原始 B0 场图，体积流程已按现有数据走无 GDC/B0 路径；即使上述投影在同输入下吻合，也不能据此宣称整条 UKB 表面流程等价。

## 真实数据对照

基准固定为同一例 UKB BOLD（490 帧、TR 0.735 秒）、同一份 MNI 2 mm clean BOLD、同一套 MNI white/pial/注册球面/ROI，以及同一个 Connectome Workbench 2.1.0（commit `f724d200fc8a43cd966de6c87ed7d1912c6e42d4`，OpenMP YES）。Workbench 命令对照核对的是相同输入和二进制下的编排与数值；原 UKB 完整发行环境所用 Workbench 版本未在本例确认。对 goodvoxels，应先固定 ribbon，比较 FNIT 与官方 `fslmaths`/`fslstats` 的二值 Dice、不同体素数及墙钟时间；对投影，再固定 goodvoxels 和球面，比较双侧顶点时间序列的 Pearson r、MAE、最大绝对误差、CIFTI 时间轴与皮层下标签。最后单列由 FS sphere 与官方 MSMAll 注册造成的差异。

| 比较 | 输出一致性 | FNIT 耗时 | 对照耗时 | 解释 |
|---|---|---:|---:|---|
| FS sphere/结构准备 | MNI pull 逐顶点反解最大残差：左 0.04972 mm、右 0.04892 mm；皮层 ROI 顶点：左 114,540、右 117,129；被试皮层下 ROI 31,091 体素 | 23.19 秒 | 无配对记录 | 原 UKB T1→MNI warp 与 ROI 不在现有数据中，尚无同输入官方精度或耗时对照 |
| tkRAS→scanner RAS / 官方 CRAS | 同一 recon-all ZIP 的 `orig.mgz` 与 `brain.finalsurfs.mgz` 仿射完全相同；双侧 white/pial 共 485,970 顶点，四张表面最大顶点距离 0.0000314 mm | NumPy 读取和转换四张表面合计约 0.14 秒，无 GIFTI 写盘 | `mris_convert`＋Workbench 四张合计约 7.22 秒，含 GIFTI 写盘 | 两侧计时范围不同，不计算加速比；这里核对的是 T1 scanner RAS，未核对不同的 MNI 非线性 warp |
| T1/FLAIR 联合偏置校正 | 真实 T1、FLAIR 和同网格脑掩膜；固定 FSL 中间输入后 `-dilall` 结果逐体素相同；脑内最终偏置场 r=0.999999951、MAE=0.00000172，恢复后 T1/FLAIR 脑内 MAE=0.00113/0.000610 强度单位 | 10.08 秒，含首次 Numba 编译与五张压缩 NIfTI 写盘 | FSL 6.0.7.22＋Workbench 57.81 秒，含 16 条命令与写盘 | 同一节点、同一输入；全视野偏置场 MAE=0.000145，残差主要来自平滑边界实现；T1/FLAIR 来自 recon-all 存档，不等同于 UKB 最初的结构预处理输入 |
| T1/FLAIR 原生髓鞘图 | 同一真实 T1/FLAIR 和双侧表面；比值图差异 0 体素、双侧 ribbon 差异 0 体素、双侧原生髓鞘图 MAE/最大误差均为 0 | 双侧髓鞘图与 32k 参考图校正全程 101.90 秒，`OMP_NUM_THREADS=8` | 官方 `-volume-math`、FSL ribbon 掩膜和双侧 `-myelin-style` 子步骤 26.32 秒 | 两侧计时范围不同，不计算加速比；32k 参考图校正尚无整链官方输出对照；本次使用 FS sphere 测试函数，未声称 MSMAll 等价 |
| MSMSulc 输入准备 | 左 120,035、右 122,950 个真实被试顶点；`native_sulc` 与官方 `mris_convert` 后取反逐点相同，双侧 MAE/最大误差均为 0；旋转球面半径均值 100 mm | 双侧完整输入准备 3.71 秒 | 官方 `mris_convert` 左 0.97、右 0.65 秒；后续 Workbench 命令未计时 | 时间范围不同，不计算加速比；本行不含 MSM 优化器 |
| 探索性 PyTorch MSMSulc 与官方 MSM 优化器 | 左侧参考 sulc 相关：初始球面 0.7901、PyTorch 实验 0.8895、官方 MSM 0.8463；PyTorch 对官方球面中位顶点距离 1.995 mm，第 95 百分位 4.670 mm | PyTorch 实验 14.6 秒、峰值 GPU 预留 0.252 GB | 官方 MSM 左侧 46 分 39.45 秒 | PyTorch 实验仍未达到球面一致性，未纳入 FNIT 正式接口；相关较高不代表个体注册等价，也没有据此推断正式流程加速 |
| DeDrift 球面组合内核 | 双侧 120,035/122,950 顶点；与独立 Workbench 球面组合命令逐点相同，双侧 MAE/最大误差均为 0 | 双侧 5.95 秒 | 独立 Workbench 左 2.77、右 2.94 秒 | 固定真实个体 FS 球面测试组合命令，群体变换采用公开 HCP DeDrift；尚非 UKB MSMAll + UKB DeDrift 结果 |
| 新注册球面的投影准备 | 用同一 FS 球面重建时，左右原生 ROI 差异均为 0 顶点，左右 32k midthickness 最大坐标差均为 0 mm | 左 2.25、右 2.30 秒 | 显式 Workbench 左 2.20、右 2.19 秒 | 复核换球面后必须重建的 ROI 与 midthickness 编排；两条路线调用同一 Workbench |
| MNI `wmparc`/皮层下 ROI | 独立重算的 FSL warp 与 FNIT pull：前景 Dice 0.9083、逐标签一致率 0.8183、19 标签平均 Dice 0.8657；固定 FNIT pull 的独立 SciPy 最近邻重采样与 FNIT 标签差异 0 体素 | 1.429 秒（GPU＋Workbench） | 2.856 秒（FSL warp＋Workbench） | 两条路线使用不同 T1→MNI warp，仅作描述性比较；`applywarp` 返回 255，但产物完整解码并通过 shape、finite、identity 核查 |
| 同 ribbon 的 goodvoxels | FNIT 178,558、FSL 178,548 个体素；Dice 0.999149、差异 304 体素 | 10.83 / 11.31 秒 | 58.41 / 73.22 秒 | 同一 490 帧 BOLD、同一 87,160 体素 ribbon；两次时间均从固定 ribbon 后的统计步骤计，节点负载有波动 |
| 同 goodvoxels/球面的左右皮层 `func.gii` | 各 32,492 顶点×8 帧；Pearson r=1、MAE=0、最大绝对误差=0，逐值相同 | 23.72 秒 | 23.87 秒 | 同一 Workbench 2.1.0、同一输入、`OMP_NUM_THREADS=8`；分别累加双侧 14 条皮层命令的耗时 |
| 皮层下 4D 与 CIFTI | 皮层下 91×109×91×8、CIFTI 8×91,282；两者 Pearson r≈1、MAE=0、最大绝对误差=0；BrainModelAxis 相同，TR 0.735 秒 | 8.28 秒 | 8.44 秒 | 同一二进制显式执行皮层下及最终 dense 命令；命令阶段总计 FNIT 32.01 秒、显式对照 32.31 秒 |
| BIDS→MNI→表面整链 | 490×91,282 CIFTI，21 个结构的全部灰质单元随时间变化；峰值 GPU reserved 17.58 GB | 外部 wall 1,341.10 秒；其中表面准备 14.67 秒、QC＋投影 807.91 秒 | 无配对完整 UKB 记录 | 整链使用 SynthMorph、ICA-AROMA、FS sphere；不能与官方 MSMAll/FIX 最终结果逐点比较 |
| FS sphere 与官方 MSMAll 最终 CIFTI | 注册方法不同，暂不作逐点等价声明 | — | — | MSMAll/DeDrift 未实现 |

goodvoxels 使用 FSL 6.0.7.22 的原始 `fslmaths`/`fslstats` 命令对照。两轮 FSL 命令均返回 **255**；每个影像产物均通过 gzip 解码、91×109×91 尺寸与 finite 核查，最终掩膜数量和 Dice 两轮一致，故表中保留其数值并如实记录退出状态。FNIT 最终 QC 源码 SHA-256 为 `68045f83e0bdf755e0fbbcdd9113fc1e0f3f5364284e623dcf1fe48f4e19b666`；固定 ribbon 后统计两轮为 10.83/11.31 秒，ribbon 本身另耗 15.38/14.53 秒。FSL 两轮为 58.41/73.22 秒；不把单次时间换算成稳定加速倍数。FNIT 阈值为 1.1719721296，FSL 为 1.1719725；当前仍有 304 个边界体素差异，需继续定位 FSL 平滑与浮点实现细节。真实数据没有负均值、负归一化系数或恰等于阈值的体素；最终 `-bin` 和 `< Upper` 语义修订后，旧/新 FNIT 掩膜逐体素相同。另一轮完整 BIDS 整链的 clean MNI 不同，其 goodvoxels 为 178,740；最终 QC 源码在同一整链输入上重算的阈值与掩膜逐值相同。

8 帧投影对照将固定输入分别交给 FNIT 编排函数和显式 Workbench 命令；两条路线都调用同一个 Workbench 2.1.0，因此逐值相同验证的是**命令编排和文件组织**。一次默认 128 线程的 8 帧 `-cifti-resample` 在运行超过 226 秒后仍未结束，已只中止该测试子进程；同输入的 8 线程命令完成于 6.09 秒。默认线程的 226 秒是删失下界，且当时存在另一条整链任务，不能用它计算正式加速倍数。最终投影源码 SHA-256 为 `724b782f5d265386dda9fac06b6e8be7685c3d748bdb92ef44a13722f4669fc2`。490 帧整链启动时的投影源码是较早版本，脚本显式设置了 `OMP_NUM_THREADS=8`，与最终源码的有效线程数和 Workbench 参数相同；QC `< Upper` 修改后对整链同输入重算，掩膜差异 0 体素。这是针对修改范围的复核，不代表最终源码又从 BIDS 完整重跑了一遍。可机器读取的标量结果见[表面验证摘要](../../validation/fmri/surface_summary.json)。
