# 静息态 fMRI：从回归后的 BOLD 到 fsLR32k

`run_surface_from_volume` 按 fMRIPrep 的 91k 输出顺序生成可分析的 CIFTI 时间序列。它读取 FNIT 体积流程已经完成 WM、CSF 或运动参数回归的 BOLD：同一份个体 EPI 数据分别重采样到 T1w 空间做皮层投影，以及 MNI152NLin6Asym 2 mm 空间提供皮层下体素。回归发生在个体 EPI 空间；**直接投到皮层的是 T1w BOLD，不是 MNI BOLD**。已有 recon-all 结构结果提供 white、pial、sphere、sulc、thickness；运行时只读取这些文件，不执行 FreeSurfer、FSL、fMRIPrep 或 Nipype。

这里的“匹配 fMRIPrep”指固定 BOLD、结构表面、注册球面和掩膜后，从 T1w 体积到 fsLR32k GIFTI、从 MNI 体积到 91k CIFTI 的步骤与数据排序。FNIT 上游采用自己的 BBR、MNI 配准、ICA-AROMA 和混杂回归，因此从原始 BIDS 起的数值不能直接称为 fMRIPrep 全流程逐值相同。默认的 PyTorch MSMSulc 球面也尚不等于官方 MSM 球面；做固定球面对照时可用 `registered_spheres` 提供左右官方球面。

[fMRIPrep 的 fsLR 工作流](https://github.com/nipreps/fmriprep/blob/e56dc9938e742c789510705372f88fdb5a8206c2/fmriprep/workflows/bold/resampling.py)规定：T1w BOLD 经 ribbon-constrained 采样、原生表面 10 mm 最近邻填补、个体皮层掩膜、`ADAP_BARY_AREA` 和 fsLR atlas ROI 掩膜。个体掩膜来自 [sMRIPrep 的 cortex mask 工作流](https://github.com/nipreps/smriprep/blob/6a83b4686953273a0379a5e8aa76bb3d82bf4c3f/src/smriprep/workflows/surfaces.py)；它不把 atlas ROI 并入原生掩膜。投影不做 fsLR 上的 30 mm 填补或 2 mm 平滑。[NiWorkflows 的 CIFTI 组装](https://github.com/nipreps/niworkflows/blob/0eb323521639665483f651cf088c86047382d656/niworkflows/interfaces/cifti.py)使用 TemplateFlow 双侧非内侧壁标签和 HCP 皮层下分区，按 LAS 方向及 HCP 体素顺序保存。FNIT 当前的高层入口已采用这一顺序。

## 安装与输入

主页 Conda 安装包含 nibabel、PyTorch 和 Connectome Workbench。另安装固定版本的 HCP/fsLR 模板及 TemplateFlow HCP dseg；安装器逐文件核对 SHA-256。

```bash
python tools/setup_fmri_surface_assets.py \
  --output-dir /absolute/path/hcp_surface_assets \
  --fmriprep
```

`--output-dir` 是模板根目录的绝对路径。`--fmriprep` 下载 `fmriprep/tpl-MNI152NLin6Asym_res-02_atlas-HCP_dseg.nii.gz`；默认 HCP 清单还提供双侧 fsLR32k 球面、atlas ROI 和官方 `MSMSulcStrainFinalconf` 对照配置。无需 FLAIR、髓鞘图、MSMAll、DeDrift 或 FIX。

| 输入 | 含义 |
|---|---|
| `volume_dir` | `run_fmri_pipeline` 的输出目录。`pipeline_report.json` 中须记录实际完成的 WM、CSF 或运动回归；`outputs.aroma_clean_native` 指向回归后的个体 EPI 4D BOLD，同时须有 `filtered_func_data_clean_MNI152_2mm.nii.gz`、`T1_brain.nii.gz` 和配准文件。MNI 模板必须是 MNI152NLin6Asym 2 mm。 |
| `recon_all` | 与体积 T1 对应的 FreeSurfer 结果目录或 UKB T1 ZIP；须有 `mri/orig.mgz`、`mri/orig/001.mgz` 和双侧 white、pial、sphere.reg、thickness；默认 MSMSulc 还需要 sphere、sulc。FNIT 核对 scanner T1 的尺寸与 affine。 |
| `hcp_assets_dir` | 上述安装器的模板根目录；包含 fsLR32k/164k 球面、ROI 和 TemplateFlow HCP dseg。 |
| `output_dir` | 新的绝对输出目录。`clean_T1w.nii.gz` 是由回归后个体 EPI BOLD 一次插值到 T1w 网格的皮层投影输入。 |
| `registered_spheres`、`registration` | `registered_spheres` 是可选的 `(左, 右)` 原生顶点顺序球面，用于固定球面对照。省略时 `registration="msmsulc"` 运行默认 FNIT 配准；`"newmsm_experimental"` 显式运行下述逐级离散候选；`"fs"` 使用 FS 初始对应关系。实验候选尚未达到官方 newMSM 数值等价。 |
| `goodvoxels` | 可选的 T1w 网格 3D 体积 ROI。省略时走 fMRIPrep 不提供 `volume_roi` 的分支；若提供，必须与 `clean_T1w.nii.gz` 网格一致。 |
| `device`、`wb_command`、`overwrite` | 分别是 PyTorch 设备、Workbench 程序名或绝对路径、是否覆盖已有目标。CUDA 默认允许 TF32；影像以 float32 保存。 |

```python
from fnit import run_surface_from_volume

result = run_surface_from_volume(
    volume_dir="/absolute/path/sub-EXAMPLE_volume",  # FNIT 体积输出；同时含回归后的个体 EPI 与 MNI 4D BOLD
    recon_all="/absolute/path/SUBJECT_ID_20263_2_0.zip",  # 与 T1 匹配的 recon-all ZIP 或 FreeSurfer 目录
    hcp_assets_dir="/absolute/path/hcp_surface_assets",  # 上述安装器生成的 HCP + TemplateFlow 模板根
    output_dir="/absolute/path/sub-EXAMPLE_surface",  # 本次 T1w BOLD、表面和 CIFTI 的保存目录
    wb_command="/absolute/path/bin/wb_command",  # Connectome Workbench 可执行文件
    device="cuda:0",  # T1w 重采样与结构准备的 PyTorch 设备
    registration="msmsulc",  # 默认运行 FNIT 的 PyTorch 脑沟球面配准
    registered_spheres=None,  # 不传外部球面；官方固定球面对照时改成左右文件路径元组
    goodvoxels=None,  # 不使用可选的 T1w 体积 ROI
    overwrite=False,  # 目标 CIFTI 已存在时停止
)
print(result.projection.dtseries)  # 91k CIFTI：时间×灰质坐标
print(result.projection.left_metric)  # 左侧 32,492 顶点×时间 GIFTI
print(result.projection.right_metric)  # 右侧 32,492 顶点×时间 GIFTI
print(result.projection.coverage_report)  # 时间帧、TR、灰质坐标覆盖率和阶段耗时 JSON
```

单被试命令行入口与上面的默认 Python 参数一致：

```bash
fnit-fmri surface \
  --volume-dir /absolute/path/sub-EXAMPLE_volume \
  --recon-all /absolute/path/SUBJECT_ID_20263_2_0.zip \
  --surface-assets-dir /absolute/path/hcp_surface_assets \
  --output-dir /absolute/path/sub-EXAMPLE_surface \
  --wb-command /absolute/path/bin/wb_command \
  --device cuda:0 \
  --registration msmsulc
```

主要输出如下；`result.qc` 在当前默认路径为 `None`，因为默认不计算可选 goodvoxels。`result.projection.subcortical_volume` 指向作为皮层下数据来源的回归后 MNI BOLD。

```text
output_dir/
├── clean_T1w.nii.gz                         # 回归后 BOLD，T1w 空间，X×Y×Z×时间
├── prepared/native/lh.white.T1w.native.surf.gii
├── prepared/native/lh.pial.T1w.native.surf.gii
├── prepared/native/lh.midthickness.T1w.native.surf.gii
├── msmsulc/L.sphere.sulc_registered.native.surf.gii  # 默认内部注册时生成；右侧同理
├── newmsm_experimental/L.sphere.discrete.native.surf.gii  # 仅选择实验分支时生成；右侧同理
├── prepared/L.roi.individual.native.shape.gii  # thickness→填孔→去孤岛的原生皮层掩膜；右侧同理
├── registered/L/midthickness.32k_fsLR.surf.gii  # 按所选球面重采样的中面；右侧同理
├── projection/L.32k.func.gii                # 左侧完整 32k 顶点时间序列
├── projection/R.32k.func.gii                # 右侧完整 32k 顶点时间序列
├── projection/space-fsLR_den-91k_bold.dtseries.nii
└── projection/coverage.json
```

## 子函数与官方命令

| 函数 | 输入与输出 | 对应的官方操作 |
|---|---|---|
| `prepare_t1w_surface_geometry(subject_dir, output_dir, overwrite=False)` | `subject_dir` 为现成 recon-all 目录；返回双侧 white/pial/midthickness 的 T1w scanner RAS GIFTI、原生顶点数。读取 `orig.mgz` 的 tkRAS→scanner RAS 仿射，不调用 FreeSurfer。 | `mris_convert` 加 T1w 空间仿射；本函数用 nibabel 读取。 |
| `prepare_fmriprep_surface_inputs(subject_dir, hcp_assets_dir, output_dir, wb_command="wb_command", overwrite=False)` | 在 T1w 几何基础上生成双侧 FS→fsLR 初始球面及个体皮层 ROI；返回 `T1SurfacePreparation.geometry/initial_spheres/individual_rois`。ROI 按 sMRIPrep 的 thickness 绝对值二值化、填孔、去孤岛步骤生成。不创建 MNI 表面或重采样 wmparc。 | `wb_command -surface-sphere-project-unproject`、`-metric-fill-holes`、`-metric-remove-islands`。 |
| `prepare_msmsulc_inputs(...)`、`run_msmsulc(inputs, output_dir, device="cuda:0")` | 前者产生双侧原生 sulc、旋转球面和 fsLR 参考文件；后者返回左右原生顶点顺序的注册球面，并保存含仿射角度、损失、耗时、显存和折叠数的报告。 | `msm --inmesh ... --refmesh ... --indata ... --refdata ... --conf MSMSulcStrainFinalconf --out ...`。FNIT 优化器尚未逐点复现 MSM。 |
| `run_newmsm_msmsulc(inputs, output_dir, device="cuda:0")` | 读取同一双侧输入，以 162→642→2,562 个控制点的离散局部搜索输出双侧原生顶点顺序球面及 `registration_report.json`。目前需显式选择；输出尚未与官方 newMSM 数值等价。 | `newmsm --inmesh ... --refmesh ... --indata ... --refdata ... --conf MSMSulcStrainFinalconf --out ...`；官方命令仅用于隔离环境中的 benchmark。 |
| `run_fmriprep_surface_projection(clean_t1w, clean_mni, left, right, left_label, right_label, hcp_dseg, output_dir, goodvoxels=None, wb_command="wb_command", overwrite=False)` | `clean_t1w` 为 T1w 4D BOLD，`clean_mni` 为 MNI152NLin6Asym 2 mm 4D BOLD；`left/right` 各含 white/pial/midthickness、注册球面、原生 ROI、32k 球面/中面/atlas ROI；三项标签输入指定 fsLR 非内侧壁 ROI 与 HCP dseg。返回双侧 32k GIFTI、91k CIFTI、原 MNI BOLD 路径、耗时及覆盖报告。 | `wb_command -volume-to-surface-mapping ... -ribbon-constrained white pial`；`-metric-dilate ... 10 ... -nearest`；`-metric-mask`；`-metric-resample ... ADAP_BARY_AREA ... -area-surfs ... -current-roi`；再次 `-metric-mask`。 |
| `create_fmriprep_cifti(clean_mni, left_metric, right_metric, left_label, right_label, hcp_dseg, output_file)` | 双侧 GIFTI 与 MNI BOLD 合成时间×灰质坐标的 CIFTI；按双侧非内侧壁顶点及 HCP 标签分别收集皮层下体素。 | NiWorkflows `GenerateCifti`。输入网格必须与 HCP dseg 匹配；输出沿用输入 TR。 |

单独运行 MSMSulc 时，先准备与 T1 对应的 recon-all 目录和 HCP 表面资源。`prepare_msmsulc_inputs` 的 `initial_spheres` 是左、右原生顶点顺序的 FS→fsLR 初始球面；它读取 `lh/rh.sphere` 和 `lh/rh.sulc`，把 sulc 符号转换为 HCP 约定，再用 Workbench 生成旋转后的 100 mm 球面。返回字典的 `L`、`R` 各含 `native_sphere`、`rotated_sphere`、`native_sulc`、`reference_sphere`、`reference_sulc`、`affine` 六条路径。注册函数读取其中的旋转球面、个体 sulc 和参考球面及 sulc；`output_dir` 是输出目录，`device` 是 PyTorch 设备。返回的 `L`、`R` 是两张原生顶点顺序的 GIFTI 球面，可直接作为 `registered_spheres` 传给高层投影函数。

```python
from fnit import prepare_fmriprep_surface_inputs, prepare_msmsulc_inputs, run_msmsulc

prepared = prepare_fmriprep_surface_inputs(
    subject_dir="/absolute/path/FreeSurfer",  # 与待投影 BOLD 对应的已完成 recon-all 目录
    hcp_assets_dir="/absolute/path/hcp_surface_assets",  # installer 下载的 HCP 表面资源目录
    output_dir="/absolute/path/work/prepared",  # T1w 表面、初始球面和皮层 ROI 的输出目录
    wb_command="wb_command",  # Connectome Workbench 可执行文件
)
inputs = prepare_msmsulc_inputs(
    subject_dir="/absolute/path/FreeSurfer",  # 提供双侧 sphere 与 sulc 的 recon-all 目录
    initial_spheres=prepared.initial_spheres,  # (左球面路径, 右球面路径)，顶点顺序不能改变
    hcp_assets_dir="/absolute/path/hcp_surface_assets",  # 提供 164k 参考球面和 sulc
    output_dir="/absolute/path/work/msmsulc_inputs",  # 保存旋转球面、sulc 与仿射矩阵
    wb_command="wb_command",  # Connectome Workbench 可执行文件
)
spheres = run_msmsulc(
    inputs=inputs,  # 上一步返回的双侧 MSMSulcInputs 字典
    output_dir="/absolute/path/work/msmsulc",  # 保存 L/R 注册球面和 registration_report.json
    device="cuda:0",  # 使用的 GPU；无 GPU 时可填 "cpu"
)
print(spheres["L"], spheres["R"])  # 两张原生顶点顺序的注册球面路径
```

`registration_report.json` 按 `L`、`R` 记录顶点数、三角形数、折叠数、2,562 个控制点、仿射旋转角度、各级目标函数值、单侧优化耗时和峰值 GPU 显存。官方 MSM 的对应单侧命令如下；`--out` 是输出前缀，右侧把 `L` 换成 `R`。官方按 NMI 仿射和三层离散优化运行，FNIT 使用独立的 PyTorch 连续优化，因此两者命令的输入对应，但算法和球面数值尚不等价。

```bash
msm --inmesh=/absolute/path/work/msmsulc_inputs/L.sphere_rot.surf.gii \
  --refmesh=/absolute/path/hcp_surface_assets/global/templates/standard_mesh_atlases/fsaverage.L_LR.spherical_std.164k_fs_LR.surf.gii \
  --indata=/absolute/path/work/msmsulc_inputs/L.sulc.native.shape.gii \
  --refdata=/absolute/path/hcp_surface_assets/global/templates/standard_mesh_atlases/L.refsulc.164k_fs_LR.shape.gii \
  --conf=/absolute/path/hcp_surface_assets/MSMConfig/MSMSulcStrainFinalconf \
  --out=/absolute/path/work/official/L.
```

### 逐级离散球面配准实验分支

`run_newmsm_msmsulc` 使用 FNIT 自己的 PyTorch 实现：面积校正的 sulc 重采样，依次使用 162、642、2,562 个控制点和 2,562、10,242、40,962 个数据点，以局部脑沟相关和三角形形变能选取离散位移。它不调用 newMSM；当前搜索是逐控制点局部更新，**尚未复现官方的标签提案及高阶联合优化**。默认 `registration="msmsulc"` 不受此实验分支影响。

`inputs` 是上例 `prepare_msmsulc_inputs` 返回的 `{"L": MSMSulcInputs, "R": MSMSulcInputs}`。每侧读取 `rotated_sphere`（100 mm 原生球面）、`native_sulc`（同一原生顶点顺序）、`reference_sphere` 和 `reference_sulc`（fsLR 164k 模板）；`native_sphere`、`affine` 由准备函数一并记录。`output_dir` 指定输出文件夹；`device` 指定 PyTorch 设备，CUDA 默认启用 TF32。返回字典的 `L`、`R` 是原生顶点顺序的注册球面路径。目录中还保存 `registration_report.json`，逐侧记录仿射角度、三个级别的控制点数、数据点数、迭代轮数、末轮位移更新数和耗时。

```python
from fnit import run_newmsm_msmsulc

spheres = run_newmsm_msmsulc(
    inputs=inputs,  # prepare_msmsulc_inputs 返回的双侧球面、sulc 和模板路径
    output_dir="/absolute/path/work/newmsm_experimental",  # 输出双侧 GIFTI 和 JSON 报告
    device="cuda:0",  # PyTorch 设备；无 GPU 时填写 "cpu"
)
print(spheres["L"], spheres["R"])  # 左右原生顶点顺序注册球面路径
```

高层 `run_surface_from_volume` 可设置 `registration="newmsm_experimental"`，自动准备输入并将这两张球面用于同一套 fsLR32k 投影。已有注册球面也可用 `registered_spheres=(spheres["L"], spheres["R"])` 明确传入。两种调用的最终输出仍是 `projection/L.32k.func.gii`、`projection/R.32k.func.gii` 和 `projection/space-fsLR_den-91k_bold.dtseries.nii`；其余参数及含义见本页高层示例。

官方对照在隔离环境中运行，每侧命令如下，右侧把 `L` 换成 `R`。所用 HCP `MSMSulcStrainFinalconf` 的 `--simval` 需从 `3` 改为 `1`，因为本次 newMSM 版本不再接受原 NMI 设定；保留其余层级、正则项和迭代次数。FNIT 安装和正常运行均不使用这个命令。

```bash
newmsm --inmesh=/absolute/path/work/msmsulc_inputs/L.sphere_rot.surf.gii \
  --refmesh=/absolute/path/hcp_surface_assets/global/templates/standard_mesh_atlases/fsaverage.L_LR.spherical_std.164k_fs_LR.surf.gii \
  --indata=/absolute/path/work/msmsulc_inputs/L.sulc.native.shape.gii \
  --refdata=/absolute/path/hcp_surface_assets/global/templates/standard_mesh_atlases/L.refsulc.164k_fs_LR.shape.gii \
  --conf=/absolute/path/work/MSMSulcStrainFinalconf.newmsm \
  --out=/absolute/path/work/newmsm_reference/L.
```

一例真实静息态数据在同一初始球面、个体 sulc 和模板 sulc 下对照。按原生顶点对应，先计算两张球面的夹角，再取中位数和第 95 百分位。时间是每侧注册的墙钟时间；FNIT 使用 H100 GPU，官方 newMSM 使用 8 个 CPU 线程，且结果精度不同，不能将耗时比解释为等价加速。

| 实现 | 左侧角差中位数 / 95% | 右侧角差中位数 / 95% | 左 / 右耗时 |
|---|---:|---:|---:|
| FNIT 逐级离散实验分支 vs 官方 newMSM | 0.494° / 1.291° | 0.495° / 1.768° | 48.48 / 42.68 秒 |
| 官方 newMSM | 0° / 0° | 0° / 0° | 312.61 / 241.95 秒 |

该实验分支尚未完成与官方 newMSM 固定其他输入后的整段 490 帧 fsLR32k 时间序列对照；不能把下文默认 FNIT 与旧版 MSM 的 CIFTI 指标当作该分支的指标。

原版完整流程的命令格式如下；`--cifti-output 91k` 指定 91,282 个灰质坐标，`--msm` 启用官方 MSMSulc，`--fs-subjects-dir` 使用已有的同被试 FreeSurfer subject。这是原软件等价入口的说明，不是下文固定输入 benchmark 的运行命令；原版从 BIDS 原始数据重算上游流程，不会直接读取 FNIT 已回归的 BOLD。[官方参数说明](https://fmriprep.org/en/latest/usage.html)与[fsLR 输出说明](https://fmriprep.org/en/latest/outputs.html)给出其空间和球面约定。

```bash
fmriprep /absolute/path/bids /absolute/path/fmriprep_derivatives participant \
  --participant-label EXAMPLE \
  --fs-subjects-dir /absolute/path/freesurfer_subjects \
  --cifti-output 91k \
  --msm \
  --output-spaces MNI152NLin6Asym:res-2 fsLR:den-32k
```

固定输入的皮层步骤对应以下 Workbench 命令。示意左侧；右侧把 `L` 文件替换为 `R`。`T1W_BOLD` 是 T1w 4D BOLD，`WHITE`、`PIAL`、`MID` 是 T1w 原生表面，`CORTEX_MASK` 是 thickness 经填孔、去孤岛后的原生 ROI，`REG_SPHERE` 是 MSMSulc 原生球面，`ATLAS_SPHERE`、`ATLAS_MID` 和 `ATLAS_ROI` 是 fsLR32k 文件。省略可选 `-volume-roi` 对应本页默认设置。

```bash
wb_command -volume-to-surface-mapping "$T1W_BOLD" "$MID" L.native.func.gii -ribbon-constrained "$WHITE" "$PIAL"
wb_command -metric-dilate L.native.func.gii "$MID" 10 L.dilated.func.gii -nearest
wb_command -metric-mask L.dilated.func.gii "$CORTEX_MASK" L.masked.func.gii
wb_command -metric-resample L.masked.func.gii "$REG_SPHERE" "$ATLAS_SPHERE" ADAP_BARY_AREA L.atlas.func.gii -area-surfs "$MID" "$ATLAS_MID" -current-roi "$CORTEX_MASK"
wb_command -metric-mask L.atlas.func.gii "$ATLAS_ROI" L.32k.func.gii
```

单独准备结构输入时，每个参数可按下面填写。原生 ROI 不随注册球面改变。选定注册球面后，使用 `wb_command -surface-resample` 将个体 T1w 中面变换到 fsLR32k；高层入口会自动完成这一步。

```python
from fnit import prepare_fmriprep_surface_inputs

prepared = prepare_fmriprep_surface_inputs(
    subject_dir="/absolute/path/FreeSurfer",  # 同被试已完成的 recon-all 目录
    hcp_assets_dir="/absolute/path/hcp_surface_assets",  # HCP 球面与 ROI 模板根目录
    output_dir="/absolute/path/surface/prepared",  # T1w 几何、初始球面与皮层 ROI 的保存目录
    wb_command="/absolute/path/bin/wb_command",  # Connectome Workbench 可执行文件
    overwrite=False,  # 已有几何文件时停止
)
print(prepared.initial_spheres)  # 左、右原生顶点顺序的 FS→fsLR 初始球面
print(prepared.individual_rois)  # 左、右个体皮层 ROI
```

已有注册后的左右 `SurfaceHemisphere` 时，可单独执行投影与组装。`left`、`right` 必须各含 T1w 空间 white/pial/midthickness、原生注册球面和 ROI，以及 fsLR32k 球面、中面和 atlas ROI；这些字段不能改成 MNI 空间表面。

```python
from fnit import SurfaceHemisphere, run_fmriprep_surface_projection, create_fmriprep_cifti

left = SurfaceHemisphere(
    white="/absolute/path/L.white.T1w.native.surf.gii",  # 左侧 T1w 白质面
    pial="/absolute/path/L.pial.T1w.native.surf.gii",  # 左侧 T1w 软脑膜面
    midthickness="/absolute/path/L.midthickness.T1w.native.surf.gii",  # 左侧 T1w 中面
    registered_sphere="/absolute/path/L.sphere.MSMSulc.native.surf.gii",  # 左侧原生顶点顺序的注册球面
    native_roi="/absolute/path/L.roi.native.shape.gii",  # 左侧原生皮层掩膜
    atlas_sphere="/absolute/path/L.sphere.32k_fs_LR.surf.gii",  # 左侧 fsLR32k 球面
    atlas_midthickness="/absolute/path/L.midthickness.32k_fsLR.surf.gii",  # 左侧 32k 中面
    atlas_roi="/absolute/path/L.atlasroi.32k_fs_LR.shape.gii",  # 左侧 fsLR32k 皮层 ROI
)
right = SurfaceHemisphere(
    white="/absolute/path/R.white.T1w.native.surf.gii",  # 右侧 T1w 白质面
    pial="/absolute/path/R.pial.T1w.native.surf.gii",  # 右侧 T1w 软脑膜面
    midthickness="/absolute/path/R.midthickness.T1w.native.surf.gii",  # 右侧 T1w 中面
    registered_sphere="/absolute/path/R.sphere.MSMSulc.native.surf.gii",  # 右侧原生顶点顺序的注册球面
    native_roi="/absolute/path/R.roi.native.shape.gii",  # 右侧原生皮层掩膜
    atlas_sphere="/absolute/path/R.sphere.32k_fs_LR.surf.gii",  # 右侧 fsLR32k 球面
    atlas_midthickness="/absolute/path/R.midthickness.32k_fsLR.surf.gii",  # 右侧 32k 中面
    atlas_roi="/absolute/path/R.atlasroi.32k_fs_LR.shape.gii",  # 右侧 fsLR32k 皮层 ROI
)

projection = run_fmriprep_surface_projection(
    clean_t1w="/absolute/path/clean_T1w.nii.gz",  # 已回归、重采样到 T1w 的 4D BOLD
    clean_mni="/absolute/path/clean_MNI152NLin6Asym_2mm.nii.gz",  # 同一 4D BOLD 的 MNI 2 mm 版本
    left=left,  # 左侧注册后的 SurfaceHemisphere
    right=right,  # 右侧注册后的 SurfaceHemisphere
    left_label="/absolute/path/L.atlasroi.32k_fs_LR.shape.gii",  # 左侧非内侧壁标签
    right_label="/absolute/path/R.atlasroi.32k_fs_LR.shape.gii",  # 右侧非内侧壁标签
    hcp_dseg="/absolute/path/fmriprep/tpl-MNI152NLin6Asym_res-02_atlas-HCP_dseg.nii.gz",  # 皮层下分区
    output_dir="/absolute/path/surface/projection",  # 双侧 32k GIFTI 与 91k CIFTI 输出目录
    goodvoxels=None,  # 可选的 T1w 3D 体积采样掩膜
    wb_command="/absolute/path/bin/wb_command",  # Connectome Workbench 可执行文件
    overwrite=False,  # 目标 CIFTI 已存在时停止
)

cifti = create_fmriprep_cifti(
    clean_mni="/absolute/path/clean_MNI152NLin6Asym_2mm.nii.gz",  # 同一时间序列的 MNI 2 mm 4D BOLD
    left_metric=projection.left_metric,  # 左侧 32k 顶点时间序列 GIFTI
    right_metric=projection.right_metric,  # 右侧 32k 顶点时间序列 GIFTI
    left_label="/absolute/path/L.atlasroi.32k_fs_LR.shape.gii",  # 左侧保留的皮层顶点
    right_label="/absolute/path/R.atlasroi.32k_fs_LR.shape.gii",  # 右侧保留的皮层顶点
    hcp_dseg="/absolute/path/fmriprep/tpl-MNI152NLin6Asym_res-02_atlas-HCP_dseg.nii.gz",  # 皮层下 HCP 标签
    output_file="/absolute/path/surface/check.dtseries.nii",  # 独立组装得到的 CIFTI 文件名
)
print(cifti)  # 时间×灰质坐标，TR 来自 clean_mni 的 NIfTI 头
```

已有 `run_surface_from_mni`/`run_surface_projection` 是独立的 MNI 体积投影接口，包含 fsLR 平滑和皮层下重采样，**不是当前高层 fMRIPrep 路径**；调用它们时输入和输出约定见函数 docstring。现有的结构偏置校正与髓鞘图函数也不参与这条路径。

## 真实数据对照

测试使用一例真实 UKB 静息态 BOLD：490 帧，TR 0.735 秒。固定回归后 BOLD、T1w white/pial/midthickness、sMRIPrep 规则生成的 cortex mask、MNI BOLD、TemplateFlow 标签与 HCP dseg。皮层直接用 T1w BOLD；皮层下从 MNI BOLD 取值。公开的[标量摘要](../../validation/fmri/fmriprep_surface_summary.json)不含影像、逐顶点值或被试标识。fMRIPrep 及 NiWorkflows 参照代码均固定提交与 SHA-256；官方命令只在隔离的对照环境执行，FNIT 运行时不导入它们。

| 对照 | 结果 |
|---|---:|
| T1w white、pial、中面、初始球面、左右原生 cortex mask 与先前独立结构准备结果 | 所有顶点与 ROI 逐值相同；结构准备 22.69 秒 |
| 固定官方 MSM 球面：高层入口真实前 8 帧 vs 同一 Workbench 命令处理的 490 帧前 8 帧 | 91,282 灰质坐标；最大绝对差 0；高层入口含 ZIP 提取 73.22 秒 |
| 固定官方 MSM 球面：490 帧 FNIT CIFTI vs 从 [NiWorkflows 官方源码](https://github.com/nipreps/niworkflows/blob/0eb323521639665483f651cf088c86047382d656/niworkflows/interfaces/cifti.py)提取的组装函数 | 坐标轴和所有数值完全相同；最大绝对差 0；90,553 个非常数坐标的时间相关均值 1.0000 |
| 490 帧 CIFTI 组装耗时：FNIT / NiWorkflows 官方函数 | 19.57 / 19.14 秒，同一主机；未观察到有意义的提速 |

只替换注册球面、保持其余输入和 Workbench 命令一致时，下面统计的是**先对每个灰质坐标的 490 帧分别算 Pearson r，再对有效坐标求均值**。两边均为常数的坐标不参与相关均值；MAE 使用所有对应值，强度单位继承 BOLD。

| 区域：FNIT PyTorch MSMSulc vs 官方 MSM 球面 | 灰质坐标 | 有效时间相关 | 平均 r | 中位 r | MAE |
|---|---:|---:|---:|---:|---:|
| 左皮层 | 29,696 | 29,688 | 0.8045 | 0.8530 | 35.99 |
| 右皮层 | 29,716 | 29,667 | 0.7814 | 0.8450 | 40.43 |
| 皮层下 | 31,870 | 31,173 | 1.0000 | 1.0000 | 0 |
| 全部 | 91,282 | 90,528 | 0.8642 | — | 24.87 |

### MSMSulc 差异定位

两次注册的初始球面顶点坐标与原生 sulc 数组逐值相同，参考球面和 sulc 也相同。[sMRIPrep 的 MSMSulc 工作流](https://github.com/nipreps/smriprep/blob/6a83b4686953273a0379a5e8aa76bb3d82bf4c3f/src/smriprep/workflows/surfaces.py)使用的 `MSMSulcStrainFinalconf` 与本次 HCP 官方对照的参数逐行相同，仅末尾空行不同。官方先做 NMI 仿射，再以脑沟相关性进行三层离散优化，控制网格依次为 162、642、2,562 个顶点，并使用高阶应变约束。FNIT 当前先优化整体旋转，再以固定 2,562 个 Fibonacci 控制点、四层 sulc 平方误差、连续 Adam 及控制三角形剪切和面积惩罚做局部配准。该惩罚改善形变，但目标函数、插值和求解方法仍与官方不同。[官方 MSM 参数说明](https://fsl.fmrib.ox.ac.uk/fsl/docs/registration/msm.html)解释了相关性、控制网格和正则项的作用。

独立运行官方配置的阶段截断：只做仿射时，左、右球面距官方最终球面的中位角差为 1.591°、2.045°；再完成第一层离散优化后，左侧仍有 0.907°。误差主要在后续离散形变中产生，单独调整旋转不能实现最终球面等价。

| 同一真实被试 | 左侧 | 右侧 |
|---|---:|---:|
| FNIT 与官方球面逐原生顶点角度差：中位数 / 第 95 百分位 | 0.905° / 2.035° | 0.972° / 2.373° |
| 注册后 sulc 与模板 sulc 的顶点相关：FNIT / 官方 | 0.843 / 0.846 | 0.836 / 0.843 |
| 球面边长绝对对数形变第 95 百分位：FNIT / 官方 | 0.135 / 0.129 | 0.142 / 0.128 |

固定官方球面后，投影及 CIFTI 逐值一致，皮层下始终一致；当前皮层差异来自球面对应关系。

同一主机上，FNIT 双侧 GPU 注册墙钟时间为 65.06 秒，峰值显存分别为 0.298、0.315 GB，保存后的球面无折叠三角形；官方 MSM 的左、右 CPU 墙钟时间为 46 分 39 秒、46 分 42 秒。两边使用不同硬件和优化算法，这些时间不能作为等精度加速比。复用已生成的个体 T1w BOLD 和 FNIT 注册球面，490 帧表面投影与 CIFTI 组装耗时 447.04 秒，包含结构准备和 Workbench 投影；EPI→T1w 重采样另计。固定输入的 Workbench ribbon 与 dilation 两侧分步计时之和为 289.56 秒；严格 cortex mask、ADAP_BARY_AREA 和 atlas mask 之和为 119.99 秒。这些分步数字来自独立运行。参数在这同一被试上筛选，尚无独立被试的复核；FNIT 的上游 BBR、MNI 配准与去噪也和 fMRIPrep 不同。**当前默认球面可生成可分析的 fsLR32k/91k 时间序列，尚未达到官方 MSM 的数值等价。**
