# FNIT MSMSulc

`fnit.msm` 独立完成双侧脑沟球面配准。先按 newMSM 的有限差分规则估计刚性初始化，再用 162→642→2,562 个控制点优化脑沟相似度和三角形应变；HOCR 降阶与 FastPD 选择联合位移。输出球面保持原生顶点顺序。运行时不调用官方 MSM 或 FreeSurfer；准备阶段使用 Connectome Workbench。当前功能为 **MSMSulc**。

默认使用 HCP/sMRIPrep 的四级配置：`simval=3,2,2,2`，最大迭代数 `50,10,15,15`。官方 newMSM 将历史仿射相似度值 3 转为 Pearson 2；FNIT 保持该行为。几何、相似度和优化标量使用 float64，写出的 GIFTI 顶点为 float32；没有使用 FP16/BF16。刚性坐标与离散成本在 PyTorch 上计算，刚性加权相似度及 HOCR/FastPD 使用包内独立 C++ 算子。默认优化路径缓存固定几何并合并传输，`execution="reference"` 保留逐块检查供回归对照；二者使用相同的算法和停止条件。

## 输入与调用

先用同一 recon-all 结果和 `fnit-setup-fmri-surface-assets --output-dir /absolute/path/hcp_surface_assets --fmriprep` 准备 HCP 参考文件。需要 `surf/lh|rh.{white,pial,sphere,sphere.reg,sulc,thickness}` 和 `mri/orig/001.mgz`；T2w、FLAIR 不参与此配准。`prepare_fmriprep_surface_inputs` 产生的 `initial_spheres` 分别是左、右 FS→fsLR 初始球面，顶点顺序与原生 mesh 相同。

```python
from fnit import prepare_fmriprep_surface_inputs
from fnit.msm import MSMSulcConfig, prepare_msmsulc_inputs, run_msmsulc

prepared = prepare_fmriprep_surface_inputs(
    subject_dir="/absolute/path/recon-all/sub-0001",  # 已完成 T1w recon-all 的目录
    hcp_assets_dir="/absolute/path/hcp_surface_assets",  # HCP fsLR 资源目录
    output_dir="/absolute/path/work/initial",       # FS→fsLR 初始球面及 ROI 工作目录
    wb_command="wb_command",                         # Workbench 命令
    overwrite=False,                                 # 是否替换准备结果
)
inputs = prepare_msmsulc_inputs(
    subject_dir="/absolute/path/recon-all/sub-0001",  # 与上一步相同的 recon-all 目录
    initial_spheres=prepared.initial_spheres,         # (左球面, 右球面)，原生顶点顺序
    hcp_assets_dir="/absolute/path/hcp_surface_assets",  # HCP 164k 参考球面和 sulc
    output_dir="/absolute/path/work/msm-inputs",    # 双侧球面、sulc、仿射矩阵工作目录
    wb_command="wb_command",                         # Workbench 命令
)
spheres = run_msmsulc(
    inputs=inputs,                                    # {"L": MSMSulcInputs, "R": MSMSulcInputs}
    output_dir="/absolute/path/work/msm-output",    # 注册球面与 JSON 报告目录
    device="cuda:0",                                 # PyTorch 计算设备，也可为 cpu
    config=MSMSulcConfig(),                           # 默认 HCP 四级配置，也可填官方配置文件路径
    execution="optimized",                          # 缓存和合并传输；reference 用于执行方式对照
)
print(spheres["L"])  # L.sphere.MSMSulc.native.surf.gii
print(spheres["R"])  # R.sphere.MSMSulc.native.surf.gii
```

`MSMSulcInputs` 每侧包括 `native_sphere`、`rotated_sphere`、`native_sulc`、`reference_sphere`、`reference_sulc`、`affine` 六个绝对路径，分别保存原生球面、FS→fsLR 旋转球面、原生脑沟图、参考球面、参考脑沟图和初始旋转矩阵。`run_msmsulc` 返回 `{"L": Path, "R": Path}`；每个 `.surf.gii` 含 N×3 顶点坐标及 F×3 三角形索引，可直接用于 Workbench 表面重采样。

`registration_report.json` 记录实际配置、仿射角度、逐级能量、更新数量、停止位置、展开操作、耗时、峰值已分配显存和写出折叠数。球面可传给 `fMRISurface_pipeline(registered_spheres=(spheres["L"], spheres["R"]))` 做固定球面投影对照；使用已注册球面时不再指定 `msm_config`。独立配准输出是工作文件，最终 fMRI 时间序列由 surface 流程写成 BIDS Derivatives。

离散迭代中的 DATA 和控制网格保留 newMSM 的展开处理。最后从 DATA 变形到原生球面后，按官方行为写出有限坐标，并分别报告求解坐标和实际 float32 GIFTI 的翻折数、最小方向比及输入退化面数。最终原生球面不再额外优化；使用前应检查这份质控报告。

### 配置参数

下面的四元组依次对应刚性初始化和三个离散阶段。`config=None` 与 `MSMSulcConfig()` 相同；路径输入通过 `MSMSulcConfig.from_file` 读取受支持的官方选项。解析器拒绝改变为 MSMAll、MCMC 等未实现的流程。

| 参数 | 默认值 | 含义 |
|---|---|---|
| `simval` | `(3, 2, 2, 2)` | 相似度；1 为 SSD，2 为 Pearson，刚性阶段的 3 按官方行为转为 2。 |
| `iterations` | `(50, 10, 15, 15)` | 每阶段最大迭代数，离散阶段按官方能量条件提前停止。 |
| `control_grid` | `(6, 2, 3, 4)` | icosphere 控制网格级别；离散控制点数为 162、642、2,562。 |
| `sampling_grid` | `(6, 4, 5, 6)` | 位移标签取样网格级别。 |
| `data_grid` | `(6, 4, 5, 6)` | 脑沟特征网格级别；离散数据点数为 2,562、10,242、40,962。 |
| `regularization` | `(0, 10, 7.5, 7.5)` | 应变正则项权重。 |
| `affine_step_size` | `0.01` | 刚性 Euler 更新的初始步长，单位为弧度。 |
| `affine_gradient_spacing` | `0.5` | 刚性有限差分的初始角度间隔，单位为弧度。 |
| `shear_modulus` / `bulk_modulus` | `0.4` / `1.6` | 三角形形状与面积应变权重。 |
| `strain_exponent` / `regularization_exponent` | `2` / `2` | 应变势和正则项的指数。 |

`MSMSulcConfig.ssd_affine()` 只将刚性阶段改为 SSD，即 `(1,2,2,2)`，用于复测采用该配置的参照。比较双方必须使用同一配置。

浮点配置按官方 `Option<float>` 的精度读取，再提升为 double 计算。因此步长 `0.01` 的有效值为 `0.009999999776482582`，shear/bulk 的有效值分别为 `0.4000000059604645` 和 `1.600000023841858`；报告保存这些实际数值。

官方配置的 `--numthreads=N` 是 CPU 执行参数，FNIT 识别该字段并使用所选 PyTorch 设备；旧 `--threads=N` 写法仍可读取。配置文件支持本页的 MSMSulc 选项，不覆盖 newMSM 的其他注册算法。

## 原版对照命令

以下命令只在独立基准环境中运行；`--conf` 为 HCP 安装包内对应的 MSMSulc 配置文件，按实际路径填写。FNIT 不调用此命令。

```bash
newmsm --inmesh=/absolute/path/work/msm-inputs/L.sphere_rot.surf.gii \
  --refmesh=/absolute/path/hcp_surface_assets/global/templates/standard_mesh_atlases/fsaverage.L_LR.spherical_std.164k_fs_LR.surf.gii \
  --indata=/absolute/path/work/msm-inputs/L.sulc.native.shape.gii \
  --refdata=/absolute/path/hcp_surface_assets/global/templates/standard_mesh_atlases/L.refsulc.164k_fs_LR.shape.gii \
  --conf=/absolute/path/hcp_surface_assets/MSMConfig/MSMSulcStrainFinalconf \
  --out=/absolute/path/reference/L.
```

## 真实数据基准

同一例真实 UKB 的双侧初始球面、sulc、HCP 模板及完整四级配置用于双方。正式精度参照为 `fsl-newmsm 1.0 h442c261_5` 的单线程输出；同输入两次独立运行的左侧球面逐位相同。官方 8 线程耗时单列，其重复球面有差异。FNIT 使用 H100；球面角差按原生对应顶点计算，490 帧时间相关先逐灰质坐标计算 Pearson r，再取均值。

最新球面、fsLR32k 时间序列、冷/热调用及 profile 汇总见 [MSM 验证页](../../validation/msm/README.md)。配准包含读取和球面/报告写盘，不包含 BOLD 投影、Python 导入与 CUDA 上下文初始化。历史完整 surface API 时间见 [全流程报告](../../validation/fmri/README.md)，不能与本次单函数时间混加。

本次定位到三个会改变配准轨迹的问题：坐标归一化后重算了官方缓存的三角形面积；官方浮点配置被直接作为 double 使用；刚性 WLS 的 GPU 除法和指数舍入使成本出现 1 ULP 分歧。前两项已在独立 MSM 子函数修复，最后一项用 FNIT 的 C++ 运算保持官方顺序。真实检查点验证了全部 77 次刚性成本及对应逐顶点值；它们的验证范围与最终球面结果分别记录。

固定官方球面时，Workbench 投影与 CIFTI 组装对独立对照逐值一致，见 [固定球面报告](../../validation/fmri/surface_fixed_sphere.public.json)。pipeline 传递注册配置，并将写出球面的质控保留在 BIDS sidecar。

## 参考文献与原实现

- Robinson 等，*Multimodal surface matching with higher-order smoothness constraints*，NeuroImage，2018，[DOI](https://doi.org/10.1016/j.neuroimage.2017.10.037)。
- Ishikawa，*Transformation of General Binary MRF Minimization to the First-Order Case*，IEEE TPAMI，2011，[DOI](https://doi.org/10.1109/TPAMI.2010.91)。
- Komodakis 等，*Fast, Approximately Optimal Solutions for Single and Dynamic MRFs*，CVPR，2007，[论文](https://www.csd.uoc.gr/~tziritas/papers/CVPR07_FastPD.pdf)。
- 原实现：[newMSM](https://github.com/rbesenczi/newMSM)、[HCP Pipelines](https://github.com/Washington-University/HCPpipelines)、[Connectome Workbench](https://github.com/Washington-University/workbench)。
