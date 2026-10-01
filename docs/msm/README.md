# FNIT MSMSulc

`fnit.msm` 独立完成双侧脑沟球面配准。先按 newMSM 的有限差分规则估计刚性初始化，再用 162→642→2,562 个控制点优化脑沟相似度和三角形应变；HOCR 降阶与 FastPD 选择联合位移。输出球面保持原生顶点顺序。运行时不调用官方 MSM 或 FreeSurfer；准备阶段使用 Connectome Workbench。当前功能为 **MSMSulc**。

默认使用 HCP/sMRIPrep 的四级配置：`simval=3,2,2,2`，最大迭代数 `50,10,15,15`。官方 newMSM 将历史仿射相似度值 3 转为 Pearson 2；FNIT 保持该行为。几何、相似度和优化标量使用 float64，写出的 GIFTI 顶点为 float32；没有使用 FP16/BF16。默认优化路径缓存固定几何并合并传输，`execution="reference"` 保留逐块检查供回归对照；二者使用相同的算法和停止条件。

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

一例真实 UKB 被试的左右初始球面、sulc 和 HCP 模板相同。官方 newMSM 使用 8 个 CPU 线程；FNIT 在 H100 上运行。球面角差以原生对应顶点计算；时间是单侧配准墙钟时间。两端硬件不同，耗时不构成等精度加速比。

| 对照 | 左侧 | 右侧 |
|---|---:|---:|
| FNIT vs 官方注册球面角差中位数 / 第 95 百分位 | 0.383° / 1.060° | 0.407° / 1.018° |
| FNIT / 官方单侧耗时 | 138.59 / 312.61 秒 | 123.70 / 241.95 秒 |
| 490 帧 fsLR32k 逐顶点时间相关均值 | 0.9414 | 0.9420 |
| 490 帧 fsLR32k MAE | 18.20 | 19.71 |
| FNIT 峰值 PyTorch 已分配显存 | 0.726 GB | 0.727 GB |
| FNIT 写出球面折叠面片 | 0 | 0 |

同一组输入连续运行官方 newMSM 两次，其自身球面角差中位数为左 0.394°、右 0.353°，皮层时间相关均值为 0.943、0.954。FNIT 数值尚未与官方逐点一致；主要待对齐的是每片数据项和离散优化轨迹。固定官方球面时，FNIT Workbench 投影和 CIFTI 组装已逐值匹配对照。标量记录见[当前摘要](../../validation/fmri/surface_current.json)。

## 参考文献与原实现

- Robinson 等，*Multimodal surface matching with higher-order smoothness constraints*，NeuroImage，2018，[DOI](https://doi.org/10.1016/j.neuroimage.2017.10.037)。
- Ishikawa，*Transformation of General Binary MRF Minimization to the First-Order Case*，IEEE TPAMI，2011，[DOI](https://doi.org/10.1109/TPAMI.2010.91)。
- Komodakis 等，*Fast, Approximately Optimal Solutions for Single and Dynamic MRFs*，CVPR，2007，[论文](https://www.csd.uoc.gr/~tziritas/papers/CVPR07_FastPD.pdf)。
- 原实现：[newMSM](https://github.com/rbesenczi/newMSM)、[HCP Pipelines](https://github.com/Washington-University/HCPpipelines)、[Connectome Workbench](https://github.com/Washington-University/workbench)。
