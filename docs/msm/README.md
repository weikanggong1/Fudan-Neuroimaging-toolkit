# FNIT MSMSulc

`fnit.msm` 独立完成双侧脑沟球面配准。`prepare_msmsulc_inputs` 从已有 T1w recon-all 的 `sphere`、`sulc` 和 HCP 参考图生成原生 sulc、旋转球面；`run_msmsulc` 按 162→642→2,562 个控制点逐级优化脑沟相似度和三角形应变，用 HOCR 降阶与 FastPD 选择联合位移，输出原生顶点顺序的注册球面。它不调用 FSL MSM 或 FreeSurfer 命令；准备阶段使用 Connectome Workbench。CUDA 默认允许 TF32。当前只实现 **MSMSulc**，不提供 MSMAll、髓鞘图或 DeDrift。

## 输入与调用

先用同一 recon-all 结果和 `fnit-setup-fmri-surface-assets --output-dir /absolute/path/hcp_surface_assets --fmriprep` 准备 HCP 参考文件。需要 `surf/lh|rh.{white,pial,sphere,sphere.reg,sulc,thickness}` 和 `mri/orig/001.mgz`；T2w、FLAIR 不参与此配准。`prepare_fmriprep_surface_inputs` 产生的 `initial_spheres` 分别是左、右 FS→fsLR 初始球面，顶点顺序与原生 mesh 相同。

```python
from fnit import prepare_fmriprep_surface_inputs
from fnit.msm import prepare_msmsulc_inputs, run_msmsulc

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
)
print(spheres["L"])  # L.sphere.MSMSulc.native.surf.gii
print(spheres["R"])  # R.sphere.MSMSulc.native.surf.gii
```

`MSMSulcInputs` 每侧包括 `native_sphere`、`rotated_sphere`、`native_sulc`、`reference_sphere`、`reference_sulc`、`affine` 六个绝对路径。`run_msmsulc` 读取旋转球面与两侧脑沟图，返回 `L`/`R` GIFTI 路径；`registration_report.json` 记录仿射角度、逐级控制点/数据点数、位移更新、耗时、显存和折叠修复。球面可传入 `fMRISurface_pipeline(registered_spheres=(spheres["L"], spheres["R"]))` 做固定球面投影对照。独立调用的工作目录是中间文件；最终 fMRI 时间序列由 surface 流程写成 BIDS Derivatives。

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
