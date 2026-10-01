# fMRI 单 run 处理代码

`fMRIVolume_pipeline` 从原始 BIDS BOLD、可选 SBRef 和同源 T1w 同时生成两类结果：T1w 原生 BOLD 分辨率与 MNI152 2 mm 的 `preproc`，以及个体 EPI 与 MNI 2 mm 的 ICA-AROMA `clean`。`fMRISurface_pipeline` 接续同 run 的 volume、已有 recon-all 几何及 HCP 资源，默认读取 `preproc`，生成 fsLR32k GIFTI、91k CIFTI、注册球面与 QC；`signal="clean"` 显式选择去噪分支。

## 公共入口与代码分工

| 模块 | 职责与复用关系 |
|---|---|
| [`end_to_end.py`](end_to_end.py) | volume 编排、preproc/clean 分支、变换保存与来源记录；返回 `FMRIVolumeResult`。 |
| [`surface_pipeline.py`](surface_pipeline.py) | surface 输入身份校验、球面配准、投影及整批发布；返回 `FMRISurfaceResult`。 |
| [`cli.py`](cli.py)、[`__init__.py`](__init__.py) | `fnit-fmri volume/surface` 与公共 Python 导出。 |
| [`bids.py`](bids.py)、[`derivatives.py`](derivatives.py) | 单 run 选择、BIDS 元数据继承、派生路径与文件发布。 |
| [`pipeline.py`](pipeline.py) | FEAT 子入口 `run_feat_core`；运动估计直接复用独立 `TorchMCFLIRT`，缩放与高通复用 `fnit.feat.temporal`。 |
| [`_anatomical.py`](_anatomical.py)、[`bbr.py`](bbr.py)、[`normalization.py`](normalization.py) | SynthStrip、严格扫描顺序 TorchFAST、解剖缓存、BBR、SynthMorph/TorchFNIRT 与空间重采样。 |
| [`aroma_pipeline.py`](aroma_pipeline.py)、[`aroma.py`](aroma.py)、[`confounds.py`](confounds.py) | 独立 FNIT MELODIC/PICA、ICA-AROMA 分类与回归、可选混杂回归。 |
| [`timing.py`](timing.py)、[`slice_timing.py`](slice_timing.py)、[`sampling_reference.py`](sampling_reference.py)、[`spatial.py`](spatial.py) | STC 参数、可选切片校正、原生 BOLD 分辨率参考网格及一次插值坐标合成。 |
| [`surface_prepare.py`](surface_prepare.py)、[`surface_fmriprep.py`](surface_fmriprep.py)、[`surface.py`](surface.py) | 已有表面到 T1w 世界坐标、Workbench ribbon/重采样与 CIFTI 组装、表面数据契约；后者不再包含旧 HCP 投影实现。 |
| [`assets_setup.py`](assets_setup.py) | 固定资源清单与下载；HCP 按 SHA-256 校验，三项 TemplateFlow 文件同时校验大小与 SHA-256，后者只从原站获取。 |
| [`motion.py`](motion.py) | 原 `estimate_motion` 等接口的兼容层；调用同一 TorchMCFLIRT，不维护另一套优化器。 |

`TorchFAST(execution="fsl")` 表示 FNIT 内部的 PyTorch 扫描顺序模式，不启动 FSL。模块名 `surface_fmriprep` 表示参考算法来源，不导入或调用 fMRIPrep/Nipype。运行时唯一外部表面计算程序是主页 Conda 环境中的 Connectome Workbench；重建目录只作为已有几何输入读取。默认保留 float32/TF32，不引入低精度或新依赖。

## 用法、真实验证与最近修复

按统一七项结构阅读[volume 文档](../../../docs/fmri/README.md)和[surface 文档](../../../docs/fmri/surface.md)：功能/流程、Python 与输入输出、CLI、原软件调用、最新真实精度与耗时/脑图、版本与 benchmark 记录、参考来源。真实整链、固定输入控制及历史测量分开记录在[验证索引](../../../validation/fmri/README.md)。

本次修复 FEAT 子入口覆盖较短 run 时遗留旧 `MAT_*` 运动矩阵的问题。只有当前拟合成功后，才替换本函数写出的矩阵文件，保留其他用户文件；不改变运动估计、矩阵定义和 BOLD 插值。触发条件、回归检查及既有真实精度/耗时见[FEAT 子功能说明](../../../docs/fmri/feat.md)。完整 volume 每次使用独立临时 FEAT 目录，原整链 benchmark 不受此旧文件问题影响。
