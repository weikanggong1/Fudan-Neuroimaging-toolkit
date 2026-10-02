# 2026-10-01：历史 STC 开启的 preproc 与 clean 测量

这两次真实完整 490 帧运行均开启 STC（reference=0.5）、关闭 SDC，同时生成 preproc 和 clean。它们保留当时源码 SHA、输出与计时，不描述当前默认 STC 关闭的运行。

## 合并后的完整 volume 运行

`3940a72` 包含 `9e77375` 的 volume、BBR、FNIRT 和缓存更新。在共享 H100 PCIe 上以 8 个 CPU 线程、float32/TF32、20 GB CUDA 额度运行公开单被试 API，首次生成完整 `clean` 与 `preproc`。实际源码、输入和输出哈希见 [合并后 volume 报告](fmriprep/fnit_volume_3940a72.public.json)。随后增加的共享 T1 输出复用检查和 surface 元数据检查没有包含在这次计时快照中。

| 项目 | 实测 |
|---|---:|
| 公开 API 墙钟，含读写和清理 | 758.198 s |
| 验证进程墙钟，另含导入、哈希与检查 | 812.27 s |
| PyTorch peak allocation / reserved | 13.305 / 16.981 GB |
| T1w preproc 网格 | 59×74×64×490，原生 BOLD 分辨率 |
| MNI152NLin6Asym preproc 网格 | 91×109×91×490，2 mm |
| 输出 | 四份 BOLD 均为 float32、有限值、490 帧、TR 0.735 s |

FEAT 核心 221.695 s，PICA/AROMA/混杂回归 123.547 s，clean MNI 重采样 51.207 s，T1w/MNI preproc 单次插值与写盘 320.732 s。自动 ICA 得到 96 个成分，71 次迭代收敛，AROMA 标记 51 个噪声成分；这些操作只作用于 `clean`。完整时间包含两类信号的计算，不能视为仅 `preproc` 的耗时。

## 合并前的完整 volume 运行

2026-10-01 以 `f12dba7` 为基底的待发布源码在共享 H100 PCIe 上调用公开 volume API，使用 8 个 CPU 线程、float32/TF32，进程 CUDA 额度为 20 GB。实际运行源码与输入/输出 SHA-256 见 [早期 volume 报告](fmriprep/fnit_volume.public.json)。这次运行早于合入 `9e77375` 的更新，保留为版本记录。此运行同时生成 `clean` 与 `preproc`，不能将其中的总时间解释为仅预处理的耗时。

| 项目 | 实测 |
|---|---:|
| 公开 API 墙钟，含读写和清理 | 793.28 s |
| 验证进程墙钟，另含导入、哈希与检查 | 850.23 s |
| PyTorch peak allocation / reserved | 13.34 / 16.98 GB |
| T1w preproc 网格 | 59×74×64×490，原生 BOLD 分辨率 |
| MNI152NLin6Asym preproc 网格 | 91×109×91×490，2 mm |
| 输出 | 两类信号均为 float32、有限值、490 帧、TR 0.735 s |

| 阶段 | 秒 |
|---|---:|
| SynthStrip | 9.19 |
| FEAT 核心 | 230.84 |
| FAST | 1.96 |
| BBR 与 T1→MNI | 28.82 |
| 混杂掩膜 | 0.21 |
| PICA、AROMA 与混杂回归 | 128.92 |
| clean MNI 重采样 | 50.79 |
| T1w/MNI preproc 单次插值与写盘 | 330.04 |

阶段计时不含全部发布与清理开销，以 API 墙钟为完整耗时。ICA 自动得到 96 个成分，70 次迭代收敛，AROMA 标记 54 个噪声成分；这些结果属于 `clean`，没有应用到 `preproc`。本表检查完整执行与输出合同，数值匹配需使用固定变换、固定球面和独立整链的相应对照。


## 历史固定输入表面投影

使用上述历史 STC 开启的 `preproc` 和共同几何、ROI、面积表面、初始化球面，单独运行表面投影与 91k CIFTI 组装。FNIT 候选为 **216.638 s**，安装于固定 25.2.4 镜像内的官方工作流为 **214.639 s**；双侧完整 490×32,492 GIFTI 与 490×91,282 CIFTI 逐值相等，最大绝对误差和 RMSE 均为 0，时间轴、BrainModelAxis、元数据及 21 个结构一致。两者计时边界不同，不用于推断完整流程加速比。见[候选报告](fmriprep/projection_candidate.public.json)与[官方报告](fmriprep/projection_reference.public.json)。

## 历史公开 surface API

`3940a72` 的完整 490 帧、STC 开启 `preproc` 运行公开 `fMRISurface_pipeline`，显式提供初始化球面。API **238.343 s** 包含几何准备、投影、CIFTI 组装和 **11 个持久输出**；验证输入复制 2.649 s 另记。QC 的 `MSM=None`，球面 `EstimatedHere=False`，CUDA allocation 为 0。见[API 报告](fmriprep/surface_api_provided.public.json)。

官方投影使用此 API 实际生成的全部几何、原生 ROI、32k 面积表面及 T1w/MNI BOLD，逐文件 SHA-256 匹配；完整 GIFTI/CIFTI、时间轴、BrainModelAxis 和内嵌元数据一致，最大绝对误差与 RMSE 均为 0。官方工作流 **223.795 s** 从准备完毕的输入开始，不用于与 API 推断加速比。这些控制未测试 MSM 估计或从原始 BIDS 开始的变换估计。见[同输入比较](fmriprep/surface_api_projection_paired.public.json)、[官方参考](fmriprep/reference_projection_3940_actual_api.public.json)与[实际输入 SHA-256](fmriprep/surface_api_actual_inputs.public.json)。
