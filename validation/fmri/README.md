# fMRI 验证索引

当前完整测量冻结到 `6f67cc06ee8c5108ef3640cbfc289f3e5a742f65`。在同一例真实 UKB 原始 BOLD、SBRef、匹配存档 T1 和已有同源表面重建上，连续运行 volume 和 surface 公开 API；生成 preproc、clean 及 91k CIFTI。完整实际墙钟 **915.876 s**，扣除单独记录的中间结果捕获后估计 **883.340 s**。volume 为 **649.045 / 616.561 s**，surface 为 **266.814 / 266.777 s**，前者是实际观测、后者是捕获扣除估计。

本轮对照分为同步骤原 FSL/FreeSurfer 指令和独立完整 fMRIPrep 原流程。各自的原始输入、输出身份、全部 490 帧精度、端到端与分步骤时间，以及脑图集中在[当前完整 benchmark](e2e_latest/README.md)。共享机器上的一次测量不能推广为固定加速比。已有 recon-all 的首次结构重建不在 FNIT 计时内；原流程实际执行的额外结构处理按原流程记录。

| 当前证据 | 用途 |
|---|---|
| [发布清单](e2e_latest/publication.public.json) | 绑定本次发布的报告、图示、脚本和计算代码；列出合并后测试及两项无关eddy更新。 |
| [完整 FNIT 执行](e2e_latest/fnit_main.public.json) | 两个公开入口连续运行的实际墙钟、捕获开销、细分阶段、全部输出检查及源码/输入/输出 SHA。 |
| [MSM 扩展构建](e2e_latest/fnit_main_build.public.json) | 实际加载的本包 FastPD 扩展、构建源码与 flags；构建不计入调用耗时。 |
| [进程完成状态](e2e_latest/fnit_main_process.public.json) | 本次成功运行的退出码。 |
| [共享 GPU 观察](e2e_latest/fnit_main_gpu_load.public.jsonl) | 包括导入、哈希与事后 QC 的进程期采样；不代表各阶段独占 GPU。 |
| [最新同步骤精度](e2e_latest/native_matched_main.public.json) | 当前main对同原始输入的原FSL/FS：22幅阶段影像、运动/仿射/形变与最终preproc/clean。 |
| [原完整clean执行](e2e_latest/native_clean.public.json) | 实际2666.886 s，另存严格退出证明与独立MELODIC控制。 |
| [独立完整fMRIPrep](e2e_latest/native_fmriprep_full_strict__run.public.json) | 6087.099 s、成功整链、额外FS/ANTs和全部最终产物。 |
| [完整volume精度](e2e_latest/full_volume_main.public.json) / [surface精度](e2e_latest/surface_full_main.public.json) | 同raw、全部490帧；MNI无损方向对齐，真实最终source SHA门禁。 |
| [两套原volume协议](e2e_latest/original_protocol_comparison.public.json) / [surface协议](e2e_latest/surface_two_original_protocols.public.json) | 原软件之间直接对照，量化不同完整方法已有的差异。 |
| [初始化尝试](e2e_latest/fnit_initialization.public.json) | 两次发生在公开 API 前的 CUDA 初始化失败单独保留，未计作完成 benchmark。 |
| [volume 功能页](../../docs/fmri/README.md#latest-real-benchmark) | 输入输出、Python、CLI、原软件步骤与当前实测。 |
| [surface 功能页](../../docs/fmri/surface.md#surface-e2e-latest) | preproc 输入、MSM、投影、CIFTI 和原软件比较。 |

## 复测入口与统计方法

[benchmark_combined_e2e.py](benchmark_combined_e2e.py)顺序调用 `fMRIVolume_pipeline` 和 `fMRISurface_pipeline`，新建 derivatives、不复用解剖缓存、不提供注册球面。捕获观察器只记录已有函数边界并保存结果，不改变计算；嵌套函数计时和并行半球时间不相加。完整变量与复测命令见[本轮方法](e2e_latest/README.md)。

[run_native_matched_pipeline.py](run_native_matched_pipeline.py)连续执行原 SynthStrip、FSL 和作者 ICA-AROMA 的 clean 链。原指令及信号定义见[同步骤协议](matched_native.md#数据协议与原命令)。[run_native_preproc.py](e2e_latest/run_native_preproc.py)用本轮原软件新估计的运动、BBR、FNIRT，从原始 4D BOLD 一次插值至 T1w/MNI；它是单独采样阶段，不是另一份 raw→CIFTI 连续耗时。

[compare_matched_pipeline.py](compare_matched_pipeline.py)比较同步骤各阶段与最终 clean；[compare_volume_e2e.py](compare_volume_e2e.py)比较双方独立 raw→MNI preproc；[compare_surface_e2e.py](compare_surface_e2e.py)比较完整皮层和皮层下信号。比较文件须通过执行报告中的输入/输出 SHA 身份校验，保持全部帧、空间网格与 TR，不追加平滑、强度拟合或临时配准。时间 r 使用 float64、排除去均值 RMS≤1e-6 的常数序列，误差保留实际保存强度；固定统计域及覆盖计数随报告保存。

## 子函数控制

| 控制 | 来源与边界 |
|---|---|
| 独立 MCFLIRT 精确缓存/CUDA graph | [最新配对报告](../mcflirt/paired_exact_latest.public.json)：两版矩阵、参数及校正值逐 bit 一致；完整 raw 流程另见本轮报告。 |
| BBR 与 T1→MNI | [当前配准报告](registration_gpu.current.public.json)及 [BBR](../../docs/fmri/bbr.md)、[normalization](../../docs/fmri/normalization.md)；计时范围与整链不同。 |
| 单次样条采样与边界 | [全部 490 帧原节点控制](fmriprep/actual_node_interpolation_full490.public.json)；固定官方变换控制，不能代替独立估计的整链精度。 |
| 切片时间校正 | [真实 490 帧控制](fmriprep/stc_real_full490.public.json)；本轮关闭 STC，未将可选 STC 的误差掩入本轮结果。 |
| MSM 与固定投影 | [优化控制](surface_gpu_parallel/README.md)；同输入算子结果与独立估计的配准分别解释。 |
| FEAT 覆盖与接口 | [整理记录](organization_20261001.public.json)：85 个节点及真实 8→2 帧覆盖检查；文件管理检查不是性能 benchmark。 |

## 最近记录

| 冻结版本 / 日期 | 报告 |
|---|---|
| `6f67cc0` / 2026-10-02 | [当前连续 volume＋surface 与原软件对照](e2e_latest/README.md)；对更新前ac692bb的39项科学输出逐bit一致。 |
| `ac692bb` / 2026-10-02 | [更新前连续报告](e2e_latest/fnit_combined.public.json)，实际832.586 s / 扣除捕获估计793.314 s；保留原冻结源码身份。 |
| `cfb7beee` / 2026-10-01 | [此前 volume 707.287 s](mcflirt_optimization.md)，包含 preproc＋clean；原同步骤 MNI clean 时间 r 均值 0.939162。 |
| `7102c187` / 2026-10-02 | [此前独立 surface](surface_e2e/README.md)和[并行更新](surface_gpu_parallel/README.md)；不能把旧 volume 与旧 surface 时间相加当连续运行。 |
| `1eb9c417` / 2026-10-01 | [早期原同步骤 clean](matched_native.md)；首次模块数值修正与后续优化分别保留来源。 |

更早的 [clean](HISTORY_20261001_clean.md)、[STC 开启](HISTORY_20261001_STCON_PREPROC.md)、[STC 关闭](HISTORY_20261001_STCOFF_PREPROC.md)与 [2026-09-30](HISTORY_20260930.md)是固定旧源码记录，不代表当前精度。DeepPrep 的单独参考在[对应页面](deepprep/README.md)，其输入与输出范围不同。

公开内容限于匿名聚合、文件哈希和经用户授权的标准空间/表面 PNG。原始影像、个体标签、完整四维影像、表面坐标及逐体素统计图留在服务器。
