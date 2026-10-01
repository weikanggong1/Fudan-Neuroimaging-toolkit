# 2026-10-01 STC 关闭的历史 preproc 验证

这些结果对应 `c3c921c` volume 与 `bac3c395` surface 的实际执行快照；后续合并更新后的完整测量见[当前验证页](README.md)。本页保留历史输入、源码、误差和计时，不将旧时间视为合并后 API 的测量。

## 固定快照 STC 关闭的完整 volume（`c3c921c`）

以固定源码 `c3c921cc4928c2c01fe4429ebc5da91546a6849b` 在物理 H100 1 运行全部 490 帧，8 个 CPU 线程、float32/TF32、20 GB CUDA 额度，新输出目录且不复用解剖缓存。启动时 GPU 空余 55,729 MiB、利用率 100%；这是共享服务器上的一次观测。实际退出为 0，四份 BOLD 均为 float32、全部有限，TR 0.735 s；实际配置确认 `slice_timing=False`。见[完整报告](fmriprep/fnit_volume_stcoff_c3c921cc.public.json)。

| 项目 | 实测 |
|---|---:|
| 公开 API 墙钟，含读写和清理 | **697.313 s** |
| 验证复制，已从 API 时间中扣除 | 50.106 s |
| 验证进程墙钟，含导入、复制、哈希和检查 | 799.48 s |
| PyTorch peak allocation / reserved | **13.305 / 16.981 GB** |
| T1w preproc 网格 | 59×74×64×490 |
| MNI152NLin6Asym preproc 网格 | 91×109×91×490，2 mm |

FEAT 核心 220.736 s，PICA/AROMA/混杂回归 124.935 s，clean MNI 重采样 48.076 s，T1w/MNI preproc 单次插值与写盘 257.034 s。完整 API 同时生成 preproc 和 clean；单次时间不推断稳定加速比。无缓存、多 run、元数据、插值、网格和 CIFTI 等合同见[该运行源码的 193 项合同门禁](fmriprep/nonmsm_contract_gate_c3c921cc.public.json)；后续 NFS 和外部 float64 场修复的[最终合同门禁](fmriprep/nonmsm_contract_gate.public.json)另列，wheel/sdist 源码与许可校验见[打包门禁](fmriprep/nonmsm_package_gate.public.json)。

## 固定快照 STC 关闭的公开 surface API 与同输入对照（`bac3c395`）

以成功 `c3c921c` volume 的完整 490 帧 `preproc` 为输入，`bac3c395` 的公开 `fMRISurface_pipeline` 显式读取共同 recon-all 几何和初始化球面。API **233.239 s**，含几何准备、投影、CIFTI 组装、11 个持久输出及临时目录清理；验证输入复制 **3.784 s** 另记。完整进程 **244.38 s**，另含导入、复制、哈希与检查。QC 的 `MSM=None`、球面 `EstimatedHere=False`，CUDA allocation 为 0；Workbench 执行表面投影。见[API 报告](fmriprep/surface_stcoff_bac3c395.public.json)。

第一次 `c3c921c` surface 尝试完成计算后，因 QC 读取仍持有临时 CIFTI 的内存映射，NFS 临时目录清理报 `EBUSY`，未成功返回。`bac3c395` 在 QC 读取时关闭内存映射；本次真实重测成功返回并清理完成，原[失败记录](fmriprep/surface_stcoff_c3c921cc_NFS_failed.public.json)保留。

官方固定 25.2.4 镜像的 fsLR 与 grayords 工作流重新读取本次成功 API 的实际 white、pial、graymid、原生 ROI、32k 面积表面、球面和 T1w/MNI BOLD，全部输入 SHA-256 逐项匹配。

| 输出 | 数值个数 | 最大绝对误差 / RMSE | 逐值一致 |
|---|---:|---:|---|
| 左 fsLR32k GIFTI，490×32,492 | 15,921,080 | 0 / 0 | 是 |
| 右 fsLR32k GIFTI，490×32,492 | 15,921,080 | 0 / 0 | 是 |
| 91k CIFTI，490×91,282 | 44,728,180 | 0 / 0 | 是 |

21 个 CIFTI 结构、时间轴、BrainModelAxis 和内嵌元数据一致；全部有限，91,251 个灰坐标随时间变化，31 个为恒定信号。FNIT 使用 Workbench 2.1.0，镜像内为 2.0.1。官方工作流 **165.374 s** 从已准备输入开始，与 API 计时边界不同。此控制验证投影与组装，不验证 MSM、运动/BBR/MNI 变换估计或原始 BIDS 整链等价。见[完整比较](fmriprep/surface_stcoff_bac3c395_projection_paired.public.json)、[官方报告](fmriprep/reference_projection_stcoff_bac3c395_actual_api.public.json)和[实际输入 SHA](fmriprep/surface_stcoff_bac3c395_actual_inputs.public.json)。

先前 STC 开启的 216.638/214.639 s 固定投影与 238.343/223.795 s 公开 API 控制已移至[历史记录](HISTORY_20261001_STCON_PREPROC.md)。

## 独立 OFF volume 的 MNI 全 490 帧差异

成功 FNIT `c3c921c` 与官方 25.2.4 从相同 raw BOLD/T1 分别估计 HMC、EPI→T1 BBR 和 T1→MNI，再生成同一 91×109×91×490 MNI 网格的原始强度 `preproc`。双方均保留 490 帧、TR 0.735 s，实际 sidecar 的 `SkullStripped=False`、`SliceTimingCorrected=False`。比较直接使用原始输出强度，不作尺度归一化、偏移或回归拟合。

| 范围 | 体素 / 数值 | 时间 r 均值 / 中位数 | RMSE / relative RMSE |
|---|---:|---:|---:|
| 完整 MNI 网格 | 902,629 / 442,288,210 | 0.316030 / 0.303131 | 939.179 / 0.215304 |
| 两方功能脑 mask 交集 | 218,795 / 107,209,550 | **0.634869 / 0.743893** | **1146.610 / 0.140257** |

FNIT mask 为 221,058 体素，官方为 250,625，交集 Dice 为 0.927721；双方 mask 和交集 SHA 均保存。时间 r 排除任一时序标准差 ≤1e-6 的常数体素，两域此次均为 0 个。完整场与共同脑内的最大绝对误差分别为 25,606.218 与 21,021.880。relative RMSE 定义为误差二范数除以官方信号二范数。

这些结果包含两方独立 HMC、BBR、解剖 mask 和归一化估计的差异（FNIT SynthMorph、官方 ANTs）；不能归因于插值器一项，也不支持完整 pipeline 数值等价。固定变换的单次插值与提供球面的投影控制各自报告。仅比较共同 MNI 网格，T1w 的不同网格没有直接逐值比较。详见[完整聚合报告](fmriprep/independent_mni_stcoff_preproc_comparison.public.json)。

## 原生 HMC 的有限只读诊断

旧 `c3c921c` 与官方 25.2.4 留存的 HMC 后原生 BOLD 在相同 88×88×64 网格比较全部 490 帧；尚未应用 EPI→T1 或 T1→MNI。共同脑 mask 为 94,631 体素，时间 r 均值 **0.954578**、中位数 **0.974913**，RMSE **247.383**、relative RMSE **0.030372**；完整网格 r 均值为 **0.958229**。原强度直接比较，没有拟合变换或尺度。见[聚合诊断](fmriprep/native_hmc_stcoff_c3c921cc_diagnosis.public.json)。

官方原始重采样输入与 raw 的 242,851,840 个有序数值逐位相同，490 帧顺序及 affine 相同；STC 关闭、dummy scans=0。两方 HMC 参考图内容与插值核不同；后续分别使用体积 WM 边界/FreeSurfer BBR 及 SynthMorph/ANTs。HMC 阶段较高的 r 没有将 MNI 差异归因到唯一实现，也未发现帧重排的直接证据。
