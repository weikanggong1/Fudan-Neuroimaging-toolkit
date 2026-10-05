# fMRI CPU 官方对照与 GPU 回归

本轮起点为 `cc9402734faeba93b3a13c29932fa1392eaccf62`。该页记录当前执行协议；结果只有在完整运行结束、输出核对和源码绑定后才列入 benchmark。

## 范围和资源

| 子任务 | 功能范围 | 官方或独立参照 |
|---|---|---|
| 1 | MCFLIRT、运动包装、BBR | FSL MCFLIRT、FLIRT BBR |
| 2 | PICA、ICA-AROMA、两种 IC 回归、混杂回归 | FSL MELODIC、原 ICA-AROMA、fsl_regfilt、官方 AFNI 3dTproject；NumPy SVD 另作数值控制 |
| 3 | MS-HBM、体积投影和输出回投 | 固定 CBIG 提交的 MATLAB MS-HBM |
| 4 | MSM/MSMSulc/MSMAll、双侧表面和 CIFTI | 固定 newMSM、Workbench、fMRIPrep 表面链 |
| 5 | 体积管线、时间和空间处理、缓存及完整输出 | 固定 fMRIPrep 预处理与匹配原 FSL 阶段 |

CPU 使用 nodecw8 的 Intel Xeon Gold 6418H。每项分配八个互不重叠的物理核心；1 线程测试仅绑定该组中的一个核心，8 线程测试绑定完整核心组。Torch、Numba、OpenMP 和 BLAS 统一预算，关闭动态增线程。同组计时串行并持有独占锁。共享节点的其它作业负载仍会记录。

nodecw8 的原 MATLAB R2018b 因许可证主机绑定失败，未进入 CBIG/HCP 计算。已验证 nodecw10 的合法 MATLAB 实际启动和正常退出；这两项 MATLAB 对照迁至 nodecw10，FNIT 也在同一节点重新运行。该节点负载约 2,500，因此另列函数时间、进程墙钟与 CPU 时间，不将 nodecw8 的 FNIT 耗时与 nodecw10 的原版耗时相除。

GPU 对照使用已验证 H100 的显式 UUID，先记录设备负载。同一设备中的本轮计时顺序执行。默认 TF32；保留既有功能的精度策略和 float32 输出，不启用 float16/bfloat16。PyTorch 显存目标为 20,000,000,000 字节。

共享 H100 的完整 GPU 回归分别保留冻结版本：MCFLIRT／BBR v4 的 12 个进程、36 次调用和 18 组完整比较见[任务 1 报告](task01_motion_bbr/gpu_v4.public.json)；混杂回归最终可选 AFNI 模式的十次调用、六组完整比较见[任务 2 报告](task02_ica_aroma/gpu_projection_20261005.public.json)；MS-HBM 的四次完整 490 帧旧新对照见[任务 3 说明](task03_mshbm/README.md)；三种 MSM 配准的 12 次完整旧新对照见[任务 4 报告](task04_msm_surface/gpu_registration_abba_final_recovery_v1.public.json)。各项旧新输出已通过其完整数值门槛；源码版本、进程／API 时钟、初始化失败重试和显存由各报告单列。共享设备时钟只作为观测值。完整 volume 的 CPU16 与 GPU4 保持性对照已验收，见下方冻结源码结果。

## 2026-10-06 完整链保持性结果

最终 volume 测量绑定同一完整真实 180 帧 BOLD、原始 T1w 和固定资产，源码为 `6f624040 → 98019133`；默认 STC 关闭。两种配准后端分别完成 CPU1/8 的 first/warm 调用，共 16 次；八组新旧及八组首跑/缓存比较，每组十个科学输出共 395,140,404 个值逐值一致，完整头、网格、dtype、有限值与来源/输入/物理核心门槛通过。

| 后端 | CPU 线程 | 基线 first / warm | 候选 first / warm |
|---|---:|---:|---:|
| FNIRT | 1 | 1351.524 / 832.104 s | 851.694 / 405.339 s |
| FNIRT | 8 | 748.712 / 611.512 s | 581.172 / 432.309 s |
| SynthMorph | 1 | 1710.378 / 804.598 s | 1243.552 / 411.036 s |
| SynthMorph | 8 | 793.044 / 595.666 s | 617.190 / 405.428 s |

四次完整 cold GPU 调用也比较同一十科学输出的全部值，两个后端均精确相同。FNIRT 旧/新 API 为 597.814/599.190 s，SynthMorph 为 421.071/415.174 s；最高 Torch allocation/reservation 为 13.323/16.182 GB，采样本任务进程树峰值为 15.804 GB，均低于 20e9 bytes。共享 GPU 487 次采样均为 100%，时间保留为单次观察。

完整记录见 [CPU16](task05_volume/final_merged_cpu_v1.public.json)、[GPU4](task05_volume/final_merged_gpu_v1.public.json)与[功能页阶段表](../../docs/fmri/README.md)。合并上游 `d2237221` 的源码为 `2530650b`，两条已测 GPU 候选执行链的 103/101 个实际导入模块均与合并代码逐文件相同；[来源核查](task05_volume/final_source_bridge_20261006.public.json)将实际冻结树与 Git tracked 树的范围分开。本轮计时保留冻结提交标签。

原版 fMRIPrep 25.2.4 的 CPU1/CPU8 first 与新进程 workflow-cache 四次完整调用已完成，STC OFF、180 帧、launcher/payload 均退出 0。CPU1 first/cache 为 19400.443/1107.668 s，CPU8 为 2957.476/453.581 s；[保存节点 CPU1](task05_volume/official_saved_nodes_cpu1_v2.public.json)及[CPU8](task05_volume/official_saved_nodes_cpu8_v2.public.json)保留完整时钟和分组。其范围包含 confounds 和额外解剖模板配准，FNIT 的完整调用另含 PICA、ICA-AROMA 与 clean；本表是 FNIT 新旧保持性结果，不计算原版整链加速比。

### 尚未达到的范围

- MCFLIRT 某些可选阶段仍慢于同线程原版；默认 180 帧 CPU8 warm 为 36.542 s，原版 app 为 35.233 s。三轴展开候选实测内核收益约 0.135%，未采用；[完整说明](../../docs/mcflirt/CPU_BENCHMARK_20261004.md)保留结果。
- MS-HBM CPU8 原版标签逐值一致，CPU1 有一个顶点差异；更改权重的已测参数有 79 个顶点差异，旧、新 FNIT 一致，属于既有官方差异。两个独立 run 的真实输入仍待提供。
- MSMAll CA/CAT 所需的个体髓鞘图和偏置图仍待提供，未用代理图替代。
- 180 帧 fresh surface 的球面和时序仍有差异；准备、刚性和前两轮离散更新已核对。后续阶段继续定位，[任务 4](task04_msm_surface/README.md)保留实际误差；固定 490 帧几何投影与 CIFTI 则逐值一致。

## 验证规则

- 使用完整真实 180/490 帧影像与完整表面顶点，不裁短帧数、不减少优化迭代、ICA 抽样或空间密度。
- 默认关闭 slice timing；可选开启功能单独测量，不改变生产默认值。
- 初轮完整单次运行用于定位耗时、精度和失败原因；不将单次观察写成稳定中位数或加速比。
- 优化必须有具体 profile 依据。接受前固定候选源码，重复 AB/BA 核对相同输入、线程、设置和完整输出；分别记录进程墙钟、API、计算和读写边界。
- 官方算法存在随机成分或不同优化器时，记录自身重复性、成分排列与符号对齐方法；旧/新 FNIT 同输入回归作为优化不退化的独立证据。
- 记录输入、程序、资产、适配器和实际执行源码 SHA-256；参考结果不进入生产估计。
- 确认尺寸、仿射、dtype、有限值、输出完整性；连续值报告最大值/P99/误差，分类报告混淆或 Dice，表面先核实顶点对应。
- CPU 专用优化同时复测 GPU 输入的完整受影响 API、输出、显存和配对时间。共享 GPU 负载下的时间明确列为观察值。

## 数据与源码

固定服务器入口为 `FNIT`，任务源码在 `workspaces/fmri_cpu_20261004`，运行在 `runs/fmri_cpu_20261004`。基线单独冻结，正式运行期间不改源码。其它任务的运行目录和 Conda prefix 保持原位置。

私有输入路径、被试标识和个体影像不进入公开报告。说明中的脑图使用已公开、许可允许的样例或既有公开图；本轮私有病例只发布聚合指标。原软件仅供隔离官方验证，新增生产功能复用 FNIT 成熟实现。来源许可不明确的外部参考不复制发布。

## 已发现并修复的验证问题

- 原 ICA-AROMA 的该版本从当前工作目录读取三个分类掩膜；初次包装将掩膜放到子目录，原程序失败。保留失败记录，将掩膜放到其要求的位置后重新运行，未修改原算法。
- volume 初次两条队列的前置导入、校验和准备没有共同队列锁，发生同组 CPU 竞争，相关计时全部作废。另加独立队列锁覆盖整个子进程，计算适配器保留原 CPU 资源锁；避免控制器与 worker 重复获取同一锁。无效尝试保留，正式数据从修复后的新目录运行。
- benchmark 统一框架将多数组 GIFTI 和 CIFTI 交给表面适配器核对面、顶点与轴。普通 metric GIFTI 仍使用数值比较；官方与冻结 FNIT 双版本测量无需伪造 candidate。

- 官方容器阶段的首轮 `strace` 使 Singularity setuid 启动器失败，未进入原版影像计算；这些尝试没有有效耗时。新隔离参考工具在容器内记录实际 payload 退出码，并要求 payload 与启动器均为零，保留 FSL 原有严格 trace 规则。原版 volume 的许可、session/task 选择与实际修复后的独立运行保持一致；原版与 FNIT 的预处理／clean 输出范围和不同核组分列，不预先宣称整链通过。
- 体积基线四组完整 API 已返回成功，单次 180 帧的 first call/cache call 耗时见 [执行回执](task05_volume/baseline_execution_20261005.public.json)。该冻结基线早于最新 robust BOLD reference 默认，仅报告执行状态；最终合并优化链、原版完整链和逐图精度仍待验收。[13 项完整 helper 官方比较](task05_volume/helper_float32_protocol_20261005.public.json)已完成，固定 helper 输入的精度独立于整链验收。
