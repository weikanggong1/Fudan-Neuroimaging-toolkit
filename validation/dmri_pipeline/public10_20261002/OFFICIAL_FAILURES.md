# 原软件 GPU EDDY 失败与完整恢复核验

本文记录固定十人双分支 benchmark 中保留的三个原软件失败。逐例精确时钟、程序和证据 SHA-256 见 [official_cuda_failures.public.json](official_cuda_failures.public.json)。初次诊断核对时间为 **2026-10-03 10:28:52 UTC**。case05/TBSS 于 **11:22:14 UTC** 完成从 raw 开始的全新原软件整链；case10/MMORF 的首次完整恢复 R1 于 **11:29:14 UTC** 再次失败，case10/TBSS 的 R1 于 **12:21:02 UTC** 完成 27 张有限指标图。case10/MMORF 的 R2 从 raw 全新运行，于 **12:28:44 UTC** 再次在 EDDY 失败。最终原参考为 **19/20**，finalizer 如实标记 `incomplete`；本次不再启动恢复。

## 失败位置与时钟

三个初次失败作业都通过原始输入准备、TOPUP 和独立原 SynthStrip，在 **FSL 6.0.7.4 的 `eddy_cuda10.2`** 返回退出码 `1`。后续 DTIFIT、AMICO NODDI 和 TBSS/MMORF 配准没有执行；没有保存最终校正 DWI 和旋转 bvec。

| Case / 分支 | GNU 完整失败时钟（秒） | pipeline API 失败时钟（秒） | EDDY 子进程（秒） | 错误标志 |
|---|---:|---:|---:|---|
| case05 / TBSS | 995.64 | 994.045500 | 680.985603 | `CudaVolume::common_assignment_from_newimage_vol after resize()`；`cudaErrorMemoryAllocation` |
| case10 / TBSS | 330.53 | 329.028125 | 6.207924 | `parallel_for failed`；`cudaErrorMemoryAllocation` |
| case10 / MMORF | 2022.95 | 2021.087209 | 1718.810261 | `CudaVolume::common_assignment_from_newimage_vol after resize()`；`cudaErrorMemoryAllocation` |

这些是异常退出前的时间，保留在失败记录中，**不进入成功配对的耗时比**。EDDY 子进程时钟包含在 pipeline API 中，API 又包含在完整命令中，三列不能相加。

## 显存采样与同输入证据

| Case / 分支 | 本作业采样峰值（MiB） | 其他进程采样峰值（MiB） | 设备已用采样峰值（MiB） |
|---|---:|---:|---:|
| case05 / TBSS | 1,186 | 64,348 | 65,386 |
| case10 / TBSS | 464 | 73,120 | 73,145 |
| case10 / MMORF | 1,186 | 72,878 | 73,669 |

采样间隔为 **5 秒**。每列的峰值可能出现在不同时间，不能相加。共享 GPU 的外部负载可能影响耗时和分配压力；这些采样不能确定报错瞬间的剩余容量，不能据此断言设备已满，也不能将低本作业采样峰值解释为实际分配需求很小。

case05 的原 MMORF 分支完整成功，其 EDDY 使用相同原始输入和参数定义。失败 TBSS 与成功 MMORF 的 **6/6 个 EDDY 上游文件 SHA-256 完全一致**：SynthStrip 掩膜、EDDY index、采集参数、TOPUP field coefficients、motion parameters 和校正 b0 对。因此，固定的输入损坏或后续 TBSS 配准不能解释该次 EDDY 失败。

日志没有记录失败分配的字节数或优化迭代。成功的冻结 EDDY 日志也只有 GPU 分配提示，不能用缺少迭代文字判定失败发生在优化前。目前能够确定的是原 CUDA 程序报告内存分配失败；容量压力、旧 CUDA/native 分配异常及内部资源问题仍需新的完整运行证据区分。

## 完整恢复核验

已于 **2026-10-03 10:25:34 UTC** 启动三个新的完整恢复作业计划：每次从相同 raw AP、PA、梯度和 T1 输入开始，使用独立结果目录，保持冻结源码、原程序、配置、参数、环境和所选 GPU 不变，并共用已有的全作业 GPU 锁。原失败报告、日志和时钟全部保留。

恢复运行用于核验同一实现的重复性，**没有改变算法**。case05/TBSS 的整链退出码为 `0`，27 张指标图均已保存且为有限值；原 EDDY 和 driver 的 SHA-256 与初次失败版本一致。实际时钟和报告哈希已核对：

| 恢复作业 | GNU 完整命令（秒） | pipeline API（秒） | EDDY 子进程（秒） | Observer（秒） | GNU 最大 RSS（KiB） |
|---|---:|---:|---:|---:|---:|
| case05 / TBSS R1 | 3399.21 | 3382.118180 | 1947.249142 | 3399.224054 | 5,854,544 |
| case10 / TBSS R1 | 3079.49 | 3064.439439 | 1275.663944 | 3079.505072 | 5,854,400 |

GNU 原记录为 `56:39.21`；observer 单列，不替代 GNU 时钟。该次完整运行的分阶段 API 记录为：

| 阶段 | 秒 |
|---|---:|
| 原始输入准备与 b0 选择 | 18.962165 |
| TOPUP | 263.237629 |
| 独立原 SynthStrip 掩膜 | 34.965031 |
| EDDY 阶段 | 1947.251280 |
| shell 选择与 DTIFIT | 12.290234 |
| AMICO 完整拟合与保存 | 30.878980 |
| FA 预处理与权重 | 0.984671 |
| FLIRT、FNIRT 与九图传播 | 1073.547089 |

EDDY 子进程和原 TBSS 的嵌套步骤已包含在对应父阶段中，不能再次相加。完整阶段明细、报告/GNU time/process metrics 的 SHA-256 见 JSON 的 `recovery.completed_attempts`。

R1 三作业已终止，为 **2 次完整成功、1 次失败**，所选原参考累计 **19/20**；case10/MMORF 的 R1 失败另行保留，R2 亦已失败，全部恢复作业均已终止。这次成功说明未改变实现的完整重复运行可以通过，不证明初次分配错误的根因，也不构成源码 bug 已修复或 FNIT 数值等价结论。后续成功时间只在输入、资源及完整运行检查通过后与对应 FNIT 结果配对，初次失败及额外恢复成本另列。

原 case01 MMORF 的构造阶段故障与此前 case02 FNIT 的 20 GB 自身 allocator 上限故障属于另外的记录，见 [PROTOCOL.md](PROTOCOL.md) 和 [case02 NODDI 组件检查](case02_noddi_memoryfix.public.json)。本次三个错误发生在独立原软件 EDDY，不混入那些故障的恢复结论。

## case10/MMORF 首次恢复 R1 的再次失败

R1 仍在 `eddy_cuda10.2` 退出 `1`，日志为 `parallel_for failed: cudaErrorMemoryAllocation: out of memory`，随后异常信息包含不可读字符；没有保存最终校正 DWI 或旋转 bvec。该次从 raw 开始的原软件全流程于 **2026-10-03 11:29:14 UTC** 失败，数值为：

| GNU 完整失败时钟（秒） | pipeline API（秒） | EDDY 子进程（秒） | Observer（秒） | GNU 最大 RSS（KiB） |
|---:|---:|---:|---:|---:|
| 395.24 | 392.962648 | 5.763072 | 395.253258 | 5,854,220 |

四个主阶段分别为输入准备 `20.338799` 秒、TOPUP `265.676975` 秒、原 SynthStrip `101.181091` 秒和 EDDY 阶段 `5.765203` 秒。它们是失败前的记录，不进入成功时间比，EDDY 子进程不与父阶段再次相加。

5 秒采样记录的本作业峰值为 `0 MiB`、其他进程峰值 `56,218 MiB`、设备已用峰值 `56,262 MiB`。最后两个采样点的其他进程占用均为 `40,104 MiB`、设备已用均为 `40,140 MiB`。短时 EDDY 调用可能未被采样捕获；本作业的零观测不表示 GPU 使用或需求为零。已有记录不足以将失败归因于设备容量耗尽或外部负载。

### 两个 case10 R1 的同输入检查

核对了正在运行的 TBSS R1 与已失败的 MMORF R1，仅读取已生成的命令、小文件和日志：

- TOPUP、SynthStrip、EDDY 的调用在只归一化各自输出目录前缀后逐项相同。
- `ref_scan_no=0`、AP/PA 原始 b0 索引 `0`、117 帧 index 全为 `1`、readout `0.1261` 秒及 EDDY seed `12345`、8 轮等设置一致。
- **5/5** 个小型输入文件字节一致：脑掩膜、TOPUP field coefficients、motion parameters、采集参数和 EDDY index。**3/3** 个 b0 相关评分日志字节一致。
- EDDY 保存的全参数文本在同样归一化输出前缀后完全相同；未读取完整 raw/corrected DWI 或影像数组。

这些证据没有发现两个分支在 EDDY 前的配置差异。TBSS R1 已于 12:21:02 UTC 完整成功，校正 DWI、旋转 bvec 和 27 张有限指标图均已保存；它是独立原软件整链成功记录。其 EDDY 子进程退出 `0`，耗时 `1275.663944` 秒。

### 后续核验建议

TBSS R1 已用相同输入和 EDDY 设置完成。MMORF R2 在新目录从 raw 完整运行，保持冻结版本、参数、环境、设备和共享锁；初次和 R1 失败都保留，没有复制 TBSS 的 EDDY 结果。R2 已再次失败，本次 benchmark 保留真实不完整的原参考，不再继续重跑或改用 CPU/其他原程序版本。

case01 原 MMORF 曾由 gdb 将故障定位到构造阶段的 64 字节分配；本次 EDDY 只有短时调用和 CUDA 分配错误，没有相同的断点、分配大小或阶段证据。两者不能直接归为同一故障，也不将 MMORF 的构造重试规则套用于 EDDY。后续若单独开展原 EDDY 故障诊断，应获取分配位置、字节数和 native backtrace；本次没有执行该项 native 调试，也没有因此延长 benchmark 的恢复次数。

## case10/TBSS 完整成功与 MMORF R2 身份核验

case10/TBSS R1 的 GNU 原记录为 `51:19.49`，pipeline API 为 `3064.439439` 秒，完整 observer 为 `3079.505072` 秒；三者分别保留。整链退出 `0`，27 张指标图均有限。其阶段 API 为：

| 阶段 | 秒 |
|---|---:|
| 原始输入准备与 b0 选择 | 17.763528 |
| TOPUP | 265.761633 |
| 独立原 SynthStrip 掩膜 | 34.049934 |
| EDDY 阶段 | 1275.666005 |
| shell 选择与 DTIFIT | 11.869623 |
| AMICO 完整拟合与保存 | 30.536803 |
| FA 预处理与权重 | 1.171175 |
| FLIRT、FNIRT 与九图传播 | 1427.619628 |

5 秒采样的本作业、其他进程和设备峰值分别为 `1,186`、`54,300`、`54,980 MiB`，它们可能来自不同采样点。该成功没有定位先前分配错误的根因。

R2 的恢复门槛实际核验通过：R1 三作业终止、TBSS 完整成功、旧 finalizer 退出。**12:26:15 UTC** 的只读检查确认 R1/R2 命令在仅归一化输出目录后相同，显式环境和所选 GPU 身份相同；**21 项**原程序、源码、模板、SynthStrip 权重及配置文件的大小和 SHA-256 全部匹配 R1。已生成的 **7 项**小型命令记录和 **4 个**b0/采集参数/index 文件也一致。R2 此时仍在 TOPUP，尚无完整报告或退出码。这是故障发生前的身份检查，不构成 R2 成功或数值等价结论。

R2 使用独立工具与结果命名空间，其他 19 个所选比较复用原有完整证据。初次和 R1/R2 失败作为历史独立列出，失败时钟不进入成功时间比，也不拼接为完整成功耗时。门槛、plan、manifest 和全部实际文件 SHA-256 见 JSON 的 `recovery.R2_identity_check`。

## case10/MMORF R2 最终失败与停止恢复

R2 于 **2026-10-03 12:28:44 UTC** 完整命令退出 `1`，仍在同一 `eddy_cuda10.2` 报告 `parallel_for failed`、`cudaErrorMemoryAllocation` 和 `out of memory`。校正 DWI 和旋转 bvec 均未保存，后续 DTIFIT、AMICO 与 MMORF 没有执行。

| GNU 完整失败时钟（秒） | pipeline API（秒） | EDDY 子进程（秒） | Observer（秒） | GNU 最大 RSS（KiB） |
|---:|---:|---:|---:|---:|
| 356.68 | 354.392185 | 5.665275 | 356.687027 | 5,854,540 |

GNU 原记录为 `5:56.68`。输入准备、TOPUP、原 SynthStrip、EDDY 阶段分别为 `19.374197`、`263.791423`、`65.558351`、`5.667620` 秒。这些异常退出时钟与对应父阶段单独保存，不进入成功配对比。

失败后的只读核验确认：R1/R2 的 **10 项**调用在仅归一化输出前缀后相同，**7 个**小型生成文件字节相同（b0 AP/PA、采集参数、TOPUP coefficients/motion、EDDY index 和 SynthStrip 掩膜），**3 个**b0 评分日志相同，EDDY 全参数文本相同。两份最终报告的参数、原始输入 SHA、原程序、配置、模板、runner 源码、SynthStrip、b0 选择和环境记录这 **9 组**定义也逐项一致。没有发现本次失败由这些输入或配置差异引起的证据。

5 秒采样的本作业峰值为 `0 MiB`、其他进程峰值 `52,424 MiB`、设备峰值 `52,444 MiB`；最后两次采样的其他进程均为 `40,656 MiB`、设备均为 `40,687 MiB`。短调用可以落在采样间隔中，零本作业观测不能解释为零需求；现有记录仍不足以确定报错瞬间的容量或 native 分配故障原因。

新 finalizer 已退出 `2`，保留全部 **20 个固定行**，并将完整结果标为 `incomplete`：FNIT **20/20** 完成，所选原参考 **19/20** 完成，case10/MMORF 的原参考失败保留。没有第三次恢复，也没有换版本、CPU 或复用其他分支的影像。失败报告/GNU time/process metrics/EDDY log 和终态汇总 SHA-256 均见 JSON。

## 原程序与资料

原程序为验证环境中的 FSL GPU EDDY；生产 FNIT 不调用它。参考 [FSL EDDY 文档](https://fsl.fmrib.ox.ac.uk/fsl/docs/diffusion/eddy/index.html)、[EDDY 原代码](https://git.fmrib.ox.ac.uk/fsl/eddy) 和本目录的 [benchmark_official.py](benchmark_official.py)。完整流程、计时边界和共享 GPU 记录规则见 [PROTOCOL.md](PROTOCOL.md)。
