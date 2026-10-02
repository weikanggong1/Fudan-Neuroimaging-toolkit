# MCFLIRT 真实数据验证

## 最新验证：精确缓存和 CUDA graph 配对控制

本轮候选运行时源码为 `8a3f2276c3e990bb60971ab38b1a93c6d7cb0ba8`，基线为 `de23971119b38b83c6a513380fd699e88ecf3fb8`。同一真实 BOLD 88×88×64×490、FEAT 保存参考、物理 GPU 1、8 线程、TF32/完整 float32，按基线/候选/候选/基线顺序启动四个独立进程。两个版本均为 8/4/4 mm 三阶段、每阶段一次坐标轮回和最终样条采样。完整报告为 [paired_exact_latest.public.json](paired_exact_latest.public.json)。

| 相同边界，秒 | 基线第 1 回合 | 候选第 1 回合 | 候选第 2 回合 | 基线第 2 回合 |
|---|---:|---:|---:|---:|
| 完整 `TorchMCFLIRT.run` | 86.37 | 52.47 | 63.48 | 109.46 |
| 其中最终样条采样 | 27.89 | 16.51 | 20.05 | 26.54 |
| API 扣除最终采样 | 58.47 | 35.96 | 43.43 | 82.92 |

API 的两回合均值为 **97.91 → 57.98 秒，减少 40.8%**；最终采样均值为 27.22 → 18.28 秒。API 包含原始四维读取/解压、准备、首次编译/graph capture、估计、最终采样和 dtype 转换，不含写盘、哈希及数值核对。API 扣除采样后仍有读取/准备/编译/capture，不是纯优化器时间。共享 GPU 在实验期间有其他任务，显存占用增加，快照保留在报告中；这些数值描述本次实测，不能泛化为固定加速比。

本轮独立 H100 双卡 focused 检查 **74 项通过、0 跳过、0 失败**，包含非默认 device/stream 的两项合同检查；源码和测试文件均与 `8a3f2276` Git 快照及运行前后哈希一致，见 [review_tests_exact_latest.public.json](review_tests_exact_latest.public.json)。

四次均为 **45,972 次 NCC**。未舍入 float64 矩阵、六列参数及全部 corrected 解码值的原始字节 SHA-256 相同，包含正负零；输入一致，运行前后 source 哈希均未改变。候选配对调用 CUDA 峰值 allocated 为 1.154 GB、reserved 为 1.286 GB。

[gpu_exact_latest.public.json](gpu_exact_latest.public.json)是另一次完整验证：未舍入矩阵及参数按 uint64 逐元素相同，所有 `MAT_*` 与 `.par` 文本相同，全部 242,851,840 个校正 int32 值相同，RMSE 和最大差均为 0。API 为 59.09 秒，其中最终采样 17.38 秒；加额外私有验证写出为 87.85 秒，固定掩膜后续为 21.12 秒，合计 108.98 秒。其写出包含完整精度 NumPy 数组，输出集合不同于原 MCFLIRT。后续仅测量给定掩膜的均值/掩膜准备、缩放、高通和保存，不含脑提取及完整 `run_feat_core` 入口。

这份完整验证默认读取头信息 TR=0.7350000143051147 秒，而冻结 FEAT 使用 BIDS 元数据的显式 TR=0.735 秒。两份强度缩放因子均为 1.4359563469270533，高通 `sigma_volumes = 100 / (2 × TR)` 不同：whole-image RMSE 为 1.4798×10⁻⁵，最大差 0.00390625，99.9493% 解码值相同，脑内时间 r 均值为 0.9999999999999936。将 TR 同样设为 0.735 秒后，全部 242,851,840 个高通 float32 值相同，RMSE 0，见 [matched_tr_exact_latest.public.json](matched_tr_exact_latest.public.json)。这两项验证分别保留，不能将默认头信息控制写成逐 bit 相同；MCFLIRT 本身的矩阵与采样未出现差异。

[sampling_exact_latest.public.json](sampling_exact_latest.public.json)另核对全部 490 帧的最终样条输出，比较 int32 截断之前的 float32 位模式。全部 242,851,840 个 uint32 值相同，RMSE 和最大差均为 0；不是依靠整数截断掩盖误差。验证用时包括双份采样、读取和逐位比较，不能当作单次采样 benchmark。

### 这轮优化改变了什么

每个固定重心的搜索保存各轴最近一次 float64 角度及旋转矩阵，平移搜索复用完整旋转，单轴角度搜索复用另两轴。角度键为原始字节，正负零不共用；仍调用共享 FLIRT 的 `_axis_rotation` 并按 `I @ Rx @ Ry @ Rz` 组合。pull 系数只缓存固定 voxel size 的两份对角矩阵，每次仍按原次序做 double 求逆、乘积，再缩窄成 float32。

每次 CUDA `run` 为 8、4 mm 参考分别保留固定采样缓冲区与原编译 NCC 的 graph，4 mm 两阶段共用 workspace。graph 保留原行、层累加次序和 `num/numA`；换帧复制完整 float32 moving。最终 motion-only 样条采样的 graph 捕获原 PyTorch 运算，保留 Constant 样条、extraslice 和输出截断；CPU 和三线性最终采样沿用原路径。仅请求某类 RMS 时计算它，80 mm 球体定义及输出格式不变。

这些改动减少重复构建和提交，没有减少 cost、放宽容差、改用 float16 或替换原 NCC。Brent 仍逐次读取标量，按原次序决定下一搜索位置。实际速度以上述完整配对测量为准，组件探测见下文。

## 原软件本轮运行与分步观察

[native_exact_latest.public.json](native_exact_latest.public.json)记录同一 raw BOLD 和 FEAT 保存参考上的原 `mcflirt -mats -plots -spline_final -report -verbose 1`。FSL 安装为 6.0.7.22，MCFLIRT 为 2111.0、NEWIMAGE 为 2601.0、MISCMATHS 为 2412.6；包版本、原源码提交、外层/内嵌 ELF 哈希均实查。OMP/OpenBLAS/MKL 环境设为 8；捕获的单线程 PID 是外层包装器，实际 C++ 子进程的线程峰值没有采集。

| 原报告消息观测区间 | 秒 | 范围 |
|---|---:|---|
| 启动至读取消息 | 0.10 | 外层进程启动。 |
| 读取消息至第一个阶段 | 5.78 | 四维读入、解压和初始矩阵准备。 |
| 第一个阶段至第二个阶段 | 4.50 | 8 mm 参考准备及估计。 |
| 第二个阶段至第三个阶段 | 24.62 | 4 mm 参考准备及第二阶段估计。 |
| 第三阶段至保存消息 | 150.58 | 第三阶段估计、最终样条采样、MAT/par 写出，无法拆分。 |
| 保存消息至进程退出 | 145.96 | 显示范围、dtype 转换、校正 NIfTI 保存及退出。 |
| 原命令总 wall | **331.54** | 包含标准输出文件写盘；哈希及事后完整性核对在计时外。 |

区间按 unbuffered stderr 消息到达管道的时间记录，含观察延迟。原程序未输出第三阶段结束/最终采样开始的独立消息，不能据此构造纯优化或 sampler 耗时。FNIT 的 57.98 秒 API 不含写盘，原 331.54 秒包括标准文件写出，不能直接相除作为相同边界加速比。

本轮原命令返回 **255**，同时校正图的 gzip CRC/size、完整网格和有限值、490 个 MAT 及 490×6 参数全部通过。独立真实首帧的 `strace` 诊断确认 C++ 子进程正常退出 0；外层 ELF 包装器第二次等待同一子进程收到 `ECHILD`，随后退出 -1。报告保留全序列的原退出码及 `valid_run=false`，另记录 `completed_outputs=true`、`command_exit_ok=false`；数值比较使用已经核对完整的输出，没有把异常退出改成成功。

最新 [native_comparison_latest.public.json](native_comparison_latest.public.json)基于本轮原输出：脑内时间 r 均值 0.99955663、中位数 0.99982518；4D RMSE 8.39578；SD 图 r 为 0.99999706、RMSE 为 0.42156。逐帧 pull RMS 均值 0.00902 mm、p95 0.01648 mm、最大 0.04011 mm。六列参数表、完整 I/O 和更新脑图见[功能说明](../../docs/mcflirt/README.md)。本轮优化与此前 FNIT 全部运动输出一致，相对原 FSL 仍有这些既有余差。

### 复现当前完整控制

使用冻结流程实际保存的参考图，并核对输入 SHA-256、voxel size、时间单位和 TR。本例原始 SBRef 与 FEAT 保存参考的像素、affine 相同，但 z 向 `pixdim` 分别约为 2.3999999 与 2.3999996 mm；微小头信息差异会改变 float32 采样及 NCC 优化选择。基线矩阵、参数和两份四维图均须覆盖全部帧。`FROZEN_FNIT_MEMORY_MATRICES`、`FROZEN_FNIT_MEMORY_PARAMETERS` 分别为基线实际保存的未舍入 float64 N×4×4、N×6 `.npy`；它们必须同时提供。`BASELINE_FNIT_GIT_REVISION` 和 `CANDIDATE_FNIT_GIT_REVISION` 记录两份实际源码。复现冻结高通时显式使用 TR=0.735 秒、周期 100 秒，避免用 NIfTI float32 TR 取代 BIDS 元数据值。

```bash
# 所有影像、逐帧结果和验证产物留在服务器私有目录。
# FEAT_SAVED_REFERENCE 是冻结流程实际保存的参考图；FRESH_OPTIMIZATION_OUTPUT 必须尚不存在。
CUDA_VISIBLE_DEVICES=0 python validation/mcflirt/benchmark_optimization.py \
  --bold "$RAW_BOLD" --reference "$FEAT_SAVED_REFERENCE" \
  --brain-mask "$FROZEN_EPI_MASK" \
  --baseline-matrices "$FROZEN_FNIT_MATRICES" \
  --baseline-parameters "$FROZEN_FNIT_PARAMETERS" \
  --baseline-memory-matrices "$FROZEN_FNIT_MEMORY_MATRICES" \
  --baseline-memory-parameters "$FROZEN_FNIT_MEMORY_PARAMETERS" \
  --baseline-corrected "$FROZEN_FNIT_CORRECTED_BOLD" \
  --baseline-filtered "$FROZEN_FNIT_FILTERED_BOLD" \
  --baseline-kind frozen_fnit \
  --baseline-source-revision "$BASELINE_FNIT_GIT_REVISION" \
  --source-revision "$CANDIDATE_FNIT_GIT_REVISION" \
  --output-dir "$FRESH_OPTIMIZATION_OUTPUT" --device cuda:0 --threads 8 \
  --tr-seconds 0.735 --highpass-cutoff-seconds 100
```

配对实验在同一物理 GPU 分别启动基线和候选进程，记录执行顺序、每次源码、资源状态和冷启动/预热条件。该 driver 默认比较全部帧，保存 `summary.public.json`、私有图像、完整精度数组、矩阵文本和 `.par`。`--frames` 用于局部调试；截短序列会改变最后一帧的粗阶段初值和完整序列高通结果。公开报告只保留匿名汇总和哈希；不公开影像、矩阵、参数及逐帧数据。

### 复现同卡完整配对

`BASELINE_FNIT_SOURCE_ROOT` 和 `CANDIDATE_FNIT_SOURCE_ROOT` 分别为两份冻结源码目录，包含 `src/fnit` 或 `fnit`。`RAW_BOLD` 是同一真实四维图，`FEAT_SAVED_REFERENCE` 是原估计实际保存的同网格三维参考；两个 revision 变量为相应完整 Git commit。`PAIRED_ANONYMOUS_REPORT` 是尚不存在的匿名 JSON。该命令只比较一个被试的两份实现，不写影像、矩阵和参数。

```bash
# 物理 GPU 1 对应各 worker 的 cuda:0；四个独立进程顺序为基线/候选/候选/基线。
CUDA_VISIBLE_DEVICES=1 python validation/mcflirt/benchmark_paired.py \
  --bold "$RAW_BOLD" --reference "$FEAT_SAVED_REFERENCE" \
  --baseline-source "$BASELINE_FNIT_SOURCE_ROOT" \
  --candidate-source "$CANDIDATE_FNIT_SOURCE_ROOT" \
  --baseline-revision "$BASELINE_FNIT_GIT_REVISION" \
  --candidate-revision "$CANDIDATE_FNIT_GIT_REVISION" \
  --rounds 2 --threads 8 --device cuda:0 \
  --output "$PAIRED_ANONYMOUS_REPORT"
```

[benchmark_paired.py](benchmark_paired.py)分别核对输入、未舍入数组哈希、cost 次数和运行时 source 前后哈希，记录每次 API/最终采样计时及负载快照。独立进程的首次编译和 graph capture 在 API 时间内，不把某一边额外预热移出计时。

### 复现未截断样条精度控制

`CANDIDATE_PYTHON_SOURCE` 是候选的实际 Python 源目录，例如仓库 `src`；`FROZEN_FNIT_MEMORY_MATRICES` 是完整 float64 N×4×4 `.npy`，推荐使用未舍入数组。`SAMPLING_ANONYMOUS_REPORT` 是尚不存在的 JSON。下面逐帧比较原 eager 与 graph 的 float32 输出，不估计运动，不保存影像或逐帧结果。

```bash
# 全部真实帧的 uint32 位模式核对；计算结果还未经过整数截断。
CUDA_VISIBLE_DEVICES=1 PYTHONPATH="$CANDIDATE_PYTHON_SOURCE" \
  python validation/mcflirt/benchmark_sampling_exact.py \
  --bold "$RAW_BOLD" --reference "$FEAT_SAVED_REFERENCE" \
  --matrices "$FROZEN_FNIT_MEMORY_MATRICES" \
  --source-revision "$CANDIDATE_FNIT_GIT_REVISION" \
  --threads 8 --device cuda:0 --output "$SAMPLING_ANONYMOUS_REPORT"
```

## 组件profile

[components_exact_latest.public.json](components_exact_latest.public.json)用真实 moving 第 0 帧和已保存第 0、100、489 帧的矩阵，在 8、4 mm 两种参考上分别测量。每个 scale 只建一份 graph workspace；原路径和 graph 使用同一个编译 NCC reducer。六组准备张量、NCC、rigid matrix 及 pull coefficients 的位模式全部相同，source 前后哈希一致。

下表为 3 个矩阵×2 轮测量的批次平均微秒/次再取中位数，范围只是固定矩阵组件。

| 组件，µs/次 | 8 mm，原 → 缓存/graph | 4 mm，原 → 缓存/graph |
|---|---:|---:|
| CPU rigid matrix，反复相同参数 | 381.99 → 22.18 | 359.41 → 22.08 |
| CPU pull 系数 | 18.84 → 8.08 | 19.54 → 8.32 |
| queued NCC 归约提交及完成 | 109.66 → 19.91 | 109.65 → 59.45 |
| 同步完整 fixed-matrix NCC | 536.92 → 538.46 | 564.60 → 560.20 |

重复相同参数使角度缓存全部命中，不能把第一行当作 Brent 实际平均。queued 行测量批量提交后的完成区间，不含每次标量同步；实际 Brent 必须逐次读取标量。同步完整 NCC 在这次 profile 中没有明显改善，GPU 利用率快照为 72–99%。首次 graph cost 8/4 mm 为 0.232/0.279 秒，含 capture。组件分别计时，不能相加推导端到端耗时，完整改善由同卡配对实测确认。

`COMPONENT_ANONYMOUS_REPORT` 为尚不存在的 JSON，其余变量与前文一致。调用次数、轮数、预热和抽取矩阵帧均显式指定；影像读取、传输、参考生成及预热在组件计时外。

```bash
# 固定真实矩阵的组件诊断，不能代替完整运动估计 benchmark。
CUDA_VISIBLE_DEVICES=1 python validation/mcflirt/profile_components.py \
  --bold "$RAW_BOLD" --reference "$FEAT_SAVED_REFERENCE" \
  --matrices "$FROZEN_FNIT_MEMORY_MATRICES" \
  --source "$CANDIDATE_FNIT_SOURCE_ROOT" \
  --source-revision "$CANDIDATE_FNIT_GIT_REVISION" \
  --frames 0 100 489 --calls 100 --rounds 2 --warmup-calls 5 \
  --threads 8 --device cuda:0 --output "$COMPONENT_ANONYMOUS_REPORT"
```

## 单次代价函数微基准

[cost_optimization.public.json](cost_optimization.public.json)是前轮融合版的历史微基准，由 [benchmark_cost.py](benchmark_cost.py)测量同一真实 BOLD（88×88×64×490）。moving 固定为第 0 帧，依次应用来自第 0、100、489 帧的冻结矩阵；参考在 GPU 上一次构建为 8 mm、4 mm 网格，CPU 使用数值相同的副本。三个版本共用相同 moving 和强度重心，两个 CUDA 版本共用冻结的 `torch.compile` NCC 归约器。输入、实际导入源码及 driver 均记录 SHA-256。本节不是本轮 graph 优化的耗时结果。

以下是**完整单次 cost**的毫秒/次。每个矩阵分别调用 100、300 次，各测 2 轮；表中取 3 个矩阵×2 种调用次数×2 轮，共 12 个批次的批次总耗时÷调用次数的中位数。

| 参考分辨率与 XYZ 网格 | 冻结 FNIT CPU，8 线程 | 冻结 FNIT CUDA 张量准备 | 融合 CUDA 准备 |
|---|---:|---:|---:|
| 8 mm，26×26×19 | 2.868 ms | 27.512 ms | 4.482 ms |
| 4 mm，52×52×38 | 6.622 ms | 27.928 ms | 4.520 ms |

**仅准备阶段**以无分配的输入捕获器替代 NCC 归约器，并在每次 CUDA 调用后同步；同样汇总 12 个批次：

| 参考分辨率 | 冻结 FNIT CPU，8 线程 | 冻结 FNIT CUDA 张量准备 | 融合 CUDA 准备 |
|---|---:|---:|---:|
| 8 mm | 2.220 ms | 27.313 ms | 4.370 ms |
| 4 mm | 4.388 ms | 27.277 ms | 4.653 ms |

完整 cost 包含变换求逆与系数构造、采样准备、NCC 归约和返回标量所需的同步；准备阶段仍包含系数构造和同步，不能作为纯 GPU kernel 时间。输入读取、参考金字塔构建、重心计算、整帧上传、编译及预热均在这些计时之外。两张表独立测量且受到当时负载影响，不能相减得到 NCC 归约的耗时。

六组真实矩阵/分辨率组合中，融合 CUDA 的参考值、moving 值、权重及最终 NCC 代价与冻结 CUDA 逐 bit 相同。CPU 与冻结 CUDA 的最大代价差为 1.79×10⁻⁷。CPU 栏是冻结 FNIT 的 PyTorch 实现；本节没有运行原 FSL。8 mm 小网格的 CPU 单次代价更短，4 mm 的融合 CUDA 单次代价更短；完整 CPU 运动校正未在这次微基准中重测。

测量使用 H100 PCIe、PyTorch 2.5.1、TF32，物理 GPU 0 对应逻辑 `cuda:0`。GPU 0 的负载快照均为 100%，SM 时钟范围为 787–1725 MHz；同服务器另一个 GPU 也有并行任务。driver 以 25 次为一块，在同一进程交替 CUDA AB/BA 顺序，并交替放置 CPU 块。结果描述当时共享负载下的固定变换求值，不能推导完整三阶段运动估计、最终重采样或原 FSL 的受控加速比。当前进程微基准显存峰值 allocated 为 0.061 GB，reserved 为 0.069 GB。

### 复现代价函数控制

`RAW_BOLD` 和 `FEAT_SAVED_REFERENCE` 与完整控制相同，须核对报告中的 SHA-256 和参考头的实际 `pixdim`。`FROZEN_FNIT_SOURCE_ROOT`、`CANDIDATE_FNIT_SOURCE_ROOT` 是包含 `src/fnit` 或 `fnit` 的源码目录；`FROZEN_FNIT_MATRICES` 是覆盖完整时序的 `MAT_####` 目录或 N×4×4 `.npy`。两个版本变量记录各自 Git revision；`COST_REPORT_JSON` 接收匿名标量报告。

```bash
# 真实影像和冻结矩阵留在私有目录；帧号选择已有矩阵，moving 固定为第 0 帧。
# 此命令包括完整 cost 和准备阶段计时；CUDA_VISIBLE_DEVICES 指定物理 GPU。
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=8 python validation/mcflirt/benchmark_cost.py \
  --bold "$RAW_BOLD" --reference "$FEAT_SAVED_REFERENCE" \
  --frozen-source "$FROZEN_FNIT_SOURCE_ROOT" \
  --candidate-source "$CANDIDATE_FNIT_SOURCE_ROOT" \
  --baseline-matrices "$FROZEN_FNIT_MATRICES" \
  --frames 0 100 489 --repetitions 100 300 --rounds 2 \
  --chunk 25 --warmup-calls 5 --threads 8 --device cuda:0 \
  --frozen-revision "$FROZEN_FNIT_GIT_REVISION" \
  --candidate-revision "$CANDIDATE_FNIT_GIT_REVISION" \
  --output "$COST_REPORT_JSON"
```

每种版本、每个矩阵先预热 5 次，编译包含在预热阶段。`--skip-preparation-timing` 可只保留完整 cost 计时，仍会核对准备张量；短时序应把 `--frames` 改为实际存在的帧号。driver 输出 JSON 到指定文件和标准输出，包含各批次原始耗时、逐 bit 比较、输入/源码哈希及负载快照，不输出影像、私有路径或矩阵数组。

## 冻结版本与早期诊断

以下报告保留各自测量时的源码哈希和计时边界，属于历史控制。

- [gpu_optimization_latest.public.json](gpu_optimization_latest.public.json)：前轮 `cfb7beee` 在物理 GPU 0 的完整 490 帧 API 为 278.45 秒，其中最终采样 32.01 秒；API 加额外私有写出为 305.87 秒，固定掩膜后续为 16.87 秒。该记录不含本轮角度缓存和 graph 优化。
- [gpu_optimization.public.json](gpu_optimization.public.json)：初次融合版本 `7456251` 在物理 GPU 1 上的同算法观察，完整 490 帧运动 API 159.27 秒，其中最终采样 24.83 秒；加额外私有验证写出为 185.00 秒，固定掩膜后续处理为 17.91 秒。它与上述 `cfb7beee` 历史报告的输入哈希和运行时源码哈希均相同，但物理 GPU、共享负载及缓存状态未匹配；两次计时不构成受控加速比。
- [full490_cpu.public.json](full490_cpu.public.json)：完整 490 帧、4 线程 CPU 估计 387.60 秒、45,794 次 cost；包括路径解压、准备和估计，未含重采样与写盘，未在当前优化后重新测量。相对原 MCFLIRT 的逐帧脑内 pull RMS 均值 0.00906 mm、中位数 0.00864 mm、最大 0.02603 mm；旋转 RMSE 为 0.0000787–0.0001182 rad，平移 RMSE 为 0.003109–0.003576 mm。
- [full490_gpu.public.json](full490_gpu.public.json)：冻结 `1eb9c417` 完整 GPU 流程的已保存结果，脑内时间 r 均值 0.99957825、中位数 0.99983382，SD 图 r 为 0.99999784；脑内 pull RMS 均值 0.00881 mm、最大 0.03008 mm。该运行在预先给定脑掩膜下的 FEAT 阶段合计 973.119 秒，包含运动校正、掩膜准备、缩放和高通，未分别记录 MCFLIRT 估计与采样。
- [first8_cpu.public.json](first8_cpu.public.json)、[first8_cuda.public.json](first8_cuda.public.json)、[first8_warm.public.json](first8_warm.public.json)：同例前 8 帧控制，只将前 7 帧与原完整 490 帧结果比较。最后一帧的粗阶段初值规则使截短序列的实验条件不同；早期冷/重复运行耗时不代表当前完整体积性能。
- [first8_sampling.public.json](first8_sampling.public.json)：固定原矩阵文本，单独核对样条及 int32 截断；脑内 RMSE 0.2816、时间 r 均值 0.9999987，中位数 1。原运行的内存矩阵未捕获，文本量化可能影响整数边界。

CPU/GPU 的相对旋转角在求角度前将 3×3 线性部分通过 SVD 投影到 SO(3)，避免文本精度和 float32 非正交误差影响 `acos(trace)`。完整链的最新处理边界与脑图见[全流程记录](../fmri/mcflirt_optimization.md)，原冻结流程见[历史记录](../fmri/matched_native.md)。

复测时先在服务器完成原 FSL MCFLIRT 运行，再执行 FNIT 的独立控制：

```bash
# 原数据仍在服务器；8帧control只输出匿名报告和私有矩阵，不生成图像。
python validation/mcflirt/compare_motion.py \
  --bold "$RAW_BOLD" --reference "$SBREF" --brain-mask "$EPI_MASK" \
  --original-matrices "$ORIGINAL_MCFLIRT_MATRICES" \
  --output-dir "$FRESH_CONTROL_OUTPUT" --device cuda:1 --frames 8
```

完整运动估计使用同一四维输入和完整原矩阵目录，比较包括最后一帧；逐帧数组只留在私有目录：

```bash
# 四线程 CPU，计时包含输入解压、准备和估计；另行核对原程序六列 .par。
OMP_NUM_THREADS=4 python validation/mcflirt/compare_motion.py \
  --bold "$RAW_BOLD" --reference "$SBREF" --brain-mask "$EPI_MASK" \
  --original-matrices "$ORIGINAL_MCFLIRT_MATRICES" \
  --original-parameters "$ORIGINAL_MCFLIRT_PARAMETERS" \
  --output-dir "$FRESH_FULL_CONTROL_OUTPUT" --device cpu --threads 4 --all-frames
```

CUDA 第一次执行会编译固定尺寸 NCC 归约；复测记录应注明冷/重复执行状态，并分别报告估计、采样及写盘的计时范围。

`render_comparison.py` 在同一 EPI 网格计算完整时序的均值、population SD 和逐体素时间 r，再使用同一原 BBR 矩阵与 FNIRT pull 将三维指标图变换到 MNI 展示。它不会将两份四维 BOLD 再次重采样。mean/SD 使用三线性展示，r/有效体素图使用最近邻；统计在 EPI 中计算，MNI 图只用于展示。示例参数如下，矩阵与参数目录必须包含全部帧：

```bash
# 输入均来自已经完成的原 MCFLIRT 与 FNIT TorchMCFLIRT，原始时序不公开。
# MATRIX_T1_TO_EPI_WORLD 是原 inverse BBR，PULL_MNI_TO_T1_RAS 是 MNI 网格的 RAS-mm pull displacement。
# PRIVATE_MAP_OUTPUT 保存指标图和逐帧数值；FIGURE_OUTPUT 仅为去标识化的 MNI PNG。
OMP_NUM_THREADS=8 python validation/mcflirt/render_comparison.py \
  --original "$ORIGINAL_MCFLIRT_BOLD" --candidate "$FNIT_MCFLIRT_BOLD" \
  --epi-mask "$ORIGINAL_EPI_MASK" --template "$MNI152_2MM_TEMPLATE" \
  --mni-mask "$ORIGINAL_MNI_MASK" \
  --reference-to-source-world "$MATRIX_T1_TO_EPI_WORLD" \
  --mni-to-t1-pull "$PULL_MNI_TO_T1_RAS" \
  --original-matrices "$ORIGINAL_MCFLIRT_MATRICES" \
  --candidate-matrices "$FNIT_MCFLIRT_MATRICES" \
  --original-parameters "$ORIGINAL_MCFLIRT_PARAMETERS" \
  --candidate-parameters "$FNIT_MCFLIRT_PARAMETERS" \
  --private-output "$PRIVATE_MAP_OUTPUT" --figure-out "$FIGURE_OUTPUT" \
  --report-out "$ANONYMOUS_REPORT_OUTPUT" --source-revision "$FNIT_GIT_REVISION" --threads 8
```

`native_cost_probe.cc` 是仅用于验证的独立小程序：链接安装好的 NEWIMAGE，读取真实帧，按 MCFLIRT 的 8/4 mm 参考重采样计算四个固定平移矩阵的 NCC。它没有复制原软件代码，也不被 FNIT 运行时调用。构建需要原 FSL 开发头文件与库，使用原软件做 benchmark 时才需要。对应输入和重采样输出均应留在私有验证目录。
