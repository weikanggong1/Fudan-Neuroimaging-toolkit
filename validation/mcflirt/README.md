# MCFLIRT 真实数据验证

## 当前优化：完整 490 帧

[gpu_optimization.public.json](gpu_optimization.public.json)记录完整 490 帧真实 BOLD 的优化验证，候选源码为冻结的 `7456251842b5b6fac4affc95c83ba61265ad0a0a`，对照为 `1eb9c417` 的已保存完整 FNIT GPU 结果。测量使用共享 H100 PCIe、`cuda:0`、8 线程、TF32 和完整 float32；8/4/4 mm 三阶段均为一次坐标轮回，最终样条采样。报告确认运行前后源码哈希一致。

全部 490 帧写出的矩阵文本和 `.par` 与冻结结果相同，SHA-256 相同。内存矩阵与冻结文本最大差为 4.44×10⁻¹²，参数比较也受到冻结文本保存精度的影响。运动校正 int32 图及固定掩膜高通 float32 图，各自在全部 242,851,840 个体素时间点上与冻结结果逐元素相同，RMSE 均为 0。此结果支持复用 [full490_gpu.public.json](full490_gpu.public.json)中相对原 FSL 的精度统计及现有脑图；逐元素一致性所比较的基线是冻结 FNIT。

| 测量边界 | 秒数 |
|---|---:|
| `TorchMCFLIRT.run`：输入读取/解压、准备、估计、样条采样和 dtype 转换，不写盘 | 159.27 |
| 其中同步记录的 `apply_motion_warp` 样条采样 | 24.83 |
| API 时间扣除样条采样，仍含输入读取、准备、估计及 dtype 转换 | 134.45 |
| API 加私有验证文件写盘，含额外完整精度 NumPy 数组 | 185.00 |
| 固定脑掩膜的后续均值/掩膜准备、缩放、高通及保存 | 17.91 |
| 运动 API、私有文件写盘和上述固定掩膜后续合计 | 202.91 |

NCC 共调用 45,972 次。CUDA allocated 显存峰值 1.150 GB，reserved 峰值 1.372 GB。API 计时使用提前打开的 NIfTI header，四维数据读取/解压仍在计时内；哈希、冻结图读取和数值比较在候选计时外。185.00 秒的文件集合包含额外私有验证产物，与原 MCFLIRT 命令的输出集合不同。202.91 秒由运动 API 和给定掩膜的 FEAT 子函数构成，范围不含脑提取、BIDS 查找、warp 估计及完整 `run_feat_core` 入口。

冻结流程的[历史 profile](../fmri/feat_profile.public.json)为运动 API 709.11 秒、采样 30.74 秒，带有额外包装调用。原 MCFLIRT 的历史独立命令为 326.10 秒，包含估计、最终采样和写盘。共享 GPU 负载、缓存状态及输出集合未匹配，故这些历史观察与本次计时不构成受控的固定加速比。

### 复现当前完整控制

使用冻结流程实际保存的参考图，并核对输入 SHA-256、voxel size、时间单位和 TR。本例原始 SBRef 与 FEAT 保存参考的像素、affine 相同，但 z 向 `pixdim` 分别约为 2.3999999 与 2.3999996 mm；微小头信息差异会改变 float32 采样及 NCC 优化选择。基线矩阵、参数和两份四维图均须覆盖全部帧。

```bash
# 所有影像、逐帧结果和验证产物留在服务器私有目录。
# FEAT_SAVED_REFERENCE 是冻结流程实际保存的参考图；FRESH_OPTIMIZATION_OUTPUT 必须尚不存在。
python validation/mcflirt/benchmark_optimization.py \
  --bold "$RAW_BOLD" --reference "$FEAT_SAVED_REFERENCE" \
  --brain-mask "$FROZEN_EPI_MASK" \
  --baseline-matrices "$FROZEN_FNIT_MATRICES" \
  --baseline-parameters "$FROZEN_FNIT_PARAMETERS" \
  --baseline-corrected "$FROZEN_FNIT_CORRECTED_BOLD" \
  --baseline-filtered "$FROZEN_FNIT_FILTERED_BOLD" \
  --baseline-kind frozen_fnit \
  --baseline-source-revision 1eb9c417febebc8bdd450590d4759454e7160141 \
  --source-revision "$CANDIDATE_FNIT_GIT_REVISION" \
  --output-dir "$FRESH_OPTIMIZATION_OUTPUT" --device cuda:0 --threads 8
```

该 driver 默认比较全部帧，保存 `summary.public.json`、私有图像、完整精度数组、矩阵文本和 `.par`。`--frames` 用于局部调试；截短序列会改变最后一帧的粗阶段初值和完整序列高通结果。公开报告只保留匿名汇总和哈希；不公开影像、矩阵、参数及逐帧数据。

## 单次代价函数微基准

[cost_optimization.public.json](cost_optimization.public.json)由 [benchmark_cost.py](benchmark_cost.py)测量同一真实 BOLD（88×88×64×490）。moving 固定为第 0 帧，依次应用来自第 0、100、489 帧的冻结矩阵；参考在 GPU 上一次构建为 8 mm、4 mm 网格，CPU 使用数值相同的副本。三个版本共用相同 moving 和强度重心，两个 CUDA 版本共用冻结的 `torch.compile` NCC 归约器。输入、实际导入源码及 driver 均记录 SHA-256。

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

- [full490_cpu.public.json](full490_cpu.public.json)：完整 490 帧、4 线程 CPU 估计 387.60 秒、45,794 次 cost；包括路径解压、准备和估计，未含重采样与写盘，未在当前优化后重新测量。相对原 MCFLIRT 的逐帧脑内 pull RMS 均值 0.00906 mm、中位数 0.00864 mm、最大 0.02603 mm；旋转 RMSE 为 0.0000787–0.0001182 rad，平移 RMSE 为 0.003109–0.003576 mm。
- [full490_gpu.public.json](full490_gpu.public.json)：冻结 `1eb9c417` 完整 GPU 流程的已保存结果，脑内时间 r 均值 0.99957825、中位数 0.99983382，SD 图 r 为 0.99999784；脑内 pull RMS 均值 0.00881 mm、最大 0.03008 mm。该运行 FEAT 阶段合计 973.119 秒，还包含脑提取、缩放和高通，未分别记录 MCFLIRT 估计与采样。
- [first8_cpu.public.json](first8_cpu.public.json)、[first8_cuda.public.json](first8_cuda.public.json)、[first8_warm.public.json](first8_warm.public.json)：同例前 8 帧控制，只将前 7 帧与原完整 490 帧结果比较。最后一帧的粗阶段初值规则使截短序列的实验条件不同；早期冷/重复运行耗时不代表当前完整体积性能。
- [first8_sampling.public.json](first8_sampling.public.json)：固定原矩阵文本，单独核对样条及 int32 截断；脑内 RMSE 0.2816、时间 r 均值 0.9999987，中位数 1。原运行的内存矩阵未捕获，文本量化可能影响整数边界。

CPU/GPU 的相对旋转角在求角度前将 3×3 线性部分通过 SVD 投影到 SO(3)，避免文本精度和 float32 非正交误差影响 `acos(trace)`。完整链的其他处理边界与脑图见[全流程记录](../fmri/matched_native.md)。

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
