# 多模态 CPU 官方 benchmark（2026-10-04）

本轮完成多模态功能的真实数据精度、完整进程耗时及 CPU 优化验证，并汇总与本次生产源码匹配的最新完整 GPU 结果。**速度目标尚未全部达到**：部分单线程配准、反场、纯表面 CLI 和小 3D 冷启动仍慢于官方；下表保留实际差距。

## 1. 功能和覆盖范围

| 功能 | 本轮覆盖 | 详细调用、参数和完整报告 |
|---|---|---|
| TorchFLIRT | 四个主例、22 个功能观察，另测连续输入/参考/双端权重和显式候选矩阵批处理 | [FLIRT](../../docs/flirt/README.md) |
| TorchFNIRT | 13 个完整真实 case × 1/8 线程；default、六级 T1、GM、三阶段 TBSS 及掩膜、强度阶数、正则和分辨率变体 | [FNIRT](../../docs/fnirt/README.md)、[26 项完整功能验收](../../docs/fnirt/assets/cpu-functional-v27-node8-20261004.public.json) |
| TorchApplyWarp | 50 组完整配对、两种真实解剖标签、完整 105 帧 DWI→MNI；共享 World 路径另测 30 组功能及完整 490 帧 BOLD | [ApplyWarp](../../docs/applywarp/CPU_BENCHMARK_20261004.md)、[World](../../docs/applywarp/WORLD_CPU_BENCHMARK_20261004.md) |
| TorchInvWarp | 28 项功能记录：18 项新官方配对、10 项经校验的官方精度参照复用；完整系数场另完成最新两预算 warm-up + 三次配对 | [InvWarp](../../docs/invwarp/CPU_BENCHMARK.md)、[最新分配优化](../../docs/invwarp/CPU_ALLOCATION_20261004.md) |
| TorchConvertWarp | 14 个系数场、稠密场、矩阵和 relative/absolute 组合 × 1/8 线程 | [ConvertWarp](../../docs/convertwarp/CPU_BENCHMARK_20261004.md) |
| convert_space | 10 组主配对、20 个功能观察；MNI、fsaverage、fsLR 的连续图、标签、双半球、掩膜与实际 CLI | [空间转换](../../docs/space_conversion/README.md#5-最新官方对照耗时和脑图2026-10-04) |

覆盖的是功能页列出的受支持功能和实际参数变体；未实现的上游选项逐项列在功能页，不将默认预设运行当作所有参数组合的证明。各页包含输入/输出结构、每个参数、带完整变量名的 Python 示例、CLI、原软件命令、分步骤时间、公开脑图和参考文献。

## 2. 输入、环境与计时

输入为完整真实 T1w、DWI、490 帧 BOLD、实际估计的形变和公开 MNI152/HCP 模板、解剖图谱。配准保留完整优化日程，4D 不删帧，反场保留完整输出网格。私有病例仅公开聚合指标；脑图使用可公开的 CC0 样例或有许可的模板。

较早结果在 nodecw10 测量；后续 FNIRT v27 和 InvWarp v28 在用户指定的 nodecw8 测量。两节点现场均核对为 Xeon Gold 6418H、96 个物理核，使用同一个固定 Conda 环境。每项官方与 FNIT 在同一节点、同一物理核亲和性及 1/8 线程预算下交替运行，不混合两节点时间。预算是上限，原版 FNIRT/InvWarp 仍主要使用一个核，报告记录实际 user/system CPU 时间。

FSL 为现场可运行的 6.0.7.4；空间转换使用固定 CBIG 原函数、MATLAB R2018b 和 Workbench 2.0.0。World 参照为指定的 fMRIPrep 25.2.4 容器。原程序只由隔离 benchmark 调用，FNIT 运行时不增加原软件依赖。Numba、NumPy、SciPy 和 PyTorch 已在主页环境中。

完整进程含启动、导入、读取、全部计算和保存。重复配对先完整预热一次、再运行三次；一次完整观察单独标记。API 时间含全部读写但排除入口启动；分步骤插桩时间与正式配对分列，包含子函数的阶段钟不可相加。nodecw10 后期约有 2,500 个可运行任务，迁移前未完成或外层暂停过的时钟不计作完整 benchmark。

## 3. 最新完整 CPU 结果

单位为秒，`1/8` 表示两种线程预算。FNIRT 和最新 DWI/World 为各一次完整观察；FLIRT 主例和 InvWarp 为三次完整进程中位数。不同协议和节点分别保留来源。

| 功能和完整真实输入 | 官方 1/8 | FNIT 1/8 | 精度及速度结论 |
|---|---:|---:|---|
| FLIRT，T1w，12-DOF corratio | 56.223 / 54.257 | 94.483 / 45.644 | world-grid RMS 0.01569 mm；单线程未达目标 |
| FLIRT，T1w，6-DOF normmi | 123.353 / 118.392 | 107.927 / 83.691 | RMS 0.48752/0.48700 mm；不称原版严格数值匹配 |
| ApplyWarp，105 帧 DWI，原网格仿射 | 49.861 / 34.281 | 8.875 / 5.541 | 全部 81,768,960 值；relative L2 1.794×10⁻⁶ |
| ApplyWarp，105 帧校正 DWI，经 TBSS 系数场→MNI 2 mm | 457.843 / 431.278 | 11.243 / 7.571 | 全部 94,776,045 值；relative L2 2.088×10⁻⁶；[报告](../../docs/applywarp/mni_dwi_105_20261004.public.json) |
| ApplyWarp，完整 T1w，trilinear float | 2.297 / 2.549 | 5.637 / 3.732 | 小 3D 完整进程未达速度目标；50 组中 10 组达标 |
| World，490 帧 BOLD→MNI | 301.475 / 56.297 | 241.778 / 46.831 | 双方 BLAS=1、帧并发=1/8；[最新两预算报告](../../docs/applywarp/world_cpu_latest_20261004.public.json) |
| FNIRT，default 四级，v27/nodecw8 | 181.041 / 172.209 | 149.123 / 57.680 | 完整文件与前版 FNIT 相同；[八项最新主配对](../../docs/fnirt/assets/cpu-primary-v27-node8-20261004.public.json) |
| FNIRT，六级 T1，v27/nodecw8 | 224.018 / 221.801 | 293.713 / 169.214 | 单线程未达目标，8 线程更快 |
| FNIRT，完整 TBSS 三阶段，v27/nodecw8 | 836.237 / 831.938 | 795.217 / 400.189 | 完整阶段链、Jacobian 和四图保存 |
| FNIRT，完整 GM 混合 LM/SCG，v27/nodecw8 | 548.107 / 556.218 | 143.569 / 63.277 | 官方完整两进程链；两预算更快 |
| InvWarp，完整 T1 系数反场，v28/nodecw8 | 35.878 / 36.679 | 50.899 / 21.661 | 脑内向量差 p95 0.033606 mm；单线程未达目标；[配对](../../docs/invwarp/assets/cpu-allocation-v2-node8-official-20261004.public.json) |
| ConvertWarp，14 个场/矩阵组合 | 逐项见报告 | 逐项见报告 | 28/28 更快，1 线程 1.263–4.512×、8 线程 1.279–4.347×；最大场差 ≤3.052×10⁻⁵ mm |
| MNI152 T1 2 mm→fsaverage164k | 8.625 / 5.000 | 2.220 / 2.158 | 连续图 relative L2 约 2–3×10⁻⁷ |
| fsaverage164k→默认 MNI | 36.909 / 34.414 | 5.101 / 4.580 | 完整输出逐值相同 |
| 两个 fsLR32k 模板通道→MNI 2 mm | 75.594 / 57.236 | 10.191 / 6.027 | 完整输出逐值相同 |
| fsaverage41k→fsLR32k，实际 CLI | 2.385 / 0.972 | 2.731 / 0.891 | 8 核总预算按双半球 4+4 分配；单线程仍较慢 |

FLIRT 连续输入/参考/双端权重的 8 线程完整观察为官方 130.495/123.073/114.900 秒、FNIT 71.481/69.269/56.782 秒；三个单线程仍慢于官方。最新显式批处理 corratio 为官方 56.220/53.410 秒、FNIT 96.298/46.169 秒；normmi 为官方 119.950/117.136 秒、FNIT 117.378/68.236 秒。全部矩阵、保存文件、统计量及搜索计数保留对应预算的旧 FNIT 结果，见[权重门槛](../../docs/flirt/cpu_weighted_exact_20261004.public.json)、[批统计门槛](../../docs/flirt/cpu_batched_statistics_exact_20261004.public.json)和[最新配对](../../docs/flirt/cpu_followup_20261004.public.json)。

完整 World 的 442,288,210 个值均有限，正确官方参照仅 113 值不同，最大差 4.883×10⁻⁴；脑内 10 值不同、RMSE 1.487×10⁻⁹，全部头和扩展一致。旧原版多线程参照有 69/50 帧失配，已排除；有效参照将双方 BLAS 固定为 1、按帧并发 1/8，保留全部输入校验和诊断。

FNIRT 新采样与 SCG 省略丢弃成本已通过全部 26 项功能的 102 个完整保存文件核对；旧全程 solver QC/停止记录未保存，不宣称这 26 项的旧新全程 QC 相同。[独立来源核验](../../docs/fnirt/assets/cpu-functional-v27-node8-source-input-output-audit-20261004.public.json)重新检查输入、源码和全部文件。另做同节点完整六级 T1 旧/新 API 诊断：333.439→283.067 秒，四图与 QC/停止状态相同；201 次采样 23.111→4.559 秒，零回退、缓存命中一次。这支持采样提速，不能替代上表官方单线程差距，见[T1 诊断](../../docs/fnirt/assets/cpu-t1-sampler-diagnostic32-node8-v27-20261004.public.json)。完整 TBSS 第三阶段的 25 轮成本轨迹、八个张量和 51 次梯度也一致；595.556→423.703 秒是该阶段独立诊断，见[报告](../../docs/fnirt/assets/cpu-scg-cost-skip-stage3-node8-20261004.public.json)。

## 4. GPU 输出、显存与时间

新增 CPU 分支保留原 CUDA 算子、dtype 和 TF32。最新冻结 v28 在同一 H100 PCIe 完成完整 InvWarp 与完整 TBSS：每项两组 AB/BA，每个进程完整预热一次后计时三次，共 32 次完整保存调用。

| v28 完整 GPU API | 旧 FNIT 两组中位数 | 新 FNIT 两组中位数 | 对应调用的 allocation 峰值 |
|---|---:|---:|---:|
| InvWarp 完整系数反场 | 4.298 / 4.094 s | 3.870 / 3.993 s | 均为 1,275,725,824 B |
| FNIRT 完整 TBSS 三阶段 | 14.866 / 13.365 s | 14.854 / 14.102 s | warm 为 3,564,715,520 B；三次测量为 3,565,562,368 B，旧新对应位置相同 |

反场 18,808,200 值，以及 TBSS 系数 2,968,896 值、另外三图各 7,221,032 值的全部数组、头、扩展和 affine 相同；输入及源码前后 SHA-256 相同。TF32 开启，逐次检查 20,000,000,000 B allocation 上限，见[完整 v28 H100 报告](gpu_cpu_final_v28_ready_retry_20261004.public.json)。这是共享 GPU 下的观察，不能保证任意负载下的固定速度；旧 v27 的部分完成和零函数调用的初始化失败没有混入本次时钟。

其余完整 GPU 结果如下，版本号表示实际测量冻结版本。FLIRT、空间转换的相应核心文件，以及 FNIRT 注册器/采样 helper、World 公共采样器与当前 SHA-256 相同。较早 ApplyWarp 和部分报告中列出的共享依赖有后续 CPU 专用修改，相关 CUDA 分支沿用其报告路径；这些记录没有改标成合并后全部源码的重新实测。除 World 的一次完整调用外，下列各例均有两组 AB/BA、共 16 次完整 API 调用；中位数来自每个进程预热后的三次调用。

| 完整 GPU API | 旧 FNIT 两组中位数 | 新 FNIT 两组中位数 | 旧新对应 allocation 峰值及报告 |
|---|---:|---:|---|
| FLIRT，12-DOF corratio | 5.405 / 5.291 s | 5.543 / 5.309 s | 2,033,769,472 B；[v25](gpu_flirt_v25_20261004.public.json) |
| FLIRT，6-DOF normmi | 10.012 / 10.021 s | 10.056 / 10.287 s | 5,441,890,816 B；[v25](gpu_flirt_v25_20261004.public.json) |
| FNIRT，default 四级 | 16.945 / 16.702 s | 16.737 / 16.868 s | warm 1,181,705,216 B、测量 1,181,704,704 B；[v27](../../docs/fnirt/assets/cuda-final-v27-20261004.public.json) |
| ApplyWarp，完整 T1w | 0.594 / 0.573 s | 0.564 / 0.540 s | 1,536,427,520 B；[v23](gpu_final_v23_20261004.public.json) |
| ApplyWarp，105 帧 DWI 原网格 | 3.930 / 3.764 s | 3.912 / 3.908 s | 719,979,008 B；[v23](gpu_final_v23_20261004.public.json) |
| ApplyWarp，105 帧 DWI→MNI | 5.836 / 6.107 s | 5.498 / 5.395 s | 777,303,040 B；[v26](gpu_inv_mni_v26_20261004.public.json) |
| World，490 帧 BOLD→MNI，一次完整 API | 34.103 s | 36.943 s | 2,002,300,416 B；[v20](gpu_world_v20_20261004.public.json) |
| MNI→fsaverage，完整双半球 | 0.250 / 0.232 s | 0.229 / 0.241 s | 6,232,576 B；[v17](gpu_final_20261004.public.json) |
| fsaverage→MNI，完整双半球 | 2.252 / 2.160 s | 2.023 / 2.017 s | 205,128,704 B；[v17](gpu_final_20261004.public.json) |

各报告核对的旧新 FNIT 完整输出及元数据相同、记录的 allocation 峰值相同。空间转换报告提供两组进程峰值及完整最终保存结果，没有逐次声明 16 份保存结果或每次峰值校验；FLIRT、ApplyWarp 的逐次完整保存门槛见各自报告。各功能页另列与官方软件的精度和计时。共享 GPU 下的时间有双向波动，不能据此保证固定加速比。ConvertWarp 的场外仿射修复增加必要计算：非平凡前后矩阵的绝对输出例，修复前/后六次 API 样本中位数为 0.06127/0.08586 秒，各进程最后调用记录的峰值为 186,100,736/225,821,696 B；该修复同时消除约 5 mm 的原场外误差。完整配置和重复记录见[ConvertWarp 最新报告](../../docs/convertwarp/benchmark_20261004.public.json)及[功能页](../../docs/convertwarp/README.md#最新独立-gpu-范围与场外修复成本)。

## 5. 数值边界

- FNIRT 和 InvWarp 使用既有 FNIT 优化器/固定点求解器，对官方的精度近似不能写成原版算法逐位等价。优化前后 FNIT 一致与对官方精度分别记录。
- InvWarp 系数例脑内向量差 median 0.010075 mm、p95 0.033606 mm、最大 7.314282 mm。稠密场→绝对坐标例的脑内 median 0.816 mm、p95 1.783 mm、最大 311.440 mm；不能将系数例推广到全部场。有效域和脑区覆盖率使用不同分母，详见功能报告。
- ApplyWarp 整数输出改为官方截断后转换；nearest 的五种 dtype 已一致。trilinear 整数边界可能有一档差异，uint8 回绕最大差可为 255。
- ConvertWarp 场外采样修为原版全场最佳拟合仿射外推；真实非平凡例旧误差约 5 mm，修后 ≤3.052×10⁻⁵ mm。
- 空间转换自定义 0.5 mm 网格的半体素取整与 Workbench 不同，差异由两套索引规则完整解释；本次默认 2 mm 输出一致。

## 6. 复现和最近更新

在新进程首次导入前设置预算，Python 同时设置 `torch.set_num_threads`。Numba 池上限大于 Torch 预算时部分 helper 保守串行，只更改 Torch 线程数不能复现本次并行路径。

```bash
# 使用调用方的完整真实 case 清单；全部路径和资源说明保留在私有运行目录。
BENCHMARK_MANIFEST=/absolute/path/cases.private.json
FNIT_CANDIDATE_ROOT=/absolute/path/frozen_candidate
FNIT_BASELINE_ROOT=/absolute/path/frozen_baseline
BENCHMARK_OUTPUT=/absolute/path/private_benchmark_output
BENCHMARK_CPUSET=35,39,43,47,51,55,59,63
NUMBA_NUM_THREADS=8 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 \
python tools/benchmark_multimodal_cpu.py run \
  --manifest "$BENCHMARK_MANIFEST" --candidate-root "$FNIT_CANDIDATE_ROOT" \
  --baseline-root "$FNIT_BASELINE_ROOT" --output-dir "$BENCHMARK_OUTPUT" \
  --cpuset "$BENCHMARK_CPUSET" --threads 1,8 --repetitions 3 --api-repetitions 3 \
  --device cpu --lock-file "$BENCHMARK_OUTPUT/../cpu-timing.lock"
```

`--manifest` 指定各功能 adapter、输入、配置和官方程序；三个 root/output 参数指定冻结源码、基线及私有产物目录；`--cpuset` 指定双方共同的可用物理核；`--threads` 是预算列表；两种 repetition 分别控制完整进程和预热 API；锁覆盖整组配对。单次完整观察另用 `--single-observation --repetitions 1 --api-repetitions 0`。原软件命令由对应 adapter 从实际 case 生成，并完整保存在私有报告。

| 冻结版本 | 改动和验证 |
|---|---|
| v28 | CPU FP64 两次独立原位加法和停止标量减少分配；四次完整真实 InvWarp 1/8 门槛、84 项定向回归及 32 次完整 H100 保存核验。[源码](source_v28_20261004.public.json)、[回归](regression_v28_20261004.public.json) |
| v27 | FNIRT CPU 保序采样和 SCG 丢弃成本省略；26 项完整功能、8 项最新官方配对；接入后全 FNIRT 测试 **373 passed，31.31 s，无跳过**，含实际本机 CUDA。[源码](source_v27_20261004.public.json)、[接入后回归](regression_promoted_fnirt27_20261004.public.json) |
| v26 | CPU prepared FP64 对角变换；187 项定向回归，完整反场及 DWI→MNI。[源码](source_v26_20261004.public.json)、[回归](regression_v26_20261004.public.json) |

合并最新 main `f1cbdab1` 后，20 个多模态生产源码文件仍与冻结 v28 相同；共享重采样、NIfTI 头、SynthMorph、FastVBM、CLI 和多模态 CPU/CUDA 回归为 **1129 passed、2 skipped，66.95 s**。两项跳过仅因本地缺少 FSL applywarp，服务器官方对照另见上表，见[集成回归](integration_latest_main_20261004.public.json)。最新 main 的公共 NIfTI 同网格写出保留原存储 forms；四种真实 FNIRT 参考构造、24 个 CPU 与 40 个 GPU 已保存输出头部均与旧新构造相同，见[共享头部核验](shared_header_latest_main_20261004.public.json)。该核验没有重新运行配准或测量时间，原 benchmark 保留实际冻结来源。

更早权重/批统计/World 布局演进和失败诊断保留在相关子功能报告。不同版本测试范围重叠，不将数量相加。未采用的 FLIRT 有限域检查削减实验逐位通过但更慢，已拒绝并保留[结果](../../docs/flirt/cpu_finite_domain_diagnostic_rejected_20261004.public.json)。本机小网格 CUDA 核对只作为正确性回归，不替代真实服务器 benchmark。

InvWarp 的 CPU1 FP64 融合采样实验通过 198 项单位检查及完整真实 30 轮逐位门槛，完整查询采样减少约 25% 耗时，但完整 API 三轮中位数为旧路径 37.692 秒、新路径 39.266 秒，反而慢 4.17%。首次进程 API 为 38.213/38.991 秒，包含各自新的 JIT 缓存初始化；未清空文件系统缓存。该候选已拒绝，未加入生产源码、依赖或永久测试，未启动新官方配对；当前仍采用上表验证的 v28。见[单份实验记录](../../docs/invwarp/assets/cpu-fused-fp64-rejected-20261004.public.json)。

## 7. 参考、许可和公开数据

各功能页末尾给出原软件代码库和论文。官方参照只在隔离工具运行；项目计算继续使用既有 PyTorch/Numba 实现。FSL 派生部分遵守 [FSL 6 许可](../../licenses/FSL-6.0.txt)及[第三方说明](../../THIRD_PARTY_NOTICES.md)。公开脑图的来源、许可和 SHA-256 见各图附带清单；私有影像、被试 ID 和个体派生脑图不进入公开文档。外置权重和模板保持固定 Release 清单、大小和 SHA-256 校验及上游许可规则。
