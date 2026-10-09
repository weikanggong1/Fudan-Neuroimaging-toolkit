# 2026-10-09：真实 100k 追踪的无损优化

[完整 pipeline](../../../docs/connectome/README.md) · [函数、参数与策略图](../../../docs/connectome/TRACKING_PERFORMANCE_OPTIMIZATION.md) · [精度与全流程历史](../../../docs/connectome/ACCURACY_OPTIMIZATION_20261003.md)

## 1. 本次采用什么

正式源码提交 `d4327049c012b7c6a510de15a48e27c1c4715281` 只改变 `_five_tissue_mrtrix` 的计算组织：一次取出八角点、批量计算权重，再按原顺序做八次加法。保持 `wx * (wy * wz)`、严格 `weight < 1e-6`、最近体素非零判断、边界及 FP32 舍入顺序。ACT、圆弧算法、随机数消费、参数与输出接口保持原实现。

真实 100k 对照中，默认模式及原有编译模式分别与冻结旧版全部输出逐字节一致；所有 74 项 GPU 回归通过。优化不新增依赖，运行时不调用 MRtrix。`compile_arc` 仍默认 False，两种模式之间本来就可能产生不同轨迹。

## 2. 输入、版本与计时范围

- 公共真实数据：OpenNeuro **ds004666 / EDDEN**。上游 [dataset_description.json](https://raw.githubusercontent.com/OpenNeuroDatasets/ds004666/master/dataset_description.json) 声明 CC0。输入是此前真实流程生成的 FOD、5TT 和 GMWMI，没有生成模拟影像。
- FOD：float32 `[104,104,72,45]`，`lmax=8`；5TT：float32 `[256,256,256,5]`，五组织通道；GMWMI：float32 `[256,256,256]`，与 5TT 同网格。NIfTI affine 用 float64；输入 SHA 与大小逐份写入 JSON。
- FNIT：PyTorch 2.5.1、CUDA 11.8、Python 3.11.16；每个进程使用**单张 H100 PCIe**、8 个 CPU 线程，FP32/TF32，无 autocast。
- 基线：`9c118b748700e04224cd5b235b23a712145eae7e`，tracking SHA `693f015e8f0e7153e8886dc49950caa4cb5f924022888ffe061f1392978e31a4`。
- 正式实现：tracking SHA `6ed4e6bedc45fa0eaf2b026541056ec2976eb7d205d397c8dff4f3e899047dbd`；两版复用同一个冻结 FOD 模块。
- 固定 `100000 seeds / seed=0 / batch_size=8192 / arc_proposals=16 / cutoff=0.1 / power=0.5 / maxlength=250 mm`；步长和最短长度按原网格默认规则。
- 每种模式在同一张卡上，每版先完整执行一次 100k 预热，再按 **AB、BA** 测两轮。默认组与编译组用不同单张卡；只比较同组的旧、新实现。

`tracking_seconds` 从输入已在 GPU 后开始，覆盖完整追踪及内部 metadata 同步。读盘、H2D、最终 D2H、打包、hash、严格比较和 TCK 写盘单列。MRtrix 的参考秒数包含完整命令的读写，因此不能直接当作相同边界的速度比。编译预热共用进程缓存，不代表两个隔离冷启动。

完整参数、Python、benchmark CLI 与官方 `tckgen` 命令见[性能说明第 2–4 节](../../../docs/connectome/TRACKING_PERFORMANCE_OPTIMIZATION.md#2-python-输入与输出)。冻结计时工具 SHA 为 `bc689db275c72321ea1dc616259f8b7b3d9fba8ef91fcbc19cbfcec721264ff2`，位于 `a1f41246`。现工具只修正快照列名 `memory_used_mib`，并记录独立 SHA；未重写已完成实验的源码标签。

## 3. 所有正式 100k 调用

| 模式 | 阶段 / 顺序 | 旧版 A / 秒 | 新版 B / 秒 |
| --- | --- | --- | --- |
| 默认 eager | 完整预热 A→B | 223.474 | 207.125 |
| 默认 eager | 热轮 1 A→B | 222.929 | 178.422 |
| 默认 eager | 热轮 2 B→A | 254.371 | 176.643 |
| 默认 eager | 热中位数 | **238.650** | **177.532** |
| 已有 compile_arc | 完整预热 A→B | 151.125 | 175.853 |
| 已有 compile_arc | 热轮 1 A→B | 125.195 | 108.288 |
| 已有 compile_arc | 热轮 2 B→A | 107.494 | 106.400 |
| 已有 compile_arc | 热中位数 | **116.345** | **107.344** |

本次默认模式观测耗时减少 **25.61%**，编译模式减少 **7.74%**。默认基线前后增加约 14%，编译基线下降约 14%；共享节点负载及首轮影响使两样本中位数不能证明稳定加速倍率。所有慢样本及失败日志保留，没有按快慢删选。完整精度、逐项时间、GPU 快照和参数见：

- [默认模式报告](ordered_default_100000.public.json)
- [已有编译模式报告](ordered_compiled_100000.public.json)

### 输出检查

各组的六次完整调用具有同一个输出 SHA 和 TCK SHA；四次可评估比较全部通过。第一份基线是参照，JSON 中 `assessed=false` 表示建立参照，无失败含义。

| 模式 | 接受流线 | 保存点数 | 输出 SHA-256 | TCK SHA-256 |
| --- | ---: | ---: | --- | --- |
| 默认 eager | 27411 | 1122881 | `0bb811ee93cca10cd8d7c3995cea860fc9e8f21c69685a5e081f2ed415beca43` | `314d3dfa159034a09199fbf4b17f457989cf2b877fb945514b122c166e1f1e08` |
| 已有 compile_arc | 27537 | 1113078 | `064c19574441001d50e38d2ce97eabe9682abaec6801646ac7615fd8d308a1d6` | `5b782a6c7b3de5a03b227c44663b25f5c660b3f06ce533fa49177d3d750946c7` |

直接比较每条路径及 point_counts、packed_points、accepted_seeds、lengths_mm、endpoints、seeds_attempted 的 dtype、shape、数值和原始字节，而不只比较 hash。FA 未提供，mean_fa 的一致表示两边均为 None，**本轮未重新测真实 FA**。SIFT2 与最终矩阵没有在本次性能更改中重跑。

### 显存与回归

下表覆盖各版三次完整 100k 调用，包含预热；tracking 与 tracking＋D2H 峰值相同，单位为十进制 GB。

| 模式 | 旧版 allocated / reserved | 新版 allocated / reserved |
| --- | --- | --- |
| 默认 eager | 1.038 / 1.202 | 1.048 / 1.202 |
| 已有 compile_arc | 1.076 / 1.216 | 1.076 / 1.216 |

Torch allocator 限为 18 GB，给 CUDA 上下文留余量。Torch 指标不含上下文、驱动和其他进程；正式 ABBA 仅有离散整卡快照，不能据此称连续进程显存峰值。本轮另用自己的进程记录 GPU 回归显存：allocated 0.037 GB、reserved 0.038 GB，NVML 采样峰值 1.938 GB；这是回归测试规模，不冒充 100k 的进程峰值。

- [GPU 回归](gpu_regression.public.json)：74 passed，pytest 9.13 秒；覆盖真实圆弧 fixture、非连续/空采样、斜切 affine、`1e-6` 上下边界、dtype/字节、输入与 RNG 不变性。
- [总验证记录](verification.public.json)：CPU 41 passed / 33 skipped；现工具元数据修正后 benchmark CPU 测试 7 passed。没有运行 FNIT 全项目测试。
- 首次 GPU runner 缺少可选 pynvml；改用标准库调用 nvidia-smi，不改环境。第二次传输漏了已有真实圆弧 fixture，72 项通过、2 项因缺文件失败；补齐并校验 SHA 后 74 项全部通过。失败记录保留。

## 4. 官方对照与脑图

同机 MRtrix3 `3.0.3-103-g026e850d`，Xeon Gold 6430、8 线程，固定相同三份输入：

| MRtrix seed | 完整 tckgen / 秒 | 接受流线 | 保存折线平均长度 / mm |
| --- | ---: | ---: | ---: |
| 0 | 16.188 | 27685 | 40.248 |
| 1 | 16.032 | 27673 | 40.291 |
| 2 | 16.286 | 27693 | 40.156 |

见[官方报告](mrtrix_same_host.public.json)。FNIT 编译模式约 107.34 秒，MRtrix 中位约 16.19 秒；当前仍有明显追踪速度差距，且前者排除外部输入/TCK I/O。历史 67–84 秒与 782–807 秒的硬件、负载和计时口径不同，没有据此宣称本次改动实现数倍加速。

![真实流线正交投影](real100k_qc.png)

图来自真实 `9c compiled baseline` 与官方 seed 0，各固定抽样最多 2000 条。本轮正式编译输出 TCK SHA 与图中 FNIT 文件相同，所以沿用原标签及脑图。保存折线 FNIT/MRtrix 平均长度为 38.967/40.248 mm，中位为 23.915/25.363 mm，KS=0.019626。保存点下采样后的折线长度不是内部积分长度。

[QC 统计](real100k_qc.public.json)、[许可与文件绑定](real100k_qc.case_manifest.json)、[复现命令](QC_REPRODUCE.md)。原 MRI/TCK 不上传仓库；图来自已核查许可的公开数据。

同软件优化前后逐字节一致已经通过；跨软件随机追踪不要求相同 seed 产生同一流线。本轮只给官方三 seed 描述和脑图，没有评估 FNIT 对 MRtrix 的完整随机重复范围、FA/端点/TDI 或 SC 矩阵，原全流程“整体未匹配”结论仍保留。

## 5. 诊断与未采用实验

128 个真实 seed 的独立 profiler 来自**未采用的活动行校准候选**：440926 个实际 CUDA kernel，去重 kernel 时间 0.9047 秒，含 profiler 开销墙钟 8.4598 秒。逐元素 kernel 343851 个、索引 kernel 58921 个；频繁的小 kernel 与主机调度仍是瓶颈。记录只累计原始 CUDA kernel，不叠加 CPU operator 的 inclusive 时间；不将这个带 profiler 的小规模秒数外推至 100k。[诊断 JSON](profile_eager_128.public.json)

| 实验 | 真实结果与采用决定 |
| --- | --- |
| 有序角点＋权重 | 算子八种规模/布局逐字节一致；观测提速 1.268–1.379 倍；10k 热中位数 26.648→24.567 秒。完整 100k 与回归通过，已采用。 |
| 仅活动行校准 | 10k、100k 输出一致；100k 编译模式热调用 113.556→116.156 秒，无收益，不采用。 |
| 仅合并 gather | 10k 输出一致，未形成稳定整体收益，不采用；不保留在运行路径。 |
| ACT boolean 编译 | 10k 输出一致，热中位数 27.124→26.594 秒；小规模预热增加约 4.73 秒，不加入默认实现。 |
| ACT 编译＋有序 5TT | 两模式完整 100k 输出一致；编译热中位数 109.955→104.440 秒，默认 255.850→223.569 秒，但共享负载和首轮影响明显。未采用，避免为这次默认优化引入新编译成本。 |

实验报告：[有序算子](ordered_operator.public.json)、[有序 10k](ordered_10000.public.json)、[活动校准 10k](active_calibration_10000.public.json)、[活动校准 100k](active_calibration_100000.public.json)、[ACT 10k](act_core_10000.public.json)、[组合编译 100k](combined_compiled_100000.public.json)、[组合默认 100k](combined_default_100000.public.json)。活动校准 patch 仅为被否决实验的来源记录，不属于生产代码。

## 6. 下一步

优先减少 FOD/5TT 采样、索引、ACT 状态与短 kernel 间的调度，同样保留随机顺序和浮点分组。后续任何融合、图捕获或 buffer 复用都需重新通过真实完整路径逐字节比较，再做同卡交错计时。百万 seed 的内存与吞吐、完整 SC 随机重复精度需独立验证，不能从当前 100k 外推。

## 7. 参考

- [FNIT tracking 源码](../../../src/fnit/connectome/tracking.py)、[benchmark 工具](../../../tools/benchmark_connectome_tracking_exact.py)。
- [MRtrix3](https://github.com/MRtrix3/mrtrix3)、[tckgen 3.0.3 参数](https://mrtrix.readthedocs.io/en/3.0.3/reference/commands/tckgen.html)。
- Tournier, Calamante & Connelly. Improved probabilistic streamlines tractography by 2nd order integration over fibre orientation distributions. ISMRM 2010, 1670.
- Smith et al. Anatomically-constrained tractography. NeuroImage 62, 1924–1938 (2012). [DOI](https://doi.org/10.1016/j.neuroimage.2012.06.005)。
- Manzano-Patron et al. EDDEN. Imaging Neuroscience (2024). [DOI](https://doi.org/10.1162/imag_a_00060)。
