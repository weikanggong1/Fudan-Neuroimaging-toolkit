# 2026-10-01 recon-all 性能修复与整例实测

本页汇总真实同输入测试，以及冻结 `1b8c36d` 从原始 T1 连续完成的两例重建：各完成 66 阶段，138 项输出齐全，现有网格检查通过。两例受控旧版基线也已完成；GPU 同边界完整命令本次缩短 3.889%，CPU 共同校验后范围缩短 1.495%，CPU 完整命令边界不同，不计算其比值。整体指标等效为 `not_assessed`，没有采用尚未正式确认的整例等效阈值。阶段数值和 SHA-256 见 [stage_summary.json](stage_summary.json)，候选完整范围见 [candidate_whole_profile.json](candidate_whole_profile.json)，GPU 实际配对见 [gpu_control_pair_summary.json](gpu_control_pair_summary.json)。

## 版本和运行范围

| 计算版本 | 绑定与实测范围 |
|---|---|
| 阶段 1 | `3d9856c…+stage1tar0444db72`：未提交修改的冻结快照，非干净 Git commit；tar SHA-256 `0444db72c248fac01bf9385c615da4f0e9fbade94ebbe2ccd2a70942f0c2180f`。两例统计、厚度和 SynthSeg 精度诊断。 |
| 冻结 `1b8c36d` | `1b8c36d25a68e253a1e59b6d02114890afa467de`；[部署源码清单](source_1b8c36d_manifest.json)。最终默认厚度的 sub-01 LH、后验缓冲区、线程测试及当前整例。 |
| 后续剖析修复 | 基于 1b8，profiling.py SHA-256 `7eed61ea…`；headcw 8 项 CPU/模拟 CUDA 检查通过，实际命令 1.645 s，非真实 GPU 重建。完整报告 SHA 和远端路径在汇总 JSON。 |
| `12a4834` 后续入口/worker 修复 | 当前说明对应的提交；公开入口失败状态 8 项本地测试通过。厚度 worker 源码 SHA-256 `ab019eb7…` 已独立完成 sub-01 LH 真实 GPU 同输入复核，保存的 1b 厚度图零差异；不改阶段 1 量值归属。当前整例仍使用冻结 1b。 |

GPU 阶段在 gpucw1 的 H100 PCIe（UUID `GPU-e25cac06-0ce8-a833-abf9-09ab18c9c9ba`）运行，CPU 为 Xeon Gold 6430；CPU 线程测试在 nodecw10 的 Xeon Gold 6418H 运行。Torch 2.5.1、CUDA 11.8、cuDNN 90100，Torch 线程预算 4；使用 float32，未启用 FP16/BF16。TF32 默认保留，SynthSeg 卷积的已验证 FP32 例外单独记录。输入、权重、资产、Conda 构建程序及独立参考程序的大小与 SHA-256 均在 [运行清单](runtime_fingerprints_1b8c36d.json)；[GPU](hardware_gpucw1_1b8c36d.json)和 [CPU](hardware_nodecw10_1b8c36d.json)快照记录当时资源。

`git diff e036f57..3d9856c -- src` 为空：3d9856c 只增加此前验证记录，算法源码与 e036 相同。本轮受控旧版可作为算法基线；GPU wrapper 对齐 cuDNN FP32 后才能比较性能，不能将旧实际 TF32→FP32 的标签变化归于优化。

## 六项问题的处理结果

| 项目 | 当前处理与证据 |
|---|---|
| 1. SynthSeg 精度覆盖 | 设备选择与精度策略分开；两次真实前向都记录 cuDNN TF32=False、matmul TF32=True、float32、autocast 关闭。[实际前向](final_1b8c36d/buffer_uncached_api/actual-forward.json)。 |
| 2. 剖析和墙钟范围 | 剖析模式同步显式目标 GPU，并计入阶段写出；公开入口墙钟包含前置校验和报告。父/子 CPU 时间与同步等待分别记录；后续失败/allocator 状态修复单独绑定版本。[43 项测试及 19 子测试](unit_tests_1b8c36d.json)、[入口失败测试](public_wrapper_failure_tests_local.json)。 |
| 3. 多图谱统计重复 | 按表面路径和版本缓存几何、邻接、法线、面积、主曲率和 no-th3 基础量；每个图谱批量汇总后一次回传。[两例完整报告](diagnostics/stats_summary.json)。 |
| 4. 已有 PyTorch 指标 | 当前 CUDA 分支本来已默认使用自有厚度、面积、曲率；CPU 仍默认使用 Conda 源码构建指标。优化厚度的完整空间候选与编译可达性，不重复实现指标。[厚度报告](thickness_summary.json)。 |
| 5. 分配缓存与缓冲区 | 保留低显存 no-cache；后验缓冲区复用通过同 FP32 输出回归。缓存启用仅有单阶段观察，尚不能设为默认或宣布缓冲区提速。[保存文件审计](final_1b8c36d/buffer_stored_dtype_audit.json)。 |
| 6. 线程、设备与双侧 | Torch/Numba 作用域恢复线程预算；原生子进程仍须逐项核对。保留 surface.defects.mgz 的 LH 初始化/RH 合并顺序，未机械并行或迁移 CPU 阶段。[线程回归](cpu_final_1b8c36d/thread_ca_sub01/report.json)。 |

## 多图谱统计：两例、双侧、每轮 12 份输出

输入是 FNIT 已生成的固定 white/pial、顶点图、注释、cortex 和统计前置文件，非原始 T1 整例。white.preaparc、最终 white 与 pial 分开缓存；相同顶点数不代表可共用坐标。缓存键包含解析后的路径及文件版本，覆盖同一路径更新的失效检查。

| 真实数据 | 轮次 | 旧版 s | 缓存版 s | 阶段速度比 | 完整文本相同 |
|---|---:|---:|---:|---:|---:|
| [sub-01](diagnostics/stats/report.json) | 1 | 95.864 | 16.934 | 5.66× | 12/12 |
| [sub-01](diagnostics/stats/report.json) | 2 | 73.893 | 12.444 | 5.94× | 12/12 |
| [sub-02](diagnostics/sub02_stats/report.json) | 1 | 127.219 | 45.661 | 2.79× | 12/12 |
| [sub-02](diagnostics/sub02_stats/report.json) | 2 | 162.184 | 44.968 | 3.61× | 12/12 |

四轮共 48/48 份 `.stats` 文件逐字节相同，九个数值列最大差均为 0；门槛是完整文本一致。两例整个配对命令耗时分别 203.600/384.544 s，包含启动、输入哈希与报告；采样的父子进程同时占用峰值分别 2,027,945,984/2,036,334,592 字节。统计函数时间包括读写，并在边界同步 GPU。

**体积定义保持原有语义。** `-no-th3` 脑区体积按每个三角面平均厚度乘 white/pial 面积和，再向三个顶点累加；单角贡献为 `mean(thickness[face]) × (white_area + pial_area) / 6`，以 float64 汇总。TH3 顶点四面体体积图是另一输出，不能替代它。空间为 surface RAS（mm），厚度 mm、面积 mm²、体积 mm³。接口与具名示例见 [统计缓存说明](../../../../docs/recon_all/SURFACE_STATS_CACHE.md)。

## 厚度：完整候选索引与可达性检查

相同 white/pial 输入、相同有序网格。优化回归预设绝对容差 1e-6 mm、相对容差 0；四个半球每个两轮均有 0 个不同数值，最大/P99 差为 0。空间索引保留完整半径候选与 20-hop 扩展规则。阶段 1 测量对应 `fb3b6651…` 源码，不能当作最终 `c5d586b5…` 或后续 `ab019eb7…` 的时间。

| 输入 | 顶点/面 | dense 两轮 s | indexed 两轮 s | 平均耗时降幅 | 同输入 Conda 最大差 mm |
|---|---:|---:|---:|---:|---:|
| [sub-01 lh](diagnostics/thickness_lh/report.json) | 105539/211074 | 38.420/34.447 | 8.858/9.574 | 74.70% | 4.77e-07 |
| [sub-01 rh](diagnostics/thickness_rh/report.json) | 104864/209724 | 33.052/27.522 | 6.689/6.456 | 78.30% | 9.54e-07 |
| [sub-02 lh](diagnostics/sub02_thickness_lh/report.json) | 119363/238722 | 48.566/42.241 | 13.398/13.448 | 70.44% | 4.77e-07 |
| [sub-02 rh](diagnostics/sub02_thickness_rh/report.json) | 118303/236602 | 39.775/41.151 | 9.087/9.254 | 77.33% | 9.54e-07 |

冻结 1b 最终默认版另测 sub-01 LH：dense 45.589 s，indexed 13.774 s，仍逐值相同；与同输入 Conda map 最大差 4.77e-07 mm。[最终版报告](final_1b8c36d/thickness/report.json)。既有 Conda 比较门槛是 `0.005 mm + 0.001 × abs(reference)`，不是本轮新建的整例等效标准；该参考也不是新运行的系统安装官方程序。

后续 `ab019eb7…` worker 修正另做一次相同冻结 sub-01 LH GPU 回归：[报告](thickness_workers_ab019/result/report.json)、[独立摘要](workers_summary.json)。同步函数及读写 13.706 s，完整命令 17.961 s，`kdtree_workers=4`；与保存的 1b 图差异数 0、最大/P99 差 0、文件 SHA 相同。与既有同输入 Conda 图最大差 4.77e-07 mm、P99 2.38e-07 mm、越门槛顶点 0，保留原浮点尾差。该进程父子同时显存采样峰值 562,036,736 字节，9 个样本、最大间隔 2.256 s、查询失败 0，allocated/reserved 为 null，连续峰值未验证。全 GPU 同期另有 33,843–46,463 MiB 占用且利用率 100%，不归给本进程；此单次共享设备观察不用于宣布 worker 提速或新源码整例通过，阶段 1 的八轮量值和哈希保持不变。

阶段 1 配对进程采样峰值 1,304,428,544–1,365,245,952 字节，包含 dense 和 indexed；不是两个实现各自独立峰值。函数墙钟包括网格读写、传输及同步，排除脚本导入和 CUDA 上下文初始化；整例实际收益见后文，不从本段速度比外推。

## SynthSeg：精度修复、缓冲区和缓存分别评价

本节只有 sub-01 冻结 conformed T1。旧实际 TF32 和修正 FP32 之间有 148 个标签体素变化，最小标签 Dice 0.9996683，TIV 增加 35.125 mm³；这是精度策略变化，不能混入同精度缓冲区回归。

| 版本/策略 | API 状态 | 函数及读写 s | 整个命令 s | 采样父子同时峰值（字节） |
|---|---|---:|---:|---:|
| [阶段1，旧实际 TF32/no-cache](diagnostics/old_effective_tf32/actual-forward.json) | CLI | 32.033 | 46.721 | 19,411,238,912 |
| [阶段1，FP32/no-cache](diagnostics/corrected_uncached_api/actual-forward.json) | 已初始化 CUDA API | 70.258 | 85.570 | 12,897,484,800 |
| [阶段1，FP32/cache](diagnostics/corrected_cached_cli/actual-forward.json) | CLI | 59.367 | 73.869 | 20,352,860,160 |
| [阶段1，FP32/cache](diagnostics/corrected_cached_api/actual-forward.json) | 已初始化 CUDA API | 58.168 | 73.905 | 20,352,860,160 |
| [1b，FP32/缓冲区/no-cache](final_1b8c36d/buffer_uncached_api/actual-forward.json) | 已初始化 CUDA API | 80.795 | 96.692 | 14,508,097,536 |
| [1b，FP32/缓冲区/cache](final_1b8c36d/buffer_cached_cli/actual-forward.json) | CLI | 64.597 | 71.573 | 18,138,267,648 |
| [1b，FP32/缓冲区/cache](final_1b8c36d/buffer_cached_api/actual-forward.json) | 已初始化 CUDA API | 42.089 | 50.164 | 18,138,267,648 |

1b 的三种缓冲区调用与修正 FP32 的原缓冲区结果：标签差 0、所有标签 Dice=1、几何差 0，保存的 MGZ 和 CSV 都逐字节相同。MGZ SHA-256 为 `50f9e58f…`，CSV 为 `aef61247…`，完整摘要见 [只读保存格式审计](final_1b8c36d/buffer_stored_dtype_audit.json)与[缓存 API 补测](buffer_cached_api_summary.json)。旧 probe 的 `same_dtype=False` 比较了保存 MGH 的 `>f4` 与内存 NIfTI 的 int32；不是仅字节序差，也不证明输出文件 dtype 不同。保留原报告，另将保存文件两侧重新读入。

1b 两次实际前向均为 float32、matmul TF32=True、cuDNN TF32=False、autocast 关闭，原图与翻转后验共用缓冲区。no-cache 和缓存 CLI 的单次观察未比对应阶段1更快，缓存 API 的单次时间较短；这些时序、上下文及共享负载不同，不能归为缓冲复用的因果加速。阶段1开启缓存采样超过 20,000,000,000 字节；1b 单阶段 cache 的采样值较低，尚未构成配对整例缓存实验。no-cache 的 allocated/reserved 不可用，记为 null，不记 0；1b cache CLI 实测 allocated/reserved 为 15,621,712,896/17,574,133,760 字节，API allocated 为 15,621,713,408 字节。详见 [精度说明](../../../../docs/recon_all/SYNTHSEG_PRECISION.md)及[独立 SynthSeg 子函数说明](../../../../docs/synthseg/README.md#recon-all-集成发现的精度设置覆盖)。

## 线程预算同输入回归

| 输入/阶段 | Numba 128 s | Numba 4 s | 输出比较 |
|---|---:|---:|---|
| [sub-01 CA normalize](cpu_final_1b8c36d/thread_ca_sub01/report.json) | 20.328 | 22.838 | 数值、dtype、几何/有序面相同 |
| [sub-02 CA normalize](cpu_final_1b8c36d/thread_ca_sub02/report.json) | 20.447 | 23.015 | 数值、dtype、几何/有序面相同 |
| [sub-01 LH standard sphere](cpu_final_1b8c36d/thread_sphere_sub01_lh/report.json) | 209.395 | 218.038 | 数值、dtype、几何/有序面相同 |

Torch 两侧均为 4；Numba 初始容量/掩码 192，测试退出恢复 192。时间包含首次 JIT 和 IO，固定先 128 后 4、未重复暖机，因此没有线程提速结论。两例 norm/ctrl_pts 全部差值 0；球面顶点数、有序面一致，坐标最大/P99 差 0。原生程序的线程数不能由 Python 的 4 推断；N4 和拓扑的固定预算保留。[线程说明](../../../../docs/recon_all/THREAD_BUDGET.md)。

## 旧整例变慢的原因：仍未确定

[历史整例重新分析](slowdown_analysis.json)将 sub-01 的主要增加定位到 MNI 非线性、WM edit 和 pretess；这不是当前版本的新整例结果。固定真实输入复测 WM edit 的两种 seed，在 GDB 下约 34.502/33.728 s，颜色表检查仅 0.101/0.033 s，体素、dtype 和几何相同，未支持“随机颜色表造成十分钟延迟”。旧监测缺少子进程 CPU/IO/等待分解，且有较大采样间隔、GPFS 时间与主机时间偏差，无法判定延迟原因；新剖析分别记录这些范围。

## 当前整例与受控基线

| 原始输入/主机 | 新版 | 同策略旧版 | 整例速度比 |
|---|---|---|---|
| sub-01 / gpucw1 | [冻结1b GPU重试](run_gpu_retry1_1b8c36d.sh)：完整命令 4972.667 s，API 4968.545 s | [e036受控完整命令](gpu_control_pair_summary.json)：5173.887 s；[复现入口](run_gpu_control_direct_20261001.sh) | 本次降时 3.889%，速度比 1.0405× |
| sub-02 / nodecw10 | [冻结1b CPU](run_cpu_final_1b8c36d.sh)：完整命令 5295.422 s，API 5292.995 s | [e036受控结果](cpu_control_pair_summary.json)：wrapper 入口全程 5374.315 s，校验后流水线 5367.985 s | 完整命令边界不同，不计算该比值；共同校验后范围本次降时 1.495% |

两例候选的版本、完整原始 JSON、每阶段及子步骤时间均在[整例剖析](candidate_whole_profile.json)。GPU 以初始化 CUDA 的 Python API 调用，CPU 以 CLI 调用；两者输入和主机不同，不能互作速度对照。GPU 主要耗时为双侧表面链 1531.251 s、球面配准 937.128 s、最终 white/pial 与指标链 777.542 s；CPU 对应为 1588.205、898.174、1124.751 s。GPU 66 阶段前后同步合计 0.009036 s，不能将同步当作本次主要瓶颈。

GPU 候选的父子进程同次查询显存峰值为 16,118,710,272 字节（16.12 GB），基线为 12,897,484,800 字节（12.90 GB）；本次候选采样峰更高，不能从单阶段缓冲复用宣布整例显存下降。候选/基线分别有 2316/2415 个样本，查询失败均为 0，最大间隔 3.118/3.372 s。no-cache 的 PyTorch allocated/reserved 不可用，未写成 0；两个采样值均低于 20,000,000,000 字节，但不保证连续峰值。现有网格检查覆盖有序面、有限坐标、Euler/边闭合及 white/pial 各自自相交；扩展质量检查的范围与阳性见下节。

GPU [完整受控配对](gpu_control_pair_summary.json)的 20 项可比条件全部成立，包括原始 T1 与程序哈希、主机/UUID、实际 Torch/Numba 掩码、FP32 前向、关闭缓存、监测器版本和采样设置。相同完整命令边界缩短 201.220 秒。统计组为 65.964→27.015 秒；注册 996.440→937.128 秒、表面初始化 1557.491→1531.251 秒、最终表面 804.342→777.542 秒。第二次归一化、MNI 等未改计算的阶段本次也变快，不能将整例差值全部归因于新缓存或厚度算法；这是共享硬件各一次整例的观察，未证明稳定因果提速。

[CPU 受控精度对照](cpu_control_precision_summary.json)的 138 项全部通过，174 个数值比较块的最大误差均为 0；七张标签图所有标签 Dice=1，68/45/70 区与全局体积指标差为 0，双侧八个表面阶段的有序面和坐标全部相同。这里指解析后的数值及几何，文件字节 SHA 可以不同。CPU 共同校验后流水线为 5367.985→5287.736 s；收益主要来自未替换 C++ 最终表面阶段的本次 152.070 s 墙钟缩短，被表面初始化增加 85.436 s 部分抵消，不能归为新算法因果提速。统计组本身为 14.360→7.417 s。

[GPU 受控精度对照](gpu_control_precision_summary.json)的 138 项均通过既有诊断门槛，但 174 个数值块中 20 个非零，完整清单保留。最大差为 RH `curv.pial` 的 0.000101447 mm⁻¹，P99 0.0000028014 mm⁻¹；面积和 TH3 顶点体积最大差分别为 9.54e-7 mm²、1.91e-6 mm³。厚度逐值相同，七张标签图零不同体素且所有标签 Dice=1；68/45/70 区与 16 项全局体积统计的已写出数值差为 0。双侧八个阶段有序几何相同；[cortex 索引只读核验](gpu_control_cortex_label_identity.json)也相同，因此扩展质量的控制版/候选穿越结果相同是由相同几何及掩膜推出的结论，不是重新搜索交点的报告。原 label 中重复索引未改。138 项通过不等于逐字节复现，也不代表相对官方整体等效。

### 固定曲率输入的重复性

[两轮 RH pial 曲率补测](curvature_repeat_rh_pial_1b/summary.json)固定本次候选几何、官方/Conda 程序、线程和 GPU UUID。官方两轮、Conda 两轮数值和文件 SHA 均相同；同版 PyTorch 两轮有 59663/104619 个不同值，最大/P99 差为 0.000102043/0.0000027120 mm⁻¹，与上述整例曲率尾差同量级。两轮 PyTorch 相对官方以及自身重复均无超出既有 `.005 + .001 × abs(reference)` 算子容差的顶点。曲率、面积模块和所用 `_normals` 函数段在旧版与 1b 中相同；候选机制包括 GPU 归约和局部线性代数，尚未定位具体算子，不能用这一阶段解释其他 19 张图或历史官方差异。保留 TF32 与当前 GPU 曲率路径。

两次外层命令为 12.551/13.406 s，GPU 函数含同步及读写为 5.112/6.998 s，采样父子峰均 1,990,197,248 B。设备同期共享负载较高，不据此判定速度因果；官方 C++ 此输入观察较快，但单个曲率阶段的观察不触发整条指标路径回退。第一次许可证环境缺失的失败也保留，未计为有效轮次。[复现正文](curvature_repeat_rh_pial_1b/analysis_source.log)、原始报告及监测与[模块说明](../../../../docs/recon_all/SURFACE_METRICS.md#curvature-repeatability-20261001)绑定全部输入和程序哈希。

两侧必须从原始 T1 与空输出目录开始，线程预算 4，GPU no-cache，并对旧版真实 SynthSeg posterior 作用域应用同一 cuDNN FP32 例外。控制仅在 benchmark wrapper 内对齐精度，未改旧算法/缓冲区，也不读官方输出。首次 GPU 尝试在 Talairach 子进程报告 CUDA OOM，exit=1，56.347 s；[失败日志](final_1b8c36d/full_sub01_monitor/command.log)与[监测](final_1b8c36d/full_sub01_monitor/monitor.json)保留，不作为完成或速度结果。

候选完整命令计时已包含校验、加载、传输、计算、读写和报告。缓存 API 探针已完成后，重复队列在同一监测目录触发 `FileExistsError`，没有启动第二次模型；保留失败日志，并以直接控制命令从空目录启动基线。没有预装脑影像软件的干净环境整例尚未验证；运行清单的哈希核对不代替隔离部署验证。

## 剩余 GPU 复用优先级

[源码与实际耗时审计](gpu_reuse_remaining_audit.json)核验冻结 1b 的 25 个文件。双侧注册为 937.13 秒，主要优化仍在 CPU；已有 CUDA 入口只覆盖部分 force 或末尾 overlap，不能通过改一个 device 参数宣称整阶段 GPU 化。standard sphere 的 finish 合计仅 3.22 秒，Jacobian/contrast 合计约 2.11 秒。两轮归一化已经使用主设备，CPU 控制点、SciPy 距离图和往返传输的内部耗时尚未拆分。

下一步先用报告中的完整 `cProfile` 命令剖析同一 LH 注册的 averaging、force 和步长搜索。注册的输入、临时目录及分侧输出独立，可随后比较双进程各 2 线程与串行 4 线程；已初始化 CUDA 不可使用 fork，主进程集中写运行报告。整个 surface 阶段仍保持顺序，因为 RH 的 `surface.defects.mgz` 依赖 LH 的共享 MRI 输出。本轮未执行注册并行或新 CUDA 内核，审计命令仅为下一步入口。

## 当前官方对照与局部异常

两例独立官方对照已完成，沿用原来的 138 项诊断门槛；[完整脑区与表面指标](official_candidate_metrics_summary.json)包含最大、P99、局部异常、原始报告和来源 SHA。没有以平均相关性代替以下结果。

| 当前候选相对官方 | sub-01 | sub-02 |
|---|---:|---:|
| 严格逐文件诊断 | 6/138 | 2/138 |
| aparc 68 区平均厚度 MAE | 0.04184 mm | 0.02169 mm |
| 面积绝对相对偏差中位数 | 1.350% | 1.060% |
| 灰质体积绝对相对偏差中位数 | 2.181% | 1.183% |
| aparc 体积分区 Dice 中位数 | 0.94929 | 0.96419 |

两例官方与候选网格首次在 `orig.nofix` 不同，最终有序顶点/面不对应，因此使用双向点到三角面距离，没有称逐顶点一致。sub-01 white 双向平均距离为 LH 0.07944/0.08616 mm、RH 0.07058/0.07105 mm，但最大值达到 5.513 mm；pial 最大 3.998 mm。sub-02 white 最大 2.900 mm、pial 最大 3.185 mm。局部脑区偏差也保留：sub-01 LH 尾侧前扣带平均厚度高 0.185 mm（8.24%）；a2009s 的 LH Jensen 参考 4 体素、候选 138 体素、交集 1，Dice=0.014085。sub-02 LH frontalpole 厚度最大绝对差 0.121 mm，RH rostral ACC 最大相对厚度偏差 5.03%。

当前检查图：[sub-01 T1 表面叠加](official_candidate_sub01/figures_readable/t1_surface_overlay.png)、[脑区误差](official_candidate_sub01/figures_readable/region_errors.png)、[局部边界](official_candidate_sub01/figures_readable/local_region_boundary.png)；[sub-02 叠加](official_candidate_sub02/figures_readable/t1_surface_overlay.png)、[误差](official_candidate_sub02/figures_readable/region_errors.png)、[边界](official_candidate_sub02/figures_readable/local_region_boundary.png)。布局修正只重绘，输入与原指标 SHA 未变；图及实际绘图源码/命令的哈希在各目录 `provenance.json` 和 `figure_sha256.json`。

`filled.mgz` 是独立半球标签空间，255/127 代表左/右半球，不能按全局 LUT 命名为胼胝体/软组织。比较器名称已修正并通过 3 项语义测试；原 Dice 报告保留，另生成仅元数据名称更正的派生 JSON，Dice 和体素未改变。

## 扩展网格检查与官方重复性

[完整补检](surface_quality_extended_summary.json)复用已有拓扑、球面面积和三角相交函数，另检查顶点 link 与 white/pial 完整空间候选。四个候选半球均为单连通分量，非流形边、异常顶点 link、重复面以及 sphere/sphere.reg 的负面积、零面积、非有限面积均为 0。white/pial 横向穿越采用事先固定的平面跨侧与正交线重叠阈值（均为 1e-6 mm），端点接触和共面/内侧壁重合单列。

| 真正横向穿越三角对（LH/RH） | 当前 1b | 历史 e036 | 官方参考 |
|---|---:|---:|---:|
| sub-01 | 147/197 | 189/226 | 175/496 |
| sub-02 | 92/119 | 92/119 | 142/197 |

候选四个半球中，双方所有顶点均在 cortex 的阳性分别为 124、171、67、75 对，穿越交线最大长度为 1.482、0.870、1.330、1.411 mm；位置、P99、局部平面深度和脑区归属都保存。平面深度不是最近表面距离或厚度误差。sub-02 历史与当前十张表面的解析有序几何全部相同，阳性细节也相同，因此该例异常已经存在。sub-01 网格不同，三角对数较少不能称质量改善；官方也有阳性不能作为豁免。原 standard gate 为 passed，扩展穿越检查为 positive，整体质量与指标等效为 `not_assessed`。

[官方重复性审计](official_repeatability_audit.json)优先复核已有同输入 white、sphere 和 pial 记录，其解析坐标重跑差为 0；部分历史记录缺少完整程序/硬件哈希绑定。本轮没有将旧参考重跑或跨环境差异解释成随机性，SynthSeg 的 148 标签变化来自已核实的 TF32/FP32 设置不同。当前官方整例重复性尚未补测。整体指标等效阈值仍待正式确认，已有局部异常和严格失败完整保留。

本次另修复比较器的 `filled.mgz` 标签名称：默认 255 为左半球、127 为右半球，不使用通用分割 LUT 的同值名称。[metadata-only 修复脚本](../correct_dice_label_metadata.py)生成 sub-01/sub-02 的新 `paired/dice_vs_*_semantics_corrected.json`，原始报告保留；脚本回填名称后断言数值、Dice、dtype、图像路径和原始哈希全部不变。仅修复说明字段，不重算或提高通过率。

## 复现命令与输入输出

以下为复现入口，不表示本页新增执行。使用同一个声明的 Conda Python、对应冻结源码 PYTHONPATH、已授权的真实数据目录；下列路径按报告设置，不下载影像、权重或许可证。输出独立诊断目录，失败应查看异常与 JSON，不能生成 complete 占位结果。统计/厚度输入采用同一 surface RAS 顶点顺序；SynthSeg 输入为冻结 conformed MRI 网格，输出分割沿用该网格与整数标签语义。整例输出使用原始 T1，新目录不得混入检查点。

```python
from pathlib import Path
import os
import subprocess

runtime_root = Path("/cwStorage/home/gongwk/Notebook_code/freesurfer_synth/work/recon_standard_20260929")  # 已安装的声明 Conda 运行目录
diagnostic_root = runtime_root / "volume_parity_20260930/fixes_20261001"  # 已有固定代码及报告
candidate_root = diagnostic_root / "code_1b8c36d"  # 使用1b时绑定实际版本，不冒充stage1
python_executable = runtime_root / "fnit_main_env/bin/python"  # 原始报告中的同一 Python
subject_directory = runtime_root / "volume_parity_20260930/full_sub01_e036f57_uuid"  # 固定的FNIT自产输入
environment = dict(os.environ)
environment.update(PYTHONPATH=str(candidate_root / "src"),
                   CUDA_VISIBLE_DEVICES="GPU-e25cac06-0ce8-a833-abf9-09ab18c9c9ba",  # 显式物理GPU
                   PYTORCH_NO_CUDA_MEMORY_CACHING="1",  # 初始化CUDA前设置低显存策略
                   OMP_NUM_THREADS="4", MKL_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4",
                   NUMBA_NUM_THREADS="4")  # 明确总Python预算；原生日志另行核对

subprocess.run([str(python_executable),
    str(candidate_root / "validation/recon_all/python_gpu_port/benchmark_surface_stats_cache.py"),
    "--subject", str(subject_directory),  # white/pial、注释、顶点图和统计前置文件
    "--baseline-source", str(runtime_root / "volume_parity_20260930/performance_e036f57_code/src"),  # 已有e036冻结源码
    "--baseline-commit", "e036f57b62b99d2af4cd8853ab2f1e6d2a9f8c68",  # 旧算法版本
    "--candidate-commit", "1b8c36d25a68e253a1e59b6d02114890afa467de",  # 本次复现使用的候选版本
    "--output", str(diagnostic_root / "reproduce_stats_1b"),  # 两轮stats和机器可读报告
    "--device", "cuda:0", "--threads", "4", "--repetitions", "2",  # 同输入交替运行
], env=environment, check=True)
```

其他入口沿用同一环境，参数均为显式具名选项：

- [厚度配对脚本](../benchmark_thickness_indexed.py)：`--white`、`--pial`（同有序网格，mm），`--baseline-source-file`（旧实现文件）、`--reference-map`（独立同输入 Conda 顶点图，仅诊断）、`--output`、`--code-commit`、`--device cuda:0`、`--threads 4`、`--repeats 2`；输出逐顶点厚度图、比较 JSON 和同步读写耗时。
- [SynthSeg脚本](../benchmark_synthseg_precision.py)：`--input`（conformed MRI）、`--weights`（已声明并校验权重）、`--baseline-seg`（同FP32诊断参考）、`--output`、`--cudnn-tf32 false`、`--allocator disabled`、`--initialized-api`、`--threads 4`、`--device cuda:0`、`--code-version`；输出分割、脑区CSV和两次真实前向设置。
- [线程脚本](../benchmark_thread_budget.py)：`--subject`（冻结FNIT目录）、`--assets`（GCA资产）、`--output`、`--stage ca` 或 `--stage sphere --hemi lh`、`--masks 128 4`、`--torch-threads 4`、`--code-commit`；输出两套 norm/ctrl 或 sphere 和严格比较 JSON。
- [当前整例GPU命令](run_gpu_retry1_1b8c36d.sh)、[CPU命令](run_cpu_final_1b8c36d.sh)及[受控旧版wrapper](../run_controlled_legacy.py)包含原始输入、权重、资产、设备、线程和输出路径。[监测器](../run_monitored.py)显式 `--gpu-uuid`、`--interval 2` 和 `--query-timeout 5`，以同次查询合计父子占用，不将不同时刻峰值相加。

[索引整理脚本](../finalize_performance_summary.py)只读取本目录原始 JSON 并核验旧 SHA，更新 `stage_summary.json` 的完成状态、时间和新证据绑定；不处理影像，也不修改原始报告或门槛。`--summary-repository-head` 是整理前的提交，实际计算版本仍分别为 e036、阶段1快照、1b 或独立 worker，不将整理提交冒充计算版本。实际执行示例：

```python
subprocess.run(
    args=["python3", "validation/recon_all/python_gpu_port/finalize_performance_summary.py",
          "--report-directory", "validation/recon_all/python_gpu_port/performance_20261001",  # 收回的真实报告目录
          "--summary-repository-head", "ef09caca4873a81ad9e39d200587773247d1fdde"],  # 整理前实际Git提交
    check=True,  # 缺失报告或原始SHA不符时停止
)
```

metadata-only 名称修复使用 `correct_dice_label_metadata.py --input 原始Dice.json --template 当前官方元数据修正.json --output 新名称修正.json`。三个路径均为显式参数：输入保持原文件，template 提供已核验的 filled 名称，输出须是新文件。脚本失败会抛异常；返回的是同数值结构的 JSON 及修复来源，没有对应官方独立 CLI。

真实精度与时间以原始 JSON 为准，表格仅显示三位小数。完成的 GPU 候选/基线最大采样间隔为 3.118/3.372 s，首次失败整例为 3.845 s；各阶段以对应 monitor 为准，未保证捕获连续峰值。不混用 GB/GiB。无新增运行依赖，现有 Torch/NumPy/SciPy/Numba/nibabel 已在主页 Conda 安装路径；Conda 独立源码构建程序继续保留。
