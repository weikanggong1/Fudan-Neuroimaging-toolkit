# recon-all 慢阶段：优化与现有函数复用

本轮从工作分支 `764607c2e34c04bdc418fa64540380ff9d22c112` 开始优化。下文整例耗时属于已完成的计算快照 `1b8c36d25a68e253a1e59b6d02114890afa467de`。新修改另以 [CPU](candidate_source_snapshot.json) 和 [GPU](gpu_candidate_source_snapshot.json) 归档及逐文件 SHA-256 绑定。原生阶段的源码审计见 [机器可读记录](native_stage_gpu_reuse_audit.json)，它本身没有测试原生替换。

## 本轮修改与实测

同输入 sub-01 LH 完整球面配准 cProfile 为 **524.378 s**；有序面及坐标与冻结 1b 保存结果完全相同。[报告](baseline_register_profile/report.json)记录：103 次有序平均累计 241.165 s，27 次图谱平滑累计 117.288 s，其中有 10,773,744 次 `torch.roll`。剖析包含记录开销，不能当作无剖析的性能基线。

据此修改了自有算子：CPU 图谱平滑使用有序 double 累加的 Numba；三跳邻域使用整数 BFS；梯度平均增加可选 CUDA float32 有序 Jacobi 内核。remesh 复用顺序 Numba 平滑、缓存静态邻接并用 heapify 建初始队列；quick sphere 复用已有角点累加内核并减少重复几何计算；两处归一化控制点清理共用同一顺序 Numba 内核。59 项本地回归通过，真实阶段配对已完成；四侧真实回归通过后，生产入口已按目标 device 选择 CUDA 或 CPU 平均。

四个真实 remesh 配对均通过：坐标、有序面、尾部及拆缩边接受数量完全相同。计时含读写和首次编译/缓存加载；gpucw1，4 线程，逐侧各一次配对观察。[前三侧](remesh_other_three_pair.json)、[sub-02 RH](remesh_sub02_rh_pair.json)保存完整轨迹与哈希。

| 输入 | 旧代码 | 新代码 | 墙钟减少 |
| --- | ---: | ---: | ---: |
| sub-01 LH | 119.284 s | 105.333 s | 11.70% |
| sub-01 RH | 115.114 s | 104.008 s | 9.65% |
| sub-02 LH | 140.460 s | 127.147 s | 9.48% |
| sub-02 RH | 140.617 s | 119.205 s | 15.23% |

真实 sub-01 LH 首轮梯度的 16,384 次平均，暖运行四次中位数为 **39.496 → 0.318 s**，包含传输和完整迭代；0/1/64 轮同样通过零容差数值回归。首次 GPU 调用含编译为 2.805 s，静态邻接构造另计。[算子报告](gpu_average_sub01_lh.json)、[外部采样](gpu_average_sub01_lh_monitor.json)记录最大采样进程显存 486,539,264 字节（0.487 GB）、366 次样本、请求间隔 0.5 s、最大间隔 1.728 s，未验证连续峰值。这是算子结果，不能推算完整配准或整例提速。

完整 sub-01 LH 配准 [同输入配对](register_pair_sub01_lh.json)为 **503.818 → 168.824 s（2.984 倍、减少 66.49%）**。输出 SHA、坐标、有序面、尾部和所有保存轨迹一致。此处计时包含读写/JIT/设备初始化和同步，外层进程为 506.791 → 171.645 s。其余三侧与冻结相同输入结果比较也已通过：sub-01 RH 155.726 s、sub-02 LH 167.212 s、RH 180.185 s；保存轨迹、面与坐标一致。这里仅有新实现的重放耗时，旧保存时间不作同时性能对照。

quick sphere 四侧 [配对](quick_sphere_four_pair.json)均通过坐标/面/尾部和全部 trace 回归：sub-01 LH/RH 为 79.449/82.657 → 70.325/74.927 s，sub-02 LH/RH 为 92.659/95.266 → 85.156/85.288 s，单次减少 8.10%–11.48%。

sub-01 CPU 归一化 [配对](normalize_sub01_pair.json)：第一轮 84.645 → 80.945 s，第二轮 94.228 → 90.358 s；最终影像及全部诊断控制图的体素、dtype、shape、仿射完全相同。sub-02 第一轮 84.114 → 80.687 s、第二轮 105.981 → 100.384 s，同样通过所有终图和诊断图回归。新整例提速和整体指标等效尚未测得。

## 自有 CPU 阶段的加速空间

| 阶段 | 当前证据 | 本轮处理 |
| --- | --- | --- |
| 球面配准 | 本次剖析：有序平均与重复 roll 合计约 358 s | 编译平滑和 BFS；实测 GPU 有序平均 |
| remesh | 本次 sub-02 RH 新实现的 collapse 约 78 s，三次 smooth 合计约 2.41 s | 优化平滑和建堆；保留动态拓扑顺序 |
| standard sphere | 旧整例 141–238 s；本轮 headcw 同输入 CPU 剖析 156.913 s | 实测距离 SSE 64.410 s、法向22.248 s、平均16.288 s；下一步缓存静态 CSR 并优化距离目标 |
| T1/brain 归一化 | 旧 GPU 整例分别 185/175 s，含 CPU 控制点选择与传输 | 有序清理已编译；两例 CPU 完整阶段同输入配对已通过 |
| SynthSeg | 旧 sub-02 CPU 368.81 s，GPU 实现已存在 | 使用已有 GPU 路径，保留经过验证的 cuDNN FP32 例外 |
| MNI 非线性链 | 旧 sub-01：模型/保存 170.74，转换 16.81，求逆 89.47，检查图 7.13 s | 已使用 PyTorch deform，需拆分加载、前向、双向组合及保存计时 |

MNI 模型当前还生成下游未读取的逆向 DenseWarp，然后按既定语义用独立构建程序求逆。可研究仅省去未消费的后向积分、组合和 D2H；两次 UNet 构成反对称 velocity，不能省略其中一次。此项只有源码定位，没有本轮性能结论。

GPU 测试显式指定 UUID，保留 4 线程、现有 allocator 策略和 TF32 默认，不启用半精度。标量有序平均不涉及 TF32 矩阵算子。外部监测按同次 NVML 查询统计父子进程合计；关闭缓存时 PyTorch 峰值记为 unavailable。共享 CPU/GPU 的单次观察不代表稳定吞吐。

## 当前时间主要花在哪里

单位为秒；左右半球用 `LH / RH` 表示。sub01 使用 gpucw1 的 CUDA，sub02 使用 nodecw10 的 CPU，两者是不同被试和硬件，不能据此计算 GPU/CPU 加速比。以下是已有整例中的阶段或子步骤墙钟，包含各自封装内的加载、写出及子进程启动；子步骤已包含在所属大阶段中。

| 阶段 | sub01 GPU | sub02 CPU | 当前生产来源 |
| --- | ---: | ---: | --- |
| N4 封装 | 123.07 | 109.73 | FNIT 独立编译的 ITK C++，CPU；不是 FreeSurfer 程序 |
| GCA EM 注册 | 212.35 | 163.16 | 固定 FreeSurfer 源码在 Conda 中编译的 C++ |
| 拓扑封装 | 81.62 / 118.82 | 70.76 / 93.92 | FNIT Python 居中前处理＋修补版 Conda C++ GA |
| white.preaparc 放置 | 193.78 / 191.72 | 359.73 / 259.85 | Conda C++；阈值已由 FNIT Python/Numba 计算 |
| 最终 white 放置 | 212.41 / 158.99 | 263.73 / 243.57 | Conda C++ |
| pial 放置 | 182.80 / 192.47 | 277.54 / 293.89 | Conda C++；完整 Python pial 可独立调用 |
| 初次 inflation | 7.52 / 7.17 | 9.32 / 9.27 | Conda C++ |
| 最终 inflation＋sulc | 8.16 / 7.08 | 9.78 / 9.51 | Conda C++ |

原始数据源是 `performance_20261001/whole/{sub01,sub02}/candidate_run.json`，本审计记录了这两份报告的 SHA-256。`topology_native_seconds` 包住了 `run_topology_ga_conda`，其中先执行 Python 居中；因此上表不能解释为裸 GA C++ 时间。`topology_python_seconds` 另外包含初次平滑、remesh 和清交，也不能重复相加。

## 1．N4：保留算法，先测内部阶段和线程

[`tools/n4_itk/n4_itk.cpp`](../../../../tools/n4_itk/n4_itk.cpp) 第47行固定 ITK 为1线程，外层 `threads=4` 不会覆盖它。第69、80、104行分别执行 N4 拟合、全分辨率 B-spline 场计算和指数/除法输出，目前没有这三部分的同次计时。Python 封装还读写两份临时 float32 raw 图，256³时每份约64 MiB；这证明有读写步骤，尚不能证明它们是主要瓶颈。

[`n4_gpu.py`](../../../../src/fnit/recon_all/n4_gpu.py) 第1–4行已明确采用组织强度残差的平滑乘性场，**不是 ITK N4 的移植**。将它设为默认会改变算法。最小的下一步是给上述三个 `Update` 增加诊断计时，在相同 ITK 构建、输入、拟合参数和硬件下比较1/4线程，保留 float32 输出、uint8 转换及后续 `nu/norm/WM` 回归。没有实测前不更改默认线程或宣称提速。

已有 [同主机 N4 重放](../n4_same_host_replay_20260930.json)区分了 FNIT/官方实现差异与跨主机差异；它不是本次线程或 GPU 替换实验。N4 参数不能通过减少迭代、拟合层级或提高收敛阈值来换取时间。

## 2．GCA 注册：复用现有候选搜索，分块批量打分

[`register_t1`](../../../../src/fnit/recon_all/mri_em_register_python.py#L22) 已串联 GCA 读取、样本生成、平移和线性搜索、EM 及 LTA 写出。历史单个冻结真实 T1 的 LTA 最大矩阵误差为 `7.45e-9`，315,638个样本的舍入后源体素映射相同，见 [阶段验证](../MRI_EM_REGISTER_VALIDATION.md)。这份记录不是当前两例的 GPU 回归。

该函数第49行调用的 [`first_em_line_search`](../../../../src/fnit/recon_all/mri_em_register_optimizer.py#L108) **只执行首次搜索方向的线搜索**。其他输入需要的后续 EM 方向/轮次尚未得到一般验证，不能把整个函数直接作为默认原生替换。

适合先复用的是 [`search_linear_iteration_source`](../../../../src/fnit/recon_all/mri_em_register_search_source.py#L35) 第54–75行：固定一个搜索轮次后，多组候选变换的样本得分互不依赖，可以按候选和样本分块批量计算。现有 [`_sample_log_values`](../../../../src/fnit/recon_all/mri_em_register_search_jit.py#L16) 已定义坐标舍入、越界和似然规则；保留它作为同输入对照，避免重写 GCA 输入准备和搜索规则。

历史 Python 整阶段中，线性搜索205.25秒、EM5.75秒；这是旧 Python 路径的剖析，不能当作当前 C++ 的内部热点比例。新的 GPU 试验应保持候选枚举、同分选择、源体素舍入及停止条件，先检验 LTA与样本映射，再检验后续归一化和 WM。特别是坐标跨过体素舍入边界时，差异会改变被采样的强度，不能统一视为浮点尾差。完整 EM 尚未完成时，可以验证新的打分内核，不应默认替换整个注册。

## 3．拓扑：已有模块是前处理和算子，完整 GA 仍用 C++

[`topology_conda_ga.py`](../../../../src/fnit/recon_all/topology_conda_ga.py#L43) 先复用 FNIT 居中 sphere，再以1线程、种子1234执行固定源码的 GA。Python 已有 MRI/曲率评分、边筛选、fitness 合成和排名，但 [`topology_first_candidate.py`](../../../../src/fnit/recon_all/topology_first_candidate.py#L1) 与 [`topology_fitness_search.py`](../../../../src/fnit/recon_all/topology_fitness_search.py#L1) 明确没有实现完整 mutation、crossover 和全部缺陷修复。

历史修补版 C++ 在双侧冻结输入上得到相同的有序网格，详见 [拓扑说明](../../../../docs/recon_all/TOPOLOGY_CONDA_GA.md)。当前没有新的裸 C++ 热点剖析。先保留完整 GA，测出具体评分内核的占比后，再复用这些 Python 算子做局部移植与核对。将首次候选筛选或球面投影直接当作完整拓扑修复，会遗漏必要步骤。

## 4．white：复用现有内核，补齐连续四轮后再接默认

`white_preaparc_conda.run_white_preaparc` 第37–39行已经调用 FNIT `write_autodet_stats`；每侧约2秒的阈值计算不是主要时间来源。后续表面放置与最终 white 仍使用 Conda C++。

[`place_white_preaparc_prefix`](../../../../src/fnit/recon_all/place_white_preaparc_python.py#L38) 是首轮1–17步的诊断函数，第53行限制步数。已有 [首轮报告](../white_python_first_pass_20260927.json) 和 [第二、三轮报告](../white_python_passes_2_3_20260928.json)，后者从各轮冻结官方入口独立重放，不是连续四轮的生产替换。当前仓库没有完整的独立 Python 最终 white 函数。

无需重复实现 MRI 准备、边界搜索、强度/弹簧/曲率项、邻域与步长决策。它们已被 white 诊断和完整 pial 复用。GPU 或自有编译内核应先替换经同输入验证的独立计算，再实现 white 的完整轮次、状态传递、清理及诊断图。只完成首轮不能写出最终 white 的标准文件名。

两侧 white.preaparc 共用 `mrisps.wpa.mgz`，最终 white 共用 `mrisps.white.mgz`。直接对同一被试目录启用线程池会产生写入竞争；并行实验需要隔离输出、保持官方覆盖语义，并遵守总CPU线程预算。

## 5．pial：完整函数已经存在，优化动态碰撞与调用开销

[`place_pial_t1`](../../../../src/fnit/recon_all/place_pial_python.py#L53) 已包含四轮优化、内侧壁固定及相交清理。标准 runner 当前默认使用 Conda C++；Python 函数仍用于独立计算和验证，不是未实现或被删除的功能。

历史冻结官方输入上，Python 双侧有序顶点和面完全一致，耗时约1,241/1,154秒，见 [完整说明](../../../../docs/recon_all/PYTHON_PIAL_PLACEMENT.md)。但 [自产输入的双引擎比较](../native_pial_candidate_20260929.json) 中，Python/C++ 的 LH/RH 最大位移达到2.5910/1.3418 mm，P99为0.2797/0.1313 mm。这是局部几何差异，不能在未定位前认定为随机性或无影响尾差。

[历史 Python 碰撞剖析](../place_surface_collision_profile_fast.json) 的一次热运行为17.867秒，其中604,419次 `query_ball_point` 累计7.945秒；[梯度热运行](../place_surface_gradient_profile_cached.json) 合计1.397秒。两份记录来自旧冻结输入诊断，不是当前 C++ 的剖析，更不能推算整例加速比。

现有 [`asynchronous_first_step`](../../../../src/fnit/recon_all/place_surface_collision.py#L395) 第449–513行依次接受顶点，每次更新都会影响后面的邻域和碰撞。第477行在 Python 循环中查询 KDTree；拒绝试步还保留旧 face-hash 状态。优先方向是复用现有几何和决策内核，将动态 broadphase、候选筛选与顺序接受放进自有编译内核，减少数十万次 Python 调用。边界、强度和梯度等读取同一固定网格快照的部分可以单独验证 GPU 批处理。

不能将逐顶点异步更新改成全顶点 Jacobi 更新，也不能缓存跨试步已经失效的碰撞数据。即使局部 GPU 核更快，完整阶段还需要计入 CPU/GPU 往返、所有试步、pinning、cleanup与文件读写。切换默认前需要两例双侧同输入回归、逐轮定位、最大/P99几何差异及局部质量检查。清理函数返回的残留相交计数也应被质量门槛检查。

## 6．inflation：已有 CPU 实现，但当前不是慢阶段

[`inflate_surface`](../../../../src/fnit/recon_all/inflate_python.py#L271) 的已有 NumPy/Numba 路径在冻结真实双侧 `-no-save-sulc` 输入上保持有序几何一致。一次 [同输入 LH 配对](../inflate_qsphere_paired_lh_headcw.json) 为原生6.986秒、Python41.710秒；较新的24.25/30.31秒 Python 记录不是配对提速结果。

当前每次原生 inflation 只有约7–10秒。另一个接口限制是该函数只写 inflated 坐标，没有生成 sulc；默认第二次 inflation 同时要求 `inflated` 和 `sulc`。因此，它不能整体替换两个默认调用。先保留当前原生路径，把资源放到实际超过100秒的阶段。

## 建议的最小推进顺序

1. 为现有 N4 的三个内部阶段增加诊断计时，做相同算法1/4线程的两例回归。
2. 用现有 GCA 输入和搜索代码试验分块候选打分，完整 EM 未验证前保留原生注册默认。
3. 基于完整 Python pial，优化动态碰撞的编译内核；white 复用同一套已验证部件，补齐连续状态与全部轮次。
4. 保留完整 topology 和 inflation，只有内部剖析与完整配对结果支持时才更换。

各阶段使用预先声明的算子门槛，并报告严格复现、优化是否引入退化和整体指标等效三个状态。尚无经确认的整体等效门槛，本审计不设定或放宽阈值。保持 TF32 默认和已有 FP32 例外，不启用 FP16/BF16；完整进程显存按20,000,000,000字节预算记录，新增编译内核或依赖需纳入主页 Conda 安装。原生替换尚无本次 CUDA benchmark；新的自有算子配对结果见本页开头。

## 原实现与方法

- [固定 FreeSurfer 源码 d932c45](https://github.com/freesurfer/freesurfer/tree/d932c45b7941662ea380a05efef580568b98d41a)：注册、拓扑及表面优化的算法对照。
- [ITK](https://github.com/InsightSoftwareConsortium/ITK)：N4 拟合与 B-spline 场的实现。
- Fischl B. FreeSurfer. *NeuroImage*. 2012;62(2):774–781. [doi:10.1016/j.neuroimage.2012.01.021](https://doi.org/10.1016/j.neuroimage.2012.01.021)。


## 最终索引改写与生产接入

[共享 CSR 四侧验证](ordered_face_csr_four_pair.json)保持整数关联和 float32 法向逐位一致，逐元素 Torch 写入改为稳定排序与向量化填表，单次构造由5.73–7.47 s降至0.085–0.099 s。包含此修改的 [LH 完整配准](register_final_csr_sub01_lh.json)为150.171 s，表面、sulc seed及全部保存轨迹严格一致。

[sub-01 RH](register_saved_sub01_rh.json)、[sub-02 LH](register_saved_sub02_lh.json)、[sub-02 RH](register_saved_sub02_rh.json)在CSR改写前已完整回归通过，候选阶段155.73/167.21/180.19 s；它们仅与冻结输出对比精度，历史时间不是同时对照。标准入口现按device选择平均后端，末尾清理及半球共享输出顺序沿用原规则。

[第二例归一化](normalize_sub02_pair.json) CPU 配对为第一轮84.114→80.687 s、第二轮105.981→100.384 s；两例全部最终影像及诊断控制图零差异。

[资源复核](runtime_fingerprints_hotspot_candidate.json)重新读取两幅原始T1、11项权重、102项资产及14个Conda程序：大小/SHA改变数为0。官方程序SHA沿用既有核验记录，本遍没有重新生成官方参考，也未验证干净部署。

[最终源码快照](final_candidate_source_snapshot.json)绑定生产选择与共享CSR；前两份快照保留作阶段对照。新整例使用该快照、原始T1和新空目录，执行、完整性、网格质量、严格复现、优化退化与整体等效分别报告。

## 本次原始 T1 整例与代码绑定

生产代码提交为 `c24852054f3321c1142b1ae88fa3d2bf68329bb3`，工作分支为 `recon-all-hotspots-20261001`。最后 CPU/GPU 共用 [快照](final_candidate_source_snapshot.json)归档 SHA-256 为 `098ccb63a3931a4f749710fb76dd9beb4751f7b8b14c0f130fa922f9698b708e`；18 个源码、测试及脚本文件与提交的 SHA 核对一致，见 [绑定](tested_commit_binding.json)。最后 CSR 接入后的 LH 完整配准重放为 [150.171 s](register_final_csr_sub01_lh.json)，输出与保存轨迹一致；这不是与前一轮 168.824 s 配对的性能比较。

第一次空目录整例因启动器未传已声明的 FS_LICENSE 路径，在 mri_em_register 许可检查失败。保留两例失败报告；补齐启动环境后，从原始 T1 和新的空目录重启 `retry1`，没有复制失败目录中的中间结果。sub-01 为 gpucw1/H100 已初始化 CUDA 的 Python API，sub-02 为 nodecw10 CPU CLI；两者均4线程。配置见 [sub-01](sub01_whole_retry1_config.json)、[sub-02](sub02_whole_retry1_config.json)，使用 [启动器](../run_full_hotspots_launcher.py)及 [只读比较器](../collect_hotspot_whole_comparison.py)。整例还在运行，完成后以包含导入、校验、加载、传输及读写的墙钟判断提速。

[热点来源与自有替代审计](../../../../docs/recon_all/HOTSPOT_ACCELERATION_AUDIT.md)明确区分原生 Conda 程序、自有 CPU/GPU 计算、成熟替代与待补连续验证的实现。输入、11个权重、102个资产和14个候选程序的大小与 SHA 当前复核差异为0；未读取许可证内容、未下载新数据。参考程序哈希沿用已完成记录，本轮未重新运行官方程序；干净隔离环境验收仍未完成。

标准球面另在 headcw、4线程、相同 sub-01 LH 输入做本轮 [cProfile](standard_sphere_profile.json)：156.913 s，坐标和有序面与保存结果完全相同。距离 SSE 累计64.410 s（2103次），法向22.248 s（212次），有序平均16.288 s（212次）。其中线搜索与其 SSE 属于嵌套耗时，不能重复相加；下一步优先缓存不变拓扑并优化逐顶点距离目标，不能沿用配准平均的124倍算子结果推算标准球面提速。
