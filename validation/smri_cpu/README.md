# sMRI 同节点 CPU 对照

## 本轮范围

冻结起点为 `1d31e7baaebbb644ab199471f7fe6282721455fd`。在 nodecw10 对照 FNIT 与独立安装的 FreeSurfer 8.2.0-1、FSL 6.0.7.4，检查已实现的公共功能、精度和完整运行时间。原软件只参与参考测试，FNIT 推理不调用这些安装。

| 任务 | 函数 | 主要输出 |
| --- | --- | --- |
| 1 | SynthStrip、SynthSR | 脑图、掩膜、距离场；合成 T1 和几何 |
| 2 | SynthSeg、SynthSegPlus、WMHSynthSeg | 逐结构标签、皮层分区、软体积、病灶概率 |
| 3 | SynthMorph | 四种配准模式、正反变换、重采样和格式转换 |
| 4 | TorchFAST、FastVBM | 三组织分割、偏置、13 张 VBM 输出 |
| 5 | segment_4_subregions、run_recon_all_python | 110 分区硬/软体积、表面、顶点与脑区指标 |

功能覆盖、参数和复现命令放在各任务目录。官方存在而 FNIT 没有实现的模式单独列明，不计为通过。例如 SynthSegPlus 当前对应普通 SynthSeg 2.0 的 `--parc`，不对应 `--robust`。

## 线程和计时

- 第一轮主对照限定相同的 8 个物理核：`0,4,8,12,16,20,24,28`，均在 NUMA node 0；排除这些核的超线程兄弟。
- 第二轮将新组分配到不同的物理核，减少长流程排队。任务 1 沿用第一轮核组；任务 2 使用 NUMA 1 的 `1,5,9,13,17,21,25,29`；任务 3 使用 NUMA 2 的 `2,6,10,14,18,22,26,30`；任务 4 后续对照使用 NUMA 3 的 `3,7,11,15,19,23,27,31`。任务 5 的 stage 使用 NUMA 0 的 `32,36,40,44,48,52,56,60`，完整 recon-all 使用 `64,68,72,76,80,84,88,92`；有限 profiler 使用另外的 NUMA 2 八核，仅作热点诊断。
- 每个配对组内，原软件、冻结基线与候选使用同一亲和性和最大线程预算，取得该核组的文件锁后串行运行。不同核组可以并行；同一对照不能跨组配对计算提速倍数。各记录保存所属轮次和实际核组。
- 记录 OMP、MKL、OpenBLAS、Numba、ITK、TensorFlow 和 Torch 的实际设置、操作系统线程数量及节点负载。八线程配置与八物理核预算不等于进程恰好只有八个 OS 线程；空闲线程池也计入线程采样。节点还有其他工作负载，以上分组不代表独占 CPU 或内存带宽。
- 原 FAST、部分 N4 和拓扑程序可能只使用一个线程。相同预算不等于每个程序实际使用八核。
- 默认采用 reference → baseline → baseline → reference 的配对顺序。优化候选与冻结 FNIT 基线另做同输入对照。
- 主要墙钟从新进程启动到所有约定输出保存结束，包含导入、读取、权重、计算与压缩。程序内计算、模型前向、JIT 冷启动和热缓存另列。
- profiler 的观测开销、限时诊断和局部影像计算不作完整函数提速结论。中断和失败记录保留。

## 真实输入和资产

主要输入为 OpenNeuro ds003138 v1.0.1 的前三张原始 T1，按既有固定清单顺序选择；另外使用 ds000114 v1.0.2 的真实 T1 配准对和亚区阶段。ds003592 的原始 FLAIR、真实 EPI 帧以及公开低场 MRI、CT 用于对应功能。原始和经过裁剪或去面的影像分别记录，不能互换结果标签。

每个输入和权重记录文件大小、SHA-256、实际几何及来源。20 项已有模型和配套文件已按 `fnit.weights.WEIGHT_FILES` 校验。未获得镜像再分发许可的数据和模板只私密复用原作者来源，不随报告上传。官方规范化、aseg 和 wmparc 只用于声明的同输入阶段诊断；原始 T1 连续链使用 FNIT 自产中间结果。

## 精度和 GPU 回归

先核对网格、affine、dtype 和输出文件，再计算逐标签 Dice、硬/软体积差以及连续图的 MAE、RMSE、P99 和最大误差。表面逐顶点比较前验证有序顶点和面对应；缺少对应时使用空间距离并明确其含义。平均相关性不能代替局部差异。

CPU 修改优先保持 CUDA 分支。每个受影响组件还在 gpucw1 的同一 H100 上进行冻结基线与候选的真实输入回归，记录目标 UUID、精度策略、实际输出、内存和配对时间。不能只凭设备参数或代码检查宣布 GPU 性能不受影响。

## 继续修复：nodecw7 与 H100

本版从已发布的 `f1cbdab1` 继续，CPU 对照迁至现场确认的 nodecw7；不把 nodecw10 时间混入新配对。各组仍限制相同八个物理核心和线程预算，同组串行。GPU 用既有 H100 共用锁；数值、完整 header、显存和实测时钟分别记录。

| 功能 | 本版已完成的修复或定位 | 完整真实输入证据 |
|---|---|---|
| [TorchFAST](fast_fixes_20261004/README.md) | CPU 标量 C 指数/对数修复；默认和一组非默认八图与官方逐值相同 | nodecw7 默认完整 CLI 官方 368.198–411.291 s、FNIT 100.174–110.173 s，中位数比 3.706；完整脑 GPU 两后端各八图/headers 相同、显存相同，共享 GPU 时间另列 |
| [SynthMorph](../synthmorph/cpu_fixes_20261004/README.md) | CPU 解码、双向契约、有序插值、oneDNN 和矩阵顺序修复；精确 NumBa 采样与实际 Conda Eigen 缓存修复已接入 | joint 192/256 的完整场、脑内及零边界门通过。最新六次完整 CPU 两图两场及全部元数据与已验收 v29 逐位相同；256 同八核 ABBA 中位数 163.094→155.086 s，本组缩短 4.91%，RSS 11.718–11.799 GB。暖采样快 5.35/6.07 倍，局部冷 256 慢 12.3%，不混作完整时间。CUDA 路线及精度策略保持，14 项隔离门通过；完整 H100 两图两场和 reserved 17.836 GB 来自已绑定父版，未重复 GPU CNN 或宣称新的 GPU 提速 |
| [WMH-SynthSeg](synth_fixes_20261004/README.md) | 推理激活生命周期、CPU 权重搬运、GPU crop 工作区管理 | 完整 CPU 同官方标签/概率/CSV 相同；GPU crop allocated/reserved 为 18.374/18.438 GB，旧输出 SHA 和 57 次卷积 kernel 全同。默认 no-crop 保留原 GPU 路径、同实例模式恢复通过，但仍约 38 GB allocated |
| [SynthSeg / 普通皮层分区](seg_memory_20261005/README.md) | CPU decoder 复制到一个目标缓冲，保留原卷积和数值；16 次完整 FNIT CPU/GPU 运行的标签、CSV 和几何旧新相同 | nodecw7 八核候选冷进程 33 类 112.952 s、parc 55.684 s、fast 40.464–43.369 s；同节点官方正常冷 CLI 分别 55.046 / 48.296 / 33.784 s，另保留 33 类首次异常的 373.087 s，不用它作加速分母。两类时钟的校验与输出范围不同，分别报告，官方速度目标未达成。CPU 对官方仍为 1/5/1 个标签差异，没有新增。真实 H100 旧新输出和 allocated/reserved 相同，reserved 最大 18.900 GB；外部负载使本轮不支持稳定 GPU 提速或进程树物理峰值结论。Plus 已有 lazy 构造覆盖 TF32 的问题另列，未在本候选中修复 |
| [球面配准](recon_fixes_20261004/FULL_REGISTRATION.md) | CPU 平均保序并行；完整左半球同输入与官方逐点比较 | 119,451 点坐标/有序面/解码几何全同官方，旧新轨迹相同。新进程旧 764.907 / 新 348.038 / 官方 261.199 s，新仍慢 33.25%；GPU 平均三轮数输出/内存相同，原始 T1 整链未新跑 |
| [C++ 持续线程组平均器](average_cpu_persistent_20261004/README.md) | 四次冻结源码完整配准 ABBA；正式源、任务、运行库与相同 CPU 资源门通过 | 四次轨迹、119,451 点坐标、有序面和几何相同；对同输入保存的官方结果也相同，负面积面为 0。候选各 67 次 eligible 调用，实测 C++ team 均为 8。平均步骤中位数 55.151→26.447 s（2.085 倍）；完整进程 914.965→963.223 s，慢 5.27%。Numba 基线没有逐调用 team 采样，绑定的配置、环境和预算均为 8；官方是历史保存参考，未重跑。不能将算子收益称作完整配准加速 |
| [GEMS](gems_cpu_epsilon_20261004/README.md) | Gaussian 广播修复已验收；定位到 mixture 缺少官方 `1e-15`，完成 CPU 混合内部精度候选的全部 recipe 对照 | 首次完整梯度相对差约 4e−8，但最终 native 通过数仅 37/105→43/105、HR 41/107→45/107；丘脑丢失三个旧通过区，HA 仍有多数区域未过。保存全部 220 行 hard/soft 指标，候选不替换默认；原 CUDA 仍缺 epsilon |
| [GEMS 同点与首步定位](gems_first_trial_20261005/README.md) | 同一真实初态复现与已安装原版的目标、完整梯度、四面体方向及单次更新 | 三处同点梯度相对差约 5e−8、cost 绝差≤0.004632。旧丘脑/左 HA 对照漏掉负行列式时的 tet 换序，修正的是参考适配器，FNIT prior 不改。右 HA 首步旧 Armijo 最大位移 0.5 voxel、原版 1 voxel，按原搜索定义的探针舍入点相同。仅首步和同点评分，尚未通过完整优化、分割或速度门 |
| [FastVBM/FNIRT 定位](gems_fixes_20261004/FNIRT_READONLY.md) | 大量误差进入非线性估计段；最终乘法不是额外误差来源 | 固定官方场的实际转换/重采样 RMSE 4.39e−6，旧完整链 0.0189811。解析/稠密 Jacobian 定义差异另列 |
| [FastVBM CPU Jacobian 修复](fnirt_jacobian_20261004/README.md) | CPU FNIRT 采用已有解析 Jacobian；GPU/SynthMorph 保留原路径 | 同官方系数对 `jout` RMSE 7.37e−8；固定真实 GM stage 的 Jacobian RMSE 0.01404→0.01259。调制图 RMSE 略降、最大误差略增；完整 GPU 后处理三图/header/QC 相同，未重测整 pipeline |
| [FNIRT 首轮实测](fnirt_first_diff_20261004/README.md) | 首次分歧为正方向 moving GM 的平滑累计顺序；翻转控制整图逐值相同 | 初始 mask、fixed 平滑相同；首轮梯度/accepted coefficient 差很小。共享真实 FP64 系数 `4→2` 转换最大差 1.35e−12 mm；[完整平滑候选](fnirt_cpu_orientation_20261004/README.md)最终退化，未接入 |
| [FNIRT 非零状态](fnirt_nonzero_state_20261004/README.md) | 同官方 accepted 参数的 bending、梯度、Hessian 与正则权重状态对照 | regularizer action 相对差约 3.2e−15，未发现量级错误；baseline/flip 完整对照最早在 PCG 80/49 轮分叉，非线性估计仍未等价 |
| [FNIRT 共享 PCG](../fnirt_pcg_shared_state_20261004/README.md) | 同 H/RHS/初值/对角/容差重放实际官方与 FNIT solver | 第二次均 24 轮；第三次官方 69、dense 80、严格列序 49、除法控制 79。所有停止门正确；实际 FP64 尾差可放大为不同参数，未改生产求解器或重跑整链 |
| [SynthSR](synth_fixes_20261004/sr_cpu_followup/README.md) | CPU 推理使用已核验的 FP32 ELU/BN 顺序及临时 channels-last 权重；保留训练、hooks、autocast 和 CUDA 路径 | 默认两分支 22,020,096 CNN 值、9,072,000 浮点/量化值全同官方，3,522 个旧超限值降为 0。四参数分支和六真实域原门均通过；正常 CPU CLI 中位数 28.660→28.340 s，RSS 10.222→7.496 GB。GPU 新旧完整输出/header/文件 SHA 与 allocated/reserved 相同；共享时间另列 |

这版没有宣称全部 sMRI 功能已等价：丘脑/HA、FNIRT 非线性估计和 recon-all 的有效顶点对应仍见[剩余清单](REMAINING.md)。SynthSR 的 EPI NPZ 浮点仍有微差但通过原门；GPU 与旧 GPU 一致，未因此宣称 GPU 同官方 CPU 逐值相同。以下保留前一轮源码绑定结果，不改标为本版新测试。

SynthSR 修复并普通合并最新 main 后，本地组件回归为 **2,092 passed、5 skipped，138.79 秒**。覆盖前述组件及最新 FLIRT/FNIRT、ApplyWarp、ConvertWarp、InvWarp 和空间转换接口；测试和实际源码 SHA 见[最新整合记录](integration_sr_final_20261004.json)。首次整合的两项 NMI 测试固定统计 scatter 调用次数，未覆盖 CPU Numba 后端；已改为逐值检查送入熵计算的完整直方图，数值门未改变，生产数学也未改。两项 warning 来自既有 FastVBM profiler 测试未设 warmup，不能使用测试时间作 benchmark。

该 2,092 项记录早于本次 joint 原始插值与正式 CPP 平均器合并，只覆盖其注明的源码。中间版 `55d7a88f` 的 **2,148 passed、5 skipped，158.32 秒**见[当时整合记录](integration_20261005.json)，当时尚未包括安全 Eigen loader 和精确 NumBa 采样。

2026-10-05 整合版 `c2ceb18e` 的组件回归为 **2,218 passed、5 skipped，158.37 秒**；包含上述两项已验收 CPU 修改、正式 CPP 平均器合同与最新 main 文档合并，实际文件哈希及测试期间不变性见[该版整合记录](integration_final_20261005.json)。该版尚未接入 SynthSeg CPU 拼接优化。

随后 `7e0890a5` 接入保留原数值的 SynthSeg CPU 拼接，整合回归为 **2,288 passed、11 skipped、3 subtests passed，155.04 秒**；包含完整 `tests/synthseg_parc`，外层墙钟 158.44 秒，源码与测试文件在执行期间均未变化，见[接入后记录](integration_seg_20261005.json)。没有采纳 GEMS mixed 或新的 raw-prior 优化目标。两个 warning 仍来自既有 profiler 测试。前一轮 **931 passed、3 skipped，69.70 秒**及其源码仍见[原整合记录](integration_20261004.json)。组件回归不代替真实影像与原软件对照，也没有重新运行原始 T1 的完整 recon-all。

## 前一轮 nodecw10 验收记录（已发布）

以下为 2026-10-04 已完成的真实输入对照；每行绑定各自的冻结源码和计时范围。完整函数精度和性能以链接中的逐图、逐区报告为准。

| 功能 | 同组 CPU 原软件 / FNIT 时间 | 当前精度与性能结论 |
|---|---|---|
| [SynthStrip / SynthSR](strip_sr_20261004/README.md) | Strip 默认两例的中位数 45.56 / 14.90 秒、49.20 / 14.03 秒 | Strip 11 场景、26 CLI：mask/brain 相同，SDT 最大差 4.34e−5 mm。SR 16 场景的量化/格式门通过，默认浮点严格门仍有 3,522 点超限；未采用改变其量化输出的布局原型。另已完成[默认真实物化对象 API](strip_sr_20261004/in_memory_20261004/README.md)，各一次，与已有 FNIT 保存输出逐值相同。 |
| [SynthSeg / Plus / WMH](../smri_cpu_20261004/t2_seg/README.md) | 普通 SynthSeg 官方 46.05–127.19 秒、候选 124.61–171.51 秒；WMH 四组原软件 65.59–180.24 秒、候选 59.33–107.16 秒 | 普通 SynthSeg 保持原 CPU 卷积后端：12/12 旧新标签/几何检查通过，内存降至约 15.1 GB，尚无稳定官方速度优势。WMH 16/16 CPU 检查通过；20 GB 预算内的 GPU WMH 对照失败，未宣称 GPU 验收完成。Plus 为普通 `--parc`，并非 robust SynthSeg+。大尺寸 CPU 1×1 投影崩溃已修复：正式大图四进程完成，最小 Dice 0.999803；两例 fast/非 fast 旧新八进程标签和 CSV 相同。最新 GPU fast/非 fast 四进程回归完成，旧新标签/101 列 CSV/几何相同；reserved 最大 18.900 GB，共享负载下不宣称稳定 GPU 提速。 |
| [SynthMorph](../synthmorph/cpu_20261004/README.md) | 四模式原软件 35.81–261.35 秒、候选 20.53–169.25 秒，各模式分别配对 | 本例各模式完整时间较短。deform 两向通过；affine/joint 逆向场或上边界尚未全过门。GPU 四模式旧新数组和 header 相同。另已完成[affine256 真实物化对象 API](../synthmorph/cpu_20261004/in_memory_affine_20261004/README.md)；对象先解码 float64 与路径直接解码 float32 的首次输入差异已定位，其他对象模式未测。 |
| [TorchFAST](../smri_cpu_20261004/task04/README.md) | 显式 `execution="fsl"`、默认分割参数：8 核预算原软件 388.22–394.79 秒、候选 121.68–132.94 秒；严格单核 391.39 / 349.99 秒 | 8 核预算相同，官方 FAST 主要单线程。默认仍有少量 PVE 差；真实非默认配置八图相同。CUDA 两后端共 32 对 gzip 文件 SHA 相同。 |
| [FastVBM](../smri_cpu_20261004/task04/fast_vbm_cpu_20261004/final_v4/README.md) | 显式 `fast_execution="fsl"`：FNIRT 完整链官方 769.37 秒 / 最终 v4 536.79 秒；v4 SynthMorph 完整链 317.74 秒 | 最终 v4 两条原始 T1 链均返回 0，各 13 图数据与科学几何/header 与 v2 相同。FNIRT 灰质/Jacobian/调制图脑内 NRMSE 约 0.0190/0.0101/0.0160；SynthMorph 对应约 0.000562/0.000352/0.000451。沿用既有官方输出与计时，非邻接配对；官方 SynthMorph 仅分支实测，无冷整链速度比。 |
| [亚区分割 / recon-all](task5/README.md) | 脑干 205.55 / 764.69 秒；丘脑 275.89 / 2464.29 秒；双侧海马/杏仁核 442.30 / 3455.41 秒；完整 recon-all 官方 4600.04 / v3 4829.70 秒 | 脑干逐区门通过，丘脑与海马/杏仁核未全过，CPU 仍明显慢于官方。后续联合优化在另一核组的旧新配对降低脑干 API 时间 8.76%，全部后验/拟合状态相同。v2 因部署缺少 tifffile 失败；v3 重跑及[完整评分](task5/recon_complete_cpu_v3/README.md)完成，138 输出齐全。68 区厚度/面积/灰质体积 MAE 为 0.017 mm / 30.559 mm² / 75.529 mm³；顶点拓扑不同，44 顶点图与 12 注释的同索引误差为 NA。本例墙钟慢 4.99%，整体数值等价未判定。[raw 全亚区 CPU v5](task5/raw_all_cpu_v5/README.md)也已完成：墙钟 6241.435 秒、API 6238.458 秒；原网格 4/105 非空区通过，HR 5/107，分别 5/3 区 NA，尚不等价；无单次官方 raw 整链计时。 |

### 内存影像入口的新增覆盖

这组补测复用已经保存的官方和 FNIT 文件入口输出，没有重复官方推理。完整进程还包含来源校验、物化及事后比较，不能与上表的冷 CLI 墙钟直接相除为新的加速比。 本次对象补测仅在 CPU 执行，没有新增 GPU 回归。

- **SynthStrip / SynthSR**：冻结 `task5_candidate_cpu_v5`（head `00fedf35`、归档 `ab9a2d58…`），各运行一次 case01 默认 CPU API。输入是默认解码为 float64、保留 K-layout 的真实自有 ndarray SpatialImage，沿各自成熟 loader 的精度政策处理。Strip 三图、SR uint8 与浮点输出均与既有 FNIT 保存结果逐值且文件 SHA 相同。Strip 对官方的数组门通过，完整 header/pixdim 不逐字节相同，qform 最大差 `1.49983e−9 mm`；SR uint8 对官方门通过，浮点原门仍失败。API 分别为 **6.606 / 29.987 秒**，全部保存为 2.898 / 1.949 秒；含物化、检查和事后比较的完整进程为 25.530 / 41.545 秒。[完整对象报告](strip_sr_20261004/in_memory_20261004/README.md)保留各阶段与几何差异。
- **SynthMorph**：冻结 `task5_candidate_cpu_v4`（head `6f1e2b38`、归档 `ffda47a7…`），仅运行一次 affine256 双向真实对象 API；API 为 **12.765 秒**，保存为 5.401 秒，新进程完整墙钟为 27.289 秒。先默认解码 float64 再转 float32 的对象输出与既有路径候选有微差，官方逆向上边界原门仍未过。两个无 CNN 控制证明首次输入分歧来自 NIfTI slope/intercept 的解码精度顺序；直接 `np.array(ArrayProxy, dtype=np.float32, copy=True)` 物化与路径的张量、规范化网络输入逐值相同，但这第三路线没有重新运行网络，完整输出未验收。其他配准模式和对象参数分支仍未覆盖。[对象与缩放精度报告](../synthmorph/cpu_20261004/in_memory_affine_20261004/README.md)保留实际微差和未测范围。

已完成的相应 GPU 回归证明其对应 CPU 修改保留已有结果；这与 GPU 对官方精度验收是两项检查。WMH 在显存预算内失败，剩余问题保留在对应报告中。重建和亚区分割的已有算法差异分别报告，时间比较不自动等于等价重建加速。

[剩余验收与改进清单](REMAINING.md)列出已测试但未通过、尚缺真实完整测量及当前接口未实现的模式。

本轮使用的服务器默认 Conda 前缀缺少 `tifffile` 和绘图用 `matplotlib`，并非主页环境完整安装后的验收。重建 v3 使用任务独立的纯 Python TIFF 依赖目录；脑图事后由已有绘图环境生成。依赖清单中已包含这两个包，`tifffile` 也加入标准 Python 安装依赖；没有修改运行中的前缀。纯 Conda 全新安装检查仍需独立进行。
