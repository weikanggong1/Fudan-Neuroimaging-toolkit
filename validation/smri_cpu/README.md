# sMRI 同节点 CPU 对照

## 最新FNIRT采样对照

[真实体积CPU候选](../fnirt_cpu_sampler_case_20261006/README.md)在16,128个记录坐标上，通过plain值、partial值和三个导数逐位门；当前helper仍有2338/813/0/913/1520个word不同。129,024个角点同输入门先通过，四次实际API及冻结源码已独立核验。候选公开在验证目录，默认注册器未替换。随后[同目标坐标/RHS控制](../fnirt_cpu_sampler_case_20261006/affine_coordinate_v3/README.md)通过12个实际消费矩阵word、16,128个坐标键、完整顺序输出、mask及129,024个角点门；候选partial值/三个导数逐位同。FSL序正梯度总相对L2由4.1597e−7降至8.7145e−10，三系数块仍有9.07e−8/3.21e−8/3.16e−8的相对差，1,177个Double word仍非逐位同。完整H/diag、求解与配准输出、fallback和GPU保护仍待验收。各API首编译与时钟分列，不报告加速倍率。

[最新除法投影控制](../fnirt_cpu_sampler_case_20261006/projection_division_v1/README.md)复用同一保存导数，48,384个投影值逐位同。固定lambda的FSL累计序g相对L2由8.7145e−10降至5.26766e−15；XYZ系数块分别3.14e−15/1.47e−15/2.16e−15，仍有1,170个Double words尾差。LM序仍3.62e−8。157项执行门与资源收尾已过，完整H/diag、求解及最终影像尚未验证，默认CPU/GPU路径不变。

[保存几何支持集诊断](../robust_register/support_points_results_20261006/README.md)已完成一次4全网格和4单点验证：原两阶段warp指标精确复现；两个差异点的round域、clamp、rint和FEQUAL整数捷径一致，插值角点记录不同。支持差异仍各1体素，原正式17/20保持；这是保存几何的观察，不证明优化器首差。[根复核](../robust_register/support_points_root_review_20261006.json)核对原回执和四个当前Git文件，没有新注册、GEMS、原软件或GPU调用。

[保存状态的 X0 单列对照](../fnirt_cpu_sampler_case_20261006/hcol_x0_v1/README.md)已完成：196项合同门通过，组合列59/1177个Double词不同，相对L2为2.94287e−15。该边界方向的候选数据项全部为正零，结果主要验证bending及接口，不能代表非零数据项或完整H。data列含冷JIT为1.362秒，worker为88.437秒；其余导入和核验时间未细分，没有API或端到端加速结论。下一步用原始输入自产内部图像与坐标，核对cf/grad缓存、FSL顺序RHS和连续PCG，再验完整CPU配准及GPU保护。

## 最新 robust QR 与 IRLS 对照

[首轮 A/b 连续对照](../robust_register/first_ab_cpu_prefix_20261007/README.md)已完成：同一真实输入的准备态、两层金字塔、半空间变换及图像、9196×6 的 A 和 9196 项 b，18 个记录边界全部逐位相同。原 SDK 的一次四阶 Schur 返回成功；参考是固定源码观察器，不是安装程序内部 trace。候选冷子进程为16.506秒，其中四次运行库身份检查约12.573秒，不能据此计算生产函数提速比。正式完整刚性/仿射仍17/20通过。

[同一 A/b 的完整 IRLS 对照](../robust_register/same_ab_irls_cpu_20261007/README.md)已定位首差：第一轮的中心值、MAD、尺度、归一化残差、权重及加权 A/b 全部逐位相同，QR 的6个Float32参数首次不同，最大绝对差8.22544e−6。双方都执行4轮并回退到第3轮；最终6个参数最大差3.29018e−5，9196项权重中2473项不同。同一权重输入的累计权重和也相差0.027832，需修复CPU归约顺序。新的CPU Float LINPACK候选在保存的同一加权A/b上求解一次，6个返回值全部逐位匹配原生；核函数为0.595毫秒，首次类型编译为0.919秒。尚未接入完整IRLS，完整warp、核团与GPU保护仍待验收。

## 最新C24完整接入结果

[完整CPU/GPU对照](seg_columns_c24_integration_20261006/README.md)已完成一次同公开T1的CPU8 ABBA和GPU AB：API中位85.499→80.621秒（缩短5.71%），分割、CSV、完整header和gzip SHA保持原输出；GPU实际设备/精度/allocated/reserved及采样内存保持。冷进程首臂20.286秒未细分检查使整体15.29%观察含检查时间差异，不能当计算加速。既有Conda冷/暖编译通过，sdist三个源文件逐字节包含；完整wheel/全新安装未测。GEMS与FNIRT严格验收仍未完成。

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

最新独立[robust配准前处理](../../docs/robust_register/PREPARATION.md)已完成同例CPU门：目标掩膜和右侧atlas的体素、13项MGH字段及存储affine均与官方相同；仅比较保存网格，gzip和可选tag另列。原报告序列化失败已修复，原两次exit1保留，未重算图像；[独立复核](../robust_register/target_preparation_20261006/ROOT_REVIEW.public.json)验证这一范围。该模块尚未接入GEMS。随后一次刚性→保存重读→仿射对照完成，133个点的最大世界坐标差为7.742e−6 / 4.487e−4 mm；刚性和仿射各有1个支持集体素不同，仿射同采样器warp相对L2为2.588e−5，超过预定1e−5。[独立候选](../robust_register/rigid_affine_20261006/README.md)17/20门通过，未采用；最终核团Dice/体积门仍未通过。

随后独立[Float反矩阵源码顺序合同](../robust_register/inverse_order_20261006/README.md)对六个已保存矩阵的determinant、reciprocal、inverse共18门逐bit通过；[root复核](../robust_register/inverse_order_root_review_20261006.json)核对原receipt与22个Git文件。旧刚体不同word仅为带符号零，四个仿射矩阵有1.19e−7至7.63e−6的非零数值差。该合同不读取MRI、不证明安装binary指令一致，旧17/20影像门保持原结论。

随后[真实求逆顺序候选](../robust_register/inverse_real_20261006/README.md)仍为17/20门通过：仿射同采样器warp相对L2为2.556643e−5，比旧候选降低约1.22%，仍超过原1e−5；两阶段各差1个支持集体素。刚性完整receipt记录26次局部求逆；仿射已保存两输出，但报告因NumPy int32不能JSON序列化而失败，其API时钟与内部flags记NA。随后只评分已保存输出，未重跑注册；[独立复核](../robust_register/inverse_real_root_review_20261006.json)核对518项绑定及原退出码。候选未采用，未形成等价提速结论。

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

[同投影Hessian对角线对照](../fnirt_cpu_sampler_case_20261006/diagonal_only_v2/README.md)已完成：未加扰及乘1.001后的1177项对角线相对L2分别为7.08786e−16、7.08078e−16；三个392项系数块约1.5e−15至2.4e−15，整体范数由scale项主导。两向量仍各有1166个Double words不同，不能称逐位同。仅执行一次对角线计算，未计算非对角项、完整H或求解；完整缓存、非线性配准及GPU保护仍待验收。

[Robust完整初始化对照](../robust_register/centroid_m0_cpu_prefix_20261006/README.md)在同一真实输入上使两幅准备图像和全部54个Double初始化值逐位匹配SDK参考。后续迭代及正式17/20完整配准门仍待解决。

本版从已发布的 `f1cbdab1` 继续，CPU 对照迁至现场确认的 nodecw7；不把 nodecw10 时间混入新配对。各组仍限制相同八个物理核心和线程预算，同组串行。GPU 用既有 H100 共用锁；数值、完整 header、显存和实测时钟分别记录。

最新[普通33类 CPU 分步骤观察](seg_cpu_profile_20261006/README.md)只运行一次当前源码的完整 API，未重复官方、parc/fast 或 GPU。当前标签、每区 Dice、完整 header、体积 CSV 和压缩 SHA 与已验收候选相同；106.377秒观察时钟中，CNN81.716秒、平滑13.136秒。正式112.952秒/官方55.046秒保留，分步骤诊断不能记作新的提速。源码、工具与543项事件聚合已[独立复核](seg_cpu_profile_20261006/ROOT_REVIEW.json)。

最新[FNIRT自有CPU归约核验](../fnirt_cpu_reductions_20261006/README.md)通过保存真实系统的465项标量、186项相对范数和自然69轮的全部向量／最终解逐位门，运行前后503项绑定通过。该候选使用现有Numba/llvmlite；没有改生产、重新组装或启动完整配准。当前17文件和28份报告/工具的Git字节已[独立复核](../fnirt_cpu_reductions_20261006/root_review.public.json)；矩阵自由接线和梯度组装差异仍待处理。

随后复用已存当前H/g/独立diagonal完成[旧新PCG两臂](../fnirt_cpu_current_replay_20261006/README.md)：自然53/49轮，同当前系统的真实残差7.82e−4/4.17e−4，均符合原1e−3停止门。与另一存档原生系统解的相对L2为11.53%/14.67%，这组跨系统距离不支持默认替换，也不能单独判定算术bug。只读保存解诊断发现，新旧差方向的逆增益为748.8；这是混合参数坐标中的单个方向，不是全局条件数或voxel误差。矩阵组装和任意方向的matrix-free回调仍待核对。

[一次当前level重建](../fnirt_cpu_level_rehydrate_20261006/README.md)已在第二accepted参数点通过：count、SSD、bending、λ/cost和完整1177项gradient/独立diagonal逐位等于已保存当前系统，linearize含evaluate为2.104秒。只运行一次evaluate/linearize，没有callback、H物化或求解；私有状态可供后续诊断，尚未验证NPZ布局恢复和任意方向算术。30份Git产物、18个当前源码及原始summary已[独立复核](../fnirt_cpu_level_rehydrate_20261006/root_review.public.json)，不作为完整配准提速或官方等价结论。

该回放的41份原报告/工具、17个当前生产源、509项前后绑定、追加10/17/5项绑定及残差标量已经[独立复核](../fnirt_cpu_current_replay_20261006/root_review.public.json)。没有重复计算模型或求解器。

随后[真实matrix-free callback上的PCG两臂](../fnirt_cpu_matrixfree_pcg_20261006/README.md)在同一恢复状态自然49/53轮停止，实际相对残差7.102e−4 / 6.770e−4均达到原1e−3门；两更新向量相对L2差6.9665%。首个方向已在预条件阶段分叉，没有原生同系统解，未接入默认。两个unit门、106次实际callback、完整绑定与原summary已[独立复核](../fnirt_cpu_matrixfree_pcg_20261006/root_review.public.json)；控制tau不是已确认的历史solve3阻尼，冷JIT顺序时钟不作速度比。随后[首方向诊断](../fnirt_cpu_initial_precondition_20261006/README.md)确认内部下限改值0；倒数再乘/直接除法造成332个1 ULP差异，同一向量下的点积/范数尾差单列。两个原首方向已逐位复现；0 callback/PCG/影像/官方，记录已[独立核验](../fnirt_cpu_initial_precondition_20261006/root_review.public.json)。完整更新、组装和非线性配准仍待处理。

后续[普通33类平滑候选](seg_cpu_blur_trial_20261006/README.md)在同一真实概率张量的三次实际逐位门全部通过，局部 worker 峰值RSS最大值10.295→5.592 GB；操作中位数6.963→6.978秒，未证明加速，暂不接入默认。只恢复一次已保存 decoder 末端，不记作完整CNN或T1实测；原始12份产物、14个当前源文件、三次位比较与时钟已[独立核验](seg_cpu_blur_trial_20261006/ROOT_REVIEW.json)。

[FNIRT平滑moving单输入控制](../fnirt_cpu_rhs_moving_control_20261006/README.md)已自然exit0：12项baseline/fixed/scalar/gradient/mask/count门全过，只换官方保存的平滑moving图。固定参数和λ下，总g相对误差8.364e−7→2.359e−7，FSL顺序控制7.366e−7→2.484e−7；三coef分量各改善约21%–30%，全部1177项仍不逐位同。只采样一次，0H/diag/PCG/native/full；官方SSD没有独立记录，不推断完整warp或分割改善。源码/operands/flags及三exit已[现场独立复核](../fnirt_cpu_rhs_moving_control_20261006/root_review.public.json)，默认生产未变。

[当前FNIRT编译CPU平滑](../fnirt_cpu_smoothing_bridge_20261006/README.md)补足旧保存图到当前Numba实现的桥接：plain→旧plain、header adapter→官方保存图两个整图各18,579,456个FP32值逐位同，三exit0。31项源/输入、12项harness与operands/flags前后相同；实际仅2次blur、1次adapter，不运行归一化/RHS/H/PCG/官方/GPU/完整配准。旧完整方向候选误差扩大仍保留，默认未采用；[独立复核](../fnirt_cpu_smoothing_bridge_root_review_20261006.json)只确认预处理身份。

[单处除法顺序控制](../fnirt_cpu_rhs_projection_control_20261006/README.md)已完成：原完整RHS先逐位重现，再仅替换signed-axis FP32除法。17个前置门和所有后门通过，scale逐位不变，但对存档官方总梯度相对L2由2.4836018e−7变为2.4838650e−7，未改善；x轻微改善、y/z轻微变差，因此不接入默认。仅1次采样/2次梯度前缀，无H/PCG/官方/GPU/完整配准。v1的nibabel头信息表示守卫错误及原exit1/1/0完整保留，v2仅修守卫后exit0/0/0；[独立复核](../fnirt_cpu_projection_root_review_20261006.json)确认32项源/输入与15项harness前后身份，尚未证明最终非线性输出一致。

[保存当前moving的scale/SSD累计控制](../fnirt_cpu_scale_accumulation_control_20261006/README.md)已完成一次四成员诊断：baseline两标量逐位复现，Z/Y/X串行Double后scale仅变3.19744e−14、SSD仅变2.13163e−14，同串行和的两种最终分组逐位相同。对当前存档native scale约7.66992e−6的差异没有实质改善；官方moving替换状态的2.54688e−6残差属于另一输入，未在本控制中重建。[根复核](../fnirt_cpu_scale_root_review_20261006.json)核对10个Git文件、原summary/cleanup字节、30源与输入及7operand前后身份。无新采样、H、PCG、原软件或GPU调用；生产未替换。

[单层64MiB卷积分块候选](seg_cpu_conv_slab_trial_20261006/README.md)已在目标Torch2.5.1的第三个短合同停止：实际权重、depth7输入的37,128个FP32值中33,144个不同，maxabs4.292e−6。未运行真实MRI层或ABBA，不形成速度或官方精度结论；14个生产文件和GPU保持原样。记录与Git源码已[独立复核](seg_cpu_conv_slab_trial_20261006/ROOT_REVIEW.json)，下一步只研究保留原GEMM矩阵布局的列缓冲复用。

[列缓冲复用接口](seg_columns_reuse_20261006/README.md)的编译后，[v2短合同](seg_columns_reuse_v2_20261006/README.md)通过6组完整FP32、13组复制及23组fallback门。真实层首次测试因6D权重哈希被误限为5D而在任何卷积前停止，[原失败](seg_columns_real_layer_20261006/README.md)保留。修正哈希守卫后，[真实MRI层ABBA](seg_columns_real_layer_v2_20261006/README.md)四臂完成：三次完整264,241,152个FP32值逐位相同，单层操作中位数15.385730→5.720358秒，本组快2.690倍，RSS最大11.397GB；14个slab的M/K/N、偏置及同一MKL provider保持。原始记录、源和时钟已[独立核验](seg_columns_real_layer_v2_20261006/ROOT_REVIEW.json)。该结果覆盖已保存输入的一层。随后[完整CPU/GPU接入对照](seg_columns_integration_20261006/README.md)完成真实T1的CPU四臂和GPU两臂：冷worker中位数107.937608→90.118916秒，缩短16.508%；API中位数105.144285→87.393578秒。两次CNN前向、分割、全部header、体积表及压缩文件SHA保持旧输出；GPU allocated/reserved为10.712/14.615GB，进程采样最大15.177GB，均与旧版相同。实际已有Conda缓存冷编译1次、暖编译0次，39个守卫/缓存合同及4个调度合同通过。对官方仍仅原1个体素差异，前景最低Dice0.999998569，软体积CSV最大差0.8mm³；对应官方同例CPU8冷CLI55.046秒，仍未达速度目标。缓存只在受测CPU层/权重/运行库下启用；共享GPU时钟不作提速结论，完整新Conda安装仍待测试。

下一层[24通道列缓冲候选](seg_columns_c24_contracts_20261006/README.md)已完成现有Conda编译和短输入检查：实际`down[0].conv1`权重的六组FP32输出、13组复制均逐位相同，30个回退守卫通过；原32层slab、MKL矩阵布局和偏置累计保持。短数值worker峰值RSS0.468GB；[root复核](seg_columns_c24_contracts_root_review_20261006.json)实际重核28项source和12项冻结文件。该短合同没有MRI、完整CNN或GPU调用。随后[原图与翻转图真实单层ABBA](seg_columns_c24_real_results_20261006/README.md)已正常完成，六次完整FP32比较逐位相同；两个pass操作中位数10.889→5.003秒，快2.177倍，RSS最大9.769GB。仅运行一次preprocess、两次conv0和八次conv1，完整CNN/原软件/GPU调用均为0；[独立复核](seg_columns_c24_real_root_review_20261006.json)保存原始回执SHA和计数。此结果尚未接入生产，仍缺完整CPU输出及GPU保护。

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
| [GEMS CPU 数据目标与有限轨迹](gems_cpu_objective_20261005/README.md) | 平滑阶段的零质量 alpha 导致额外通道归一化改变 mesh 数据项梯度；验证闭包保留 raw prior mass | 真实固定阶段 initial/1/3/37 同点门通过，完整梯度相对差约 2–3e−8。验证闭包还依赖 CPU epsilon、FP64 reference/插值、FP32 owner 与私有 L-BFGS，未接入当前生产。37 步对官方轨迹最大坐标差 0.508 voxel、RMSE 0.008893 voxel、目标值高 242.54，仍未收敛；两者无非正 Jacobian。前 3 步和续 34 步累计 136.523 s，另 2.340 s 为恢复评分，不是 fresh 完整函数 benchmark。继续定位 optimizer，未测试新的完整 recipe 或逐核团 Dice/体积 |
| [FastVBM/FNIRT 定位](gems_fixes_20261004/FNIRT_READONLY.md) | 大量误差进入非线性估计段；最终乘法不是额外误差来源 | 固定官方场的实际转换/重采样 RMSE 4.39e−6，旧完整链 0.0189811。解析/稠密 Jacobian 定义差异另列 |
| [FastVBM CPU Jacobian 修复](fnirt_jacobian_20261004/README.md) | CPU FNIRT 采用已有解析 Jacobian；GPU/SynthMorph 保留原路径 | 同官方系数对 `jout` RMSE 7.37e−8；固定真实 GM stage 的 Jacobian RMSE 0.01404→0.01259。调制图 RMSE 略降、最大误差略增；完整 GPU 后处理三图/header/QC 相同，未重测整 pipeline |
| [FNIRT 首轮实测](fnirt_first_diff_20261004/README.md) | 首次分歧为正方向 moving GM 的平滑累计顺序；翻转控制整图逐值相同 | 初始 mask、fixed 平滑相同；首轮梯度/accepted coefficient 差很小。共享真实 FP64 系数 `4→2` 转换最大差 1.35e−12 mm；[完整平滑候选](fnirt_cpu_orientation_20261004/README.md)最终退化，未接入 |
| [FNIRT 非零状态](fnirt_nonzero_state_20261004/README.md) | 同官方 accepted 参数的 bending、梯度、Hessian 与正则权重状态对照 | regularizer action 相对差约 3.2e−15，未发现量级错误；baseline/flip 完整对照最早在 PCG 80/49 轮分叉，非线性估计仍未等价 |
| [SynthSegPlus CUDA 精度作用域](seg_tf32_20261005/README.md) | 默认 CPU/GPU 完整三图、CSV、header 与上一版严格相同；构造及异常退出恢复调用方策略，新增 Python True/False/None 选项 | 同病例 False 普通分割对官方 0 voxel 差，fast 2 voxel 差；CSV 尾差 0.20/0.346 mm³。默认 Torch 显存相同；共享 GPU 时间仅作观察 |
| [GEMS CPU Double 续段](gems_cpu_double_continuation_20261006/README.md) | actual Double point、QR、Gaussian 与保存状态恢复；37步同点 objective/gradient/prior/coverage 原门通过 | 不同末点最大差 0.660 voxel、RMSE 0.009264，cost 高35.49；与前一控制相比 cost 差降低但坐标差增大。仅诊断，未接入默认生产或完成新的 ROI 验收 |
| [FNIRT 恢复状态的矩阵作用](../fnirt_cpu_matrixfree_restore_20261006/README.md) | 68数组/臂的原stride与逻辑值恢复，3unit列和7个优化CPU/原Torch action逐位通过 | 14callback/7CSC，0solver/native/new assembly；真实方向对CSC最大差3.55e−15。3.515秒含冷JIT，不是速度对照；未接生产，完整非线性差异仍待解决。[独立复核](../fnirt_cpu_matrixfree_restore_20261006/root_review.public.json) |
| [GEMS CPU 完整右侧 HA 未过门](gems_native_cpu_rha_failure_20261006/README.md) | 未采纳的显式 CPU 候选四真实点/EM/GPU 保护门后，完整 recipe 只执行一次，exit0 | API3110.729秒/RSS11.838GB；native9/28、HR6/28通过原Dice≥0.95/硬体积差≤5%门。逐区CSV、固定网格、五阶段耗时和真实脑图已[独立核验](gems_native_cpu_rha_failure_20261006/ROOT_REVIEW.json)。初始对齐等定义差异已定位，未接入生产；官方同节点新墙钟缺失 |
| [FNIRT 新共享探针](../fnirt_shared_followup_20261006/README.md) | 固定真实系统的严格CSC/除法/独立官方reductions组合自然69轮，全部向量/scalar与1177解bits相同 | 仍是隔离诊断，不是自有生产修复。当前第二接受点H rel1.03e−9、RHS rel8.36e−7；只换Jte后7.37e−7，完整非线性估计尚未等价 |
| [FNIRT 共享 PCG](../fnirt_pcg_shared_state_20261004/README.md) | 同 H/RHS/初值/对角/容差重放实际官方与 FNIT solver | 第二次均 24 轮；第三次官方 69、dense 80、严格列序 49、除法控制 79。所有停止门正确；实际 FP64 尾差可放大为不同参数，未改生产求解器或重跑整链 |
| [SynthSR](synth_fixes_20261004/sr_cpu_followup/README.md) | CPU 推理使用已核验的 FP32 ELU/BN 顺序及临时 channels-last 权重；保留训练、hooks、autocast 和 CUDA 路径 | 默认两分支 22,020,096 CNN 值、9,072,000 浮点/量化值全同官方，3,522 个旧超限值降为 0。四参数分支和六真实域原门均通过；正常 CPU CLI 中位数 28.660→28.340 s，RSS 10.222→7.496 GB。GPU 新旧完整输出/header/文件 SHA 与 allocated/reserved 相同；共享时间另列 |

这版没有宣称全部 sMRI 功能已等价：丘脑/HA、FNIRT 非线性估计和 recon-all 的有效顶点对应仍见[剩余清单](REMAINING.md)。SynthSR 的 EPI NPZ 浮点仍有微差但通过原门；GPU 与旧 GPU 一致，未因此宣称 GPU 同官方 CPU 逐值相同。以下保留前一轮源码绑定结果，不改标为本版新测试。

SynthSR 修复并普通合并最新 main 后，本地组件回归为 **2,092 passed、5 skipped，138.79 秒**。覆盖前述组件及最新 FLIRT/FNIRT、ApplyWarp、ConvertWarp、InvWarp 和空间转换接口；测试和实际源码 SHA 见[最新整合记录](integration_sr_final_20261004.json)。首次整合的两项 NMI 测试固定统计 scatter 调用次数，未覆盖 CPU Numba 后端；已改为逐值检查送入熵计算的完整直方图，数值门未改变，生产数学也未改。两项 warning 来自既有 FastVBM profiler 测试未设 warmup，不能使用测试时间作 benchmark。

该 2,092 项记录早于本次 joint 原始插值与正式 CPP 平均器合并，只覆盖其注明的源码。中间版 `55d7a88f` 的 **2,148 passed、5 skipped，158.32 秒**见[当时整合记录](integration_20261005.json)，当时尚未包括安全 Eigen loader 和精确 NumBa 采样。

2026-10-05 整合版 `c2ceb18e` 的组件回归为 **2,218 passed、5 skipped，158.37 秒**；包含上述两项已验收 CPU 修改、正式 CPP 平均器合同与最新 main 文档合并，实际文件哈希及测试期间不变性见[该版整合记录](integration_final_20261005.json)。该版尚未接入 SynthSeg CPU 拼接优化。

随后 `7e0890a5` 接入保留原数值的 SynthSeg CPU 拼接，整合回归为 **2,288 passed、11 skipped、3 subtests passed，155.04 秒**；包含完整 `tests/synthseg_parc`，外层墙钟 158.44 秒，源码与测试文件在执行期间均未变化，见[接入后记录](integration_seg_20261005.json)。没有采纳 GEMS mixed 或新的 raw-prior 优化目标。两个 warning 仍来自既有 profiler 测试。前一轮 **931 passed、3 skipped，69.70 秒**及其源码仍见[原整合记录](integration_20261004.json)。组件回归不代替真实影像与原软件对照，也没有重新运行原始 T1 的完整 recon-all。

2026-10-06 接入 SynthSegPlus 精度作用域的 `76bb9546` 整合回归为 **2,386 passed、11 skipped、3 subtests passed，165.89 秒**，外层墙钟169.99秒。绑定源码和测试在执行期间保持不变，详见[本次记录](integration_seg_tf32_20261006.json)；真实原始T1的CPU/GPU门见独立精度报告。组件测试不代替MRI速度benchmark。

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
