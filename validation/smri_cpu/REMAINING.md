# 本轮尚未通过或缺少完整实测的项目

此清单对应 2026-10-04 同节点 CPU 工作，详细指标和源码身份见[总报告](README.md)。已完成对照与尚未达到目标分开记录；不以旧 GPU 结果填补本轮 CPU 测量。

| 工作包 | 当前尚未通过 | 尚缺的真实完整覆盖 |
|---|---|---|
| SynthStrip / SynthSR | SR 默认 CPU 完整 CNN、浮点和量化已与官方相同；四参数和六真实域原门通过。EPI NPZ 浮点保留微差，max 0.000339508、RMSE 1.76756e−5；GPU 旧新相同但与官方 CPU 的既有 TF32 差异仍在。Strip GPU TF32 对官方 CPU 仍有 59 个 mask 差异 | 其他模型、参数、域/格式的完全物化对象入口；默认对象补测完成，最新 CPU 和 GPU 精度/正常 CLI/RSS/显存见[SynthSR 完整报告](synth_fixes_20261004/sr_cpu_followup/README.md) |
| SynthSeg / Plus / WMH | 普通 Seg 尚无官方 CPU 速度优势。最新 CPU 单缓冲拼接已接入，16 次完整 FNIT 旧新对照标签/CSV/几何相同；八核 nodecw7 官方正常冷 CLI 33/parc/fast 为 55.046/48.296/33.784 s，候选冷 worker 为 112.952/55.684/40.464–43.369 s，时钟与输出范围分别列明。保留官方 1/5/1 个标签差异、无新增。Plus CPU 崩溃已修复；2026-10-06 新增显式 cuDNN TF32 策略并修复 lazy 构造覆盖调用方状态，默认 CPU/GPU 完整三图、CSV 与旧版严格相同。WMH crop 的完整 GPU 20 GB 与旧输出门已通过，no-crop 预算与模式隔离结果见[最终报告](synth_fixes_20261004/README.md) | 继续定位卷积计算顺序与低显存实现；Plus 最新精度作用域与默认完整对照见[报告](seg_tf32_20261005/README.md)；共享 GPU 配对观察不作为稳定速度倍数。非默认 `min_pad`；CPU keep-geometry/色表独立计时；WMH no-crop 同精度低显存路线及跨输入低显存覆盖。最新[完整对照](seg_memory_20261005/README.md)支持 Torch 显存相同，不等于物理进程树峰值或独占 GPU 速度验收 |
| SynthMorph | 四模式默认 CPU 门通过；joint 192/256 的完整场、脑内及零边界门通过。精确 NumBa 采样及安全 Eigen loader 已完成真实源码绑定：六次完整 CPU 两图两场/完整元数据与验收版逐位相同，物化 API 输入未改。256 八核 ABBA 中位数 163.094→155.086 s，本组缩短 4.91%；暖采样快 5.35/6.07 倍，局部冷 256 慢 12.3%。14 项 GPU 隔离门通过，既有完整 H100 输出相同、reserved 17.836 GB，未重跑 GPU CNN；官方 init+mid-space 自身异常另列 | 其他对象模式/参数；当前 world 边界全 490 帧对照和独占资源的重复完整速度评估。真实两帧 DWI 和 14 项同场 apply 已测；[当前源码与完整回归](../synthmorph/cpu_fixes_20261004/README.md)区分最新实测和沿用父版证据 |
| TorchFAST / FastVBM | FAST CPU `fsl` 默认及已测非默认八图与官方逐值一致，原少数 PVE 差已消除。默认 `tensor` 仍非原序算法。FNIRT VBM 非线性估计尚未等价，Morph 完整 VBM 13 图也非逐位相同 | 更新 FAST/Morph 修复后的完整 VBM 两链、官方 Morph 冷完整链、pipeline AB-BA、其他 FAST 组合和完整 VBM GPU。FNIRT[实证定位](gems_fixes_20261004/FNIRT_READONLY.md)排除了重采样和最终乘法作为主因；首轮 accepted、共享非零参数和两次共享 PCG 轨迹已对照；浮点敏感性已实证，原完整后续状态仍缺 |
| 亚区 / recon-all | 丘脑与双侧 HA 逐区门未全过，CPU 仍慢。更接近官方的裁剪预处理候选存在逐区退步，未接入；缓存无 CPU 收益，已撤回。Gaussian 广播 bug 已修复，但四 recipe 不进入该分支。完整 recon v3 仍慢 4.99%，皮层分区和拓扑不同；44 顶点图、12 注释记 NA | 原始 T1 的 CPU `balanced` 全亚区；有效顶点对应、DKT/a2009s/BA 具名逐区摘要、三图谱同公式 no-TH3 体积；其他半球策略、批量入口和当前整合版完整 CPU/GPU 回归。同完整左半球输入的 sphere.reg 已与官方逐点一致、保存负面积面为 0；348.038 s 对官方 261.199 s 仍慢 33.25%，未新跑上游或整链 |

## 后续定位顺序

1. SR 的 CPU 原固定门已通过，正常 CLI 与旧版基本持平、RSS 降低约 26.7%；GPU 完整旧新输出及显存相同。继续扩展对象模式时保留原数值门，EPI NPZ 的非逐值差异和 GPU 对官方 CPU 的 TF32 差异独立报告。
2. GEMS 已在[实际首个同状态 mesh evaluation](gems_first_state_20261004/README.md)定位到原 likelihood 求和后缺少 `1e-15`，并区分 FP32 几何误差。[混合内部精度候选的全部 recipe](gems_cpu_epsilon_20261004/README.md)已完成：native 37/105→43/105、HR 41/107→45/107，但丘脑丢失三个旧通过区、HA 仍有多数区域未过，因此不替换默认。完整后验、网格、Gaussian、objective/solver、RSS、时钟及 220 行 hard/soft 指标保存不变。[真实同点与首步定位](gems_first_trial_20261005/README.md)已核对安装版实际 options、扩展和 recipe 源码：三处完整梯度相对差约 5e−8；旧丘脑/左 HA 的百分比梯度差来自参考适配器漏掉负行列式下的 tet 换序，不是 FNIT prior 符号 bug。右 HA 首次最大移动是旧 Armijo 0.5 voxel、安装版 1 voxel，说明搜索定义仍不同。带历史的有限 L-BFGS / Double 控制已完成，最新同点和末点差异见下节；新的显式 CPU recipe 已完成[唯一右侧完整输出](gems_native_cpu_rha_failure_20261006/README.md)，native 9/28、HR 6/28 过门，未采用；安装版的初始 robust 对齐、mask、Gaussian 支持和后处理与 FNIT 存在明确定义差异，按保存状态和单项输入继续定位。仅 epsilon 的 FP32 候选还没有完整 recipe 结果。CUDA 保留旧路径，尚缺 epsilon。
3. FNIRT 已用固定 GM、官方 FLIRT、模板/mask 保存首层初始化、目标、梯度、Hessian 和 accepted coefficients，并完成一次共享系数网格转换；[首差报告](fnirt_first_diff_20261004/README.md)定位到 CPU 平滑方向。[完整平滑候选](fnirt_cpu_orientation_20261004/README.md)最终误差扩大、耗时增加，未采纳；[非零参数探针](fnirt_nonzero_state_20261004/README.md)未发现 regularizer 或 Gram 的量级错误。[共享 Hessian/RHS 逐轮重放](../fnirt_pcg_shared_state_20261004/README.md)已完成两个有限续跑系统：69/80/49/79 轮分叉可由 FP64 顺序尾差触发，全部符合原停止门。原 stock 后续内部缓存状态仍缺，尚不能把它当作最终非线性误差的唯一原因。CPU 解析 Jacobian 定义已[修复并验证](fnirt_jacobian_20261004/README.md)，同系数 RMSE 7.37e−8；非线性估计仍不等价。先测受影响阶段，再重测完整 VBM。
4. WMH 已验收 crop 的 20 GB 工作区生命周期；no-crop 保持原 GPU 数值，进一步低显存优化仍须通过完整输出及实际 cuDNN 算法回归。
5. CPU recon 的双侧球面配准占旧整例墙钟约 36.4%。平均算子已保序并行，[完整左半球同输入配准](recon_fixes_20261004/FULL_REGISTRATION.md)与官方逐点相同，旧新接受轨迹也相同。[正式 C++ 持续线程组 ABBA](average_cpu_persistent_20261004/README.md)已完成：冻结 1,239 文件、任务/运行库身份、同节点八核资源和全部轨迹/几何门通过，平均步骤快 2.085 倍；完整新进程仍慢 5.27%，性能目标未通过。双侧整链、其他输入和分区边界仍需核对。拓扑不同的逐顶点厚度/面积/体积/曲率仍为 NA；先建立有效对应。

## 原软件存在、当前接口未实现的模式

SynthSegPlus 目前是普通 SynthSeg 2.0 `--parc`，未实现 robust SynthSeg+、官方 QC、目录批处理、CT、Photo 或 v1；`--parc --color-lut` 组合也未实现。TorchFAST 目前为三组织 T1，不含 T2/PD、多通道、其他组织类数及部分原版先验模式。这些属于功能范围，不能记作 benchmark 通过。

## GEMS CPU：保留拟合轨迹的候选优化

前一阶段[数据目标与有限轨迹诊断](gems_cpu_objective_20261005/README.md)证实，平滑阶段的零质量 alpha 使额外通道归一化改变 CPU mesh 数据项梯度。保留 raw mass 后，真实阶段 initial/1/3/37 同点评分通过，但还依赖冻结 CPU epsilon、FP64 reference/插值、独立 FP32 ownership 与私有 L-BFGS；当前生产未接入这套前提，不能单改归一化就宣布通过。该 FP32 控制第37步对官方轨迹最大差 0.508 voxel、RMSE 0.008893 voxel、目标值高242.54，未收敛；随后 Double 控制与保存状态审计见下节，保留这份前版结果供对照。新完整 recipe、逐区 Dice/体积与速度仍未验收，CUDA 保留当前实现。

[已保存优化状态的只读审计](gems_optimizer_state_20261005/README.md)按官方 Double 梯度重建方向，并从点增量推断步长，37 步增量最大残差为 1.85e−13 voxel；这不是原生内部 trace。首步官方 Double 点舍入后虽与 Python FP32 点相同，原几何上的梯度已差 0.1233%，到第四步出现明显步长分叉。随后前三步、第4步及续至37步的 CPU Double 点、投影和历史控制已完成，最新结果列在下节；不改变 GPU，不以同点门代替完整分割验收。

本轮[完整脑干同输入 stage](task5/README.md)中，强度网格拟合为 689.308 秒，约占该 recipe 的 91%。owner、EM 和 Gaussian 的细项优先级来自冻结 v2 的五次 core evaluation 限额诊断，主动停止退出码为 75；它们不是最新联合候选或 raw v5 的完整热点占比，也不与 recipe/API 重复累加。[联合候选](task5/cpu_compact_20261004/README.md)已有完整后验与拟合状态恒等核验，API 从 733.021 降至 668.801 秒。

固定 EM 数据缓存已完成 nodecw7 实测，完整状态相同但墙钟/RSS均无收益，因此未采纳。其余候选仍待验证：

- 四面体 owner IDs 和拓扑逐值相同时，复用自有 IDs、稳定排序及分段布局；保留原点到四面体、四面体到顶点的累计顺序和已验收 FP64 梯度归约。
- alpha 无梯度且形状、stride、版本及 owner 不变时，缓存 alpha 的 gather；动态几何、原 FP32 插值、边界判定、归一化和背景规则保持。GPU 路径不进入这些 CPU 缓存。

每项替换仍须通过完整后验、网格、Gaussian、objective 和 solver 接受/停止状态的核验；局部热循环时间不能代替完整函数耗时。

## 2026-10-06：有限诊断与候选接入

- **普通33类平滑候选**：一份真实末端概率的[ABBA回放](seg_cpu_blur_trial_20261006/README.md)三次实际逐位比较全部通过；局部最大RSS10.295→5.592GB，降低45.68%，操作中位数6.963→6.978秒，无速度收益。生产未接入、未补跑完整T1。随后[64MiB高分辨率卷积候选](seg_cpu_conv_slab_trial_20261006/README.md)在目标Torch2.5.1的depth7短合同出现maxabs4.292e−6，按原逐位门停止；真实MRI层和ABBA均未执行。[列缓冲复用接口](seg_columns_reuse_20261006/README.md)的编译及[v2短合同](seg_columns_reuse_v2_20261006/README.md)已通过；subclass/forward-AD和fallback守卫已核对。[真实层v1](seg_columns_real_layer_20261006/README.md)因哈希维度守卫错误在0卷积时停止，未作为数值失败。随后[新v2真实MRI层ABBA](seg_columns_real_layer_v2_20261006/README.md)的三次完整FP32比较逐位相同，单层中位数15.386→5.720秒，快2.690倍，RSS≤11.397GB。原slab14/last10、M/K/N、偏置和LP64 provider保持。[完整CPU33/GPU保护](seg_columns_integration_20261006/README.md)已完成六臂：冷worker107.938→90.119秒，完整分割、header、CSV及压缩SHA无新增差异；GPU输出、实际设备/精度和显存相同。已有Conda真实冷编译通过，受测CPU资格下接入；对官方速度目标仍未通过，freshConda安装尚未验证。下一处[C24层短合同](seg_columns_c24_contracts_20261006/README.md)已完成编译、六组实际权重bit0及13复制/30回退门；它只使用短合成输入，不作benchmark。随后[真实MRI原图与翻转图单层ABBA](seg_columns_c24_real_results_20261006/README.md)已完成：六次完整FP32逐位比较差异0，两个pass的操作中位数10.889→5.003秒，快2.177倍；最大RSS9.769GB、输出shape/stride和精度状态保持。只采集两次conv0并测试八次conv1，没有完整CNN、原软件或GPU运行；尚未接入生产，完整CPU输出与GPU保护仍待验证。
- **普通33类 SynthSeg CPU**：一次当前源码的[完整分步观察](seg_cpu_profile_20261006/README.md)通过输出逐值、header、CSV、压缩 SHA、两次前向、8次原 CPU join 和精度恢复门。观察API106.377秒中，两次CNN81.716秒、两次blur13.136秒；显式slab复制仅0.927秒。优先处理高分辨率卷积和group33平滑；单纯减少Python循环不足以解决差距。未采用改变标签的oneDNN候选，正式速度门仍未通过。
- **SynthSegPlus**：真实原始 T1 的 CPU 4 臂、GPU 8 臂完成；默认两模式旧新三图、完整 header、体积 CSV 及压缩文件 SHA 相同，Torch allocated/reserved 峰值相同。新 `cudnn_tf32=False` 普通模式对该病例官方分割差 0 voxel，fast 差 2 voxel；CSV 最大差分别 0.20 / 0.346 mm³。`None` 继承调用方 cuDNN=False 的输出与 False 相同；不推广为全部输入逐位相同，也不把共享 GPU 时钟作为加速验收。[精度报告](seg_tf32_20261005/README.md)。
- **GEMS CPU Double**：真实 T1 派生标签阶段的第37个候选点同点评分通过；不同末点的 cost 差从前一 FP32 控制的 +242.54 降至 +35.49，最大坐标差却从 0.508 增至 0.660 voxel。[Double 续段与计划](gems_cpu_double_continuation_20261006/README.md)。显式CPU候选已在独立工作树实现，固定真实四点评分、强度EM及默认GPU短程保护已过；新增materialize/mask合同发现的两处CPU Double输出dtype冲突已修复，51项合同通过。修复版[唯一完整右侧HA recipe](gems_native_cpu_rha_failure_20261006/README.md)已 exit0：API3110.729秒、峰值RSS11.838GB；native9/28、HR6/28同时通过每区Dice≥0.95、硬体积差≤5%，完整精度未过门。固定网格及逐区CSV已[独立核验](gems_native_cpu_rha_failure_20261006/ROOT_REVIEW.json)；没有对应官方新墙钟，默认生产未采用。[独立robust前处理](../../docs/robust_register/PREPARATION.md)已通过同例目标掩膜、右侧atlas数据及MGH几何门，原报告JSON bug已修复并保留失败记录，没有重算图像。[独立刚性/仿射候选](../robust_register/rigid_affine_20261006/README.md)已完成一次官方两命令/自身两阶段对照，17/20门通过；各阶段支持集差1体素，仿射warp相对L2为2.588e−5超原门，候选未采用。133点最大位移差分别7.742e−6/4.487e−4mm；一致的掩膜Dice计数不等于逐体素标签相同。随后[固定Float求逆顺序合同](../robust_register/inverse_order_20261006/README.md)18/18通过，仅6个已保存4×4矩阵、0MRI/注册；旧刚体word差为signed zero，仿射才有非零数值差。随后[真实候选](../robust_register/inverse_real_20261006/README.md)沿用原评分尺仍17/20通过，仿射warp相对L2为2.556643e−5，旧候选降低约1.22%但未过1e−5；两支持集仍各差1体素。仿射输出已保存而JSON报告失败，内部API时钟/flags为NA；仅追加保存输出评分，没有重跑注册。矩阵合同不能当作核团或配准验收。下一步定位求解/保存几何的首差，再接入GEMS核对最终核团及stage mask、超参数、后处理。本例1mm未触发前处理resize，GPU新接口尚未实测。

- **FNIRT CPU**：前序[共享系统探针](../fnirt_shared_followup_20261006/README.md)使用隔离官方算术上下文；[自有CPU归约](../fnirt_cpu_reductions_20261006/README.md)在同一保存原生系统通过465项标量、186项相对范数及自然69轮的全部向量／最终解逐位门。新增[当前已组装系统两臂及保存解诊断](../fnirt_cpu_current_replay_20261006/README.md)完成，旧/新自然53/49轮，均符合原1e−3停止门；对另一存档原生系统解的距离11.53%/14.67%为跨系统比较，不支持默认替换。509项绑定和10项追加输入前后相同，未重复MRI、组装或官方运行。随后[一次当前solve3状态重建](../fnirt_cpu_level_rehydrate_20261006/README.md)通过全部scalar和1177项gradient/独立diagonal逐位门，已保存私有checkpoint；没有调用callback或求解。当前对官方RHS约7.4e−7、任意方向matrix-free乘积与完整非线性估计仍未解决；随后[checkpoint布局与同状态callback](../fnirt_cpu_matrixfree_restore_20261006/README.md)已通过：三个unit列与存档H逐位同，七个方向的优化CPU/原Torch动作逐位同；对CSC仅有maxabs≤3.55e−15舍入差。0求解/新组装/native，生产未接。随后[真实matrix-free PCG两臂](../fnirt_cpu_matrixfree_pcg_20261006/README.md)已自然49/53轮停止，实际相对残差均小于原1e−3，更新向量相对L2仍差6.9665%，未接入默认。[首方向诊断](../fnirt_cpu_initial_precondition_20261006/README.md)已分离局部原因：内部下限改值0，倒数再乘/直接除法产生332个1 ULP差异，另有同向量点积/范数尾差；这些还不能解释完整更新或非线性误差。随后[唯一平滑moving单输入控制](../fnirt_cpu_rhs_moving_control_20261006/README.md)12门通过：固定参数/λ的总g误差8.364e−7→2.359e−7，FSL顺序7.366e−7→2.484e−7；三coef分量仅各改善21%–30%，仍不逐位同。没有H/diag/PCG/native/full，官方图不进入生产。随后[当前编译CPU平滑桥接](../fnirt_cpu_smoothing_bridge_20261006/README.md)已对两个完整18,579,456值数组逐位匹配，3exit0，仅2次blur/1次adapter，没有完整配准。旧完整orientation候选最终误差扩大仍不采用。随后[仅导数投影除法控制](../fnirt_cpu_rhs_projection_control_20261006/README.md)已完成：17前门通过、原b17ee梯度逐位复现，12267个FP32导数值bits改变后，总g误差2.4836018e−7→2.4838650e−7未改善；scale绝对误差2.54688e−6完全保留。v1头信息守卫混比的失败记录保留，v2仅修读取表示；不运行solver/full，不采用候选。下一步优先核对逐项组装和scale累计顺序，再看求解轨迹及最终非线性输出。

## 保存几何支持集的后续诊断

[一次支持集诊断](../robust_register/support_points_results_20261006/README.md)已完成：4次原sampler全网格warp先精确复现历史指标，再4次单点逐bit验证。刚体和仿射各1个差异点都观察到插值角点或非零角点状态变化；round域、clamp、rint和FEQUAL捷径一致。原17/20门仍失败，未修改生产，也未定位优化器最早分叉。下一步核对求解与保存几何的首个差异，不能以这两点观察替代最终核团Dice/体积验收。

## FNIRT scale累计假设的后续结果

[一次保存状态控制](../fnirt_cpu_scale_accumulation_control_20261006/README.md)已排除该状态下的累计与最终factor分组作为主要差异：scale只变3.19744e−14，当前存档native差约7.66992e−6，同一串行和的最终两组FP64 bits完全相同。没有重建此前官方moving替换的输入，不把该结论扩展到其2.54688e−6残差。后续需绑定原oracle身份并核对实际Robj/ScaledRef/Mask中间缓存及系数梯度，完整非线性精度仍未过门。

## 初始刚体配准的首差观察计划

[只读源码核对](../robust_register/earliest_branch_readonly_20261006/README.md)确认官方活动 robust 路径使用 LINPACK QR，候选使用 Torch reduced QR；停止语义相同，尚未捕获同状态求解轨迹。准备图像/M0观察器已独立构建并完成[真实同例对照](../robust_register/prepared_m0_capture_20261006/README.md)：准备图像/几何及Rsrc/Rtrg逐位同，质心/M0有最大4.12e−13的Double尾差。下一步核质心归约及首个A/b、LINPACK QR解，再验证最终影像；正式17/20及原阈值不变。

## 最新普通33类CPU优化

[C24完整接入](seg_columns_c24_integration_20261006/README.md)在已接受C72基础上，CPU8 API中位85.499→80.621秒（5.71%改善）；完整CPU/GPU输出、几何、CSV及GPU精度/内存保持。冷进程首臂检查差异单列，未声称15.29%都是计算收益。仍需达到同线程官方CPU速度、更多真实shape/线程与输入覆盖，以及完整wheel/全新Conda安装；不扩大受测资格。

## FNIRT 冷缓存的实际差异

[固定参数点的冷初始化对照](../fnirt_native_cache_prefix_result_20261006/README.md)已通过实际运行库、输入/源码身份与资源回收门；这与数值等价分列。掩膜逐体素同，重采样图有2425/16128个FP32 word不同，maxabs9.9182e−5、relative L2 2.1212e−7。Deriv会改写原生重采样缓存，本轮仍未恢复历史solve3调用上下文；尚不能把这些尾差作为最终配准差异的唯一原因。下一步先核对参考图、scale参数与ScaledRef，再比较同输入采样，最后验证梯度和完整非线性输出。生产/GPU未修改，完整官方CPU速度目标仍未通过。

## Robust prepared/M0 的 SDK 现状

[现场 SDK 审查](../robust_register/prepared_m0_sdk_review_20261006/README.md)已找到固定源码、Conda 编译器、原 libutils 与11个静态依赖；20个原源码/头文件/许可证身份相同。但原 robust 对象实存与 Ninja 声明均为0。随后[局部Conda构建](../robust_register/prepared_m0_sdk_build_20261006/README.md)完成9个对象和M0前缀参考程序；尚不支持完整Schur，guard入口即停止，原链接失败记录保留。实际原生准备态/M0已exit0，23个加载ELF身份通过。outside字段宽度与懒加载模块名两处验证入口错误已修正，原数学不改；[双臂比较](../robust_register/prepared_m0_capture_20261006/README.md)完成，准备图像/几何逐位同，M0最大尾差4.12e−13，17/20原门保持。这是独立源码参考，不是安装版内部binary trace。

[随后参考图/ScaledRef控制](../fnirt_saved_scaled_reference_20261006/README.md)已通过：参考图字节及scale参数相同，16128个按原dtype乘法派生的ScaledRef逐位同。当前首差范围进一步缩小到重采样/导数缓存与后续组装；完整非线性输出仍未验收。该控制0native/solver/GPU，不作速度结论。

## 最新FNIRT采样修复候选

[同真实体积四API对照](../fnirt_cpu_sampler_case_20261006/README.md)已通过：候选plain值及partial值/三个导数的逐位差异均为0，当前helper的2338/813/0/913/1520个word差异在实际API中复现。129,024个角点、坐标/floor/fraction/derivedvalid均先过门。默认生产未替换。随后成熟checkpoint坐标复现旧SHA，但与实际原记录XX的16128个唯一位键交集0，采样/RHS均未执行；先从源定义核对公共/内部体素方向，再比较RHS，最后检验完整真实配准和GPU。历史moving身份差异与旧2425word差异仍分列，不能把局部通过记作端到端等价。
