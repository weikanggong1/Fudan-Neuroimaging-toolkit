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
2. GEMS 已在[实际首个同状态 mesh evaluation](gems_first_state_20261004/README.md)定位到原 likelihood 求和后缺少 `1e-15`，并区分 FP32 几何误差。[混合内部精度候选的全部 recipe](gems_cpu_epsilon_20261004/README.md)已完成：native 37/105→43/105、HR 41/107→45/107，但丘脑丢失三个旧通过区、HA 仍有多数区域未过，因此不替换默认。完整后验、网格、Gaussian、objective/solver、RSS、时钟及 220 行 hard/soft 指标保存不变。[真实同点与首步定位](gems_first_trial_20261005/README.md)已核对安装版实际 options、扩展和 recipe 源码：三处完整梯度相对差约 5e−8；旧丘脑/左 HA 的百分比梯度差来自参考适配器漏掉负行列式下的 tet 换序，不是 FNIT prior 符号 bug。右 HA 首次最大移动是旧 Armijo 0.5 voxel、安装版 1 voxel，说明搜索定义仍不同。带历史的有限 L-BFGS / Double 控制已完成，最新同点和末点差异见下节；新的 opt-in CPU recipe 候选正在实现，固定状态通过后再做单侧完整输出验收。仅 epsilon 的 FP32 候选还没有完整 recipe 结果。CUDA 保留旧路径，尚缺 epsilon。
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

- **普通33类平滑候选**：一份真实末端概率的[ABBA回放](seg_cpu_blur_trial_20261006/README.md)三次实际逐位比较全部通过；局部最大RSS10.295→5.592GB，降低45.68%，操作中位数6.963→6.978秒，无速度收益。生产未接入、未补跑完整T1。下一步只验证高分辨率卷积的更小slab候选，F32逐位门通过前不替换。
- **普通33类 SynthSeg CPU**：一次当前源码的[完整分步观察](seg_cpu_profile_20261006/README.md)通过输出逐值、header、CSV、压缩 SHA、两次前向、8次原 CPU join 和精度恢复门。观察API106.377秒中，两次CNN81.716秒、两次blur13.136秒；显式slab复制仅0.927秒。优先处理高分辨率卷积和group33平滑；单纯减少Python循环不足以解决差距。未采用改变标签的oneDNN候选，正式速度门仍未通过。
- **SynthSegPlus**：真实原始 T1 的 CPU 4 臂、GPU 8 臂完成；默认两模式旧新三图、完整 header、体积 CSV 及压缩文件 SHA 相同，Torch allocated/reserved 峰值相同。新 `cudnn_tf32=False` 普通模式对该病例官方分割差 0 voxel，fast 差 2 voxel；CSV 最大差分别 0.20 / 0.346 mm³。`None` 继承调用方 cuDNN=False 的输出与 False 相同；不推广为全部输入逐位相同，也不把共享 GPU 时钟作为加速验收。[精度报告](seg_tf32_20261005/README.md)。
- **GEMS CPU Double**：真实 T1 派生标签阶段的第37个候选点同点评分通过；不同末点的 cost 差从前一 FP32 控制的 +242.54 降至 +35.49，最大坐标差却从 0.508 增至 0.660 voxel，尚无新的完整 ROI 验收。保存状态审计显示第5步起线搜索 alpha 分叉，微小几何差后的梯度变化进入历史；目前不能确认唯一原因。[Double 续段与计划](gems_cpu_double_continuation_20261006/README.md)。下一阶段在独立工作树实现 opt-in CPU 候选，先固定真实状态接线与 GPU 保护，再只做一个完整右侧 HA recipe；最终每区 Dice≥0.95、硬体积差≤5%，不以内部37步坐标差单独判断最终核团失败。

- **FNIRT CPU**：前序[共享系统探针](../fnirt_shared_followup_20261006/README.md)使用隔离官方算术上下文；最新[自有CPU归约](../fnirt_cpu_reductions_20261006/README.md)已在同一保存系统通过465项标量、186项相对范数及自然69轮的全部向量／最终解逐位门，运行前后503项绑定通过。当前第二接受点RHS相对差仍约7.4e−7；生产矩阵自由回调接线和完整组装／配准尚未解决，未改变停止容差。下一步复用已存当前H/g/独立diagonal，只比较旧新求解算术，不重复组装或官方运行。
