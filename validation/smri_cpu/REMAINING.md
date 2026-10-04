# 本轮尚未通过或缺少完整实测的项目

此清单对应 2026-10-04 同节点 CPU 工作，详细指标和源码身份见[总报告](README.md)。已完成对照与尚未达到目标分开记录；不以旧 GPU 结果填补本轮 CPU 测量。

| 工作包 | 当前尚未通过 | 尚缺的真实完整覆盖 |
|---|---|---|
| SynthStrip / SynthSR | SR 默认 CPU 完整 CNN、浮点和量化已与官方相同；四参数和六真实域原门通过。EPI NPZ 浮点保留微差，max 0.000339508、RMSE 1.76756e−5；GPU 旧新相同但与官方 CPU 的既有 TF32 差异仍在。Strip GPU TF32 对官方 CPU 仍有 59 个 mask 差异 | 其他模型、参数、域/格式的完全物化对象入口；默认对象补测完成，最新 CPU 和 GPU 精度/正常 CLI/RSS/显存见[SynthSR 完整报告](synth_fixes_20261004/sr_cpu_followup/README.md) |
| SynthSeg / Plus / WMH | 普通 Seg 尚无稳定官方 CPU 速度优势，保留少数官方标签/体积差；Plus CPU 崩溃已修复。WMH crop 的完整 GPU 20 GB 与旧输出门已通过，no-crop 预算与模式隔离结果见[最终报告](synth_fixes_20261004/README.md) | 非默认 `min_pad`；CPU keep-geometry/色表独立计时；WMH no-crop 同精度低显存路线及跨输入低显存覆盖 |
| SynthMorph | 四模式默认 CPU 门已通过；joint 192/256 的完整场、脑内及零边界门均通过，完全物化 API 和路径相同。小网格卷积 primitive、归约与矩阵组合顺序已定位并修复；完整 GPU 新旧数组与几何相同，reserved 17.836 GB。当前两对相邻 CPU joint 仍约慢 5%，精确插值 NumBa 优化正在验证；官方 init+mid-space 分支自身异常另列 | 完整 CPU 精确插值候选的时间与源码绑定、其他对象模式/参数；当前 world 边界全 490 帧对照。真实两帧 DWI 和 14 项同场 apply 已测；[本版四模式与对象回归](../synthmorph/cpu_fixes_20261004/README.md)绑定实际源码 |
| TorchFAST / FastVBM | FAST CPU `fsl` 默认及已测非默认八图与官方逐值一致，原少数 PVE 差已消除。默认 `tensor` 仍非原序算法。FNIRT VBM 非线性估计尚未等价，Morph 完整 VBM 13 图也非逐位相同 | 更新 FAST/Morph 修复后的完整 VBM 两链、官方 Morph 冷完整链、pipeline AB-BA、其他 FAST 组合和完整 VBM GPU。FNIRT[实证定位](gems_fixes_20261004/FNIRT_READONLY.md)排除了重采样和最终乘法作为主因；首轮 accepted、共享非零参数和两次共享 PCG 轨迹已对照；浮点敏感性已实证，原完整后续状态仍缺 |
| 亚区 / recon-all | 丘脑与双侧 HA 逐区门未全过，CPU 仍慢。更接近官方的裁剪预处理候选存在逐区退步，未接入；缓存无 CPU 收益，已撤回。Gaussian 广播 bug 已修复，但四 recipe 不进入该分支。完整 recon v3 仍慢 4.99%，皮层分区和拓扑不同；44 顶点图、12 注释记 NA | 原始 T1 的 CPU `balanced` 全亚区；有效顶点对应、DKT/a2009s/BA 具名逐区摘要、三图谱同公式 no-TH3 体积；其他半球策略、批量入口和当前整合版完整 CPU/GPU 回归。同完整左半球输入的 sphere.reg 已与官方逐点一致、保存负面积面为 0；348.038 s 对官方 261.199 s 仍慢 33.25%，未新跑上游或整链 |

## 后续定位顺序

1. SR 的 CPU 原固定门已通过，正常 CLI 与旧版基本持平、RSS 降低约 26.7%；GPU 完整旧新输出及显存相同。继续扩展对象模式时保留原数值门，EPI NPZ 的非逐值差异和 GPU 对官方 CPU 的 TF32 差异独立报告。
2. GEMS 已在[实际首个同状态 mesh evaluation](gems_first_state_20261004/README.md)定位到原 likelihood 求和后缺少 `1e-15`，并区分 FP32 几何误差；当前 CPU-only epsilon 与混合内部精度候选仍须通过完整 recipe 的逐区分割、solver、后处理和时长门。现有标签命名空间及固定评价网格保留；工作 T1 更接近不能代替最终标签门。[候选、完整逐区退步与脑图](gems_fixes_20261004/README.md)均保留。
3. FNIRT 已用固定 GM、官方 FLIRT、模板/mask 保存首层初始化、目标、梯度、Hessian 和 accepted coefficients，并完成一次共享系数网格转换；[首差报告](fnirt_first_diff_20261004/README.md)定位到 CPU 平滑方向。[完整平滑候选](fnirt_cpu_orientation_20261004/README.md)最终误差扩大、耗时增加，未采纳；[非零参数探针](fnirt_nonzero_state_20261004/README.md)未发现 regularizer 或 Gram 的量级错误。[共享 Hessian/RHS 逐轮重放](../fnirt_pcg_shared_state_20261004/README.md)已完成两个有限续跑系统：69/80/49/79 轮分叉可由 FP64 顺序尾差触发，全部符合原停止门。原 stock 后续内部缓存状态仍缺，尚不能把它当作最终非线性误差的唯一原因。CPU 解析 Jacobian 定义已[修复并验证](fnirt_jacobian_20261004/README.md)，同系数 RMSE 7.37e−8；非线性估计仍不等价。先测受影响阶段，再重测完整 VBM。
4. WMH 已验收 crop 的 20 GB 工作区生命周期；no-crop 保持原 GPU 数值，进一步低显存优化仍须通过完整输出及实际 cuDNN 算法回归。
5. CPU recon 的双侧球面配准占旧整例墙钟约 36.4%。平均算子已保序并行，[完整左半球同输入配准](recon_fixes_20261004/FULL_REGISTRATION.md)与官方逐点相同，旧新接受轨迹也相同。双侧整链、其他输入和分区边界仍需核对。拓扑不同的逐顶点厚度/面积/体积/曲率仍为 NA；先建立有效对应。

## 原软件存在、当前接口未实现的模式

SynthSegPlus 目前是普通 SynthSeg 2.0 `--parc`，未实现 robust SynthSeg+、官方 QC、目录批处理、CT、Photo 或 v1；`--parc --color-lut` 组合也未实现。TorchFAST 目前为三组织 T1，不含 T2/PD、多通道、其他组织类数及部分原版先验模式。这些属于功能范围，不能记作 benchmark 通过。

## GEMS CPU：保留拟合轨迹的候选优化

本轮[完整脑干同输入 stage](task5/README.md)中，强度网格拟合为 689.308 秒，约占该 recipe 的 91%。owner、EM 和 Gaussian 的细项优先级来自冻结 v2 的五次 core evaluation 限额诊断，主动停止退出码为 75；它们不是最新联合候选或 raw v5 的完整热点占比，也不与 recipe/API 重复累加。[联合候选](task5/cpu_compact_20261004/README.md)已有完整后验与拟合状态恒等核验，API 从 733.021 降至 668.801 秒。

固定 EM 数据缓存已完成 nodecw7 实测，完整状态相同但墙钟/RSS均无收益，因此未采纳。其余候选仍待验证：

- 四面体 owner IDs 和拓扑逐值相同时，复用自有 IDs、稳定排序及分段布局；保留原点到四面体、四面体到顶点的累计顺序和已验收 FP64 梯度归约。
- alpha 无梯度且形状、stride、版本及 owner 不变时，缓存 alpha 的 gather；动态几何、原 FP32 插值、边界判定、归一化和背景规则保持。GPU 路径不进入这些 CPU 缓存。

每项替换仍须通过完整后验、网格、Gaussian、objective 和 solver 接受/停止状态的核验；局部热循环时间不能代替完整函数耗时。
