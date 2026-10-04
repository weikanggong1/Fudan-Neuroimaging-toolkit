# 本轮尚未通过或缺少完整实测的项目

此清单对应 2026-10-04 同节点 CPU 工作，详细指标和源码身份见[总报告](README.md)。已完成对照与尚未达到目标分开记录；不以旧 GPU 结果填补本轮 CPU 测量。

| 工作包 | 已测试、尚未通过 | 尚缺的完整真实输入覆盖 |
|---|---|---|
| SynthStrip / SynthSR | SR 默认浮点输出有 3,522 点超过原定容差；Strip GPU TF32 对官方 CPU 有 59 个 mask 差异。未采用改变 SR 量化输出的布局、BN 原型。[各一次默认真实物化对象 API](strip_sr_20261004/in_memory_20261004/README.md)已完成，与已有 FNIT 保存数组及文件 SHA 相同；复用官方参考的 Strip 数据门、SR 量化门通过，SR 浮点原门仍失败，Strip 官方完整 header/qform 微差另列 | 其他模型、参数、域/格式的完全物化对象入口；SR 本轮 GPU 完整对照。默认对象补测的完整进程含事后比较，不代替冷 CLI 速度对照 |
| SynthSeg / Plus / WMH | 普通 Seg 尚无稳定官方 CPU 速度优势；仍有少数官方标签/体积差。大图 Plus CPU 崩溃已修复，真实 CPU 与最新 GPU fast/非 fast 旧新回归通过。WMH GPU 在 20 GB allocator 预算内失败 | 非默认 `min_pad`；CPU keep-geometry/色表独立计时。WMH 四模式网络已测，新 header 对两个模式重新核验，其余计时仍绑定此前源码 |
| SynthMorph | affine 逆向上边界、joint 逆向场与上边界未全部过门；rigid 零动态范围边界有三点非零；官方 init+mid-space 分支自身异常另列。[一次 affine256 双向真实物化对象 API](../synthmorph/cpu_20261004/in_memory_affine_20261004/README.md)已完成，float64 先解码再转 float32 与路径直接 float32 解码产生首次输入微差，保存输出不逐值同 | rigid/deform/joint 及其他对象参数分支；直接 float32 物化路线的完整网络输出（仅无 CNN 前处理逐值同）。World 链越界周期采样；当前 CPU 边界实现的完整 490 帧对照。真实两帧 DWI 和 14 项同场 apply 已测 |
| TorchFAST / FastVBM | FAST `fsl` 默认参数仍有少量 PVE 差；接口默认 `tensor` 与官方差异更大。FNIRT VBM 尚未等价；Morph 三张标准图误差较小，但与官方 13 图没有逐位相同。最终 v4 两链完整补测已完成，输出与 v2 相同 | 官方 Morph 冷完整链；pipeline AB-BA、其他 FAST 组合、默认 tensor VBM、显式 brain/mask 和本轮完整 VBM GPU |
| 亚区 / recon-all | 丘脑与双侧海马/杏仁核逐区门未全过；CPU stage 仍慢于官方。脑干联合优化降低时间 8.76%，完整后验和拟合状态相同。recon v3 已完整执行并[完成评分](task5/recon_complete_cpu_v3/README.md)，墙钟慢 4.99%，皮层分区和几何仍有差异。44 顶点图、12 注释因顶点对应不成立记 NA；raw v4 CPU 卷积崩溃已修复，v5 新空目录运行中 | 原始 T1 的 CPU 全亚区整例、`balanced`；有效顶点对应、DKT/a2009s/BA 具名逐区摘要、三图谱同公式 no-TH3 体积；其他半球策略、批量入口及当前整合版完整 CPU/GPU 回归 |

## 后续定位顺序

1. 完成正在测量的最终整合流程，先报告执行状态、输出完整性、逐区和几何精度，再解释耗时。
2. 对丘脑、海马/杏仁核定位每次 mesh evaluation、Gaussian 更新和后处理输出首次分歧。现有标签命名空间及固定物理评价网格已核验，不通过调整评价网格提高 Dice。
3. GEMS CPU 的强度网格拟合仍占主要时间。稳定整数排序和梯度归约可继续评估，但需保留原排序及累计顺序，并通过完整后验/拟合状态核验后才替换。
4. SynthMorph 逆向场按网络预测、仿射分解和场合成逐阶段隔离；边界和脑内误差分别验收。
5. WMH 保持当前精度，定位 GroupNorm 的显存峰值；20 GB 内完整运行成功前，GPU 状态保留为未通过。
6. 完整 CPU recon v3 的双侧球面配准占墙钟约 36.4%，先定位其查找、成本和线搜索热点，并核对最终分区边界。拓扑不同的顶点指标保持 NA，建立有效对应后才能评价逐点厚度、面积、体积与曲率；不能用表面距离替代。

## 原软件存在、当前接口未实现的模式

SynthSegPlus 目前是普通 SynthSeg 2.0 `--parc`，未实现 robust SynthSeg+、官方 QC、目录批处理、CT、Photo 或 v1；`--parc --color-lut` 组合也未实现。TorchFAST 目前为三组织 T1，不含 T2/PD、多通道、其他组织类数及部分原版先验模式。这些属于功能范围，不能记作 benchmark 通过。

## GEMS CPU：保留拟合轨迹的候选优化

本轮[完整脑干同输入 stage](task5/README.md)中，强度网格拟合为 689.308 秒，约占该 recipe 的 91%。owner、EM 和 Gaussian 的细项优先级来自冻结 v2 的五次 core evaluation 限额诊断，主动停止退出码为 75；它们不是最新联合候选或 raw v5 的完整热点占比，也不与 recipe/API 重复累加。[联合候选](task5/cpu_compact_20261004/README.md)已有完整后验与拟合状态恒等核验，API 从 733.021 降至 668.801 秒。

后续候选尚未实现或测量：

- Gaussian M-step 缓存固定 EM 图像的展平、有限值/非零掩膜及筛选数组；图像、bias、dtype 或版本变更时失效。保留原质量求和、均值矩阵乘法和显式差值协方差计算。
- 四面体 owner IDs 和拓扑逐值相同时，复用自有 IDs、稳定排序及分段布局；保留原点到四面体、四面体到顶点的累计顺序和已验收 FP64 梯度归约。
- alpha 无梯度且形状、stride、版本及 owner 不变时，缓存 alpha 的 gather；动态几何、原 FP32 插值、边界判定、归一化和背景规则保持。GPU 路径不进入这些 CPU 缓存。

每项替换仍须通过完整后验、网格、Gaussian、objective 和 solver 接受/停止状态的核验；局部热循环时间不能代替完整函数耗时。
