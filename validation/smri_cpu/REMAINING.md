# 本轮尚未通过或缺少完整实测的项目

此清单对应 2026-10-04 同节点 CPU 工作，详细指标和源码身份见[总报告](README.md)。已完成对照与尚未达到目标分开记录；不以旧 GPU 结果填补本轮 CPU 测量。

| 工作包 | 已测试、尚未通过 | 尚缺的完整真实输入覆盖 |
|---|---|---|
| SynthStrip / SynthSR | SR 默认浮点输出有 3,522 点超过原定容差；Strip GPU TF32 对官方 CPU 有 59 个 mask 差异。未采用改变 SR 量化输出的布局、BN 原型 | 内存影像入口的独立完整原版对照；SR 本轮 GPU 完整对照 |
| SynthSeg / Plus / WMH | 普通 Seg 尚无稳定官方 CPU 速度优势；仍有少数官方标签/体积差。大图 Plus CPU 崩溃已修复且真实 CPU 回归通过，最新 GPU 回归进行中。WMH GPU 在 20 GB allocator 预算内失败 | 非默认 `min_pad`；CPU keep-geometry/色表独立计时。WMH 四模式网络已测，新 header 对两个模式重新核验，其余计时仍绑定此前源码 |
| SynthMorph | affine 逆向上边界、joint 逆向场与上边界未全部过门；rigid 零动态范围边界有三点非零；官方 init+mid-space 分支自身异常另列 | World 链越界周期采样；当前 CPU 边界实现的完整 490 帧对照。真实两帧 DWI 和 14 项同场 apply 已测 |
| TorchFAST / FastVBM | 默认 FAST 少量 PVE 差仍保留；默认 `tensor` 与官方差异更大。FNIRT VBM 尚未等价；Morph 三张标准图误差较小，但与官方 13 图没有逐位相同。最终 v4 两链完整补测已完成，输出与 v2 相同 | 官方 Morph 冷完整链；pipeline AB-BA、其他 FAST 组合、默认 tensor VBM、显式 brain/mask 和本轮完整 VBM GPU |
| 亚区 / recon-all | 丘脑与双侧海马/杏仁核逐区门未全过；CPU stage 仍慢于官方。脑干联合优化降低时间 8.76%，完整后验和拟合状态相同。recon v3 已完整执行并[完成评分](task5/recon_complete_cpu_v3/README.md)，墙钟慢 4.99%，皮层分区和几何仍有差异。44 顶点图、12 注释因顶点对应不成立记 NA；raw v4 CPU 卷积崩溃已修复，v5 新空目录运行中 | 原始 T1 的 CPU 全亚区整例、`balanced`；有效顶点对应、DKT/a2009s/BA 具名逐区摘要、三图谱同公式 no-TH3 体积；其他半球策略、批量入口及当前整合版完整 CPU/GPU 回归 |

## 后续定位顺序

1. 完成正在测量的最终整合流程，先报告执行状态、输出完整性、逐区和几何精度，再解释耗时。
2. 对丘脑、海马/杏仁核定位每次 mesh evaluation、Gaussian 更新和后处理输出首次分歧。现有标签命名空间及固定物理评价网格已核验，不通过调整评价网格提高 Dice。
3. GEMS CPU 的强度网格拟合仍占主要时间。稳定整数排序和梯度归约可继续评估，但需保留原排序及累计顺序，并通过完整后验/拟合状态核验后才替换。
4. SynthMorph 逆向场按网络预测、仿射分解和场合成逐阶段隔离；边界和脑内误差分别验收。
5. WMH 保持当前精度，定位 GroupNorm 的显存峰值；20 GB 内完整运行成功前，GPU 状态保留为未通过。

## 原软件存在、当前接口未实现的模式

SynthSegPlus 目前是普通 SynthSeg 2.0 `--parc`，未实现 robust SynthSeg+、官方 QC、目录批处理、CT、Photo 或 v1；`--parc --color-lut` 组合也未实现。TorchFAST 目前为三组织 T1，不含 T2/PD、多通道、其他组织类数及部分原版先验模式。这些属于功能范围，不能记作 benchmark 通过。
