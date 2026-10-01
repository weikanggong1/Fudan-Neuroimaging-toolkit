# DicL GPU速度优化：真实1000人验收

本轮完成敏感行回退、LARS分段CUDA图和顺序字典逐元素融合。最终源码在同一1000人的VBM、FA、MD缓存投影上完成全部DicL拟合，观测耗时 **63.65秒**，分配显存峰值 **2.31 GiB**。字典与LASSO门槛三模态通过，FA/MD的OMP30重建仍未通过。结果说明优化没有损害本次效果，尚不能称整个sklearn字典学习或BigFLICA全流程等价。

## 1. 输入、参数与比较范围

完整掩膜体素数为VBM **157,901**、FA **222,257**、MD **222,261**；影像网格为91×109×91、2 mm。复用已保存的float64 R500投影，固定D200、seed0、batch32、alpha1、最多1000 epoch、稀疏路径预算1000，沿用sklearn的tol=0.001和max_no_improvement=10。两端实际停止于VBM123、FA149、MD286批，共558批。CPU参考是服务器实际安装的scikit-learn1.7.1，只拟合过一次；本轮复用该参考。GPU为共享H100 PCIe、PyTorch2.5.1/CUDA11.8。

输入沿用历史实验“每体素跨被试去均值，再除以模态整体RMS”的规范；本轮计时只覆盖缓存投影之后的DicL。没有重新运行mMIGP、FLICA、原始NIfTI读取、C20有效成分或最终脑图，也没有启动30,000人测试。匿名[数值报告](dicl_speed_optimization_real1000_20261001.json)保存全部顺序观测、指标和文件哈希。

公开GPU入口通常生成float32投影，而当前GPU DicL把这些投影提升为float64标准化和计算；CPU fit_dicl保留float32输入的mean/std与训练dtype。因此本轮float64投影验收不能覆盖公开默认float32规范。当前DicL仍为原有float64，TF32不会加速这些运算，没有引入float16。

## 2. 已完成的三项优化

### 2.1 只回退敏感行

每个样本的稀疏编码独立求解。此前ADMM抛光后，只要一行接近sklearn路径节点，就将整个32行批次重跑兼容LARS；现在保留已经接受的编码，仅重跑敏感行，再按原顺序写回。真实MD前32批中，7次近节点回退只有8个敏感样本：原先重算224行，现在重算8行；加上前四批冷启动，总回退136行。batch32、字典/A/B更新、shuffle、停止规则和RNG顺序保留。

### 2.2 围绕LU求解捕获CUDA图

兼容LARS的每次事件分为求解前和求解后两段CUDA图；密集LU仍调用原PyTorch求解。保留原节点停止、条件插值、斜率舍入、原子退出、每四事件一次主机完成检查以及预算末尾检查。完整D×D求解复杂度仍在。图缓存每线程最多4项；输入精度/设备及TF32、确定性选项参与缓存区分，缓冲区重复使用和跨流调用有完成事件保护。这只涉及LARS工作区，完整fit的并发安全仍须单独核验。

实际服务器能力检查中，D200/float64的solve_ex及lu_factor_ex+lu_solve，batch1均可捕获并正确回放，batch2和batch32均在捕获阶段失败，普通预热和捕获后的CUDA健康检查正常。这个结果针对PyTorch2.5.1/CUDA11.8/H100组合，不推断其他后端也失败。因此采用分段图保留eager LU，没有为图捕获添加正则化、调整求解数学或忽略错误。完整能力结果见JSON的capability_h100.json；它属于能力检查，不是性能benchmark。

### 2.3 融合顺序字典更新的逐元素操作

200个原子仍按Gauss–Seidel顺序更新。每原子的cuBLAS matvec及torch.linalg.vector_norm完全保留，只融合matvec后的减法/除法/加法，以及范数后的clamp/division。关闭浮点运算收缩，使用round-to-nearest除法；NaN按torch.clamp_min传播，stride和尾部掩膜受到检查。死原子和RNG分支保留原Torch逻辑。

Triton是现有Conda环境的triton==3.1.0依赖，配套PyTorch2.5.1，没有新增安装要求。CPU或缺少Triton时保留Torch更新。融合支持float32/float64，但本轮真实数据速度及效果验收为float64。

## 3. 真实MD前32批：保留全部ABBA观测

每组都从保存的CPU初始字典和RNG开始，以基线A、候选B、候选B、基线A顺序运行。未包含完整SVD、整模态预加载或FLICA。表中“设置”包含本次构造/编译/图准备；“训练”是32批wall时间；“LARS流”是CUDA事件计时，包含该流等待，并非独占GPU纯计算时间。“总秒”为设置加训练；LARS流时间已包含在训练中，不再相加。

| 方案 | 顺序 | 设置秒 | 训练秒 | 总秒 | LARS流秒 |
|---|---|---:|---:|---:|---:|
| 仅敏感行回退 | A1 | 0.338 | 9.756 | 10.094 | 7.531 |
| 同组 | B1 | 0.405 | 5.490 | 5.894 | 3.682 |
| 同组 | B2 | 0.289 | 5.333 | 5.623 | 3.613 |
| 同组 | A2 | 0.287 | 13.260 | 13.547 | 10.741 |
| 敏感行回退＋融合更新 | A1 | 0.687 | 32.789 | 33.476 | 24.523 |
| 同组 | B1 | 3.230 | 19.047 | 22.277 | 11.280 |
| 同组 | B2 | 0.215 | 15.853 | 16.068 | 9.951 |
| 同组 | A2 | 0.201 | 26.658 | 26.859 | 20.290 |
| 敏感行回退＋分段图 | A1 | 0.452 | 6.587 | 7.038 | 5.615 |
| 同组 | B1 | 0.309 | 4.093 | 4.402 | 3.045 |
| 同组 | B2 | 0.356 | 1.680 | 2.036 | 1.244 |
| 同组 | A2 | 0.214 | 4.973 | 5.187 | 4.319 |
| 分段图＋融合更新实验 | A1 | 0.425 | 4.763 | 5.188 | 4.094 |
| 同组 | B1 | 1.197 | 8.649 | 9.846 | 5.836 |
| 同组 | B2 | 0.266 | 7.696 | 7.961 | 4.891 |
| 同组 | A2 | 0.239 | 11.766 | 12.005 | 9.322 |
| 首次整合源码 | A1 | 0.614 | 10.304 | 10.917 | 7.855 |
| 同组 | B1 | 2.896 | 6.581 | 9.477 | 4.402 |
| 同组 | B2 | 0.268 | 5.646 | 5.914 | 3.732 |
| 同组 | A2 | 0.309 | 9.701 | 10.010 | 7.463 |

共享GPU负载随时间明显变化。例如分段图＋融合实验的首次候选训练8.649秒，比此前基线4.763秒更慢；同组末次基线又升到11.766秒。没有删除这组较慢结果。最终整合源码这组A1/A2为10.304/9.701秒，B1/B2为6.581/5.646秒；这支持该时窗内的改善，但不足以给出独占GPU或CPU/GPU受控加速倍数。首次候选设置2.896秒包含冷构造开销，完整耗时中保留这项。

所有五组观测的候选相对旧基线，最终字典/A/B/全部编码相对L2差分别约9.30e-14、1.27e-13、9.43e-14、6.92e-14；RNG哈希一致，同一版本重复结果逐位一致。这里的“重复逐位一致”不表示基线与候选所有值逐位一致。候选相对CPU第32批字典差为1.37e-7，与旧实现水平一致。

## 4. 真实checkpoint的字典更新配对控制

为区分共享负载和融合贡献，使用同一真实第32批checkpoint，单独配对调用顺序updater。每对交替先运行Torch或融合版本。初始化/字典/A/B相同，输出逐位一致，相对差0。

| 配对 | 先运行 | Torch wall毫秒 | 融合wall毫秒 | Torch流毫秒 | 融合流毫秒 |
|---:|---|---:|---:|---:|---:|
| 1 | torch | 33.439 | 6.538 | 32.917 | 4.752 |
| 2 | fused | 17.207 | 8.924 | 12.900 | 6.855 |
| 3 | torch | 17.279 | 11.027 | 12.895 | 6.805 |
| 4 | fused | 17.272 | 11.239 | 12.839 | 6.826 |
| 5 | torch | 16.868 | 11.242 | 12.438 | 6.820 |
| 6 | fused | 16.730 | 11.235 | 12.309 | 6.785 |

六对中融合版本均更快。流计时中位数由12.867降至6.812毫秒，中位数之比约1.89。这是单一真实checkpoint上的updater局部改善，不外推为全部DicL或BigFLICA的同等提速。

## 5. 完整三模态拟合：首轮与最终源码复测

| 观测 | 三模态秒 | 分配显存峰值GiB | 主机RSS峰值GiB | 来源与边界 |
|---|---:|---:|---:|---|
| 历史CPU参考 | 103.900911 | — | — | 8线程，复用既有参考；投影已在内存 |
| 历史GPU精度版 | 141.290862 | 2.615365 | 1.756474 | 既有观测，早期预加载边界 |
| 首次整合完整版 | 58.776645 | 2.312058 | 0.989925 | 核心0221ff… |
| 最终源码完整复测 | 63.654240 | 2.312060 | 1.021446 | 核心56735e… |

GPU完整调用计时包含投影I/O、标准化、SVD初始化和观察器字典快照；不是移除观察器后的纯生产速度测试。四行处于不同运行时间窗、共享GPU负载和计时边界；CPU投影读取边界也不同。因此58.78/63.65秒与历史103.90/141.29秒只作为观察，不能标为CPU/GPU或版本间受控加速。两次优化版分配器硬上限为18 GiB，峰值约2.31 GiB，满足此次预算；没有以此保证任意维度或并发数都满足20 GiB。

### 最终源码额外修复及复测结果

回归中发现，新增LARS图在float32任意alpha时把alpha与容差分别舍入后相加，可能改变原eager路径的节点停步阈值；R500、alpha=10.0000005可复现。已改为按原主机标量顺序相加，再一次转换到节点阈值缓冲区。

首次完整核心SHA-256：`0221ff0f8d816cc40f70232f8052097e9d8c923d8c635cea0def10c730e9e03f`。修复后的最终核心SHA-256：`56735ea2ecd7543e8078a51c9a772f1db3d55fdfed132194015f8982ed76474f`。最终源码已重新完成三模态GPU训练和效果评估，两阶段退出码均为0。最终与首轮的三模态初始、原始和归一化字典全部逐位一致，相对L2差均为0；修复未改变本次float64/alpha1结果。最终训练仍在123/149/286批停止。

评估报告的source_sha256记录评估进程加载的模块，GPU_training_source_sha256记录被评估字典的实际训练模块；两者分别为旧参考415c09…及最终56735e…。训练来源以后一字段和训练report为准，未把评价进程加载的模块冒充训练源码。

## 6. 数值验收：OMP门槛仍未通过

复用每模态2000个评估体素，重新核对它们未进入本次两端实际训练批次；这些体素已参加全数据标准化和SVD，不是独立留出被试/体素的泛化测试。字典Hungarian匹配和符号对齐后，本次匹配均为原顺序。两端评估均使用相同CPU编码器；GPU字典的OMP30检查不表示新增GPU OMP接口。

| 指标 | 固定门槛 | VBM | FA | MD |
|---|---|---:|---:|---:|
| 原始字典相对L2差 | ≤0.001 | 1.223e-07 | 0.000107154 | 0.000584811 |
| 归一化字典相对L2差 | ≤0.001 | 1.22307e-07 | 0.00010715 | 0.000584876 |
| 原始最小原子余弦 | ≥0.999999 | 0.999999999999964 | 0.999999978217310 | 0.999999646118476 |
| 归一化最小原子余弦 | ≥0.999999 | 0.999999999999964 | 0.999999978049693 | 0.999999646395262 |
| LASSO重建相对L2差 | ≤0.001 | 1.57201e-07 | 0.000131198 | 0.00064721 |
| LASSO目标值相对差 | ≤1e-05 | 1.31722e-11 | 5.89927e-08 | 4.28122e-07 |
| OMP30重建相对L2差 | ≤0.001 | 1.32886e-07 | 0.0244475（未通过） | 0.0533408（未通过） |

三模态字典、余弦、LASSO目标值及LASSO重建均通过；VBM通过全部门槛，FA/MD的OMP30重建差仍为 **2.4447%/5.3341%**，大于0.1%门槛。没有放宽阈值，也未用近似相等的LASSO损失替代OMP检查。重建差定义为||GPU字典重建−CPU字典重建||/||CPU字典重建||，并非两端残差之差。

本轮优化不是OMP差异修复。前轮[效果匹配报告](dicl_match_real1000_20261001.md)已观察到少量OMP支持集切换放大重建差，以及初始化/在线字典算术微扰的累计。当前优先继续追踪这些分叉；不能从此次耗时下降推出完整MiniBatchDictionaryLearning输出等价。

## 7. 回归、复现与来源

最终源码定向回归 **146 passed、52.63秒、无跳过**，本地RTX3060/PyTorch2.4.1/sklearn1.2.2/NumPy1.26.4。回归覆盖LARS节点/插值/退化回退、图预算/缓存/流隔离、敏感行重启、顺序融合、非连续输入、尾部掩膜、NaN/Inf、死原子RNG和无Triton回退。这些生成夹具用于检查实现；以上真实精度和时间来自服务器真实1000人投影。

已有[benchmark_dicl_match.py](benchmark_dicl_match.py)暴露cpu、gpu、eval三个独立阶段，输入为三模态*_projected.h5，各data为体素×500的float64矩阵。完整命令、参数、独立验证环境及输出结构沿用[效果匹配报告的复现部分](dicl_match_real1000_20261001.md#复现与参考实现)；保持CPU8线程、seed0、batch32、D200、alpha1及预算1000。保存CPU参考后重复GPU或评估即可，无需重跑CPU。实际数组、排列和字典保持私密，只发布匿名聚合与哈希。

本轮真实运行保留私密运行器SHA-256 `a14c750231828c53736ea58ec03f38c0c6a1415d190cd7a620dadff9cb3b6e71`；公开脚本SHA-256 `004c21644bc30c467cfc0ed8881c6618ed85a10f9689dc5f499d8d4439ba631b`。两者不是同一文件，本轮没有声称公开脚本已在真实数据上从头冷启动重跑。

算法参考：[Mairal等，Online Learning for Matrix Factorization and Sparse Coding，JMLR2010](https://jmlr.org/papers/v11/mairal10a.html)。原软件接口及参数见[sklearn1.7系列MiniBatchDictionaryLearning文档](https://scikit-learn.org/1.7/modules/generated/sklearn.decomposition.MiniBatchDictionaryLearning.html)、[1.7.1字典学习源代码](https://github.com/scikit-learn/scikit-learn/blob/1.7.1/sklearn/decomposition/_dict_learning.py)与[LARS源代码](https://github.com/scikit-learn/scikit-learn/blob/1.7.1/sklearn/linear_model/_least_angle.py)。本项目CPU对应pipeline.fit_dicl，GPU对应[fit_dicl_gpu_streaming](../../src/fnit/bigflica/dicl_torch.py)；不调用原神经影像软件运行时。1.7系列文档页面可能显示后续补丁版本；本次精确1.7.1的依据是实际安装环境和1.7.1源码tag。

## 8. 尚待完成

1. 建立同输入CPU float32参考，统一CPU/GPU标准化和dtype，再验收TF32；不能从本轮float64结果外推。
2. 定位在线更新的首个数值分叉和OMP支持集切换，继续按现有0.1%重建门槛改进。
3. 固定共享GPU负载、I/O冷暖与观察器边界，再做可复现的完整DicL速度对照。
4. 验收FLICA C20有效成分及最终脑图后，再推进大样本与end-to-end结论。

## 9. 复用CPU参考的最短GPU与评估命令

在仓库根目录、已安装FNIT和scikit-learn1.7.1的验证环境执行。投影目录应已有vbm/fa/md_projected.h5；CPU参考目录使用此前保存的完整参考，新的GPU及评估输出目录必须尚不存在。`CUDA_VISIBLE_DEVICES=0`选择待测GPU，输入目录保持私密。

```bash
projection_directory=/absolute/private/mmigp_R500  # 三模态float64投影缓存
cpu_reference_directory=/absolute/private/cpu_reference  # 已完成的CPU参考
new_gpu_result_directory=/absolute/private/new_gpu_result  # 新建训练输出
new_evaluation_directory=/absolute/private/new_evaluation  # 新建对照输出

# 完整三模态GPU DicL：保持默认seed0、D200、batch32、alpha1和预算1000。
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTHONPATH=src \
  python validation/bigflica/benchmark_dicl_match.py gpu \
  "$projection_directory" "$cpu_reference_directory" "$new_gpu_result_directory"

# 复用CPU参考，检查字典、LASSO与OMP30；每模态默认2000个评估体素。
OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTHONPATH=src \
  python validation/bigflica/benchmark_dicl_match.py eval \
  "$projection_directory" "$cpu_reference_directory" \
  "$new_gpu_result_directory" "$new_evaluation_directory"
```
