# BigFLICA：真实 UKB 输入、GPU 对照与内存测试

2026-09-30 在用户指定的 UKB 多模态目录核验，共有 87,720 个被试目录；其中 37,182 个同时含 `VBM_2mm.nii.gz`、`dti_FA_2mm_mmorf.nii.gz`、`dti_MD_2mm_mmorf.nii.gz` 和 `zstat1s.nii.gz`。抽查影像为 `91×109×91`、2 mm MNI 网格。任务图文件是 **zstat**，不是 tstat。

性能和成分数判断以 **2,050 人、VBM/FA/MD 三模态完整掩膜**为当前主要样本；18 人小掩膜结果仅用于接口、数值定位和新被试调用检查。两种样本的输入和输出范围在各节分别注明，不互相外推。

同输入精度测试取排序后的前 18 名四模态齐全被试拟合，下一名只用于 `apply_model`。每模态从前三名共同非零体素中每隔 20 个取一个：VBM 7,896、FA 11,113、MD 11,114、zstat1 14,688 个体素。掩膜用于快速同输入核验，不是推荐的全脑生物学掩膜。参数固定为 `n_components=3`、`migp_dim=10`、`dicl_dim=40`、`dicl_max_iter=20`、`flica_max_iter=100`、`top_voxels=300`、`random_state=0`。原 notebook 的最大迭代数为 1000；本报告结论只适用于所列参数。影像、被试名单和脑图保留在用户远程验证目录；仓库仅保存不含被试 ID 的汇总结果。

参考算法是用户 notebook 中的 NumPy/SciPy mMIGP、sklearn `MiniBatchDictionaryLearning` 与[原作者 FLICA 变分推断](https://github.com/weikanggong/BigFLICA/blob/master/FLICA_cpu.py)。上游检出为 `125d44451f288977810105edd29f4869ee5d8317`，含 NumPy 兼容修订。原 `BigFLICA_cpu.py` 用 SPAMS DicL；本次比的是用户指定的 **sklearn 版本**。原回归函数将 t 值命名为 Z；两边的成分图均按相同自由度换成带符号标准正态 z 后比较。

## 同输入精度

| 路径与比较 | 被试 course 绝对相关 | 四模态 × 三成分 z 图 | 解释 |
|---|---:|---:|---|
| FNIT CPU 与 notebook CPU | 三项均为 1.0 | 相关 ≥0.9999999999999993；MAE 1.43–2.17×10⁻⁸ | 保留原 FLICA 数值核心；[原始核对](map_compare.json)。 |
| GPU 直接体素 FLICA 与原 FLICA 直接读取相同标准化体素 | 最低 0.999999978 | 最低相关 0.999999973；MAE 5.3×10⁻⁵–2.29×10⁻⁴ | 跳过 mMIGP/DicL；[逐模态结果](raw_cpu_compare100.json)。 |
| GPU mMIGP+DicL+FLICA，**仅在验证中**将 mMIGP 特征向量符号对齐到 CPU 参考 | 最低 0.999994827 | 最低相关 0.999995042；MAE 0.00122–0.00263 | 字典学习批次均为 32；[逐模态结果](stream64_aligned_compare.json)。 |
| GPU 压缩模式，独立运行且不使用 CPU 参考符号 | 0.351、0.596、0.892 | 相关最低 0.223；MAE 0.402–1.039 | 默认 batch=32 的[完整结果](gpu_default32_final.json)；不能称为参考输出等价。 |

直接体素 GPU 路径生成 `18×3` 的 course、12 张原网格 z NIfTI、12 张 top-300 阈值 NIfTI 与 PNG，以及冻结的掩膜、均值、标准差和空间载荷。下一名未参与拟合的被试通过 CUDA `apply_model` 得到三个有限 course。原 BigFLICA 没有等价的新被试函数，所以不声称预测效度或后验逐点一致。结果见 [GPU 完整运行](gpu_raw_full.json)；CPU 对照仅拟合 FLICA 和计算 z 图，未生成相同数量的 PNG。

压缩路径中，float64 分块 mMIGP 的投影与原 CPU mMIGP 在相同符号下相对误差约 4.5–6.0×10⁻¹⁵。GPU Lasso-LARS 在同一投影输入上，VBM 字典对 sklearn 的相对误差 1.04×10⁻⁵，见[字典对照](dicl64_one.json)；100 个真实小批次的 CPU 张量 LARS 与 sklearn 字典相对差约 2.86×10⁻⁹。GPU FLICA 的同初始化 100 次更新相对误差约 2.45×10⁻⁹。其余差异主要来自 mMIGP 特征向量的任意正负号：只改符号并继续用原 CPU sklearn/FLICA，三项 course 从 0.266–0.741 变为 1.0，见[控制实验](orientation_effect.json)。固定随机种子的字典学习并不自动消除这个符号变化。符号对齐实验读取了 CPU 参考 `U.npy`，用于检验各 GPU 数值步骤，**不是独立 GPU 运行模式的成绩**。

另用**当前默认 batch=32 的 GPU mMIGP 投影**作同输入控制：10 列中 4 列与 CPU 特征向量符号相反；对齐后 `U` 相对误差为 6.19×10⁻¹⁵。两组只在符号上不同，后续均使用相同的 CPU sklearn DicL 和原 FLICA。未对齐时，最优一对一成分匹配的 course 相关为 0.351/0.596/0.892，几乎复现独立 GPU 运行；对齐后为 1/1/1。未对齐的三个 course 子空间主角为 10.4°/33.1°/61.6°；用 GPU course 最优线性拟合中心化 CPU course 后，Frobenius 相对残差为 0.574。因而差异不只是成分置换或整体符号，而是符号改变后 DicL/FLICA 拟合到了不同的子空间。该实验在后续阶段使用 CPU，仅用于定位原因；[不含被试 ID 的汇总](orientation_current_default32.json)。

## 耗时与资源

以下都是单次观测。CPU 为 headcw Xeon Gold 6418H；GPU 为 gpucw1 H100 PCIe，当时有其他 GPU 作业，占用和计算负载会影响耗时。不同路径输出边界不同时不计算“总加速比”。

| 阶段 / 路径 | 秒 | 说明 |
|---|---:|---|
| notebook CPU：读取/标准化、mMIGP、sklearn DicL、FLICA | 0.99 / 0.012 / 4.45 / 0.357 | 四阶段合计 5.81；不含成分 NIfTI/PNG 与新被试模型；[原始记录](benchmark.json)。 |
| FNIT 历史 CPU 测量：读取/标准化、mMIGP、sklearn DicL、FLICA、成图、投影模型 | 1.14 / 0.195 / 5.08 / 0.457 / 2.72 / 0.053 | 冷启动带全部输出墙钟 9.71；[当时的源码快照记录](cpu_final_metadata.json)，不能视为当前源码的重新测量。 |
| GPU 直接体素：读取/标准化、FLICA、成图、投影模型 | 1.28 / 43.74 / 3.43 / 0.213 | 墙钟 48.76；进程峰值显存 0.157 GiB、内存 0.882 GiB；原 CPU 直接体素 FLICA 同输入为 2.39 秒。 |
| GPU 压缩、默认 batch=32：读取、mMIGP、DicL、FLICA、成图、投影模型 | 1.37 / 1.12 / 474.47 / 7.15 / 3.67 / 0.189 | 墙钟 488.10；显存峰值 0.156 GiB、内存 0.985 GiB；[完整结果](gpu_default32_final.json)。 |
| GPU 压缩、batch=32，仅 DicL 四模态 | 486.33 | 符号对齐实验；VBM 单模态 119.46 秒，对照 CPU 1.52 秒。当前 PyTorch LARS 在小批量上明显慢于 sklearn。 |
| GPU 压缩、batch=1024：读取、mMIGP、DicL、FLICA、成图、投影模型 | 1.43 / 2.73 / 153.07 / 7.29 / 3.43 / 0.211 | 墙钟 168.32；显存峰值 0.156 GiB、内存 0.977 GiB。批次不同于 notebook，不作精度等价结论；[完整运行](gpu_canonical_full.json)。 |
| 上一行相同输入、参数与缓存，仅将成分数改为 2 | 9.33 总计 | `mmigp_reused=true`、`dicl_reused=true`；再次运行 FLICA、成图和投影模型。 |
| 当前源码入口，GPU 压缩、batch=4096；再改为两个成分 | 33.23 / 9.36 总计 | 冷启动生成 12 张 z 图；再次运行复用 mMIGP/DicL。此大批次仅核对最新代码路径，不用于 notebook 精度比较；[运行记录](gpu_final_smoke.json)。 |

**2,050 名真实被试分块测试**使用 VBM 和 FA 两个模态、同一核验掩膜：标准化 HDF5 实际数据 311,747,600 字节，逐被试建库 68.39 秒；mMIGP 2.17 秒，进程峰值显存 0.125 GiB、内存 0.743 GiB。初版 6 次 GPU 子空间迭代的特征对残差为 3.47×10⁻⁴；新增自适应迭代在第 18 次达到 3.03×10⁻⁹，mMIGP 2.83 秒，显存峰值 0.095 GiB，见[分块基准](large2050.json)和[改进结果](large2050_refined.json)。测试覆盖了 `N>2048` 的随机特征分解分支，但这两个模态在所选小掩膜下本就可以放入 32 GB 内存；“完整大模态不进内存”由逐被试 HDF5 写入及逐体素块读取实现，**未**做 37,182 人 × 全脑体素的端到端耗时实测。按 100 万掩膜体素估算，float64 规范化缓存约 277 GiB/模态，实际磁盘容量必须单独安排。

直接体素模式在大于 2,048 人时使用随机特征分解，并将原 FLICA 的自动空间自由度估计固定为 1；此大样本分支尚未完成与原软件的精度对照。大样本端到端的受控 CPU/GPU 加速比尚不能报告；后文的 2,050 人 C3 冷启动只给共享节点观察时间。真实 UKB 派生 PNG 从远程复制到本地工作区曾被自动审批拒绝，理由是可能含敏感数据；本仓库不附脑图，原图保留在用户远程结果目录。

## 中规模速度门槛与算法修正

用户要求 mMIGP、DicL 和 FLICA 各阶段的 GPU 版本均明显快于 CPU 后才运行 30,000 人全样本。因此 30,000 人仅固定了私密随机被试清单和掩膜，未进行全样本模型拟合。以下两组真实被试沿用该清单的前 500 人或前 2,050 人。四模态完整掩膜分别含 VBM 157,901、FA 222,257、MD 222,261、zstat1 293,754 个体素，共 896,173 个。500 人和 2,050 人逐被试构建 float64 HDF5 分别耗时 52.3 秒和 220.3 秒，峰值内存分别为 0.582 和 0.688 GiB；影像及含被试 ID 的缓存保留在私密远程目录。

| 阶段和真实输入 | CPU 最快实测 | GPU 实测 | 精度及结论 |
|---|---:|---:|---|
| mMIGP，500 人、四完整掩膜、R=100 | 同节点 8 线程 15.27 秒 | 13.49 秒 | `U` 相对差 1.99×10⁻¹⁴，四投影相对差最大 1.36×10⁻¹⁴；仅快 1.13 倍。[对照](mmigp500_full_cpu_gpu.json) |
| mMIGP，2,050 人、四完整掩膜、R=100 | 同节点 8 线程 68.19 秒 | 54.59 秒 | `U` 相对差 7.64×10⁻⁸，四投影相对差最大 3.17×10⁻⁸；仅快 1.25 倍。[对照](mmigp2050_full_cpu_gpu.json) |
| DicL，2,050 人的小掩膜 VBM 投影 7,896×100，D=50、batch=32 | 同节点 sklearn 单线程 4.09 秒 | PyTorch GPU 33.37 秒 | GPU 改用 sklearn 同种子的随机 SVD 初始化后，最终字典相对差 4.48×10⁻⁹；速度未达标。[受控训练](dicl_r100_control.json)、[GPU 结果](dicl_r100_vbm_gpu_rsvd.json) |
| DicL，500 人完整掩膜 VBM 投影 157,901×100，D=50、batch=32 | 同节点 sklearn 最快 8 线程 8.13 秒 | PyTorch GPU 34.21 秒 | 最终字典相对差 1.45×10⁻⁷；速度未达标。[GPU 结果](dicl_gpu_rsvd1_vbm_500full.json) |

mMIGP 以 float64 分块处理，每个特征对的相对残差及三段耗时写入 `eigen_diagnostics.json`，大样本未达 `1e-8` 会停止而不会缓存结果。2,050 人完整掩膜时 GPU 的协方差、特征分解、投影分别耗时 24.02、0.97、28.70 秒；投影阶段与 CPU 的 26.41 秒接近，说明当前总墙钟受磁盘读写限制。GPU 两卡当时被其他作业高度占用，因此上述耗时仅是同机观测，不能外推为空闲 H100 的速度。

DicL 原 GPU 初始化使用精确协方差特征分解；在 R=100、D=50 时，这与 sklearn 随机 SVD 的初始字典相对差 0.433，并经非凸训练放大到最终字典约 0.924。使用相同初始字典、相同标准化输入和批次顺序时，CPU/GPU 训练 112 步的最终字典相对差为 4.45×10⁻⁹。现已在 GPU 上按 sklearn 的随机矩阵、幂迭代和符号规则实现初始化，初始字典相对差 1.55×10⁻¹³。[初始化控制](dicl_r100_init_compare.json) 和 [训练控制](dicl_r100_control.json) 区分了精度问题与速度问题。18 人四模态同输入对照中，新版 DicL 字典相对 sklearn 为 7.8×10⁻¹⁰ 至 2.8×10⁻⁹；下游 course 与 z 图最小相关均约 0.999995。[回归结果](dicl_r10_rsvd_regression.json)

FLICA 的 C=20、R=100、D=50 设置在 2,050 人小掩膜和 500 人完整掩膜上分别只剩 2 和 1 个有效 course 秩；原 CPU 实现本身也塌缩，因此不能将这两次试验当成有效 20 成分模型。流程现保存 `flica_reconstruction.json` 并拒绝该输出。有效模型需先确定合适的 R/D，再测 GPU FLICA 端到端耗时。其小张量 CUDA Graph 候选对当前 GPU 的全部返回数组逐元素相同，但全新进程捕获加运行 100 轮仍慢于 CPU，故尚未接入正式流程。

### 低秩问题的原源码核对

指定 notebook 的正式示例使用 **R=1000、D=500**；上面 C=20 的中规模试验用 R=100、D=50/100/200，不能把其有效秩外推到正式维度。2,050 人四模态小掩膜中，每个模态均有 2,050 个非零被试行；R=100、D=200 的四个 DicL 字典及加权合并矩阵有效秩均为 99，合并矩阵第 20 奇异值为首值的 0.808。FLICA 前 10 轮保留 20 个成分，到 30 轮只剩 2 个。

为排查移植错误，用公开原版 [FLICA_cpu.py](https://github.com/weikanggong/BigFLICA/blob/master/FLICA_cpu.py)（原文件 SHA-256 `5c82361a381597980f7a81ac89cdf0f7ef3de2639937df3d7021a562d6f5a061`）和 FNIT `flica_vb.py` 读取**完全相同的真实字典**，分别运行 10、30、100 轮。仅修复原文件在现代 NumPy 中的 `np.int` 和最终 free-energy 求和兼容问题；这两处不参与 H 更新。两份实现的 H、四个 X、四个 W 逐数组误差全为 0，DD 相同，有效秩均为 20→2→2。[逐轮聚合结果](upstream_flica_parity.json)。这排除了该试验中 FLICA 移植和 GPU 算术导致的低秩；尚须用 notebook 的 R=1000/D=500 及相同数据查明较小 R/D 与输入选择的影响。

### float32 与 TF32 试验

2,050 人四完整掩膜的 mMIGP 同用 7.514 GB float32 HDF5 缓存、R=100、90 次子空间迭代：同机 8 线程 CPU **31.31 秒**，严格 float32 GPU **21.71 秒**，仅快 **1.44 倍**。两端都达到 `1e-6` 特征对相对残差门槛；GPU 对 CPU 的 U 相对误差 `5.85e-5`、最大子空间主角 `0.014°`，四模态投影相对误差最大 `2.66e-5`。[分阶段及内存记录](mmigp2050_full_float32.json)。GPU 另测 TF32 为 22.31 秒，残差 `2.21e-4`，并未提速；正式 mMIGP 的 float32 计算已关闭 TF32。float32 缓存现可直接从逐被试 NIfTI 生成，原有缓存转换的 45 秒不属于新运行必需步骤；直接生成的总耗时尚未实测。

独立试验中，2050 人小掩膜 C=20/R=100/D=50 的 FLICA 严格 float32 与 TF32 最终 H 相对差 `9.39e-4`；两者都只剩 2 个有效成分，因此这组 3.7–3.8 秒的 GPU 墙钟**不是合格 20 成分模型的速度成绩**。[独立精度记录](flica_float32_tf32_probe.json)。DicL 的严格 float32/Triton 候选已完成同输入核验；其精度达到下面所列水平，但速度未通过门槛。

### 原版 FLICA 的成分剪枝

在同一组真实 R=100、D=200 四模态字典上，公开原 Python FLICA 和 FNIT 标量噪声模式均在第 30 轮由 20 个有效成分降至 2 个，说明这不是 GPU 移植误差。逐式核对发现原 Python 版的 PCA 初始 H 使用协方差特征值，而 [FSL MATLAB 源码](https://fsl.fmrib.ox.ac.uk/fsl/docs/utilities/flica_2013-01-15.tar.gz) 使用奇异值；仅改这一处反而剩 1 个。原 Python 版的逐被试噪声模式还存在广播和不安全视图错误，FNIT 已修复该内部路径，并以每个被试的 H 协方差做单元测试。此模式不是指定 notebook 使用的标量噪声模型。

修复后的逐被试模式在上述四模态输入保留 18/20 个成分，zstat1 的重建范数比仅 `3.69e-6`；去掉 zstat1 后，三个结构模态为 20/20。要求 22 或 24 个成分仍只留下 18 或 19 个；单独提高 zstat1 权重则几乎舍弃 FA/MD。按官方支持的显式 H 初始化可暂时得到四模态 20/20，但两次等价初始旋转的 course 最低相关只有 `0.445`，脑图最低相关为 `0.025–0.289`，不能作为稳定结果。500 人完整掩膜、R=100/D=50 的独立检查只留下 4/20。详细输入规模、模态重建、噪声分布和历史结果见[维度与源码审计](flica_dimensionality_audit.json)。

### 修正 FLICA 自由度拟合前的端到端与符号取向

修正 GPU FLICA 自由度拟合前的 18 人四模态 C=3/R=10/D=40 独立运行：CPU 全流程 `8.20` 秒，GPU `65.48` 秒；GPU DicL 为 `57.55` 秒，是主要瓶颈。两端有效秩均为 3，但 course 最低绝对相关 `0.351`，12 张 z 图最低相关 `0.223`。[逐阶段和逐图汇总](end18_float32_cache_current.json)。两端 mMIGP 特征向量的数学符号任意，符号差异会改变后续非凸 DicL/FLICA 的拟合路径。若 CPU/GPU **双方**独立统一最大绝对载荷为正，course 最低相关 `0.999999964`、z 图最低相关 `0.999999956`；但双方此时都不复现未修改 notebook 的输出。[受控符号试验](sign_equivariance_real18.json)。

DicL 的严格 float32/Triton 候选在同一 18 人输入下，和 CPU float32 的 course 最低相关 `0.999928`、z 图最低相关 `0.999925`，但 GPU `19.86` 秒仍慢于 CPU `6.48` 秒，故没有替换正式 float64 路径。TF32 会产生更大偏差；此前看似 float32 的异常运行实际上被设备选择函数重新开启了 TF32。[严格精度汇总](dicl_strict32_cpu32_18_c3.json)。上述计时均为共享节点单次观测，30,000 人全掩膜运行仍等待 mMIGP、DicL、FLICA 各阶段通过速度与有效成分门槛。

### 去掉 task 的三结构模态基准

按同一 2,050 名真实被试，只保留 VBM（157,901 体素）、FA（222,257）和 MD（222,261）；以下计时均复用已生成的 float32 HDF5，不包含 NIfTI 解码和首次建库。完整掩膜 mMIGP 的 R=100、8 线程 CPU 两次为 `22.09/21.64` 秒，严格 float32 GPU 两次为 `15.47/14.17` 秒，按中位数快约 `1.48` 倍。GPU 与 CPU 的 U 相对差 `5.45e-5`、最大子空间主角 `0.00285°`，三模态投影相对差为 `1.71e-5/1.16e-5/8.31e-6`；两端均满足 `1e-6` 特征对残差门槛。[同输入 ABBA 记录](mmigp2050_three_modal_float32.json)保留当时源码 SHA-256：`streaming.py` 为 `b4cb6404...`，当前版本为 `104cfc35...`，属于历史冻结源测量；[脚本](benchmark_three_modal_mmigp_real2050.py)说明测试方法，不保证逐字节复现当时结果。

先只在 FLICA 阶段去掉 zstat1，使用此前**四模态 mMIGP**产生的 VBM/FA/MD 三个真实 D=200×R=100 字典，逐被试噪声模式 C=20 迭代 101 次后保留 `20/20` 个有效成分。GPU 独立初始化修正后，自由度 DD 对 CPU 的相对差 `1.40e-16`，20 条 course 最小绝对相关 `0.9999999999999785`；CPU 初始化加拟合 `2.57` 秒，GPU `3.93` 秒，仍慢约 `1.53` 倍。VBM/FA/MD 重建范数比分别约为 `0.37/0.48/0.48`，保留 20 个成分并不等于达到所需重建质量。[修正后同输入记录](flica2050_dd_fixed_fourderived_three_structural.json)。该 FLICA 试验不能充当三模态端到端结果。

真正从 VBM/FA/MD 三模态重新拟合 mMIGP R=100，再在三个完整掩膜上用 sklearn DicL D=200 拟合，CPU 与独立 GPU FLICA `R` 模式 C=20 运行 101 次更新后都只有 **17/20** 个有效成分。两端自由度 DD 相对差 `1.14e-16`，17 条有效 course 的匹配绝对相关约为 1，VBM/FA/MD 重建范数比均约为 `0.026/0.348/0.289`；流程按有效秩门槛拒绝输出。CPU 初始化加拟合 `2.80` 秒，GPU `3.75` 秒。这表明此前只在 FLICA 阶段去掉 task 的 `20/20` 不能外推到三模态完整链路。[同输入 CPU/GPU FLICA 记录](flica2050_dd_fixed_true_three_modal.json)。

在同一组三模态 mMIGP 投影上，D=200、最多 1000 轮的 CPU sklearn DicL 分别耗时 VBM `138.74`、FA `12.78`、MD `156.89` 秒，总计 `308.41` 秒。float64 GPU DicL 的 VBM 耗时 `654.42` 秒，为 CPU 同模态的 `4.72` 倍；最终字典相对差 `4.29e-6`，200 个对应原子的最小绝对余弦 `0.99999999967`。GPU 显存使用在结束前采样约 `2.85` GiB，未测得全程精确峰值。速度门槛已失败，GPU FA/MD 未继续运行，**不存在**实测三模态 GPU DicL 总耗时。[阶段与精度记录](dicl2050_three_modal_r100_d200_partial.json)和[可复现脚本](benchmark_three_modal_dicl_real2050.py)。

真实 VBM 投影的逐批剖析显示，LARS 活动集每次事件都调用 PyTorch 稠密线性求解；前三批 LARS 耗时 `0.845/0.194/0.257` 秒，而依次更新 200 个原子仅耗 `0.046/0.014/0.014` 秒。64 原子紧凑求解在未溢出的局部批次仅快约 `1.1–1.3` 倍；融合原子更新的单步微测虽快约 `1.9` 倍，覆盖的耗时太少；严格 float32 也未缩短 LARS。它们都不足以扭转完整 VBM 训练的 `4.72` 倍劣势，因此未替换生产路径。[脱敏微测记录](dicl_vbm_r100_d200_gpu_microprofile.json)明确标注后续微批次加入了稳定项，不代表完整训练轨迹。

进一步的有限敏感性检查表明，三字典的第 20 奇异值/首奇异值分别为 `0.677/0.778/0.820`，未在输入阶段缺秩。普通联合 PCA 的 20 维投影可达到 `0.550/0.520/0.478` 的三模态重建范数比。将 FLICA 请求成分改为 17、使用 `PCAnew` 初始化、固定空间自由度为 1 或把 VBM 输入乘 2，100 轮后的有效秩分别为 `16/17`、`1/20`、`10/20` 和 `13/20`；这些简单改动均未通过 20 成分验收。它提示后续应检查变分更新中的模态竞争及成分剪枝，而不能只调一个权重掩盖结果。[完整汇总](three_structural_cpu_flica_diagnostic_real2050.json)与[可复现 CPU 诊断脚本](diagnose_three_modal_flica.py)。

同一真实字典上追踪 FNIT 修复广播错误后的逐被试噪声更新，1/10/20/30/50/101 次后的有效秩为 `20/20/19/18/17/17`；第 10 次的 VBM 重建范数比已降至 `0.0082`，到第 101 次仍只有 `0.0262`。因此提前截断在 10 次虽可保留 20 个非零 course，却不是收敛后的 20 成分解，也不能修复 VBM 模态被弱化的问题。[逐轮脱敏结果](three_modal_rank_trajectory_public.json)和[复现脚本](diagnose_three_modal_rank_trajectory.py)。

### 2,050 人公开接口冷启动

另从同一 2,050 人的原始 VBM/FA/MD NIfTI、三个完整掩膜，独立运行 CPU `run_bigflica`，不把上述阶段试验缓存伪装成公开 API 缓存。C=3/R=100/D=200、逐被试噪声模式的首次运行用时 `732.75` 秒：读取及标准化 `342.59`、mMIGP `43.37`、sklearn DicL `321.61`、FLICA `1.04`、z 图 `4.37`、固定投影模型 `15.92` 秒；进程峰值 RSS `1.91` GiB，未超过 `32` GiB 上限。输出 `2050×3` course、三个模态共九张 z-stat NIfTI、对应阈值图与 PNG；逐图 NIfTI 与 `.npy` 在掩膜内完全一致，一名未参训真实被试的 `apply_model` 返回有限的三个值。[公开调用聚合报告](three_structural_public_cpu_real2050.json)和[实跑时冻结脚本](benchmark_public_cpu_real2050_frozen.py)，脚本 SHA-256 为 `60743e67...`。

同一节点的 H100 上也从原始 NIfTI 独立冷启动 GPU `run_bigflica`，使用同一有序被试清单、掩膜和参数，float32 标准化及 `max_gpu_gb=19`。GPU 总墙钟 `1045.08` 秒：读取及标准化 `139.85`、mMIGP `14.17`、PyTorch DicL `871.61`、FLICA `1.20`、z 图 `4.23`、投影模型 `12.23` 秒。峰值主机 RSS `1.17` GiB、CUDA 分配 `1.11` GiB、保留 `1.92` GiB。GPU 也生成 `2050×3` course、九张 z-stat NIfTI 和阈值图，并对同一名留出被试完成 `apply_model`。两次运行均在 gpucw1、CPU 限八线程，依次使用新输出目录；CPU 标准化为 float64，GPU 为 float32，操作系统页面缓存未控制，GPU 开始时有外部作业占满 GPU 计算。因此 GPU 总时间为 CPU 的 `1.43` 倍、DicL 阶段为 `2.71` 倍，**只是本次共享节点的观察值**，不是受控加速比。[GPU 聚合报告](three_structural_public_gpu_real2050.json)和[GPU 实跑脚本](benchmark_public_real2050.py)，脚本 SHA-256 为 `76b474e1...`。后一脚本可作为 CPU/GPU 同入口重跑模板，但它并非上述 CPU 测量时的逐字节脚本；其 docstring 保留了原文件名。

两份报告记录了测量时源码 SHA-256。本轮随后把公开入口默认显存预算调为 19 GiB；实测调用原本就显式传入 19 GiB，运行参数没有变化。

两条公开链路的输入签名及有序被试清单相同。mMIGP U 逐列无符号翻转，CPU/GPU 相对 L2 差 `2.18e-5`；VBM/FA/MD 投影差分别为 `6.92e-6/4.75e-6/3.39e-6`。经过 Hungarian 原子匹配和符号校正，三个 DicL 字典的相对 L2 差仍为 `0.549/0.453/0.896`，匹配原子的绝对余弦中位数为 `0.935/0.939/0.604`。三个成分的最优匹配 course 绝对相关为 `0.719/0.132/0.9997`；第二张 z 图在 VBM/FA/MD 的符号及排列校正后相关为 `-0.155/-0.048/-0.265`。分歧从 DicL 阶段扩大并传到 FLICA、脑图；仅凭两条独立运行还不能区分上游输入扰动与 DicL 实现差异，下面的同投影控制进一步区分。此前在**同一 float32 VBM 投影**上的受控 GPU/CPU DicL 字典相对差 `4.29e-6`，说明本次大差异不能仅按 GPU 算术误差解释。[逐阶段、课程与 z 图聚合对照](three_structural_public_cpu_gpu_compare_real2050.json)及[对照脚本](compare_public_cpu_gpu_real2050.py)。

为固定完整三模态的中间输入，又直接读取本次 GPU 公开运行生成的**同一 float32 mMIGP 投影**，转为 float64 后交给 CPU sklearn DicL；GPU DicL 内部也使用 float64 算术。VBM/FA/MD 三个字典相对 GPU 字典的误差分别只有 `1.05e-5/1.55e-6/6.56e-5`，同序原子的最小绝对余弦均大于 `0.99999998`。再以同一 GPU U 拟合 CPU FLICA C3，2050 人三个 course 与 GPU 成分最优匹配后的相关均大于 `0.9999999984`，三模态九张 z 图的相关也均大于 `0.9999999984`。这说明**固定投影时** DicL 和 FLICA 实现高度一致；独立全链的失配由上游微小投影差异在非凸 DicL 中放大。CPU 三模态 DicL 耗时共 `163.37` 秒，GPU 公开运行中 DicL 为 `871.61` 秒；GPU 共享负载和缓存条件不同，这两个数只记录本次观察，不能作受控加速比。[同投影控制报告](three_structural_f32_same_projected_control_real2050.json)及[复现脚本](diagnose_public_gpu_projected_dicl_cpu_real2050.py)。

这两个 C=3 模型虽然通过数值秩门槛，CPU 的 VBM/FA/MD 重建范数比仅 `7.93e-6/0.195/0.196`，GPU 仅 `4.69e-5/0.183/0.111`，因此只用于大样本调用链功能检查。用**本次各自新产生的字典**另跑 C=20 的 FLICA，101 次更新后 CPU float64 仅 `16/20`、GPU float32 仅 `11/20` 个有效成分，均未通过验收；也不能把它们与此前 float32 阶段 CPU 字典的 `17/20` 合并。[CPU C20 诊断](three_structural_public_cpu_c20_probe_real2050.json)、[GPU C20 诊断](three_structural_public_gpu_c20_probe_real2050.json)。

两套 CPU 字典的差别也经过定位：float64 与 float32 mMIGP 的 U 相对差为 `3.73e-5`，100 列没有符号翻转，而同序 sklearn DicL 字典相对差达到 VBM `0.769`、FA `0.629`、MD `0.990`。旧四模态与新三模态 float64 标准化缓存每模态抽查首、中、尾三个数据块完全相同；显著分歧出现在 mMIGP 精度和之后的非凸字典学习链路，尚不能把 C20 秩变化归因于某一个单独步骤。[输入及字典对照](three_structural_dicl_cpu_f64_vs_f32_real2050.json)。

另以这次 float64 字典有限尝试 C24/C25/C28，101 次更新分别得到 `16/18/20` 个有效成分；C28 在 101、201、501 次后均为 `20/28`，恰有八行 H 的范数不超过最大值的 `1e-6`，保留列集合在这三次中相同。旧 float32 字典的 C28 在 101、201 次也得到 `20/28`，但保留列集合与 float64 不同。过完备拟合后筛选可作为后续研究方向；当前公开 API 会拒绝这八条退化列，且删列后的模型不能称作原软件 **C20 同参数**输出。[小范围参数试验](three_structural_public_cpu_overcomplete_probe_real2050.json)、[float64 稳定性](three_structural_public_cpu_c28_stability_real2050.json)、[float32 稳定性](three_structural_old_cpu_f32_c28_stability_real2050.json)和[复现脚本](probe_overcomplete_flica_real2050.py)。

公开调用链也用 18 名真实被试、三个小核验掩膜验证了纯结构模态的 `R` 模式，阈值图取前 `100` 个体素：Python API 首次运行 `30.03` 秒，其中标准化 `0.90`、mMIGP `0.78`、DicL `24.48`、FLICA `0.90`、脑图 `2.52` 秒；请求的 `3/3` 个成分均有效。CLI 复用 mMIGP/DicL 缓存后 `8.35` 秒。九张 z-stat NIfTI 均为 float32，与内存 z 图逐体素往返误差 `0`；输出不含 zstat1。[公开入口与输出检查](three_structural_public_R_smoke_real18.json)。这是小掩膜功能核验，不能替代 2,050 人完整掩膜的 20 成分验收。

同一 18 人输入的 CPU 公开调用也保留 `3/3` 成分，但与上述 GPU 输出的 course 经最优排列和符号对齐后，三个相关只有 `0.284/0.097/0.359`，九张 z 图相关范围 `0.060–0.450`，未达到端到端数值一致。CPU 在 headcw 运行 `9.00` 秒、GPU 在 gpucw1 运行 `30.03` 秒，属于不同节点且 GPU 有共享负载，**不能**当作加速比。[逐成分对照](three_structural_public_R_cpu_gpu_real18.json)。

这一小样本机制试验发现，mMIGP 的十列 U 数值逐列几乎相同，五列方向相反；统一方向后的 U 相对误差为 `1.61e-5`，三模态投影相对误差为 `1.28e-5–1.56e-5`。原 GPU 字典仅在训练后翻转方向，仍与 CPU 差约 `1.02–1.05`；在 DicL **训练前**统一方向，再用 CPU sklearn DicL 和 CPU FLICA 受控重跑，course 最低相关 `0.9999829`，九张 z 图最低相关 `0.9999809`。仅从 GPU float32 规范化缓存重算 SciPy mMIGP 方向，也得到与原 CPU 相同的十列符号；这提供小样本兼容路径，不能作为大样本 GPU 算法的速度成绩。该受控试验没有重跑修正方向后的 GPU DicL/FLICA，因此**尚未**证明独立 GPU 端到端输出等价。[逐阶段符号隔离](three_structural_orientation_real18.json)。

从 GPU `R` 模式模型投影一名未参加训练的真实被试，在 CPU 上得到形状 `(3,)` 且全部有限的 course，耗时 `0.332` 秒。[新被试调用检查](three_structural_public_R_heldout_apply_real1.json)。以上 18 人结果和本项单被试调用均为小掩膜功能与机制核验，不构成 2,050 人完整掩膜、20 成分模型的验收。

## 2026-09-30：增量 LARS 与 CUDA Graph（587944e版本）

本轮把每次 LARS 路径事件的完整 `32×200×200` LU 求解，改为活动集逆矩阵的增量更新：加入原子时使用 Schur 补，移除原子时作逆矩阵降阶。每四次事件用 CUDA Graph 重放，减少 Python 调度；检查活动方程残差和枢轴，异常时从当前小批次重新运行原 PyTorch LU 求解器。字典原子全部有效时，按原顺序重放原子更新图；存在未使用原子时保留原来的 NumPy 随机数顺序和重采样过程。没有更换 Lasso 目标、批次、种子或提前停止条件，也没有引入新依赖。

正式实现的同输入测试使用 **2,050 人 VBM/FA/MD 完整掩膜的同一份 float32 mMIGP 投影**，R100/D200、batch32、seed0、最多1000 epoch。两端 DicL 内部沿用已有 float64 算术，本轮未尝试 float16。CPU sklearn 先实际拟合，后续开发试验复用其字典与测量记录，核对三个投影文件的 SHA-256，不重复拟合 CPU。实测环境为 PyTorch 2.5.1、NumPy 1.26.4、sklearn 1.7.1；项目 Conda 文件仍固定 sklearn 1.5.2，因此本表不是指定 Conda 环境的完整重建测试。

| 模态 | CPU sklearn 秒（本轮已测、随后复用） | 正式 GPU 实现秒 | 字典相对 Frobenius 误差 |
|---|---:|---:|---:|
| VBM：157,901 体素 | 139.05 | 199.33 | 1.0645×10⁻⁵ |
| FA：222,257 体素 | 13.38 | 16.46 | 1.5480×10⁻⁶ |
| MD：222,261 体素 | 13.41 | 22.93 | 6.5621×10⁻⁵ |
| 合计 | **165.84** | **238.72** | — |

三个模态共 2,541 个小批次，均未触发 LU 回退。相对旧 GPU 字典的误差分别为 `1.12e-7/7.62e-9/1.59e-8`；与 CPU 对应原子的最小绝对余弦均大于 `0.99999998`。沿用同一个 U，再运行 CPU/GPU C3 FLICA 和空间回归，course 及九张 z 图的同序成分在符号校正后相关均大于 `0.9999999984`。GPU DicL 峰值分配显存 `1.23 GiB`、保留显存 `1.77 GiB`，主机 RSS `1.10 GiB`。见[正式实现聚合报告](dicl_incremental_real2050.json)与[阶段复现脚本](benchmark_dicl_incremental_real2050.py)。报告的实现 SHA-256 对应提交 `587944e` 的 `dicl_torch.py`；后续ADMM版本另列下文。

开发期间仅替换 LARS 的首次完整三模态试验耗时 `147.02/11.95/17.86` 秒，随后图捕获覆盖全部原子条件分支的试验为 `189.89/11.87/12.74` 秒。这些是不同时间段的共享节点观测，不能据此决定原子更新图是否加速。正式代码仅在所有原子有效时捕获原有更新操作，未采用覆盖全部条件分支的版本。另试了 `torch.compile` 融合事件，首次设置增加 `14.32` 秒，五个批次相对普通图捕获没有稳定的优势，未加入运行时。

正式测量时 H100 持续有外部作业、利用率为100%；空闲显存由约19.7 GiB下降到17.2 GiB，我们自己的分配峰值仍只有1.23 GiB。旧公开 GPU 运行的 DicL 为871.61秒，本次为238.72秒，观察到耗时减少约72.6%；两次负载和测量边界不同，**这不是受控加速比**。本次 GPU 仍比已测 CPU 慢约44%，没有达到“各阶段明显快于CPU再跑30,000人”的门槛，未启动30,000人试验。

公开 `run_bigflica` 另复用已核对的标准化、mMIGP 和本轮正式代码生成的字典缓存，完成后续输出检查：`2050×3` course、九张原网格 z-stat NIfTI、九张 top-1000 阈值 NIfTI 和九张 PNG 均存在；z NIfTI 与数组在掩膜内逐点相同；另一名未参训真实被试的 `apply_model` 返回三个有限值。此检查显式导入阶段字典缓存，没有重测冷启动总时间。核验脚本初次误把仅包含course值的TSV当作被试清单；修正后从数据目录选择未参训被试，并继续检查已生成的同一模型，没有重复拟合或成图。见[公开缓存与输出报告](dicl_incremental_cache_api_real2050.json)和[复现脚本](check_incremental_cache_api_real2050.py)。

GPU DicL 缓存目录及签名升级为 `rsvd2invgraph`；旧 `rsvd1` 字典不会作为新版本缓存复用，mMIGP 缓存仍可复用。回归测试覆盖混合零输入、图工作区重用、自定义alpha、失效逆矩阵回退，以及未使用原子的随机数顺序和 sklearn 原子更新；本地 PyTorch 2.4.1 的 CPU/CUDA 测试共26项通过。H100/PyTorch2.5.1的精度与内存结论来自上面的真实完整拟合。

### 仍未通过的验收

- **独立端到端一致性**：本轮固定 mMIGP 投影以核验下游，没有消除 CPU/GPU 各自生成投影时的小差异经非凸 DicL 放大的问题；不能把本轮结果写成独立全链等价。
- **20个有效成分与模态重建**：本轮同投影的CPU/GPU C20都只有 `11/20` 个有效成分；VBM重建范数比约 `7.10e-5/7.09e-5`，仍明显弱于FA/MD。此处的CPU输入不同于此前float64公开全链，不能与其16/20结果混为一谈。C3仍只作功能检查。
- **CPU速度门槛**：需要继续减少LARS事件内的GPU小核与同步，进一步检查未使用原子重采样的标量传输；完整阶段胜出后才扩大样本。不能仅凭局部稀疏编码加速宣布全部阶段胜出。

可复现阶段测试需要已经建立的三模态 R100 投影目录和旧GPU字典目录；不把原始整模态载入内存。将CPU基线参数设为 `-` 会重新拟合CPU对照；已有本脚本格式的基线时可传其私密输出目录，报告会明确标注时间复用：

```bash
OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=1 \
  python validation/bigflica/benchmark_dicl_incremental_real2050.py \
  /absolute/private/mmigp_100 \
  /absolute/private/old_gpu_dictionaries \
  - /absolute/private/new_stage_benchmark
```

`new_stage_benchmark`保存私密字典、FLICA日志和诊断，只有不含被试ID的 `aggregate.json` 可上传。输出目录须尚不存在；投影输入与复用CPU基线的SHA-256不符会停止。该脚本还检查C3 course、各模态z图以及C20秩，不生成NIfTI和PNG；公开缓存检查脚本专门核对这些文件和留出被试调用。

### 成熟稀疏编码策略的候选试验

查阅 [SPORCO BPDN](https://sporco.readthedocs.io/en/latest/modules/sporco.admm.bpdn.html) 的批量 ADMM 方程与 [PyTorch Lasso](https://github.com/rfeinman/pytorch-lasso) 的 FISTA、Split Bregman 实现。前者源码为 BSD-3，后者为 MIT 且作者明确标注仍在开发；没有直接将整套库作为运行时依赖，也没有复制发布其源码。验证脚本独立用 PyTorch 实现 ADMM 方程，保留 `0.5*||code@dictionary-data||²+||code||₁` 目标，复用20步 CUDA Graph，字典改变时刷新矩阵分解。

[候选报告](bpdn_probe_real2050.json)来自上述2,050人真实完整掩膜投影，每模态前32个体素，最终字典恢复单位范数约束。`rho=1`、200步的编码相对 LARS 误差为 `1.28e-10/7.57e-11/9.59e-11`，最优性条件最大残差均小于 `2e-14`；候选耗时 `0.0217/0.0220/0.0289` 秒，LARS为 `0.0616/0.0486/0.0436` 秒。图设置另需 `0.072/0.093/0.078` 秒。此处是共享GPU上的单批观测，候选时间包含矩阵分解但不含图设置，不能推断完整训练加速。其他rho在200步时仍有 `1e-6–1e-4` 级误差，说明不能用固定少量迭代直接替换。

此初步候选当时未覆盖初始SVD字典、原子重采样、全部训练批次、最终DicL字典及FLICA脑图，未作为默认求解器。后续的活动集校正和正式运行时验收另列下文。

复现命令（只读取已存在的私密投影和本轮阶段字典）：

```bash
CUDA_VISIBLE_DEVICES=1 python validation/bigflica/benchmark_bpdn_real2050.py \
  /absolute/private/mmigp_100 /absolute/private/new_stage_benchmark \
  /absolute/private/bpdn_probe.json
```

三个参数依次是含 `*_projected.h5` 的目录、含 `*_gpu_dictionary.npy` 的目录、聚合JSON输出路径。不输出被试ID或影像，不替代完整阶段benchmark。

## 2026-09-30：ADMM 活动集校正与回退

在SPORCO BPDN方程基础上，新增批量ADMM识别活动集，再求解带既有 `1e-10` 正则项的活动方程，检查系数符号、最优性条件和LU枢轴。固定rho=1，每20步重放CUDA Graph，最多检查20次；前四个批次、未归一化初始化字典、不可用的图工作区或检查失败时重新运行原LARS。批次、目标函数、初始化、随机数、原子重采样和sklearn停止规则保持原有设置。重复原子的非唯一表示即使通过最优性条件，也会因枢轴检查而回退，避免改变原软件选择的表示。只使用已有PyTorch依赖，未加入SPORCO或PyTorch Lasso运行时。

正式实现再次读取相同的2,050人完整掩膜三模态float32投影，内部使用既有float64精度，与前述已测CPU字典和时间比较。**这是完整DicL阶段对照，未重新做原始NIfTI到mMIGP的独立全链比较。**

| 模态 | CPU sklearn秒（复用同输入实测） | 正式GPU秒 | 字典相对Frobenius误差 | LARS回退 / 小批次 |
|---|---:|---:|---:|---:|
| VBM | 139.05 | 123.35 | 1.0612e-5 | 4 / 2308 |
| FA | 13.38 | 10.24 | 1.5401e-6 | 4 / 111 |
| MD | 13.41 | 10.65 | 6.5626e-5 | 4 / 122 |
| 合计 | **165.84** | **144.23** | — | 12 / 2541 |

每模态四次回退均对应预设早期批次。GPU显存分配峰值1.29 GiB、保留峰值1.81 GiB、主机RSS峰值1.14 GiB。相对旧GPU字典误差 `1.06e-7/3.48e-8/2.83e-8`。沿用同一U运行C3后，course和九张z图在符号校正后相关均超过 `0.9999999984`。正式代码SHA-256为 `fa6f41c3ca7b2a477b986ec2c72482cf748f3e8002318e883fc56b1eb4364b52`，与[聚合报告](dicl_bpdn_real2050.json)完全一致；[阶段脚本](benchmark_dicl_incremental_real2050.py)现针对正式ADMM运行时计数。

这一正式运行的总耗时低于复用CPU基线约13%，但不是受控加速比：GPU利用率持续100%，外部显存占用在VBM期间减少约12 GiB；CPU时间没有在同一时窗重测。稍早的ADMM校正候选完整拟合观察为 `170.85/15.68/17.36` 秒，合计203.89秒，同样通过字典、course和z图精度检查。两次实现及外部负载不同，不能将时间变化全部归因于新增保护检查，也不能声称已达到各阶段稳定、大幅快于CPU的门槛。**未启动30,000人试验。**

公开API使用新版字典与核对过的上游缓存，输出 `2050×3` course、九张z-stat NIfTI、九张top-1000阈值NIfTI和九张PNG；掩膜内z-stat与保存数组逐点相同。另一名未参训真实被试的 `apply_model` 返回三个有限值。该输出检查耗时22.19秒，明确复用了上游和阶段字典，不能当作冷启动总耗时。见[公共缓存与输出报告](dicl_bpdn_cache_api_real2050.json)及[检查脚本](check_incremental_cache_api_real2050.py)。

本地PyTorch2.4.1的CPU/CUDA回归测试共28项通过，新增自定义alpha、混合零输入、工作区复用和重复原子非唯一解回退；H100/PyTorch2.5.1完成正式真实阶段拟合。现有项目Conda依赖不变；与sklearn1.7.1的对照不能写成已用固定sklearn1.5.2环境完成完整重建。新版缓存为 `rsvd3bpdn`，旧DicL不自动复用，mMIGP仍可复用。

仍待完成：独立CPU/GPU的mMIGP→DicL→FLICA全链一致性；C20有效成分与VBM重建（本轮CPU/GPU仍为11/20，VBM重建范数比约7.10e-5/7.09e-5）；持续受控的完整阶段速度验收。本轮改善稀疏求解，不改变FLICA模型或放宽退化成分检查。
