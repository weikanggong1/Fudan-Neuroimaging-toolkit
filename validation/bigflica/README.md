# BigFLICA：真实 UKB 输入、GPU 对照与内存测试

2026-09-30 在用户指定的 UKB 多模态目录核验，共有 87,720 个被试目录；其中 37,182 个同时含 `VBM_2mm.nii.gz`、`dti_FA_2mm_mmorf.nii.gz`、`dti_MD_2mm_mmorf.nii.gz` 和 `zstat1s.nii.gz`。抽查影像为 `91×109×91`、2 mm MNI 网格。任务图文件是 **zstat**，不是 tstat。

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
| FNIT 当前 CPU：读取/标准化、mMIGP、sklearn DicL、FLICA、成图、投影模型 | 1.14 / 0.195 / 5.08 / 0.457 / 2.72 / 0.053 | 冷启动带全部输出墙钟 9.71；[当前源码重跑](cpu_final_metadata.json)。 |
| GPU 直接体素：读取/标准化、FLICA、成图、投影模型 | 1.28 / 43.74 / 3.43 / 0.213 | 墙钟 48.76；进程峰值显存 0.157 GiB、内存 0.882 GiB；原 CPU 直接体素 FLICA 同输入为 2.39 秒。 |
| GPU 压缩、默认 batch=32：读取、mMIGP、DicL、FLICA、成图、投影模型 | 1.37 / 1.12 / 474.47 / 7.15 / 3.67 / 0.189 | 墙钟 488.10；显存峰值 0.156 GiB、内存 0.985 GiB；[完整结果](gpu_default32_final.json)。 |
| GPU 压缩、batch=32，仅 DicL 四模态 | 486.33 | 符号对齐实验；VBM 单模态 119.46 秒，对照 CPU 1.52 秒。当前 PyTorch LARS 在小批量上明显慢于 sklearn。 |
| GPU 压缩、batch=1024：读取、mMIGP、DicL、FLICA、成图、投影模型 | 1.43 / 2.73 / 153.07 / 7.29 / 3.43 / 0.211 | 墙钟 168.32；显存峰值 0.156 GiB、内存 0.977 GiB。批次不同于 notebook，不作精度等价结论；[完整运行](gpu_canonical_full.json)。 |
| 上一行相同输入、参数与缓存，仅将成分数改为 2 | 9.33 总计 | `mmigp_reused=true`、`dicl_reused=true`；再次运行 FLICA、成图和投影模型。 |
| 当前源码入口，GPU 压缩、batch=4096；再改为两个成分 | 33.23 / 9.36 总计 | 冷启动生成 12 张 z 图；再次运行复用 mMIGP/DicL。此大批次仅核对最新代码路径，不用于 notebook 精度比较；[运行记录](gpu_final_smoke.json)。 |

**2,050 名真实被试分块测试**使用 VBM 和 FA 两个模态、同一核验掩膜：标准化 HDF5 实际数据 311,747,600 字节，逐被试建库 68.39 秒；mMIGP 2.17 秒，进程峰值显存 0.125 GiB、内存 0.743 GiB。初版 6 次 GPU 子空间迭代的特征对残差为 3.47×10⁻⁴；新增自适应迭代在第 18 次达到 3.03×10⁻⁹，mMIGP 2.83 秒，显存峰值 0.095 GiB，见[分块基准](large2050.json)和[改进结果](large2050_refined.json)。测试覆盖了 `N>2048` 的随机特征分解分支，但这两个模态在所选小掩膜下本就可以放入 32 GB 内存；“完整大模态不进内存”由逐被试 HDF5 写入及逐体素块读取实现，**未**做 37,182 人 × 全脑体素的端到端耗时实测。按 100 万掩膜体素估算，float64 规范化缓存约 277 GiB/模态，实际磁盘容量必须单独安排。

直接体素模式在大于 2,048 人时使用随机特征分解，并将原 FLICA 的自动空间自由度估计固定为 1；此大样本分支尚未完成与原软件的精度对照。大样本端到端 CPU/GPU 加速比尚不能报告。真实 UKB 派生 PNG 从远程复制到本地工作区曾被自动审批拒绝，理由是可能含敏感数据；本仓库不附脑图，原图保留在用户远程结果目录。
