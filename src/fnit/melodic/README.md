# MELODIC 源码目录

单被试空间 PICA。`ica.py` 实现方差归一化、Laplace PPCA 自动定阶、对称 pow3 ICA、Gaussian/Gamma 混合模型和后验阈值；`bids.py` 负责单 run 的 BIDS Derivatives 输入与输出。fMRI volume 调用同一个 `decompose_spatial_ica` 内核，运行时不启动 FSL。

输入为预处理后的四维 BOLD 和同网格三维脑掩膜。输出是 K 张空间成分、后验图、阈值图、T×K 时间序列和来源 JSON；路径接口为 `run_melodic_bids`。数据读写使用 nibabel，计算使用 PyTorch。PCA/ICA 采用 float64，其余函数的默认 TF32 设置保持不变。

本轮修复了额外空间去均值、PPCA 尾谱、随机初始化、输出尺度/排序及 Gamma 后验尾部裁剪。固定一例真实 490 帧原 FEAT 输入与掩膜，原 MELODIC 和 FNIT 都得到 95 个成分、40 步迭代；配对后时间相关中位数 0.999999978，空间相关中位数 0.999999970，阈值支持 Dice 中位数 0.999562。完整 pipeline 的输入由各自前处理生成，结果另行比较。

全部参数、输入输出结构、单被试示例、原 MELODIC 命令、真实脑图和文献见 [专属说明](../../../docs/melodic/README.md)；[固定输入报告](../../../validation/fmri/ica_fixed_input.public.json)；[完整流程对照](../../../validation/fmri/matched_native.md)。
