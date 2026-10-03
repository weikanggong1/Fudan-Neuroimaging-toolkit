# dMRI 参数图流程源码目录

`DMRIPipeline.run()` 处理一名受试者的 UKB 格式 AP/PA 数据；`run_bids()` 从原始 BIDS 的 `dwi/`、可选 `fmap/` 和 `anat/` 选择同一被试数据，再调用相同的 TorchTOPUP、TorchEDDY、TorchDTIFIT、TorchAMICONODDI 和 TBSS/FNIRT 或 T1+tensor MMORF 计算链。`noddi_fit_method="classic"` 可选连续 Watson 拟合，默认 `"amico"`。TBSS 不依赖 T1w；MMORF 需要 T1w。`bids.py` 只解析输入并建立内部链接；运行时不启动 FSL、FreeSurfer、AMICO 或 NODDI Toolbox 程序。

单被试 Python 与命令行示例、输入和输出定义见[功能页](../../../docs/dmri_pipeline/README.md)。[真实数据验证](../../../validation/dmri_pipeline/README.md)记录 BIDS 无 T1w 的 TBSS 1 mm 整链、带 T1w 的 MMORF 2 mm 整链、1 mm MMORF 在共享 GPU 上的显存失败边界，以及既有 UKB TBSS 与 FSL 的九图数值对照；不同起点和网格的参考结果不能合并为端到端等价结论。

公开十人双分支最新状态与完整复现见[固定队列验证](../../../validation/dmri_pipeline/public10_20261002/README.md)。冻结 `bf339a0` 的 20 次 FNIT raw 整链全部完成，已比较 10/10 TBSS 和 9/10 MMORF，共 432/450 张图通过几何和有限值检查。case10 MMORF 的原 GPU EDDY 初次、R1、R2 完整尝试均报 CUDA 分配失败，六次原参考失败时钟全部保留。AMICO 内存修复只改变临时矩阵调度；不能把完成标记或较短耗时解读为数值等价，也不能把冻结队列重标为合并后的 main 运行。

2026-10-02 起，两条配准分支均使用 PyTorch SynthStrip 的 b0 脑 mask：有 PA 时输入 TOPUP 校正 AP/PA b0 的 float32 均值，AP-only 时只平均原 AP 的 `b<100` volume。`synthstrip_weights=` / `--synthstrip-weights` 指定标准 PT 文件或目录，None 使用 FNIT 本地解析；首次加载校验标准大小与 SHA-256，实例内复用模型，MMORF 的 T1 也使用该实例。`eddy/nodif_brain_mask_report.json` 和总报告 `brain_mask` 记录输入、权重、原网格、CPU/GPU、计时及 allocator 峰值范围。

有 PA 的 UKB TOPUP 准备仍裁去奇数 z 输入的末片；EDDY 准备会在 iout 与原 AP 网格不一致时直接报错。因此该组合当前要求偶数 z，公开十人数据为 68 片，前一独立单例为 72 片；AP-only 分支不执行 TOPUP 裁剪。原软件完整 driver 的主要对照需显式指定 `--brain-extractor synthstrip` 和官方入口、权重；默认 `bet` 保留为历史协议，完整调用见功能页。

2026-10-02 的 `b3ccafe0` 集成版已经从同一真实 `104×104×72×105` raw AP/PA 完整重跑，在新目录生成全部 27 图。主要参考复用本轮已独立完成的官方 TOPUP、其自身 b0 均值的官方 CPU SynthStrip、FSL GPU EDDY、FSL DTIFIT／官方 Python AMICO／FSL TBSS 结果；同 raw、官方环境和参考协议未变，两侧各自使用 EDDY 旋转梯度。双方各 8 CPU 线程，参考 CPU 0–7、FNIT CPU 8–15，本任务 H100 GPU 1 阶段串行；最新 main 在参考完成后运行，没有本任务时间重叠。当前 FNIT／参考处理时间为 404.74／2055.53 s，完整进程 408.54／2073.54 s；计时包含输入处理和最终写盘，启动与报告统计另列。

主要协议的 mask Dice=0.9999834，native FA／MD r=0.999051／0.999642，固定模板脑区 standard 九图 r=0.988013–0.998676，固定模板 skeleton 九图 r=0.993251–0.998960。另列当前 FNIT 对历史 FSL BET 链的标准九图 r=0.846458–0.959704；两协议不合并为数值等价结论，旧阈值/BET 范围也不能用于隔离单项修复的因果效应。当前精度报告重新计算，27 对图几何和有限值检查全部通过；433 个实际 Python 源文件与 `b3ccafe0a48f4c1c396ae6285d0bdbe07a7664c9` 全部匹配。TOPUP core／sampler SHA-256 仍以 `d6b9838c`／`ee19a764` 开头，集成版与合并前清理版的 27 张解码数组及 header binary block 相同。整链 allocator 已分配／保留峰值为 14.65145／19.98166 GB，设定预算为 20,000,000,000 bytes，排除 context 和其他进程。[此前单例整链报告与脑图](../../../validation/dmri_pipeline/end_to_end_synthstrip_topup_20261002.md)记录真实分步骤时间、误差、完整版本和哈希。

合并前清理前／清理版处理时间分别为 516.31／499.55 s，完整进程为 521.75／503.37 s；只在这些早期运行中参考 CPU 配准与 FNIT GPU 阶段有重叠。不同源码历史单列，不合并时间，也不把共享系统中的时间差全部归于 FNIRT／ApplyWarp 更新。旧阈值/BET 报告继续保留独立范围。

相同真实官方 TOPUP b0 均值的独立 SynthStrip 核验已完成：FNIT H100 GPU 与官方 CPU mask Dice=0.9999907776，差 5 voxel，shape/affine 一致，见[固定输入结果](../../../validation/dmri_pipeline/synthstrip_fixed_input_20261002.public.json)。模型加载＋推理与包含冷 NFS 读取的完整进程分别计时，完整边界见功能页；不据此计算整链加速比。[53 项 CPU 回归](../../../validation/dmri_pipeline/synthstrip_cpu_tests_20261002.public.json)在受支持 Python 3.11 环境通过。

另保留合并 main 带入的 FNIRT 优化历史：原 mask/TOPUP 的完整真实 dMRI 流程对冻结 `7473452` 的 66 项科学输出检查通过，包含重新估计 FNIRT 和九图传播；FastVBM、volume 同期单独核验。其输入、计时和科学 QC 见[统一验证页](../../../validation/registration_lossless_20261002/README.md)，不并入新 SynthStrip 协议的精度或时间。
