# dMRI 参数图流程源码目录

`DMRIPipeline.run()` 处理一名受试者的 UKB 格式 AP/PA 数据；`run_bids()` 从原始 BIDS 的 `dwi/`、可选 `fmap/` 和 `anat/` 选择同一被试数据，再调用相同的 TorchTOPUP、TorchEDDY、TorchDTIFIT、TorchAMICONODDI 和 TBSS/FNIRT 或 T1+tensor MMORF 计算链。`noddi_fit_method="classic"` 可选连续 Watson 拟合，默认 `"amico"`。TBSS 不依赖 T1w；MMORF 需要 T1w。`bids.py` 只解析输入并建立内部链接；运行时不启动 FSL、FreeSurfer、AMICO 或 NODDI Toolbox 程序。

单被试 Python 与命令行示例、输入和输出定义见[功能页](../../../docs/dmri_pipeline/README.md)。[真实数据验证](../../../validation/dmri_pipeline/README.md)记录 BIDS 无 T1w 的 TBSS 1 mm 整链、带 T1w 的 MMORF 2 mm 整链、1 mm MMORF 在共享 GPU 上的显存失败边界，以及既有 UKB TBSS 与 FSL 的九图数值对照；不同起点和网格的参考结果不能合并为端到端等价结论。

2026-10-02 起，两条配准分支均使用 PyTorch SynthStrip 的 b0 脑 mask：有 PA 时输入 TOPUP 校正 AP/PA b0 的 float32 均值，AP-only 时只平均原 AP 的 `b<100` volume。`synthstrip_weights=` / `--synthstrip-weights` 指定标准 PT 文件或目录，None 使用 FNIT 本地解析；首次加载校验标准大小与 SHA-256，实例内复用模型，MMORF 的 T1 也使用该实例。`eddy/nodif_brain_mask_report.json` 和总报告 `brain_mask` 记录输入、权重、原网格、CPU/GPU、计时及 allocator 峰值范围。

本轮合并新 main 前完成的独立整链复测，以同一真实 `104×104×72×105` raw AP/PA 分别生成全部 27 图。主要参考为官方 TOPUP、其自身 b0 均值的官方 CPU SynthStrip、FSL GPU EDDY、FSL DTIFIT／官方 Python AMICO／FSL TBSS；两侧各自使用 EDDY 旋转梯度。双方各 8 CPU 线程，参考 CPU 0–7、FNIT CPU 8–15，H100 GPU 1 阶段串行；参考 CPU 配准与 FNIT GPU 阶段重叠。清理版 FNIT／参考处理时间为 499.55／2055.53 s，完整进程 503.37／2073.54 s；计时包含输入处理和最终写盘，启动与报告统计另列。合并后的新源码需以其独立 fresh 运行绑定结果。

主要协议的 mask Dice=0.9999834，native FA／MD r=0.999051／0.999642，固定模板脑区 standard 九图 r=0.988013–0.998676，固定模板 skeleton 九图 r=0.993251–0.998960。另列最终 FNIT 对历史 FSL BET 链的标准九图 r=0.846458–0.959704；两协议不合并为数值等价结论，旧阈值/BET 范围也不能用于隔离单项修复的因果效应。27 对图几何和有限值检查全部通过；最终 TOPUP core／sampler SHA-256 以 `d6b9838c`／`ee19a764` 开头，清理前与最终 FNIT 的 27 张解码数组及 header binary block 相同。整链 allocator 已分配／保留峰值为 14.65145／19.98166 GB，设定预算为 20,000,000,000 bytes，排除 context 和其他进程。[最新整链报告与脑图](../../../validation/dmri_pipeline/end_to_end_synthstrip_topup_20261002.md)记录真实分步骤时间、误差、完整版本和哈希；旧阈值/BET 报告保留为独立历史记录。

相同真实官方 TOPUP b0 均值的独立 SynthStrip 核验已完成：FNIT H100 GPU 与官方 CPU mask Dice=0.9999907776，差 5 voxel，shape/affine 一致，见[固定输入结果](../../../validation/dmri_pipeline/synthstrip_fixed_input_20261002.public.json)。模型加载＋推理与包含冷 NFS 读取的完整进程分别计时，完整边界见功能页；不据此计算整链加速比。[53 项 CPU 回归](../../../validation/dmri_pipeline/synthstrip_cpu_tests_20261002.public.json)在受支持 Python 3.11 环境通过。
