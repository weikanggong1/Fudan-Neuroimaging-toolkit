# dMRI 参数图流程源码目录

`DMRIPipeline.run()` 处理一名受试者的 UKB 格式 AP/PA 数据；`run_bids()` 从原始 BIDS 的 `dwi/`、可选 `fmap/` 和 `anat/` 选择同一被试数据，再调用相同的 TorchTOPUP、TorchEDDY、TorchDTIFIT、TorchAMICONODDI 和 TBSS/FNIRT 或 T1+tensor MMORF 计算链。`noddi_fit_method="classic"` 可选连续 Watson 拟合，默认 `"amico"`。TBSS 不依赖 T1w；MMORF 需要 T1w。`bids.py` 只解析输入并建立内部链接；运行时不启动 FSL、FreeSurfer、AMICO 或 NODDI Toolbox 程序。

单被试 Python 与命令行示例、每项输入及输出文件的定义见[功能页](../../../docs/dmri_pipeline/README.md)。[当前真实数据验证](../../../validation/dmri_pipeline/README.md)分别记录两条分支的九图精度、标准空间图、阶段时间及内存，并明确区分匹配 FSL EDDY 输入的下游对照与官方 FSL 配准对照。不同起点的参考结果不能合并为端到端等价结论。
