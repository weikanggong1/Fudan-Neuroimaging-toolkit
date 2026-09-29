# dMRI 参数图流程源码目录

`DMRIPipeline.run()` 处理一名受试者的 UKB 格式 AP/PA 数据；`run_bids()` 从原始 BIDS 的 `dwi/`、可选 `fmap/` 和 `anat/` 选择同一被试数据，再调用相同的 TorchTOPUP、TorchEDDY、TorchDTIFIT、TorchAMICONODDI 和 TBSS/FNIRT 或 T1+tensor MMORF 计算链。`noddi_fit_method="classic"` 可选连续 Watson 拟合，默认 `"amico"`。TBSS 不依赖 T1w；MMORF 需要 T1w。`bids.py` 只解析输入并建立内部链接；运行时不启动 FSL、FreeSurfer、AMICO 或 NODDI Toolbox 程序。

单被试 Python 与命令行示例、输入和输出定义见[功能页](../../../docs/dmri_pipeline/README.md)。[真实数据验证](../../../validation/dmri_pipeline/README.md)记录 BIDS 无 T1w 的 TBSS 1 mm 整链、带 T1w 的 MMORF 2 mm 整链、1 mm MMORF 在共享 GPU 上的显存失败边界，以及既有 UKB TBSS 与 FSL 的九图数值对照；不同起点和网格的参考结果不能合并为端到端等价结论。
