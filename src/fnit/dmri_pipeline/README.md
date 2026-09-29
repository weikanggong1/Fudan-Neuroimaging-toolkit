# dMRI 参数图流程源码目录

`DMRIPipeline` 处理一名受试者的 UKB 格式 AP/PA 数据：TorchTOPUP、TorchEDDY、TorchDTIFIT、TorchAMICONODDI，然后选择 TBSS/FNIRT 或 T1+tensor MMORF 配准。运行时只调用 FNIT 已实现的函数，不启动 FSL、FreeSurfer 或 AMICO 程序。

单被试 Python 与命令行示例、每项输入及输出文件的定义见[功能页](../../../docs/dmri_pipeline/README.md)。[当前真实数据验证](../../../validation/dmri_pipeline/README.md)分别记录两条分支的九图精度、标准空间图、阶段时间及内存，并明确区分匹配 FSL EDDY 输入的下游对照与官方 FSL 配准对照。不同起点的参考结果不能合并为端到端等价结论。
