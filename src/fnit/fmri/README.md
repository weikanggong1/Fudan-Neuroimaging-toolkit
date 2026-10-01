# fMRI 单 run 处理

体积接口 `fMRIVolume_pipeline` 从原始 BIDS BOLD、SBRef 和 T1w 生成个体 EPI 与 MNI152 2 mm 清理图、脑掩膜、BBR 矩阵和 provenance JSON。运动校正调用独立 `TorchMCFLIRT`；组织估计调用 `TorchFAST(execution="fsl")`。

输入、输出、逐项参数、Python/命令行示例与原软件命令见 [体积说明](../../../docs/fmri/README.md)；[真实全流程对照](../../../validation/fmri/README.md)。表面后续处理见 [surface 说明](../../../docs/fmri/surface.md)。
