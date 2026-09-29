# Connectome 模块

用户入口、BIDS 输入、单/多 atlas 命令、输出结构和真实数据对照集中在[功能文档](../../../docs/connectome/README.md)。

`UKBConnectome_pipeline.run_bids()` 处理原始 BIDS DWI/T1；`UKBConnectome_pipeline.__call__()` 接收已校正 DWI。`pipeline.py` 负责响应、FOD、解剖、追踪与多 atlas 矩阵；`bids.py` 负责选片及 TOPUP/EDDY/recon-all 阶段复用。逐算子的独立验证保留在 [`validation/connectome/`](../../../validation/connectome/) 中。
