# Connectome 模块

用户入口、BIDS 输入、单/多 atlas 命令、输出结构和真实数据对照集中在[功能文档](../../../docs/connectome/README.md)。

`UKBConnectome_pipeline.run_bids()` 处理原始 BIDS DWI/T1；`UKBConnectome_pipeline.__call__()` 接收已校正 DWI。`pipeline.py` 负责响应、FOD、解剖、追踪与多 atlas 矩阵；`bids.py` 负责选片及 TOPUP/EDDY/recon-all 阶段复用。逐算子的独立验证保留在 [`validation/connectome/`](../../../validation/connectome/) 中。

当前 `tracking.py` 调用 FNIT 从固定 MRtrix 源码独立构建的 CPU `tckgen`，不查找 PATH 中的 MRtrix；其余成熟 PyTorch 阶段继续使用所选设备。输入、参数、程序校验和安装见[追踪算子说明](../../../docs/connectome/TRACKING_OPERATORS.md)，本版真实整链与官方比较见[原生追踪评测](../../../validation/connectome/native_tracking_20261009/README.md)。
