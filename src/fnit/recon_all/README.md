# Python recon-all 模块

公开单被试入口：`fnit.recon_all.native_free.run_recon_all_python(...)`，命令行为 `fnit-recon-all`。多被试并行入口：`fnit.recon_all.run_recon_all_python_batch(...)`，仅提供 Python API。两者使用外置权重和模板，不调用 FreeSurfer 可执行程序。

当前整例输出是近似版；拓扑、表面放置与球面配准尚未与官方对齐。一个 T1 的严格比较通过 6/138 项。安装、参数、输出、设备分配和已测量差异见[完整说明](../../../docs/recon_all/README.md)与[整例报告](../../../validation/recon_all/python_gpu_port/NATIVE_FREE_CONNECTED_20260927.md)。
