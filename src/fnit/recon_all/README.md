# Python recon-all 模块

单被试入口：`fnit.recon_all.native_free.run_recon_all_python(...)`，命令为 `fnit-recon-all`。多被试并行入口：`fnit.recon_all.run_recon_all_python_batch(...)`，仅提供 Python API。两者读取外置权重与模板，并调用 Conda 从固定 FreeSurfer 源码编译的少数 C++ 程序，无须安装官方 FreeSurfer 运行包。

默认重建仍为近似版本。可选 `native_white_preaparc=True` / `--native-white-preaparc` 接入 CPU MNI/辅助分割、`brain.finalsurfs`、Conda 预白质放置及 Python 三轮 `smoothwm`；需 `native_topology=True` 和两张可选 MNI152 图像。最终 `white/pial` 与皮层统计仍未通过整例一致性验收。最新完整 T1 重建 v3 的严格 138 项比较通过 19 项。安装、参数和输出见[完整说明](../../../docs/recon_all/README.md)及[整例报告](../../../validation/recon_all/python_gpu_port/native_cpp_conda_20260927/v3_e2e_20260927/BENCHMARK.md)。
