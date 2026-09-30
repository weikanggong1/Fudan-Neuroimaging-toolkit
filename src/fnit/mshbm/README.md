# MS-HBM：fsLR32k 单被试 17 网络划分

公开入口为 `fnit.mshbm` 和 `fnit-mshbm`。完整中文说明、输入输出结构、具名 Python 示例、命令行参数、CBIG MATLAB 对应关系、真实数据计时与脑表面对照图见 [`docs/mshbm/README.md`](../../../docs/mshbm/README.md)。

本目录只保留当前实现：`core.py` 为 NumPy/SciPy CPU 推断，`cli.py` 负责单被试入口，`volume.py` 负责 MNI↔fsLR32k 投影，`output.py` 写标签和网络 TSV，`assets_setup.py` 从固定 CBIG 原站部署投影资源，`assets/hcp40_fslr32k_17.npz` 为 CBIG HCP_40 固定资产。
