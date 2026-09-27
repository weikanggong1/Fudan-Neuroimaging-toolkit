# Python recon-all 验证状态

[同一 T1 整例对照和计时](python_gpu_port/NATIVE_FREE_CONNECTED_20260927.md)是当前 `fnit-recon-all` 入口的主要证据。Python 核心流程跑通，但严格 FreeSurfer 8.2 固定配置仅 **6/138** 项一致，52 项缺失，80 项有差异。候选耗时 1805.7 秒；归档官方流程 6789.6 秒。输出尚不等价，两个时长不构成等价流程加速结论。

[差异根因、逐点与数值容差及验收顺序](python_gpu_port/DISCREPANCY_AND_TOLERANCE_20260927.md)区分当前整例路径和冻结官方输入下的独立阶段验证。

[comparison.json](python_gpu_port/native_free_sub01_20260927/comparison.json)含逐体素、逐表面、逐顶点指标及逐脑区差异；[strict_138.json](python_gpu_port/native_free_sub01_20260927/strict_138.json)列出固定配置全部输出；[run.json](python_gpu_port/native_free_sub01_20260927/run.json)与[benchmark.json](python_gpu_port/native_free_sub01_20260927/benchmark.json)保留哈希和阶段时间。原生与 Python 的独立阶段配对结果见[移植记录](python_gpu_port/README.md)，完整验收要求见[发布门槛](python_gpu_port/RELEASE_GATES.md)。

较早的混合运行包试验报告属于历史验证材料，不代表当前 Python 入口的数值状态。主流程不需要这些运行包或 FreeSurfer license。
