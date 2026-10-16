# FNIT 用户文档

- [统一 Python API](API_REFERENCE.md)：`FNIT_*` 主入口、类与函数的调用方式、模式选项及兼容接口。
- [功能导航与安装](../README.md)：按任务选择功能，配置 Conda 环境和运行资源。
- [最新 benchmark 索引](BENCHMARK_INDEX.md)：进入各功能的真实精度、耗时和脑图报告。
- [资源安装](ASSETS.md)：固定 Release、权重、模板、SHA-256 和许可边界。
- [代码结构](ARCHITECTURE.md)：公共接口与内部模块的关系。

## 阅读功能页

每个功能页依次说明：功能；Python 调用、输入和输出；CLI；原软件对应命令；当前真实精度、时间和脑图；近期变化；参考与许可。pipeline 另含流程图。

主要入口统一从 `fnit` 导入，名称以 `FNIT_` 开头。配置、结果类型和低层算子保留高级兼容接口。MSMSulc/MSMAll 通过 `FNIT_newmsm` 选择，AMICO/classic NODDI 通过 `FNIT_noddi` 选择。

真实数据报告保留原测量版本。文档中的通用目录、环境标签用于说明文件结构，不是原实验服务器的连接信息。[隐私整理记录](../validation/documentation/public_api_20261016/README.md)区分原始文件身份与发布副本身份。
