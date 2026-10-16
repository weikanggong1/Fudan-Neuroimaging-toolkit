# 代码结构与公共接口

[返回首页](../README.md) · [统一 Python API](API_REFERENCE.md) · [新增功能](ADDING_FUNCTIONS.md)

所有主要处理入口从 `fnit` 按需导入，名称以 `FNIT_` 开头；完整清单、调用方式及兼容说明以[统一 Python API](API_REFERENCE.md)为准。模型类保留构造后复用的方式，文件函数保留原参数合同。结果类型、配置对象和细分算子继续位于各功能模块。

## 源码与文档

| 目录 | 内容 |
|---|---|
| `src/fnit/` | 按功能组织的实现；根 `__init__.py` 按需公开统一入口。 |
| `src/fnit/cli.py` | 共享命令行解析及分派；功能自己的 CLI 见对应手册。 |
| `docs/<功能>/README.md` | 功能、Python/CLI、输入输出、原软件、当前验证和参考。 |
| `tests/<功能>/` | 数值、几何、参数、缓存与输出合同检查。 |
| `validation/<功能>/` | 绑定实际运行版本的真实数据报告、历史记录和可公开脑图。 |
| `assets/`、`licenses/` | 资源清单、许可和再分发边界；模型权重不写入源码树。 |

## 处理与资源

- 通用影像运算、线性和非线性配准以成熟子函数组合，pipeline 负责流程与缓存。
- fMRI volume、surface、dMRI、VBM 和 connectome 的流程图及实际输入范围见各功能页。
- connectome 的 iFOD2/ACT 使用 FNIT 管理的固定 MRtrix 源码构建缓存；其余组件保持各自已实现路径。首次构建和离线复用见[追踪手册](connectome/TRACKING_OPERATORS.md)。
- recon-all 的当前生产路径、仍存在的差异及原始 T1 整例记录见[重建手册](recon_all/README.md)；不能用阶段结果代替整例等效。
- 权重按显式路径、`FNIT_WEIGHTS`、配置目录和用户缓存定位，不自动读取系统脑影像软件中的模型。固定 Release、大小、SHA-256 与原站下载例外见[资源安装](ASSETS.md)。

## 命名与验证

本轮统一主要 Python 入口和文档命名，不改变配准、分割、推断、追踪或统计定义。MSMSulc/MSMAll 合并为 `FNIT_newmsm` 的模式选项，AMICO/classic NODDI 合并为 `FNIT_noddi` 的方法选项。

[benchmark 索引](BENCHMARK_INDEX.md)只提供报告导航；每条数值仍绑定原运行源码、输入、设备及计时范围。接口改名与隐私整理不代表重新执行 MRI benchmark。
