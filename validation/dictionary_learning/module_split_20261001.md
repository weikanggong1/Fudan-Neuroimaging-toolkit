# 字典学习独立模块拆分记录

字典学习现属于 Post analysis，代码位于 `fnit.dictionary_learning`。BigFLICA 的 CPU、CPU 流式流程及 CUDA 流程调用该模块，原参数、字典输出和缓存版本保持一致。

## 接口与文件

| 内容 | 当前位置 |
|---|---|
| CPU 数组拟合 | [cpu.py](../../src/fnit/dictionary_learning/cpu.py)：`fit_dictionary_learning`，兼容名 `fit_dicl` |
| GPU HDF5 流式拟合 | [torch_backend.py](../../src/fnit/dictionary_learning/torch_backend.py)：`fit_dictionary_learning_streaming`，兼容名 `fit_dicl_gpu_streaming` |
| 参数与调用示例 | [独立功能页](../../docs/dictionary_learning/README.md) |
| 效果、速度和复现 | [独立验证索引](README.md)与 [benchmark_dicl_match.py](benchmark_dicl_match.py) |

旧 `src/fnit/bigflica/dicl_torch.py` 已移走；BigFLICA 中原 CPU 拟合实现已移到独立模块。`fnit.bigflica.fit_dicl` 保留为重导出，既有调用仍使用同一函数。字典学习模块可直接导入，不加载 BigFLICA。完整 FLICA、C20、脑图和模型对照仍在 [BigFLICA 验证目录](../bigflica/README.md)。

## 核对计算没有变化

对照拆分前提交 `571fc9ea57551bb55f89f23f65176f3b0d2c7b10`：

- CPU `fit_dicl` 的函数源码逐字一致。
- GPU 整个模块的语法树一致，仅将对 BigFLICA 的设备选择导入改为相同函数，并增加共享的缓存版本常量。核对包含条件内的 Triton 核、类、初始化、随机数与停止规则。
- GPU 字典缓存版本仍为 `rsvd5rowgraph`，通过原输入签名检查的缓存可复用。
- 新入口使用项目已有依赖，无需下载权重或模板。

原 GPU 核心 SHA-256 为 `56735ea2ecd7543e8078a51c9a772f1db3d55fdfed132194015f8982ed76474f`；独立 GPU 文件为 `6f225f0ac9727934fe6fd2f298d586e7aea178a5e8db756c63e7acc8a8e72a99`，CPU 文件为 `b6a4cde5d0b51dbb51bc8253cbdf816a3076103eb7c88cf68db53795eff09f97`。文件路径及导入改变会改变文件哈希，计算内容保持一致。[机器可读记录](module_split_20261001.json)保存对应字段。

## 回归结果

**152 passed，52.59秒，无跳过。**本地环境为 RTX3060、PyTorch2.4.1、NumPy1.26.4、sklearn1.2.2。覆盖 CPU/CUDA 稀疏求解、字典更新、缓存与流复用、独立导入及别名、CPU 参考模块调用边界、BigFLICA 参数与输出检查；测试夹具用于回归验证。

```bash
PYTHONPATH=src python -m pytest -q \
  tests/test_dictionary_learning* \
  tests/test_bigflica.py \
  tests/test_bigflica_normalization_and_design.py \
  tests/test_bigflica_raw_routing.py
```

独立 benchmark 和保留的两个 BigFLICA 集成运行器已通过语法检查，独立 benchmark 的 CLI 参数检查通过。

## 真实数据证据沿用范围

此次拆分保持计算原样，沿用此前同一1000人、VBM/FA/MD、float64 R500/D200 的真实验收，未重复整套 benchmark。此前最终 GPU 完整拟合观测为63.65秒，分配显存峰值2.31 GiB；各模态字典与 LASSO 门槛通过，FA/MD 的 OMP30 重建差仍为2.44%/5.33%。全部测量、来源哈希与共享负载边界见[原速度报告](dicl_speed_optimization_real1000_20261001.md)。

独立报告保留实际测量时的源码和运行器哈希；新的公开运行器记录当前独立 CPU/GPU 文件来源。模块拆分没有改变 float32 默认入口、OMP30、压缩 C20 及最终脑图的剩余验收状态。
