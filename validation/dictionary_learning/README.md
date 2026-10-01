# 字典学习：独立代码与验证

字典学习属于 Post analysis，可单独处理数组或 HDF5 矩阵。CPU 入口为 `fnit.dictionary_learning.fit_dictionary_learning`，GPU 入口为 `fit_dictionary_learning_streaming`；兼容名称 `fit_dicl` 与 `fit_dicl_gpu_streaming` 指向相同函数。BigFLICA 通过这些入口调用，mMIGP、FLICA、脑图回归和新被试模型仍由 BigFLICA 管理。

[功能说明](../../docs/dictionary_learning/README.md)记录输入、输出、每个参数和完整示例；[CPU 源码](../../src/fnit/dictionary_learning/cpu.py)及[PyTorch 源码](../../src/fnit/dictionary_learning/torch_backend.py)是唯一实现。GPU 入口接受每模态一个 `*_projected.h5`，其中 `data` 为样本×特征二维矩阵，不要求输入来自 mMIGP；输出为每模态的原子×特征字典。FNIT 的每原子去均值和整体 RMS 归一化属于该函数输出合同。

## 拆分检查

本次移动代码和报告不改变字典学习数学。[模块拆分记录](module_split_20261001.md)及[机器可读证明](module_split_20261001.json)核对旧函数与新函数、CPU 源码、独立导入、公开别名和 BigFLICA 调用关系。下面的真实数据耗时与精度来自拆分前实测，不写成此次重新运行。

## 独立阶段证据

| 范围 | 数据与结论 | 报告 |
|---|---|---|
| sklearn 效果匹配 | 同一1000人 VBM/FA/MD 完整掩膜、float64 R500/D200；独立初始化的字典与 LASSO 通过原容差；FA/MD OMP30 重建仍未通过 | [说明](dicl_match_real1000_20261001.md)、[数值](dicl_match_real1000_20261001.json)、[32批误差轨迹](dicl_match_trace_20261001.png) |
| 调度与字典更新优化 | 同一缓存投影，敏感行回退、分段 CUDA 图和顺序更新融合；最终 GPU 完整拟合观测63.65秒、分配显存峰值2.31 GiB；效果保持上述范围 | [说明](dicl_speed_optimization_real1000_20261001.md)、[数值](dicl_speed_optimization_real1000_20261001.json) |

原 CPU 三模态参考为103.90秒，历史 GPU 精度版为141.29秒。参考与候选位于不同时间窗，GPU 有共享负载，CPU/GPU 投影读取及观察器边界不同；这些耗时不能视为受控硬件加速比。六组真实 checkpoint 更新配对的融合版本输出逐位一致，局部流计时中位数12.87→6.81毫秒，不能外推为全流程同等提速。

报告继续保留实际测量时的旧路径、SHA-256、源码版本和测试名称，作为来源记录。当前源文件位于 `src/fnit/dictionary_learning/`，回归测试位于 `tests/test_dictionary_learning_*.py`。JSON 中的旧源码路径属于历史测量键，不是当前运行时导入。

## 复现独立阶段

[benchmark_dicl_match.py](benchmark_dicl_match.py)提供 `cpu`、`gpu`、`eval` 三个独立阶段，只调用字典学习模块，不导入 BigFLICA。脚本输入为三个 float64 的体素×500 HDF5 投影；参数、私密输出结构、冻结 CPU 参考复用、共同初始化和评估样本控制见[效果报告的复现章节](dicl_match_real1000_20261001.md#复现与参考实现)。精确参考环境是 sklearn1.7.1；主页 Conda 环境的固定版本不同，报告给出独立验证环境建立命令。

新脚本核对当前独立 CPU/GPU 源文件哈希。历史报告保留旧运行器哈希，脚本入口迁移不表示公开脚本从头重跑过1000人。数组、初始化、随机排列、字典和评估体素索引留在私密环境；仓库只发布匿名聚合和哈希。

## BigFLICA 集成证据

含 FLICA、C20、最终脑图或缓存模型的报告与运行器继续保留在 [BigFLICA 验证目录](../bigflica/README.md)：

- [2,050人 DicL→FLICA 阶段对照](../bigflica/dicl_bpdn_real2050.json)及[运行器](../bigflica/benchmark_dicl_real2050.py)包含 C3/C20 拟合和 z 图检查。
- [同投影控制](../bigflica/three_structural_f32_same_projected_control_real2050.json)及[定位运行器](../bigflica/diagnose_public_gpu_projected_dicl_cpu_real2050.py)包含 C3 course 与脑图比较。
- [缓存及输出检查](../bigflica/dicl_bpdn_cache_api_real2050.json)覆盖模型复用和新被试调用。

## 仍待验收

当前真实数据字典学习内部使用 float64。公开 BigFLICA GPU 的 float32 投影先提升为 float64，而 CPU float32 入口保留其原 mean/std 和训练精度；上述 float64 同输入比较不能覆盖公开默认 float32 规范。FA/MD 的 OMP30 重建差仍约2.44%/5.33%，尚不能称整个 `MiniBatchDictionaryLearning` 输出等价。压缩流程有效 C20、最终 course/脑图和大样本性能属于 BigFLICA 独立验收，模块拆分没有改变这些结论。
