# 完整默认首缺陷 GA：状态缓存摊销回归

本轮已退出0并完成完整默认首缺陷GA的逐位门。每缺陷准备一次显式状态上下文，495次候选评分、最终评分及写回全部共享；默认仍None。不混面索引缓存，不改变随机流、终止、剔除或源float32更新。

- [启动/版本收据](launch.json)：仅自有 FNIT 最小包，不含原版源码/程序、影像或许可证。846 基线 Python 文件经逐项 SHA 核验。
- [必要适配器合同](adapter_contracts.json)：2 项本地身份/异常恢复合同，8.29 秒，不代替 MRI benchmark。
- [完整机器摘要](reports/summary.json)：495+final全部评分、29条零基代记录、随机流1,362,381次、80完整字段及三个NPZ原文件全部逐位相同；工厂1次、完整装配497次，final score/write各1次。全部新产物落盘后才读取旧结果。
- [原数值白名单](reports/metrics.json)、[上下文计数与每次装配](reports/state_cache.json)、[496次评分/分段时间](reports/candidate_timings.json)、[新运行共享负载](reports/shared_load.json)、[首候选参考绑定](reports/source_reference_binding.json)。原私有报告/NPZ/影像没有发布，白名单保留数值类型及原SHA。
- 完整CLI含工厂、导入、JIT、上游生成及读写：2263.422952→2057.242890秒，观察缩短9.1092%；连接搜索/final API2182.039244→1979.388001秒，观察缩短9.2872%。暖评分中位3.240205→3.266672秒略慢；同主机四线程但CPU亲和性/运行时段不同，不能把总差异当作受控单核或整例收益。
- [性能图](reports/complete_first_ga.png)和[本轮自产MRI叠加](reports/first_defect.png)均绑定当前实际产物。自身首/末位移不是官方误差，脑图不代替最终网格QC。
- child峰值RSS1,722,355,712→1,736,519,680字节；父子同时合计CPU内存峰值未测，CUDA隐藏/显存null。全部缺陷、原始T1、整体指标等效与600秒目标未评估。
- [显示工具缺失的第一次失败](brain_plot_import_failure.json)与[最小公开传输/导出版本](public_transfer_receipt.json)保留；后续显示工具在独立目录运行，没有改变冻结算法。短阶段曾变慢的v2/v3报告继续保留于[原状态缓存页](../../../../docs/recon_all/TOPOLOGY_CANDIDATE_STATE_CACHE.md)。

[七节功能页](../../../../docs/recon_all/TOPOLOGY_GENETIC_STATE_CACHE.md)；[独立完整验证入口](../../../../tools/benchmark_recon_topology_genetic_state_cache.py)。
