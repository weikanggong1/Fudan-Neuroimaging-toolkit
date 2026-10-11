# GA最终缺陷写回：当前证据

本轮独立实现搜索后最优状态回填、最后评分/DVS恢复，再次完整几何匹配及未使用顶点清理。复用现有Python/Numba算子，没有运行原版阶段程序。生产默认未改变。

- [本地合同及12项固定源控制顺序核对](local_contracts.json)：11/11通过，0.015秒；HEAD72ba5336，实际finalizer及当时candidate cycle依赖SHA逐项记录。
- [完整中文API、空间/单位、失败行为和具名示例](../../../../docs/recon_all/TOPOLOGY_GA_FINALIZE.md)。
- [合同复现工具](run_local_contracts.py)要求新的输出路径；不覆盖旧结果。
- [真实最终段v2数值报告](reports/finalization_v2.json)：公开ds000114 sub-07/ses-test/LH、FNIT79a41cdd上游，声明的冻结初态/全局先验加自产MRI校准，两条独立序列60类数值输出逐位重复。
- [首候选严格源门](reports/initial_source_gate_v2.json)：全部自产结果生成之后比较参考，没有将after或参考分数作为输入。
- [完整新进程和资源记录](reports/process_v2.json)：104.396738秒；CPU120–123/4线程，CUDA隐藏，子进程RSS1,537,044 KiB，GPU未测为null；保留v1模块依赖缺失的EXIT1记录。
- [参考程序与重复性来源](reports/reference_provenance.json)：程序、构建和驱动SHA；观测程序主动在首候选之后停止，不是完整官方GA时间。
- [真实T1叠加及误差图](figures/final_patch.png)、[图像/输入/工具哈希回执](figures/figure_receipt.json)与[具名复现脚本](plot_frozen_final.py)。影像和私有NPZ未提交。
- [实际传出数值与图像的SHA回执](EXPORT_RECEIPT.json)仅覆盖白名单文件；出版清单另外绑定本地代码、测试与说明。

本地合同验证“最后评分→选择性恢复→再次几何→清理”的顺序与状态。第二次几何保留最后评分的H/K/E/fx/curvbak、不重算SFF/反馈；新面FACE.area保持源0，不替换为缓存面积。最终分数与搜索记录不同会如实返回。

真实两条序列的最终API分别8.380493/8.357907秒；最后评分/恢复4.422251/4.385028秒，再次完整几何3.919553/3.933976秒。最终坐标、有序面和法向与同序列已经通过源门的首候选几何逐位相同；固定排序没有完整种群搜索，最终自身分数均为−97.76077795028687，38项RP计数各加一次，没有额外剔除。完整最终源码捕获、完整GA/原始T1整例、脑区指标等效、600秒目标、同期GPU显存及干净环境隔离未在此验收。

测试实际版本为72ba5336加精确哈希覆盖层；`finalization_v2.json`绑定实际加载的全部FNIT模块，`process_v2.json`绑定运行包。v1只因缺少自身诊断依赖在导入处退出，v2算法字节未改变。冷首候选16.984秒与后续4.371秒分列，不能把JIT或整个104.397秒诊断当作生产阶段提速。
