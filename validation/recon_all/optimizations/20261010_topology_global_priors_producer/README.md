# 自产全局 K/dot 先验：当前真实阶段验证

## 范围与输入

公开 OpenNeuro ds000114 sub-07 ses-test 左半球，FNIT `79a41cdd` 自产的未修复网格及快速球面，同一有序 110,534 顶点、221,116 面。从两份输入独立生成 canonical/mixed 法向、有序邻域、首次原域 SFF1、两轮 ORIGINAL 平滑及实际 metric，然后生成全部全局先验。实际两输入 SHA 与固定参考输入相同。

完整候选保存后才读取固定 `d932c45b7941662ea380a05efef580568b98d41a` 源码独立构建程序的既有两次捕获。FNITFIT1 的几何与顶点 K 在后续 SFF2 相位，捕获中的 K/dot 先验仍来自首次 SFF1 和平滑 ORIGINAL dot 阶段；候选没有读取参考 K、先验或参数作为运算输入。

## 完成门

- v3 驱动退出 0：完整面、平滑 ORIGINAL 坐标及原一环 CSR 建立有序对应，全部不同值/字节和最大/P99 误差为 0。
- 11 个先验字段逐位相同：两个曲率的 bins/PDF/bin_size，联合 PDF 与几何，以及 dot bins/PDF/bin_size。dot 消费 663,348 条有向边。
- 两次自产完整重新计算的全部输出数组字节相同；两个固定参考捕获全文件 SHA 相同。
- 新补充均值检查从自产 PDF 调用成熟 HISTO mean，K1/K2 为 −0.09168962389230728/−0.03310907259583473，与 MAT1 捕获的实际匹配参数 float32 位模式相同；两次参考参数也相同。
- metric pending `[0]`，没有未解决顶点或坐标改动。17 契约通过，JUnit 7.556 秒。

完整 GA、整例运行、最终网格质量及整体脑区指标等效未评估。本阶段不依赖原版 C++ 程序；原程序仅生成 benchmark 参考。

## 耗时、资源和版本

Python 3.11.16、NumPy 1.26.4、Numba 0.67.0、Torch 2.5.1，同 CPU 节点四核亲和性、总线程 4。已有 JIT 缓存与共享负载：首次完整 API 22.517566 秒，重复 18.659587 秒；两次生成、加载、检查点写入和比较的驱动 43.756364 秒，导入另计 0.400679 秒。解释器启动和最终 JSON 写出未计入驱动时间。没有 CUDA 运算或初始化，显存为未测/null，不写成测得 0。

参考观察器 5.072093/5.221656 秒包括其他准备/捕获相位，不能与 FNIT API 边界计算速度比。没有配对整例提速结论。

实际源码是 `72ba5336ddde011a50cfe202b560479732816c30` 时冻结的六个最小 FNIT 适配器/工具/解析器/收据/启动文件，加只读复用的已部署 FNIT 依赖。每个实际运行模块 SHA、原输入/程序/捕获 SHA、三次源码清单和包 SHA 在 [source_binding.json](reports/source_binding.json) 中。当前发布提交仅作为交付位置，不回填测试版本。原日志、候选 NPZ 和运行源码清单保留在私有任务目录；公开副本移除主机标识，保留完整指标、计时、参数、退出码及哈希。

## 失败版本保留

| 版本 | 原始退出与定位 |
| --- | --- |
| v1 | EXIT 1；工具比较 raw current 与参考 smoothed ORIGINAL，相位不一致。先验比较被同域门阻断 |
| v2 | EXIT 1；相位/输入核对完成，工具导出 joint PDF 的 origin/spacing 顺序错误。其他 PDF/bins/counts 已零差异 |
| v3 | EXIT 0；只改工具导出顺序，适配器与计算依赖 SHA 未改变，全部 11 项通过 |

不改写 v1/v2 报告，也不把工具相位错误归为 GPU/随机差异。

## 复现与收据

- [完整接口、具名参数和流程图](../../../../docs/recon_all/TOPOLOGY_GLOBAL_PRIORS.md)
- [v1 原始数值报告](reports/report_v1.json)、[v2](reports/report_v2.json)、[v3](reports/report_v3.json)
- [实际匹配均值回归](reports/prior_means_v1.json)、[环境](reports/environment.json)、[交付摘要](reports/delivery_summary.json)
- [契约版本绑定](reports/contracts_v3.json)、[JUnit](reports/contracts_v3.junit.xml)
- [公开真实脑图/PDF](figures/global_priors_brain_and_pdf.png)、[脑图收据](figures/figure_receipt.json)
- [绘图复现](scripts/plot_global_priors.py)、[均值检查复现](scripts/verify_global_prior_means.py)
- [发布文件 SHA 清单](PUBLIC_EXPORT_MANIFEST.json)

复现工具 `tools/benchmark_recon_topology_global_priors_producer.py` 只消费同输入 nofix/qsphere，所有参数通过显式长选项给出；完整示例、参数、空间和失败行为见专页。生产评分仅接 `global_topology_priors()` 的自产结果，不使用 benchmark 参考。
