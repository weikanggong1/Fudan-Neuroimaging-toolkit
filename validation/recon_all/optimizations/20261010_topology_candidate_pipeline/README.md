# 当前连续候选回归

此目录发布公开 ds000114 sub-07 ses-test LH、FNIT 79a41cdd 自产上游的同输入候选回归。实际算法基线 a6fcf88d 加三个具名新文件；模块 SHA `32a8dba28654df9d2bbe5bff1b45921df4ae5ca89010645f8cf8c03232d6aef3`，参考源码固定 d932c45b、隔离程序SHA `2df26fc8f997e13eee8d1dd97c46b452e2d25980645927b1d4de31f2eec9bec7`。全部完整哈希见公开report绑定；原始MRI/NPZ/私有收据留在授权计算目录。

| 版本 | 实际结果与边界 |
| --- | --- |
| v1 | 候选生成完整；检查器把原法向列3:6误作fx6:9导致失败。保留该失败，不归因算法/随机性 |
| v2 | 仅修事后检查器列定义；新空目录完整两调用EXIT0，所有几何/局部场/距离/反馈/五项F32与double fitness全bits一致 |
| v3 | 同算法另做第二调用透明cProfile，EXIT0全bits一致；计时包含剖析开销，不与v2宣称倍率 |

v2全110,534顶点XYZ/法向/K1K2/fx/curvbak、有序216,860面、白灰PDF和局部MRI/logP、21×19×16距离均0差，双调用自身逐bit重复，fitness=−97.760777950286865。600有向共边检查通过。首候选138新面由本链生成，未读取参考后态填入。

v2冷/热候选API18.466938/5.748928s，写出0.344392/0.346958s；原MRI/no-fix/缺陷识别与校准准备45.720757s单计。fresh进程74.922992s包含导入、加载、两次API、读写和事后比较。CPU8–11四核、GPU隐藏、共享负载；峰值RSS1,145,812KiB，不代表GPU显存或整例性能。

v3透明剖析的第二API8.038040s，重连+面发现5.711145s；110,539次索引校验和110,544次unique，全N行及全F索引重建构成主要可改进点。已编译相交351,038次合计0.207506s；不应首先重写此算子。各完整计时与前置准备分列在 [v3报告](reports/v3/report.json)。

## 证据与复现

- [v1原失败派生指标](reports/v1/report.json)、[v2严格结果](reports/v2/report.json)、[v3透明剖析](reports/v3/report.json)。
- 每版本 `EXPORT_RECEIPT.json` 是服务器白名单导出结果：455/455/730数值、布尔及null叶的值和类型保持，原JSON SHA保留。所有公开payload SHA见 `PUBLIC_REPORT_MANIFEST.json`。
- 三项有意义的本地单元契约覆盖共边/边界、DISCARD/外部状态、输入无别名。模拟契约不代替此真实回归。
- [完整具名API、格式、默认值与复现命令](../../../../docs/recon_all/TOPOLOGY_CANDIDATE_PIPELINE.md)。

![自产候选MRI叠加及严格误差](candidate.png)

图中99个顶点由自产新增面唯一顶点集合选择；MRI体素Z=49、69点位于±2体素范围。最大匹配位移4.053288mm、对相同源候选误差0；这不等于对原no-fix形状或最终质量验收。

## 未完成范围

初始全属性及global K/dot先验仍冻结；独立raw base/global-prior producer、跨候选DVS/RP集成、种群选择/停止/淘汰与完整GA、最终网格和原始T1整例均未在本报告验收。整体指标等效 `not_assessed`。没有新增依赖或生产默认切换，没有运行原FS阶段程序生成候选。
