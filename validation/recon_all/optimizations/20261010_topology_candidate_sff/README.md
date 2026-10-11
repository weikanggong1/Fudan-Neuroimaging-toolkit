# 候选二阶曲面拟合：当前证据

当前源码同输入真实回归已完成：公开 ds000114 sub-07/ses-test 左半球第一份完整拓扑候选，110,534 顶点、216,860 面，99 个拟合顶点。

- `reports/v1/report.json`：自产 K1/K2、主方向、完整五项评分、有向距离及反馈的逐值/逐位比较；参考和自身均双重复。
- `reports/v1/source_manifest.json`：实际测试的 29 文件冻结 SHA，保留当时 HEAD，不改为交付 HEAD。
- `reports/reference_*_receipt.json`：独立固定源码观察器构建、两次执行与捕获 SHA；仅隔离 benchmark。
- `contracts-v1.*`：冻结时 14 项契约；`contracts-v2.*`：最终 15 项契约，代码模块未改，仅新增病态回退检查。
- `comparison.csv` / `timing.csv`：精度与完整边界的派生表。
- `source_rules.json`：原算法位置、数学库及类型边界；不发布原 C++ 主体。
- `build_post_sff_observer.py` / `plot_candidate_sff.py`：独立观察器构建与可视化复现；需要私有已授权参考输入。
- `figures/candidate_sff.png`：真实候选位置和曲率、主方向误差图。

## 判定边界

SFF 和评分输出严格逐位一致；准备矩阵有 7 个正负零位差、数值全部相同，因此完整内部逐位复现仍为 false。完整 GA、最终网格质量、原始 T1 整例和整体指标等效未判定。没有可比的原 SFF 独立子段计时，不报告加速倍数。

原始影像、冻结数组、程序二进制、权重、许可证和服务器连接信息均不发布。本组来自已授权公开基准的去敏汇总。生产 API 只消费自身阶段输入，不读参考输出。
