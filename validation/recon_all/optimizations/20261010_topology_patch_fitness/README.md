# 拓扑 fitness 当前标量回归

完整功能与七节中文说明见 [功能页](../../../../docs/recon_all/TOPOLOGY_PATCH_FITNESS.md)。

本目录保留固定 `d932c45` 完整 fitness 规则和公开 T1 `sub07` 自产左半球首个真实候选的同输入评分回归。参考两次实际退出 0；新完整标量合成与参考的 double 字节相同，10,000 次候选调用重复相同。32 项契约通过。生产 GA 及最终网格质量门不改变，本项不验收完整 GA 或整体指标等效。

## 实际结果

| 项目 | 结果 |
| --- | --- |
| 首个完整候选 | 99 顶点、138 面；局部体积 21×19×16 |
| 源与新 fitness | −97.760777950286865，double 字节相同 |
| 旧 Python float 容器 | 本例无差异 |
| 旧 NumPy float32 容器 | −97.76077842712402，误差 −4.76837158203125e−7 |
| 源重复性 | 原核心 TP、字段位模式、完整捕获和 fitness 两次相同 |
| 新 API 时间 | 10,000 次 0.565269 s，平均 56.527 μs/次 |
| 新冷脚本含导入、读写及上述循环 | 0.710216 s |
| 参考观察器含上游与首次完整候选捕获 | 6.227574 / 5.112346 s，不能与标量时间相除 |
| GPU、整例耗时、最终网格质量 | 本项未使用 / 未测量 / 未判定 |

选中顶点/面面积均为 0。四个未被 fitness 消费的面积加权诊断为原 NaN，保留 JSON null 与 `0xffc00000` 原位模式；五个核心字段有限。中间候选的评分语义与最终网格 QC 分开。

## 文件与复现

- [完整规则](source_rules.json)、[实际冻结包 SHA](source_manifest_v1.json)：候选模块 `d9f90820…`，旧模块 `cd57434e…`；参考程序 `2df26fc8…`。
- [最终同输入比较](reports/comparison.json)、[容器类型首差](reports/input_type_comparison.json)、[三个子进程退出/全读写墙钟](reports/driver_receipt.json)。
- [参考 0](reports/reference/repeat0.json)、[参考 1](reports/reference/repeat1.json)：来自首个完整实际候选，不是初始统计或 unMRI-only 捕获；包含输入、源、程序、原字段与文件 SHA。
- [公开去敏清单](reports/export_manifest.json)：仅比较报告主机名替换成角色，原数值字段与失败状态保留；原私有字节 SHA 同时记录。
- [真实候选图](figures/real_candidate_fitness.png) 与 [绘图收据](figures/figure_receipt.json)：surface RAS/mm 解剖投影、选中顶点及有序评分项，不代表最终修复成功。
- `contracts-v1.*` 是 31 项早期契约；`contracts-v2.*` 是 32 项最终契约，含本例 NumPy float32 首加法回归。

```bash
# 读取已派生的完整原候选标量并验证第二次参考，输出必须是新文件。
python tools/benchmark_recon_topology_fitness.py \
  --reference-json validation/recon_all/optimizations/20261010_topology_patch_fitness/reports/reference/repeat0.json \
  --repeated-reference-json validation/recon_all/optimizations/20261010_topology_patch_fitness/reports/reference/repeat1.json \
  --output-json /tmp/new_complete_fitness.json \
  --repeat-calls 10000
```

`prepare_fitness_reference.py` 从私有完整观察器目录核对原输入/捕获 SHA，派生规范 JSON；`probe_scalar_input_types.py` 复用隔离旧模块；`export_fitness_reports.py` 只去敏；`plot_fitness_candidate.py` 复用已有 `load_statistics_capture()`。这些脚本是独立 benchmark，冻结参考和检查点不能进入生产。

原观察器 v1 打印 `-nan` 造成非标准 JSON，v2 保留原位模式并用 null 编码，源算法与核心值不变；两版独立运行记录分别保留。绘图首次因错误文件路径在读取前退出 1，后续正确路径成功；见 [完成及失败边界](completion_receipt.json)。
