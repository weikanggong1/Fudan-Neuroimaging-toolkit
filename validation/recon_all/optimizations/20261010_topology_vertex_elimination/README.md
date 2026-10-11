# GA 最差顶点剔除候选

## 1．功能简介

独立 NumPy 实现完整剔除操作。静态 F32 阈值显式传入和返回；遵守
原有效条目数、未使用条目的副作用及原停止条件。不是完整 GA。

## 2．Python 输入输出

完整参数、索引域、类型、源 signed-char 存储、输出结构、失败行为和
中文具名示例见[功能页](../../../../docs/recon_all/TOPOLOGY_VERTEX_ELIMINATION.md)。
RP 输入必须是同一原缺陷顺序的自身生产状态，不能用未来参考结果填补。

## 3．命令行

```bash
PYTHONPATH=src python -m unittest discover \
  -s tests/recon_all \
  -p 'test_topology_vertex_elimination.py' \
  -v
```

这是本地数组和状态合同，不能代替真实影像 benchmark。实际代码与
测试的 SHA、HEAD、测试数量和合同时间在 `local_contracts.json`。

## 4．原软件对照

固定 d932 `mrisurf_defect.cpp` 的 `deleteWorstVertices` 是内部函数，
没有单独 CLI。完整原参考命令与源码链接见功能页。未执行原程序，
也未链接其库。参考程序只能用于独立 benchmark。

## 5．当前验证

七项合同已通过：小缺陷返回0与副作用、F32阈值传递、原配额、索引
停止、同分顺序、signed-char、跳过负映射、别名顺序及非法输入事务。
真实同输入 before/after、参考重复性、阶段速度比、误差图和整例效果
均未测试，不写成通过。严格复现、整体指标等效均 `not_assessed`。
显存未采样为 `null`。未增加 Conda 依赖或新的安装/隔离验收。

## 6．版本记录

2026-10-10：新增完整剔除候选与七节功能说明；不改生产默认、源搜索
顺序或QC门。源规则异常仅记录，没有根据预期结果“修正”成另一算法。

## 7．参考

原函数、RP统计和自动拓扑修复论文均见功能页。发布范围仅为 FNIT
独立实现、测试、中文说明和无私有路径的合同报告；不含原软件代码、
程序、库或影像。
