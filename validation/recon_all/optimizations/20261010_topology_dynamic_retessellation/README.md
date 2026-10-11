# 有序动态重连、候选面与新检查点的真实验证

当前目录绑定首次完整真实缺陷，而非最终GA或recon-all整例。
输入为ds000114 sub07 ses-test LH，79a41cdd连续链自产。
使用唯一source-before状态（110534V/216722F/4723边），不是原闭合面表。

- 动态v3 cold6.118350s、repeat3.884241s：完整有序面/邻接/边状态/selected/marks0bitdiff。
- 新参考两次捕获仍与原7种fullfit捕获SHA相同；仅增加只读记录。
- 局部MRI准备phase0/phase1与before-SFF真实vtotal状态分别保存，不能把其参考输出当自产上游。
- `all`首候选已真实验证；`some`和overlap仅结构契约，完整MRI形变/GA搜索/最终QC未覆盖。
- [中文七节说明](../../../../docs/recon_all/TOPOLOGY_DYNAMIC_RETESS.md)。

`reports/PUBLIC_EXPORT_MANIFEST.json`绑定20份安全JSON。服务器先按严格字段
白名单移除编译命令、主机、内部路径、日志和任意构建元数据，再复用发布器
保真导出。原始收据SHA及选中字段一致性保留；未下载或发布MRI/参考数组。

`build_retessellation_observer.py`用于固定源码开发者benchmark树，新目录构建，
捕获FNITRTS1完整动态前/后；不属于生产实现。原v1失败、v2部署失败保留。
`tools/benchmark_recon_topology_dynamic_retessellation.py`是可复现同输入runner，
完整API/具名CLI和限制见中文说明。`cold/repeat`不是独占性能门，未提供
原retess独立计时，因此不报告加速倍数，也不估算600秒整例。
