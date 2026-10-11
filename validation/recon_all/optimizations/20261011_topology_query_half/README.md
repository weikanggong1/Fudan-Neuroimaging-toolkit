# 拓扑查询端复用的真实阶段回归

## 已完成范围

- `fourth/metrics.json`：同第三缺陷自产检查点→完整第四首候选，两个fresh
  NumBa进程，cold/warm/audit各72字段和全部控制逐位相同。
- `wiring/metrics.json`：实际库Python/NumBa各初始5评分+最终1评分；81字段、
  4个NPZ、完整控制及20,877次RNG bounds/value序列相同。
- `fourth/query_half_candidate.png`：实际190个候选终态顶点在自身T1上的叠加。
  它表示自产坐标，不是原程序误差或完整网格QC。
- `first/metrics.json`：完整默认首缺陷退出0，495搜索+final1、497重连、
  29trace(index0–28)、80字段/3NPZ以及全部496分数和控制逐位相同。
- `first/query_half_first.png`：完整评分曲线与99个自产最终顶点的自身T1叠加。

九缺陷、orig和原始T1整例不由本目录验收；当前既有任务保持各自冻结版本。

## 计时边界

第四既有Numba/查询端复用的CLI为134.428856/134.494223s，热闭环
3.826088/3.263971s；cold19.219048/20.654912s，audit4.980915/5.469966s。
两个fresh子进程顺序、CPU84–87、各4线程、无CUDA。共享负载存在；保留冷
和审计的负结果，不把一次热调用观察称完整GA加速。

六评分Python/查询端Numba CLI115.622033/119.246340s；完整search+final
API37.881616/39.917858s，未测到阶段提速。算法计时、加载、准备、写出
及RSS原数值见机器报告。

完整首缺陷的已完成Numba基准/查询端CLI为1357.057126/1217.853505s，
报告前完整API1349.788190/1208.901601s，search+final阶段
1277.391707/1134.773795s。两个共享负载窗口只有同预算单次观察，未做
ABBA，不将本次较短墙钟解释为稳定吞吐、全部九缺陷或整例加速。

## 来源与保真

第四manifest `f1f1cd47…`、直接六评分manifest `0d2864c5…`。
完整首缺陷manifest `5bd768f2…`，协议`e3ad3121…`。
三协议分别执行，不把新helper SHA回标旧运行。查询helper `1a9cc1f9…`；
成熟dynamic `74c381ac…`保持不变。扫描完整谓词序列SHA `9acfa671…`；
六评分完整随机序列 `39b81ccb…`。

第四旧helper实际`6cc1e3d6…`；完整首缺陷基准实际`33a919df…`，分别保留。
完整首缺陷的RNGcount=1,362,381、29trace和控制一致，但工具未捕获全部
逐draw值；不以count代替逐draw序列证明，不称官方winner逐位一致。
实际80字段为first32/final40/search8，完整文件SHA与既有Numba产物相同。

`EXPORT_RECEIPT.json`绑定实际原报告/COMPLETE/EXIT以及导出metrics SHA/size。
`PLOT_RECEIPT.json`再绑定同run、同metrics、实际加载MRI变换模块与PNG。
下载后全部SHA/size再次核对。选中数值/布尔/null的值和类型不变；完整私有
记录留服务器，不公开MRI、NPZ、日志、服务器地址/账号/内部路径。

第四/直接六评分实际执行导出器`61537d41…`、绘图器`0a24c449…`；
完整首缺陷实际执行导出器`eacdec74…`、绘图器`6dd57af3…`。后者新增旧
full-cli基准结构的严格绑定，旧报告没有回写新工具SHA。

完整七节说明和具名调用见
[查询端复用](../../../../docs/recon_all/TOPOLOGY_BLOCKER_QUERY_REUSE.md)。
