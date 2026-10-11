# 有序 GA 交叉与变异：当前验证边界

## 1．功能简介

独立 NumPy/Python 实现完整边排列的交叉、变异，复用已验证的
`rank_patch_fitness`。当前只验证算子和字段合同，不接生产默认；完整
网格、fitness、状态恢复、种群和最终 QC 属于连续 GA 的后续验收。

## 2．Python 输入和输出

全部参数、索引域、返回结构、错误行为和具名中文示例见
[功能页](../../../../docs/recon_all/TOPOLOGY_GA_ORDERING.md)。随机数必填
显式回调，不能以 NumPy 默认 RNG 或相同 seed 代替源随机序列。
任意位置交换保留源临时位置表未更新的行为；它不是最终逆排列。

## 3．命令行和复现

```bash
PYTHONPATH=src python \
  validation/recon_all/optimizations/20261010_topology_ga_ordering/run_local_contracts.py \
  --fixed-source fixed_reference/mrisurf_defect.cpp \
  --output new_local_contracts.json
```

`fixed_source` 是仅本地只读的固定 d932 原源码，校验 SHA 后核对八处
原随机调用锚点；不提供时不核对源码。`output` 必须为新路径。报告
记录实际测试前后 HEAD、实现与测试 SHA；失败保留 JSON 并非零退出。
参考 observer、完整 `FNITGAO1` schema 和真实同输入回归 CLI 见功能页。

## 4．原软件对照

对应 `mrisCrossoverDefectPatches`、`mrisMutateDefectPatch` 和
`defectPatchRank` 内部步骤，无独立官方命令。参考 observer 透明记录
原调用参数和返回值，前三次交叉和前三次变异后有意停止。停止不表示
拓扑完成；原源码、库和程序只在独立参考环境保留。

## 5．实测范围

`local_contracts.json` 是实际本地数组 API／二进制 schema 合同报告。
真实 GA before/after 状态捕获尚待更广 headcw 脚本上传授权；当前没有
真实 MRI 阶段误差、原程序耗时、脑图或整例提速结论。严格真实复现和
整体指标等效均未判定。显存未采样，JSON 使用 `null`，不是零显存。

## 6．更新记录

2026-10-10：完成独立交叉、变异、显式随机轨迹和排名复用；提供严格
完整字段解析与本地复现脚本。未移除原始 T1 的既有失败项，未改变
验收阈值或默认 profile，也未新增 Conda 依赖。

## 7．参考

固定源码、原文链接及 automated manifold surgery 参考文献见功能页。
本目录仅发布 FNIT 验证脚本、源规则和无私有路径的报告，不发布原
软件源码、库、程序或真实影像。
