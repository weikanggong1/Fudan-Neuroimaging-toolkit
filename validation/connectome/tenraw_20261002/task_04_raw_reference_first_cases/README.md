# 独立官方原始链：首两例五重复实物审计

## 1. 本次范围

2026-10-02 22:22 UTC 的实际快照：公开 ds001226 固定 snapshot 的 CON01、CON03 已分别完成五轮官方 iFOD2/ACT、SIFT2、标量采样和八套 atlas 四矩阵，每例 198 条命令全部 exit0。两组控制器正在继续 CON04/CON05，其余病例等待真实上游合同。这是十例工作的一份阶段记录。

原始 T1w、AP/PA DWI、梯度和 JSON 来自 canonical 清单，SHA `cc33e925a9e07362103b51f7bb70363a380d89b11a19545837676ba9a4ffae70`。官方链消费自己生成的 corrected DWI、FOD、FA、5TT/GMWMI 和 atlas；旧固定 FNIT PT 五重复属于另一个算子定位报告。

| 病例 | seed 0–4 接受轨迹数（每轮尝试 100000） | 追踪每轮 s | SIFT2 每轮 s | 五轮组件 worker wall s | 官方 CPU EDDY s |
|---|---|---:|---:|---:|---:|
| CON01 | 14554 / 14350 / 14434 / 14559 / 14573 | 65.40–66.27 | 11.99–12.04 | 427.83 | 2550.45 |
| CON03 | 11997 / 12038 / 11937 / 12274 / 12099 | 62.87–63.88 | 10.59–10.68 | 406.84 | 2550.93 |

追踪在 nodecw10 固定 `-nthreads 0`，下游 8 线程。两组同时运行，时间包含这次实际服务器负载。worker wall 是五轮追踪至矩阵、读回和输出校验的组件时间；EDDY 是此前 gpucw1 上官方 `eddy_cpu` 的单独命令时间。recon-all、TOPUP、解剖与 FOD 时间在各自 producer 报告中，不能把这里的组件时间当作全 raw pipeline 时间。

## 2. 已验证什么

只读审计在 CON01/03 分别用时 5.007 / 4.395 s，未调用任何官方命令或 GPU。它重验 canonical raw 原文件、actual producer 合同、worker/两 helper/七 binary SHA、13 个原始 MRtrix reader 记录、五份 TCK 和标量摘要、160 份矩阵及 atlas/nodes 元数据；完成 report 的 SHA 在审计前后相同。

- [CON01 原样机器报告](sub-CON01_official_repeat_audit.json)
- [CON03 原样机器报告](sub-CON03_official_repeat_audit.json)
- [来源文件实物 SHA 快照](actual_source_snapshot.json)
- [两组实际输出来源表快照](case_origin_binding_snapshot.json)

每个 atlas 的报告保留官方自身全部十对比较和六指标完整范围。错误门槛是官方最大观察误差，相似性门槛是官方最小观察相似性。这里没有 FNIT 结果，cross 与 FNIT 自身重复均为 `not_assessed`；官方自身范围不能单独证明 FNIT 匹配。这是五次有限样本观测，不是总体置信区间。

八套节点数在这两例为 84 / 84 / 164 / 376 / 414 / 216 / 554 / 1054，以实际 `nodes.tsv` 定义行列。其他病例可有不同缺席 ROI，不硬编码维度。

## 3. 复现只读审计

```bash
# 路径全部指向实际已完成产物；不重跑追踪、SIFT2 或矩阵。
raw_manifest=/path/to/canonical/input_manifest.json
raw_manifest_sha256=cc33e925a9e07362103b51f7bb70363a380d89b11a19545837676ba9a4ffae70
completed_official_case=/path/to/actual/group_B/sub-CON03
frozen_reference_worker=/path/to/frozen/tools/reference/benchmark_connectome_raw_official.py
new_audit_report=/path/to/new/sub-CON03_official_repeat_audit.json

CUDA_VISIBLE_DEVICES='' python tools/reference/audit_connectome_raw_repeats.py \
    --manifest "$completed_official_case/reference_manifest.json" \
    --raw-manifest "$raw_manifest" --raw-manifest-sha256 "$raw_manifest_sha256" \
    --case-id sub-CON03 --worker "$frozen_reference_worker" \
    --output "$new_audit_report"
```

`--manifest` 是真实完成参考报告；`--raw-manifest` 与 SHA 绑定原始病例；`--case-id` 必须一致；`--worker` 是当时执行的冻结源码及相邻 helper；`--output` 必须为新文件。工具源 SHA 写入审计结果。标量 QC 如有非有限值会保留计数，只描述有限子集分布，不改变原标量或矩阵。

## 4. 本轮 metadata 修正记录

此前 dependency preflight 在 5TT 第四个非空间 channel 的 `spacing=NaN` 处无法严格写 JSON。commit `8afae8e1` 仅将该 undefined metadata 记为 `null` 并记录原 NaN 类型；前三个物理空间轴仍要求有限正间距，原 NIfTI 和 `mrinfo` 文件字节/SHA不变。与 `588e8e33` 比较，`parameters()` / `main()` 的科学命令 AST 未变；19 个 CPU 契约测试通过（8.37 s），测试不作为科学 benchmark。旧失败 preflight 和无 tmux 的零派发尝试均在来源快照中保留。

两组实际 worker SHA `0c6191ec2fc950352548d2244a8ee1c03ab22cdd42a2f347bd36ffcb83dd487e`，控制器 SHA `f73e3840235f42b18bd902b6811dcd6834020d2bec1fa3aafe1fa6b51eafcf89`。新只读审计器独立冻结，不修改任何 producer 或运行中的 worker。只读 collector 另有 2 个标准库来源协议测试，包含实际 manifest 的 `size_bytes` 字段和原始/worker/audit/病例摘要错配拒绝，0.008 s；这些是协议回归。

## 5. 参考入口

[独立官方 raw 工具、公式、命令和文献](../../../../tools/reference/benchmark_connectome_raw_official.md)；[FNIT 五种子固定输入结果](../task_04_repeat_fivefnit_reference/README.md)。后续新 formal FNIT 输出来源经总控冻结后，用 raw 全链 envelope 分别比较相同病例的八套四矩阵。
