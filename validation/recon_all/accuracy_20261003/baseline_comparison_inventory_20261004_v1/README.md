# 816 / 8f 真实案例比较清单

现场读取时间：2026-10-04T12:59:37.365934+00:00。读取 canonical INDEX、冻结 cohort、原始及重放 config/launch/completion、实际 pipeline JSON、官方 done SHA、比较 checkpoint/execution_binding/strict 报告；只读核对，未启动算法或 GPU。

| 案例 | 816 原始 | 816 重放 | 可用816完成产物 | 官方完成/现存138项 | 现存完整官方数值比较 |
|---|---|---|---|---|---|
| ds000030_sub-10159 | complete | 无 | 是 | True/138 | complete，138检查/2通过；输入metadata待纠正 |
| ds000030_sub-10171 | complete | 无 | 是 | True/138 | complete，138检查/2通过；输入metadata待纠正 |
| ds000030_sub-10189 | failed | complete | 是 | True/138 | 未发现 |
| ds000030_sub-10193 | complete | 无 | 是 | True/138 | 未发现 |
| ds000030_sub-10206 | failed | 无 | 否 | True/138 | 未发现 |
| ds000114_sub-04 | complete | 无 | 是 | True/138 | 未发现 |
| ds000114_sub-05 | complete | 无 | 是 | True/138 | 未发现 |
| ds000114_sub-06 | failed | failed | 否 | True/138 | 未发现 |
| ds000114_sub-07 | failed | complete | 是 | True/138 | 未发现 |
| ds000114_sub-08 | failed | complete | 是 | True/138 | 未发现 |

816 共有8例实际完成、138项齐全：10159、10171、10189（重放）、10193、sub-04、sub-05、sub-07（重放）、sub-08（重放）。10206和sub-06没有完成的816整例，不能用其他源码结果替换。8例可以供未来同输入3a完成产物做直接前后比较，但已有官方误差指标只覆盖10159、10171；其余6例还没有现存完整官方数值比较。

8f sub-06单列：实际源码8f3e51f58c37d28e1283d5bb8e248e93530cd398，整例complete、138项齐全，官方比较complete18阶段，strict检查138项、5项通过。仅可与未来3a同输入sub-06比较；不是816结果。原binding顶层输入SHA错误保留，已有独立metadata_correction_verified及原binding SHA绑定；该纠正只核验元数据，没有重跑数值。

10159和10171的原execution_binding顶层input_sha256均误记8aa986c658663d2c26fa482a2ce303c4eecc28fbd69e4f6e9bace438be058200。实际输入现算SHA、两侧config/launch、官方completion与cohort一致，且比较两侧config/completion SHA与实际回执匹配；checkpoint均complete18阶段。因此可安排独立CPU verify-only纠正，当前清单不把该metadata缺陷隐藏为已修正。binding_verified_only不计为数值比较完成。

机器清单 inventory.json逐例保存输入SHA、实际config和source commit/archive SHA、launch/completion/pipeline SHA、138存在性及严格比较通过数量。官方版本为FreeSurfer8.2.0 d932c45，816源码为816e5610417a4c587caf321049438a9554139016。未重跑全资源绑定核验，读取的已完成报告可用于规划，后续正式比较仍须完整执行绑定。

138个文件存在、138个检查已执行与138个检查通过是不同指标。未来3a必须有自身complete执行回执及同输入比较；当前整体等效仍为not_assessed，不能依据现存报告宣称新候选已完成或已等效。
