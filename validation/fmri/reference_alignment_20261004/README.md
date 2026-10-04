# BOLD参考图与皮层采样对齐：实际验证

本轮使用 ds001226 v5.0.1 的 CON01、CON06，真实 T1w 与完整180帧 BOLD。新增 robust 参考对齐官方 NiWorkflows1.14.4 的选帧、裁剪、漂移归一与中位数，运动校正复用 TorchMCFLIRT。相同输入的成熟皮层采样已与官方流程逐值比较；完整流程的残差另行保留。

配准方向为 **BOLD→T1：TorchFLIRT+TorchBBR；T1→MNI：TorchFLIRT+TorchFNIRT**（显式 `registration_backend="fnirt"`）。当前 API 解剖配准默认仍为 SynthMorph；本轮均明确选 FNIRT。

| 实验与样本 | 实际结果 | 详细报告 |
|---|---|---|
| 参考图：两例、完整原空间三维参考 | robust与官方的空间相关均超过0.9998；精确数值与阶段时钟见报告 | [参考图与真实脑图](../../../docs/fmri/bold_reference.md) |
| 固定同输入皮层采样：两例，每例11项完整保存输出 | 所有float32值相同，最大绝对差和RMSE为0；完整91k轴相同 | [采样端点、时钟与脑图](surface/SURFACE_ALIGNMENT.md) |
| 自产连续volume→surface：每例middle/robust各一次 | 两例CIFTI时序r中位数0.717636→0.737295，NRMSE9.248912%→9.170536% | [四次完整对照](CONTINUOUS_BENCHMARK.md) |

| 自产连续链：相同两例中位数 | middle | robust |
|---|---:|---:|
| API至保存终点，分钟 | 11.35 | 11.99 |
| MNI2mm逐位置时间r病例均值 | 0.511970 | 0.520368 |
| MNI NRMSE，参考为官方fMRIPrep | 24.044193% | 23.845726% |
| 91k CIFTI逐位置时间r病例均值 | 0.717636 | 0.737295 |
| CIFTI NRMSE，参考为官方fMRIPrep | 9.248912% | 9.170536% |

连续 API 自动生成新的 volume 再进行 surface，复用自身原完整重建与 final MSM 球面；上述时间排除冷重建/MSM求解。共享 H100 与CPU条件、阶段/外层时钟、完整180帧及固定脑域、常数序列和来源守卫均在具名报告中记录。固定同输入采样结果不等同于完整流程数值等价；仍保留 CPU/Workbench 步骤。

[原冻结FNIT0.16.0、官方fMRIPrep25.2.4与DeepPrep25.1.0的十例三方报告](../threeway_20261004/final10/REPORT.md)已完成并独立保存，不能将其计入本轮新robust策略的两例统计。所有正式结果均来自真实成品；小型测试只用于代码合同验证。
