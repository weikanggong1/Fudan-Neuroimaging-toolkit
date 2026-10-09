# 当前自产 smoothwm 的完整 inflation 接入回归（2026-10-09）

公开 ds000114 sub-06/sub-07，输入为已完成原始 T1 空目录整例 `803aec50` 的 FNIT 自产 smoothwm。这里重新计算的是完整标准 inflation/sulc 阶段；不是新的原始 T1 整例。固定 Conda 源码构建 mris_inflate 是同输入参考，未调用预装 FreeSurfer。

## 结果

| 范围 | 结果 |
|---|---|
| 四网格完整 API | 每侧 native/Torch/Torch/native；8 次 Torch 与 native 有序坐标、sulc、几何头均零差异 |
| 参考稳定性 | 每个 native 重跑均解码几何与 sulc 相同 |
| 四网格新解释器 CLI | 13.975 / 10.836 / 12.107 / 10.227 s；输出均零差异 |
| sub-06 双侧组 ABBA | native 46.299/44.547 s，Torch 27.270/17.765 s；中位数45.423→22.518 s，缩短50.427% |
| sub-07 双侧组 ABBA | native 37.682/33.902 s，Torch18.248/18.768 s；中位数35.792→18.508 s，缩短48.291% |
| 已初始化父 CUDA | 8组父关闭缓存/live SHA保持；两侧新worker实际启用缓存、每侧2线程、TF32开启且无半精度 |
| 严格复现 / 优化退化 / 整体等效 | 同输入通过 / 本次未见 / 未判定；整例加速未测 |

同机 A100-SXM4-80GB / Xeon Platinum8369B，CPU64–67，总4线程。完整组含私有复制、fresh exec、导入、CUDA初始化、JIT、完整API、搬运、读写和发布。首个Torch组为空JIT，其后复用。CPU/GPU存在共享负载，不能把这些阶段观察作为稳定吞吐或跨组比较。

## 精度与坐标

输入/输出都是同序三角网格，surface RAS/mm；sulc为与顶点顺序对应的FP32有符号累积投影/mm。sub-06 LH/RH为130679/132459顶点，261354/264914面；sub-07为114247/114951顶点，228490/229898面。最大/P99/RMSE距离均0。图为全部顶点的矢状投影，不能代替三维自相交检查；有序面保持，不扩展网格质量验收。

![当前四网格与误差](figures/smoothwm_inflated_error.png)

## 报告、代码与资源

- [完整API配对](v6_complete_pair/summary.json)、[冷CLI](v6_cold_cli/summary.json)、[真实双侧组ABBA](v7_groups/summary.json)。
- [输入清单](public_inputs_manifest.json)、[实际源码/程序/硬件身份](source_hardware_receipt.json)、[图身份](figures/figure_receipt.json)。数值核心沿用已提交8742ce8c，完整runner SHA1186db4d、Torch积分SHAa782b5ec；child缓存调度SHAc618fc80。
- [去敏感清单](public_export_manifest.json)、[5,227项数值复核](publication_receipt.json)：27份原报告数字、布尔、null保持，私有原包SHA87d0125aa0f69d2ceeaca08fd6cde01bedd469334682586843a22464729fa3f0保留。包不含影像、网格、权重、许可证和凭据。

各Torch worker allocated315,037,184–358,849,024字节、reserved394,264,576–473,956,352字节，未相加冒充同期值。目标卡采样峰值12,583,960,576字节、全部计算进程同期合计上界12,557,746,176字节含其他任务。PID namespace归属未确认，父子树峰值null；名义0.5s、最大实际5.678s、0查询失败。未进行连续峰值保证、整例20GB验收或物理独立安装隔离验收。

缓存关闭的旧生产反例仍有效：相同完整Torch算法会因临时分配明显慢，不能直接切默认。当前候选只在标准inflated/sulc对应fresh worker局部启用缓存；nofix/quicksphere保持原接口，注册默认inherit。

全部具名参数、语义、复现命令、失败行为和参考见[中文专页](../../../../../docs/recon_all/INFLATE_TORCH_20261009.md)。
