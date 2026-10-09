# 两例完整 MNI/网格组并行验证

本轮使用公开ds000114的sub06/sub07、冻结803aec50自产原始T1链的orig/crop/affine及最终八个表面。固定A100 GPU1、CPU亲和性4–7，总四计算线程；串行4+4、并行父2+子2，每例A→B→B→A。生产公式不变，新增 fresh exec 调度并修复非当前CUDA设备的Triton作用域。

| 被试 | 串行中位，秒 | 并行中位，秒 | 组加速 | 墙钟减少 |
| --- | ---: | ---: | ---: | ---: |
| sub06 | 160.421 | 94.287 | 1.701× | 41.23% |
| sub07 | 142.240 | 72.892 | 1.951× | 48.75% |

## 精度与验证范围

8次完整文件API均exit0；6个对照均forward/inverse/check体素、最大/P99为0，shape/affine/dtype/头/文件SHA相同；现有完整生产网格验证结果相同。所有API父current0→0、TF32/allocator环境与存活CUDA张量保持。5项helper和2项双GPU正常/异常恢复合同通过。

两例公共完整inverse旧current1/新current0，同物理GPU1：全域0差异、头/affine/dtype/SHA与Voronoi/soap迭代统计同。两例未初始化父CUDA冷CLI98.857/97.674秒，仍0差异，父CUDA不初始化。

完整组计时包含校验、输入/权重/程序SHA、加载、GPU同步、fresh exec/import、传输、写出和join；最小fixture副本准备与结果比较单列外计。每次冷进程墙钟/RSS见api_v3/*.time，不能只用局部模型时间评价并行。共享节点及GPU负载有波动；ABBA仅两次/模式，不能外推稳定吞吐。

## 资源、版本与未验项目

线程/精度/输入输出及源码SHA见每次JSON和source_binding.json。GPU目标UUID记录在采样报告；host/container PID归属未解决，tree peak为null。整卡占用包括其他任务，cache-off时Torch allocated/reserved不可用；没有本helper低于20,000,000,000字节的证明。

这是冻结同输入完整两阶段测试，不是原始T1整例。尚未验证接线后的整例提速、干净独立环境、整体指标等效，也未重新运行官方程序。旧官方比较仅作为既有算法参考，不能改标本轮结果。

## 保留的失败与复现

- failure_v1：环境缺GNU time，退出127，未运行算法。
- failure_v2：54.664秒报Triton指针不可访问，CUDA0/current与CUDA1 tensor不匹配，并非OOM。
- api_v3：局部device作用域修复后8次ABBA与7项合同。
- inverse_v4：两例相同前向场的公共逆场API回归。
- cli_v5：两例父CUDA未初始化的完整并行组。

[参数与Python/CLI示例](../../../../docs/recon_all/MNI_MESH_PARALLEL.md)。reproduce.sh复用已有公开检查点、声明资产与明确冻结源码；不下载影像或权重，不调用官方程序。MANIFEST.json验证报告文件，source_binding.json验证候选/基线源文件；不存在影像、权重、许可证或凭据载荷。

![同输入MNI检查图](mni_check_comparison.png)
