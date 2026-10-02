# 本轮官方 raw10 参考链只读依赖检查

2026-10-03 的实际检查只验证环境、源码、程序与原始输入，不执行追踪、不初始化 CUDA，不作为端到端精度或性能结论。

| 运行环境 | 七个固定 MRtrix 程序 `-version` | 原始数据 | 完整 producer 合同 | 实际耗时 |
|---|---|---|---|---:|
| nodecw10 | 全部 exit0；版本与已审计参考相同 | 101 个唯一文件 SHA 与 NIfTI header 通过 | 未齐，required_dependency_ready=false | 5.8754 s |
| gpucw1 系统运行库 | 全部因旧 libstdc++ 缺失 GLIBCXX 返回1 | 同一101个文件通过 | 未齐，required_dependency_ready=false | 3.1925 s |

固定二进制和运行库文件均未修改。实际官方参考在 nodecw10 的既有认证通道运行；各次真实返回码、程序 SHA、配置/工具 SHA、完整 producer 状态与原报告摘要见 [机器记录](actual_dependency_preflight.json)。失败环境记录完整保留。尚未启动新 raw10 官方追踪/下游。上游原始 B0 frame index 正在明确绑定；合同完成后须用新冻结配置再检查，不隐式消费旧路径。

最新 CPU 协议回归：32 passed / 1 skipped，3.94 s。小数组 fixture 验证输入合同、失败处理与统计工具，不作为真实科学 benchmark。固定输入的 CON03 差异另见 [独立参考报告](../task_04_repeat_reference/README.md)。
