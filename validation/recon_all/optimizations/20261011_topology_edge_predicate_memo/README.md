# 2026-10-11：不可变四顶点谓词memo实验

功能、全部参数和七节说明见[独立功能页](../../../../docs/recon_all/TOPOLOGY_EDGE_PREDICATE_MEMO.md)。当前只新增可选模块、合同和诊断工具，未更改生产调用或默认后端。

本地50项合同通过，0失败/错误/跳过，pytest墙钟8.96秒，JUnit8.96秒；覆盖近零/退化、三种float32尺度的全部有向四元组、动态增删与valid变化、完整两轮重连和审计SHA、cap外完整求值、不可变域/元信息与并发拒绝，以及诊断hook/比较器失败保全。机器记录见[本地合同](local_contracts.public.json)。小固定几何不是MRI benchmark。

另有3项typed导出合同通过，1.44秒；使用伪造schema收据，不是MRI benchmark。导出器要求双方收据已齐，独立复核24项全字段/控制/序列门，只发布预定义数值和SHA。

冻结v5已完成首identity候选真实配对，复用既有三个自产输入，不读取官方结果或旧候选作为计算输入。两臂独立空Numba缓存、相同两线程CPU预算，分别执行cold/warm/audit；双方完成前不读取另一臂结果。

全部24项比较及独立导出通过：完整36候选/40状态数组、6控制字段、原输入与实际来源SHA、有序谓词序列和自身重复一致。机器结果见[metrics.public.json](metrics.public.json)，完整来源与产物哈希见[execution_receipt.public.json](execution_receipt.public.json)。科学基线Git为`e9e89cac07cc65ef37f8c3223e53f03968f2af05`，只覆加memo模块；生产路径未变。

本次没有提速：冷CLI130.129983→131.456263秒，warm完整候选API8.434330→9.535094秒。每轮351,038次谓词、命中0、缓存250,000项、cap外求值101,038次；完整独特键数未知。峰CPU RSS1,796,235,264→1,836,576,768字节。共享环境单配对只能作为本次观测，完整495次搜索、真实跨候选复用、最终GA、网格质量与原始T1整例均未验收。未运行新官方对照，本实验不使用CUDA，显存指标为`null`。

复现入口为[launch_first_identity_pair.sh](launch_first_identity_pair.sh)、[诊断工具](../../../../tools/benchmark_recon_topology_edge_predicate_cache.py)和[独立导出器](export_metrics.py)。冻结协议和SOURCE_MANIFEST保持测试前字节，协议中的准备状态不代表当前完成状态；完成状态以机器结果和收据为准。
