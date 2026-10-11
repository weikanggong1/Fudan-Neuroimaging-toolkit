# 初始前三候选memo专项

首次v6真实前三候选配对及离线重核已完成，原运行的3/114 JSON逐字节失败和EXIT1保留。111个原门相同，3个完整prefix JSON对象门另核通过；没有重跑计算。该范围不是完整GA或原始T1整例。生产默认与v5负结果均保持原样。

预设三项FNIT自产输入、科学基线Git e9e89cac、同cap250000、seed1234连续VNL流、两线程CPU、GPU隐藏；baseline/memo分别为冷新进程，进程内部独立执行cold/warm/audit三条初始三候选链。每链评估器和memo只准备一次，三个slot保留真实累积反馈，链间不共享缓存。

- `protocol.public.json`：运行前冻结完整36/40/6、排序/反馈、全随机流、45artifact及114比较门。
- `source_manifest.public.json`：实际源码/工具SHA和模块名映射，拒绝安装目录兜底。
- `export_metrics.py`：双方完整收据后独立重核，导出typed指标；`test_export_metrics.py`只验证离线schema和隐私。
- `launch_prefix_pair.sh`：只接受私有路径参数，不包含服务器入口或影像路径。
- `local_contracts.public.json`、`local_source_import.public.json`：本地门和导入收据，不是MRI benchmark。
- `execution_protocol.public.json`：首次科学运行前冻结的CPU67–68资源重绑定，SHA17e747；原CPU92–93协议不改。
- 协议内proposal/pending标记是启动前冻结状态；实际完成、失败和离线重核以收据为准，原协议不回写结果。原`launch_prefix_pair.sh`绑定92–93未用于这次执行；按actual协议直接调用冻结工具，完整外层worker由收据SHA绑定。
- `metrics.public.json`：原111门、3原失败与3对象门、完整随机流、逐slot缓存计数和冷/热/IO/RSS。memo实际命中34.44%，warm扫描0.626→1.291秒，warm完整prefix22.958→23.114秒；未证明提速，不提升默认。
- `execution_receipt.public.json`：原外层355.465秒、退出码1、实际资源与104文件全源绑定；两臂冷CLI162.318→159.541秒是单次共享环境观察，不能归因memo加速。
- `original_failure_receipt.public.json`：由已核机器metrics/外层收据投影原111通过、3失败、原比较/启动器SHA及退出码1，与新重核分开记录。
- `recheck_prefix_json_order.py`、`test_recheck_prefix_json_order.py`：复用完整原比较器的一项JSON对象键序修复，逐类型/F64位检查；不改原科学源/产物。
- `export_execution_receipt.py`及测试：只导出typed完整时钟和来源，保留原失败状态。
- `local_json_order_recheck_contracts.public.json`：声明源码路径下59项合同通过，含不同PYTHONHASHSEED冷进程和科学数组一位变化拒绝门；`local_misconfigured_test_attempt.public.json`保留此前遗漏PYTHONPATH导致的本地环境失败。
- `frozen/v6_rebind/tools/benchmark_recon_topology_edge_predicate_population_prefix.py`：测试时原029入口。当前tools入口只修正prefix JSON键序，不能把其新SHA写成本次科学测试来源。

完整中文输入输出、坐标/单位、失败条件、示例、时钟范围和对应原软件见[七节说明](../../../../docs/recon_all/TOPOLOGY_EDGE_PREDICATE_POPULATION_PREFIX.md)。GPU隐藏，显存字段null；没有新官方结果、最终网格、整例或隔离安装通过结论。
