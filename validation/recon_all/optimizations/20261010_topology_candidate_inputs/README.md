# 每表面缓存与每缺陷候选输入：真实阶段验证

## 范围与完成门

公开 ds000114 sub-07 ses-test 左半球，110,534 顶点、221,116 面、首缺陷 4723 条有序候选边。自产 nofix/MRI、完整自产全局 K/dot 先验及自产 ETable 检查点进入准备函数；自产检查点写出以后才读固定源码参考。原版程序只在独立 benchmark 中使用。

- v1：EXIT 0，24 组消费输入/29 组自身每缺陷重复逐位相同；后续完整 GA 接线在首候选前发现 kwargs 漏必需 `volume/ras_to_voxel`。v1 的阶段通过不证明完整消费者可调用。
- v2：补这两个 context 只读共享字段，数值计算未改；新的完整签名绑定与 readonly identity 检查通过，24/29 组仍逐位相同。整个自产 NPZ 文件 SHA 与 v1 一样。
- 内部矩阵 16 个数值精确，7 个正负零位模式不同；`all_internal_bitwise_reproduction=false` 保留。被消费的输出严格复现为 true。
- 15 原合同及补齐后的 16 合同通过；跨缺陷当前几何传递只有合同证据。完整 GA、最终网格、整例速度与指标等效在本报告范围内均未评估。

## 文件

| 文件 | 实际内容 |
| --- | --- |
| [report_v1.json](reports/report_v1.json) / [report_v2.json](reports/report_v2.json) | 原机器 JSON 原 bytes，阶段数值、误差、重复性、实际模块与程序 SHA、时间边界 |
| [complete_v1.json](reports/complete_v1.json) / [complete_v2.json](reports/complete_v2.json) | 与对应报告 SHA 绑定的完成标记；另有原 EXIT 文本 |
| [source_manifest_v1.json](reports/source_manifest_v1.json) / [source_manifest_v2.json](reports/source_manifest_v2.json) | 实际 d76310419eb71b2d59e39d980de0e9e829ed9d41 加五文件最小 overlay；原版源码/库/程序没有转移入该候选包 |
| [source_binding.json](reports/source_binding.json) | 包大小/SHA、参考程序、阶段相位、说明纠正的 AST 检查与允许的发布字段 |
| [reference_repeats_v2.json](reports/reference_repeats_v2.json) | 四种参考相位两次文件 SHA，一一相同；不是整例参考重复性 |
| [contracts.json](reports/contracts.json) | 三次本地合同、实际 source/test/JUnit SHA；公共 XML 只去掉 hostname 字段 |
| [runtime_versions_v2.json](reports/runtime_versions_v2.json) | 未改 runtime prefix 的事后版本核对，不冒充计时中采样 |
| [当前脑图与误差](figures/v2/candidate_inputs_brain_and_error.png) / [图收据](figures/v2/candidate_inputs_brain_and_error.json) | 绑定 v2 报告与自产检查点；全部局部体素误差 0、内部矩阵正负零诊断 |

原影像、NPZ、二进制捕获、原程序、原代码和私有启动器留在授权服务器；公开表仅保留输入/程序 SHA、数值、完成门、版本和验证图。源码基线是实际打包时版本，不用后来发布 HEAD 替代。

## 复现

接口、全部参数、命名空间、数据结构、空间与单位、失败条件和具名示例见 [七节说明](../../../../docs/recon_all/TOPOLOGY_CANDIDATE_INPUTS.md)。验证工具为 [benchmark_recon_topology_candidate_inputs.py](../../../../tools/benchmark_recon_topology_candidate_inputs.py)，离线图工具为 [plot_recon_topology_candidate_inputs.py](../../../../tools/plot_recon_topology_candidate_inputs.py)。实际参考分别是创建局部 MRI 前后、原域全局统计及候选匹配参数；不能换成最终 subject 结果。

```bash
PYTHONPATH=src OMP_NUM_THREADS=4 NUMBA_NUM_THREADS=4 \
  MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
  python -m pytest -q tests/recon_all/test_topology_candidate_inputs.py \
  --junitxml=new_contract_output.xml
```

实际 stage 源码绑定最终 `topology_candidate_inputs.py` SHA `b3fc6e1e56549c3a98cf15cb31cfa42ae39949f45576796821ffbc415a9c7527`。完整 driver v1/v2 为 20.422733/14.135993 秒，另计导入 1.937002/5.390987 秒；首次 per-defect API 为 1.661410/0.029227 秒，重复为 0.014940/0.015733 秒。共享 CPU、跨时段、JIT 状态不同，且 v2 增加消费者签名检查，不能把这些差值当算法提速。CPU 四线程，没有初始化 CUDA，显存未测 null。

完整 GA 由另一独立任务运行；这里只交付其完整必需输入，不能据本阶段证明 10 分钟整例或最终拓扑质量通过。
