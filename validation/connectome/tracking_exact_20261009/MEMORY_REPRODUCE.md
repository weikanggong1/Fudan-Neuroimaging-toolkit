# 完整 100k 追踪的进程显存审计

`audit_real100k_memory.py` 只运行一次本轮正式候选，使用冻结 benchmark 的真实 NIfTI 加载、tracking、CPU snapshot 与输出摘要函数。从 CUDA 初始化开始采样，只保留当前进程在所选 GPU 上的显存；覆盖输入 H2D、完整 tracking 和 snapshot D2H。监测秒数不加入正式 ABBA 表。

输出 JSON 记录输入/源码 SHA、参数、Torch allocated/reserved、进程显存采样、错误/缺失和最大间隔，并核对固定输出 SHA。不会将其他进程的显存相加，也不会把缺失值当零；没有保存 PID、GPU UUID 或文件路径。采样峰值不能证明采样间隔内没有更短峰值。

## 复现

输入是[本轮报告](README.md#2-输入版本与计时范围)中同 SHA 的真实 ds004666 FOD、5TT 和 GMWMI。冻结工具和候选源码必须匹配已记录 SHA；当前生产源码可直接作为候选。该验证工具没有独立 Python 计算接口，不增加生产 API。

```bash
FROZEN_BENCHMARK=/data/frozen/benchmark_tracking_v5.py
REAL_FOD=/data/inputs/fod_reference.nii.gz
REAL_FIVE_TISSUE=/data/inputs/five_reference.nii.gz
REAL_GMWMI=/data/inputs/gmwmi_reference.nii.gz
MEMORY_REPORT=/data/results/memory_audit.json

# 导出实际完成正式测试的 v5 工具；不是当前改过列名的工具。
git show a1f412462586faba6fb6eec6d705c71037ff2384:tools/benchmark_connectome_tracking_exact.py \
  > "$FROZEN_BENCHMARK"

python validation/connectome/tracking_exact_20261009/audit_real100k_memory.py \
  --benchmark-module "$FROZEN_BENCHMARK" \
  --candidate-tracking src/fnit/connectome/tracking.py \
  --fod-module src/fnit/connectome/fod.py \
  --fod "$REAL_FOD" --five-tissue "$REAL_FIVE_TISSUE" --gmwmi "$REAL_GMWMI" \
  --device cuda:0 --n-seeds 100000 --batch-size 8192 --seed 0 \
  --output "$MEMORY_REPORT"
```

| 参数 | 意义 / 默认值 |
| --- | --- |
| `benchmark-module` | 必需：冻结 v5 Python 工具，SHA 必须为 `bc689db2…`。 |
| `candidate-tracking` | 必需：正式 tracking 源码，SHA 必须为 `6ed4e6be…`。 |
| `fod-module` | 必需：两版正式测试使用的同一 FOD/SH 实现。 |
| `fod` | 必需：真实 float32 `[104,104,72,45]` WM FOD NIfTI。 |
| `five-tissue` | 必需：真实 float32 `[256,256,256,5]` 组织 PVE NIfTI。 |
| `gmwmi` | 必需：真实 float32 `[256,256,256]` 界面权重，与 5TT 网格一致。 |
| `output` | 必需：审计 JSON 输出路径；失败也保留已完成记录。 |
| `device` | 单 CUDA 设备，默认 `cuda:0`；CPU 不能通过此显存审计。 |
| `n-seeds / batch-size / seed` | 默认且验收配置固定为 `100000 / 8192 / 0`。其他配置不冒充本轮 gate。 |
| `compile-arc` | 默认关闭；开启时配置不匹配默认模式 gate。编译模式的正式峰值另见 ABBA 报告。 |
| `sample-interval-seconds` | 请求采样周期，默认 `0.5` 秒，实际间隔包含查询延迟。 |
| `query-timeout-seconds` | 单次 nvidia-smi 查询超时，默认 `3` 秒；超时保留。 |
| `max-sample-gap-seconds` | 监测覆盖最大允许间隔，默认 `2` 秒；必须不小于请求周期。 |

Torch allocator cap 固定为 18,000,000,000 bytes。通过采样审计还要求：本进程采样峰值严格小于 20,000,000,000 bytes，H2D 到 D2H 的首尾被有效样本覆盖，中途没有失败/缺失，实际最大间隔不超过声明值，并且默认输出 SHA 相同。`continuous_upper_bound_proven=false` 始终保留。

## 与官方和版本的关系

这不是新追踪算法，也不增加原软件调用；MRtrix 比较及命令见[主验证说明](README.md#4-官方对照与脑图)。采用既有 Conda 环境的 PyTorch、nibabel、NumPy 和系统 nvidia-smi，不需要 pynvml。测量结果及脚本 SHA 独立记录，不改正式时间样本；未测的配置不外推容量。
