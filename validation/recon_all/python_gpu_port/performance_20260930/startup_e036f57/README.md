# e036f57 GPU 启动诊断

本组记录失败，不属于有效整例 benchmark。生产源码固定为 `e036f57b62b99d2af4cd8853ab2f1e6d2a9f8c68`，仍使用 gpucw1 物理 GPU1、同一 Conda 环境、原始 sub-01 T1、权重及资产。没有降低精度、更换 GPU 或读取官方结果。

| 尝试 | 实际结果 | 外层退出码与时间 | 进程采样 |
| --- | --- | --- | --- |
| `full_sub01_e036f57` | 进入 FNIT 前，4 字节 float32 张量初始化报 CUDA OOM | 1，5.397 s | 两行采样均为 0；不能解释为零显存 |
| `full_sub01_e036f57_retry` | 同一位置失败 | 1，5.577 s | 同上 |
| `full_sub01_e036f57_retry2` | 初始化成功；SynthStrip 完成，Talairach 子进程加载模型时 OOM；FNIT 报 `failed_stage=input_talairach` | 1，77.687 s；FNIT 内部 50.101 s | 36 行，最大 2,447,376,384 字节；未捕获连续峰值 |

第三次包装器在 CUDA 初始化前显式设置 PyTorch 四线程；原生产 API 进入后本来就会设置四线程。此改动用于一致的启动线程预算，不能认定为 OOM 修复。Talairach 子进程沿用既有缓存开启、四线程及 FP32 例外。三个失败日志、采样 CSV、退出摘要、实际启动脚本和第三次失败报告保留在本目录；源码归档哈希见[源码清单](../source_e036f57_manifest.json)。

## 短对照

[线程设置对照](cuda_init_probe_e036f57.json)采用六个独立进程，交替保持默认线程或先设置四线程。五次成功，一次在小张量启动时失败，不能将线程设置认定为稳定解决方法。[分配与模块加载对照](cuda_allocation_probe_e036f57.json)进一步检查空分配、填充、CPU→GPU 复制、缓存开关、EAGER/LAZY 模块加载；六次中只有一次成功，其余在首次 `cudaMemGetInfo` 已报 OOM。对照设置 `CUDA_LAUNCH_BLOCKING=1`，不用于性能计时或生产运行。

当时驱动为 535.216.03，NVML 报 GPU1 约 28–31 GiB 空闲，GPU 利用率 100%。因此上述采样和小张量结果不能支持“FNIT 模型已超过 20 GB”的结论，也不足以指定某个驱动、线程或 allocator 为原因。缓存和模块加载对照未稳定消除失败；没有据此修改生产策略。

随后[设备映射对照](cuda_device_probe_e036f57.json)交替使用数字 `1` 与 NVML UUID，六个独立进程全部成功，实际 PyTorch UUID 均对应物理 GPU1，未发现编号错位。新版启动器用该 UUID 同时指定计算和采样设备，API 侧车另存实际 UUID；这明确设备身份，不证明此前 OOM 的原因或已稳定解决。后续新空目录 `full_sub01_e036f57_uuid` 的整例单独记录，不续接以上失败目录。

两个诊断脚本固定服务器路径，只调用 PyTorch 和 nvidia-smi，不调用脑影像软件。输入为当前环境及声明源码，输出为本目录 JSON；每一行保留所执行 Python 语句、环境选项、退出码、完整错误和秒数，无独立官方命令。它们不写标准重建目录，也不作为精度 benchmark。GPU 整例仍需从原始 T1 和新空目录完成，不能续接第三次失败目录来声称连续整例。
