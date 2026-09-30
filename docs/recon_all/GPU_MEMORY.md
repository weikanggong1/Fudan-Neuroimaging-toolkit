# recon-all 显存记录与验收口径

整例显存记录指定物理 GPU 上**同一时刻父子进程合计的实际占用**，另报可用的 PyTorch `max_memory_allocated`、`max_memory_reserved`。统一保存字节数，并展示十进制 GB 和二进制 GiB；预算为 20,000,000,000 字节。`nvidia-smi` 周期采样只能给出观察最大值，不能保证捕获连续峰值，也不能将不同时刻的父、子峰值相加。

标准 CUDA runner 默认关闭 PyTorch 的 CUDA 分配缓存，以压低进程显存占用。此模式下 PyTorch 的 allocated/reserved 峰值接口返回 0，运行报告用 `gpu_memory_mode=no_cuda_allocator_cache` 标识，并不将 0 当作显存峰值；Talairach 子进程单独保留缓存与其统计。完整进程显存需要外部采样。当前 `279e09f` 的 sub-01 已初始化 CUDA API 整例，指定物理 GPU 1 上父子进程同时刻合计最大采样值为 19,411,238,912 字节（19.41 GB、18.08 GiB），共 2824 行；实际间隔中位数 2 s、最大 20 s。观察值低于 20,000,000,000 字节，连续峰值仍未验证。[完整运行记录](../../validation/recon_all/python_gpu_port/current_full_runs_20260930.json)。默认允许 TF32；SynthStrip、SynthSeg、Talairach affine 和 MNI 非线性的 FP32 例外按阶段记录。未启用 FP16/BF16。

2026-09-29 的单例开发运行在 SynthSeg 阶段记录到 allocated **21,089,711,616 字节**、reserved **24,761,073,664 字节**，已超过 20 GB；该运行仍在执行表面阶段，不能当作完成的整例。用同一真实 T1 和 GPU 作配对测试：第一份 float32 后验在第二次推理时暂存 CPU，分割图逐体素一致、体积 CSV 逐字节一致，allocated 降至 **18,843,125,248 字节**；再关闭分配缓存，`nvidia-smi` 周期采样的进程最大占用为 **18,450 MiB（约 19.35 GB）**。这只是 SynthSeg 单阶段，不证明整例低于 20 GB。[SynthSeg 配对记录](../../validation/recon_all/python_gpu_port/synthseg_memory_20260929.json)。Talairach affine 已改在子进程执行，冻结 `synthstrip.mgz` 的 XFM、LTA 与先前直接调用逐字节相同；其 allocated/reserved 峰值为 4,715,548,672/4,884,267,008 字节。连续整例的输出已核对；父子进程的合计 GPU 连续峰值仍须测量。[Talairach 同输入记录](../../validation/recon_all/python_gpu_port/talairach_child_20260929.json)。

本轮 CLI 基线与已初始化 CUDA 的 Python API 分别使用固定启动器，从原始 T1 和空目录开始，源码、退出码与 CSV 均保留。默认关闭缓存的父进程不能提供有效 allocated/reserved；报告中可用的 Talairach 子进程峰值也不能当作整个调用的 PyTorch 峰值。非默认逻辑 `cuda:1` 的探针只核查统计指向指定设备；完整缓存开启的 Python API 尚未验证。[计时、采样与复现方法](BENCHMARK_METHODS.md)。

## 参考文献与原实现

- [PyTorch CUDA 内存统计接口](https://docs.pytorch.org/docs/stable/generated/torch.cuda.memory.memory_stats.html)。
- [PyTorch 原实现代码库](https://github.com/pytorch/pytorch)。
