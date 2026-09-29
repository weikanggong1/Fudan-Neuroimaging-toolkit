# recon-all 显存记录与验收口径

整例显存以**同一进程的实际 GPU 占用**和 PyTorch `max_memory_allocated`、`max_memory_reserved` 同时记录，统一保存字节数，并展示十进制 GB 和二进制 GiB。`nvidia-smi` 周期采样只能给出观察最大值，不能保证捕获连续峰值。

标准 CUDA runner 默认关闭 PyTorch 的 CUDA 分配缓存，以压低进程显存占用。此模式下 PyTorch 的 allocated/reserved 峰值接口返回 0，运行报告用 `gpu_memory_mode=no_cuda_allocator_cache` 标识，并不将 0 当作显存峰值；Talairach 子进程单独保留缓存与其统计。完整进程显存需要外部采样。现版整例的外部进程采样与 20 GB 预算仍待验收。默认允许 TF32；SynthStrip、SynthSeg 和 Talairach affine 的 FP32 例外按阶段记录。未启用 FP16/BF16。

2026-09-29 的单例开发运行在 SynthSeg 阶段记录到 allocated **21,089,711,616 字节**、reserved **24,761,073,664 字节**，已超过 20 GB；该运行仍在执行表面阶段，不能当作完成的整例。用同一真实 T1 和 GPU 作配对测试：第一份 float32 后验在第二次推理时暂存 CPU，分割图逐体素一致、体积 CSV 逐字节一致，allocated 降至 **18,843,125,248 字节**；再关闭分配缓存，`nvidia-smi` 周期采样的进程最大占用为 **18,450 MiB（约 19.35 GB）**。这只是 SynthSeg 单阶段，不证明整例低于 20 GB。[SynthSeg 配对记录](../../validation/recon_all/python_gpu_port/synthseg_memory_20260929.json)。Talairach affine 已改在子进程执行，冻结 `synthstrip.mgz` 的 XFM、LTA 与先前直接调用逐字节相同；其 allocated/reserved 峰值为 4,715,548,672/4,884,267,008 字节。下一次连续整例仍须测量父子进程的合计 GPU 占用，并核对后续输出。[Talairach 同输入记录](../../validation/recon_all/python_gpu_port/talairach_child_20260929.json)。

既有独立 SynthSeg 真实输入实验曾观察到 19,152 MiB（约 18.70 GiB、20.08 GB），历史 v2 整例曾观察到 20,824 MiB。这些数字来自旧版本和周期采样，不能作为当前流程满足 20 GB 的证据；原始 JSON 仍留在历史验证目录供核查。

## 参考文献与原实现

- [PyTorch CUDA 内存统计接口](https://docs.pytorch.org/docs/stable/generated/torch.cuda.memory.memory_stats.html)。
- [PyTorch 原实现代码库](https://github.com/pytorch/pytorch)。
