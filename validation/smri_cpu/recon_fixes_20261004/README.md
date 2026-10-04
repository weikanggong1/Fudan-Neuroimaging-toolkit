# 球面配准：保持逐点结果的 CPU 梯度平均并行化

[公开接口与原软件命令](../../../docs/recon_all/SPHERE_REGISTRATION_PERFORMANCE.md) · [CPU 记录](results/cpu_average.json) · [GPU 记录](results/gpu_average.json)

## 功能与实现

本版只优化 `RegistrationGradientAverager` 的 CPU 有序平均。每轮读取旧梯度、写入新缓冲区，不同顶点可独立并行；顶点内的邻接累加顺序、每步 FP32 舍入和最后的倒数乘法保持原值。没有改线搜索、步长接受、邻接或轮数。

Numba 线程上限取调用方 Torch 线程数与配置上限的较小值，并在正常返回和异常时恢复调用方线程掩码；小于 8,192 个顶点用一线程减少并行开销。CUDA 仍使用已有 Triton 内核。

## 真实完整表面结果

输入来自已完成 FNIT 重建的真实左半球：119,451 个顶点、238,898 个三角面。sphere 和 smoothwm 面序相同；先按既有距离/面积目标生成实际非零梯度，再给旧/新平均器相同输入。不是随机梯度或模拟表面。

CPU 使用 nodecw7 同一组八个物理核 `3,7,11,15,19,23,27,31`。每个轮数按旧→新→新→旧运行；下表为两次调用中位数。首调用 JIT 单列，不加到暖调用时间。

| 轮数 | 旧 CPU | 新 CPU | 逐点比较 |
|---:|---:|---:|---|
| 1 | 0.013541 s | 0.003137 s | 输出逐值相同，输入未修改 |
| 16 | 0.123674 s | 0.021065 s | 同上 |
| 256 | 1.317464 s | 0.314495 s | 同上 |

256 轮本组观测为 **4.189 倍**。旧/新首次调用含 JIT 为 0.284/0.767 s，新并行内核首次编译更久。所有梯度数组和源码的 SHA 在记录中。

这是平均算子的测试。没有在这版重新测完整双侧配准或 recon-all，也未解决原有拓扑、顶点对应及官方皮层分区差异；不能将算子倍数当作等价重建的加速。

## GPU 回归与参数

同一完整真实梯度的 1/16/256 轮 GPU 旧/新输出逐值相同，并与相应 CPU SHA 相同。256 轮旧/新中位数为 0.007663/0.007829 s；共享 GPU 下毫秒级波动不作为稳定速度变化。运行峰值 allocated/reserved 为 25.273/46.137 MB，CUDA 内核源码未改。

内部调用的 `neighbors` 为有序 `(N,K)` int64 邻接，`degrees` 为 `(N,)` int64 有效邻居数；`gradient` 为 `(N,3)` float32；`iterations` 为非负轮数；`device` 选择 CPU 或 CUDA。返回新梯度，保留 shape/dtype/输入设备，不修改输入。

```python
from fnit.recon_all.mris_register_average_numba import RegistrationGradientAverager

# 以下张量来自同一真实表面的既有配准准备步骤
gradient_averager = RegistrationGradientAverager(neighbors, degrees, device="cpu")
averaged_gradient = gradient_averager(gradient, iterations=256)
```

这个内部算子不新增单独产品 CLI。完整配准的 Python/CLI、原 `mris_register -curv -threads 8` 调用、输入/输出和参考文献见[功能说明](../../../docs/recon_all/SPHERE_REGISTRATION_PERFORMANCE.md)。[重放脚本](benchmark_gradient_average.py)接收真实 sphere/smoothwm 和两版源码路径，保存算子 ABBA、输出哈希与首调用时间。

## 最近版本与剩余范围

| 版本 | 变化 |
|---|---|
| 本版 | CPU 顶点间并行，原邻接顺序和 FP32 结果保留 |
| `f1cbdab1` | CPU 有序串行平均；CUDA 已接入有序 Triton 平均 |
| `c2485205` | 配准按 `device` 选择平均设备，静态 CSR 共享 |

本版的 27 项平均、blur 和 nonlinear 相关检查通过。官方全流程指标和有效顶点对应仍按[剩余清单](../REMAINING.md)验收，不由该算子测试替代。
