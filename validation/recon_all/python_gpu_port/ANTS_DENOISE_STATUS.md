# `AntsDenoiseImageFs` 去噪阶段：仓库内 Numba 实现

当前 `fnit-recon-all` 的 `ants_denoise` 阶段调用 [`denoise_volume`](../../../src/fnit/recon_all/ants_denoise_python.py)。运算核心位于 [`ants_denoise_core.py`](../../../src/fnit/recon_all/ants_denoise_core.py)，使用 NumPy 和 Numba 在 CPU 上执行；运行时不导入 ANTsPy，也不调用外部 `AntsDenoiseImageFs`。输入限定为三维 `uint8` MGH/MGZ，与当前 `recon-all` 的 `brain.mgz` 相同。

## 实际 T1 配对结果

2026-09-27 在 `headcw` 使用同一幅真实 T1 派生的 `brain.mgz`（256³，`uint8`，SHA-256 `1b7360d069b76c8a5a63f93f3c296dddeb4db725c4e2625e11ebfc156279f401`）。官方输出 `antsdn.brain.mgz` 的 SHA-256 为 `378548a3f58f74878450525a734c15ff19ce935dff913133f52aa149bb91d2ce`。Python CLI 完整读入、去噪并写盘后的对照见[机器可读结果](ants_denoise_numba_20260927.json)。

| 检查项 | 结果 |
| --- | ---: |
| 输出体素逐点相同 | 16,777,216 / 16,777,216 |
| 输出数组 SHA-256 | 两侧均为 `56c914332ae338a17ada33bc1ff10c78d8a8f36348ea80fabd954cfd2fc0e951` |
| 输入到输出发生变化的体素 | 两侧均为 1,039,074 |
| 仿射、MGH 前 284 字节 | 完全相同 |
| Python 输出与输入尾部标签 | 完全相同 |
| Python 输出与官方尾部标签 | 不同；官方解压后多 1 字节 |

压缩文件 SHA-256 不相同：Python 保留输入尾部标签，官方 `MRIwrite` 改写了尾部。体素、几何和 MGH 固定头部已通过，不能据此称 `.mgz` 文件字节相同。

同机一次观测：当前 Python CLI 从启动到写盘 **19.72 秒**，函数内部计时 **19.14 秒**，峰值 RSS **558,476 KiB**；此前相同输入的官方原生命令为 **27.58 秒**，旧 ANTsPy 包装为 **28.25 秒**。不同调用虽固定同一数据和 `headcw`，但不是重复测量；只说明这次单阶段观测，不代表整例 recon-all 的加速。CPU 是该实现的执行设备。

[函数说明、参数、输出与官方命令](../../../docs/recon_all/DENOISE_NUMBA.md)列出可复现的调用方式。其余 recon-all 步骤仍需各自通过同输入验收。
