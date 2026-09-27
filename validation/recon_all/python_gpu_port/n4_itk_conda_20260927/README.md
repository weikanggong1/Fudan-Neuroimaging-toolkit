# N4 Conda C++ 单阶段实测

2026-09-27 在 headcw 用固定的真实 sub01 T1 `orig.mgz`（SHA-256 `d79723f94bfc149ff36c89094a3d734b888a03cecbc32dc57a22992e8a5e817f`，256³ uint8）测试。仓库源码经 Conda GCC 11.4.0、ITK 5.4.7 从干净目录编译；产物 SHA-256 为 `b6a50763053193e5ada2e50ca7e257c2df8d3c9b05073d55f1a06d7164e18117`。`ldd` 没有 FreeSurfer、SimpleITK 或缺失库；在 gpucw1 的 glibc 2.17 节点启动成功。没有安装或调用 FreeSurfer/ANTsPy/SimpleITK 运行包。官方程序只用于隔离对照。

| 输出 | 与同输入官方结果一致的体素 | 最大灰度差 | 仿射 / MGH 前 284 字节 |
| --- | ---: | ---: | --- |
| `nu0.mgz` | 16,777,207 / 16,777,216 | 1 | 相同 / 相同 |
| 经现有 Python 后处理的 `nu.mgz` | 16,777,208 / 16,777,216 | 2 | 相同 / 相同 |

`nu0.mgz` 的 9 处均比官方低 1，`nu.mgz` 的 8 处均位于这 9 个位置。输出保留输入 MGH 尾部，因此 `nu0.mgz` 尾部长度 1346 字节，官方为 1345；最终 `nu.mgz` 与所用官方后处理归档尾部也不同。文件哈希和具体比较结果在 [`report.json`](report.json)。此结果不能称为逐体素或完整文件一致。

一次真实输入墙钟：Python CLI（包括 NiBabel 解压、临时 raw I/O、C++ N4、MGZ 写盘）107.89 秒，最大 RSS 426,380 KiB；单独 Python `make_nu` 后处理 1.06 秒。此前同一输入的官方 N4 单次墙钟为 168.26 秒，但运行日期和共享负载不同，不能据此声称受控加速。N4 不使用 GPU；这份测试没有运行完整重建或皮层指标。完整入口的 N4 二进制由 `native_bin_dir/fnit_n4_itk` 提供。构建命令、API、输入输出和官方命令见[函数说明](../../../../docs/recon_all/N4_ITK_CONDA.md)。
