# TorchFAST 去除 Surfa：真实 T1 配对验证

[功能说明](../../../docs/fast/README.md) · [机器报告](report.json) · [此前 FSL 算法基准](../README.md)

## 输入与条件

使用公开示例 `sub-02_brain.nii.gz`，即已去脑单帧 T1w，形状 **156×256×256**，文件 SHA-256 为 `ada1fc44fd439bf42736decadab78dad73aabf3619e417cc88a7634c7c3e8078`。旧版与新版均在 headcw 的同一 Conda 环境中运行 PyTorch 2.5.1、CPU、4 线程，FAST 算法文件 SHA-256 均为 `9ffed0035a618ec4d6dc1eea8deeadaddd071ced9b04b66593cb6bd4d1102638`。只改影像读取、内存体与 NIfTI 写出包装，不改分割算法及默认参数。

旧版包装层源码 SHA-256 为 `4eab9eabba7b57a4fc39bd496884eaca219acfd30f7c14f73e347277ec2d4ff7`；新版为 `54d3555c67a20e4df080cc7ba24d21d009cc877da4986204bff0fb3699409ffd`。新版执行时阻断 `surfa` 导入，运行结束时进程中没有 Surfa 模块。旧 Surfa 内存体仍可作为 Python 输入；新代码没有为了兼容而调用 Surfa API。

## 八张输出

两版各保存 CSF/GM/WM PVE、硬分类、PVE 分类、mixel 类型、偏置场和校正图为 `.nii.gz`。每张图都比较了全部 **10,223,616** 个体素、数据类型、仿射矩阵、NIfTI 文件头和压缩文件字节：

| 输出 | 不同体素 | 最大绝对差 | 仿射最大绝对差 | 完整文件字节 |
|---|---:|---:|---:|---|
| CSF PVE | 0 | 0 | 0 | 相同 |
| GM PVE | 0 | 0 | 0 | 相同 |
| WM PVE | 0 | 0 | 0 | 相同 |
| 硬分类 | 0 | 0 | 0 | 相同 |
| PVE 分类 | 0 | 0 | 0 | 相同 |
| mixel 类型 | 0 | 0 | 0 | 相同 |
| 偏置场 | 0 | 0 | 0 | 相同 |
| 校正图 | 0 | 0 | 0 | 相同 |

初版候选虽已达到零体素与零仿射差，NiBabel 默认的 qform/sform 编码和单位字段仍改变文件头。最终实现只在 FAST 结果包装中继承输入 NIfTI 的文件头，再按输出设置 float32/int32 类型；八个压缩文件因此与旧版逐字节相同。共享的 SynthStrip `Volume.save()` 没有改动。

## 阶段时间与运行入口

| 单次墙钟阶段 | 旧 Surfa 包装层 | 新 NiBabel 包装层 |
|---|---:|---:|
| 读取输入 + FAST 推理 + CPU 回传 | 38.39 s | 46.00 s |
| 保存八张压缩 NIfTI | 1.57 s | 2.02 s |

这是共享节点不同时刻的单次运行，没有配对负载控制；**不据此判断新旧速度优劣**。此前 10 例 H100 与 FSL FAST 的计时属于旧包装层的[历史算法基准](../README.md)，不充当本次 GPU 或 FSL 复验。曾尝试在 gpucw1 GPU1 运行配对；CUDA 初始化与 `cudaMemGetInfo` 即报告 OOM，本次没有 GPU 精度或速度验收。

目标 Conda 环境下，禁用 Surfa 导入后 FAST API 和 CLI 相关回归测试 **7 项通过**。正常依赖环境的完整 FAST 回归测试 **8 项通过**，其中一项仅检查未提供的 `batch` 命令报错。另以同一真实 T1 的 64³ 裁切，在 Surfa 导入阻断下执行独立 `fnit fast -b -B` 命令，成功写出八张 64³ 图。旧 Surfa 内存体输入的兼容性也单独检查。该裁切仅用于命令入口测试，精度对比使用上述完整真实 T1。

本记录验证 TorchFAST 影像包装层，不代表 FSL FAST 逐体素等价，也不代表完整 recon-all 已通过。原始影像与八张输出保留在验证节点，不提交仓库；可审查数值与文件 SHA-256 在[机器报告](report.json)。
