# TorchMCFLIRT

单被试三阶段 BOLD 运动校正。输入四维 NIfTI 和可选三维参考，输出校正图、FSL input→reference 矩阵及 `.par`。

CUDA 采样复用 TorchFLIRT 的精确 float32 运算，将 MCFLIRT 的行范围、逐次坐标累加、三线性采样、边界降权和参考值读取合并为一个 Triton kernel；NCC 计数和归约、三阶段 Brent 求解及逐帧初值传递沿用已有实现。同一次调用复用各帧重心；完整 float32 数据满足 4 GiB 和空闲显存四分之一的缓存限额时，三个阶段共用已上传帧，否则逐帧上传。

六自由度搜索复用未改变角度的 CPU 旋转矩阵及固定 voxel size 的采样矩阵。CUDA 每次调用、每个参考分辨率保留一份工作区，用 graph 重放原 NCC 归约；motion-only 最终样条采样也重放原 PyTorch 运算。三线性最终采样沿用既有路径。全部缓存只在本次调用中使用；RMS 位移在请求输出时计算。

接口与输出结构不变。项目 [Conda 环境](../../../environment.yml)已包含 Triton；CPU 和无法导入 Triton 的 CUDA 采样沿用 PyTorch 张量路径。

详细参数、Python 与命令行示例、原实现和真实数据对照见 [功能说明](../../../docs/mcflirt/README.md)；[benchmark](../../../validation/mcflirt/README.md)。
