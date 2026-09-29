# fslmaths 真实影像对照

[机器可读结果](real_data_20260929.json)保存了 2026-09-29 的 48 组裁剪块对照、5 组完整影像对照、2 组真实 BOLD 时间滤波对照、3 组 3D/4D 混合运算，以及整数输出和 `-range` 的检查。各组记录两端耗时、逐体素误差、图像形状和仿射差；主要运算另有 FNIT CUDA 峰值分配。输入文件的 SHA-256 已写入结果文件。

输入是已有 dMRI 处理记录中的真实 T1w 脑图和 6 分量 tensor，以及运动校正后的真实 BOLD 时间序列；裁剪块由原图的连续空间体素组成，没有生成模拟强度。完整 T1w 为 176×256×256，完整 tensor 为 104×104×72×6，BOLD 裁剪块为 16×16×16×490。FSL 6.0.7.4 在 headcw CPU 上执行，FNIT PyTorch 2.5.1 在 gpucw1 的 H100 GPU 0 上执行。FSL 只用于验证，FNIT 的运算路径不调用 FSL。

示例配对命令如下；两个命令使用同一张原始图和同一个操作序列：

```bash
# FSLDIR、FSLOUTPUTTYPE：官方 FSL 6.0.7.4 的安装位置和输出格式
export FSLDIR=/path/to/FSL/6.0.7.4
export FSLOUTPUTTYPE=NIFTI_GZ

# INPUT：真实 T1w 脑图；FSL_OUT：官方输出文件
INPUT=/path/to/verified/t1_brain.nii.gz
FSL_OUT=/path/to/benchmark/fsl_thr_bin.nii.gz
fslmaths "$INPUT" -thr 100 -bin "$FSL_OUT"

# FNIT_OUT：PyTorch 输出文件；--device：使用 H100 GPU 0
FNIT_OUT=/path/to/benchmark/torch_thr_bin.nii.gz
fnit-fslmaths --device cuda:0 "$INPUT" -thr 100 -bin "$FNIT_OUT"
```

GPU 0 在最终完整影像测试前已有 56,019 MiB/81,559 MiB 被其他进程占用，利用率为 98%。测试脚本在调用 FNIT 前执行 `torch.cuda.set_per_process_memory_fraction(0.20, 0)`；这限制当前进程的 CUDA 分配器上限。完整影像五组的实测峰值分配不超过 0.430 GiB，48 组裁剪块测试不超过 0.163 GiB，BOLD 时间滤波测试不超过 0.106 GiB。FSL CPU 与拥挤 GPU 的时间不是同硬件计算内核对照；计时含启动、NIfTI 读写和 gzip 压缩。

报告只证明列出的图像、选项组合和软件版本上的数值与时间。运行其他尺寸、其他核或多个组合前，应继续用同输入配对验证。模块功能、输入输出和选项边界见[使用说明](../../docs/fslmaths/README.md)。
