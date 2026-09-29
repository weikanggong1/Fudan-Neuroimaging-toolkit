# TorchEDDY 真实数据验证

[功能与用法](../../docs/eddy/README.md) · [机器报告](report.public.json) · [比较脚本](../../tools/validate_fsl2111_strict.py) · [源码对照](source_audit.md)

2026 年 9 月 29 日在 gpucw1 用一例真实 UKB 格式 AP/PA DWI 验证公开 `TorchEDDY` 的数值路径。AP 图像为 104×104×72×105，包含 5 个 b0、50 个 b≈1000、50 个 b≈2000；共同脑掩膜含 242,316 个体素。两次运行固定相同的 AP、bval、bvec、掩膜、FNIT TOPUP 系数与运动文件、acqparams、index、参考帧及 GP 种子 12345。输入及两份校正图的 SHA-256 见[机器报告](report.public.json)。不发布原始 DWI。

参考是 FSL 6.0.7.4 的 `eddy_cuda10.2`。FNIT 数值实现依据 [FSL EDDY 2111.0 源码](https://git.fmrib.ox.ac.uk/fsl/eddy)；服务器可执行文件的精确源码提交未核实。因此这里是对指定可执行文件的实际输出比较，不声称对所有 FSL 版本通用。

| 指标 | FNIT 对 FSL GPU | 验收界限 |
|---|---:|---:|
| 脑内 4D voxel×volume Pearson r | 0.999738 | ≥0.999 |
| 脑内 4D MAE，原图像强度单位 | 19.47 | ≤50 |
| b≈1000 每体素跨方向 r 中位数 | 0.996331 | ≥0.99 |
| b≈2000 每体素跨方向 r 中位数 | 0.998301 | ≥0.99 |
| 平移参数 MAE | 0.03590 mm | ≤0.10 mm |
| 旋转参数 MAE | 0.0003973 rad | ≤0.001 rad |
| 旋转 b-vector 平均无符号夹角 | 0.03842° | ≤0.10° |
| 官方离群切片召回 | 14/15，93.3% | ≥90% |

两侧在共同脑掩膜内的非零输出掩膜完全一致。FNIT 标出 14 个离群切片，均在 FSL 的 15 个之内，因此 outlier map **并非完全相同**；b-vector 最大夹角为 0.515°。运动 RMS 的 MAE 为 0.06960 mm，restricted movement RMS 为 0.03543 mm。输出已明显超过浮点舍入误差，不能称为逐体素等价。FSL 同输入、不同随机种子的两次 GPU 运行之间，4D r 为 0.999913、MAE 为 15.11，可作为本例随机波动的参照；这不消除 FNIT 与固定种子 FSL 的差异。

| 程序 | 计算设备 | wall time | 计时边界 |
|---|---|---:|---|
| FSL `eddy_cuda10.2` | gpucw1 GPU | 10:21.19 | 从启动至结果写盘 |
| FNIT TorchEDDY | gpucw1 H100 | 10:38.75 | Python 导入和 CUDA 初始化后，`TorchEDDY.run` 至结果写盘 |

两次均为真实病例完整 105 帧，使用同一 GP 种子 12345。FNIT 计时不含 Python 导入和 CUDA 初始化，FSL 计时包含进程启动；共享 GPU 负载也未隔离，因此这两个数不能直接计算稳定加速比。本版 FNIT 的 CUDA allocation 峰值为 4,843,410,944 字节（约 4.51 GiB）。

本版把 cubic B-spline 权重中的布尔索引改为 `torch.where` 分段计算。一次真实体积的参数更新在修改前触发 2,615 次 `nonzero`；profiler 记录的 CUDA 同步调用从 2,716 次降至 124 次。同一进程在同一真实 DWI 体积上交替调用旧/新样条采样，单次耗时分别为 89–159 ms 和 5.5–7.6 ms，结果逐值相同。相同病例、GPU、种子的单轮 EDDY 试验，wall time 从 249.68 s 降至 147.69 s；它们是共享 GPU 上先后运行的两次观察值。完整八轮的校正 NIfTI 与修改前 SHA-256 相同，旋转 b-vector、16 列参数、两种 RMS 和离群图也逐值相同；对 FSL 的全部精度指标保持不变。

先前整链通过公开 `fnit.eddy.TorchEDDY` 接口运行。TOPUP 系数和脑掩膜与上面的单独基准逐文件 SHA-256 相同；GP 未指定种子，对固定种子 FSL GPU 达到 4D r=0.999727、MAE=21.46、两组 shell 中位数 r=0.996272/0.997858、离群切片重合 14/15，预设阈值全部通过。整链影像指标仍可参考；整链当时的时间早于本次等值加速改动，不代表新版的整链速度。

比较命令读取两份输出前缀、相同 mask 和 bval，核对 shape、affine、4D 图像、两组 shell 的体素内方向相关、旋转梯度、参数、RMS 与三份离群切片图：

```bash
python tools/validate_fsl2111_strict.py \
  --strict-root eddy/fnit_data \
  --fsl-root eddy/fsl_data \
  --mask nodif_brain_mask.nii.gz \
  --bvals AP.bval \
  --output eddy_comparison.json \
  --assert-targets
```

`--strict-root` 和 `--fsl-root` 都是不带 `.nii.gz` 的输出前缀；脚本据此读取对应 `.eddy_*` 文件。`--assert-targets` 在表中任一预设界限未通过时以非零状态退出。机器报告保存实测精度及输入、输出哈希，不包含病例号或源影像路径。

图像对照已在计算节点生成，但尚未进入公开仓库；仓库现有的旧 EDDY 图不对应新版输出，已移除。数值表使用完整 4D 图像，不依赖切片图。
