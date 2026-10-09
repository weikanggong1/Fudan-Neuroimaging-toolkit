# 固定主曲率派生图验证（2026-10-09）

当前范围是K1/K2→BE/C/FI/S，不包含离散主曲率、完整curv.stats或原始T1整例。
已有连续局部拟合不能替代默认离散主曲率，因此本轮不改变生产默认。

## 机器可读证据

- `cpu_real_maps.json`、`gpu_real_maps.json`：公开ds000114 sub-07/sub-08双侧，
  4线程、3轮ABBA/BAAB。两张输入、四张冻结参考、代码和参考程序均记录哈希。
  BE/C/S逐位相同，FI最多1ULP；预设2ULP子算子容差16/16通过。
  Torch CPU/GPU与独立Numba源公式16/16逐位相同。
- `cpu_cuda_unit_tests.txt`：CPU/CUDA表达式与摘要、nibabel写出回归6项通过。
- `benchmark.py`：驻留计算、搬运API与完整读取/写出的不同计时范围。
- `refresh_precision.py`：从保存的CPU/GPU四图再核查原始FP32位模式，区分
  数值相等与signed zero位模式，保持原计时与原脚本哈希。
- `plot_real_maps.py`和`sub07_lh_curvature.png`：真实surface RAS网格的C及FI尾差图。

Numba公式循环不是原生完整curv.stats计时。冻结图来自现有Conda源码构建
程序，原生程序现场SHA-256记录在报告中，本轮未重新运行官方程序或验证
其重复性。摘要只有单元回归，尚无完整标准统计文本真实数据通过结论。
GPU显存只统计当前PyTorch峰值，不统计父子进程合计。

## 复现

```bash
PYTHONPATH=src python validation/recon_all/optimizations/20261009_curvature_derivatives/benchmark.py \
  --subject /data/sub07/attempt_01/subject /data/sub08/attempt_01/subject \
  --reference-program /data/fixed-native/bin/mris_curvature_stats \
  --device cuda:0 \
  --threads 4 \
  --repeats 3 \
  --code-version ACTUAL_COMMIT_AND_PATCH_HASH \
  --output /data/new-diagnostic/curvature
```

`--subject`为冻结自产subject目录，要求双侧smoothwm.K1/K2和BE/C/FI/S图；
`--reference-program`仅现场记录程序哈希，不执行参考程序；`--device`默认
cuda:0，支持CPU诊断；`--threads`固定Torch/Numba线程；`--repeats`为配对轮数；
`--code-version`标识实际冻结代码；`--output`是新的诊断目录，不能指向参考图。
报告没有把未测项标成0或通过。

完整中文接口、空间、单位、失败行为和原软件命令见
[派生图说明](../../../../docs/recon_all/CURVATURE_DERIVATIVES_TORCH_20261009.md)。
