# 单幅 T1 前段的逐阶段数值定位

这份记录对应 FNIT 工作分支提交 `b8cd17bb441307d88dda78ba8f56e77695b595a8`。两例真实 T1 从原始 NIfTI 连续运行到 `filled.mgz`，参考结果由独立的 FreeSurfer 8.2 benchmark 生成。代码、输入、11 项权重、102 项资产和候选及参考程序的大小与 SHA-256 见[机器清单](../../validation/recon_all/python_gpu_port/volume_parity_20260930/provenance.json)。[逐阶段比较脚本](../../validation/recon_all/python_gpu_port/compare_volume_prefix.py)、[Conda N4 浮点比较](../../validation/recon_all/python_gpu_port/compare_n4_float.py)和[四组注册诊断](../../validation/recon_all/python_gpu_port/replay_em_cross.py)保留了复现入口。四组交叉输入只用于隔离诊断；生产流程没有读取官方输出。

## 本次修复：conform 的第一处差异

`conform_volume(input_file, output_file, *, device="cuda:0", slab=8) -> None` 读取三维 MGH/MGZ 格式的 `rawavg.mgz`，在 PyTorch 设备上完成默认 1 mm 冠状位、三线性重采样，写出 uint8 的 `orig.mgz`。输入为原始 T1 网格；输出通常是 256³ 的 conform 网格，仿射以 mm 表示 RAS 坐标。函数不返回图像；输入不是三维 MGH/MGZ、数据类型不是 float32/uint8 或 `slab < 1` 时抛出 `ValueError`。显存由 `slab` 控制，默认一次处理 8 个输出切片。它对应 FreeSurfer 的 `mri_convert --conform` 默认路径；[原实现](https://github.com/freesurfer/freesurfer/tree/d932c45b7941662ea380a05efef580568b98d41a)中需同时检查 `mri_convert`、`MRIgetResampleMatrix` 和 VNL 矩阵求逆。

内部 `_det3(m)` 接收 3×3 float32 矩阵并返回 float32 行列式。修复改变了余子式乘法的求值顺序，使随后的单精度求逆与参考 VNL 顺序一致；没有加入被试或体素特例。sub-02 的原始导入、原网格强度缩放、目标网格和仿射均已相同。旧代码求得的目标到源体素变换第 2 行平移为 `-1.283172607421875`，参考为 `-1.2831649780273438`；修复后相同。旧差异使 195 个接近半整数的插值值量化到相邻 uint8 值。修复后两例 `orig.mgz` 的 16,777,216 个体素、仿射和数据类型均与参考一致。[修复前](../../validation/recon_all/python_gpu_port/volume_parity_20260930/sub02/conform_before.json)和[修复后](../../validation/recon_all/python_gpu_port/volume_parity_20260930/sub02/conform_after.json)记录了坐标、插值及写出检查。真实 sub-02 仿射的逆矩阵回归在 `tests/recon_all/test_conform_gpu.py` 中。

```python
from fnit.recon_all.conform_gpu import conform_volume

conform_volume(
    input_file="/data/sub02/mri/rawavg.mgz",  # 三维原始网格 T1，float32 或 uint8
    output_file="/data/sub02/mri/orig.mgz",   # conform 网格的 uint8 MGH/MGZ
    device="cuda:0",                         # PyTorch 重采样使用的设备
    slab=8,                                  # 每次处理的输出切片数
)
```

## 两例连续前段

下表是相对各自官方参考的不同体素数。修复前列重新读取了此前完整整例的文件，来源分别为 `9ae7939` 和 `9ae7939` 加后来发布于 `b66ff97` 的 SynthSeg CPU 修复；它们是**旧源码产物**。修复后列是本提交从原始 T1、空目录连续运行的 18 个前段阶段。体积均为 256³，仿射相同；sub-01 的 `norm`、`brain` 和 `wm.seg` 的 P99 体素绝对误差分别为 1、2 和 1，其余表列体积的 P99 为 0。最大值和完整路径哈希见[sub-01](../../validation/recon_all/python_gpu_port/volume_parity_20260930/sub01/connected_prefix_compare.json)与[sub-02](../../validation/recon_all/python_gpu_port/volume_parity_20260930/sub02/connected_prefix_compare.json)报告。

| 输出 | sub-01 修复前 → 后 | sub-02 修复前 → 后 |
| --- | ---: | ---: |
| `orig/001.mgz`、`rawavg.mgz` | 0 → 0 | 0 → 0 |
| `orig.mgz` | 0 → 0 | **195 → 0** |
| `nu.mgz` | 2 → 2 | **40,320 → 29** |
| `T1.mgz` | 2 → 2 | **223,968 → 27** |
| `brainmask.mgz` | 2 → 2 | **85,003 → 17** |
| `norm.mgz` | 490,692 → 490,692 | **251,873 → 41** |
| `brain.mgz` | 703,485 → 704,759 | **471,291 → 4,502** |
| `wm.seg.mgz` | 207,731 → 207,890 | **121,685 → 2,318** |
| `wm.asegedit.mgz` | 140,107 → 140,235 | **91,149 → 1,805** |
| `wm.mgz` | 140,229 → 140,364 | **91,222 → 1,812** |
| `filled.mgz` | 4,956 → 4,881 | **2,987 → 112** |

`wm.seg`、`wm.asegedit` 与 `wm` 是带非零强度值的白质图，表中的体素数比较其数值，前景 Dice 则比较非零支持集，不把每个灰度级误称为标签。修复后 `wm` 前景 Dice 为 0.991984 / 0.999861。真正以 127/255 标记左右半球的 `filled` 按标签计算 Dice：sub-01 为 0.994238 / 0.993535，sub-02 为 0.999927 / 0.999847。前段完成并不表示严格整例验收通过。

## 下一处差异：N4 的浮点输出

在同一主机、完全相同的 `orig.mgz`、几何、四级各 50 次迭代、4 倍缩小和 1 个 ITK 线程下，官方 N4 与 Conda `fnit_n4_itk` 的 float32 输出已在 uint8 量化**之前**出现差异。sub-01 有 640,426 个不同浮点体素，最大差 `3.0517578125e-05`，量化后 `nu0` 只差 2 个体素；sub-02 分别为 2,496,335 个、`7.62939453125e-05` 和 30 个。各自的浮点输出用同一舍入规则可逐体素复现各自 uint8 输出，故当前不能把原因归于写盘或舍入。详见[sub-01](../../validation/recon_all/python_gpu_port/volume_parity_20260930/sub01/n4_float.json)和[sub-02](../../validation/recon_all/python_gpu_port/volume_parity_20260930/sub02/n4_float.json)。当前 Conda 构建使用 ITK 5.4.7，参考程序含 ITK 4.13.2；版本及浮点实现差异仍需隔离实验确认，不能直接定为根因。[N4 API、参数、安装和官方命令](N4_ITK_CONDA.md)另有逐项说明。

## nu 与 brainmask 的交叉输入

隔离目录用同一个 Conda 源码构建的 `mri_em_register -uns 3 -mask brainmask.mgz nu.mgz RB_all_2020-01-02.gca transforms/talairach.lta`，随后对四组结果调用相同的 FNIT `run_ca_normalize`。下表以“官方 nu + 官方 brainmask”这一组为基准：

| 输入变化 | sub-01 LTA 最大差；norm 不同体素 | sub-02 LTA 最大差；norm 不同体素 |
| --- | ---: | ---: |
| 仅改为 FNIT nu | 0.000573516；490,692 | 7.45×10⁻⁹；30 |
| 仅改为 FNIT brainmask | 0；0 | 0；0 |
| 两者均改为 FNIT | 0.000573516；490,692 | 7.45×10⁻⁹；30 |

[sub-01](../../validation/recon_all/python_gpu_port/volume_parity_20260930/sub01/em_cross.json)和[sub-02](../../validation/recon_all/python_gpu_port/volume_parity_20260930/sub02/em_cross.json)机器报告保存输入、程序、图谱哈希、每组 LTA、norm 与耗时。sub-01 的两个 `nu` 体素使 EM 解切换到不同的局部结果，再传播到大量 `norm` 体素；这不说明 `mri_em_register` 自身实现有错。相同官方输入重跑的 LTA 两例均与归档官方 LTA 逐元素一致；sub-01 的 norm 完全一致，sub-02 的 norm 尚有 11 个体素差 1，仍须隔离其归一化阶段。

## 耗时、精度和验收边界

单次前段连续运行：sub-01 在 gpucw1 的 H100 GPU 与 4 CPU 线程上用时 1,227.80 秒；sub-02 在 nodecw10 的 CPU 与 4 线程上用时 993.28 秒。[阶段耗时](../../validation/recon_all/python_gpu_port/volume_parity_20260930/sub01/connected_prefix_run.json)分别列出 N4、SynthSeg、EM 等。GPU 使用进程父子树在**同一时刻**每 2 秒采样，最大为 19,411,238,912 字节，低于 20,000,000,000 字节；该值是采样最大值，不保证捕获连续峰值。测试期间 GPU 与其他作业共享，不据此声称加速。默认 TF32；SynthStrip、SynthSeg 和 Talairach affine 保留 FP32 例外，没有使用 FP16/BF16。

本页仅确认前段连续运行。整例输出完整性、网格质量、严格 138 项数值比较和最终脑区指标须独立报告；在当前提交的新整例完成前，旧版完整整例结果不能充当现版验收。物理隔离、无预装脑影像软件的环境尚未验证。

## 参考

- Fischl B. FreeSurfer. *NeuroImage*. 2012;62:774–781. [doi:10.1016/j.neuroimage.2012.01.021](https://doi.org/10.1016/j.neuroimage.2012.01.021)。
- [FreeSurfer 固定源码](https://github.com/freesurfer/freesurfer/tree/d932c45b7941662ea380a05efef580568b98d41a)。
- Tustison NJ, et al. N4ITK: Improved N3 Bias Correction. *IEEE Trans Med Imaging*. 2010;29:1310–1320. [doi:10.1109/TMI.2010.2046908](https://doi.org/10.1109/TMI.2010.2046908)。
- [ITK N4 原实现](https://github.com/InsightSoftwareConsortium/ITK)。
