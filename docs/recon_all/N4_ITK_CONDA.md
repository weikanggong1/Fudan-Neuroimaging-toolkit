# N4 偏置场校正：Conda C++ 与 Python 入口

FreeSurfer 8.2 在 recon-all 中通过 `AntsN4BiasFieldCorrectionFs` 将 `orig.mgz` 校正为 `nu0.mgz`。当前 FNIT 用仓库内 [`n4_itk.cpp`](../../tools/n4_itk/n4_itk.cpp) 调用 Conda ITK 5.4.7 的 N4 类，使用 4 倍缩小、四级各 50 次迭代、收敛阈值 0、覆盖全体素的掩膜和单 CPU 线程。NiBabel 包装器处理 MGH/MGZ、几何、uchar 取整和临时文件。不调用 SimpleITK、ANTsPy 或已安装的 FreeSurfer。

## 安装与编译

标准流程从仓库根目录执行：

```bash
conda env create -f environment.yml
conda activate fnit
bash tools/setup_recon_all_native_conda.sh
```

`$CONDA_PREFIX/bin/fnit_n4_itk` 是安装后的程序。它链接当前 Conda 环境的 ITK 动态库，运行时也须使用该环境。只排查 N4 时可用 `bash tools/build_n4_itk_conda.sh /path/to/native-build` 单独编译；这一步不需要 FreeSurfer 源码或许可证。输入输出临时 raw 文件位于目标目录，运行结束后自动删除；256³ float32 输入和输出各约 64 MiB。

## 输入、输出和调用

`correct_volume(input_file, output_file, *, binary) -> None` 接受：

| 参数 | 含义 |
| --- | --- |
| `input_file` | 三维 `.mgh` 或 `.mgz` 原始 T1，通常为 `mri/orig.mgz`。体素转成 float32 后送入 N4。 |
| `output_file` | 待写入的 uchar 三维 `.mgh` 或 `.mgz`，通常为 `mri/tmp/nu0.mgz`；沿用输入的仿射与头信息。 |
| `binary` | 本仓库源码经 Conda 编译生成的 `fnit_n4_itk` 可执行文件路径。 |

函数返回 `None`；图像写入 `output_file`。输出数据类型为 uint8，维度和仿射与输入相同。MGH 尾部沿用输入，完整文件字节不保证与官方相同。Python 调用：

```python
from fnit.recon_all.n4_itk import correct_volume

correct_volume(
    input_file="/data/sub01/mri/orig.mgz",  # 三维原始 T1
    output_file="/data/sub01/mri/tmp/nu0.mgz",  # 输出的 N4 校正图
    binary="/path/to/native-build/bin/fnit_n4_itk",  # Conda 编译程序
)
```

单阶段命令行：

```bash
python -m fnit.recon_all.n4_itk \
  --i /data/sub01/mri/orig.mgz \
  --o /data/sub01/mri/tmp/nu0.mgz \
  --binary /path/to/native-build/bin/fnit_n4_itk
```

官方同一子步的参考命令为：

```bash
AntsN4BiasFieldCorrectionFs -i orig.mgz -o nu0.mgz --dtype uchar
```

官方 recon-all 后续还通过 `mri_nu_correct.mni --ants-n4` 做均值恢复和 uchar 归一化。FNIT 的 [`make_nu`](N4_WRAPPER_VALIDATION.md) 接收 `orig.mgz`、`nu0.mgz`、`talairach.xfm`，写入最终 `nu.mgz`。

## 连通前缀的 Python API

`run_input_n4_chain(t1, subject_dir, weights_dir, assets_dir, *, n4_binary, device, threads) -> dict` 从单幅 T1 顺序生成 `orig.mgz`、SynthStrip 图、Talairach 变换、`nu0.mgz` 和 `nu.mgz`。`t1` 是输入 NIfTI；`subject_dir` 是不存在或为空的输出被试目录；`weights_dir` 和 `assets_dir` 是已校验的外置权重与模板目录；`n4_binary` 是本页编译程序；`device` 是 PyTorch 推理设备；`threads` 是 CPU 线程数。返回字典包含各输出路径、`n4_seconds`、`n4_wrapper_seconds`、均值缩放和直方图分箱。示例：

```python
from fnit.recon_all.input_n4_chain import run_input_n4_chain

result = run_input_n4_chain(
    t1="/data/sub01_T1w.nii.gz",  # 单幅 T1 NIfTI
    subject_dir="/scratch/subjects/sub01",  # 空的被试目录
    weights_dir="/path/to/weights",  # 外置权重目录
    assets_dir="/path/to/assets",  # 外置模板目录
    n4_binary="/path/to/native-build/bin/fnit_n4_itk",  # Conda N4 程序
    device="cuda:0",  # 网络推理设备
    threads=4,  # CPU 阶段线程数
)
# result["nu0"] 和 result["nu"] 是写入磁盘的图像路径。
```

`run_input_brainmask_chain(...)` 接受相同的七个输入，在上述字典中再返回 `T1`、`brainmask`、归一化和掩膜阶段耗时；`run_input_ca_normalize_chain(...)` 也接受相同输入，继续返回 `talairach_lta`、`norm`、`ctrl_pts` 和 GCA 阶段报告。两条扩展链本次没有重跑；当前数值验收只覆盖固定输入的 N4 和 `nu` 后处理。完整整例使用[主入口](README.md)的 `native_bin_dir`，无须单独传 `n4_binary`。

## 真实 T1 比较

固定输入为 sub01 的 256³ `orig.mgz`，SHA-256 `d79723f94bfc149ff36c89094a3d734b888a03cecbc32dc57a22992e8a5e817f`。一次 headcw 测试中，`nu0.mgz` 相对官方有 **9 / 16,777,216** 个体素不同，均低 1 个灰度级；最终 `nu.mgz` 有 **8 / 16,777,216** 个体素不同，最大差 2。两步仿射及 MGH 前 284 字节相同。完整数值和哈希见[验证报告](../../validation/recon_all/python_gpu_port/n4_itk_conda_20260927/README.md)。

包装器单次耗时 107.89 秒，之前同输入的官方 N4 单次耗时 168.26 秒；两次处于不同共享负载，不能作为受控加速比。N4 仍运行于 CPU。

在 2026-09-29 的自产上游整例中，当前 `nu.mgz` 与归档官方结果仅有 2 个体素不同。另将固定 FreeSurfer 源码中的 `AntsN4BiasFieldCorrectionFs` **在同一 Conda 中编译**并运行于相同 `orig.mgz`，经同一 FNIT `make_nu` 后仍有 42 个体素不同；因此标准流程保留当前 N4 实现。[同输入的程序哈希和体素数](../../validation/recon_all/python_gpu_port/n4_source_probe_20260929.json)可复查。两处差异在连续链中足以影响 EM 注册；对冻结官方 `nu` 和 `brainmask` 重放当前 Conda `mri_em_register` 时，LTA 16/16 个矩阵元素与官方一致，见[输入敏感性记录](../../validation/recon_all/python_gpu_port/em_register_input_sensitivity_20260929.json)。这些结果不能代替整例皮层指标验收。

在 gpucw1 上以相同 `orig` 的体素和 MGH 头独立重跑官方 N4，后处理的 `nu` 与归档官方图逐体素一致，FNIT 同主机输出仍差 2 个体素；同一官方程序在 headcw 上重跑则差 34 个体素。[配对记录](../../validation/recon_all/python_gpu_port/n4_same_host_replay_20260930.json)区分了实现差异与主机差异。官方重跑只作为隔离的参考，不进入 FNIT 标准路径；两处前段差异也不足以单独解释后续所有表面误差。

## 参考文献与原实现

- Fischl B. FreeSurfer. *NeuroImage*. 2012;62(2):774–781. [doi:10.1016/j.neuroimage.2012.01.021](https://doi.org/10.1016/j.neuroimage.2012.01.021)。
- [FreeSurfer 固定源码提交](https://github.com/freesurfer/freesurfer/tree/d932c45b7941662ea380a05efef580568b98d41a)。
- [ITK N4 原实现代码库](https://github.com/InsightSoftwareConsortium/ITK)。
