# Conda 最终 white 表面放置

`fnit.recon_all.final_white_conda.run_final_white(subject_dir, hemi, binary, assets_dir, threads=4)` 调用 Conda 从 FreeSurfer 8.2 源码编译的 `mris_place_surface`，生成单侧最终 white 表面。该函数已接入 `fnit-recon-all --native-white-preaparc` 的球面注释之后；当前串联尚未完成从原始 T1 到最终表面的整例验收。

## 输入与输出

函数先检查被试目录中的以下七个文件：

| `subject_dir` 内的输入 | 用途 |
| --- | --- |
| `mri/brain.finalsurfs.mgz`、`mri/wm.mgz`、`mri/aseg.presurf.mgz` | 强度、白质和边界搜索分割 |
| `surf/H.white.preaparc` | 有序输入网格及 rip 参照 |
| `surf/autodet.gw.stats.H.dat` | 被试灰质/白质阈值 |
| `label/H.cortex.label`、`label/H.aparc.annot` | 皮层 rip 掩膜及脑区注释 |

`H` 对应半球。函数写入 `surf/H.white` 与 `mri/mrisps.white.mgz`。后一诊断图由双侧调用共用，第二侧会覆盖第一侧，和 recon-all 相同。许可证通过外部 `FS_LICENSE` 传入；wrapper 将 `FREESURFER_HOME` 设为 `assets_dir`、`SUBJECTS_DIR` 设为被试目录的父目录。运行不依赖系统安装的 FreeSurfer，但需要该源码编译程序。

Python API：

```python
from fnit.recon_all.final_white_conda import run_final_white

result = run_final_white(
    subject_dir="/path/to/subjects/sub01",  # 含 mri/surf/label 的被试目录
    hemi="lh",  # 左半球；右半球用 rh
    binary="/path/to/recon-cpp-build/bin/mris_place_surface",  # Conda 编译的程序
    assets_dir="/path/to/assets",  # 外置 FreeSurfer 数据目录
    threads=4,  # CPU 线程数
)
```

`result` 的 `output` 是 `surf/H.white` 路径，`outvol` 是 `mri/mrisps.white.mgz` 路径，`seconds` 是放置阶段的墙钟秒数。

独立命令：

```bash
python -m fnit.recon_all.final_white_conda /path/to/subjects/sub01 lh \
  --binary /path/to/recon-cpp-build/bin/mris_place_surface \
  --assets-dir /path/to/assets --threads 4
```

命令行前两个位置参数分别是 `subject_dir`（被试目录）和 `hemi`（`lh`/`rh`）；`--binary` 指定 Conda 程序，`--assets-dir` 指定外置数据目录，`--threads` 指定 CPU 线程数（默认 4）。命令会打印上述返回字典。

保存的 FreeSurfer 8.2 `recon-all.log` 中对应命令从 `subject_dir/mri` 运行：

```bash
mris_place_surface --adgws-in ../surf/autodet.gw.stats.lh.dat \
  --seg aseg.presurf.mgz --threads 4 --wm wm.mgz \
  --invol brain.finalsurfs.mgz --lh \
  --i ../surf/lh.white.preaparc --o ../surf/lh.white \
  --white --nsmooth 0 --rip-label ../label/lh.cortex.label \
  --rip-bg --rip-surf ../surf/lh.white.preaparc \
  --aparc ../label/lh.aparc.annot --restore-255 --restore-255 \
  --outvol mrisps.white.mgz --rip-bg-lof
```

重复的 `--restore-255` 与末尾 `--rip-bg-lof` 均来自原日志。右侧将 `lh` 改为 `rh`。顺序可对照[固定源码的 recon-all](https://github.com/freesurfer/freesurfer/blob/d932c45b7941662ea380a05efef580568b98d41a/scripts/recon-all)；保存命令与首差分析见[早期诊断](../../validation/recon_all/python_gpu_port/native_cpp_conda_20260927/WHITE_PREAPARC_FIRST_DIVERGENCE_20260927.md)。

## 真实 T1 同输入测试

使用仓库中去标识的 `examples/data/sub-01_T1w.nii.gz`（SHA-256 `f20410a4efd8e6a05cd04d55730a4a5492ecf9ad1b234fe0fd4661e448270c6a`）。gpucw1 的隔离被试从保存的官方 FreeSurfer 8.2 被试软链接**全部七个阶段输入**，只检验 Conda 最终 white 命令与 wrapper，不检验自产 T1 到 white 的连续流程。可执行文件 SHA-256 为 `9a42f5d7b70a066daf12e67fb6a0924048b4778186ea772e8976722b226b35d5`；私有许可证由环境传入，没有复制。

| 左侧输出相对保存的官方结果 | 实测值 |
| --- | ---: |
| 有序顶点 / 面 | 双方均为 106,622 / 213,240；面顺序完全一致 |
| 体积几何信息 | 完全一致 |
| 顶点三维欧氏位移：均值 / P99 / 最大值 | 0.000358662 / 0.005435218 / 0.751783 mm |
| 位移 >0.1 mm 的顶点 | 57 / 106,622 |
| `mrisps.white.mgz` | 0 / 16,777,216 体素差；仿射和完整压缩文件 SHA-256 一致 |
| Conda wrapper 墙钟 / 峰值 RSS | 225.64 s / 1,142,784 KiB |
| 保存的官方命令墙钟 | 225.85 s；日期和负载不同 |
| 重新运行的官方 8.2 程序墙钟 / 峰值 RSS | 168.86 s / 1,200,960 KiB |
| 重跑官方 `lh.white` 对照保存结果 | 319,866 / 319,866 坐标分量一致；面及体积几何一致；最大位移 0 mm |
| 重跑官方 `mrisps.white.mgz` | 0 / 16,777,216 体素差；完整压缩文件 SHA-256 一致 |

Conda 输出接近官方，但未达到顶点完全一致；其误差与早期冻结输入诊断相同。官方程序使用相同七个输入重跑时则逐坐标重现保存的 white 和诊断图，表明本次 Conda 尾部误差并非官方程序在该试验中的随机重跑差异。三次耗时处于不同共享节点负载，不构成受控速度比。[原始对照](../../validation/recon_all/python_gpu_port/final_white_conda_20260927/official_frozen_lh.json)记录所有输入路径、SHA-256、输出指标和 wrapper 耗时；[运行日志](../../validation/recon_all/python_gpu_port/final_white_conda_20260927/official_frozen_lh.log)保留优化过程及资源用量，私有许可证路径已遮盖。用 [compare.py](../../validation/recon_all/python_gpu_port/final_white_conda_20260927/compare.py) 的 `CANDIDATE_SUBJECT OFFICIAL_SUBJECT WRAPPER_LOG OUTPUT_JSON` 参数可复核。另有[官方自重复对照](../../validation/recon_all/python_gpu_port/final_white_conda_20260927/official_binary_repeat_lh.json)和[日志](../../validation/recon_all/python_gpu_port/final_white_conda_20260927/official_binary_repeat_lh.log)。自产全输入及右侧仍待测试。

## 数值首差

冻结输入下，两份日志在第 0 轮、放置子迭代 008 首次出现打印数值差：官方 SSE 为 `231646.9`，Conda 为 `231647.0`。此前打印的 SSE 在显示精度内一致。两者都使用 4 线程，预处理数量和边界摘要相同，四轮外层优化的步长接受/拒绝顺序也相同。更早的 `VMPeak` 与耗时差异仅涉及资源。小的算术偏差可在后续迭代累积，但现有日志不能定位具体运算。

ELF `.comment` 显示官方程序含 GCC 4.8.5 对象，Conda 程序含 GCC 11.4.0 对象。Conda CMake 缓存为 `Release`（`-O3`），编译器默认参数包含 `-O2 -march=nocona -mtune=haswell -ftree-vectorize`；官方二进制已去符号，无法取得完整编译参数。Conda 构建链接其 `libgomp` 和 ITK VNL，官方程序链接系统 `libgomp` 且不链接 ITK VNL；两者的 `libm` 均为 `/lib64/libm.so.6`。官方二进制的源码修订号未从程序本身确认。当前不能将 57 个超阈值顶点归因于单一编译选项、库、线程调度或源码版本，因此未调整构建参数。
