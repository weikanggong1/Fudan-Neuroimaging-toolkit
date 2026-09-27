# Conda pial.T1 表面放置

`fnit.recon_all.pial_t1_conda.run_pial_t1(subject_dir, hemi, binary, assets_dir, threads=4)` 调用 Conda 从 FreeSurfer 8.2 源码编译的 `mris_place_surface --pial`，生成单侧 `pial.T1`。这是独立阶段；当前 `fnit-recon-all` 调度器尚未调用。其八个前置文件须由更早的被试阶段产生。

## 输入、输出与调用

`subject_dir` 是 FreeSurfer 风格的被试目录，`hemi` 为 `lh` 或 `rh`，`binary` 指向 Conda 编译的可执行文件，`assets_dir` 为外置 FreeSurfer 数据目录，`threads` 为正整数。函数启动前检查：

| `subject_dir` 内的输入 | 用途 |
| --- | --- |
| `mri/brain.finalsurfs.mgz`、`mri/wm.mgz`、`mri/aseg.presurf.mgz` | 强度、白质和边界搜索分割 |
| `surf/H.white` | 有序最终 white 网格，也用于排斥和内侧壁固定 |
| `surf/autodet.gw.stats.H.dat` | 灰质/白质阈值 |
| `label/H.cortex+hipamyg.label`、`label/H.cortex.label` | 皮层 rip 与内侧壁固定掩膜 |
| `label/H.aparc.annot` | 传给原生命令的脑区注释 |

`H` 对应半球。函数写入 `surf/H.pial.T1`，返回 `{"output": path, "seconds": wall_time}`；它不生成最终 `pial`、厚度、面积、顶点体积、曲率或脑区统计。许可证保持在外部 `FS_LICENSE`；wrapper 向程序传入 `FREESURFER_HOME=assets_dir` 和 `SUBJECTS_DIR=subject_dir.parent`。[Conda C++ 阶段说明](CONDA_CPP_STAGES.md)列出构建方法。

Python API：

```python
from fnit.recon_all.pial_t1_conda import run_pial_t1

result = run_pial_t1(
    "/path/to/subjects/sub01", "lh",
    "/path/to/recon-cpp-build/bin/mris_place_surface",
    "/path/to/assets", threads=4,
)
```

独立命令：

```bash
python -m fnit.recon_all.pial_t1_conda /path/to/subjects/sub01 lh \
  --binary /path/to/recon-cpp-build/bin/mris_place_surface \
  --assets-dir /path/to/assets --threads 4
```

保存的 FreeSurfer 8.2 `recon-all.log` 中对应命令从 `subject_dir/mri` 运行：

```bash
mris_place_surface --adgws-in ../surf/autodet.gw.stats.lh.dat \
  --seg aseg.presurf.mgz --threads 4 --wm wm.mgz \
  --invol brain.finalsurfs.mgz --lh --i ../surf/lh.white \
  --o ../surf/lh.pial.T1 --pial --nsmooth 0 \
  --rip-label ../label/lh.cortex+hipamyg.label \
  --pin-medial-wall ../label/lh.cortex.label \
  --aparc ../label/lh.aparc.annot \
  --repulse-surf ../surf/lh.white --white-surf ../surf/lh.white \
  --restore-255
```

右侧将文件名及半球参数的 `lh` 改为 `rh`。阶段顺序见[固定的 recon-all 源码](https://github.com/freesurfer/freesurfer/blob/d932c45b7941662ea380a05efef580568b98d41a/scripts/recon-all)。

## 真实 T1 同输入测试

参考被试来自去标识的 `examples/data/sub-01_T1w.nii.gz`（SHA-256 `f20410a4efd8e6a05cd04d55730a4a5492ecf9ad1b234fe0fd4661e448270c6a`），由 FreeSurfer 8.2 重建。隔离的左侧测试被试从保存的官方被试软链接**全部八个前置文件**；官方 pial 几何不是输入。因此测试的是正确上游输入下的 Conda C++ `pial.T1` 命令，不是自产 T1 到 pial 的连续重建。[对照 JSON](../../validation/recon_all/python_gpu_port/pial_t1_conda_20260927/official_frozen_lh.json)记录各输入 realpath 和 SHA-256；[运行日志](../../validation/recon_all/python_gpu_port/pial_t1_conda_20260927/official_frozen_lh.log)记录优化过程和资源。可用 [compare.py](../../validation/recon_all/python_gpu_port/pial_t1_conda_20260927/compare.py) 的 `CANDIDATE_SUBJECT OFFICIAL_SUBJECT WRAPPER_LOG OUTPUT_JSON` 参数复核。

| 左侧输出相对保存的官方结果 | 实测值 |
| --- | ---: |
| 有序顶点 / 面 | 双方均为 106,622 / 213,240；面顺序完全一致 |
| 体积几何信息 | 完全一致 |
| 完全一致的坐标分量 | 28,062 / 319,866 |
| 顶点三维欧氏位移：均值 / P99 / 最大值 | 0.0296608 / 0.229365 / 1.669266 mm |
| 位移 >0.1 mm 的顶点 | 5,571 / 106,622 |
| Conda wrapper 墙钟 / 峰值 RSS | 245.32 s / 828,564 KiB |
| 保存的官方 `recon-all.log` 阶段耗时 | 3.68 min（220.8 s）；另一次运行与负载 |

Conda 程序 SHA-256 为 `9a42f5d7b70a066daf12e67fb6a0924048b4778186ea772e8976722b226b35d5`，wrapper 源码 SHA-256 为 `61450061862bf4b77d42b879e59c4f220c45826a1fa806317b16b149d011e1a0`。八个输入的哈希与真实路径均和保存的官方被试对应。Conda 日志有 42 个被接受的步长；官方和 Python 同输入测试为 41 步。该 Conda 输出未达到逐顶点一致，现有证据不足以用它替换已通过同输入检验的 Python 阶段。

独立的 [Python pial 放置](PYTHON_PIAL_PLACEMENT.md)曾在正确官方输入上逐顶点重现保存的官方左侧几何，在 headcw 耗时 1,240.8 s；另一轮官方 C++ 重跑耗时 116.05 s。保存日志的 220.8 s、官方重跑的 116.05 s、本次 Conda 的 245.32 s 处于不同运行和负载，不能作为受控速度比。该局部试验中 Conda C++ 比精确 Python CPU 实现快，但几何未通过。自产全输入、右侧及下游顶点图和脑区统计仍待验收。
