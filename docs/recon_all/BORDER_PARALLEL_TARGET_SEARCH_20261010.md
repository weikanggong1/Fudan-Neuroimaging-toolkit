# 表面目标搜索的独立顶点并行

## 1. 功能简介

`compute_border_values_first_pass()` 新增显式
`target_search_backend="serial"/"parallel"`。两种后端使用同一 Num ba 搜索
函数体；并行仅分配互不依赖的顶点目标搜索，不改变同一个顶点内部的候选
扩展、峰筛选、float32 舍入、原有五轮有序目标平均或表面接受顺序。
默认 `serial` 保留。两 dispatcher 使用不同 JIT 缓存身份，
`fastmath=False`，没有新依赖，不启用半精度，也不修改调用方线程预算。

```mermaid
flowchart LR
    A[固定 MRI / 同序表面状态] --> B[逐顶点完整边界搜索]
    B --> C[串行或既定线程并行]
    C --> D[原有有序目标平均]
    D --> E[原有四轮梯度与有序接受]
    E --> F[完整源清理与输出]
```

## 2. Python 调用、输入输出和参数

```python
import numpy as np
from fnit.recon_all.place_surface_border import compute_border_values_first_pass
from fnit.recon_all.thread_budget import thread_budget

with thread_budget(threads=2):              # 每半球 worker 的总 CPU 线程预算
    border_state = compute_border_values_first_pass(
        volume=prepared_volume,             # uint8 (W,H,D)，已有预处理 MRI
        segmentation=aseg_volume,           # int32 (W,H,D)，同 MRI 网格标签
        surface_xyz=current_coordinates,    # float32 (N,3)，surface RAS，毫米
        surface_normals=current_normals,     # float32 (N,3)，当前单位法向
        original_xyz=original_coordinates,   # float32 (N,3)，固定原始坐标，毫米
        ripped=ripped_vertices,              # bool (N,)，True 为排除顶点
        previous_values=previous_targets,    # float32 (N,)，上轮强度，初始 -1
        sras2vox=surface_ras_to_voxel,        # float32 (4,4)，毫米到体素索引
        thresholds=np.asarray(
            [inside_hi, border_hi, border_low, outside_low, outside_hi],
            dtype=np.float64,
        ),                                  # 自产灰白统计的五项强度阈值
        hemisphere="lh",                    # lh/rh，决定对侧标签停止条件
        surface="white",                    # white/pial，默认 white
        sigma=2.0,                          # 梯度尺度，体素，默认 2
        max_thickness=10.0,                  # 搜索限距，毫米，默认 10
        target_search_backend="parallel",   # 显式独立搜索并行；默认 serial
    )
```

返回六项：`(N,) float32` 目标强度、法向距离/mm、峰导数，
`(N,3) float32` 目标 surface RAS/mm，`(N,) int32` 标记和
`(N,) float32` 实际 sigma。峰导数回退时是 0，不是完整上轮 `mean`；
完整放置不使用该诊断量。强度与坐标 dtype 转换和原契约相同。
MRI 须同网格，顶点数组须同序有限值；函数不承担全部 shape/越界检查。
非法半球、表面或搜索后端抛 `ValueError`，不可转换输入抛 NumPy 异常；
JIT、并行运行异常直接传播，无静默回退。

完整 `place_white_preaparc()`、`place_final_white()` 和 `place_pial_t1()`
均新增相同具名参数，默认 `serial`；初始及轮间四次搜索全部传递该选项。
其他输入、坐标及参数分别见[preaparc](PYTHON_WHITE_PREAPARC.md)、
[最终 white](PYTHON_FINAL_WHITE.md)和 [pial](PYTHON_PIAL_PLACEMENT.md)。
返回增加 `target_search_backend` 和 `target_search_threads`，后者为实际
Num ba 并行掩码，串行报告 1，不等于整个进程或多库总线程数。
`thread_budget()` 在退出时恢复调用方状态；双半球每 worker 应各自分配
两线程，不能分别使用整个进程总预算。

## 3. 命令行

子函数没有独立生产 CLI。完整 preaparc 接口增加
`--target-search-backend serial/parallel`，只能和 `--complete` 使用；
原首轮诊断接口不扩展此选项。

```bash
# 完整四轮：两个 CPU 线程，显式 GPU 只负责既有候选与源清理。
NUMBA_NUM_THREADS=2 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 \
python -m fnit.recon_all.place_white_preaparc_python \
  /data/fnit/subject lh /data/diagnostic/lh.white.preaparc \
  --complete \
  --max-steps 400 \
  --output-volume /data/diagnostic/lh.mrisps.wpa.mgz \
  --target-search-backend parallel \
  --sampling-backend cpu \
  --regularization-backend cpu \
  --candidate-backend torch_snapshot \
  --candidate-grid-cells-per-axis 3 \
  --retained-mht-backend compiled \
  --cleanup-marking-backend source_torch \
  --cleanup-candidate-grid-cells-per-axis 3 \
  --device cuda:0
```

完整冻结对照用
`validation/recon_all/python_gpu_port/benchmark_border_parallel_stages.py`，
输入是自产 subject、哈希匹配的已有 benchmark、单独冻结代码 overlay；
每次使用全新输出目录。参考仅在阶段结束后读取。

```bash
# 同一个完整阶段依次运行串行控制和并行候选，包含 JIT、读写与同步。
python validation/recon_all/python_gpu_port/benchmark_border_parallel_stages.py \
  --source-subject /data/fnit/self-generated-subject \
  --baseline-directory /data/benchmark/existing-same-input-case \
  --candidate-directory /data/frozen/explicit-target-backend \
  --output-directory /data/benchmark/new-parallel-stage \
  --code-base-commit ACTUAL_BASE_COMMIT \
  --stage preaparc \
  --hemisphere lh \
  --threads 2 \
  --device cuda:0 \
  --order serial_parallel
```

## 4. 原软件调用

该算子对应固定 `mris_place_surface` 内部
`MRIScomputeBorderValues_new()`，没有独立官方命令。完整白质和 pial
命令分别见上方三个阶段专页。固定源表达式的外侧 1 mm float32 修复
仍保留，见[误差定位与 12 次完整同输入回归](WHITE_BORDER_TARGET_PRECISION_20261010.md)。
并行方案没有更换原阈值或改用近似 GPU 采样。

## 5. 当前版本精度、时间和资源

已完成的四线程隔离回放使用公开 sub06 LH 四轮已录取原生状态：
16 次热 ABBA 调用的全部六项输出逐元素相同。各轮串行中位数之和
31.069 s，并行 8.955 s，局部搜索快 3.47 倍；首次 JIT 16.542 / 9.411 s
另计。此回放不包含目标平均、完整碰撞或写出，不能称完整阶段或整例提速。
机器收据见[报告](../../validation/recon_all/optimizations/20261010_preaparc_border_precision/parallel_targets_report.json)。

新接口在两线程预算下完成同一个公开 sub06 LH 自产 `79a41cdd` 输入的
完整串行/并行配对，冻结基础 `54f681dc` 与四项实际 overlay 源码 SHA。
双方初始及轮间四次搜索的六项字段、全部接受坐标 SHA/试步决定、四轮
终点、最终完整清理及有序几何全部相同；预处理 MRI/阈值亦相同。
37 步 preaparc 与 44 步 pial 最终源相交门均为 0，与哈希录取的已有
同输入 Conda 参考最终坐标相同。本次没有新增原生重复或原生配对计时。

| 两线程完整 API | 串行控制 | 并行候选 | 本次墙钟减少 |
|---|---:|---:|---:|
| preaparc，37 步 / 17,27,34,37 | 254.914 s | 230.663 s | 9.51% |
| pial，44 步 / 27,33,37,44 | 326.640 s | 321.003 s | 1.73% |

完整时钟含自产阈值（preaparc）、读取/传输/首次 JIT、全部四轮、最终清理、
写出、透明轨迹记录和边界同步；导入、CUDA 初始化与输入复制/哈希在计时前。
这是每阶段一次共享负载配对，不能外推为稳定吞吐或整例提速。pial 碰撞
230.386→236.120 s，仍占主要时间；目标更新 23.393→21.111 s，较小的
完整收益符合当前瓶颈。preaparc 目标更新 55.289→48.101 s，这一子段
还包含 rip、法向和有序目标平均，不能称纯搜索时间。

搜索仍默认 serial；显式 parallel 的两个完整阶段通过不会更改生产默认。
实际 Num ba 掩码为两线程、后端 TBB，退出时恢复；不调整 Torch interop。
TF32 两项开启，没有半精度。三项 CUDA 分配器环境变量均未设置，
使用环境默认策略；preaparc allocated/reserved 716,507,648 / 1,539,309,568
字节，pial 716,382,720 / 3,082,813,440 字节（各自控制和候选相同）。
父子进程合计 GPU 占用未测，不据此宣布整例 20 GB 通过。
全部机器收据与源清单见[完整报告](../../validation/recon_all/optimizations/20261010_border_parallel_search/)。

相关真实表面/MRI 修复图见[白质目标定位页](WHITE_BORDER_TARGET_PRECISION_20261010.md#5-当前精度耗时与可视化)。
当前图是源舍入修复前后；新并行阶段的独立精度结论来自完整机器收据，
两后端最终所有顶点坐标相同，不另以近邻形状指标代替逐顶点检查。

## 6. 更新和 benchmark 记录

| 版本 / 范围 | 完成状态 |
|---|---|
| `199cfd52` 原表达式修复 | 两例四侧 preaparc、final white、pial 共 12 次完整同输入最终几何 0 差异 |
| 独立四线程 prange 回放 | 六项字段相同；搜索 31.069→8.955 s，不是整阶段提速 |
| 显式双 dispatcher 接口 | 默认 serial，47 项聚焦契约通过；两线程完整 preaparc/pial 轨迹、目标和最终几何全 0 差异 |

旧冻结源码和失败记录保留；新接口独立冻结，不覆盖已运行整例。

## 7. 参考文献和源码

- [固定原边界搜索源码](https://github.com/freesurfer/freesurfer/blob/d932c45b7941662ea380a05efef580568b98d41a/utils/mrisurf_mri.cpp)。
- [固定完整放置程序](https://github.com/freesurfer/freesurfer/blob/d932c45b7941662ea380a05efef580568b98d41a/mris_make_surfaces/mris_place_surface.cpp)。
- Dale AM, Fischl B, Sereno MI. Cortical surface-based analysis. I.
  Segmentation and surface reconstruction. *NeuroImage* 1999;9:179–194.
  [DOI](https://doi.org/10.1006/nimg.1998.0395)。
