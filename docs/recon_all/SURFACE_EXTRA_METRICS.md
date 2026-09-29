# 表面 Jacobian、白质灰质对比与 SNR

[返回阶段索引](CONDA_CPP_STAGES.md) · [固定输出清单](../../src/fnit/recon_all/expected_outputs.py)

标准流程在球面配准后写 Jacobian，在最终 white、厚度和 aparc 注释完成后写白质灰质对比及脑区 SNR。三项由 FNIT Python 实现，运行时不调用系统 FreeSurfer。

## 球面面积比：`jacobian_map(...)`

`original` 是同侧 `surf/H.white.preaparc`，`mapped` 是 `surf/H.sphere.reg`；两者必须有相同顶点及有序三角面。`output` 写 `surf/H.jacobian_white`，每顶点一个 float32 面积比，顺序与输入网格一致；`device` 指定 CPU 或 CUDA。函数无返回值，文件保留面顶点序号对应关系。

```python
from pathlib import Path
from fnit.recon_all.surface_jacobian_gpu import jacobian_map

jacobian_map(
    original=Path("/data/subjects/sub01/surf/lh.white.preaparc"),  # 原始表面，surface RAS
    mapped=Path("/data/subjects/sub01/surf/lh.sphere.reg"),  # 同序号配准球面
    output=Path("/data/subjects/sub01/surf/lh.jacobian_white"),  # 逐顶点面积比
    device="cpu",  # 此短阶段默认在 CPU 执行
)
```

对应官方命令：`mris_jacobian surf/lh.white.preaparc surf/lh.sphere.reg surf/lh.jacobian_white`。

## 白质灰质对比：`write_contrast_percentage(...)`

`subject` 包含 `mri/rawavg.mgz`、`mri/orig.mgz`、`surf/H.white`、`surf/H.thickness` 和 `label/H.cortex.label`；`hemi` 为 `lh` 或 `rh`；`output` 为 `surf/H.w-g.pct.mgh`；`device` 指定采样设备。函数返回输出 `Path`。输出是 `(顶点数, 1, 1)` 的 MGH float32 图，按同侧 white 的顶点顺序对齐，不是 3D 体素分割。计算在 white 法线向内 1 mm 与厚度 30% 处采样 `rawavg`，再计算成对归一化差的百分比。

```python
from pathlib import Path
from fnit.recon_all.vol2surf_contrast_python import write_contrast_percentage

output_path = write_contrast_percentage(
    subject=Path("/data/subjects/sub01"),  # MRI、white、厚度和皮层标签所在目录
    hemi="lh",  # 左半球；右侧使用 "rh"
    output=Path("/data/subjects/sub01/surf/lh.w-g.pct.mgh"),  # 顶点百分比图
    device="cpu",  # 体积插值设备
)
```

对应官方 `pctsurfcon --s sub01 --lh-only`；其内部依次调用 `mri_vol2surf --projdist -1`、`mri_vol2surf --projfrac 0.3` 和 `mri_concat --paired-diff-norm --mul 100`。这些是官方命令关系，FNIT 不执行它们。

## 脑区 SNR：`write_surface_snr_stats(...)`

`subject` 须已有 `label/H.aparc.annot`、`surf/H.w-g.pct.mgh` 和 `surf/H.area`；`hemi` 为半球；`output` 写 UTF-8 文本 `stats/H.w-g.pct.stats`。函数返回输出 `Path`，其中每个非空脑区一行，包括顶点数、面积、均值、标准差、范围和 SNR。

```python
from pathlib import Path
from fnit.recon_all.segstats_surface_snr_python import write_surface_snr_stats

stats_path = write_surface_snr_stats(
    subject=Path("/data/subjects/sub01"),  # 已有注释、对比图和面积图的被试目录
    hemi="lh",  # 左半球；右侧使用 "rh"
    output=Path("/data/subjects/sub01/stats/lh.w-g.pct.stats"),  # 输出文本表
)
```

对应官方命令：`mri_segstats --in surf/lh.w-g.pct.mgh --annot sub01 lh aparc --sum stats/lh.w-g.pct.stats --snr`。

## 真实 T1 的同输入结果

在官方冻结的真实 T1 双侧 MRI、表面和标签输入上，2026-09-29 新主页 Conda 环境得到了下列结果：[逐项数值与耗时](../../validation/recon_all/python_gpu_port/surface_metrics_wiring_20260929.json)。

| 输出 | 左侧 | 右侧 | FNIT 单次墙钟 |
| --- | ---: | ---: | --- |
| `jacobian_white`，相关系数 | 0.9999999999999629 | 0.9999999999999643 | 0.146 / 0.071 s |
| `jacobian_white`，最大绝对差 | 1.43×10⁻⁶ | 1.43×10⁻⁶ | 同上 |
| `w-g.pct.mgh`，不同 float32 顶点 | 0 / 106,622 | 0 / 105,541 | 2.135 / 1.097 s |
| `w-g.pct.stats`，不同数据行 | 0 / 35 | 0 / 35 | 0.110 / 0.149 s |

此前同机独立 CLI 比较中的 `mris_jacobian` 官方观察值为左侧 0.53 秒、右侧 0.51 秒；FNIT Python 冷启动为 2.08/1.79 秒。白质/灰质两次 `mri_vol2surf` 官方 CLI 分别观察到左侧 0.587/0.615 秒、右侧 0.813/0.626 秒；FNIT 冷启动为 2.512/1.802 和 2.348/2.056 秒。SNR 官方 CLI 为 0.37/0.35 秒、FNIT 冷启动为 0.20/0.23 秒。这些旧计时和上表暖进程计时范围不同，不拼接成速度比。冻结上游结果也不代表现版整例验收。

## 参考文献与原实现

- Fischl B. FreeSurfer. *NeuroImage*. 2012;62(2):774–781. [doi:10.1016/j.neuroimage.2012.01.021](https://doi.org/10.1016/j.neuroimage.2012.01.021)。
- [FreeSurfer 原实现代码库](https://github.com/freesurfer/freesurfer/tree/d932c45b7941662ea380a05efef580568b98d41a)。
