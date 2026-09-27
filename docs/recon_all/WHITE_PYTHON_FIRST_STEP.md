# 白质面放置：Python 首步探针

此实现仅复现 `mris_place_surface --white` 第一轮的第一个优化步，供逐顶点定位差异。输出文件是首步诊断曲面，不是 `lh.white.preaparc`，不能接入正式 `recon-all`。生产流程仍使用已验收的 Conda 原生白质面程序。

## 输入与调用

`first_white_preaparc_step` 读取同一被试目录下五个文件：`surf/<hemi>.orig`（输入网格）、`surf/autodet.gw.stats.<hemi>.dat`（灰白质阈值）、`mri/brain.finalsurfs.mgz`（强度）、`mri/wm.mgz`（白质掩膜）、`mri/aseg.presurf.mgz`（预表面分割）。顶点按原始曲面顺序处理，不重新编号。

```python
from pathlib import Path
from fnit.recon_all.place_white_preaparc_python import first_white_preaparc_step

result = first_white_preaparc_step(
    subject_dir=Path("/data/subjects/sub-01"),  # 输入被试目录，含 surf/ 与 mri/
    hemi="lh",  # 半球：lh 或 rh
    output=Path("/data/check/lh.white.step1"),  # 输出 FreeSurfer 首步诊断曲面
    diagnostics=Path("/data/check/lh_step1.npz"),  # 可选：各力项与 SSE 的 NumPy 快照
)
```

返回字典包含 `output`、`hemisphere`、`steps=1`、`vertices`、`faces`、`ripped_vertices`、`held_vertices`、`initial_sse`、`initial_rms`、`step_sse`、`step_rms`、`seconds` 和 `stage_seconds`。`stage_seconds` 按 `prepare`、`initial_objective`、`gradient`、`collision`、`step_objective`、`write` 分段，单位均为秒。`diagnostics` 非空时写入 `.npz`：`initial`、`after_collision` 是 `(V,3)` float32 顶点坐标；`ripped` 是 `(V,)` bool；`target_values` 是 `(V,)` float32；各力项键 `intensity`、`averaged`、`self_repulsion`、`pre_normal_spring`、`normal_spring`、`curvature`、`tangential_spring` 是 `(V,3)` float32 梯度或累积梯度；四个 SSE/RMS 标量为 float64。

单被试命令行：

```bash
python -m fnit.recon_all.place_white_preaparc_python \
  /data/subjects/sub-01 lh /data/check/lh.white.step1 \
  --diagnostics /data/check/lh_step1.npz
```

前三个位置参数依次是被试目录、半球和诊断曲面路径；`--diagnostics` 指定可选的中间数组文件。命令行向标准输出打印同样的结果字典。

等价的官方完整命令如下；为限制为一个优化步，实测时将固定版 FreeSurfer 源码中的 `parms.niterations` 改为 1，并把每个力项和碰撞后坐标写到隔离目录。原始输入、优化公式与原生依赖库不变，未把诊断程序加入发布包。

```bash
mris_place_surface \
  --adgws-in ../surf/autodet.gw.stats.lh.dat \
  --wm wm.mgz --threads 4 --invol brain.finalsurfs.mgz --lh \
  --i ../surf/lh.orig --o ../surf/lh.white.preaparc --white \
  --seg aseg.presurf.mgz --restore-255 --nsmooth 5 \
  --rip-bg-no-annot --rip-bg --rip-bg-lof --restore-255 \
  --outvol mrisps.wpa.mgz
```

`--adgws-in` 指定自动阈值；`--wm`、`--invol`、`--seg` 分别指定白质掩膜、强度像和分割像；`--lh` 选择左半球；`--i`/`--o` 是曲面输入/输出；`--white` 选择白质优化；`--threads` 设置 OpenMP 线程；`--restore-255` 还原高强度体素；`--nsmooth` 指定放置前五次网格平滑；`--rip-bg-no-annot`、`--rip-bg`、`--rip-bg-lof` 指定冻结区域；`--outvol` 保存预处理体积。

## 自排斥算子

`mean_vertex_spacing(vertices, ripped, neighbor_offsets, neighbors)` 计算未冻结顶点到其一、二环邻点的平均距离，返回 Python `float`，单位 mm。`vertices` 是 `(V,3)` float32，`ripped` 是 `(V,)` bool，压缩邻接表 `neighbor_offsets` 是 `(V+1,)` int32，`neighbors` 是一维 int32 顶点编号。

`vertex_buckets_current(vertices, ripped, resolution=1.0)` 返回 `(bucket_offsets, bucket_members)`，类型分别是 `(V+1,)` int32 和一维 int32，表示每个顶点所在哈希桶中的候选顶点，并保持原生插入顺序。梯度使用默认 1 mm 格点；SSE 用当前平均顶点距离作 `resolution`，因为固定版源码两处创建 MHT 时的分辨率不同。

`self_repulsion_gradient(vertices, ripped, bucket_offsets, bucket_members, neighbor_offsets, neighbors, weight=5.0)` 返回 `(V,3)` float32 力；`self_repulsion_energy` 的参数相同，返回 float64 加权 SSE。这两个函数跳过已冻结顶点及一、二环邻点。

```python
spacing = mean_vertex_spacing(
    vertices=xyz,  # 当前 (V,3) float32 曲面
    ripped=ripped,  # 当前 (V,) bool 冻结标记
    neighbor_offsets=neighbor_offsets,  # 一、二环 CSR 起点
    neighbors=neighbors,  # 一、二环顶点编号
)
bucket_offsets, bucket_members = vertex_buckets_current(
    vertices=xyz,  # 当前曲面顶点
    ripped=ripped,  # 冻结标记
    resolution=spacing,  # SSE 使用当前平均邻点距离，单位 mm
)
energy = self_repulsion_energy(
    vertices=xyz,  # 当前曲面顶点
    ripped=ripped,  # 冻结标记
    bucket_offsets=bucket_offsets,  # 每个顶点的候选桶起点
    bucket_members=bucket_members,  # 候选顶点编号
    neighbor_offsets=neighbor_offsets,  # 一、二环 CSR 起点
    neighbors=neighbors,  # 一、二环顶点编号
    weight=5.0,  # FreeSurfer 白质自排斥权重
)
force_offsets, force_members = vertex_buckets_current(
    vertices=xyz,  # 当前曲面顶点
    ripped=ripped,  # 冻结标记
    resolution=1.0,  # 梯度使用原生默认 1 mm 格点
)
force = self_repulsion_gradient(
    vertices=xyz,  # 当前曲面顶点
    ripped=ripped,  # 冻结标记
    bucket_offsets=force_offsets,  # 每个顶点的候选桶起点
    bucket_members=force_members,  # 候选顶点编号
    neighbor_offsets=neighbor_offsets,  # 一、二环 CSR 起点
    neighbors=neighbors,  # 一、二环顶点编号
    weight=5.0,  # FreeSurfer 白质自排斥权重
)
```

## 真实数据核对

输入是同一 T1 的 FreeSurfer 官方重建被试 `a_official` 左半球，固定源码提交 `d932c45b7941662ea380a05efef580568b98d41a`。冻结输入的 SHA256 已记录在 `validation/recon_all/python_gpu_port/white_python_first_step_20260927.json`。报告只比较第一轮第一步，不能推出完整 `white.preaparc`、`sphere` 或厚度等终点一致。

输入文件 SHA256 见 JSON；其中 `lh.orig` 为 `ef1bbc58ad65e2e50a71bd7727112ff4614eb840009cd1060384c598a5cd1126`，`brain.finalsurfs.mgz` 为 `c239ba0c807bd381662d388e198715da44103f0c724b3877a7c9623d4ccff9e7`。网格有 106,622 个顶点、213,240 个面、5,611 个冻结顶点。

| 首步检查 | 固定源码原生 | Python 修正前 | Python 修正后 |
| --- | ---: | ---: | ---: |
| 初始 SSE | 4,367,904.930716817 | 4,367,087.002610897 | 4,367,904.930706604 |
| 首步 SSE | 2,527,918.821121097 | 2,524,570.074212057 | 2,527,918.819044066 |
| 初始 SSE 绝对差 | — | 817.928106 | 0.0000102 |
| 首步 SSE 绝对差 | — | 3,348.746909 | 0.0020770 |
| 墙钟时间 | 48.87 s | 115.48 s | 98.55 s |

差异源于固定源码分别用两种顶点哈希表：自排斥梯度用默认 1 mm 分辨率，SSE 则用当前未冻结顶点到其一、二环邻点的平均距离。第一次 Python 实现将两者都设为 1 mm。修正后，初始 SSE 相对差为 `2.34e-12`，首步 SSE 相对差为 `8.22e-10`；本次单步核对采用 SSE 相对容差 `1e-8`，两项通过。

修正后，初始坐标、冻结标记、强度梯度、符号平均后的梯度、自排斥后的梯度、法向弹簧后梯度、曲率后梯度和切向弹簧后梯度都逐位一致，梯度阶段每项均为 319,866/319,866 个 float32 元素。首次诊断曾把最终梯度数组传入碰撞代码，后者按设计原位改写梯度；已改为先复制快照。首个实际坐标差异出现在碰撞更新后：106,622 个顶点中 1,083 个有浮点末位差，最大欧氏距离 `7.62939453125e-6 mm`，超过 `0.0001 mm` 的顶点为 0。首步 RMS 原生为 `10.931342242095482`，Python 为 `10.931342237271664`。此处按 `0.0001 mm` 顶点容差验收单步；逐位一致尚未达成。

Python 修正后内部运行 94.23 s，其中准备与目标寻找 67.78 s、初始 SSE 2.18 s、梯度 1.92 s、碰撞 18.85 s、首步 SSE 2.74 s、写出 0.75 s。原生 48.87 s 是探针整条命令的墙钟时间，含额外 RAM 快照、输出体积和随后交点清理；Python 98.55 s 含首次运行部分 Numba 缓存读取及 Python 启动，未实现整轮交点清理。两者的计时范围不同，不能据此宣称同功能加速。当前 Python 路径运行在 CPU，仍需验证第 2–5 步及完整四轮后的几何、厚度、面积和脑区指标。
