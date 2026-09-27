# 白质面放置：Python 首轮 17 步

`place_white_preaparc_prefix` 运行 `mris_place_surface --white` 第一轮的前 1–17 步。它是逐顶点核对用的独立实现，当前只验收了真实 T1 的左半球首轮；输出是指定步数后的诊断曲面，不能当成完整 `white.preaparc` 接入生产重建。实现只用 nibabel、NumPy 和 Numba，运行在 CPU。已安装 FreeSurfer 仅用于隔离的对照试验，发布代码不会调用它。

## 输入、输出与调用

`subject_dir` 是被试目录，须包含 `surf/<hemi>.orig`（原始顶点和三角面）、`surf/autodet.gw.stats.<hemi>.dat`（灰白质强度阈值）、`mri/brain.finalsurfs.mgz`（强度像）、`mri/wm.mgz`（白质掩膜）和 `mri/aseg.presurf.mgz`（预表面分割）。`hemi` 取 `lh` 或 `rh`；`output` 指定 FreeSurfer 曲面文件，保留原始顶点编号和三角面；`steps` 是第一轮步数，取 1–17；`diagnostics` 可省略，设置后写出 `.npz` 中间数组。

```python
from pathlib import Path
from fnit.recon_all.place_white_preaparc_python import place_white_preaparc_prefix

result = place_white_preaparc_prefix(
    subject_dir=Path("/data/subjects/sub-01"),  # 输入被试目录，含上述 surf/ 和 mri/ 文件
    hemi="lh",  # 输入半球：lh 或 rh
    output=Path("/data/check/lh.white.step17"),  # 输出第 17 步的诊断曲面
    steps=17,  # 第一轮执行步数：1–17
    diagnostics=Path("/data/check/lh_step17.npz"),  # 可选的逐步坐标、梯度和目标值快照
)
```

只核对第一步时，可调用 `first_white_preaparc_step`；它的输入文件与输出结构相同，固定执行一步。

```python
from pathlib import Path
from fnit.recon_all.place_white_preaparc_python import first_white_preaparc_step

result = first_white_preaparc_step(
    subject_dir=Path("/data/subjects/sub-01"),  # 输入被试目录
    hemi="lh",  # 输入半球：lh 或 rh
    output=Path("/data/check/lh.white.step1"),  # 输出第一步的诊断曲面
    diagnostics=Path("/data/check/lh_step1.npz"),  # 可选的第一步力项与坐标快照
)
```

两个函数都返回字典：`output` 是曲面路径，`hemisphere` 是半球，`steps` 是实际接受的步数，`vertices`、`faces`、`ripped_vertices`、`held_vertices` 是网格/冻结/阻挡数量；`initial_sse`、`initial_rms` 和 `step_sse`、`step_rms` 分别是初始与末步目标值；`seconds` 是函数总耗时。`per_step` 列表逐步给出 `step`、`trials`、`sse`、`rms`、`next_dt`、`reductions`、`held_vertices`。`stage_seconds` 以秒列出 `prepare`、`initial_objective`、`gradient`、`collision`、`step_objective`、`write`。若第一轮因第三次缩步提前结束，`steps` 为实际步数。

`.npz` 中 `initial` 和 `stepN_initial`、`stepN_after_collision` 是 `(V,3)` float32 顶点坐标；`stepN_tangential_spring` 是 `(V,3)` float32 最终梯度；`ripped` 是 `(V,)` bool，`target_values` 是 `(V,)` float32；`stepN_sse`、`stepN_rms` 是标量。另有第一步的 `intensity`、`averaged`、`self_repulsion`、`pre_normal_spring`、`normal_spring`、`curvature`、`tangential_spring` 和 `after_collision` 快照，便于定位第一处差异。

命令行运行同一函数：

```bash
python -m fnit.recon_all.place_white_preaparc_python \
  /data/subjects/sub-01 lh /data/check/lh.white.step17 \
  --steps 17 --diagnostics /data/check/lh_step17.npz
```

三个位置参数依次为被试目录、半球和诊断曲面路径；`--steps` 为第一轮步数，默认 1；`--diagnostics` 为可选 `.npz` 路径。标准输出是上述返回字典的 JSON。

## 与官方命令的关系

官方对照命令在被试的 `mri/` 目录运行：

```bash
mris_place_surface \
  --adgws-in ../surf/autodet.gw.stats.lh.dat \
  --wm wm.mgz --threads 4 --invol brain.finalsurfs.mgz --lh \
  --i ../surf/lh.orig --o ../surf/lh.white.preaparc --white \
  --seg aseg.presurf.mgz --restore-255 --nsmooth 5 \
  --rip-bg-no-annot --rip-bg --rip-bg-lof --restore-255 \
  --outvol mrisps.wpa.mgz
```

`--adgws-in` 读灰白质阈值；`--wm`、`--invol`、`--seg` 分别读白质掩膜、强度像和分割像；`--lh` 选左半球；`--i`、`--o` 指定输入/输出曲面；`--white` 选白质放置；`--threads` 控制 OpenMP 线程；`--nsmooth` 指定前置五次网格平滑；`--rip-bg-no-annot`、`--rip-bg`、`--rip-bg-lof` 控制冻结区域；`--restore-255` 还原高强度体素；`--outvol` 写预处理体积。官方命令继续后续三轮与清理，Python 目前只覆盖第一轮前缀。

## 自排斥算子

`mean_vertex_spacing` 接收当前 `(V,3)` float32 顶点、`(V,)` bool 冻结标记，以及一、二环压缩邻接表 `neighbor_offsets`（`(V+1,)` int32）和 `neighbors`（一维 int32），返回平均邻点距离，单位 mm。`vertex_buckets_current` 用顶点、冻结标记和哈希格分辨率构建候选表，返回 `(bucket_offsets, bucket_members)` 两个 int32 数组。`self_repulsion_energy` 返回加权标量 SSE；`self_repulsion_gradient` 返回 `(V,3)` float32 力，二者输入都是曲面、冻结标记、候选表、邻接表和权重。梯度使用原生默认 1 mm 格点，SSE 使用当前平均邻点距离。

```python
from fnit.recon_all.place_surface_self_repulsion import (
    mean_vertex_spacing, vertex_buckets_current,
    self_repulsion_energy, self_repulsion_gradient,
)

spacing = mean_vertex_spacing(
    vertices=xyz,  # 输入当前 (V,3) float32 顶点
    ripped=ripped,  # 输入 (V,) bool 冻结标记
    neighbor_offsets=neighbor_offsets,  # 输入一、二环 CSR 起点
    neighbors=neighbors,  # 输入一、二环顶点编号
)
energy_offsets, energy_members = vertex_buckets_current(
    vertices=xyz,  # 输入当前顶点
    ripped=ripped,  # 输入冻结标记
    resolution=spacing,  # 输入 SSE 哈希格分辨率，单位 mm
)
energy = self_repulsion_energy(
    vertices=xyz,  # 输入当前顶点
    ripped=ripped,  # 输入冻结标记
    bucket_offsets=energy_offsets,  # 输入每个顶点的候选桶起点
    bucket_members=energy_members,  # 输入候选顶点编号
    neighbor_offsets=neighbor_offsets,  # 输入一、二环 CSR 起点
    neighbors=neighbors,  # 输入一、二环顶点编号
    weight=5.0,  # 输入官方白质自排斥权重；输出为标量 SSE
)
force_offsets, force_members = vertex_buckets_current(
    vertices=xyz,  # 输入当前顶点
    ripped=ripped,  # 输入冻结标记
    resolution=1.0,  # 输入梯度哈希格分辨率，单位 mm
)
force = self_repulsion_gradient(
    vertices=xyz,  # 输入当前顶点
    ripped=ripped,  # 输入冻结标记
    bucket_offsets=force_offsets,  # 输入每个顶点的候选桶起点
    bucket_members=force_members,  # 输入候选顶点编号
    neighbor_offsets=neighbor_offsets,  # 输入一、二环 CSR 起点
    neighbors=neighbors,  # 输入一、二环顶点编号
    weight=5.0,  # 输入官方白质自排斥权重；输出为 (V,3) float32 力
)
```

## 真实 T1 验证

输入是 `a_official` 左半球的同一组冻结文件，106,622 顶点、213,240 个面、5,611 个冻结顶点。`lh.orig` 的 SHA256 为 `ef1bbc58ad65e2e50a71bd7727112ff4614eb840009cd1060384c598a5cd1126`，`brain.finalsurfs.mgz` 为 `c239ba0c807bd381662d388e198715da44103f0c724b3877a7c9623d4ccff9e7`；五个输入的完整哈希和逐步数组结果在 [机器报告](../../validation/recon_all/python_gpu_port/white_python_first_pass_20260927.json)。固定源码提交 `d932c45b7941662ea380a05efef580568b98d41a`。安装版二进制 SHA256 为 `67139c3590f961f879a4214cd01f173b42986c7b178d1851dfa3f08e8acf12aa`。

第一步的输入坐标、冻结标记和强度、自排斥、弹簧、曲率等力项，与固定源码探针逐位一致；首个差异发生在碰撞更新，最大顶点距离 `7.63e-6 mm`。固定源码探针与 Python 的若干后续步骤会渐渐分开，下表仅摘录关键步；机器报告保存全部 17 步。

| 步 | Python/固定源码 SSE 绝对差 | 碰撞后最大顶点差 | 超过 0.0001 mm 顶点 |
| ---: | ---: | ---: | ---: |
| 1 | 0.002077 | 0.00000763 mm | 0 |
| 5 | 0.008814 | 0.00001789 mm | 0 |
| 8 | 0.052028 | 0.00009289 mm | 0 |
| 9 | 0.062029 | 0.00011309 mm | 1 |
| 13 | 0.287193 | 0.09435462 mm | 37 |
| 15 | 0.070376 | 0.11154280 mm | 89 |
| 17 | 0.806826 | 0.03613443 mm | 341 |

第 17 步还用安装版 FreeSurfer 8.2 的进程 RAM 曲面单独核对：GDB 将本轮步数设为 17，在第一轮后跳过后续三轮并写出当前曲面。捕获标记、退出状态和二进制哈希均在机器报告中。Python 与这个官方曲面的逐顶点最大欧氏差为 **`1.640763e-5 mm`**，99 百分位为 `5.96e-7 mm`，没有顶点超过 `0.0001 mm`；319,866 个坐标分量中 316,573 个逐位相同。固定源码探针与安装版的末步最大差反而是 `0.036134 mm`，341 个顶点超过 `0.0001 mm`。两种原生构建在较后步骤已有差异，原因尚未单独锁定；因此完整输出验收以安装版实际 RAM 曲面为准，固定源码探针用于定位力项和决策。

## 当前 Conda C++ 候选的首轮尾差

另对当前全候选输入重放 `white_candidate_no_official_topology_fslicense` 直接截取生产 Conda 二进制的首轮 RAM 曲面，而非使用 9 月 24 日的旧候选。二进制 SHA256 为 `9a42f5d7b70a066daf12e67fb6a0924048b4778186ea772e8976722b226b35d5`；GDB 在首轮 `MRISpositionSurface` 返回后写曲面，退出码 0，用时 `103.47 s`。候选 `lh.orig` 顶点与面逐位相同，几何元数据除文件名均相同；阈值文件字节相同；`brain.finalsurfs`、`wm`、`aseg.presurf` 的体素及仿射均与官方相同。各文件当前 SHA 和路径见机器报告，不能用压缩文件哈希不同推断体素不同。

该**生产** Conda 二进制的首轮曲面与固定源码探针 319,866/319,866 个坐标分量逐位相同。相对安装版官方首轮，它有 341 个顶点超过 `0.0001 mm`，最大差 `0.036134 mm`；Python 首轮相对官方为 0 个、最大 `0.00001641 mm`。已有全候选链的完整 `white.preaparc` 相对官方最大差 `0.630004 mm`、50 个顶点超过 `0.1 mm`；三次平滑后的 `smoothwm` 最大差 `0.188170 mm`、18 个顶点超过 `0.1 mm`。因此偏差在生产 Conda C++ 的首轮已经出现，后续轮次或交点清理会继续改变尾部误差；尚未逐轮捕获，不能将全部终点偏差仅归因于首轮。

运行于 gpucw1（Intel Xeon Gold 6430），Python 17 步墙钟 `672.98 s`，内部 `668.48 s`：准备 `71.63 s`，初始目标 `3.14 s`，梯度 `63.35 s`，碰撞 `473.83 s`，逐步目标 `51.24 s`，写出 `5.29 s`。安装版 GDB 截取首轮墙钟 `75.27 s`；原始官方日志的首轮放置为 `1.3 min`；固定源码逐步写 RAM 探针为 `122.57 s`。三者的诊断输出和计时边界不同，现阶段只能判断 Python 首轮明显较慢，耗时集中于动态碰撞。完整四轮、右半球、最终拓扑清理、`sphere.reg`、厚度、面积和脑区统计仍未验收，不能将首轮精度外推到最终重建。
