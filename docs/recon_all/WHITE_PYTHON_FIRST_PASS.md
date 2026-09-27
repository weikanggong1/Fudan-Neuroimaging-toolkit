# 白质面放置：Python 前三轮逐步诊断

`place_white_preaparc_prefix` 运行 `mris_place_surface --white` 第一轮的前 1–17 步。它是逐顶点核对用的独立实现，公开函数只覆盖真实 T1 左半球首轮；隔离验证脚本还核对了第二、第三轮全部接受步及第三轮入口。输出是诊断曲面，不能当成完整 `white.preaparc` 接入生产重建。实现只用 nibabel、NumPy 和 Numba，运行在 CPU。已安装 FreeSurfer 仅用于隔离的对照试验，发布代码不会调用它。

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

该**生产** Conda 二进制的首轮曲面与固定源码探针 319,866/319,866 个坐标分量逐位相同。又分别截取第二轮、第四轮优化刚结束的官方与 Conda RAM 曲面，按原顶点编号比较：

| 白质阶段 | Conda/官方最大顶点差 | 超过 0.0001 mm | 超过 0.1 mm |
| --- | ---: | ---: | ---: |
| 第一轮末 | 0.036134 mm | 341 | 0 |
| 第二轮末 | 0.222230 mm | 2,224 | 2 |
| 第四轮末、交点清理前 | 0.630004 mm | 8,892 | 50 |
| 保存的 `white.preaparc` | 0.630004 mm | 8,892 | 50 |

第四轮末与保存结果在官方、Conda 各自内部都 319,866/319,866 个坐标分量逐位相同，`MRISremoveIntersections` 在这组数据上没有移动顶点。最终 50 个超过 `0.1 mm` 的顶点中，只有 1 个在第一轮末已超过 `0.0001 mm`，第二轮末则有 37 个；误差在后续优化中出现并放大，不能将最终 50 个尾差直接归为首轮的 341 个小误差。Python 首轮相对官方仍为 0 个顶点超过 `0.0001 mm`、最大 `0.00001641 mm`；完整的后三轮尚未实现，不能据此推断最终曲面。全候选三次平滑后的 `smoothwm` 相对官方最大差 `0.188170 mm`，18 个顶点超过 `0.1 mm`。

## 首轮状态替换的因果试验

首轮放置的交接状态还包括顶点法线。对安装版第一轮末的进程 RAM 读取法线，先用同版 Conda 头文件核对 `MRIS` 与 `VERTEX` 偏移，再验证 RAM 的 XYZ 与写出的曲面逐位相同。101,011 个未冻结顶点上，Python 根据首轮末曲面重算法线相对官方 RAM 最大差 `7.36e-5`、没有顶点超过 `0.0001`；当前 Conda 首轮法线最大差 `0.01928`、486 个顶点超过 `0.0001`。冻结顶点未参与以下状态替换。

[隔离诊断用的 C++ helper](../../validation/recon_all/python_gpu_port/white_state_handoff_probe.cpp) 通过 `fnit_replace_white_state(surface, path)` 在第一轮返回点写入状态。`surface` 是当前进程的 `MRIS*`，`path` 是连续 `(106622,6)` little-endian float32 文件，列顺序为 `x,y,z,nx,ny,nz`；它只更新未冻结顶点，返回 `0` 表示成功，`1` 表示文件或顶点数不符，`2` 表示短读，`3` 表示文件仍有剩余数据。只在隔离 GDB 对照中加载，未编入 FNIT，也不从生产代码调用。

三次试验使用同一候选输入、同一个生产 Conda 二进制，在第一轮 `MRISpositionSurface` 返回后注入一组状态，再让原程序运行后三轮和清理。零改动对照写回 Conda 自己的状态，最终与保存候选逐坐标相同；另两组分别使用 Python 首轮状态、安装版官方首轮 RAM 状态。所有注入均返回 `0`，四轮均完成。机器报告记录 helper/二进制/状态文件 SHA256、断点及逐顶点结果。

| 第一轮交接状态，其后三轮均为 Conda C++ | 最终对官方平均差 | 99 百分位 | 最大差 | >0.1 mm |
| --- | ---: | ---: | ---: | ---: |
| 原 Conda 状态；零改动对照逐位复现 | 0.000420574 mm | 0.006998961 mm | 0.630004 mm | 50 |
| Python 首轮 XYZ + 重算法线 | **0.000027106 mm** | **0.000054070 mm** | **0.165034 mm** | **2** |
| 安装版官方首轮 XYZ + RAM 法线 | 0.000140201 mm | 0.000019933 mm | 0.532148 mm | 43 |

换成 Python 首轮状态后，原先 50 个超过 `0.1 mm` 的顶点全都改善并降至阈值以下；新的两个超阈值顶点为 `58217` 和 `58940`。这证实首轮状态对现有尾差有实质影响，但该替换仍未达到完整白质面的逐顶点验收。把**精确**官方 XYZ/法线写入同一 Conda 后续程序反而得到 43 个尾差，表明仅交接这六列不足以构成等价的完整首轮状态，也可能是后续碰撞路径对微小差异敏感；现有试验不能区分这两种解释。三个 GDB 单命令用时分别为 `263.17`、`265.65` 和 `291.00 s`，存在诊断及共享节点负载，不能作为生产速度对比。

运行于 gpucw1（Intel Xeon Gold 6430），Python 17 步墙钟 `672.98 s`，内部 `668.48 s`：准备 `71.63 s`，初始目标 `3.14 s`，梯度 `63.35 s`，碰撞 `473.83 s`，逐步目标 `51.24 s`，写出 `5.29 s`。安装版 GDB 截取首轮墙钟 `75.27 s`；原始官方日志的首轮放置为 `1.3 min`；固定源码逐步写 RAM 探针为 `122.57 s`。三者的诊断输出和计时边界不同，现阶段只能判断 Python 首轮明显较慢，耗时集中于动态碰撞。完整四轮、右半球、最终拓扑清理、`sphere.reg`、厚度、面积和脑区统计仍未验收，不能将首轮精度外推到最终重建。

## 第二轮入口与完整优化步

这一诊断从安装版 FreeSurfer 8.2 第一轮末的 RAM 曲面出发，冻结同一组真实 T1 输入。它不调用完整重建，也不把官方程序接入 FNIT 生产代码。官方二进制、五个输入文件的 SHA256 和断点命令见[第二轮机器报告](../../validation/recon_all/python_gpu_port/white_python_passes_2_3_20260928.json)。原始 `VERTEX` 内存各 49,472,608 字节，保存在 gpucw1 诊断目录；报告记录哈希，不提交含进程指针的 RAM 原件。

官方第二轮开始时，先从 5,611 个增加到 5,728 个冻结顶点，再以 `sigma=1` 搜索边界并平均目标强度五次。Python 用第一轮末官方 RAM 中的坐标、法线、上一轮目标值和冻结标记作为输入。新的冻结标记逐点相同；106,622 个目标强度和目标距离、319,866 个目标坐标分量均逐位相同。100,894 个未冻结顶点的 `sigma` 逐位相同。辅助字段 `mean` 在其中 129 个顶点不同；本次第 18 步的坐标和 SSE 核对未受到可测影响，后续轮次尚未验证。

第一轮末坐标与第二轮入口坐标、法线逐位相同。第 18 步用 `n_averages=2`、`dt=0.5`；Python 法线在未冻结顶点上的 302,682 个分量与官方逐位相同。放置后 319,866 个坐标分量中 319,865 个逐位相同，顶点 62122 的一个分量差 `9.54e-7 mm`，没有顶点超过 `0.0001 mm`。

第二轮一开始，`MRISpositionSurface` 会调用 `MRIScomputeMetricProperties` 和 `MRISstoreMetricProperties`，把**当轮当前曲面**面积设为弹簧 SSE 的基准。原诊断代码错误地沿用首轮前的 `57,089.18 mm²`，于是第二轮初始切向弹簧项被乘以 `0.720327`，造成总 SSE 少约 47,586。改用当轮入口面积 `79,254.53 mm²` 后，官方通过 `FREESURFER_logSSE=1` 输出的三项和 Python 如下：

| 阶段 | 官方 SSE | Python SSE | 绝对差 | 官方/ Python 坐标 |
| --- | ---: | ---: | ---: | --- |
| 第二轮入口 | 669,703.502519 | 669,703.502519429 | `4.29e-7` | 入口坐标逐位相同 |
| 第 18 步 | 482,625.560794 | 482,625.560786897 | `7.10e-6` | 319,865/319,866 分量逐位相同 |

初始自排斥、切向弹簧、强度三项的绝对差分别为 `2.45e-7`、`4.04e-7`、`7.79e-8`；第 18 步分别为 `6.93e-6`、`8.93e-7`、`6.58e-8`。启用与不启用 SSE 逐项日志的两次官方运行，第 18 步坐标、法线、目标值和距离等已解析字段全部逐位相同；两份原始 RAM 文件的 SHA 不同是其中还包含进程地址，不能当成数值差。

复核脚本位于 `validation/recon_all/python_gpu_port/`。`capture_white_installed_second_pass_boundary.py` 的 `--subject`、`--binary`、`--license`、`--out` 分别是被冻结的官方被试、安装版参考二进制、私有许可证文件和隔离输出目录；默认在第二轮优化前截取 RAM，`--after-first-step` 则把第二轮迭代上限暂设为 1，截取第 18 步后状态。它只用于对照，不属于安装后的 FNIT 调用链。

`validate_white_second_pass_boundary.py` 以 `--subject`、`--first-raw`（官方首轮末 RAM）、`--second-raw`（官方第二轮入口 RAM）和 `--out`（JSON 结果）调用；它在同目录写出 `.npz`，含 `(V,)` 冻结标记、目标值、距离、梯度幅度、标记、sigma，以及 `(V,3)` 目标坐标。`validate_white_second_step.py` 接收 `--subject`、`--before-raw`（第二轮入口 RAM）、`--after-raw`（第 18 步 RAM）、`--boundary-npz`（上述 Python 目标）、`--official-log`（逐项 SSE 日志）和 `--out`（JSON 结果）；同目录 `.npz` 含初始坐标/法线、各力项、候选和碰撞后坐标。两者的具体命令参数和值保存在机器报告和脚本源中。

在 gpucw1，Python 第二轮入口目标准备用时 `25.32 s`，第 18 步单步诊断用时 `58.19 s`。后续从同一入口重新运行完整第二轮，用时 `340.82 s`：准备 `12.40`、初始 SSE `1.78`、梯度 `26.05`、碰撞 `275.85`、每步 SSE `24.21 s`。单步和完整重放是独立进程，Numba 缓存与准备范围不同，不可相加。官方 GDB 从 `lh.orig` 重放首轮并逐步保存第二轮的九份 RAM，用时 `122.89 s`；Python 从官方第二轮入口 RAM 起跑，官方还包含首轮与诊断写盘，不能直接计算加速比。Python 耗时集中于碰撞，当前没有提速证据。

### 第二轮第 18–26 步

官方每个接受步的 RAM 坐标和标准输出均已截取。第二轮第 19 步先拒绝一次 `dt=0.5`，然后以 `dt=0.25` 接受；第 23 步后降至 `dt=0.125`，第 26 步后停止。Python 的接受、拒绝和停止位置一致。官方对第 19–26 步只打印一位小数 SSE，因此该列只能验到打印精度；第 18 步另有前表所示的六位小数逐项日志。

| 步 | 官方 SSE（打印值） | Python SSE | 逐位相同坐标分量 / 319,866 | 最大顶点差（mm） | >0.0001 mm 顶点 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 18 | 482625.6 | 482625.560787 | 319,865 | 0.000000954 | 0 |
| 19 | 338996.1 | 338996.050211 | 319,865 | 0.000000954 | 0 |
| 20 | 305732.2 | 305732.218245 | 319,865 | 0.000000954 | 0 |
| 21 | 252125.7 | 252125.727818 | 319,862 | 0.000000954 | 0 |
| 22 | 244985.9 | 244985.938615 | 319,850 | 0.000000954 | 0 |
| 23 | 258980.8 | 258980.785788 | 319,828 | 0.000004672 | 0 |
| 24 | 225960.5 | 225960.516823 | 319,792 | 0.000007629 | 0 |
| 25 | 220529.9 | 220529.933046 | 319,751 | 0.000007650 | 0 |
| 26 | 220338.9 | 220338.897238 | 319,706 | 0.000007844 | 0 |

首次坐标非逐位一致仍在第 18 步顶点 62122，只有一个 float32 分量差 `9.54e-7 mm`；首次高精度 SSE 差也是第 18 步的 `7.10e-6`。后八步 SSE 都落在官方一位小数打印值的 ±`0.05` 内，不能据此声称后八步 SSE 逐位相同。第二轮末 106,622 个顶点均未超过 `0.0001 mm`。

### 第三轮入口

再用同组输入在官方进程第三次调用放置优化器之前截取 RAM。官方第二轮末至第三轮入口的 `xyz`、`orig`、法线共三组 319,866 个坐标分量逐位未变。Python 从**官方第 26 步 RAM** 出发，重新冻结顶点，以 `sigma=0.5` 搜索边界，并对目标强度做五次邻域平均。该输入选择单独检验第三轮入口算子；若从 Python 第 26 步坐标连续接入，仍有上表的微小浮点差。

| 字段 | Python 与官方 |
| --- | --- |
| 冻结顶点 | 5,744/5,744，逐点相同 |
| 标记顶点 | 逐点相同 |
| 目标强度、目标距离 | 各 106,622/106,622 逐位相同 |
| 目标坐标 | 319,866/319,866 分量逐位相同 |
| 活跃顶点的 sigma | 100,878/100,878 逐位相同 |
| 活跃顶点的 `mean` | 100,662/100,878 逐位相同，216 个不等，最大差 14.116126 |
| 已冻结顶点中残留的 sigma | 133 处不等；不计入上面的活跃 sigma 验收 |

Python 第三轮入口目标准备用时 `23.79 s`（体积准备 `4.15`、冻结 `8.07`、边界搜索 `5.08`、平均 `6.49 s`）。官方 GDB 从原始输入重放前两轮再截取入口，用时 `135.11 s`；起点不同。`mean` 字段未逐位匹配，以下八步检验它是否进入当前优化轨迹。

### 第三轮第 27–34 步

Python 从上述官方第三轮入口 RAM 坐标出发，使用自己计算的目标强度与活跃 sigma。这两项已逐点匹配；`mean` 没有输入 Python 梯度或目标函数。官方每个接受步另存 RAM，并用 `FREESURFER_logSSE=1` 独立重放，取得六位小数的自排斥、弹簧和强度 SSE。第 28 步先拒绝 `dt=0.5` 再接受 `dt=0.25`；第 31、34 步后缩步。Python 的八个接受步、缩步和结束位置均与官方相同。

| 步 | 官方 SSE | Python SSE | 逐位相同坐标分量 / 319,866 | 最大顶点差（mm） |
| ---: | ---: | ---: | ---: | ---: |
| 27 | 417800.562461 | 417800.562461 | 319,865 | 0.000000954 |
| 28 | 285414.094242 | 285414.094243 | 319,863 | 0.000000954 |
| 29 | 247318.912133 | 247318.912039 | 319,860 | 0.000000954 |
| 30 | 227257.904359 | 227257.904306 | 319,851 | 0.000001349 |
| 31 | 224447.697160 | 224447.701073 | 319,830 | 0.000007632 |
| 32 | 210614.310101 | 210614.314082 | 319,783 | 0.000007720 |
| 33 | 210453.108581 | 210453.149022 | 319,746 | 0.000007807 |
| 34 | 212909.045554 | 212909.110241 | 319,690 | 0.000007910 |

八步都没有顶点超过 `0.0001 mm`。SSE 从第 27 步就有 `3.52e-7` 的数值差；到第 34 步差 `0.064687`，首次跨过官方一位小数的打印边界。单独把**官方第 34 步坐标**输入同一 Python 目标函数，得到 `212909.045554475`，与原生六位小数值只差 `4.75e-7`。换成 Python 自己的坐标得到 `212909.110241`；两套坐标间的 SSE 差 `0.064686` 中，自排斥项占 `0.063706`，强度项占 `0.000987`，弹簧项约 `-0.000007`。第 33 步在官方坐标上重算的 SSE 也与原生值一致至六位小数。因此第 34 步打印差来自微小坐标差在自排斥项的累积，而非同坐标目标函数的可测偏差。

固定 FreeSurfer 源码提交 `d932c45b7941662ea380a05efef580568b98d41a` 中，本命令的 `l_grad` 默认为零；强度梯度实现的 `v->mean` 读数被注释，`mean` 对应的 SSE 项只在 `l_grad` 非零时启用。安装版实测也只记录自排斥、弹簧和强度三项。这解释了入口 216 个活跃 `mean` 差异没有造成当前第三轮的同坐标 SSE 差；不代表其它参数路径可忽略该字段。

Python 第三轮独立重放 `362.54 s`：准备 `16.11`、初始 SSE `3.06`、梯度 `25.93`、碰撞 `291.18`、逐步 SSE `25.78 s`。官方 GDB 从 `lh.orig` 重放前三轮并存八份 RAM 为 `172.51 s`，另一次重放前三轮记录精细 SSE 为 `184.70 s`。计时起点与诊断 IO 不同，不计算正式加速比；Python 的碰撞核明显较慢。

### 离完整 white.preaparc 还差什么

本轮在第三轮入口重新使用**官方**第二轮末坐标。接入生产前，要将 Python 首轮、第二轮、第三轮连续传递，检验微小坐标差会不会在后续轮次放大。官方还有第四轮（`sigma=0.25`、`n_averages=0`）；需核对该轮重新冻结/目标搜索、每个接受及拒绝步、SSE 和坐标。第四轮后还要核对 `MRISremoveIntersections`、写出的 `white.preaparc` 顶点/面/元数据，以及右半球。当前公开函数仍只覆盖首轮，未形成可接入生产的完整白质面替换。

### 复核脚本的输入和输出

`capture_white_installed_second_pass_steps.py` 和 `capture_white_installed_third_pass_boundary.py` 的命令行参数均为 `--subject`（含五个冻结输入的被试目录）、`--binary`（仅隔离验证用的官方 `mris_place_surface`）、`--license`（私有许可证路径）和 `--out`（诊断目录）。前者写 `capture.log`、`capture_report.json` 与 `step018.raw` 至 `step026.raw`；后者写同名日志/清单和第三轮入口 `vertices.raw`。原生 `raw` 记录每顶点 464 字节，含进程指针，只保留在私有诊断目录，不入库。

`validate_white_second_pass_steps.py` 使用 `--subject`（输入被试）、`--boundary-raw`（官方第二轮入口 RAM）、`--boundary-npz`（Python 第二轮目标值与 sigma）、`--official-steps`（九个官方逐步 RAM 文件的目录）、`--out`（JSON 路径）。它在 `--out` 同位置写 `.npz`：`initial`、`step18` 至 `step26` 是 `(V,3)` float32 坐标，`ripped` 是 `(V,)` bool，`target_values` 是 `(V,)` float32。JSON 逐步列出接受/拒绝、SSE/RMS、坐标误差和分段耗时。

`validate_white_third_pass_boundary.py` 使用 `--subject`、`--second-final-raw`（官方第 26 步 RAM）、`--third-boundary-raw`（官方第三轮入口 RAM）、`--python-second-npz`（Python 第二轮逐步快照）和 `--out`（JSON 路径）。同位置的 `.npz` 含 `ripped`、`values`、`distances`、`mean`、`marked`、`sigma`（均为 `(V,)`）及 `target`（`(V,3)`）；JSON 给出逐字段相同数、最大差、首次差异位置和分段耗时。实际路径、脚本 SHA256、五个输入的 SHA256 和完整命令均在[机器报告](../../validation/recon_all/python_gpu_port/white_python_passes_2_3_20260928.json)。

`capture_white_installed_third_pass_steps.py` 和 `capture_white_installed_third_pass_sse.py` 的 `--subject`、`--binary`、`--license`、`--out` 与前述官方截取脚本相同。前者写 `step027.raw` 至 `step034.raw`、日志和清单；后者启用高精度 SSE 日志，写 `capture.log` 和含每步三项 SSE 的 `capture_report.json`，不写 RAM。两者都在第三轮结束后退出，不运行第四轮。

`validate_white_third_pass_steps.py` 的 `--subject` 是冻结被试；`--boundary-raw` 是官方第三轮入口 RAM，`--boundary-npz` 是 Python 第三轮目标，`--official-steps` 指向官方八个逐步 RAM，`--out` 指定 JSON。相邻 `.npz` 中 `initial`、`step27` 至 `step34` 为 `(V,3)` float32 坐标，`ripped`、`target_values` 为 `(V,)`。JSON 按步列出试探次数、接受 dt、SSE/RMS、坐标误差及分段时间。`analyze_white_third_sse_terms.py` 另接收 `--python-steps-npz`（上述逐步数组），以及相同的 `--subject`、`--boundary-raw`、`--boundary-npz`、`--official-steps` 和 `--out`（JSON）；输出第 33、34 步分别在官方与 Python 坐标上的三项 SSE。脚本和数据 SHA256、完整命令在[机器报告](../../validation/recon_all/python_gpu_port/white_python_passes_2_3_20260928.json)。

当前公开的 `place_white_preaparc_prefix` 仍只运行首轮，不输出完整 `white.preaparc`。这些隔离探针只提供同输入的阶段证据，未用于 FNIT 生产调用链。
