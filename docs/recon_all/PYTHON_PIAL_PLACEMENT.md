# Python pial.T1 表面放置

`fnit.recon_all.place_pial_python.place_pial_t1` 实现 FreeSurfer 8.2 `mris_place_surface --pial` 的四轮几何优化，以及内侧壁固定和表面相交修复。它在 CPU 上使用 NumPy、Numba 和 nibabel，可单独调用并用于同输入算法核对。标准 `fnit-recon-all` 在最终 white 放置后使用[Conda 源码构建的原生 pial](NATIVE_PIAL_PLACEMENT.md)；原因及同输入差异见该页。GPU 采样后端和静态索引优化的历史验证分别记在本文后续章节及表面热点报告；本轮候选仍须绑定自己的完整阶段结果。

## 输入、输出与调用

`subject` 是 FreeSurfer 格式的被试目录，`hemisphere` 为 `lh` 或 `rh`。以下七个文件须由各自的前序阶段生成：

| 相对于 `subject` 的输入 | 用途 |
| --- | --- |
| `surf/{hemi}.white` | 提供有序顶点、面、体积几何和表面标签；pial 优化从其坐标开始。 |
| `surf/autodet.gw.stats.{hemi}.dat` | 被试的灰白质强度阈值。 |
| `label/{hemi}.cortex+hipamyg.label` | 放置与相交修复使用的顶点 rip 掩膜。 |
| `label/{hemi}.cortex.label` | 优化后将内侧壁固定回 white。 |
| `mri/brain.finalsurfs.mgz` | 放置所用强度图与体素几何。 |
| `mri/wm.mgz` | 强度预处理使用的白质类别。 |
| `mri/aseg.presurf.mgz` | 边界搜索所用分割。 |

```python
from fnit.recon_all.place_pial_python import place_pial_t1

report = place_pial_t1(
    subject="/path/to/subjects/sub01",  # 含七个前置文件的被试目录
    hemisphere="lh",  # 左半球；右半球用 rh
    output="/path/to/subjects/sub01/surf/lh.pial.T1",  # 输出表面路径
)
```

`output` 可省略，默认写入 `subject/surf/{hemi}.pial.T1`。函数输出一个 FreeSurfer 三角表面：保留输入 white 的**有序面、完整体积几何标签和辅助尾部**，以放置后的 pial 坐标替换顶点坐标。`report` 含 `output`（路径字符串）、`hemisphere`（半球）、`steps`（累计迭代步数；达到缩步上限的还原步也计一次）、`pass_ends`（四轮的累计步数）、`cleanup`（相交次数、轨迹和平滑次数）及 `seconds`（墙钟耗时）。`max_steps` 默认 200；未收敛时抛出异常，不写入未完成的表面。

函数不生成厚度、面积、曲率、体积或 atlas 统计；这些指标须由后续阶段计算。它也不生成所需的 white、标签或 MRI 输入。标准 runner 已在上游完成最终 white 放置；自产上游连续输入仍需验收，不能据此沿用下述冻结同输入的逐点结果。按官方顺序，还需先完成 `white.preaparc` 放置、皮层和海马杏仁核标签、sphere 配准与 aparc 注释、最终 white 放置。本 pial 函数不直接读取 `white.preaparc` 或 `aparc.annot`，但它们属于上述上游流程。

本模块没有独立 Python CLI；上面的具名实参调用是使用入口。在 `subject/mri` 目录中，使用 FreeSurfer 8.2 和 `FS_LICENSE` 的对应官方命令为：

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

右侧将文件名和半球参数中的 `lh` 改为 `rh`。Python 函数复现该命令中固定的 pial 放置分支；该分支使用显式 rip 标签，因此 Python 函数不需要注释文件。

## 冻结真实 T1 输入的同阶段对照

参照被试是由真实 T1 完成的 FreeSurfer 8.2 `a_official`，其 `mri/orig.mgz` SHA-256 为 `c99c246200cc35479b6b8cd691457985b66f2062d4a37a0c3592ff1b678b985c`。原始被试文件保存在 headcw 的 `work/reconall_benchmark_pair_ac_20260924/official_subjects/a_official`；Git 中没有存放患者影像。候选函数读取该被试保存的 white、brain.finalsurfs、WM、aseg、标签和阈值文件，**不读取官方 pial 几何、优化日志、决策或内存检查点**。输入哈希见[清单](../../validation/recon_all/python_gpu_port/native_cpp_conda_20260927/place_geometry_pair/input_manifest.json)。这是一项冻结官方上游输入的阶段验证，不是当前候选上游的整例验证。

| 同输入阶段 | Python 输出与官方对照 | 观察耗时 |
| --- | --- | ---: |
| LH pial.T1 | 独立接受 41 步；四轮结束于 26/32/36/41；有序顶点 106,622/106,622、float32 坐标 319,866/319,866、有序面 213,240/213,240 完全一致。保留源码写入方式后，体积几何字段与完整尾部字节一致。 | Python 在 headcw 耗时 1,240.8 s；另一次 headcw 官方 C++ pial 耗时 116.05 s。并发负载不同，未据此计算受控速度比。 |
| RH pial.T1 | 独立接受 41 步；四轮结束于 26/31/35/41；有序顶点 105,541/105,541、float32 坐标 316,623/316,623、有序面 211,078/211,078 完全一致。体积几何字段与完整尾部字节一致。 | Python 耗时 1,154.2 s；另一次 headcw 官方 C++ RH 重放耗时 91.77 s，运行负载发生变化。 |

输出文件的 SHA-256 可能因首条表面注释记录生成来源而不同。[双侧几何、尾部与耗时报告](../../validation/recon_all/python_gpu_port/native_cpp_conda_20260927/place_geometry_pair/python_pial/) 保存了逐项对照。以相同的官方 white 和 cortex 标签，将该阶段输出接入现有 Python 顶点指标函数后，相对归档官方图的 p99 绝对误差：厚度不超过 2.38e-7 mm、pial 面积不超过 2.38e-7 mm²、顶点体积不超过 4.77e-7 mm³；两侧相应最大误差分别不超过 4.77e-7 mm、9.54e-7 mm² 和 1.91e-6 mm³。曲率函数仍有小幅算法误差：LH/RH 的 p99 为 2.71e-5/2.15e-5，最大值为 4.54e-4/2.03e-4。双侧共八张图均没有超过既定容差的顶点（面积：0.001 + 0.001 × |reference|；其他图：0.005 + 0.001 × |reference|）。[指标图比较脚本](../../validation/recon_all/python_gpu_port/compare_pial_metric_maps.py)与 [LH/RH JSON 报告](../../validation/recon_all/python_gpu_port/native_cpp_conda_20260927/place_geometry_pair/python_pial/)保留逐图计数。这些冻结输入结果尚不能代表当前默认 runner 的指标或吞吐。此项 CPU pial 优化在有限观测中明显较慢；待上游候选链匹配后，碰撞检测和 KDTree 是主要提速对象。

所需 Python 包为 `nibabel`、`NumPy`、`SciPy`、`PyTorch`、`Numba`，由主页 `environment.yml` 安装。`place_pial_t1` 不调用外部 FreeSurfer 可执行程序；自产上游的 Python 与 Conda C++ 同输入比较见[标准 pial 记录](NATIVE_PIAL_PLACEMENT.md)。

## 参考文献与原实现

- Fischl B. FreeSurfer. *NeuroImage*. 2012;62(2):774–781. [doi:10.1016/j.neuroimage.2012.01.021](https://doi.org/10.1016/j.neuroimage.2012.01.021)。
- [FreeSurfer 固定源码提交](https://github.com/freesurfer/freesurfer/tree/d932c45b7941662ea380a05efef580568b98d41a)。


## 当前串行优化：复用静态索引

实测候选8178bf2。完整四轮函数复用已有FaceNormalTopology，固定面CSR只准备
一次；每个当前坐标版本仍重算法向。原表面法向沿用自己的累加定义，累加前
不归一化各面，不能用current法向替代。原始white坐标和rip标签的桶索引固定，
每次对当前坐标重新生成键；Numba查询替代每轮两个Python逐顶点字典循环。
保留float32加1000再向零取整、桶内原顶点顺序、所有候选及原排斥梯度。
异步接受、拒绝试步状态、步长、目标值、四轮切换及最后相交清理均保留。
生产默认仍为Conda完整pial；该替换必须先通过同输入回归，不能把组件加速
直接计入生产整例提速。

新增内部OriginalVertexBuckets(original, ripped)输入(N,3)float32原始surface
RAS/mm坐标和(N,)bool标签mask，复制并冻结整数键和mask。query(current)输入
同N的有限坐标，返回offsets(N+1)与candidate_ids(M) int32；M为全部候选数，
没有固定上限。原表面或rip改变须新建；shape或有限性不合法抛ValueError。
vertex_buckets(current, original, ripped)保留原入口，单次使用相同完整算法。
original_vertex_normals(vertices, triangles, *, topology=None)接收同序有限坐标
和整数三角面，返回(N,3)float32单位法向；无面顶点为零，不兼容拓扑抛异常。
_query_vertex_buckets输入当前/排序整数键、原顶点ids及固定rip，返回同CSR。
这些都是上游mris_place_surface内部步骤，没有独立官方CLI或GPU/精度选项。

~~~python
import nibabel.freesurfer.io as surface_io
import numpy as np
from fnit.recon_all.place_surface_normals import FaceNormalTopology
from fnit.recon_all.place_surface_repulsion import OriginalVertexBuckets, original_vertex_normals
vertices, faces = surface_io.read_geometry("/data/self/surf/lh.white")  # 有序三角面，surface RAS/mm
ripped = np.zeros(len(vertices), dtype=bool)  # 示例保留全部顶点；生产来自真实rip标签
normal_topology = FaceNormalTopology(triangles=faces, nvertices=len(vertices))  # 仅缓存整数拓扑
original_normals = original_vertex_normals(vertices=vertices, triangles=faces, topology=normal_topology)  # 原表面法向定义
candidate_index = OriginalVertexBuckets(original=vertices, ripped=ripped)  # 本次white与固定rip
candidate_offsets, candidate_ids = candidate_index.query(current=vertices)  # 下一轮传其真实当前坐标
~~~

真实完整回归已执行sub01左侧41步，报告绑定源码/输入SHA，固定4线程、200步上限。
未更改默认white：现有Python white只覆盖前缀，无法代替完整四轮优化。
已有GPU厚度/面积/曲率继续由主runner调用，不重新实现；整例实测另行报告。


## 61926c7：同一顶点批量碰撞

当前真实剖析：完整41步含cProfile为1687.49秒，其中异步放置累计1419.07秒，
最终相交清理140.51秒；桶构造17.57秒。累计时间含子函数，不能再相加。
这是CPU剖析，不作为无剖析性能值。

新增_vertex_collision_batch把同一顶点关联面一次送入KD查询，再复用原
_candidate_collision进行编译循环。current不修改，所有面仍按原序；整个
顶点检查完成后才接受坐标，保持异步顶点顺序。KD显式return_sorted=False，
半径仍依次加原maximum_radius与1mm，完整候选无上限。存在拒绝试步MHT时
继续执行原逐面路径，保留其状态、候选重试和清理。没有同时更新顶点或GPU
近似。内部输入为当前float32(N,3)坐标、int32(F,3)面、有序关联face_ids、
vertex、float32(3,)终点、当前固定KD树及最大面半径mm；返回bool。
_incident_face_geometry返回关联面坐标、double中心/半径和float32边界；
_incident_faces_collide输入这些量及完整候选CSR并返回首次碰撞bool。
没有独立官方CLI、精度/线程新参数；限制是只用于无拒绝试步状态的顶点。

32项专项测试通过；四半球真实原法向和white/pial坐标下的候选CSR完全相同。
桶查询约0.016–0.027秒，旧代码0.222–0.369秒，准备成本另列；这是组件计时。
候选完整无剖析冷JIT耗时1232.634秒，41步、四轮结束26/32/36/41及清理2→6→0均与优化前相同；有序面、全部坐标和文件SHA完全一致。无剖析、分别使用空JIT缓存的同主机4线程配对为1453.060→1232.634秒，单次观察减少15.17%（1.179倍），文件SHA相同。共享负载下只有一次新/旧顺序配对，未验证稳定吞吐；同输入三方结果已完成：Python的全部坐标及有序面与官方8.2.0完全相同，当前Conda与官方P99为0.225255mm、最大1.482209mm，与冻结Conda几何相同。原始报告见[三方JSON](../../validation/recon_all/optimizations/20261001_serial/stage5/native_reference.json)。Conda176.787秒、官方133.863秒；三者计时均含阶段读写与加载，Python另含冷JIT。当前Python明显较慢，不默认替换。生产仍保留完整Conda white/pial，整例提速单独测量。
第一版只缓存整数索引的完整候选剖析已中止，保留日志，不计入性能结果。


## 2026-10-03：缩步上限后的拒绝试步

特别说明成熟完整 pial 子函数的控制流缺口：上游 `MRISpositionSurface` 的第三次缩步后若 RMS 仍升高，会恢复本步起始坐标并结束当前轮；原 Python 函数在相同情况下报“rejected every trial”。本轮候选保留当前坐标、进入下一轮，最后仍执行内侧壁固定和相交清理；非终止拒绝继续重试。未更换生产 Conda 程序，也未增加 GPU/CPU 回退或更改精度。默认无诊断回调时不建立试步轨迹列表。

现有所有参数：`subject` 是含本文七项输入的目录；`hemisphere` 必须是 `lh` 或 `rh`；`output=None` 默认 `surf/{hemi}.pial.T1`；`max_steps=200` 是四轮累计上限，未收敛抛 `RuntimeError`；`sampling_backend="cpu"` 可选 `cpu/torch/triton`；`regularization_backend="cpu"` 可选 `cpu/torch`，GPU固定网格梯度见[专页](PYTORCH_PLACEMENT_REGULARIZATION.md)；`candidate_backend="tree"` 可选 `tree/snapshot/torch_snapshot`，完整GPU候选见[碰撞专页](PYTORCH_PLACEMENT_COLLISION.md)；`device=None` 仅CPU允许，GPU采样、Torch正则梯度或Torch候选须显式指定；`trace_callback=None` 可指定只读函数 `(step, pass_index, coordinates_copy, diagnostics)`；`profile=False` 可显式开启同步目标GPU的分步骤墙钟剖析，字段见[正则梯度专页](PYTORCH_PLACEMENT_REGULARIZATION.md)。坐标为 `(N,3)` float32、surface RAS/mm，面为输入 white 的 `(F,3)` 有序整数索引；MRI沿用其conform网格。诊断字典的 `trials` 列表新增 `trial`、`dt_used`、`dt_next`、`sse`、`rms`、`reduced`、`rejected`、`stop`、`reductions`，不改变已有诊断字段。终止拒绝时回调坐标是已还原的当前坐标，SSE/RMS保留该步起点值；拒绝试步本身的指标在 `trials` 内。最终报告增加实际 `regularization_backend`，标准CLI不变，此内部优化器仍没有独立官方CLI。

```python
from fnit.recon_all.place_pial_python import place_pial_t1
trial_records = []
pial_report = place_pial_t1(
    subject="/data/self/sub01",  # FNIT自产完整七项前置输入
    hemisphere="lh",  # 当前半球
    output="/data/diagnostics/sub01/lh.pial.T1",  # 独立输出表面，surface RAS/mm
    max_steps=200,  # 四轮累计迭代上限
    sampling_backend="triton",  # 已有GPU强度采样；不启用半精度
    candidate_backend="snapshot",  # 完整动态碰撞候选
    device="cuda:0",  # 显式GPU
    trace_callback=lambda step, pass_index, coordinates_copy, diagnostics:
        trial_records.append((step, pass_index, diagnostics)),  # 保存各试步接受/拒绝状态
)
```

服务器7项单元回归通过（10.22 s，6项缓存+1项完整四轮控制流）；控制流测试强制终止拒绝，验证四轮均完成、顶点恢复、面和输出保留，不代替真实benchmark。初次测试因测试表面缺体积几何失败，补齐合法测试输入后通过，原失败记录仍保留服务器会话输出。同输入完整阶段序列已在 gpucw1 排队：white.preaparc/最终white分别比较官方与当前Conda；完整pial比较官方/Conda/Python，先LH后RH。Python white只有前缀，不参与完整white三方。各stage单独取得并释放共享锁，输入始终从同一旧自产冻结目录复制，官方输出不喂入下一项生产或候选。此处是算法诊断，不能冒充本轮10例整例。

官方固定CLI不输出每轮完整坐标快照，因此本轮native报告逐轮步长、SSE/RMS、拒绝次数和清理消息；Python额外保存四轮结束表面与全部试步。仅有日志标量时不宣称完整坐标逐轮对应。机器报告和复现入口见 `validation/recon_all/accuracy_20261003/task_05/placement_probe.py` 与 `run_placement_stages.sh`；阶段真实结果、GPU速度及10例指标尚待返回，整体等效仍为 `not_assessed`。
