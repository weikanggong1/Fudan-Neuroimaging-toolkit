# 拓扑缺陷的有向距离前置

## 1．功能

`defect_signed_distance()` 用自有 Python/Numba 计算固定拓扑算法的完整 active `new_loop3` 距离。复用项目已有角度伪法向和有序相邻面函数；保留源三角面、边、角点的距离分支以及相同绝对距离时严格正值优先的归并。不是二值分割的 EDT，也不按普通点到三角面距离替换符号规则。

```mermaid
flowchart LR
  A[当前 orig 坐标 / 面法向 / 原有序 incident CSR] --> B[扩展 bbox 选面]
  B --> C[复用角度及边伪法向]
  C --> D[面内投影 / 边 / 角点]
  D --> E[有向最小距离 XYZ]
  E --> F[既有体积似然与 curvbak]
```

本模块是距离前置，预备 MRI/PDF 生成、候选控制器和完整 GA 由其他模块负责。函数不运行原版程序、不读参考、不写文件、不改变几何、设备或全局线程设置。`backend="serial"` 保持默认；显式 `backend="parallel"` 只将 X 轴分成互不重叠的体素区间，每个体素仍按完整原面序计算。GA 候选之间的状态更新保持串行。

## 2．Python 调用、输入与输出

```python
from fnit.recon_all.topology_defect_signed_distance import defect_signed_distance

distance_result = defect_signed_distance(
    original_vertices=current_original_coordinates,  # float32(N,3)，source orig surface RAS，mm
    faces=current_triangular_faces,                  # 整数(F,3)，与当前坐标对应
    face_normals=current_source_face_normals,         # float32(F,3)，当前 getFaceNorm 字段，无量纲
    incident_offsets=original_incident_offsets,       # 整数(N+1,)，原 vt.f/vt.n CSR 偏移
    incident_faces=original_incident_face_indices,    # 整数(E,)，保持原 incident 次序及重复
    incident_corners=original_incident_corner_indices, # 整数(E,)，顶点在对应面的0/1/2角点
    volume_shape=local_defect_volume_shape,           # 三个正整数，局部 mri_defect 的 XYZ shape
    volume_origin_mm=local_defect_volume_origin,      # float32(3,)，mri_defect xstart/ystart/zstart，mm
    volume_index_scale_per_mm=local_defect_index_scale, # 正float32(3,)，mri_defect xsize/ysize/zsize，mm⁻¹
    ordered_candidate_faces=None,                    # 默认全部面；可用同源保守空间索引，不得遗漏扩展bbox相交面
    backend="serial",                                # 默认有序串行；parallel使用当前Numba线程预算按X轴并行
)
current_signed_distance = distance_result["distance_mm"]  # float32(X,Y,Z)，mm；固定1000为未处理哨兵
```

全部参数除 `ordered_candidate_faces=None` 和 `backend="serial"` 外必填。`backend` 仅接受 `serial`/`parallel`，不自动选择或回退。并行使用调用者当前的 Numba 线程数，最多同时处理 X 个非空区间；调用者须与其他任务共享总线程预算。线程数在 Python 层读取，编译内核可保存 JIT cache，不改变全局线程设置。坐标不是 scanner RAS；局部 `mri_defect` 的 scale 是坐标到索引的**乘数**。普通 MGH 存储头的 spacing/origin 可能不同，不能替代它。源索引是 float32 减/乘后加 double 0.5、向零截断，不是 NumPy 的四舍五入；反向采样是 float32 `origin + index / scale`。

返回字典：

| 字段 | 结构、单位和含义 |
| --- | --- |
| `distance_mm` | `(X,Y,Z)` float32/mm；在每个局部体素选择原分支最小绝对距离，严格正值优先；未处理为1000。 |
| `buffer_bounds` | `(B,7)` int64：fno、未裁剪 i_min/i_max、j_min/j_max、k_min/k_max；执行时再裁剪到体积。 |
| `buffer_coordinates` | `(B,9)` float32/mm；每行原 x0/x1/x2、y0/y1/y2、z0/z1/z2。 |
| `other_edge_faces` | `(B,3)` int64；复用原 incident 次序找到三个边的相邻面，边界使用源 self-face 回退。 |
| `query_vertices` | `(Q,)` int64；所有选中面消费的顶点。 |
| `query_pseudonormals` | `(Q,3)` float32；同几何角度加权伪法向，不再单位化。 |
| `scope` | 距离前置范围，不表示完整 unMRI/GA。 |

两个后端返回相同字段，使用独立数组，输入不修改。非法后端、形状、索引、CSR、角点关系、非有限几何或非正 scale 抛 `ValueError`；非有限计算结果抛 `FloatingPointError`。不加 epsilon、不丢弃 rip 面、不将捕获的公共初始零距离作为最小距离初值；active 源私有映射从 NPY=1000 开始。每次调用重新生成当前几何的 bbox、邻面和伪法向，不跨候选保存距离或坐标缓存。

## 3．命令行

子项没有生产 CLI。隔离完整回归工具：

```bash
python tools/benchmark_recon_topology_signed_distance.py \
  --capture isolated_reference/repeat0/distance.bin \
  --native-repeat isolated_reference/repeat1/distance.bin \
  --volume-capture isolated_reference/same_candidate_volume.bin \
  --observer-receipt isolated_reference/build.public.json \
  --output isolated_diagnosis/new_signed_distance \
  --threads 4
```

`capture/native-repeat` 是两个独立参考进程的完整 `FNITDST1/2` 状态；`volume-capture` 是**同一候选**的 `FNITVOL1`，提供预备 MRI/PDF 和反馈初值；新 `FNITDST2` 另记录实际 `mri_defect` 采样几何，并与它逐字节交叉核验。工具自己的距离送入既有体积聚合，不把参考距离传入生产函数。`observer-receipt` 绑定固定源码与独立编译程序；`output` 必须不存在。`threads` 默认4，复用现有线程预算并恢复调用者。私有 NPZ/MRI 不发布。

## 4．原软件调用

对应固定 d932 的 `mrisComputeDefectMRILogUnlikelihood_wkr` active 距离分支及 `updateDistanceElt*`。内部辅助步骤没有独立 CLI；隔离参考使用 Conda 固定源码构建的 `mris_fix_topology` 只读观察器，种子1234、4线程。观察器记录首次真实 unMRI 之前/之后距离，然后停止于 GA 搜索前。它不进入候选运行依赖。

保留源 `new_loop3` 分支、`SIGN(0)=-1`、三边的严格较小替换顺序、中心近似 `.33f/.3333333f`、`.98f/1.01f` 跳过规则。没有通过数学上更漂亮的公式替换源运算。

## 5．真实精度、耗时与图示

当前 v3 冻结输入是公开 ds000114 sub-07 ses-test 左半球，原始自产链79a41cdd；完整110,534顶点、216,860面、650,580 incident 项、1,628选面和903伪法向顶点，距离体积21×19×16。

- 完整距离图、候选 bbox、坐标、有序邻面、903×3伪法向的不同元素数和最大/P99误差均为0，逐位一致。
- 自产距离送入既有体积聚合：八个源字段与完整110,534元素 `curvbak` 逐位一致；全部候选数组的两次完整重复一致。
- 原参考两次 raw capture SHA 不同，来自 OMP 缓冲区追加次序；按 fno 对应的几何及最终距离相同，不称完整捕获逐字节重复。

成熟伪法向函数有一项类型错误：旧实现把原 C++17 实际 `sqrt(float)`/`acos(float)` 重载和两次 float 除法提升到 double。原翻译单元类型探针和非直角三角面 rawbits 合同确认后，对应辅助函数按 float32 恢复原顺序。v2的656/2709尾差消失；距离核心源码66b84289未改，不降低比较门槛。[类型探针](../../validation/recon_all/optimizations/20261010_topology_source_unmri_observer/reports/numeric_type_probe_20261010_v2/types.json)和[当前完整报告](../../validation/recon_all/optimizations/20261010_topology_signed_distance/reports/v3/report.json)绑定各自实际源码及程序 SHA。

| v3 候选范围 | 首次 API 内观察（s） | 热重复 API 内观察（s） |
| --- | ---: | ---: |
| 全216,860面 bbox 扫描 | 0.224471 | 0.013059 |
| 相邻面查询 | 0.005455 | 0.001718 |
| 角度伪法向 | 0.005896 | 0.003284 |
| 有向距离内核 | 0.457269 | 0.060770 |
| 验证、复制及返回 | 0.003187 | 0.002759 |
| **完整距离 API** | **0.696277** | **0.081590** |

首次 API 已可用6项合同产生的部分 JIT cache；不同内核仍有首次成本，不作为完全冷启动。后续体积聚合0.522922s；导入1.582254s另列。工具含导入、校验、加载、两次 API、NPZ 和绘图5.690778s，不含前置 bootstrap 和最后 JSON。嵌套时间不与完整 API 重复相加。原参考两次观察器6.483830/5.633459s含其他准备与捕获，无距离独立 timer，因此不作原函数加速比。CPU四线程；候选亲和性100–103、参考8–11，同主机共享负载未隔离。GPU未用、显存未测，不声称整例10分钟或整体指标等效。

![当前实际有向距离与源对照](../../validation/recon_all/optimizations/20261010_topology_signed_distance/reports/v3/distance.png)

图仅隐藏双方未处理的 NPY=1000 哨兵以展示脑内局部距离，计算数组及比较仍保留全部哨兵。首版诊断误用距离存储头作为采样几何，导致选面完全不同，失败收据及旧图保留。正确 geometry 来自同一候选 `mri_defect`；两张局部图的 shape 相同、头含义不同，不能混用。

第四缺陷首候选的[此前冻结剖析](../../validation/recon_all/optimizations/20261011_topology_ga_nine_performance/README.md)中，普通暖完整五项0.534501秒、完整距离0.520358秒；带cProfile的 `_distance` 自身0.459655秒。它们仍绑定旧源码，不能改标为新后端实测。

新增后端的[本轮独立五组真实回归](../../validation/recon_all/optimizations/20261011_topology_distance_slabs_fourth/README.md)已完成：同自身第三缺陷完整检查点和三个原输入，第四首候选每组冷/热/审计三次，72候选字段＋6距离数组的shape/dtype/bytes及完整控制全部相同，自重复、零随机消费和2,656,862次谓词顺序通过。

| 本轮后端 | 首次距离 API / s | 暖距离 API / s | 暖完整 candidate cycle / s | fresh CLI（三次及全部自产前置）/ s |
| --- | ---: | ---: | ---: | ---: |
| 原串行 | 3.677842 | 0.475060 | 3.772341 | 148.295691 |
| 新串行 | 3.855348 | 0.458801 | 3.710041 | 144.793849 |
| 新 parallel2 | 3.512684 | 0.223670 | 3.037098 | 127.196744 |
| 新 parallel4 | 3.941938 | 0.144773 | 3.152171 | 133.807767 |

暖距离2线程约快2.124倍、4线程约快3.281倍；4线程首次距离比原串行慢。完整cycle是实测观察，但未改的retess也从2.092180降至1.842294秒，不能把全部cycle/CLI下降归因于距离。外围总预算4，parallel2只在距离期间用mask2；原生产每worker2线程的完整接线回归尚未测。CUDA隐藏，GPU allocated/reserved为null；本轮没有原版程序或整例运行。诊断包装器明确记录 `global_function_replacement=True`，结束恢复函数及mask，不是生产接口接线。默认serial保持，整体指标等效未判定。

本地[距离并行合同](../../tests/recon_all/test_topology_signed_distance_parallel.py)与原距离合同共 14 项、48 个子项通过，耗时 18.06 秒。另将 Git `5def77e6` 中原串行源码 `66b84289…` 与候选源码 `11c360b0…` 比较：7 类输入 × 串行/并行1/2/4线程共28组，全部6个返回数组的 shape、dtype 和 bytes相同，调用者线程预算恢复。它们是边界与实现回归，不是真实 T1 benchmark 或整例速度结果。

## 6．更新与 benchmark

- v1：6项合同通过，完整真实回归失败；记录错误诊断几何引起的6363不同体素，不把失败隐藏成浮点尾差。
- v2：采样几何正确后，距离和后续完整反馈逐位通过；内部伪法向尚未全部严格通过。参考两次 OMP 行序差异保留原始SHA。
- v3：正式伪法向类型修复及新 FNITDST2 完整采样输入冻结，六组内部字段、完整距离、八项聚合及反馈全部逐位通过。历史 v1/v2 不改标。
- 2026-10-11：新增显式体素区间并行后端。每个体素的面序、float32中间值、符号零、严格正值tie、中心近似和剪枝保持；默认串行。局部合同覆盖1/2/4线程、非整除区间、单X维、空候选、空写区间、范围裁剪、重复有序面、近零和坐标改变；第四冻结同输入15次完整78字段门通过。完整GA和整例待验证。
- 串行距离已由完整候选评分调用；预备MRI/PDF、动态控制器、拒绝状态恢复与最终网格分别验证。新增并行后端尚未接入生产调度，整例耗时尚未改变；不将历史冻结报告或原型包装当作现版整例验收。

## 7．参考文献与原代码

[固定 active unMRI 源函数](https://github.com/freesurfer/freesurfer/blob/d932c45b7941662ea380a05efef580568b98d41a/utils/mrisurf_defect.cpp#L1838-L3030)，[距离归并原函数](https://github.com/freesurfer/freesurfer/blob/d932c45b7941662ea380a05efef580568b98d41a/utils/mrisurf_metricProperties.cpp#L11079-L11195)。Ségonne F, Pacheco J, Fischl B. IEEE TMI 2007;26:518–529，[DOI](https://doi.org/10.1109/TMI.2006.887364)。来源与许可见项目既有通知；不发布原 `.cpp` 或原版二进制。
