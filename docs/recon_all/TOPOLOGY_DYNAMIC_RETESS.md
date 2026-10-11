# 拓扑缺陷的有序动态重连与候选面发现

## 1. 功能简介

`retessellate_ordered_edges` 对一个完整缺陷执行源顺序的边接受、相交拒绝、
顶点分类和重试；`retessellate_and_detect_candidate` 再发现、检查 attachment
并定向新三角面。计算使用自有 Python/Numba，复用已有球面相交和面发现
函数，不执行原版程序。不从闭合原网格猜测缺陷基底，不以径向投影、复制
文件或删除困难顶点代替动态重连。

```mermaid
flowchart LR
    A[自产缺陷基底和原有序邻接] --> B[自产完整边表和排序]
    B --> C[有序接受 / 阻挡 / 顶点分类]
    C --> D[源重试与临时边状态收敛]
    D --> E[选择 retain inside 和完整 border]
    E --> F[面发现 / attachment / 球面定向]
    F --> G[MRI形变 / 自产主曲率 / 五项fitness]
    G --> H[GA搜索 / 最终网格质量]
```

本页完成 C–F 的首次真实候选回归；G–H 不属于该接口。`some` 模式和 overlap
重试有结构契约，当前真实数据只覆盖 `all`、无 overlap 的首候选，不能据此
声称所有搜索分支或完整拓扑修复已经通过。

## 2. Python 调用、全部输入与输出

```python
from fnit.recon_all.topology_retessellation_dynamic import retessellate_and_detect_candidate

candidate_state = retessellate_and_detect_candidate(
    canonical=canonical_vertices,  # float32 (N,3)，自产 canonical surface RAS
    base_faces=retained_base_faces,  # int32 (F,3)，已移除全部待修缺陷面
    neighbors=ordered_base_neighbors,  # N行整数一环；保留源顺序和无面支撑边
    candidate_edges=complete_candidate_edges,  # int32 (K,2)，自产完整MRI评分边表
    edge_used=initial_edge_states,  # int32 (K,)，源0…6状态
    ordering=candidate_edge_order,  # int32 (K,)，完整原边编号排列
    inside_vertices=mapped_inside_vertices,  # int32 (I,)，当前网格索引原顺序
    inside_discarded=initial_discard_flags,  # bool (I,)，源status==DISCARD，不再判标签
    border_vertices=mapped_border_vertices,  # int32 (B,)，完整原border
    hull_vertices=mapped_hull_vertices,  # int32 (C,)，前B项等于border
    vertex_rip=initial_vertex_rip,  # bool (N,)，保留原不可用顶点语义
    fixed_mapping=None,  # 缺省全true；实验optimal-map可显式传bool (N,)
    mode="all",  # 缺省all；some包含动态顶点分类和删除邻接
    overlaps=None,  # 缺省无表；可提供K行自产原边编号，保留顺序和重复项
    overlap_complete=None,  # 有overlap表时必需bool (K,)，原完整性标志
    blocker_scan_backend="python",  # 默认参考扫描；可显式numba完整编译扫描
    blocker_scan_audit=False,  # True逐条核原谓词顺序，包含审计开销
)
new_faces = candidate_state["faces"]  # 全部保留面加自产有序新面，int32
selected_vertex_ids = candidate_state["selected_vertices"]  # retain inside随后全部border
selected_face_ids = candidate_state["selected_faces"]  # 原面表之后新增面编号
```

`canonical` 必须为有限、非零三维点；算法转换 float32，坐标采用原 canonical
surface RAS 的尺度。`base_faces`、邻居和全部域列表必须是整数索引，浮点或
溢出的大整数在转换前拒绝；不会静默截断。邻接必须对称、无自环或重复。
base 的面顺序与编号保留，不能从面重新构造邻接而丢掉没有面支撑的原边。

重复候选可显式使用`base_context=None`的新可选参数。先调用
`prepare_retessellation_base(canonical=..., base_faces=..., neighbors=...)`，
再同时传它自身的三个不可变对象与context；普通调用不变。缓存只维护
完整基底的有序邻接与入射，按触及行复制；不会跨坐标迭代缓存动态状态。
结果新增`prepared_base_used`和`prepared_copied_rows`。完整输入、失败规则
与真实热ABBA见[不可变基底缓存](TOPOLOGY_PREPARED_BASE_CACHE.md)。

`blocker_scan_backend`默认`python`，显式`numba`将同一个完整首阻挡扫描
放入已有Numba谓词核；每次加删邻接同步查询行，不缩减候选或变更全N
返回状态。`blocker_scan_audit`接受bool/np.bool_，默认False；开启后返回
实际谓词调用数与完整有序五列SHA，普通Python不开审计时计数为None。
组合接口分别报告`retessellation_seconds`与`face_detection_seconds`。
完整参数、真实第四缺陷72字段及全序列门见[有序阻挡扫描](TOPOLOGY_ORDERED_BLOCKER_SCAN.md)。

`edge_used` 的源编号是 unused=0、fixed=1、original=2、new=3、both=4、
temporary-new=5、temporary-both=6。每轮按 `ordering` 执行，更新使用即时
邻接；不是把所有边同时更新。完整 overlap 表只补检 hull[B:]，不完整表
检查 inside+完整 hull。缺省无表也检查全部后者，未删掉远端阻挡边。

| 返回字段 | 结构与含义 |
|---|---|
| `faces` | int32 (F,3)；边接口保留基底，组合接口追加新面 |
| `neighbors` | N行独立Python整数列表，当前有序一环 |
| `edge_used` / `marks` | int32 (K,) / (N,)，当次源状态 |
| `selected_vertices` | int32，实际retain inside随后完整border，原顶点顺序 |
| `selected_edges` | int32，按ordering收集的new/both原边编号 |
| `ninside` / `ndiscarded` | 当次新增inside数 / 当次未用非初始discard数 |
| `removed_vertices` | int32，some模式本次顺序移除的顶点；不重新编号 |
| `temporary_faces` | some分类使用的临时面；不冒充最终新面 |
| `retessellation_trace` / `rounds` | 完整自产有序边判定记录 / 实际轮数 |
| `patch_faces` / `selected_faces` | 组合接口的新面(int32 M×3) / 新面表编号 |
| `face_decisions` | 组合接口的面包含、edge incidence和接受记录 |
| `complete_ga` | 固定false；未做完整GA搜索或最终输出验收 |

输入不修改，不写标准文件，不创建 MRI/曲率/PDF。非法 shape、dtype、索引、
非对称邻接或排序抛 `ValueError`；不收敛重试抛 `RuntimeError`。最终网格的
连通性、非流形、自相交、形状和 sphere 翻折须另验收。中间候选不能因
某项子阶段相同而放行成为最终生产输出。

## 3. 命令行与检查点

```bash
# 隔离诊断：唯一算法输入是before；after/statistics只作事后严格比较。
python tools/benchmark_recon_topology_dynamic_retessellation.py \
  --before /workspace/benchmark/retess.before.bin \
  --after /workspace/benchmark/retess.after.bin \
  --statistics /workspace/benchmark/patch_statistics.bin \
  --output /workspace/diagnostics/new-candidate
```

前三者为完整参考诊断文件，最后一个必须为全新目录。`--statistics` 缺省
None，显式提供时覆盖完整面发现回归；只比有序面/索引，不把参考后态坐标
传入算法。输出 JSON 数值报告、冷/重复 NPZ 和两种自产 trace。时间包含完整
算法输入验证；解释器导入/解析/写出另计在 process 收据，不冒充局部 kernel。

新增只读检查点解析器均采用具名 `capture_path`，返回独立数组；格式、
截断、尾部、索引与 CSR 错误抛 `ValueError`：

- `load_retessellation_capture`：FNITRTS1，完整前/后 XYZ、canonical、面、
  邻接、原边状态/顺序、inside status、border/hull及实际selected V/E。
- `load_neighbor_capture`：FNITNBR1，候选 SFF 前的有序 vnum/vtotal/nsizeCur
  和 origXYZ、法向、e1/e2、旧k1/k2、有符号面积。旧主曲率不是自产拟合结果。
- `load_volume_preparation_capture`：FNITPDF1，原 origXYZ/完整ETable、MRI-only
  矩阵、两组织校准 bins/counts/bin_size、resolution/均值、局部几何。
  phase0为初mask；phase1为完整强度和白灰logPDF，参考不供生产回灌。

## 4. 原软件对应步骤

这是 `mris_fix_topology -ga` 的内部动态重连，没有独立官方 CLI：

```bash
# 仅隔离benchmark参考，生产不运行。
mris_fix_topology -mgz -sphere qsphere.nofix -inflated inflated.nofix \
  -orig orig.nofix -out diagnostic.orig.premesh -ga -seed 1234 \
  -threads 4 subject lh
```

固定版本 d932c45b7941662ea380a05efef580568b98d41a；实际函数为
`retessellateDefect_wkr → detectDefectFaces → orientDefectFaces`。
观察器只加只读记录，在首次完整fitness后有意停止，不生成GA最终结果。
构建源码和原程序、加入观测后的程序分别绑定SHA；参考墙钟包含上游及
捕获写入，不能称原版retess子步骤耗时。

## 5. 当前真实数据精度和时间

公开 ds000114 sub07 ses-test LH；输入来自79a41cdd原始T1连续链。CFFF同机、
CPU8–11/总4线程、CUDA隐藏，固定源seed1234。原MRI、权重、程序及输入SHA
保留在[机器可读报告](../../validation/recon_all/optimizations/20261010_topology_dynamic_retessellation/README.md)。

| 范围 | 当前实测 |
|---|---|
| 唯一retess before | 110,534V / 216,722F / 650,530邻接项 / 4,723候选边 |
| 实际域 | 67inside、61border、125hull；modeALL，无overlap |
| 动态边v2 | cold4.631389s / repeat2.847185s；所有邻接/边状态/selected/marks0bitdiff |
| 动态边+完整面发现v3 | cold6.118350s / repeat3.884241s |
| 完整新面 | 追加138面→216,860F；全有序面/selectedFaces逐bit同参考 |
| 实际候选选择 | 38inside+61border；ninside38、当次ndiscard0与源相同 |
| 参考两次retess捕获 | 5.523981 / 5.222239s；before/after字节均相同 |
| MRI preparation两次捕获 | 5.665037 / 5.559619s；两phase全部数组字节相同 |
| 局部MRI | shape21×19×16、scale2mm⁻¹；校准白灰bin_size实际均0，原样保留 |
| MRI白/灰mean | 85.1091003418 / 62.6942176819；mean-log −3.2196474075 / −3.7139668465 |
| before-SFF邻域 | 99selected；vnum=vtotal，度数3–24；完整CSR650,882项 |
| 同候选旧7种捕获 | 新观察器加入后全部SHA不变；原算法输出未被观测改动 |
| 文件格式契约 | 本地9/9、0.022s；声明环境9/9、0.010s（进程0.994142s） |
| 实际数学overload | log(float)/ceil(float)均4bytes；通用fixture bits3197324049/1073741824 |

标准初始化邻域650,882与另一个post-fitness CSR650,580不同，不可混为同状态。
本组没有原retess单独计时，不能给加速倍数；未运行整例、最终质量和整体
指标等效暂未判定。记录的自产 trace 尚未与逐边源 trace 独立比较；当前严格
门是完整前→后状态及完整有序面一致，不夸大为全部源决策都已逐条观测。

## 6. 最近更新与benchmark记录

- v1观察器：局部prepare固定anchor数假设错误；retess编译缩进警告失败，保留。
- v2部署：错误改写旧依赖目录，尚未计算即失败；保留旧目录。
- v3 retess观察器：修正只读C++记录、真实before/after两次稳定。
- prepare v3：记录int→float narrowing/缩进警告失败；v4修正后真实两phase稳定。
- 动态v1：首次完整有序边状态通过；v2在转换前拒绝浮点/大整数索引，并按
  源保留非triangle临时边的continue，真实输出仍完全一致。
- 动态v3：复用成熟面发现/定向，新增138有序面同输入完全一致。
- dependency-v4：只用已提交HEAD依赖4b20a92f…替换旧脏wrapper，完整有序
  边/面/邻接/选择仍0bitdiff（冷6.011898s、重复4.008200s）。所用两个导入
  函数AST相同，未把wrapper实验作为隐藏依赖。新[连续候选几何接口](TOPOLOGY_CANDIDATE_GEOMETRY.md)继续25smooth→40match→法向，另做真实回归。
- ordered-scan：新增显式Numba完整首阻挡扫描，默认Python保持；真实第四
  首候选72字段/有序面与控制/2,656,862谓词序列完全一致。新增9契约、
  旧14契约分别通过；完整默认首缺陷仍需独立完成门，不替换旧运行源。

这些是诊断与自有组件版本，不是新的recon-all整例或默认切换。旧失败记录、
原观察器程序和原始收据留在授权benchmark目录；公开报告仅含去敏标量/哈希，
不发布MRI、NPZ、原程序、编译argv、内部主机/路径或许可证。

## 7. 参考文献与源码

- [FreeSurfer固定拓扑源码](https://github.com/freesurfer/freesurfer/blob/d932c45b7941662ea380a05efef580568b98d41a/utils/mrisurf_defect.cpp)
- [FNIT源码仓库](https://github.com/weikanggong1/Fudan-Neuroimaging-toolkit)
- [拓扑MRI边评分](TOPOLOGY_MRI_SCORE.md)、[完整fitness诊断](TOPOLOGY_FIRST_FITNESS_DIAGNOSTICS.md)、[准备后五项评分](TOPOLOGY_PATCH_FITNESS_ASSEMBLY.md)

本实现沿用项目已有NumPy/Numba依赖，无新增安装包。固定原源码只承担
独立参考验证；本文的生产可调用函数不链接或启动FreeSurfer阶段程序。
