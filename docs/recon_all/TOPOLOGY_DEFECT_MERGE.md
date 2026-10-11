# 邻近缺陷的独立有序合并

## 1. 功能简介

`merge_neighboring_defects()`补齐原组件分割之后、retention和初始基底之前的邻近缺陷合并。它复用已有组件BFS和原面顺序邻接，用NumPy保存源遍历、合并和字段写回规则，不调用原版C++程序。

固定源码关闭`ADD_EXTRA_VERTICES`与`FIND_ENCLOSING_LOOP`，但开启`MERGE_NEIGHBORING_DEFECTS`。因此单纯的`defect_component_labels()`只完成前一步，不能直接作为完整GA的最终缺陷域。

```mermaid
flowchart LR
  A[自产球面与原表面面积] --> B[已有组件分割]
  B --> C[原组件sum面积写val]
  C --> D[按border接触计数合并]
  D --> E[保留concat inside与有序border]
  E --> F[retention和显式顺序初始基底]
```

## 2. Python 调用、输入和输出

```python
from fnit.recon_all.topology_defect_merge import merge_neighboring_defects

merged_defect_report = merge_neighboring_defects(
    canonical_vertices=centered_spherical_vertices,  # 同顶点顺序、自产中心化球面坐标
    initial_vertex_normals=precenter_metric_vertex_normals,  # 进入拓扑函数前、未中心化球面的法向
    vertex_normals=canonical_metric_vertex_normals,  # 中心化后，实际合并之前的球面法向
    original_vertex_area=raw_original_vertex_areas,  # 原nofix表面的origarea，不能传球面面积
    faces=ordered_original_faces,  # 原有序三角面，顶点编号不变
    component_labels=premerge_component_labels,  # 已有分割的0与连续1..D标签
)
```

| 输入 | 格式、空间和单位 |
| --- | --- |
| `canonical_vertices` | 有限float32 `(N,3)`，surface RAS/mm，已完成源球面中心化 |
| `initial_vertex_normals` | 有限float32 `(N,3)`，未中心化canonical metric法向，无单位；初始化组件border法向 |
| `vertex_normals` | 有限float32 `(N,3)`，中心化后球面metric法向，无单位；合并时计算内部法向均值 |
| `original_vertex_area` | 有限非负float32 `(N,)`，raw表面原面积，mm² |
| `faces` | 整数 `(F,3)`，原面顺序和N顶点索引 |
| `component_labels` | 整数 `(N,)`，0为外部，连续1..D为完整连通premerge组件 |

全部参数必填，没有可调算法阈值、设备或精度默认。完整raw metric和组件发现由上游提供，本函数不从参考文件取属性。

| 输出 | 结构和含义 |
| --- | --- |
| `regions` | tuple；每项含有序int32 `vertices`/`border`、全KEEP0 `status`、旧0起始`source_component`、F32 `area`及`centroid`/`normal` |
| `labels` | compact int32 `(N,)`，未清空旧组件按编号重新排序，0保持外部 |
| `source_order` / `vertex_translation` | int32 `(N,)`，外部升序后接各合并列表原concat顺序，以及逆映射 |
| `initial_vertex_val` | F32 `(N,)`，保留原分割组件sum面积在merge前写入的val；合并不回写 |
| `source_marks` | int32 `(N,)`，合并函数退出时原未压缩组件标记；后续清mark由调用者处理 |
| `merge_events` | int32 `(M,3)`，目标旧编号、并入旧编号、接触次数 |
| `seconds` / `source_macros` / `phase` | 本API墙钟、固定宏及字段相位说明 |

`area`沿用源字段：未合并时为origarea有序F32总和；合并时变成原有序总和除内部顶点数。`centroid`为surface RAS/mm的F32均值；合并后的`normal`为内部法向均值，源代码未再单位化。不能把这些字段统一改成面积总和或单位法向。

合并列表可能在原网格上不直接连通，合并是通过border接触判定的；重新从compact标签BFS会改变原concat顺序。后续基底必须消费显式`regions[*]["vertices"]`，不能假设旧的单组件BFS顺序仍正确。

输入不改变。非法类型、索引、非连续或断连premerge标签抛`ValueError`；F32统计溢出抛`FloatingPointError`。原宏分支固定，不支持额外加顶点或包围环配置，不替代retention、GA、最终网格质量检查，也不改变CUDA/TF32/线程设置。

## 3. 命令行调用

无独立生产CLI。只读源码与本地合同可复跑：

```bash
PYTHONPATH=src python validation/recon_all/optimizations/20261010_topology_defect_merge/run_local_contracts.py \
  --fixed-source /benchmark/fixed/utils/mrisurf_defect.cpp \
  --output /benchmark/new-run/defect-merge-contracts.json
```

`--fixed-source`可选，只读核SHA与规则锚点；`--output`必须为新JSON路径。新增依赖为空，使用主页Conda环境现有NumPy和已安装FNIT。

## 4. 对应原软件

对应`mris_fix_topology -ga`内部的`mrisMergeNeighboringDefects`，不具有单独原CLI。参考路径仅在独立benchmark中使用固定版本、输入、种子和线程；生产不执行它。

```bash
mris_fix_topology -mgz -ga -seed 1234 -threads 4 reference_subject lh
```

原macro header及源文件SHA随验证报告记录，不从“没有额外顶点”推断“无需合并”。

## 5. 精度和耗时

本轮6个本地合同覆盖接触阈值、回查并入更早组件、原concat顺序、F32面积累计、merge前val保留、空域及失败行为。[合同报告](../../validation/recon_all/optimizations/20261010_topology_defect_merge/local_contracts.json)绑定实际代码SHA与HEAD。

初版六合同和真实链保存其原源码SHA，不改标为新接口结果。后续只读核对启动链发现两个法向相位：`mris_fix_topology.cpp`先计算未中心化canonical metric，再进入`MRIScorrectTopology`中心化；初始化组件border法向仍使用此前字段，实际merge则消费重新计算的中心化法向。新接口显式分开两者，新增相位合同；旧链的normal自身仍未验收，顶点映射、面积和K门分别报告。

真实原始表面→merged域→完整基底的连续比较由[基底准备](TOPOLOGY_GENETIC_BASE_PREPARE.md)组织；该阶段结果单列，不能用组件合同或后续K1/K2冻结同输入通过替代。完整GA、原始T1整例、整体指标等效和显存未在本页宣布通过。该函数不生成脑图，表面与缺陷叠加由完整拓扑链报告提供。

## 6. 最近更新记录

- 2026-10-10：补齐已开启的邻近缺陷合并；保留原合并顺序、F32平均字段及merge前val。旧组件探测仍用于真正的premerge阶段，不删除仍承担验证作用的函数。
- 同日审查retention内层`j<ninside`：与固定源码一致，没有border索引越界；未因疑似问题改变距离规则。
- 该独立合并接口仍处于实验连续接线阶段；生产默认未改。
- 同日补齐必填`initial_vertex_normals`，保留未合并组的旧相位法向；真实新版本由基底准备链另行冻结验证。

## 7. 参考文献和源码

- [固定源邻近合并](https://github.com/freesurfer/freesurfer/blob/d932c45b7941662ea380a05efef580568b98d41a/utils/mrisurf_defect.cpp)，7518–7709附近的`mrisMergeNeighboringDefects`，9234–9363附近的组件字段初始化。
- [固定构建宏](https://github.com/freesurfer/freesurfer/blob/d932c45b7941662ea380a05efef580568b98d41a/include/mrisurf_base.h)，额外顶点、包围环及邻近合并开关。
- [Fischl等，2001，Automated manifold surgery](https://doi.org/10.1109/42.906426)，皮层网格拓扑修复方法。
