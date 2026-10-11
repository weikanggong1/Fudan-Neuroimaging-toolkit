# 有序 GA 初始基底：完整独立 producer

## 1．功能简介

本任务补齐原拓扑修复在候选搜索之前的完整基底构建。算法复用既有映射
和源一环处理，增加五个向量、十五个标量、独立入射顺序及源零分配状态。
完整函数和流程图见[功能页](../../../../docs/recon_all/TOPOLOGY_GENETIC_BASE.md)。

## 2．Python 调用与输入输出

```python
from fnit.recon_all.topology_genetic_base import create_genetic_base

base = create_genetic_base(
    faces=source_faces,                         # 原有序整数(F,3)
    defect_labels=source_component_labels,      # 0外部，1…D连通缺陷
    defect_status=source_retention_flags,       # 缺陷内0 KEEP／1 DISCARD
    source_vertex_fields=source_vector_fields, # 五个float32(N,3)真实完整状态
    source_scalar_fields=source_scalar_fields, # 十五个float32(N,)完整copied属性
    source_neighbors=source_one_ring,           # 原有序一环，不能由剩余面重建
    source_incident_faces=source_incident_rows, # 原有序入射面编号
    source_incident_corners=source_corner_rows, # 同顺序角点0／1／2
    source_vertex_rip=source_rip,                # 原状态标志，新rip不直接复制
    source_metadata=source_metadata,            # surface RAS／体积／半球元数据
    vertex_translation=None,                   # 自动复用已有映射函数
    face_translation=None,                     # 自动复用已有映射函数
)
```

所有输入、单位、失败规则和完整返回值在功能页逐项列出。不能拿平滑
原网格的法向代替恢复 TMP 后的法向，不能用参考 after 填补缺失属性。

## 3．命令行

复现工具是 `tools/benchmark_recon_topology_genetic_base.py`，完整具名参数
见功能页。参考格式为 FNITBAS1；解析器包含 shape、capacity、角点、
截断和尾部检查。新目录保存私有候选 NPZ、typed JSON 和 COMPLETE。
生产不新增原软件命令调用。

## 4．原软件对照

原步骤属于固定 d932 `MRIScorrectTopology()`，没有独立 CLI。
`build_base_observer.py` 复用固定完整 fitness 观察器的编译收据，在真正
TMP restore＋metric 后和初始构建后读取状态；不重算或改写源网格。
只读 source SHA、精确规则和行号保存在 `source_rules.json`。

## 5．当前精度、时间和图

8项完整 API 合同和2项捕获格式合同通过（0.019秒）。测试包括原顺序悬空边、独立
入射排序、全部 copied 位模式、NaN payload 只复制、体积几何缓存例外、全缺陷状态和非法
映射／截断拒绝。它们是接口测试，不能代替真实脑影像 benchmark。

目前新增真实 before／after 观察脚本上传仍待明确授权；没有将不存在的
捕获写成通过结果。原程序重复、真实完整阶段时间、脑图、完整 GA和
整例未完成，严格同输入及整体指标等效未判定。

## 6．版本和记录

2026-10-10：首次完整 producer 候选；复用成熟函数，不改生产默认。
源码和合同报告在真实运行时冻结绑定；未复制发布原 C++ 文件或程序。
本目录 pending 状态不会标成真实成功。

## 7．参考

[固定 mrisurf_defect.cpp](https://github.com/freesurfer/freesurfer/blob/d932c45b7941662ea380a05efef580568b98d41a/utils/mrisurf_defect.cpp)
和[功能页](../../../../docs/recon_all/TOPOLOGY_GENETIC_BASE.md)的文献及
零分配、类型定义链接。原程序仅允许进入独立 benchmark，生产边界不变。
