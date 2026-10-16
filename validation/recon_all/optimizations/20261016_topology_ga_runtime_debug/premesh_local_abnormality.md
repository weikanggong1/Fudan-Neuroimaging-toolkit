# sub-07 右半球同输入 premesh 局部异常

本页只分析 2026 年 10 月 16 日已完成的 **同五份 FNIT 自产上游输入**阶段诊断，不重新计算距离，也不评价新整例。比较对象是 Conda 独立编译 C++ 拓扑程序和当前 Python GA 各自输出的 `rh.orig.premesh`。二者分别有 109,539／109,503 个顶点和 219,074／219,002 个面，顶点对应关系不存在；下列数值均为**双向顶点到完整目标三角面距离**，不是同索引顶点误差，也不是连续 Hausdorff 距离。空间为 surface RAS，长度单位为 mm。

## 证据与局部位置

完整私有距离数组保留在 `$FNIT/runs/<private-run>/diagnostics/topology_ga_runtime_debug_20261016/premesh_distance_v1/ds000114_sub-07_rh/all_vertex_distances.private.npz`，SHA-256 为 `d5907c378366b6d47a404fda25d8efdea704c9a8536ba43e84f7ccf54eff9a87`。对应完整 `report.json` SHA-256 为 `c9c8016a1a4b2bf0a0a7150f9132521e9ab06869a20e5b3a9147567fb7bfb9bf`，同输入网格质量检查 SHA-256 为 `86da654d31bd8007ca1f0f9acdb875a1ed1e62ecb55bfa73ff84842e21769022`。C++ 表面 SHA-256 为 `468a4a6a0fd8d43129dfc16a6f2c5c8f316b65169ec796033f9d156e2f5309e4`，Python 表面为 `0d91f5d8f11b67c4072bc4f186a08871e1bca95a860b769d2578be316f0c14e6`。

| 测量方向 | 平均 | P99 | 最大 | 距离 > 0.1 mm | 距离 > 1 mm |
| --- | ---: | ---: | ---: | ---: | ---: |
| C++ → Python 三角面 | 0.000965 | 0 | 7.871200 | 246 | 6 |
| Python → C++ 三角面 | 0.000572 | 0 | 0.824361 | 202 | 0 |

C++→Python 的最大值位于 C++ 顶点 99,505，surface RAS 坐标约 `[29.023, 45.086, 2.920]` mm；它所在的 `>0.1 mm` 连通簇有 18 个顶点，包围盒约 `[21.500, 30.335, 2.920]` 至 `[29.023, 45.086, 13.255]` mm。该顶点只有 3 个一环邻居，三条边长 **15.306–17.725 mm**，邻居自身到 Python 三角面的距离为 0.297–0.680 mm。其三个邻接三角形面积分别为 10.121、11.520、3.949 mm²。C++ 网格共 16 条边长 `>5 mm`，其中 3 条 `>10 mm`；Python 网格没有 `>5 mm` 的边，最长边为 2.993 mm。两网格边长 P99.9 分别为 1.510 和 1.498 mm。C++ 面积最大 11.520 mm²，Python 面积最大 1.986 mm²。

原始 `rh.orig.nofix` 中距这个 C++ 最大误差顶点最近的顶点约 0.759 mm；该观察不足以确定长边在何处形成，也不足以判定哪一种 premesh 与官方最终形状一致。需要沿 `orig.nofix → orig.premesh → orig → white → pial` 查看同一区域，以及与**相同输入**的官方参考比较。

## 质量含义及待验收项

两份 premesh 的现有结构门均通过：闭合、单连通、Euler 数 2，无边界、非流形顶点、面方向冲突或零面积面。该结构门没有检查长边、细长三角形和 premesh 自相交；因此不能因其通过而忽略上述局部突起。premesh 的后续 remesh、相交清理、最终 `orig` 与 white/pial 仍须另行验收，不把这个阶段结果直接算作整例退化或最终指标不等效。

新整例完成后，建议按以下顺序作**只读**质量审查，并保留每个输入文件 SHA、检查程序 SHA、原始计数、局部位置和耗时：

1. 对双侧 `orig.premesh`、`orig`、`white`、`pial` 记录唯一边长度的最大值、P99.9、`>5 mm` 数量，以及三角形面积最大值；`>5 mm` 仅用于追踪本次异常，**不是新的验收阈值**。核对上述坐标附近的长边是否在最终表面消失，并生成局部 T1 叠加图。
2. 用现有 [`mesh_structure`](../../../../src/fnit/recon_all/place_surface_quality.py) 检查闭合、连接性、Euler、边界、非流形和退化面；用同模块的 `proper_pairs(first_vertices=xyz, faces=faces)` 对最终 `orig`、`white`、`pial` 检查严格自相交。生产入口 [`_validate_meshes`](../../../../src/fnit/recon_all/native_free.py) 已对 white/pial 调用 `mark_intersections`，可先复核其原有报告；[`independent_self_intersections`](../../../../tools/recon_evaluation_checks.py) 是现成的只读四表面复查 API。完整 `proper_pairs` 可能较慢，待 GPU 整例完成后执行。
3. 先确认 white/pial 同半球有序面一致，再用 `proper_pairs(first_vertices=pial_xyz, second_vertices=white_xyz, faces=faces)` 检查严格相互穿越。现有 [`audit_recon_white_pial_outputs.py`](../../../../tools/audit_recon_white_pial_outputs.py) 展示了阶段输出的调用方式，但其 CLI 需要专门的候选/参考目录布局；普通整例可直接调用上述函数。只数自相交无法替代跨表面检查。
4. 对同一 T1 的 C++ 与 Python 运行，比较最终表面和双向三角面距离、局部异常、分区 Dice 及厚度/面积/体积指标。仍然禁止在面数或顶点顺序不同的网格上做同索引差值；不得用本次 premesh 平均值单独宣布最终指标等效。

距离计算复用 [`premesh_bidirectional_distance.py`](premesh_bidirectional_distance.py) 和既有 [`_point_to_mesh`](../../accuracy_20261003/task_04/reference_surface_distance.py)；本页的额外极值、一环、连通簇及边长统计仅从已保存距离数组与两份表面做轻量只读汇总，未修改生产表面，也未在进行中的 GPU 整例上重算距离。
