# 完整首缺陷 MRI 边评分验证

## 1．验证范围

公开 ds000114 sub-07 LH，79a41cdd 原始 T1 连续自产 nofix、qsphere 和 brain。同输入阶段回归，不是原始 T1 整例，也不是完整 GA。函数、具名示例、全部输入/输出、坐标及失败行为见[中文专页](../../../../docs/recon_all/TOPOLOGY_MRI_SCORE.md)。

## 2．Python 输入输出

`source_topology_ras_to_voxel(volume_header=...)` 使用 MRI-only tkregister inverse；`prepare_topology_edge_inputs(...)` 独立准备全网格强度和法向；`source_mri_edge_scores(...)` 计算完整候选边的 22 点评分。候选保存完成后才读取独立参考，参考不参与生产算法。完整原始 MRI、压缩 checkpoint 和观察二进制留在已授权计算目录，不进入 Git。

## 3．复现脚本

`tools/benchmark_recon_topology_mri_edges.py` 生成完整首缺陷与边表；本目录三个活跃观察 builder 独立源码编译参考；`compare_samples.py` / `compare_preparation.py` 分离采样、坐标与灰白准备差异。`recover_capture_receipt.py` 修复已完成观察的事后收据，保留原驱动失败，不改变原数据。全部参数以具名 CLI 传入。

`export_reports.py` 复用已有 JSON 发布器，额外按调用者提供的私有前缀脱敏；完整根先替换，避免用户名先替换破坏路径匹配。公开收据逐个校验 SHA 和数值/布尔/null 的顺序及类型。原收据 SHA 在 [PUBLIC_EXPORT_MANIFEST.json](reports/PUBLIC_EXPORT_MANIFEST.json) 中保留，26 份 JSON 收据全量公开。

## 4．参考实现

固定 FreeSurfer d932c45 源码、已有 FNIT 三处数值适配、独立 Conda 编译程序。不是系统安装官方程序。两次独立参考固定同主机、CPU affinity 8–11、4 线程和 seed 1234；观察器在首次完整边表排序后停止，不运行遗传控制。编译、链接、源文件和程序实际 SHA 完整保留，不拷贝官方结果充当候选。

## 5．当前精度和耗时

- [v3 完整候选](reports/v3/candidate.json)：4,723 评分、used 标记与排序端点全一致；评分 max/P99=0，重复 0 差异。
- [完整采样回放](reports/v3_diagnostics/samples.public.json)：4,851 边×22=106,722 次坐标和插值全一致。
- [全准备字段](reports/v3_diagnostics/preparation.public.json)：110,534 顶点的平均原坐标、原法向、白灰中值及实际矩阵均数值一致；221,116 有序面一致。后续原程序清标与初始缺陷 label 的 1,996 标记差单独记录，不参与评分。
- 准备含 IO/JIT 51.023 s；完整边评分首次 0.330 s，热调用 0.00688 s；候选包含 checkpoint/比较的 harness 74.941 s。参考范围不同，不能宣称整步或整例提速。

完整初始失败重连的 [T1 局部图](../20261010_topology_retessellation/reports/figures/v5_intersecting_patch_T1.png) 仍显示 17 个相交面；没有以新评分的阶段通过替代几何 QC。另一侧、第二例、全部 fitness/GA 和整体等效尚未验收。

## 6．最近版本与失败记录

v1/v2 都有 4,467/4,723 个评分不同且排序不同。v2 正确化 `sqrt(float)` 未消除误差。准备观察先排除坐标与法向，再发现表面放置矩阵与 topology 的 MRI-only 变换语义不同。v3 纠正变换并保留实际类型规则后全部采样与排序一致。

未激活的静态 cache 观察失败保存在 [inactive_branch_failed.json](reports/inactive_branch_failed.json)，不能归因于实际缓存。实际 `MATRIX_ALLOCATION` 观察两次均程序退出 0，全部采样/准备/评分字节与旧参考相同；原驱动事后 SHA 类型错误退出 1，独立[恢复收据](reports/v3_diagnostics/capture_recovery/capture_recovery.public.json)保留此边界。公开 JSON 以 `.public` 命名，服务器侧已脱敏并校验；原导出文件名和真实原 SHA 留在清单中。

## 7．来源与后续缺口

[固定拓扑源码](https://github.com/freesurfer/freesurfer/blob/d932c45b7941662ea380a05efef580568b98d41a/utils/mrisurf_defect.cpp)及参考文献见专页。下一步是真实 MRI face/vertex/unMRI、法向与曲率 fitness、USE_SOME 分类、mutation/crossover、种群控制和全缺陷接受。当前 `summary.json` 明确 `complete_ga_solver=false` 和 `overall_metric_equivalence=not_assessed`。
