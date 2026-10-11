# 带符号顶点面积：法向/曲率似然接口修复

## 1．功能与边界

修复已成熟的两个似然子项中过严的面积输入限制。固定源码平面几何重新从带符号面面积累加顶点面积；原评分直接保留这些权重。当前实现允许负顶点和负总面积，既有 NumBa 核心按原有序 double 累加后检查精确零分母，没有新增 Python 遍历、绝对值或顶点删除。最终网格质量检查与候选评分分开。

## 2．Python、输入与输出

接口及全部具名输入、类型/空间/单位与示例见[功能页](../../../../docs/recon_all/TOPOLOGY_NORMAL_CURVATURE_LIKELIHOOD.md)。法向使用同几何 origXYZ、normal、area和有序一环；曲率复用已计算k1/k2与固定PDF。返回四个源float32字段与完整逐顶点诊断，不改变几何/PDF/权重。空patch仍0，非空真实有序总面积0时明确ValueError。

## 3．命令行

沿用 `tools/benchmark_recon_topology_normal_likelihood.py`，显式提供 `--capture`、`--native-repeat`、`--observer-receipt`、`--output` 和 `--threads 4`。输出必须是新目录，原始捕获/NPZ不公开；生产算子无独立CLI。

## 4．原软件规则

固定 `d932c45b7941662ea380a05efef580568b98d41a`。`mrisOrientPlane` slow.h 674–682、fast.h 1117–1130按带符号面面积求顶点面积；原两似然直接保留权重。完整文件/行范围SHA见 `source_rules.json`，不发布原头文件。原参考仍是隔离 Conda 源码构建观察器，不进入生产。

## 5．真实回归与时间

公开 `ds000114 sub-07 ses-test LH`，完整初始快照来自绑定79a41cdd的自产链。110,534顶点、108,538选中顶点、663,348邻接项。四字段与参考全部0差异/max0/P990，FNIT逐顶点两次0差异；原两进程完整capture SHA相同。实际快照没有触发负顶点面积，负权重分支由固定源码与合同支持，尚无真实动态负面积patch门。

终版源码 `202c4813…`，10合同通过。热完整API法向 **0.033995s**、曲率 **0.011615s**；含加载/校验/两次API/NPZ/hash **1.730107s**，绘图 **5.279011s**，工具入口墙钟 **11.662781s**。参考组件未单独计时，不能计算原生提速；共享4线程，GPU不用、显存null，完整GA和整体等效未评估。

![修复版真实初始表面似然](reports/aggregate.png)

## 6．版本与来源

- 原 `3ebc601d` / `bfb461f2…` 初始实现记录不重标，旧19payload清单改为按该Git提交绑定。
- signed v1 `8c5dbaed…` 允许负权重但额外Python遍历；完整初始回归0差异，收据保留 `reports/python_guard_v1/`。
- signed v2 `202c4813…` 在原有Numba累加中检查零分母；同快照完整重新回归0差异，不用首版结果冒充终版。
- `PUBLIC_EXPORT_MANIFEST` 保留每个原始JSON SHA、公开SHA与数值/布尔/null叶级保真；`PUBLICATION_MANIFEST`绑定当前独占源码和全部公开载荷。

## 7．参考与许可

[固定指标源码](https://github.com/freesurfer/freesurfer/blob/d932c45b7941662ea380a05efef580568b98d41a/utils/mrisurf_defect.cpp#L1502-L1623)、[slow metricProperties](https://github.com/freesurfer/freesurfer/blob/d932c45b7941662ea380a05efef580568b98d41a/utils/mrisurf_metricProperties_slow.h)、[fast metricProperties](https://github.com/freesurfer/freesurfer/blob/d932c45b7941662ea380a05efef580568b98d41a/utils/mrisurf_metricProperties_fast.h)。文献和来源许可见功能页与仓库通知。载荷没有原MRI、NPZ、原程序、许可证内容或原软件源码。
