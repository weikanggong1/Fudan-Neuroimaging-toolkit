# recon-all 第二轮候选后端的接线

[重建入口](README.md) · [持久半球调度](PERSISTENT_HEMISPHERE_PIPELINE.md) · [GPU 距离阶段](TOPOLOGY_GPU_DISTANCE_FITNESS.md)

## 1. 功能与流程

`backend="python"` 在完整单 T1w 流程中允许显式选择 GCA 图谱上下文复用、拓扑距离计算、拓扑状态装配和双半球进程生命周期。四项只选择已有完整算法的执行方式；默认仍为 `False`、`serial`、`legacy`、`groups`。生产输入只能是原始 T1w、FNIT 自产中间结果及已声明资源。原版程序只用于隔离对照。

```mermaid
flowchart LR
  A[原始 T1w 与校验资源] --> B[GCA 注册与归一化]
  B --> C[自产 filled 与双侧原始网格]
  C --> D[完整拓扑 GA 距离评分与状态装配]
  D --> E[white 球面配准 pial]
  E --> F[138 项输出及质量检查]
```

## 2. Python 调用、参数与输出

```python
from fnit.recon_all.native_free import run_recon_all_python

reconstruction_report = run_recon_all_python(
    t1="/data/public_sub-06_T1w.nii.gz",  # 原始单幅三维 T1w NIfTI
    subject_dir="/data/new_sub-06",  # 不存在或为空的被试输出目录
    weights_dir="/data/verified_weights",  # 已校验 SHA-256 的模型权重
    assets_dir="/data/verified_assets",  # 已校验 SHA-256 的模板和图谱
    backend="python",  # 自有 Python/Numba/PyTorch 连续候选
    device="cuda:0",  # 当前进程可见 GPU 的明确逻辑编号
    threads=4,  # 整例 CPU 总预算；双侧各最多两个线程
    topology_backend="python-ga",  # 完整自有有序 GA，允许选择下面两个后端
    topology_distance_backend="gpu_defect_signed_distance",  # 当前候选的完整 GPU 距离场
    topology_state_backend="packed",  # 每次候选重新装配当前完整状态
    gca_reuse_atlas_context=True,  # 只在本次 Torch GCA 调用内复用不变图谱数据
    hemisphere_execution="persistent",  # 每侧一次私有拷贝和一个连续 worker
)
```

| 本轮参数 | 默认值与可选值 | 输入约束及作用 |
| --- | --- | --- |
| `gca_reuse_atlas_context` | `False` / `True` | 布尔值；`True` 只用于显式 `cuda:N` 上的 Torch GCA，同次调用复用不变图谱分类器，不复用不同被试的动态状态。 |
| `topology_distance_backend` | `serial` / `parallel` / `gpu_defect_signed_distance` | 仅 `topology_backend="python-ga"` 可用非默认值；前两者为同公式 CPU 执行，GPU 选项要求显式 `cuda:N`，每个候选重新计算完整距离场。距离单位 mm。 |
| `topology_state_backend` | `legacy` / `packed` | 非默认值仅用于完整 `python-ga`；`packed` 每个候选重新打包、验证有序面和邻接，不缓存变化几何。表面坐标为 surface RAS，单位 mm。 |
| `hemisphere_execution` | `groups` / `persistent` | `persistent` 仅支持 `backend="python"` 与两个独立半球 worker；公共入口在 Python 配方内固定两个 worker。总线程至少 2。 |

`t1` 为含有效 affine 的三维影像；入口生成 1 mm conformed 体积网格。`filled` 等 MRI 保持该网格，表面是有序三角面、surface RAS/mm，顶点指标使用对应表面的顶点次序。权重、资产的文件名、大小和 SHA-256 由安装清单核验。完整公开参数、标准输出路径和返回字典见[重建入口](README.md)。本轮四项在报告中保留请求和实际后端；成功返回含 `output_validation`、`mesh_validation`、`hemisphere_scheduling`、各阶段耗时和路径的字典，并写 `fnit-native-free-run.json`。非法设备、后端组合、缺资源或网格质量失败会抛异常；失败报告保留实际完成的阶段，不补写标准文件。

## 3. 命令行

```bash
fnit-recon-all /data/public_sub-06_T1w.nii.gz /data/new_sub-06 \
  --weights-dir /data/verified_weights --assets-dir /data/verified_assets \
  --backend python --device cuda:0 --threads 4 \
  --topology-backend python-ga \
  --topology-distance-backend gpu_defect_signed_distance \
  --topology-state-backend packed --gca-reuse-atlas-context \
  --hemisphere-execution persistent
```

路径均为调用方已授权的本地路径；输出目录须新建。这里没有单独的“第二轮”CLI：这些参数进入同一个 `fnit-recon-all`。安装使用主页 [`environment.yml`](../../environment.yml)；不要求原版阶段程序。

## 4. 原软件对应步骤

GCA 对应独立参考中的 `mri_em_register`，拓扑 GA 对应 `mris_fix_topology -mgz -ga`，皮层整链对应 `recon-all -all`。图谱上下文复用、距离后端和状态打包均为这些命令内部的实现选择，**没有独立的原软件 CLI 参数**。参考程序与产物不能送入生产候选。

## 5. 真实验证、时间和版本边界

现有[GPU 距离同输入实验](TOPOLOGY_GPU_DISTANCE_FITNESS.md)在一例真实冻结缺陷上完成完整首缺陷配对：CPU/GPU 均有 495 次搜索评分及 1 次最终评分，完整 CLI 为 1275.139/1225.616 秒，最终状态逐字节一致。这只验证该阶段当时的冻结源码，不是四项组合或当前整例提速。状态打包、GCA、半球调度各有单独阶段记录；本页不把其局部收益相加。

[旧 f016 两例整例报告](../../validation/recon_all/optimizations/20261014_gpucw1_f016_partial/README.md)记录 sub-07 完成且 138/138 输出、23/138 严格逐文件通过，sub-06 在拓扑质量门失败。它不是随后修复的整例结果，也没有证明本页四项组合通过；该版本的最终脑区误差、网格局部异常、历史官方时间及受外部进程污染的显存样本均在报告中单列。当前组合的两例空目录整例、同条件速度、进程树峰值和整体指标等效均待评估，不能用阶段测试代替。公开报告未附私有 T1 脑图；真实第四缺陷的[距离差图](../../validation/recon_all/optimizations/20261011_fnit4_gpu_fitness/fourth_real_roi.png)只对应旧阶段输入。

## 6. 更新与参考

- 2026-10-11：完整 GPU 距离阶段的同输入门和后端选择已分别记录；默认选择未因此改变。
- 2026-10-14：冻结 f016 两例运行仅一例完成，拓扑质量和整例等效问题保留。
- 本次文档补齐现有接口约束；未新增科学运行或重标旧 benchmark。

实现见 [`python_profile.py`](../../src/fnit/recon_all/python_profile.py)、[`native_free.py`](../../src/fnit/recon_all/native_free.py)、[`hemisphere_pipeline.py`](../../src/fnit/recon_all/hemisphere_pipeline.py)。原算法见固定[FreeSurfer 源码](https://github.com/freesurfer/freesurfer/tree/d932c45b7941662ea380a05efef580568b98d41a)及 Fischl, Liu & Dale, *IEEE TMI* 20:70–80 (2001), [DOI:10.1109/42.906426](https://doi.org/10.1109/42.906426)；GPU 核采用 [Triton](https://github.com/triton-lang/triton)。
