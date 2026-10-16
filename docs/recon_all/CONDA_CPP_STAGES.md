# 标准单 T1 流程的阶段索引

[总入口与输出结构](README.md) · [Conda 安装](CONDA_CPP_BUILD.md) · [固定 138 项清单](../../src/fnit/recon_all/expected_outputs.py)

表中的 `H` 表示 `lh` 或 `rh`；`subject` 是单个被试目录。官方命令用于说明算法对应关系；FNIT 运行时只调用 Conda 中从固定源码构建的程序和仓库自己的 Python 实现。官方内部步骤没有独立命令时，以所列主命令为准。命令列说明算法对应关系，不代表该步骤在 CUDA 配方中一定调用原生程序；具体后端由参数及实际运行报告确定。当前 `93affde8` 原生表面测速配方的完整参数和自有/原生分工见[显式 Conda 表面配方](NATIVE_TOPOLOGY_FINISH.md)。

| 阶段 | FNIT 输入 → 输出 | 对应官方程序或命令 | 详细说明及真实数据证据 |
| --- | --- | --- | --- |
| N4 校正 | `mri/orig.mgz` → `mri/tmp/nu0.mgz` | `mri_nu_correct.mni` 所用 N4 校正 | 本轮使用 SimpleITK Python API/CPU；[完整 Python N4](N4_SIMPLEITK.md)、[Conda N4](N4_ITK_CONDA.md) |
| MNI152 非线性变换 | 裁剪 `orig.mgz`、`aff.lta`、固定模板和 deform 权重 → 前向/逆向 warp、`test.nii.gz` | `mri_synthmorph -m deform`、`mri_warp_convert`、`mri_ca_register -invert-and-save`、`mri_convert -at` | [输入输出、参数及真实 T1 对照](MNI_NONLINEAR_CHAIN.md) |
| GCA 注册、WM 分割与编辑 | conform T1、脑掩膜、GCA/aseg → LTA、`wm.seg.mgz`、`wm.asegedit.mgz` | `mri_em_register -uns 3 -mask brainmask.mgz nu.mgz RB_all_2020-01-02.gca transforms/talairach.lta`；`mri_segment`；`mri_edit_wm_with_aseg` | 本轮 GCA 为 Torch 评分+Python EM，WM 为自有优化分割和 Python/Numba 编辑；[原生阶段历史同输入结果](../../validation/recon_all/python_gpu_port/native_cpp_conda_20260927/wm_chain_20260927/REPORT.md) |
| 拓扑修复 | `surf/H.orig.nofix`、`H.inflated.nofix`、`H.qsphere.nofix`、`mri/brain.mgz`、`wm.mgz` → `H.orig.premesh`；默认 Python remesh/清理，显式 conda-cpp 则 preclean→原生 remesh3→postclean→结构与双谓词门→`H.orig` | `mris_fix_topology -ga -seed 1234 -threads 1 -mgz -sphere qsphere.nofix -inflated inflated.nofix -orig orig.nofix -out orig.premesh subject H` | [拓扑](TOPOLOGY_CONDA_GA.md)、[显式原生收尾](NATIVE_TOPOLOGY_FINISH.md) |
| 非零自相交修复 | Python 路径修复 `H.orig`；conda-cpp 路径清理私有 premesh/remesh 候选，完整门通过后发布 | `mris_remove_intersection INPUT OUTPUT` | [自有修复](INTERSECTION_REPAIR.md)、[原生候选及发布门](NATIVE_TOPOLOGY_FINISH.md)；真实同输入与整例证据按版本区分 |
| 缺陷体积映射 | `H.orig.nofix`、`H.defect_labels`、皮层标签、`orig.mgz` → `mri/surface.defects.mgz` | `mri_label2vol --defects ...` | 本轮显式 `defects_backend=torch`，保留 LH→RH 共享累积；[输入输出与真实 T1 配对](DEFECTS_VOLUME.md) |
| 预白质与最终白质表面 | `H.orig`、MRI、皮层标签/注释 → `H.white.preaparc`、`H.smoothwm`、`H.white` | `mris_place_surface --white ...` | 本轮使用独立 Conda C++ 放置；smoothwm 三轮平滑仍为自有 CPU。[预白质](WHITE_PREAPARC_CONDA_CHAIN.md)、[最终 white](FINAL_WHITE_CONDA.md) |
| nofix/标准 inflation | `H.smoothwm.nofix` → `H.inflated.nofix`；`H.smoothwm` → `H.inflated`、`H.sulc` | `mris_inflate -no-save-sulc ...`；`mris_inflate ...` | 本轮 nofix 使用 Conda C++，标准 inflation 使用完整自有 Torch；[标准 GPU inflation](INFLATE_TORCH_20261009.md) |
| quick/standard sphere、配准 | `H.inflated.nofix`、`H.inflated`、`H.smoothwm`、folding atlas → `H.qsphere.nofix`、`H.sphere`、`H.sphere.reg` | `mris_sphere -q ...`、`mris_sphere ...`、`mris_register ...` | 本轮为自有 Python/Numba + GPU averaging，球面 dense 收尾使用 Torch；[标准球面](../../validation/recon_all/python_gpu_port/SPHERE_STANDARD_STATUS.md)、[配准](../../validation/recon_all/python_gpu_port/MRIS_REGISTER_STATUS.md) |
| pial | `H.white`、`brain.finalsurfs.mgz`、MRI/标签 → `H.pial.T1`、`H.pial` | `mris_place_surface --pial ...` | 本轮使用标准 Conda C++ 程序；[标准四轮 pial](NATIVE_PIAL_PLACEMENT.md)、[Python 单阶段 pial](PYTHON_PIAL_PLACEMENT.md) |
| 主要顶点图 | 同序最终 white/pial → thickness、area/area.pial、curv/curv.pial、area.mid、volume | `mris_place_surface --thickness/--area-map/--curv-map ...` 等对应步骤 | 本轮 CUDA 上仍用自有 PyTorch，不随 native placement 切换；[CUDA 厚度、面积和曲率](SURFACE_METRICS.md) |
| 图谱曲率与统计 | `H.sphere.reg`、folding atlas、`H.smoothwm`、`H.curv`、`H.sulc` → `H.avg_curv`、`H.smoothwm.*.crv`、`stats/H.curv.stats` | `mrisp_paint -a 5 folding-atlas.tif#6 H.sphere.reg H.avg_curv`；`mris_curvature_stats -m --writeCurvatureFiles -G -o stats/H.curv.stats -F smoothwm subject H curv sulc` | 本轮 avg_curv、curv.stats 及 smoothwm 八张曲率图使用 Conda C++；[同输入结果](CURVATURE_OUTPUTS.md)、[历史两例与三方比较](../../validation/recon_all/optimizations/20261002_parallel/FINAL_RESULTS.md) |
| Jacobian、灰白对比与 SNR | `H.white.preaparc`、`H.sphere.reg`、`rawavg.mgz`、最终 white/厚度/注释 → `H.jacobian_white`、`H.w-g.pct.mgh`、`stats/H.w-g.pct.stats` | `mris_jacobian`；`pctsurfcon` 中的 `mri_vol2surf`/`mri_concat`；`mri_segstats --snr` | [输入输出与双侧真实 T1 同输入结果](SURFACE_EXTRA_METRICS.md) |
| 脑区、ribbon 与统计 | 已放置表面、aseg、注释图谱 → `label/H.*.annot`、`mri/ribbon.mgz`、`mri/aparc+aseg.mgz`、`stats/*.stats` | `mris_ca_label`、`mri_surf2volseg`、`mri_segstats` 的对应阶段 | [后处理同输入结果](../../validation/recon_all/python_gpu_port/ASEG_RIBBON_RUNNER_20260928.md)、[皮层体积映射](../../validation/recon_all/python_gpu_port/ATLAS_VOLUME_RUNNER_20260928.md)、[BA/VPnl 接线实测](../../validation/recon_all/python_gpu_port/exvivo_wiring_20260929.json) |

阶段证据分三级：**同输入**把官方冻结上游交给两种实现，验证单个算子；**连续链**使用 FNIT 自产上游，检查误差传播；**整例**从真实原始 T1 开始检查全部 138 个文件、顶点指标、脑区值、耗时及资源。表中的历史同输入结果只证明各自阶段，不能替代现版整例。

每个函数的精确参数、坐标空间、返回字典和具名参数示例以所链接的阶段文档及源码docstring为准。非零相交输入已修复到零，但未取得官方同输入输出。历史源码8d750e2已从两例原始T1连续生成138项输出，生产基本网格检查通过；严格数值、white/pial穿越与sphere/reg局部质量和脑区指标单列在[该版结果](../../validation/recon_all/optimizations/20261002_parallel/FINAL_RESULTS.md)。输出存在和执行完成不代表整体指标等效已通过。本轮原生收尾配方尚未完成原始 T1 整例验收，不能据历史结果宣布 2,000 秒目标达成。当前生产基本网格门覆盖同序面、有限坐标、Euler/闭合和 white/pial 自相交；sphere/sphere.reg 局部翻折、white↔pial 穿越及脑区误差由事后评估另报，不能写成已由该门验收。

## 参考文献与原实现

- Fischl B. FreeSurfer. *NeuroImage*. 2012;62(2):774–781. [doi:10.1016/j.neuroimage.2012.01.021](https://doi.org/10.1016/j.neuroimage.2012.01.021)。
- [FreeSurfer 官方 recon-all 说明](https://www.freesurfer.net/fswiki/recon-all)。
- [FreeSurfer 固定源码提交](https://github.com/freesurfer/freesurfer/tree/d932c45b7941662ea380a05efef580568b98d41a)。
