# 标准单 T1 流程的阶段索引

[总入口与输出结构](README.md) · [Conda 安装](CONDA_CPP_BUILD.md) · [固定 138 项清单](../../src/fnit/recon_all/expected_outputs.py)

表中的 `H` 表示 `lh` 或 `rh`；`subject` 是单个被试目录。官方命令用于说明算法对应关系；FNIT 运行时只调用 Conda 中从固定源码构建的程序和仓库自己的 Python 实现。官方内部步骤没有独立命令时，以所列主命令为准。

| 阶段 | FNIT 输入 → 输出 | 对应官方程序或命令 | 详细说明及真实数据证据 |
| --- | --- | --- | --- |
| N4 校正 | `mri/orig.mgz` → `mri/tmp/nu0.mgz` | `mri_nu_correct.mni` 所用 N4 校正 | [N4](N4_ITK_CONDA.md) |
| GCA 注册、WM 分割与编辑 | conform T1、脑掩膜、GCA/aseg → LTA、`wm.seg.mgz`、`wm.asegedit.mgz` | `mri_em_register -uns 3 -mask brainmask.mgz nu.mgz RB_all_2020-01-02.gca transforms/talairach.lta`；`mri_segment`；`mri_edit_wm_with_aseg` | [原生阶段历史同输入结果](../../validation/recon_all/python_gpu_port/native_cpp_conda_20260927/wm_chain_20260927/REPORT.md) |
| 拓扑修复 | `surf/H.orig.nofix`、`H.inflated.nofix`、`H.qsphere.nofix`、`mri/brain.mgz`、`wm.mgz` → `H.orig.premesh`，再由 FNIT remesh 写 `H.orig` | `mris_fix_topology -ga -seed 1234 -threads 1 -mgz -sphere qsphere.nofix -inflated inflated.nofix -orig orig.nofix -out orig.premesh subject H` | [拓扑](TOPOLOGY_CONDA_GA.md) |
| 非零自相交修复 | `surf/H.orig` → 同路径修复表面 | `mris_remove_intersection H.orig H.orig` | [输入输出与当前证据](INTERSECTION_REPAIR.md)；非零真实输入仍需单独配对验收 |
| 缺陷体积映射 | `H.orig.nofix`、`H.defect_labels`、皮层标签、`orig.mgz` → `mri/surface.defects.mgz` | `mri_label2vol --defects ...` | [输入输出与真实 T1 配对](DEFECTS_VOLUME.md) |
| 预白质与最终白质表面 | `H.orig`、MRI、皮层标签/注释 → `H.white.preaparc`、`H.smoothwm`、`H.white` | `mris_place_surface --white ...` | [预白质](WHITE_PREAPARC_CONDA_CHAIN.md)、[最终 white](FINAL_WHITE_CONDA.md) |
| quick/standard sphere、配准 | `H.inflated.nofix`、`H.inflated`、`H.smoothwm`、folding atlas → `H.qsphere.nofix`、`H.sphere`、`H.sphere.reg` | `mris_sphere -q ...`、`mris_sphere ...`、`mris_register ...` | [标准球面](../../validation/recon_all/python_gpu_port/SPHERE_STANDARD_STATUS.md)、[配准](../../validation/recon_all/python_gpu_port/MRIS_REGISTER_STATUS.md) |
| pial 与主要顶点图 | `H.white`、`brain.finalsurfs.mgz`、MRI/标签 → `H.pial.T1`、`H.pial`、厚度、面积、体积、曲率 | `mris_place_surface --pial ...`、`mris_place_surface --thickness/--area-map/--curv-map ...` | [标准四轮 pial](NATIVE_PIAL_PLACEMENT.md)、[Python 单阶段 pial](PYTHON_PIAL_PLACEMENT.md)、[面积体积](../../validation/recon_all/python_gpu_port/RELEASE_GATES.md) |
| 图谱曲率与统计 | `H.sphere.reg`、folding atlas、`H.smoothwm`、`H.curv`、`H.sulc` → `H.avg_curv`、`H.smoothwm.*.crv`、`stats/H.curv.stats` | `mrisp_paint -a 5 folding-atlas.tif#6 H.sphere.reg H.avg_curv`；`mris_curvature_stats -m --writeCurvatureFiles -G -o stats/H.curv.stats -F smoothwm subject H curv sulc` | [输入输出与真实 T1 同输入结果](CURVATURE_OUTPUTS.md)；现版整例待验证 |
| Jacobian、灰白对比与 SNR | `H.white.preaparc`、`H.sphere.reg`、`rawavg.mgz`、最终 white/厚度/注释 → `H.jacobian_white`、`H.w-g.pct.mgh`、`stats/H.w-g.pct.stats` | `mris_jacobian`；`pctsurfcon` 中的 `mri_vol2surf`/`mri_concat`；`mri_segstats --snr` | [输入输出与双侧真实 T1 同输入结果](SURFACE_EXTRA_METRICS.md) |
| 脑区、ribbon 与统计 | 已放置表面、aseg、注释图谱 → `label/H.*.annot`、`mri/ribbon.mgz`、`mri/aparc+aseg.mgz`、`stats/*.stats` | `mris_ca_label`、`mri_surf2volseg`、`mri_segstats` 的对应阶段 | [后处理同输入结果](../../validation/recon_all/python_gpu_port/ASEG_RIBBON_RUNNER_20260928.md)、[皮层体积映射](../../validation/recon_all/python_gpu_port/ATLAS_VOLUME_RUNNER_20260928.md)、[BA/VPnl 接线实测](../../validation/recon_all/python_gpu_port/exvivo_wiring_20260929.json) |

阶段证据分三级：**同输入**把官方冻结上游交给两种实现，验证单个算子；**连续链**使用 FNIT 自产上游，检查误差传播；**整例**从真实原始 T1 开始检查全部 138 个文件、顶点指标、脑区值、耗时及资源。表中的历史同输入结果只证明各自阶段，不能替代现版整例。

每个函数的精确参数、坐标空间、返回字典和具名参数示例以所链接的阶段文档及源码 docstring 为准。`mris_remove_intersection` 的非零相交输入和图谱曲率的 FNIT 自产上游整例尚未验收。

## 参考文献与原实现

- Fischl B. FreeSurfer. *NeuroImage*. 2012;62(2):774–781. [doi:10.1016/j.neuroimage.2012.01.021](https://doi.org/10.1016/j.neuroimage.2012.01.021)。
- [FreeSurfer 官方 recon-all 说明](https://www.freesurfer.net/fswiki/recon-all)。
- [FreeSurfer 固定源码提交](https://github.com/freesurfer/freesurfer/tree/d932c45b7941662ea380a05efef580568b98d41a)。
