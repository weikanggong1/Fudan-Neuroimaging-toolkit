# Conda C++ 阶段：功能、调用与验收

本页记录从 FreeSurfer 8.2 源码提交 `d932c45b7941662ea380a05efef580568b98d41a` 编译的八个 CPU 程序。以下配对均指**冻结同输入**，不能替代同一 T1 的端到端验证。构建方法见[Conda 编译说明](CONDA_CPP_BUILD.md)。默认网络和部分 Python 算子可在 GPU 运行，但整例并非全 GPU。

## 调用

```python
from fnit.recon_all.native_free import run_recon_all_python

report = run_recon_all_python(
    "subject_T1w.nii.gz", "/scratch/subjects/sub01",
    "/path/to/weights", "/path/to/assets",
    device="cuda:0", threads=4,
    native_bin_dir="/path/to/recon-cpp-build/bin",
    native_topology=True, native_sphere=True,
    native_surface_metrics=True, native_registration=True,
)
```

等价单被试 CLI：

```bash
export FS_LICENSE=/path/to/private/license.txt
fnit-recon-all subject_T1w.nii.gz /scratch/subjects/sub01 \
  --weights-dir /path/to/weights --assets-dir /path/to/assets \
  --device cuda:0 --threads 4 \
  --native-bin-dir /path/to/recon-cpp-build/bin \
  --native-topology --native-sphere --native-surface-metrics --native-registration
```

多被试只提供 `fnit.recon_all.batch.run_recon_all_python_batch` Python API。`native_bin_dir` 必填并启用前三个程序；表面选项启用其余五个。每次运行记录程序路径、SHA-256 和 `stages[].seconds`，每半球表面细分耗时在 `surfaces[hemi]`。

下表为当前 Python runner 的功能和对应原生命令。`S=sub01`、`H=lh`（右侧用 `rh`），`SURF=$SUBJECTS_DIR/$S/surf`，`ATLAS=/path/to/assets/average`。命令中的可执行文件指官方 FreeSurfer 8.2；实际 runner 调用 `native_bin_dir` 的 Conda 编译版。体积命令在被试 `mri` 目录运行，表面命令在 `scripts` 目录运行，并设置外置 `FREESURFER_HOME`、`FS_LICENSE`、`SUBJECTS_DIR`。

| 程序与开关 | 功能及输出 | 官方等价命令 |
|---|---|---|
| `mri_em_register`；必需 | `nu` 与 GCA 配准，写 `transforms/talairach.lta` | `mri_em_register -uns 3 -mask brainmask.mgz nu.mgz "$ATLAS/RB_all_2020-01-02.gca" transforms/talairach.lta` |
| `mri_segment`；必需 | 从去噪脑图生成 `wm.seg.mgz` | `mri_segment -wsizemm 13 -mprage antsdn.brain.mgz wm.seg.mgz` |
| `mri_edit_wm_with_aseg`；必需 | 加入 EntoWM、ACJ 和皮层下约束，写 `wm.asegedit.mgz` | `mri_edit_wm_with_aseg -keep-in -fix-ento-wm entowm.mgz 3 255 255 -fix-acj aseg.presurf.mgz 255 255 -fill-seg-wm -fix-scm-ha 1 wm.seg.mgz brain.mgz aseg.presurf.mgz wm.asegedit.mgz` |
| `mris_fix_topology`；`native_topology=True` | 根据 `orig.nofix`、`inflated.nofix`、`qsphere.nofix` 生成 `$H.orig` | 当前 runner 与官方整例均调用：`mris_fix_topology -ga -seed 1234 -threads 1 -mgz -sphere qsphere.nofix -inflated inflated.nofix -orig orig.nofix -out orig "$S" "$H"`。Conda 版同输入仍未通过逐点验收。 |
| `mris_inflate`；`native_sphere=True` | 生成 `inflated.nofix`，再生成 `inflated` 和 `sulc` | `mris_inflate -no-save-sulc "$SURF/$H.smoothwm.nofix" "$SURF/$H.inflated.nofix"`；`mris_inflate "$SURF/$H.smoothwm" "$SURF/$H.inflated"` |
| `mris_sphere`；`native_sphere=True` | 从 `inflated` 生成正式 `sphere`；快速球面已改为 Python | `mris_sphere -threads 4 -seed 1234 "$SURF/$H.inflated" "$SURF/$H.sphere"` |
| `mris_place_surface`；`native_surface_metrics=True` | 在已有 white/pial 上计算厚度、面积和曲率五张顶点图；不生成几何、`area.mid` 或顶点体积 | 五条命令见下方 |
| `mris_register`；`native_registration=True` | 用 folding atlas 配准 `sphere.reg` | `mris_register -curv -threads 4 "$SURF/$H.sphere" "$ATLAS/$H.folding.atlas.acfb40.noaparc.i12.2016-08-02.tif" "$SURF/$H.sphere.reg"` |

### 八个程序的输入与输出格式

| 程序 | 主要输入 | 输出结构 |
|---|---|---|
| `mri_em_register` | `nu.mgz`、`brainmask.mgz`、`RB_all_2020-01-02.gca` | `mri/transforms/talairach.lta`；4×4 仿射与体积几何元信息。 |
| `mri_segment` | `mri/antsdn.brain.mgz` | `mri/wm.seg.mgz`；与输入同空间维度的 uint8 白质候选体积；本次验证输入为 256³。 |
| `mri_edit_wm_with_aseg` | `wm.seg.mgz`、`brain.mgz`、`aseg.presurf.mgz`、`entowm.mgz` | `mri/wm.asegedit.mgz`；与输入同空间维度的 uint8 编辑后白质体积；本次验证输入为 256³。 |
| `mris_fix_topology` | `surf/H.orig.nofix`、`H.inflated.nofix`、`H.qsphere.nofix` 及 `mri/filled.mgz` | `surf/H.orig`；按顶点编号排序的三角网格，附 MGH volume geometry。 |
| `mris_inflate` | `surf/H.smoothwm.nofix` 或 `surf/H.smoothwm` | 对应 `surf/H.inflated.nofix` 或 `surf/H.inflated`；第二次另写 `surf/H.sulc`，每顶点一个 float32 值。 |
| `mris_sphere` | `surf/H.inflated` | `surf/H.sphere`；有序球面三角网格。快速 `qsphere.nofix` 由同目录 Python `write_quick_sphere` 产生。 |
| `mris_place_surface` | 有序 `surf/H.white` 和 `surf/H.pial` 网格 | `surf/H.thickness`、`area`、`area.pial`、`curv`、`curv.pial`；每文件按顶点索引排列 float32 值。当前只调用五张指标图模式。 |
| `mris_register` | `surf/H.sphere`、`surf/H.curv`、`average/H.folding.atlas...tif` | `surf/H.sphere.reg`；与输入相同面序的有序注册球面网格。 |

`H` 表示 `lh` 或 `rh`；顶点文件的可比性要求两侧候选与官方先有相同顶点数量和顺序。上述输出保留各自实际头部与元信息，不能只比较数值数组。Python `write_quick_sphere(input_inflated_nofix, output_qsphere_nofix) -> None` 读取有序三角网格，写同面序球面网格；其官方等价命令为 `mris_sphere -q -p 6 -a 128 -seed 1234 ...`。[v3 真实 T1 双侧同输入审计](../../validation/recon_all/python_gpu_port/native_cpp_conda_20260927/v3_e2e_20260927/quick_sphere_lh.json)、[RH](../../validation/recon_all/python_gpu_port/native_cpp_conda_20260927/v3_e2e_20260927/quick_sphere_rh.json)记录输入、输出、顶点数、面数、误差和耗时。

`mris_place_surface` 的五条命令：

```bash
mris_place_surface --thickness "$SURF/$H.white" "$SURF/$H.pial" 20 5 "$SURF/$H.thickness"
mris_place_surface --area-map "$SURF/$H.white" "$SURF/$H.area"
mris_place_surface --area-map "$SURF/$H.pial" "$SURF/$H.area.pial"
mris_place_surface --curv-map "$SURF/$H.white" 2 10 "$SURF/$H.curv"
mris_place_surface --curv-map "$SURF/$H.pial" 2 10 "$SURF/$H.curv.pial"
```

## 冻结同输入精度与时间

时间均为单次运行，来自不同节点/日期和共享负载，不作为稳定加速比。

| 阶段 | 与官方配对结果 | 官方 / 当前耗时 |
|---|---|---:|
| `mri_em_register` | LTA 4×4 的 16 个 float64 元素逐位相同；文件创建时间不同 | 369.43 / 299.21 s |
| `mri_segment` | `wm.seg` 0/16,777,216 体素差，MGH 头部和仿射一致 | 51.85 / 40.45 s；连续 WM 测试当前 81.63 s |
| `mri_edit_wm_with_aseg` | `wm.asegedit` 0/16,777,216 体素差 | 历史官方日志 28.42 / 独立当前 45.99 s；连续测试当前 52.88 s |
| `mris_fix_topology`（**不带** `-ga`） | LH 117,777、RH 119,990 顶点的有序坐标及面索引均相同 | LH 43.2 / 53.3 s；RH 43.9 / 45.7 s |
| `mris_fix_topology -ga -seed 1234`（**官方整例模式**） | 准确上游输入上的 Conda 版 LH `orig.premesh` 为 101,737/203,470 顶点/面，官方为 101,689/203,374；RH 为 100,655/201,306，官方为 100,555/201,106。官方二进制在同路径同输入重跑能复现官方历史网格。首差位于首个缺陷的 GA 候选评分。**未通过**。 | 尚未形成可比时间 |
| `mris_inflate` | LH 117,777 顶点坐标、235,550 面及 117,777 个 sulc 值逐元素相同 | 12.21 / 12.19 s |
| Python `write_quick_sphere`（代替 Conda `mris_sphere -q`） | v3 真实 T1 的双侧同一 `inflated.nofix` 上，LH 102,764/102,764 顶点、205,560/205,560 面及 RH 101,454/101,454 顶点、202,936/202,936 面均 exact；旧 Conda LH `-q` 平均顶点差 0.2402 mm。 | Python LH/RH 139.24/139.03 s（共享负载；不作配对速度比） |
| `mris_sphere`（正式球面） | LH 面序相同，353,331/353,331 坐标标量不同；对应顶点距离均值 2.435 mm，P95 3.767 mm，最大 8.204 mm。**未通过**。 | 376.62 / 338.37 s |
| `mris_place_surface` 顶点图 | 双侧五图共 1,188,540 值逐元素相同，十个文件哈希相同；输入 white/pial 仍需验收 | 十命令合计 84.77 / 80.20 s |
| `mris_register` | 固定同一 `sphere/curv` 与 atlas，LH 353,319/353,331 坐标标量不同；顶点距离均值 0.1616 mm，P95 0.4633 mm，最大 2.8237 mm。**未通过**。 | 273.28 / 274.34 s |

在已修正的同一 T1 上游输入上，`brain`、`antsdn.brain`、`wm.seg`、`wm.asegedit`、`wm`、`filled` 六张体积图均为 0/16,777,216 体素差，MGH 头部、类型、仿射一致；步骤命令、每步耗时及哈希见[WM/filled 报告](../../validation/recon_all/python_gpu_port/native_cpp_conda_20260927/wm_chain_20260927/REPORT.md)。`mri_ca_normalize` 的 Python 实现在冻结同输入上使 `norm`、`ctrl_pts` 逐值一致，用时 35.47/91.44 s（Python/官方），见[单步报告](../../validation/recon_all/python_gpu_port/native_cpp_conda_20260927/CA_NORMALIZE_CANDIDATE_SAME_INPUT.md)。

旧无 `-ga` [拓扑配对](../../validation/recon_all/python_gpu_port/native_cpp_conda_20260927/topology_source_vs_official_same_input.json)、[GCA 配对](../../validation/recon_all/python_gpu_port/native_cpp_conda_20260927/gca_source_vs_official_same_input.json)、[顶点图配对](../../validation/recon_all/python_gpu_port/native_cpp_conda_20260927/vertex_metric_source_vs_official_same_input.json)、[球面/配准逐点配对](../../validation/recon_all/python_gpu_port/native_cpp_conda_20260927/SPHERE_REGISTRATION_SAME_INPUT.md)分别证明局部边界。球面首差已定位到首次 `MRISintegrate` 更新；单线程、禁向量化和仅修改首个缩放值均未通过最终逐点验收，见[有界诊断](../../validation/recon_all/python_gpu_port/native_cpp_conda_20260927/SPHERE_FIRST_DIVERGENCE_20260927.md)。white/pial 放置、真实 sphere.reg、顶点面积/体积/曲率及脑区统计还需从同一 T1 连续验收。[上次 v2 整例结果](../../validation/recon_all/python_gpu_port/native_cpp_conda_20260927/post_cc/BENCHMARK.md)只是历史基线。
