# recon-all 的 Conda C++ 与 Python 表面阶段

当前从 FreeSurfer 8.2 固定源码提交 `d932c45b7941662ea380a05efef580568b98d41a` 编译六个 CPU 程序；快速球面、正式球面和球面配准使用仓库现有 Python API。下表的精度比较均用去标识的真实 OpenNeuro ds000114 sub-01 T1w 衍生影像（NIfTI SHA-256 `f20410a4efd8e6a05cd04d55730a4a5492ecf9ad1b234fe0fd4661e448270c6a`）及其官方 FreeSurfer 8.2 中间结果。冻结同输入通过不代表从 T1 到脑区指标已通过连续整例验收。

## 单被试与批量调用

```python
from fnit.recon_all.native_free import run_recon_all_python

report = run_recon_all_python(
    t1="subject_T1w.nii.gz",  # 单幅 T1w NIfTI
    subject_dir="/scratch/subjects/sub01",  # 空被试输出目录
    weights_dir="/path/to/weights",  # 外置模型目录
    assets_dir="/path/to/assets",  # 外置模板和图谱目录
    device="cuda:0",  # PyTorch 推理设备
    threads=4,  # CPU 线程数
    native_bin_dir="/path/to/recon-cpp-build/bin",  # Conda 编译程序目录
    native_topology=True,  # 拓扑修复及 Python remesh
    native_sphere=True,  # 膨胀与 Python 球面
    native_surface_metrics=True,  # Conda 顶点指标程序
    native_registration=True,  # Python 球面配准
)
# report 是含阶段耗时、输出路径及实现来源的运行记录字典。
```

```bash
export FS_LICENSE=/path/to/private/license.txt
fnit-recon-all subject_T1w.nii.gz /scratch/subjects/sub01 \
  --weights-dir /path/to/weights --assets-dir /path/to/assets \
  --device cuda:0 --threads 4 \
  --native-bin-dir /path/to/recon-cpp-build/bin \
  --native-topology --native-sphere --native-surface-metrics --native-registration
```

CLI 前两个位置参数依次是 T1w 输入和空输出目录；`--weights-dir`、`--assets-dir`、`--native-bin-dir` 分别指外置模型、模板和 Conda 编译程序目录；`--device` 选 PyTorch 设备，`--threads` 指 CPU 线程数。`FS_LICENSE` 是外部许可证路径。输出目录中的 `mri/`、`surf/`、`label/`、`stats/` 与运行报告结构见[主文档](README.md#输入与输出结构)。

`native_bin_dir` 提供三个必需程序；`native_topology` 启用 Python 居中球、Conda `mris_fix_topology_fnit` 及 Python remesh，`native_sphere` 启用 `mris_inflate` 加 Python 双球面，`native_surface_metrics` 启用 `mris_place_surface` 的五张指标图，`native_registration` 启用 Python 球面配准。后两个球面开关仍沿用原 CLI/Python 名称以保持调用兼容，内部数值实现已换成 Python。`native_sphere` 和 `native_registration` 均要求 `native_topology`。多被试仅提供 `fnit.recon_all.batch.run_recon_all_python_batch` Python API，返回按输入顺序排列的单被试报告；输入/输出目录及报告结构见[主文档](README.md#输入与输出结构)。

## Conda 程序的输入与输出

`S=sub01`、`H=lh`（另一侧为 `rh`）、`SURF=$SUBJECTS_DIR/$S/surf`、`ATLAS=/path/to/assets/average`。下列命令为官方等价调用；实际程序位于 `native_bin_dir`，需要外部 `FS_LICENSE`。体积命令在被试 `mri` 目录运行，拓扑在 `scripts` 目录运行。所有数值输出均保留空间几何或网格元信息。

| 程序；当前启用方式 | 输入 | 输出及格式 | 官方等价命令 |
|---|---|---|---|
| `mri_em_register`；必需 | `nu.mgz`、`brainmask.mgz`、GCA 图谱 | `mri/transforms/talairach.lta`，4×4 仿射及体积几何 | `mri_em_register -uns 3 -mask brainmask.mgz nu.mgz "$ATLAS/RB_all_2020-01-02.gca" transforms/talairach.lta` |
| `mri_segment`；必需 | `antsdn.brain.mgz` | `wm.seg.mgz`，与输入同空间维度的 uint8 MGZ | `mri_segment -wsizemm 13 -mprage antsdn.brain.mgz wm.seg.mgz` |
| `mri_edit_wm_with_aseg`；必需 | `wm.seg.mgz`、`brain.mgz`、`aseg.presurf.mgz`、`entowm.mgz` | `wm.asegedit.mgz`，与输入同空间维度的 uint8 MGZ | `mri_edit_wm_with_aseg -keep-in -fix-ento-wm entowm.mgz 3 255 255 -fix-acj aseg.presurf.mgz 255 255 -fill-seg-wm -fix-scm-ha 1 wm.seg.mgz brain.mgz aseg.presurf.mgz wm.asegedit.mgz` |
| `mris_fix_topology_fnit`；`native_topology` | `H.orig.nofix`、`H.inflated.nofix`、`H.qsphere.nofix`、`brain.mgz`、`wm.mgz` | `surf/H.orig.premesh`，有序顶点和三角面、MGH volume geometry | `mris_fix_topology -ga -seed 1234 -threads 1 -mgz -sphere qsphere.nofix -inflated inflated.nofix -orig orig.nofix -out orig.premesh "$S" "$H"`；[居中球与构建差异](TOPOLOGY_CONDA_GA.md) |
| `mris_inflate`；`native_sphere` | `surf/H.smoothwm.nofix`，后续 `surf/H.smoothwm` | `H.inflated.nofix`；再生成 `H.inflated` 网格和每顶点 float32 `H.sulc` | `mris_inflate -no-save-sulc "$SURF/$H.smoothwm.nofix" "$SURF/$H.inflated.nofix"`；`mris_inflate "$SURF/$H.smoothwm" "$SURF/$H.inflated"` |
| `mris_place_surface`；`native_surface_metrics` | 已有同顶点顺序的 `surf/H.white`、`surf/H.pial` | `H.thickness`、`H.area`、`H.area.pial`、`H.curv`、`H.curv.pial`，各为按顶点索引排列的 float32 图；此处不生成几何 | 五条命令见下方 |

```bash
mris_place_surface --thickness "$SURF/$H.white" "$SURF/$H.pial" 20 5 "$SURF/$H.thickness"
mris_place_surface --area-map "$SURF/$H.white" "$SURF/$H.area"
mris_place_surface --area-map "$SURF/$H.pial" "$SURF/$H.area.pial"
mris_place_surface --curv-map "$SURF/$H.white" 2 10 "$SURF/$H.curv"
mris_place_surface --curv-map "$SURF/$H.pial" 2 10 "$SURF/$H.curv.pial"
```

## Python 拓扑和球面函数

- `write_centered_topology_sphere(input_qsphere, output_centered) -> dict`：生成有序面与 footer 不变的 GA 居中球，返回路径、网格数量、迭代次数、秒数；[真实 T1 双侧居中球 exact 和 GA 结果](TOPOLOGY_CONDA_GA.md)。
- `write_quick_sphere(input_inflated_nofix, output_qsphere_nofix) -> None`：读 `surf/H.inflated.nofix` 的有序网格，写同面序 `H.qsphere.nofix`；官方命令 `mris_sphere -q -p 6 -a 128 -seed 1234 INFLATED.NOFIX QSPHERE.NOFIX`。v3 真实 T1 同输入的[左](../../validation/recon_all/python_gpu_port/native_cpp_conda_20260927/v3_e2e_20260927/quick_sphere_lh.json)/[右](../../validation/recon_all/python_gpu_port/native_cpp_conda_20260927/v3_e2e_20260927/quick_sphere_rh.json)顶点、面完全一致。
- `remesh_surface(input_orig_premesh, output_orig, iterations=3) -> None`：读 `H.orig.premesh` 的有序网格与元信息，完成三轮按 FreeSurfer 顺序的边拆分、坍缩和切向平滑，写 `H.orig`，即皮层白质网格的正式拓扑。官方命令 `mris_remesh --remesh --iters 3 --input "$SURF/$H.orig.premesh" --output "$SURF/$H.orig"`。真实 T1 [双侧冻结同输入验收](../../validation/recon_all/python_gpu_port/REMESH_VALIDATION.md)的顶点、面及 volume geometry 逐位一致；官方 LH/RH 14.46/19.53 秒，Python 100.87/97.16 秒，是不同负载下的观察值。当前调度在 Conda topology 之后调用该函数并单列 `topology_remesh_seconds`。随后调用 `remove_intersection_surface(orig, orig)`，对应官方 `mris_remove_intersection "$SURF/$H.orig" "$SURF/$H.orig"`；当前真实 T1 左侧检测 0 个相交面、输出文件 SHA 不变，用时 2.48 秒。官方该 T1 双侧均为零相交；其他输入若出现相交则显式报错，迭代修复分支尚未实现，另记 `topology_intersection_seconds`。
- `run_standard_sphere(inflated, smoothwm, output, finish_device="cpu") -> dict`：读相同面序的 `surf/H.inflated` 与 `H.smoothwm`，写有序三角网格 `H.sphere`。返回字典含输入/输出路径、`finish_device`、初始负面积比例、投影/原始度量/JIT/修复/总耗时、每步 `{index, stage, weight, averages, dt, seconds}` 列表及最后修复的负面计数。官方命令 `mris_sphere -threads 4 -seed 1234 INFLATED SPHERE`。只在最后重叠修复使用 `finish_device`；主优化在 CPU/Numba。独立 CLI：`python -m fnit.recon_all.sphere_standard_run INFLATED SMOOTHWM SPHERE --finish-device cpu --report REPORT.json`。
- `run_register_sphere(sphere, smoothwm, sulc, atlas, output, overlap_device="cpu") -> dict`：读球面/白质网格、每顶点 sulc 图及外置 folding-atlas TIFF，写有序注册球面 `H.sphere.reg`。返回字典含五个路径、输入/输出 SHA-256、临时 sulc 种子 SHA-256、`overlap_device`、`sulc_pass`/`smoothwm_pass` 的有序更新轨迹与耗时以及总耗时；临时种子在返回前删除。官方命令 `mris_register -curv -threads 4 SPHERE ATLAS SPHERE.REG`。主优化在 CPU PyTorch/Numba；独立 CLI：`python -m fnit.recon_all.mris_register_run SPHERE SMOOTHWM SULC ATLAS SPHERE.REG --overlap-device cpu --report REPORT.json`。

CPU 用于两个末端重叠修复，是已通过冻结输入验收的模式；此前一次 LH CUDA 末端试验因共享 GPU 显存不足中断。当前整例 `fnit-native-free-run.json` 在 `surfaces[H].standard_sphere_report` 和 `sphere_registration.reports[H]` 记录上述完整字典，`stages[].seconds` 给整例墙钟。该组合需要真实有符号 `sulc` 和准确上游网格；冻结输入验收不能抵消拓扑或 white/pial 偏差。

## 真实 T1 的冻结同输入精度与耗时

时间是不同日期与共享负载下的实际运行，不作为等价整例加速比。严格比较先检查有序顶点/面和 MGZ 几何，再比较顶点图。

| 阶段 | 精度 | 官方 / 当前墙钟 |
|---|---|---:|
| `mri_em_register` | 4×4 LTA 16 个 float64 元素逐位相同 | 369.43 / 299.21 s |
| `mri_segment` | `wm.seg` 0/16,777,216 体素差，MGH 头部及仿射一致 | 51.85 / 40.45 s（连续 WM 链当前 81.63 s） |
| `mri_edit_wm_with_aseg` | `wm.asegedit` 0/16,777,216 体素差 | 旧官方日志 28.42 / 当前 45.99 s（连续 WM 链 52.88 s） |
| `mris_fix_topology_fnit` 官方 `-ga -seed 1234` | 准确同输入下双侧 `orig.premesh` 及 Python remesh 后 `orig` 的有序顶点、坐标、面全 exact；LH/RH 最终 `orig` 106,622/213,240 和 105,541/211,078。[详细证据](TOPOLOGY_CONDA_GA.md) | 官方拓扑 LH/RH 60.98/82.10 s；Python 居中球 6.34/3.55 s + Conda C++ 56.92/95.48 s + Python remesh 100.05/119.32 s，非同期负载 |
| `mris_inflate` | LH 同输入 117,777 顶点、235,550 面及 117,777 个 sulc 值全部一致 | 12.21 / 12.19 s |
| Python quick sphere | v3 双侧输入：LH 102,764 顶点/205,560 面、RH 101,454/202,936，全部有序坐标和面一致；更新后的连续左侧输出也逐点 exact | 官方归档 LH/RH 83.88/96.84 s（日志 `FSRUNTIME@`）；Python 139.24/139.03 s；非同期共享负载 |
| Python standard sphere | 冻结官方双侧输入最终有序顶点、面及 volume geometry 全部逐位一致；LH 243 步、RH 134 步 | 官方 LH/RH 240.64/109.72 s；Python 372.82/220.17 s |
| `mris_place_surface` 五图 | 双侧十文件共 1,188,540 顶点值逐元素一致、SHA-256 一致；前提是同一 white/pial 几何 | 官方/Conda 十命令合计 84.77/80.20 s |
| Python sphere registration | 冻结官方球面/smoothwm/sulc/atlas：LH 106,622、RH 105,541 个有序顶点、面及 volume geometry 全部一致；两段优化轨迹与原生一致 | 官方 LH/RH 157.62/161.62 s；Python 372.85/347.36 s |

[修复后的自产输入左半球标准球面复验](../../validation/recon_all/python_gpu_port/candidate_sphere_first_difference_20260927/full_stage/README.md)显示：同候选输入的 Python 与官方有序顶点及面全部一致，单次墙钟分别为 884.56/328.61 秒。与归档官方球面的平均 2.959 mm 差异在两种同输入实现中相同；[上游审计](../../validation/recon_all/python_gpu_port/candidate_sphere_first_difference_20260927/full_stage/UPSTREAM_FIRST_DIFFERENCE.md)将首个已保存几何差异定位在 `white.preaparc`。这项通过只覆盖左侧完整 `sphere` 单阶段，未覆盖 `sphere.reg` 和整例。

[WM/filled 连续同输入报告](../../validation/recon_all/python_gpu_port/native_cpp_conda_20260927/wm_chain_20260927/REPORT.md)证明上游六张体积图逐体素一致。Python 标准球面[完整验证](../../validation/recon_all/python_gpu_port/SPHERE_STANDARD_STATUS.md)及球面配准[完整验证](../../validation/recon_all/python_gpu_port/MRIS_REGISTER_STATUS.md)保留每步轨迹和输出哈希。另有真实 T1 的[左半球 Python sphere→sphere.reg 连续链](../../validation/recon_all/python_gpu_port/native_cpp_conda_20260927/sphere_chain_lh_20260927/REPORT.md)：Python 写入的球面直接输入 Python 配准，106,622 个有序顶点、213,240 个面和 volume geometry 与官方逐位相同；球面和配准 Python API 分别耗时 372.82/369.81 秒，原生参考 240.64/157.62 秒为不同负载下观察值。右侧独立连续链仍未验收。当前整例还使用近似 white/pial；历史[white.preaparc 首差诊断](../../validation/recon_all/python_gpu_port/native_cpp_conda_20260927/WHITE_PREAPARC_FIRST_DIVERGENCE_20260927.md)在冻结官方 MRI/网格/标签输入、连续调用 Conda 预表面与最终 white 放置命令时，测得左侧最终 white 0.0221 mm 平均顶点位移，并将首个可见预表面差异定位于第 6–7 步浮点力/积分路径。当前可选链调用 Conda white.preaparc 后执行 Python 三轮 smoothwm 平滑，再以 smoothwm 近似最终 white；它并未运行该诊断中的最终 white 优化器，[候选前缀对照](WHITE_PREAPARC_CONDA_CHAIN.md)和[连通 LH 报告](../../validation/recon_all/python_gpu_port/white_connected_prefix_20260927/README.md)给出各自误差。独立[双侧 Python pial.T1](PYTHON_PIAL_PLACEMENT.md)已在正确官方上游输入上 exact，但尚未接入当前整例，因为 white 与其标签/阈值链未验收。先前 Conda `mris_sphere`/`mris_register` 数值差异仅作[历史诊断](../../validation/recon_all/python_gpu_port/native_cpp_conda_20260927/SPHERE_REGISTRATION_SAME_INPUT.md)，当前构建和调度不再调用这两程序。[发布验收门槛](../../validation/recon_all/python_gpu_port/RELEASE_GATES.md)仍要求同一 T1 的最终 138 项逐文件核对。
