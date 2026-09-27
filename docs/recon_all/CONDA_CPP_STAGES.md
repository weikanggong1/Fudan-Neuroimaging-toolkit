# Conda C++ 阶段：功能、调用与验收

本文记录 FreeSurfer 8.2 源码提交 `d932c45b7941662ea380a05efef580568b98d41a` 中六个程序的整例接入，以及第七个可单独运行的 `mri_segment`。构建方法见 [Conda 编译说明](CONDA_CPP_BUILD.md)。它们在 CPU 上运行；外部目录只提供数据、许可证和在 Conda 中编译的程序。以下“精度相同”均指**同一个冻结命令输入**上的源码编译版与官方版，不表示从 T1 到最终指标已与官方 `recon-all` 一致。

## Python 用法

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

单被试等价 CLI 为：

```bash
export FS_LICENSE=/path/to/private/license.txt
fnit-recon-all subject_T1w.nii.gz /scratch/subjects/sub01 \
  --weights-dir /path/to/weights --assets-dir /path/to/assets \
  --device cuda:0 --threads 4 \
  --native-bin-dir /path/to/recon-cpp-build/bin \
  --native-topology --native-sphere --native-surface-metrics --native-registration
```

`native_bin_dir` 本身启用 `mri_em_register`；其他四个开关按表启用。`native_sphere` 和 `native_registration` 均要求 `native_topology`。注册若不启用 `native_sphere`，其输入仍是近似径向球面，因此完整原生球面链应同时启用两者。单被试函数返回 `report`，同时写入 `subject_dir/fnit-native-free-run.json`；其中 `stages[].seconds` 是整例阶段耗时，`surfaces[hemi]` 记录细分原生耗时。每次运行还记录原生程序路径和 SHA-256。多被试仅提供 `fnit.recon_all.batch.run_recon_all_python_batch` Python API，接受相同原生参数。

## 六个程序

下表的命令与当前接入的参数一致。`FS_BIN` 指**官方 FreeSurfer 8.2** 的 `bin`，用于配对参考；实际运行时 Python 用 `native_bin_dir` 下的同名 Conda 编译程序。`S=sub01`、`H=lh`，右半球以 `rh` 替换。运行 `mri_em_register` 时工作目录是 `$SUBJECTS_DIR/$S/mri`；其余命令在 `$SUBJECTS_DIR/$S/scripts`。表中 `SURF=../surf`、`ATLAS=/path/to/assets/average`。须设置 `SUBJECTS_DIR`、指向纯数据资产的 `FREESURFER_HOME` 和外部 `FS_LICENSE`。被试目录输入由上述 Python 流程先生成。

| 程序与 Python 开关 | 功能及输出 | 官方等价命令（`$FS_BIN`） |
|---|---|---|
| `mri_em_register`; `native_bin_dir=...` | 在 `nu.mgz` 与 GCA 图谱间估计仿射，读取 `brainmask.mgz`，写 `mri/transforms/talairach.lta`。 | `mri_em_register -uns 3 -mask brainmask.mgz nu.mgz /path/to/assets/average/RB_all_2020-01-02.gca transforms/talairach.lta` |
| `mris_fix_topology`; `native_topology=True` | 根据 `orig.nofix`、`inflated.nofix`、`qsphere.nofix` 修复每半球网格，写 `$SURF/$H.orig`。上游 `filled` 与这些输入仍来自近似链。 | `mris_fix_topology -threads 1 -mgz -sphere qsphere.nofix -inflated inflated.nofix -orig orig.nofix -out orig "$S" "$H"` |
| `mris_inflate`; `native_sphere=True` | 两次使用：先从 `$H.smoothwm.nofix` 生成 `$H.inflated.nofix`；修复后从 `$H.smoothwm` 生成 `$H.inflated` 和 `$H.sulc`。 | `mris_inflate -no-save-sulc "$SURF/$H.smoothwm.nofix" "$SURF/$H.inflated.nofix"`；`mris_inflate "$SURF/$H.smoothwm" "$SURF/$H.inflated"` |
| `mris_sphere`; `native_sphere=True` | 两次使用：先产生拓扑修复所需 `$H.qsphere.nofix`；再由 `$H.inflated` 产生正式 `$H.sphere`。 | `mris_sphere -q -p 6 -a 128 -seed 1234 "$SURF/$H.inflated.nofix" "$SURF/$H.qsphere.nofix"`；`mris_sphere -threads 4 -seed 1234 "$SURF/$H.inflated" "$SURF/$H.sphere"` |
| `mris_place_surface`; `native_surface_metrics=True` | 在当前 `white/pial` 上计算每半球 `thickness`、`area`、`area.pial`、`curv`、`curv.pial` 共五张顶点图。此开关**不**生成 white/pial 几何，也不计算 `area.mid` 或 `volume`。 | 五条精确命令见下方。 |
| `mris_register`; `native_registration=True` | 用 `$H.sphere` 和 folding atlas 作曲率球面配准，写 `$H.sphere.reg`，供后续 atlas 标注。 | `mris_register -curv -threads 4 "$SURF/$H.sphere" "$ATLAS/$H.folding.atlas.acfb40.noaparc.i12.2016-08-02.tif" "$SURF/$H.sphere.reg"` |

`mris_place_surface` 的五条官方命令（当前接入也逐条调用同一参数）：

```bash
"$FS_BIN/mris_place_surface" --thickness "$SURF/$H.white" "$SURF/$H.pial" 20 5 "$SURF/$H.thickness"
"$FS_BIN/mris_place_surface" --area-map "$SURF/$H.white" "$SURF/$H.area"
"$FS_BIN/mris_place_surface" --area-map "$SURF/$H.pial" "$SURF/$H.area.pial"
"$FS_BIN/mris_place_surface" --curv-map "$SURF/$H.white" 2 10 "$SURF/$H.curv"
"$FS_BIN/mris_place_surface" --curv-map "$SURF/$H.pial" 2 10 "$SURF/$H.curv.pial"
```

## 同输入精度与耗时

| 程序 | 冻结输入精度 | 官方 8.2 / Conda 编译版时间 | 验收状态 |
|---|---|---:|---|
| `mri_em_register` | LTA 4×4 矩阵 16/16 个 float64 元素逐位相同，最大绝对差 0；文件哈希只因一行创建时间而不同。 | 369.43 / 299.21 s，双侧共用的一次命令 | 同输入算法输出通过。 |
| `mris_fix_topology` | LH 117,777 顶点、235,550 面；RH 119,990 顶点、239,976 面；两侧顶点坐标及面索引逐元素相同。输出文件哈希不同。候选输入修复后仍有 LH 79、RH 12 个三角面相交。 | LH 43.2 / 53.3 s；RH 43.9 / 45.7 s（程序 `FSRUNTIME` 日志，`-threads 1`） | 同输入几何通过；上游近似输入未达到官方整例拓扑质量。 |
| `mris_place_surface` | 两侧五张图共 10 文件、1,188,540 个顶点值全部相同；10 文件 SHA-256 也各自相同。 | 10 条命令合计 84.77 / 80.20 s | 同输入五张指标图通过；当前 white/pial 仍是近似几何。 |
| `mris_inflate` | LH 117,777 个有序顶点坐标、235,550 个面及 117,777 个 sulc 值逐元素相同；sulc 文件 SHA 相同。 | 12.21 / 12.19 s | 同输入几何与 sulc 通过；inflated 文件创建时间字节不同。 |
| `mris_sphere` | LH 面序相同，但 353,331/353,331 个坐标标量不同；对应顶点距离均值 2.435 mm、P95 3.767 mm、最大 8.204 mm。官方同输入同路径重跑的有序坐标完全相同。 | 376.62 / 338.37 s | **未通过**同输入逐点几何验收。 |
| `mris_register` | 固定同一份 `sphere/curv` 和 atlas，LH 面序相同，但 353,319/353,331 个坐标标量不同；`sphere.reg` 对应顶点距离均值 0.1616 mm、P95 0.4633 mm、最大 2.8237 mm。 | 273.28 / 274.34 s | **未通过**同输入逐点配准验收；下游 atlas 指标仍需核对。 |

现有 Python `mri_ca_normalize` 在当前候选的冻结同输入上与官方 `norm.mgz` 和 `ctrl_pts.mgz` 逐值、头部均一致，单次墙钟为 35.47/91.44 s（Python/官方）；因此保留该 Python 子函数，不增加 C++ 构建目标。功能、调用命令和完整数值见 [CA normalize 配对报告](../../validation/recon_all/python_gpu_port/native_cpp_conda_20260927/CA_NORMALIZE_CANDIDATE_SAME_INPUT.md)。

前三行来自 [GCA 配对 JSON](../../validation/recon_all/python_gpu_port/native_cpp_conda_20260927/gca_source_vs_official_same_input.json)、[拓扑配对 JSON](../../validation/recon_all/python_gpu_port/native_cpp_conda_20260927/topology_source_vs_official_same_input.json)、[顶点图配对 JSON](../../validation/recon_all/python_gpu_port/native_cpp_conda_20260927/vertex_metric_source_vs_official_same_input.json)；后三行来自 [球面链同输入配对](../../validation/recon_all/python_gpu_port/native_cpp_conda_20260927/SPHERE_REGISTRATION_SAME_INPUT.md)及其 [逐点 JSON](../../validation/recon_all/python_gpu_port/native_cpp_conda_20260927/sphere_registration_same_input.json)、[官方自重复 JSON](../../validation/recon_all/python_gpu_port/native_cpp_conda_20260927/sphere_official_repeat_same_input.json)。另有 [球面 `abs` 编译行为隔离试验](../../validation/recon_all/python_gpu_port/native_cpp_conda_20260927/SPHERE_ABS_INT_TRIAL.md)：首个缩放值改为官方结果，但最终逐点误差反而增大，故未合入生产。[最新首差诊断](../../validation/recon_all/python_gpu_port/native_cpp_conda_20260927/SPHERE_FIRST_DIVERGENCE_20260927.md)进一步定位到第一次 `MRISintegrate` 更新：修复缩放后，更新前 117,777/117,777 顶点一致，更新后 91,799 个坐标标量不同；单线程和禁向量化均未消除差异，仍无可合入的逐点一致补丁。共享 gpucw1 负载没有控制，单次命令时间不能作为稳定加速比。历史官方整例的双侧 `mris_register` 日志约 833.1 s，和这次 LH 同输入配对不是同统计范围，不作比值。

这些配对只隔离了程序本身。真正的 T1 级验收还须比较官方与候选的 `aseg/filled`、white/pial 有序网格、`sphere.reg`、每个顶点的厚度/面积/体积/曲率、atlas 和逐脑区统计；当前 Python white/pial 放置及 `area.mid`、顶点 `volume`、后续统计仍可能主导误差。旧纯 Python 整例 6/138 严格通过的结果见[历史报告](../../validation/recon_all/python_gpu_port/NATIVE_FREE_CONNECTED_20260927.md)，不能替代新 Conda C++ 整例验收。

## 第七个单独程序：`mri_segment`

`mri_segment` 从 `antsdn.brain.mgz` 生成 `wm.seg.mgz`，官方命令为 `mri_segment -wsizemm 13 -mprage antsdn.brain.mgz wm.seg.mgz`。当前整例尚未生成去噪输入，故该目标已由 Conda 编译但不在 `fnit-recon-all` 调度中。在固定官方输入上，Conda 版与官方 **16,777,216/16,777,216 体素**、MGH 头部及仿射一致，单次耗时 40.45/51.85 秒；现有 Python 函数也逐体素一致，耗时 105.82 秒。具体 Python 用法、外置数据环境变量、命令和哈希见 [同输入报告](../../validation/recon_all/python_gpu_port/native_cpp_conda_20260927/MRI_SEGMENT_CONDA_PAIR.md)。
