# Conda 拓扑 GA 与 Python sphere 前处理

启用 `native_topology` 后，流程调用在 Conda 内编译并固定版本的 `mris_fix_topology_fnit`。它的 Python 前处理复现 FreeSurfer 8.2 缺陷搜索开始时使用的 sphere。源码编译补丁调整了居中坐标的交接，以及缺陷平滑中 `tanh`、`sqrt` 的 float/double 调用。未修改的 Conda `mris_fix_topology` 仅保留用于诊断对照。该阶段在 CPU 上运行，运行时不要求系统安装 FreeSurfer。

## 输入、输出与用法

`write_centered_topology_sphere(input_qsphere, output_centered)` 读取有序 FreeSurfer 三角表面 `qsphere.nofix`，写出面和尾部不变的新表面。返回字典含 `input`、`output`、`vertices`、`faces`、`iterations` 和 `seconds`。它使用 nibabel、NumPy、Numba；本函数本身不修复拓扑，也不读取官方中间数据。

`run_topology_ga_conda(subject, hemisphere, binary, assets)` 要求 FreeSurfer 格式的被试目录下有：

| 输入 | 用途 |
| --- | --- |
| `surf/{hemi}.orig.nofix` | 原始空间的有序网格 |
| `surf/{hemi}.inflated.nofix` | 膨胀表面坐标 |
| `surf/{hemi}.qsphere.nofix` | 快速球面网格 |
| `mri/brain.mgz`, `mri/wm.mgz` | 配对的强度图和白质掩膜 |
| `binary` | 本仓库 Conda 构建的修补版 `mris_fix_topology_fnit` |
| `assets` | 仅含数据的 FreeSurfer 8.2 查找资产 |

函数写入 `surf/{hemi}.topology-centered.sphere`、`surf/{hemi}.orig.premesh` 和 `scripts/{hemi}.topology-ga-fnit.log`；返回字典含 `hemisphere`、`preflight`（前处理报告）、`output`、`log`、`command` 和 `native_seconds`。Conda 程序**必须**设置 `FNIT_CENTERED_COORDS`；若未设置则中止，避免退回至另一种居中结果。Python 函数会自动设置，并核查程序的替换标记和输出。

激活 Conda 构建环境后，在仓库根目录运行：

```bash
bash tools/build_recon_all_fs_cpp_conda.sh /path/to/clean/freesurfer-8.2-source /path/to/conda-build
python -m fnit.recon_all.topology_conda_ga /path/to/subjects/sub01 lh \
  /path/to/conda-build/bin/mris_fix_topology_fnit /path/to/assets \
  --report /path/to/lh-topology.json
```

`rh` 同理。可选的 `native_topology=True` recon-all runner 选择该修补版程序，调用同一个阶段 API，再以 `remesh_surface(..., iterations=3)` 生成 `surf/{hemi}.orig`。默认 runner 的其他表面和分割步骤仍有近似处理；启用本阶段并不等于端到端重建已验收。

从 `subject/scripts` 目录运行时，对应的 FreeSurfer 8.2 命令为：

```bash
mris_fix_topology -threads 1 -mgz -sphere qsphere.nofix \
  -inflated inflated.nofix -orig orig.nofix -out orig.premesh \
  -ga -seed 1234 -threads 1 sub01 lh
mris_remesh --remesh --iters 3 ../surf/lh.orig.premesh ../surf/lh.orig
```

## 真实 T1 的冻结同输入结果

原始真实 T1 MRI 的 SHA-256 为 `c99c246200cc35479b6b8cd691457985b66f2062d4a37a0c3592ff1b678b985c`。FNIT 修正后的 `filled`、`norm` 生成了 LH/RH 共八张 `orig/inflated/smoothwm/qsphere.nofix` 表面；相对已完成的 FreeSurfer 8.2 被试，这八张表面逐点一致。以下拓扑阶段对照读取这些独立生成的表面，同时通过临时被试目录读取**冻结的官方 `brain.mgz` 和 `wm.mgz`**。因此，结果只隔离了拓扑阶段，不能作为当前候选输入整例重建的证据。Git 中没有存放患者影像。

| 相同准确输入 | LH `orig.premesh` 顶点/面 | RH 顶点/面 | 相对官方结果 |
| --- | ---: | ---: | --- |
| 官方 FreeSurfer 8.2 | 101,689 / 203,374 | 100,555 / 201,106 | 参照 |
| 未修补的 Conda C++ | 101,737 / 203,470 | 100,655 / 201,306 | 两侧均不同 |
| Python 精确 sphere + 原版 Conda C++ | 101,689 / 203,374 | 100,575 / 201,146 | LH 顶点、面顺序及 float32 坐标完全一致；RH 不同 |
| Python 精确 sphere + double-`tanh` Conda C++ | 101,689 / 203,374 | 100,548 / 201,092 | LH 一致；RH 仍不同 |
| Python 精确 sphere + double-`tanh`/`sqrt` Conda C++ | 101,689 / 203,374 | 100,555 / 201,106 | 双侧有序顶点、面和 float32 坐标完全一致 |

Python 前处理所得 sphere 的坐标和有序面与官方诊断用居中 sphere **完全一致**：LH 为 102,764/205,560，RH 为 101,454/202,936。应用两项 C++ 数学修正后，LH `orig.premesh` 的 101,689 个顶点、305,067 个 float32 坐标分量和 203,374 个有序面完全一致；RH 相应为 100,555、301,665 和 201,106。文件 SHA-256 因表面注释记录生成来源而不同。仅将 `tanh` 改为 double 的受控基线，RH 的首差发生在平滑第 0 个缺陷后的首次 crossover：三个顶点最多相差 1.19e-7 mm；随后 MRI 匹配放大误差并改变 GA 决策。将 type-2 平滑器的 `sqrt` 参数提升为 double 后，该 crossover 和最终 RH 网格恢复一致。

最终 Conda C++ 阶段的观测耗时为 LH/RH 56.92/95.48 s；另一次调用的 Python 前处理耗时为 6.34/3.55 s。较早官方同输入运行耗时 60.98/82.10 s。共享节点负载有变化，这些数据不是受控速度比较。冻结配对输入包含官方 MRI/WM 和 FNIT 独立生成的 nofix 表面；由此尚不能验收当前连通流程的 white/pial 和 atlas 指标。

连通的 Python remesh 还处理了新生成的双侧 `orig.premesh`，调用时没有提供官方表面。相对归档官方 `orig`，LH 顶点 106,622/106,622、float32 坐标 319,866/319,866、有序面 213,240/213,240 完全一致；RH 分别为 105,541/105,541、316,623/316,623、211,078/211,078。在共享的 headcw 节点耗时 100.05/119.32 s。[较早的独立 remesh benchmark](../../validation/recon_all/python_gpu_port/REMESH_VALIDATION.md)使用另一张冻结 premesh 比较 Python 与官方命令，其耗时不能与本次生成的 premesh 配对。

两侧体积几何元数据字段在数值上相同；其中 `filename` 正确指向各自被试的 `mri/wm.mgz`，所以官方与临时被试的路径不同。候选 `orig` 缺少 425 字节的官方 FreeSurfer 构建/运行来源标签，完整尾部字节及文件 SHA-256 因而不同。[双侧 orig 几何记录](../../validation/recon_all/python_gpu_port/native_cpp_conda_20260927/place_geometry_pair/accurate_filled_initial/curvature_trial/lh_fnit_sqrt_orig_vs_official.json)和[尾部及元数据报告](../../validation/recon_all/python_gpu_port/native_cpp_conda_20260927/place_geometry_pair/accurate_filled_initial/curvature_trial/orig_metadata_report.json)区分了这两类差异；RH 几何 JSON 位于 LH 文件旁。

[双侧 JSON 证据](../../validation/recon_all/python_gpu_port/native_cpp_conda_20260927/place_geometry_pair/accurate_filled_initial/curvature_trial/)记录了坐标和面逐项对照、耗时及最后一次 `sqrt` 修正前的诊断检查点。Conda 构建在打补丁前固定源码哈希；`ldd` 与 SHA-256 输出存于构建目录。本阶段没有分配 GPU 显存。
