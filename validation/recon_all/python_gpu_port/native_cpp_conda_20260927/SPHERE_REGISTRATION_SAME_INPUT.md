# FreeSurfer 球面链：Conda 源码版与官方 8.2 同输入配对

日期：2026-09-27；节点：gpucw1；单被试左半球。Conda 版从 FreeSurfer 8.2 源码提交 `d932c45b7941662ea380a05efef580568b98d41a` 编译，官方版来自 `/public/software/apps/Freesurfer/8.2.0-1/bin`（build-stamp `d932c45`）。这三个原生程序都在 **CPU** 上运行。冻结输入来自当前候选流程的 `sphere_registration_preflight/subjects/sub01`，其上游网格仍是近似流程的产物；本实验只隔离三个程序的同输入效果。

## 功能、调用和官方对照

设置 `FS_LICENSE`、`FREESURFER_HOME`（15 项纯数据资产）、`SUBJECTS_DIR`、`OMP_NUM_THREADS=4` 后，在 `$SUBJECTS_DIR/sub01/scripts` 运行下列命令。`SRC_BIN` 为 Conda 编译的 `fs_cpp/bin`，`FS_BIN` 为官方 8.2 `bin`；每条命令将前缀由 `SRC_BIN` 换成 `FS_BIN` 即是本次官方等价命令。Python 单被试入口通过 `native_sphere=True` 启用展开和成球，通过 `native_registration=True` 启用曲率球面配准；详见 [Conda C++ 阶段说明](../../../../docs/recon_all/CONDA_CPP_STAGES.md)。

```bash
"$SRC_BIN/mris_inflate" "$SURF/lh.smoothwm" "$SURF/lh.inflated"          # 生成 inflated 和 lh.sulc
"$SRC_BIN/mris_sphere" -threads 4 -seed 1234 "$SURF/lh.inflated" "$SURF/lh.sphere"
"$SRC_BIN/mris_register" -curv -threads 4 "$SURF/lh.sphere" "$ATLAS/lh.folding.atlas.acfb40.noaparc.i12.2016-08-02.tif" "$SURF/lh.sphere.reg"
```

三阶段**各自**复制同一冻结被试到 Conda/官方两个独立目录，删除该阶段目标文件，并在执行前核对所有文件的相对路径及 SHA-256 清单完全相同；球面与注册阶段使用预检的冻结 `inflated/sulc`、`sphere/curv`，不使用上一个配对命令的输出。两版使用同一 atlas 文件，SHA-256 `92f1dc820d778a67c143d4ea82597fd71a89ddc738dfa555b7ca32ed21e95c97`。两个程序串行运行；输出用 nibabel 按顶点次序解码，逐元素比较坐标、面和 sulc，`/usr/bin/time` 测墙钟。

| 阶段 | Conda / 官方墙钟 | 同输入有序输出 | 判定 |
| --- | ---: | --- | --- |
| `mris_inflate` | 12.19 / 12.21 s | `lh.inflated` 的 117,777 个顶点坐标和 235,550 个面逐元素相同；117,777 个 `lh.sulc` 值及 sulc 文件 SHA 完全相同。 | 本输入的几何和 sulc **通过**。inflated 文件 SHA 不同，首个差异是创建时间字段。 |
| `mris_sphere` | 338.37 / 376.62 s | 面索引相同；353,331/353,331 个坐标标量不同。对应顶点距离均值 2.435 mm、中位 2.434 mm、P95 3.767 mm、P99 4.172 mm、最大 8.204 mm；110,213/117,777 个顶点距离超过 1 mm。 | **未通过**逐点几何验收。 |
| `mris_register` | 274.34 / 273.28 s | 面索引相同；353,319/353,331 个坐标标量不同。对应顶点距离均值 0.1616 mm、中位 0.1116 mm、P95 0.4633 mm、P99 0.9757 mm、最大 2.8237 mm；1,119 个顶点距离超过 1 mm。 | **未通过**逐点 sphere.reg 验收。 |

球面坐标都落在半径约 100 mm 的球面上。`mris_sphere` 两版日志均显示 `-seed 1234`、4 线程，但最早可见的投影尺度已经不同（源码 0.332、官方 0.333）。刚体配准后对应顶点距离均值仍为 0.543 mm、P95 0.838 mm，差异不能仅用球体整体旋转解释。`mris_register` 的首轮搜索最优位置相同，但 SSE 日志为源码 250105.9、官方 250106.0，后续局部优化出现差异。`readelf -p .comment` 显示 Conda 二进制包含 GCC 11.4.0 编译对象，官方二进制为 GCC 4.8.5；`ldd` 分别解析到 Conda 的 `libgomp.so.1` 与系统 `/lib64/libgomp.so.1`。这些是实际构建差异，但尚未证明某个编译选项或运行库是坐标偏差的单一原因。

为检验官方球面程序自身是否可重现，我们保存第一次官方输出，把**相同被试路径**恢复成首次运行前逐文件 SHA 完全相同的输入，继续用 `-seed 1234 -threads 4` 重跑。第二次耗时 368.66 s；两次官方 `lh.sphere` 的坐标和面逐元素完全相同，三维距离最大值 0。文件 SHA 因创建时间等字节不同。此单次重复排除了本实验中官方 `mris_sphere` 自身随机波动作为 2.435 mm 平均差异的解释；未测试 Conda 版或官方 `mris_register` 的自重复。

`mris_register` 配对固定使用同一份预检 `lh.sphere`，因此 0.1616 mm 的同输入差异独立于上面两版 `mris_sphere` 的 2.435 mm 差异。整条链将球面差异继续送入注册，可能改变 atlas 边界；本配对没有重新生成 annotation，不能推断逐区厚度、面积和体积已经相同。其 T1 级差异应看整例比较报告。三项单次墙钟来自共享 gpucw1，未控制外部负载，不构成稳定提速结论。

数据：[六条命令的输入哈希、二进制哈希、计时与逐点差异](sphere_registration_same_input.json)；[官方 sphere 自重复对照](sphere_official_repeat_same_input.json)。原始日志与完整复制的被试位于 `/cwStorage/home/gongwk/Notebook_code/freesurfer_synth/work/reconall_cpp_conda_20260927/sphere_registration_pair_20260927/`，其中 `status=exit=0`、`sphere/repeat_official/status=exit=0`。
