# 候选输入下 sphere 的首次偏离

本诊断使用去标识的真实扫描 `examples/data/sub-01_T1w.nii.gz`（SHA-256 `f20410a4efd8e6a05cd04d55730a4a5492ecf9ad1b234fe0fd4661e448270c6a`）。候选 `lh.inflated` 和 `lh.smoothwm` 来自 FNIT 的 white/smoothwm 前缀；各探针使用其逐字节副本，SHA-256 分别为 `f3a2128de469dc3c8d06e2649f85f420b26c84f95aea3c7912f677b4e57afd30` 和 `6d1a5d30639f3678c2e825d69b816f9bc35d5411546145f6c0bcab29a5fd0b91`。FreeSurfer 8.2 `mris_sphere` 程序的 SHA-256 为 `c34ca308a7fa03acdb3f689bf6125cf3d0198c37c3a62992a29f68e631c73612`。

此前使用这组相同候选输入完成了官方和 Python 的完整 sphere 运行，[结果](../candidate_sphere_annotation_20260927/candidate_sphere_official_control_audit.json)中同索引顶点位移的均值/P99/最大值为 1.652636/2.952841/3.557925 mm，面片顺序一致。本次只查首次偏离，没有重跑完整 sphere。

## 短探针和来源

原生 SSE 探针设置 `DIAG=0x10000040`、`DIAG_VERBOSE=1`、`FREESURFER_logSSE=1`，执行 `mris_sphere -threads 4 -seed 1234 surf/lh.inflated lh.sphere`，记录四次 `sses:` 决策后停止，耗时 29.56 s。[采集脚本](../experimental/capture_standard_sphere_second_sse.py)的 SHA-256 为 `532b6a74cfbe5b548ff6d229d602e9eff7198b7dfec190ae6577ab3ba27215e7`。原始日志长 150,418 字节，SHA-256 为 `24914631586dc3a85b22d10db0620885a5ee83d0a560f5b67d5d674311fa7357`；[压缩副本](native_first4_sse.log.gz)可用 `gzip -dc` 原样还原。[计时文件](native_first4_sse_time.txt)保留停止条件。

另一原生探针在相同输入副本上执行 `mris_sphere -threads 4 -seed 1234 -w 1`，生成 `sphere0000`–`sphere0004` 后停止，耗时 30.03 s（SIGTERM）。[快照清单](snapshot_probe.json)记录命令、输入/程序哈希和五份输出哈希；网格保留在清单所列 gpucw1 scratch 目录。[Python 四步比较程序](../experimental/probe_standard_sphere_default_continuous.py)的 SHA-256 为 `bbf2f475da6b06c2a2f70d82a436ca0be0a302ffab4e9b1ace1b340f0e621648`。它从候选文件计算目标距离、梯度和线搜索，并与原生快照比较；在第一个不相等的更新处停止。[结果 JSON](python_vs_native_first4.json)含代码哈希、SSE 各项、决策和逐顶点差异。其中 `finish` 模块哈希仅为来源记录；本次未执行该步骤。

普通官方候选完整运行未保存逐次更新网格。输入哈希一致及先前在冻结官方输入上对 `-w 1` 的验证，都不能证明本次候选 `-w 1` 前缀与普通完整运行逐位相同。SSE 日志和网格快照也来自两个独立进程；首步决策在日志打印精度内一致。

## 首次偏离

| 候选输入下的初始修复 | 原生日志 | Python |
| --- | ---: | ---: |
| 初始 `sphere0000` 坐标 | 319,866 个分量 | 319,866/319,866 精确相同 |
| 初始负面积 SSE | 275.835616 | 275.835615519 |
| 初始距离 SSE | **26.863932** | **26.863064785** |
| 初始总 SSE | **302.699548** | **302.698680304** |
| 首次线搜索选中试探 | index 2，dt 1027.938 | index 2，dt 1027.937440996 |
| 首次更新后坐标 | 319,866 个分量 | 308,844/319,866 精确相同；最大顶点误差 0.000011444 mm |

偏离从**首次更新前的距离 SSE** 开始。初始网格精确相同，负面积项在日志六位小数精度内相同，距离项解释了打印出的总 SSE 差额。尚需分辨差额来自目标距离矩阵，还是当前球面距离与 SSE 求值。把 Python 首步 `dt` 拟合到原生网格得到 1027.937658，最大顶点误差仍为 0.000011444 mm；仅调整步长不能解决已观察到的差异。

原生前四次 `dt` 为 1027.938、18451.210、399.497、34325.025；独立 Python 候选完整运行记录为 1027.937441、18450.785169、399.364756、34319.906153。两者 index 0–2 的阶段均为 `initial_repair`、梯度平均数 1024，index 3 的平均数为 256。第 0 次更新后坐标已不同，后续 `dt` 不能作为同状态精度比较。本次未修改生产代码。

两个失败尝试不计入数值结果：`-v` 未附取值，只打印帮助；合用 `DIAG` 和 `-w 1` 时，程序写出 `sphere0000` 后因 `MRISwriteIntoVolume` 诊断错误退出，未完成更新。因此 SSE 和网格分别采集。

## 首行距离与源码核对

[逐项审计](initial_row0_metric_audit.json)对原生首次 `logSSE:1` 打印的第 0 顶点数据进行比较。Python 在相同候选 `smoothwm` 上调用 `sample_standard_metric_matrix` 和 `average_standard_metric`，得到 8,268,920 个目标距离；构建耗时 25.81 s（含 JIT）。随后在精确相同的原生 `sphere0000` 上调用 `_spherical_distance`。第 0 顶点的 79 个目标距离、79 个当前距离均与原生日志**在打印的六位小数上逐项一致**。这不能证明全矩阵逐位相同，也不能解释 0.000867 的加权距离 SSE 差额。

[FreeSurfer 8.2 对应源码](https://github.com/freesurfer/freesurfer/blob/d932c45/utils/mrisurf_sseTerms.cpp#L1442-L1622)把目标距离和当前距离存为 float，距离比例、每顶点 SSE 和总 SSE 使用 double；逐顶点循环会跳过被标记的顶点及无效距离，并可能应用 `vsmoothness`。Python 当前的 `_distance_sse` 则按行顺序对全部距离求和。原生日志只打印第 0 顶点的距离，没有逐顶点 SSE；现有文件无法判断其余样本/顺序、当前距离或筛选条件何处首次不同。候选 `smoothwm` 相对官方输入有小幅坐标扰动，角度采样的阈值分支可能改变后续全局随机序列；这仍是待检假说，不能据此修改采样或 SSE 算法。之前在冻结官方输入上验证的 8,268,920/8,268,920 目标距离精确一致，不代表本次候选输入也精确一致。

最小后续核查是复用已有的 [GDB 内存采集脚本](../experimental/capture_standard_sphere_trial_memory.py)，以相同程序和候选输入、`SPHERE_LOGSSE_HIT=1` 停在第一次 SSE 后，只在 scratch 保存原生 `native_offsets.bin`、`native_target.bin`、`native_current.bin` 和 `native_xyz.bin`。先逐位比较全部目标距离及行长度；若相同，再比较当前距离，并用原生数组在 Python 重算 SSE。若目标距离不同，定位第一行差异后再检查邻居 ID 与采样阈值。该探针尚未执行，不需要重跑完整 sphere，也不应在分叉位置查明前调参或修改生产实现。
