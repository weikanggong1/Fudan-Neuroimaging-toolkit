# Conda C++/Python recon-all：接入 `mri_cc` 后的整例验收

同一 T1（SHA-256 `f20410a4efd8e6a05cd04d55730a4a5492ecf9ad1b234fe0fd4661e448270c6a`）从空被试目录运行，`fnit-native-free-run.json` 状态为 `complete`，退出码 0。该 v2 配置调用六个从固定 FreeSurfer 8.2 源码提交用 Conda 编译的 **CPU** 程序，并用 Python/CUDA 和外置权重、模板运行其余阶段；没有安装的 FreeSurfer 运行包。运行中另接入已有 Python `mri_cc`，使 `aseg.auto` 和 `aseg.presurf` 与官方逐体素一致。

## 整例时间与精度

| 观测 | 当前配置 | 官方参考 |
|---|---:|---:|
| 阶段报告总耗时 | 2648.95 s，30 阶段 | 6789.6 s，归档同 T1 运行 |
| 进程实际墙钟 | 2654.95 s | 非同次配对 |
| 固定 138 项输出严格通过 | **8/138** | 参考自身 |
| 存在但不通过 / 缺失 | 79 / 51 | — |

8 项通过输出为 `mri/orig.mgz`、`mri/orig/001.mgz`、`mri/rawavg.mgz`、`mri/nu.mgz`、`mri/T1.mgz`、`mri/synthseg.rca.mgz`、`mri/aseg.auto.mgz`、`mri/aseg.presurf.mgz`。`brainmask.mgz` 相差 35/16,777,216 体素；最终 `aseg.mgz` 相差 104,986，`wm.mgz` 相差 435,674，`filled.mgz` 相差 92,188。最终 `aseg.mgz` 仍是 float32，官方为 int32。左侧 `orig` 顶点 116,148/106,622（当前/官方），右侧 118,913/105,541；面数与所有有序顶点图因此无法做同索引逐点验收。所有 18 项表面、46 项顶点图、12 项标签和 23 项统计均未通过严格比较；其中部分文件缺失。[逐文件严格结果](strict_138.json)与[空间诊断](comparison.json)保留每项差异。

双侧逐顶点图的分布均值同样明显不同；因顶点数与网格拓扑不同，下表**不是逐点对应误差**。每张图的全部文件检查及每脑区厚度、面积、体积、曲率数值在[空间诊断](comparison.json)中。

| 顶点图均值 | 左官方 / 当前 | 右官方 / 当前 |
|---|---:|---:|
| 厚度 mm | 2.0785 / 2.8996 | 2.0243 / 2.8969 |
| 白表面顶点面积 mm² | 0.7063 / 0.5041 | 0.7109 / 0.5065 |
| 顶点皮质体积 mm³ | 1.6821 / 2.5108 | 1.6579 / 2.4591 |
| 白表面曲率 | −0.0252 / −0.0477 | −0.0229 / −0.0308 |

`lh.aparc.stats` 的全局平均厚度为官方 2.23998 mm、当前 2.78072 mm，白表面积为 70,554.5/52,346.0 mm²。全部 34 个左侧 aparc 脑区仍具有官方的同名行，但数值无一整行严格匹配。

最大的阶段耗时为左表面 555.18 s、右表面 507.32 s、GCA 仿射 349.81 s、右球面配准 298.53 s、左球面配准 218.80 s 和 N4 198.52 s。双侧表面时间包含球面、拓扑与指标子步骤，不可与这些子步骤再次相加。每一步当前耗时、官方归档命令耗时、138 项明细和来源哈希见[完整 benchmark](BENCHMARK.md)及[机器摘要](benchmark_summary.json)。两次整例并非同负载配对，且输出不等价，**2648.95/6789.6 的墙钟比值不是等价重建的加速比**。

## 阶段归因与剩余工作

Python `mri_cc` 在独立同输入配对中使 `aseg.auto` 的 16,777,216 个体素与官方完全一致；此处新整例的严格比较也证实 `aseg.auto`、`aseg.presurf` 通过。剩余主要差异来自当前流程把 SynthSeg 标签直接二值化为 `wm` 和 `filled`，以平滑网格/法向射线近似 white/pial，并缺少官方 `antsdn.brain → mri_segment → WM 编辑 → mri_fill`、重网格化、真实 white/pial 放置、ribbon 回填与完整 atlas 后处理。固定输入上原生拓扑和五种顶点图可与官方一致，但输入网格拓扑已不同。Conda 版 `mris_sphere` 和 `mris_register` 在相同输入上仍有坐标差异；详见[六阶段配对](../SPHERE_REGISTRATION_SAME_INPUT.md)。

已另将 `mri_segment` 作为**第七个独立 Conda 构建目标**验证：同输入 16,777,216 个体素及 MGH 头部、仿射与官方一致，单次耗时 Conda C++ 40.45 s、现有 Python 105.82 s、官方 51.85 s；当前整例尚未产生它需要的去噪输入，因此没有把这个独立时间算入本整例。[独立报告](../MRI_SEGMENT_CONDA_PAIR.md)给出确切命令和哈希。

隔离后处理探针在完整 v2 被试的副本上调用现有 Python `volmask`、低信号重标记和 ribbon 回填：候选 `ribbon` 与官方相差 227,112 个体素，最终 `aseg` 相差 142,536 个体素，比未回填的 104,986 个多 37,550。该实验反而恶化，因此没有接入默认流程；先修上游 white/pial 几何与 `entowm` 支持的皮层标签。[后处理完整报告](../post_cc_chain_v2/README.md)保存输入哈希、输出差异、类型、仿射与阶段时间。

本次 GPU 进程采样的最大驻留量为 **20,824 MiB**，超过 20 GiB 目标；采样在前置阶段启动后才开始，不能称为完整运行连续峰值。独立 SynthSeg FP32 显存试验见[显存审计](../GPU_MEMORY_PROFILE.md)，前置残留的阶段定位另行记录。未启用 float16/bfloat16；SynthSeg 为保持体素一致单独关闭 cuDNN TF32，其余 PyTorch 路径维持项目默认 TF32 策略。

v2 运行 JSON 固定记录了当时实际调用的六个二进制 SHA-256。整例结束后为加入第七目标先在 headcw、后在 gpucw1 重新执行构建脚本，两组二进制哈希均与整例时不同，见[headcw 哈希](../bin_seven.sha256)和[gpucw1 哈希](../bin_seven_gpucw1.sha256)；这些后续构建未用于 v2 计时或 138 项比较，也不自动继承其整例数值证据。第七目标 `mri_segment` 已在两次重编后另做同输入复测。

来源：[完整运行 JSON](run.json)、[退出状态](e2e.status)、[进程计时](e2e.time)、[比较状态](compare.status)、[运行源码哈希](source.sha256)。这些文件对应同一 v2 输出；v1 的 2706.64 秒、6/138 结果保留在上一层目录，作为接入 `mri_cc` 前的历史基线。
