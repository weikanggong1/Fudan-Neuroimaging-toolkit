# recon-all 阶段状态（2026-09-28）

本页记录截至停止任务时已纳入 main 的实现与验收边界。目标仍是同一 T1 下核对官方 FreeSurfer 8.2 的体素分割、双侧有序 white/pial/sphere.reg、逐顶点厚度/面积/体积/曲率、逐脑区统计及整例时间。阶段测试使用真实 T1；冻结官方上游文件的同输入结果只证明该阶段的计算，不证明 FNIT 从原始 T1 产生等价上游。

## 已完成的阶段结果

| 部分 | 当前实现与设备 | 本阶段证据 | 时间范围 |
| --- | --- | --- | --- |
| N4、初始分割与入口依赖 | 仓库 Conda ITK C++ N4 在 CPU；SynthStrip、33 类 SynthSeg 和部分体素算子可用 PyTorch/CUDA；recon-all 入口已移除 Surfa、SimpleITK、ANTsPy、DIPY 运行时调用 | 同一真实 T1 的 N4 `nu0` 有 9/16,777,216 个体素差 1，`nu` 有 8 个体素差 1–2；禁用上述模块的入口导入检查通过，见[N4](N4_ITK_CONDA.md)与[导入记录](../../validation/recon_all/python_gpu_port/runtime_imports_20260928.json) | N4 单次 107.89 秒；先前官方同阶段 168.26 秒，日期/负载不同 |
| 独立图像与配准函数 | WMH-SynthSeg、SynthSR、TorchFAST、FLIRT 与 connectome 自动配准已移除 Surfa 运行时路径 | 真实输入的新旧 FNIT 输出按各自合同相同；FLIRT 图像和矩阵逐字节相同，connectome b0/T1 矩阵 16/16 元素相同。见[FLIRT](../../validation/flirt_no_surfa_20260928/README.md)、[connectome](../../validation/connectome_registration_no_surfa_20260928/README.md) | 各报告为单次配对；不据此声称稳定提速 |
| 白质面 Python 放置 | CPU NumPy/Numba 逐步诊断；生产流程尚未调用完整 Python white | 左侧第三轮 step27–34 坐标最大差 7.91e-6 mm、0 顶点超过 1e-4 mm；第34步自产坐标 SSE 比官方高 0.064687，在官方同坐标重算仅差 4.75e-7。见[逐轮报告](WHITE_PYTHON_FIRST_PASS.md) | 第三轮 Python 362.54 秒；官方 GDB 重放含前三轮与 RAM 写出 172.51 秒，计时边界不同 |
| ribbon 与最终 aseg | Python CPU；已接入 runner，按官方阶段顺序生成三张 ribbon、hypos 和最终 aseg | 冻结真实 T1 的五张图各 16,777,216 个体素与官方全同，仿射差 0。见[同输入报告](../../validation/recon_all/python_gpu_port/ASEG_RIBBON_RUNNER_20260928.md) | 本次 ribbon 2.95 秒、hypos 3.10 秒、aseg 修正 3.84 秒，均含 I/O |
| aparc/a2009s/DKT 与 wmparc 体积图 | Python CPU；已用双侧 white/pial、皮层标签和注释替换 runner 的最近 white 顶点近似 | 冻结真实 T1 的四张图各 16,777,216 个体素与官方全同，仿射差 0。见[同输入报告](../../validation/recon_all/python_gpu_port/ATLAS_VOLUME_RUNNER_20260928.md) | 本次三张皮层图合计 11.74 秒、wmparc 4.26 秒，含 I/O |
| 可选最终表面串联 | `--native-white-preaparc` 已按 preaparc→皮层标签→sphere.reg/注释→Conda C++ 最终 white→Python pial→顶点图接线 | 单侧冻结官方输入的最终 white 已写出；平均/最大逐顶点位移 0.000359/0.752 mm，与既有 Conda 最终 white 诊断相符。Python pial 在约 17 分钟运行后按停止要求终止，未取得此新串联的 pial 或顶点图结果 | 本次 Conda white 日志约 2.80 分钟；pial 和后续图没有完成计时 |

历史上从原始 T1 完成的 v3 整例耗时 **2903.79 秒**，138 项中通过 19 项、缺失 47 项、不同 72 项。该结果使用旧 N4 和旧表面调度；[历史报告](../../validation/recon_all/python_gpu_port/native_cpp_conda_20260927/v3_e2e_20260927/BENCHMARK.md)不能作为当前版的整例精度或速度。当前版没有重新运行完整 T1，也没有新的 138 项结果。已经完成的冻结输入阶段不能合并成“端到端一致”或“等价重建加速”。

## 剩余工作

1. 完成 Python white 的第四轮、前三轮连续传递、最终交点清理与写出，并在右半球和第二例真实 T1 上核对。当前独立 API 仍只覆盖首轮前缀；可选生产链暂用 Conda C++ 最终 white。
2. 用 FNIT 自产的同一 T1 前序文件连续运行双侧最终 white、Python pial、厚度/面积/体积/曲率图；核对有序顶点与面、逐点误差、局部异常及每个脑区统计。可选新串联本阶段只完成接线，尚未跑通。
3. 完成标准 sphere.reg 的最终配准与注释验收，补足其余官方后处理分割、atlas、统计文件。已有 Python 单阶段结果不能替代完整输出检查。
4. 清除通用 SynthMorph、FNIRT、FastVBM、TBSS 等路径及 pyproject/Conda 环境中的 Surfa 依赖；此次通用 SynthMorph affine 和 FNIRT 的在途实验没有纳入阶段发布。
5. 在允许重新运行完整 T1 后，固定同一输入、设备、线程和资源，逐文件核对 138 项以及逐顶点和逐脑区指标，再逐阶段记录 CPU/GPU 峰值显存、墙钟及官方同条件耗时。当前没有可发布的等价重建速度比。

本阶段保留经验证的局部改动与明确标注的可选实验路径。没有安装 FreeSurfer 运行包的依赖；三个必需和若干可选程序仍须从固定 FreeSurfer 源码在 Conda 环境编译。整个仓库目前仍安装 Surfa，尚不满足全仓库无 Surfa 安装要求。
