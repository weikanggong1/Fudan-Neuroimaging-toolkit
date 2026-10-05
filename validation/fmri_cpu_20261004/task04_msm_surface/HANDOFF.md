# 第 4 项执行交接

## 2026-10-06 当前状态

固定 490 帧投影/CIFTI 的 CPU1/8 全值与轴对照已通过；完整 180 帧 surface 已结束，球面和时序仍有差异。nodes/spectra 六组原版与十二组旧新全矩阵核对已完成，最终三种配准的十二次 GPU 旧新回归坐标/faces 一致。右侧细网格的完整保存几何重放确认两源顶点受上游 double 舍入影响；固定相同几何时三种 FNIT 选面与官方 Octree 一致。CA/CAT 个体 myelin/bias 输入仍缺。最新数值和范围以 [README](README.md)、[功能矩阵](FEATURE_MATRIX.md)及其中报告为准。

## 2026-10-05 历史交接与恢复协议

以下保留当时的失败、排队及准备状态，供追溯；这些状态不代表上方最新结果。

2026-10-05 已取回注册候选六项、features 候选八项的完整精度、实际源码与输入 SHA，全部输入稳定；原 nodes/spectra 六项输出已写出。配准 CPU1/8 都保持严格 native1 全坐标与有序 faces 逐位一致，原 MSMAll native8 波动另列。固定投影与完整 surface 的原版 command_0 在候选启动前失败；等待根任务集中取回实际 stderr。共享 tmux 路由由根任务统一执行，子任务不再发送远程命令。GPU 配对由协调者统一分配。完整数值见 [本轮报告](README.md)与[最新聚合回执](completion_status_20261005.public.json)，历史 GPU 保留实际源码范围。

## 可以启动的完整对照

- `adapter.py` 兼容 `tools/benchmark_multimodal_cpu.py`；公开适配器不含个体路径。
- 自有服务器 `runs/fmri_cpu_20261004/task04_msm_surface/preparation/registration.manifest.private.json` 已包含默认四级 MSMSulc、WRN C 一级 MSMAll 与三级 MSMAll 三例；每例双侧完整顶点和原配置，候选不读取任何原版输出。
- 同目录 `registration.input_metadata.private.json` 现场校验 29 个输入/配置/程序文件。原版 newMSM 在 nodecw8 `--help` 正常退出并包含 `inmesh/refmesh/conf`；原版依赖 `LD_LIBRARY_PATH` 已绑定其自己的环境。
- 原版球面输出为 `L.sphere.reg.surf.gii` / `R.sphere.reg.surf.gii`。原版逐侧串行用当前总预算，候选按当前 API 双侧执行，双方限于同组物理核。
- 基线与候选已独立构建 FastPD 扩展：`-O3 -std=c++17 -fno-fast-math -ffp-contract=off`；源码和构建产物 SHA、实际 import 路径保存在本轮私有构建清单。固定原 WLS 全包逐位通过。
- 配对比较覆盖 sphere coords、有序 faces、source 顶点对应、radial signed determinant 的绝对/相对方向变化、角差和弦长差。metric GIFTI、node TSV、CIFTI 全值/axis/保存 dtype/完整时间轴亦可比较。两项小型控制只验证工具能识别不同顶点和正确 CIFTI 轴，不是 MRI benchmark。

## 接着准备的真实功能

1. 全 490 帧 VN/DR/DR+VN/WRN 已使用固定 HCP v4.7 原 MATLAB 函数在 nodecw10 实际完成 CPU1/8 候选八项 maps/weights 对照。原生 nodes/spectra 六项完整输出也已写出；同 manifest 的完整 nodes 数值比较待根任务运行最新 collector。旧基线 WRN CPU8 的输入不变检查失败，不验收该行。
2. 完整 490 帧固定 sphere 的投影/CIFTI 官方与旧/新 FNIT 已在 nodecw8 同锁排队。参照使用原 fMRIPrep 25.2.4 镜像，程序、脚本、TemplateFlow 和完整几何逐文件绑定。
3. 公共 180 帧 whole surface 已在同锁排队：fresh surface 输出，以同一完整 volume 和 recon-all 为起点，包含准备、MSMSulc、投影、CIFTI 与 QC。不是新 raw→volume 或 recon-all 耗时。
4. CA/CAT 仍缺可证明来源的真正个体 myelin 与每轮 bias。旧目录有 `MyelinMap` 名字的候选，但是否 T1/FLAIR proxy、是否仅模板投影不能从名字推断。先核对 lineage，不能当作实测分支通过。

## 静态热点候选

CPU 的 radial containing-face search、adaptive 权重构建、逐标签多次 tensor 分配/小矩阵、ordered scalar reduction、最后 native warp 和 GIFTI 写盘值得按完整 baseline profile 定位。WRN 多次 pseudoinverse / 全 grayordinate 中间数组可检查内存访问；改变 pinv 容差或累加顺序之前必须用真实原版作门槛。GPU 分支保持现有算法/运算顺序；不能凭静态分析承诺任何收益。

## 已落实的 CPU 修改

- containing-face 查询改为独立行 Numba 运算，复用完整静态几何，保留面编号 tie break 和 scalar 顺序；无梯度 CPU 才使用，CUDA 与梯度保留原路径。
- 真实原 SDK Point gate 发现 CPU 向量除法的末位舍入会改变共享边归属。CPU float64 的 normalize、tangent、投影除法和 unsigned area 使用 literal double 顺序；原完整首轮球面逐位匹配，首轮成本最大差约 `7e-14`。这项 gate 不是完整配准通过。
- WRN 缓存全 BOLD 的固定 spatial/temporal demean，保持 d7–d21 全 15 组和原算法。额外两份完整 CPU 双精度观察矩阵约 710 MB；CUDA 或任何回归输入需要梯度时保留原运算序列。
- 2026-10-05 本地 Point/selector/WRN 检查 30 项、CPU/CUDA sphere execution 11 项通过，见 [聚合测试记录](focused_checks_20261005.public.json)。追加 float32 查询显式 tensor guard；运行中的服务器候选冻结不覆盖。GPU 完整 API 由协调者分配锁。

## 后续交付

`FEATURE_MATRIX.md` 列支持范围和资源缺口。已更新 MSM 三页与本目录七节说明，分别记录原版函数、fresh、完整 API 与原多线程精度波动。候选锁内 Python 源码树 SHA `72059515e650f7db02084fd41816ab78abf61bfbc1286e8f398a2ebdfd5b294f`；最终本地 selector guard 快照另有 source SHA，冻结目录不覆盖。surface 全链、原 nodes 全矩阵和最终 GPU 完整配对仍需实际完成。

## CPU 特征热点诊断准备

`profile_feature_cpu.py` 对私有 manifest 的完整 VN/DR/WRN 调用做 cProfile 和 PyTorch CPU operator 聚合。沿用 task04 的物理核与锁，仅在已有队列结束后运行；所有输出仍写新的私有目录。profile 包含仪器开销，时间只用于定位，不能替代正式配对。远端恢复后先查看 `_cifti`/保存、`_demean`、`_pinv`、`_node_timeseries`/`_spatial_maps` 及 CPU matmul/std 的实际分布。当前未根据静态成本改写 pinv、归约顺序或中间精度。

最新本地 MSM 源码逐文件 SHA 和完整三例 GPU 私有 manifest 已交给协调者；对最终快照独立构建 FastPD。`collect_report.py` 同时核对实际节点矩阵，只有匹配 features manifest 和完整原 nodes 输出的记录才参与对比。
