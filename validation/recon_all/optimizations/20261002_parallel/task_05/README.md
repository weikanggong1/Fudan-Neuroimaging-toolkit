# 任务 5：MNI warp GPU 后处理验证

最新两例自产连续 MNI 阶段已完成：GPU 与重新运行的 Conda 的变形场、前向场、逆向场和检查图全部数值零差异，空间及类型一致；本次完整阶段观察加速为 3.294× / 3.247×。默认后端仍为 `conda`；不得把本页阶段结果解释为整例成功或整体等效。

## 实现与输入

[功能文档](../../../../../docs/recon_all/MNI_WARP_GPU.md)说明全部空间、参数、示例、CLI、原软件命令与参考。候选保留完整有序散射、全域 Voronoi 填充、soap-bubble 控制点/边界/停止规则及 SynthMorph 两次反对称 FP32 前向；没有使用负位移、反向网络或固定点替代。调用方可选择 `postprocess_backend="gpu"`，不需要新增依赖；共享调度与主页安装由协调者接入。

输入为两例冻结 FNIT 自产的 orig/crop/aff/deform/LTA，不读取官方结果参与计算。资源核对见 [resource_audit.json](resource_audit.json)：固定权重与模板大小及 SHA-256 全部相符；这个审计绑定早期未提交 v3，不改标成最新执行。正式执行源码由各报告中的 `code_commit` 与逐文件 SHA 标识。公开仓库只保存代码和数值元数据，不包含真实影像、模板、权重或许可证。

## 最新完整自产阶段：v12 已执行

[直接 GPU/Conda 比较](stage_comparison_v12.json)、[完整四次执行](stage_pairs_v12.json)、[14 项测试](unit_v12.log)均绑定执行快照 `ed4901f`，数值源代码沿用 `765186b`。两个新被试目录只复制 orig/crop/aff；每次重建 SynthMorph 模型，重新产生 deform/LTA、完整前向场、256³ 逆向场和最终检查图，没有把冻结 deform 当生产输入。执行顺序为 sub01 GPU→Conda、sub02 Conda→GPU。

| 完整阶段，含模型构造、权重加载、计算与读写 | sub01 GPU | sub01 Conda | sub02 GPU | sub02 Conda |
| --- | ---: | ---: | ---: | ---: |
| 模型与 deform 保存 | 12.380029 s | 10.653780 s | 11.280623 s | 10.989070 s |
| 前向转换 | 4.309824 s | 16.656071 s | 4.071649 s | 24.223876 s |
| 完整求逆 | 18.704451 s | 88.022341 s | 20.639197 s | 80.188735 s |
| 最近邻检查图 | 1.301132 s | 7.261019 s | 1.349576 s | 6.599326 s |
| 完整函数墙钟 | 37.365728 s | 123.093841 s | 37.820771 s | 122.812862 s |

墙钟包括上述步骤以外的 LTA 写入和输出网格检查；图像比较、SHA 与残差诊断在函数计时后进行。分步 JSON/CSV 分别为 [sub01 GPU](stage_v12_sub01_gpu.json) / [CSV](stage_v12_sub01_gpu.csv)、[sub01 Conda](stage_v12_sub01_conda.json) / [CSV](stage_v12_sub01_conda.csv)、[sub02 GPU](stage_v12_sub02_gpu.json) / [CSV](stage_v12_sub02_gpu.csv)、[sub02 Conda](stage_v12_sub02_conda.json) / [CSV](stage_v12_sub02_conda.csv)。

两例新 GPU 与新 Conda 直接比较：deform 各 13,961,808、forward 各 25,590,063、inverse 各 50,331,648、check 各 8,530,021 个元素全部零差异，affine 最大差为 0，dtype / intent / 单位一致；各自与冻结结果也数值零差异。文件 SHA 不全部相同，未宣称逐字节一致。完整链的全域与脑内组成残差仍与上述同 forward 基线相同，没有新增数值退化。整体等效为 `not_assessed`，138 项验收未在该阶段运行。

[最新算子](operators_v12.json) / [CSV](operators_v12.csv)：两例转换和同冻结 forward 检查图均数值零差异、几何完全一致，消除了 v4 sub01 的 1.525879e-5 mm 差异；转换 4.574612 / 4.274279 s，检查图 1.311324 / 1.329956 s。[初始化记录](context_bootstrap_v12.json)通过单次分阶段启动，无模型自动重试。

[全命令显存监测](context_v12_monitor.json) / [逐次采样](context_v12_gpu_samples.csv)：440.407633 s 包括测试、算子、四次完整阶段及验证，不能作为一次阶段耗时；1,120 样本，查询失败 0，最大间隔 3.192424 s。父子进程合计采样峰值 13,103,005,696 字节（13.103 GB / 12.203 GiB）。Torch 的全命令累计 allocated / reserved 峰值分别 9,268,777,984 / 12,272,533,504 字节；每阶段报告的是到该时刻的累计值，未重置成独立阶段峰值。本轮观测在 20 GB 预算内，NVML 采样不证明连续峰值上界。allocator 显式开启，文件系统及 Numba/Triton 缓存未清空，模型每次重建，阶段前 empty_cache；不称无缓存冷启动。内部两次反对称前向的直接观测和重复完整阶段另由 v13 执行。

## 历史已执行证据：v4

[operators_v4.json](operators_v4.json)及[CSV](operators_v4.csv)绑定 `97354aa`；后续边界及几何修正尚需新报告验证。

| 两例冻结同输入 | sub01 | sub02 |
| --- | ---: | ---: |
| 前向转换墙钟（含加载/计算/传输/保存） | 6.090627 s | 5.967084 s |
| 位移不同元素数 / 总数 | 2,713,491 / 25,590,063 | 0 / 25,590,063 |
| 位移 max / P99 | 1.525879e-5 / 1.525879e-5 mm | 0 / 0 mm |
| 同一冻结 forward 检查图墙钟 | 2.041710 s | 1.921015 s |
| 检查图不同体素数 / 总数 | 0 / 8,530,021 | 0 / 8,530,021 |

两例几何、dtype、intent、单位一致。输出描述及 gzip 元数据不同，文件 SHA 不相等；数值严格一致与文件严格一致分别评价。sub01 前向数值严格复现未通过，不能用预先声明的 1e-4 mm 排错范围替代零差异严格门槛。检查图使用相同冻结 forward，该结果不代表候选前向场产生的检查图已经验证。

[显存监测](operators_v4_monitor.json)：命令全程 27.445750 s，同次查询合计父子进程采样峰值 754,974,720 字节（0.755 GB / 0.703 GiB），76 次样本，失败 0，请求间隔 0.25 s、最大间隔 1.450328 s。采样不能证明连续峰值上界；尚不能用于完整模型阶段显存预算结论。[逐次显存与全 GPU 负载](operators_v4_gpu_samples.csv)已同步归档；全 GPU 与进程树分别查询，不能当作同一时刻的连续峰值。allocator 禁用时 Torch allocated/reserved 标为 unavailable，不以零冒充零显存。

CUDA 完整中心/边缘控制点单元回归与成熟 CPU 参考逐元素一致、完整迭代数相同，v3/v4 各 5 项通过。[v8 CPU 日志](cpu_v8.log)：公共 API 回归 12 项通过、2 CUDA 项未执行；覆盖 native nearest/rint、最后半体素散射、错误 affine、MGZ 大端、向量编码、spacing、多帧拒绝。小网格只属于单元测试，不替代真实 benchmark。

[真实散射边界诊断](coordinate_domain_v9.json)：两例各 8,530,021 个节点，旧提前夹取影响节点与 native rint 拒绝节点均为 0、全部坐标有限。这证明修正未改变这两例的散射坐标域，不等于证明完整逆场一致。诊断为单线程非计时检查，实际函数来自 v9 的 `6b5fc69` 快照，验证脚本为 `5bf8205` 新增且记录自身 SHA。

[脑内范围来源](brain_masks.json)：冻结 brainmask>0 与原图几何完全一致，脑内节点分别为 1,269,264 / 1,478,712；保存掩膜 SHA 和空间，不用掩膜剔除全域残差。

## 完整同 forward 求逆：已执行

[GPU JSON](inverse_v6.json) / [CSV](inverse_v6.csv)、[Conda 与官方 JSON](reference_v6.json) / [CSV](reference_v6.csv)均绑定 `c160cdd`。两例各 50,331,648 个分量，GPU 冷/热、冻结结果、重新执行 Conda 和官方 8.2 结果全部数值零差异，几何、dtype、intent、单位相同。GPU 与 Conda 文件 SHA 也一致；官方 SHA 不同，不能宣布官方文件逐字节一致。

| 同输入完整求逆（含读写） | sub01 | sub02 |
| --- | ---: | ---: |
| GPU 首次调用 | 34.819153 s | 22.335556 s |
| GPU 同进程重复 | 19.272995 s | 21.162687 s |
| Conda 新执行 | 91.023136 s | 81.226545 s |
| 官方 8.2 新执行 | 98.916541 s | 88.722363 s |

同进程重复相对 Conda 的本次观察收益为 4.723× / 3.838×；首次调用为 2.614× / 3.637×。首次/重复并非清空全部缓存的独立冷/热测量：之前执行过单元回归，第二例继承第一例的 JIT 缓存。未做 AB/BA 重复，不将本次值概括为稳定整阶段加速。

完整 Voronoi 为 91 / 73 轮；两例 xyz soap 各执行 50 轮。第一次全尺寸 GPU 填充/平滑计时含 JIT 为 15.936 s，后续为 0.520 / 1.418 / 0.661 s；总函数时间仍包含有序 CPU 散射、位移编码与 gzip 保存。重复调用保存占 11.540 / 13.065 s，是当前主要剩余耗时。

| 组成残差，所有实现一致 | sub01 | sub02 |
| --- | ---: | ---: |
| 全域 P99 / max | 127.447237 / 243.190948 mm | 100.167336 / 201.521957 mm |
| 脑内 P99 / max | 0.256041 / 11.080381 mm | 0.269760 / 4.896737 mm |

全域分别 16,777,216 体素，剔除数为 0；脑内严格沿用 frozen brainmask>0。两例脑内均值 0.069798 / 0.068674 mm，>0.1 mm 节点 145,947 / 190,816（仅排错分箱，不是等效门槛）。全域最大在 [0,0,1] / [0,1,0]，不在脑内。已有脑内最大异常为 [123,188,104] / [123,188,111]；[CPU 单线程定位 1](residual_location_sub01.json)和[定位 2](residual_location_sub02.json)的统计与 GPU 报告一致。私有三平面图显示最大点位于脑掩膜下缘的稀疏保留区；不因此剔除它们。所有三方结果含这些相同异常，优化新增数值退化为零，整体指标等效仍为 `not_assessed`。

[GPU 监测](inverse_v6_monitor.json)记录全命令 156.032 s（包括四次求逆、哈希及排错，不能当作一次函数耗时），采样峰值 1,006,632,960 字节，399 样本，失败 0，最大间隔 3.743 s；[参考监测](reference_v6_monitor.json)全命令 419.303 s、峰值 933,232,640 字节、1,142 样本、失败 0、最大间隔 2.337 s。对应[GPU 采样](inverse_v6_gpu_samples.csv)与[参考采样](reference_v6_gpu_samples.csv)保留外部总占用；均非连续峰值上界。

[v6 前置单元](unit_v6.log) 7 项通过。v7 几何修正与 v11 完整阶段的两个尝试分别在首次 CUDA 分配失败，见 [v7 日志](unit_v7_bootstrap_failure.log) / [v11 日志](unit_v11_bootstrap_failure.log)：pytest/model 未开始，没有该次速度或数值结果。随后[allocator 关闭](bootstrap_uncached_staged.json)和[allocator 开启](bootstrap_cached_staged.json)的分阶段诊断均通过 init / set-device / properties / memory-info / 4 字节分配 / synchronize；[两次状态](bootstrap_pair_status.json)记录返回码均为 0。此前失败原因尚未确认。v12 在相同已初始化进程内执行完整回归，不做模型自动重试。

## 待执行与门槛

1. v6 同 forward 完整 GPU / Conda / 官方求逆、全域/脑内排错已完成。
2. v12 最新前向/检查图真实回归、完整自产 GPU/Conda AB/BA、实际显存与 I/O 分步秒数已完成。
3. v13 补充直接内部双前向观测与完整阶段重复测量。
4. 原始 T1 空目录两例整例、138 项严格诊断、安装与共享调度接入由协调者执行。本任务未改变 138 门槛，整体等效为 `not_assessed`。

本轮完整 MNI 阶段有直接配对测量；原始 T1 整例尚无本任务的新测结果，不能从 MNI 局部秒数推算整例提速。

## 复现

[benchmark.py](benchmark.py)接受 JSON：`cases`（各含 `id`,`subject`）、`assets`,`native_bin`,`official_bin`,`output`,`code_commit`，mode 为 operators / inverse / reference / all。[context_validation.py](context_validation.py)将单次分阶段 CUDA 初始化、14 项测试、两例算子、完整阶段 AB/BA 置于同一进程；初始化失败立即停止，完整模型逐次重建，sub01 GPU→Conda、sub02 Conda→GPU。整个命令外层持锁和监测。

[stage_benchmark.py](stage_benchmark.py)另需 `weights`，mode 为 stage / stage-conda / baseline，output 必须是新目录。`stage` 与 `stage-conda` 都只复制 orig/crop/aff，重新运行完整两次网络前向和各自完整后处理，可做完整阶段对照；`baseline` 是冻结 deform/LTA 的隔离原生后处理计时，范围不同。完整阶段还记录自产 deform 与冻结 deform 数值差、资源 SHA、实际 Torch/interop 线程和 CPU affinity。文件系统与 JIT 缓存没有清空，不宣称无缓存冷启动。

所有性能命令须在同一共用本地文件锁内顺序执行，进程启动前绑定物理 GPU UUID，OMP/BLAS/Numba/Torch 总线程预算 4。用项目已有 [run_monitored.py](../../../python_gpu_port/run_monitored.py) 包装记录过程树显存。环境为 Torch 2.5.1 / CUDA 11.8 / Triton 3.1.0；冷启动诊断先建立同设备单元素 CUDA 上下文，不隐藏失败、自动重试或改变精度。[bootstrap.json](bootstrap.json)保留实际诊断结果；[早期 CPU JIT 后首次 CUDA 分配失败原始日志](unit_cold_failure.log)保留 2 failed / 3 passed，失败发生在首次 GPU 张量建立处；原因未确认，未计入速度测量。原 GPFS 锁返回 ENOLCK 的尝试未执行被测命令；后续统一使用共用本地文件锁。

[visualize.py](visualize.py)提供三平面检查图与全域位移误差图，固定中间切片、误差色限 1e-4 mm；只输出用户指定私有目录，影像不进入公开仓库。真实最终脑图待连续链结果产生。

## 版本记录

- `97354aa`：完整 GPU 后处理与初始单元回归；v4 真实转换/检查图。
- `22ed0bc`：native nearest 双精度半整数与 border rint。
- `c160cdd`：成熟逆场散射最后半体素边界兼容修复。
- `0dde2c5`：原生 NIfTI 中心 FP32 累加顺序。
- `765186b`：scanner RAS affine、MGZ 大端与完整 FS 向量编码校验。
- `6b5fc69`：独立完整自产连续 MNI 阶段及隔离后处理 baseline 验证入口。
- `ed4901f`：v12 同初始化进程测试、最新算子和两例完整 GPU/Conda 阶段，严格数值复现通过。
- `d327ee9`：v13 验证脚本增加内部两次 DeformNetwork 前向只读观测，数值实现不变。
