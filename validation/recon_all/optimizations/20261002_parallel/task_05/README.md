# 任务 5：MNI warp GPU 后处理验证

截至本记录，完整候选已提交，真实求逆、三方对照及自产连续链仍在共享锁队列中。默认后端仍为 `conda`；不得把本页阶段结果解释为整例成功或整体等效。

## 实现与输入

[功能文档](../../../../../docs/recon_all/MNI_WARP_GPU.md)说明全部空间、参数、示例、CLI、原软件命令与参考。候选保留完整有序散射、全域 Voronoi 填充、soap-bubble 控制点/边界/停止规则及 SynthMorph 两次反对称 FP32 前向；没有使用负位移、反向网络或固定点替代。调用方可选择 `postprocess_backend="gpu"`，不需要新增依赖；共享调度与主页安装由协调者接入。

输入为两例冻结 FNIT 自产的 orig/crop/aff/deform/LTA，不读取官方结果参与计算。资源核对见 [resource_audit.json](resource_audit.json)：固定权重与模板大小及 SHA-256 全部相符；这个审计绑定早期未提交 v3，不改标成最新执行。正式执行源码由各报告中的 `code_commit` 与逐文件 SHA 标识。公开仓库只保存代码和数值元数据，不包含真实影像、模板、权重或许可证。

## 已执行证据

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

## 待执行与门槛

1. 同一冻结 forward：两例完整 GPU 求逆冷/热，以及固定源码 Conda / 官方 8.2 的隔离参考；报告全域与脑内残差 max/P99、最大异常位置、脑内归属与越界节点，不掩膜剔除全域数据。
2. 修正几何与量化后的前向/检查图真实回归。
3. 最新完整自产 SynthMorph→转换→求逆→检查图连续阶段，两次 FP32 网络前向、实际过程显存、I/O 与分步秒数；新目录保护冻结输入。
4. 原始 T1 空目录两例整例、138 项严格诊断、安装与共享调度接入由协调者执行。本任务未改变 138 门槛，整体等效为 `not_assessed`。

当前没有新测得的完整阶段或整例加速比。不能将旧 188–196 s 减去本页局部秒数推算总提速。

## 复现

[benchmark.py](benchmark.py)接受 JSON：`cases`（各含 `id`,`subject`）、`assets`,`native_bin`,`official_bin`,`output`,`code_commit`，mode 为 operators / inverse / reference / all。[stage_benchmark.py](stage_benchmark.py)另需 `weights`，mode 为 stage / stage-conda / baseline，output 必须是新目录。`stage` 与 `stage-conda` 都只复制 orig/crop/aff，重新运行完整两次网络前向和各自完整后处理，可做完整阶段对照；`baseline` 是冻结 deform/LTA 的隔离原生后处理计时，范围不同。完整阶段还记录自产 deform 与冻结 deform 数值差、资源 SHA、实际 Torch/interop 线程和 CPU affinity。文件系统与 JIT 缓存没有清空，不宣称无缓存冷启动。

所有性能命令须在同一共用本地文件锁内顺序执行，进程启动前绑定物理 GPU UUID，OMP/BLAS/Numba/Torch 总线程预算 4。用项目已有 [run_monitored.py](../../../python_gpu_port/run_monitored.py) 包装记录过程树显存。环境为 Torch 2.5.1 / CUDA 11.8 / Triton 3.1.0；冷启动诊断先建立同设备单元素 CUDA 上下文，不隐藏失败、自动重试或改变精度。[bootstrap.json](bootstrap.json)保留实际诊断结果；[早期 CPU JIT 后首次 CUDA 分配失败原始日志](unit_cold_failure.log)保留 2 failed / 3 passed，失败发生在首次 GPU 张量建立处；原因未确认，未计入速度测量。原 GPFS 锁返回 ENOLCK 的尝试未执行被测命令；后续统一使用共用本地文件锁。

[visualize.py](visualize.py)提供三平面检查图与全域位移误差图，固定中间切片、误差色限 1e-4 mm；只输出用户指定私有目录，影像不进入公开仓库。真实最终脑图待连续链结果产生。

## 版本记录

- `97354aa`：完整 GPU 后处理与初始单元回归；v4 真实转换/检查图。
- `22ed0bc`：native nearest 双精度半整数与 border rint。
- `c160cdd`：成熟逆场散射最后半体素边界兼容修复。
- `0dde2c5`：原生 NIfTI 中心 FP32 累加顺序。
- `765186b`：scanner RAS affine、MGZ 大端与完整 FS 向量编码校验。
- `6b5fc69`：独立完整自产连续 MNI 阶段及隔离后处理 baseline 验证入口。
