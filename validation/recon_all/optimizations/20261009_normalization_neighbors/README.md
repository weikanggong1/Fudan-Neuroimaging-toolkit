# 三维控制点 GPU 邻域：完整归一化回归

本目录是公开 ds000114 两例的冻结同输入阶段测试。输入来自 `803aec50248385b3e4170cc8cdaa9667035f7288` 原始 T1 连续链的自产 MRI；不是官方中间结果。旧 FNIT 的 `T1.mgz` / `brain.mgz` 仅在新计算完成后读取作比较。

## 测试范围

- `allocator_abba_v2`：原算法的缓存关闭→启用→启用→关闭，共12次完整文件 API。
- `neighbor_abba_v4`：缓存关闭下，CPU邻域→GPU邻域→GPU邻域→CPU邻域，首次归一化两例、第二次归一化 sub06，共12次。
- `neighbor_sub07_aseg_v6`：sub07 当前自产第二次归一化输入完成后，用相同冻结源码补齐四次配对。
- `candidate_memory_v5`：相同候选启用缓存的两个单次诊断，只用于读取可用的 PyTorch allocated/reserved，不用于宣布缓存提速。
- `aggregate.json`：实际 API 秒数、每轮输入/控制图 SHA 一致性、输出比较和版本绑定。每次完整原始报告与汇总 CSV 都保留。

测试在同一 A100-SXM4 80GB、显式 `cuda:1`、CPU亲和4–7和四线程预算下进行，默认TF32，没有FP16/BF16。GPU邻域采用 double 累加后存储 float32，保持原 SciPy 邻域算子的语义。源图和缓冲在当前 ROI 内复用，CPU动态控制图仍逐轮反馈，不进行独立轮并行。

## 计时和验收

`full_API.seconds` 包括读取、校验、传输、首次函数 JIT/缓存加载、原两轮迭代和压缩写出，并在外层同步目标设备。SHA收集、结果比较、解释器导入和CUDA初始化不在此列。另存的 `process_time.json` 包括整条诊断进程及比较/采样收尾，不能当作纯生产CLI冷启动时间。

每个输出比较不同体素数、最大/P99误差、shape/dtype、affine、MGH头和完整文件SHA；每轮 float32 输入与 uint8 控制图 SHA 和选择计数另外比较。本轮规则为该算子输出0差异，不降低阈值。当前官方冻结源码参考仍见[归一化功能页](../../../../docs/recon_all/NORMALIZATION.md)；本目录没有新生成官方同输入输出，不替代官方重复性或整例验收。

第二轮未修改的 ridge/初始偏场耗时波动较大，必须保留四次实测，不能仅凭中位值认定稳定提速。原始 T1 整例耗时、138项严格诊断、最终脑区指标及整体等效由完整运行单独报告。

## 显存和共享负载

关闭CUDA缓存时，allocated/reserved不可用，报告为null。容器PID到宿主GPU进程的归属未确认，进程树显存同样为null；整卡上界含其他任务，不能当成FNIT实际占用。保留请求间隔0.5秒、实际最大间隔、失败样本和整卡记录。启用缓存的额外单次报告提供本进程PyTorch分配统计，但没有证明原始T1整例合计显存低于20,000,000,000字节。

## 复现与来源

运行 [reproduce.sh](reproduce.sh) 前配置已授权的自产检查点、旧/新冻结源码、环境Python和新输出目录。脚本不下载数据、不运行参考软件、不写入原检查点。功能、全部参数、具名示例和公开脑图见[中文说明](../../../../docs/recon_all/NORMALIZATION_GPU_NEIGHBORS.md)。

`public_export_manifest.json` 记录私有原件和去路径发布件的SHA；`$FNIT`是统一私有工作根的占位符，主机名替换为 `A100_BENCHMARK_HOST`，数值、输入/输出、源码SHA均保留。未发布MRI文件、许可证、凭据或服务器地址。

数据来源：[OpenNeuro ds000114](https://openneuro.org/datasets/ds000114)，CC0。脑图只显示该公开输入的自产归一化结果与误差；切片为原网格第三轴中间位置，没有重采样为标准解剖方向。
