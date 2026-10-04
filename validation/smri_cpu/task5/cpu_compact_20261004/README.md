# GEMS CPU 合并查找与固定点缓存

## 改动

GEMS 使用四面体图谱拟合核团。原 CPU 路径对每个 compact 批次启动一次所有者查找，并重新拼接固定体素坐标。本版把同一次拟合的全部真实点合并为一次 CPU Numba 并行查找，缓存坐标、候选表和整数任务顺序。查找结果仍交给原 PyTorch 插值、成本和梯度计算。

候选顺序、FP32 FMA、首次平局、NaN、奇异四面体与边界容差保持。缓存随原张量版本、批次对象、四面体数量或返回坐标变化重建。索引刷新产生新的缓存。CPU autocast、非 FP32 或 owner hint 模式使用原路径。CUDA 保留原分支，不导入新增 CPU 模块。本版没有改精度、图谱、拟合步数或停止规则。

公开 API、CLI 和输入输出结构不变，见[亚区功能页](../../../../docs/subregions/README.md)。新增模块是内部实现，不需要用户额外指定参数或下载资源；复用项目已有 Conda 环境中的 Numba。

## 真实查找验收

在 nodecw10、8 个物理核 `34,38,42,46,50,54,58,62` 上回放公开 T1 的真实图谱拟合数据。原实现捕获完整坐标、当前四面体矩阵和输出；回放逐值比较 selected ID、point 与 coverage。首次 JIT 和缓存准备另列，warm 顺序为旧、新、新、旧。

| 真实查找范围 | 批数 / 点数 | 旧 warm 两次（秒） | 新 warm 两次（秒） | 逐值结果 |
|---|---:|---:|---:|---|
| 既有原始查找 trace | 6 / 119,232 | 0.05792 / 0.05163 | 0.03183 / 0.03344 | 全相同 |
| 完整粗分割查找 | 138 / 210,348 | 1.41161 / 1.40560 | 0.94882 / 0.91106 | 全相同 |
| 完整强度拟合查找 | 253 / 2,005,240 | 2.05970 / 1.99241 | 1.59243 / 1.66615 | 全相同 |

机器记录：[119k](lookup_119k.public.json)、[完整粗分割](lookup_coarse_complete.public.json)、[完整强度拟合](lookup_intensity_complete.public.json)。这些是组件诊断，不是完整分割 benchmark；两次真实捕获均在取得所需 trace 后主动结束，没有完整 pipeline 输出。局部中位时间分别改善约 1.68、1.51、1.24 倍，不替代整体速度结论。

局部合同测试共 28 项通过，覆盖上述离散规则、缓存失效、CPU autocast/FP64 回退、线程预算异常恢复，以及原 PyTorch priors 和 vertex/alpha 梯度逐值一致。

## 完整 stage 联合候选

完整 stage 使用相同公开 norm/aseg/wmparc 检查点、同图谱、brainstem `fast` 配置，显式保存高分辨率标签和后验。为避开仍在运行的海马任务，本次旧/新 CPU 配对使用 nodecw10 的同一 8 个物理核 `0,4,8,12,16,20,24,28`，串行获取同一 CPU 锁。原软件旧对照使用 `32,36,40,44,48,52,56,60`，不作为本次提速的分母；原等待队列通过 stop-file 在执行前暂停，没有改变其冻结源码或已有记录。

完整候选同时加入 CPU 固定 log-prior 缓存，文件由主任务维护。完整耗时应称为**联合优化**，不单独归因于合并查找。[冻结源码身份](source_identity.public.json)保存五个相关模块的 SHA-256 和全部 473 个 Python 文件清单的 SHA-256；正式运行不覆盖冻结源码。

完整旧/新新进程调用后，另存最终 vertices、Gaussian means/covariances、objective history、Jacobian 与 solver stats。比较原网格及高分辨率标签、21 通道后验、affine/zooms、labels/volumes 表与这些拟合参数；GPU 同输入也要求旧/新输出逐值相同、精度策略不变并记录显存。参数导出和评分在 API 计时结束后进行，冷 CLI 墙钟包含其开销。

**完整 CPU 和 GPU 回归均通过。** CPU 全部标签、完整 21 通道后验、几何、体积表、最终 vertices/Gaussian 参数、61 个 objective 值、solver stats 和最小 Jacobian 逐值相同，所有数组差异数、绝对最大误差及 RMSE 均为 0。

| 完整 CPU 调用 | 旧版本（秒） | 联合候选（秒） |
|---|---:|---:|
| 分割计算 | 726.88383 | 662.43540 |
| 常规产物保存 | 6.08822 | 6.27135 |
| API（包含通常的产物保存） | 733.02104 | 668.80116 |
| 冷 CLI 墙钟（另含参数导出） | 735.89981 | 671.73259 |

本次 API 比值为 1.0960，耗时减少 8.76%；只有一个完整配对，结论限于这一真实脑干 stage。记录见[完整 CPU 对照](cpu_stage_joint.public.json)。CPU 合并查找和固定 log-prior 缓存共同构成此结果。

新的旧版本输出还与此前 v1 完整结果比较：原网格和高分辨率标签及几何逐值相同，labels/volumes 表字节相同，见[既有输出核对](old_v1_output_check.public.json)。因此本次联合候选保留主报告中的官方分割精度：四个脑干亚区 Dice 为 0.994760、0.996973、0.995968、0.964871，硬体积差分别为 0.1014%、0.5056%、0.5604%、3.3333%，均满足每区 Dice ≥0.95、硬体积差 ≤5%。此前 v1 没有保存后验和 fit 参数，不能补称这些历史字段已经比较；本次新配对已实际保存并比较完整字段。

### 同输入 GPU 回归

H100 上相同默认 TF32、8 个 CPU 线程预算，旧/新各一次完整分割。原网格标签、高分辨率标签、完整 21 通道后验、几何、体积表和最终 mesh/Gaussian/objectives/solver stats 全部逐值相同。后验 shape 为 `144×194×156×21`，没有只选四个输出亚区。两臂自身 allocated 峰值均为 2,641,245,184 字节，reserved 峰值均为 3,940,548,608 字节，均小于 20 GB。

旧/新 API 为 36.10824 / 33.49710 秒，冷 CLI 为 40.12353 / 36.88312 秒；仅一次配对，记录实际结果，不据此声明 GPU 加速。新增 CPU 查找和 log-prior 缓存均在 CUDA 上保留旧计算路径。详细逐数组哈希与精度策略见[GPU 回归](gpu_stage_joint.public.json)。

第一次 GPU wrapper 在 allocator 初始化前重置峰值失败，两个 job 均未进入 GEMS；修复记录初始化后另开目录成功运行，原失败记录保留，不计作算法耗时。

## 复现工具

- `compact_lookup_replay.py`：输入私密真实 capture 目录，输出匿名 JSON；`--threads` 固定线程预算，`--report` 需新路径。
- `capture_complete_compact.py`：复用既有真实 worker；`--stage first` 捕获首次完整 compact，`--stage intensity` 保留粗拟合并捕获强度阶段首次完整 compact；`--capture-directory` 需新目录。均为主动截断诊断。
- `gems_stage_contract.py`：`--worker` 后跟既有正式 worker 全部参数，API 计时结束后导出拟合参数与 GPU 内存策略。
- `compare_stage_contract.py`：输入两臂完整产物目录，以 `--baseline`、`--candidate`、`--report` 指定；评分不计入正式时间。

```bash
FNIT_CPU_PYTHON=/absolute/path/conda/bin/python
FNIT_REAL_CAPTURE_DIR=/absolute/path/private/real_intensity_trace
FNIT_LOOKUP_REPORT=/absolute/path/new_lookup_report.json

# capture 必须来自真实 T1 的同一次完整查找，不能拼接不同 mesh evaluation
"$FNIT_CPU_PYTHON" validation/smri_cpu/task5/compact_lookup_replay.py \
  --capture-directory "$FNIT_REAL_CAPTURE_DIR" \
  --report "$FNIT_LOOKUP_REPORT" --threads 8
```

## 下一热点

完整强度 trace 含约 200 万个所有者。稳定梯度分支每轮还对这些 ID 执行 stable argsort；可评估按非负整数 ID 的稳定计数排序，保留完全相同排列及原 FP64 segment-reduce 顺序。该方案尚未实现或验收，不属于本次联合候选。

## 原软件与更新记录

原软件独立参考为 FreeSurfer 8.2 `segment_subregions`；对应调用、数据来源和参考文献见[主报告](../README.md)。FNIT 计算没有调用原软件。

- 2026-10-04：新增合并 CPU 查找和固定点缓存；三组真实 trace 逐值通过，包含固定 log-prior 缓存的完整 CPU 联合候选由 733.02 降至 668.80 秒，完整 CPU/GPU 输出及拟合状态逐值一致。
- 前一版本：每批 CPU Numba owner lookup 已通过 119,232 个真实点和完整脑干标签门；完整 CPU stage 仍慢于官方。

原实现：[FreeSurfer SAMSEG 亚区源码](https://github.com/freesurfer/samseg/tree/2ce2b6be69f2954ea704e593a5be79c284a3a8c3/samseg/subregions)。本模块保留原 PyTorch 插值和求解，未新增原软件代码副本。
