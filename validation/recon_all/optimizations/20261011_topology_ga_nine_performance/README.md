# 九缺陷 GA 的在途性能诊断

## 1. 功能与范围

本目录记录既有完整九缺陷运行的计时与检查点，只读检查不会改变正在运行的源码、输入、候选、线程或随机流。第四缺陷的独立首候选重放另列；不是九缺陷完成、最终 `orig` 或原始 T1 整例报告。

流程：自身原始 nofix/qsphere/MRI → 九缺陷逐序搜索与最终写回 → 五轮 SOAP → 压缩 CURRENT → 有序 CSR 相交清理 → 恢复 ORIGINAL/premesh → 标准 remesh 三轮 → ORIGINAL 相交与最终 orig 门。当前报告快照停在第四缺陷搜索，后三步尚未启动。

## 2. 输入与输出

- `existing_timer_snapshot.json`：既有评分、每缺陷完成回执、检查点 SHA、采样资源及冻结源核验；只包含数字、公开数据输入哈希和必要的源哈希。
- `dispatch_audit.json`：旧冻结 v3 观测包装与后来显式 `face_detection_cache` 接口的调用顺序核对。两种源码身份分别绑定，不将正在运行的旧包装改标为新接口实测。
- `metrics.json` / `profile.json`：已完成第四缺陷首候选独立重放的白名单指标和 cProfile 表；`EXPORT_RECEIPT.json`绑定原私有报告及全部公开 payload。
- 独立首候选输入为自身第三缺陷完整 typed NPZ 检查点与相同 nofix/qsphere/brain；保留全部属性、逻辑面、容量、原顺序邻接和角点，不从参考后态补字段。校准和全局先验由原输入重新生成。
- 冷、热和 cProfile 重放都选择原 `population_index=0` 首候选 identity，实际没有随机抽样。各次从同一完整初态独立准备；不会推进原搜索。
- 显存 allocated/reserved 与无法归属的父峰值记录为 `null`。CPU 隐藏 CUDA，不用零代替 unavailable。

## 3. 调用

诊断工具 `tools/benchmark_recon_topology_fourth_candidate_profile.py` 接受具名参数：

```python
from pathlib import Path
from benchmark_recon_topology_fourth_candidate_profile import run

run(
    input_surface=Path("own_subject/surf/lh.orig.nofix"),  # 自身原始面
    input_qsphere=Path("own_subject/surf/lh.qsphere.nofix"),  # 同有序面/顶点
    brain=Path("own_subject/mri/brain.mgz"),  # 自产归一化 uint8 MRI
    state_checkpoint=Path("own_ga/defect-002.private.npz"),  # 第三缺陷完整写回
    producer_progress=Path("own_ga/progress.jsonl"),  # 核对所属缺陷和 SHA
    frozen_source_manifest=Path("diagnostic_source/SOURCE_MANIFEST.private.json"),
    expected_source_manifest_sha256=diagnostic_manifest_sha256,
    expected_checkpoint_sha256=own_checkpoint_sha256,
    output=Path("new_diagnostic"),  # 必须不存在
    defect_number=3,  # zero-based：第四缺陷
    threads=4,  # 独立四核预算
)
```

CLI 同名参数使用连字符，例如 `--state-checkpoint`、`--expected-checkpoint-sha256`。所有参数必填；该固定协议只接受缺陷编号 3 和总线程 4。完整重放命令与私有路径留在服务器新工作区；不发布 MRI、内部地址或日志。

## 4. 对应原步骤

这是 `mris_fix_topology -ga` 内部候选 `mrisDefectPatchFitness` 的诊断，不存在独立官方候选 CLI。本轮计算复用 FNIT 自有 NumPy/Numba；不启动原版阶段程序。

## 5. 现有计时与实际门

快照 UTC `2026-10-10T19:57:43`，冻结 v3 共 **856** 个 Python 文件全部 SHA/大小保持。三个完成检查点均与各自进度记录相符；完整九缺陷 EXIT/COMPLETE 不存在。strong orig 只有 WAITING，`algorithm_started=False`。

| zero-based 缺陷 | 已保存评分数 | 单评分均值/秒 | 去首调用均值/秒 | 完整缺陷 API/秒 |
| --- | ---: | ---: | ---: | ---: |
| 0 | 496（495搜索＋final） | 1.417535 | 1.388751 | 1208.789674 |
| 1 | 332（331搜索＋final） | 0.994243 | 0.995567 | 676.677215 |
| 2 | 452（451搜索＋final） | 0.932792 | 0.933735 | 939.237140 |
| 3 | 203（在途） | 4.447550 | 4.442266 | 未完成 |

第四缺陷首调用后仍较慢，已有日志不能把持续耗时解释为首次 JIT。该快照旧 observer 只有每评分 fitness/总秒数。随后独立首候选重放已完整退出0，补齐成熟函数的内部计时：同自身第三缺陷检查点 `94df28a5…`，17,213边、1,428inside、162hull、局部MRI51×53×54。冷、暖、cProfile三次完整72字段的shape/dtype/bytes与NPZ SHA均相同；double fitness均为−119.29359245300293，结果保存之后与原在途第四首评分逐字节相同。首identity实际0随机draw。

| 独立重放边界 | 实际秒数 |
| --- | ---: |
| 完整新解释器 CLI（含全部准备/三次/读写） | 146.671383 |
| raw nofix/qsphere/MRI → 自产基底 | 51.894878 |
| 自产全局先验/共享采样 | 9.783787 |
| 当前第四缺陷边/局部先验 | 5.292340 |
| 普通暖 candidate cycle | 6.573730 |
| 普通暖 candidate geometry | 5.082906 |
| 普通暖 retess+detect | 4.771748 |
| 普通暖25 smooth /40 match | 0.058679 /0.106417 |
| 普通暖完整法向 | 0.039330 |
| 普通暖完整五项评分（其中距离） | 0.534501（0.520358） |
| 冷 cycle /cProfile cycle | 19.322134 /9.013904 |

cProfile包含观测开销，不能将其9.01秒当暖吞吐。在此观测内，`retessellate_ordered_edges`自身3.501737秒/累计6.171228秒，已有Numba `_edge_intersects`仍被Python循环调度 **2,656,862次**，累计1.731605秒；全N tuple→list独立输出复制0.866550秒。后者须保留，不能为了减少计时跳过完整返回状态。`_candidate_state`累计1.707679秒，距离 `_distance`自身0.459655秒。主要可复用优化点是把完整首阻挡扫描移入已有谓词的Numba调用域，保持全部query/邻接、源首blocker、动态加删边及nearzero规则。

该诊断CPU84–87/总4线程、CUDA隐藏，峰值RSS2,057,604KiB；前后容器累计限流计数/时间均未变化。这不证明原九缺陷过去运行期间没有限流，也不将独立首候选计时直接除以在途某轮得到稳定速度比。

当前进程为运行态，4 线程，亲和性 8–11；累计用户 CPU 4046.48 秒、系统 25.14 秒、经过 4074.0 秒。既有一秒采样含 4077 次 R、4 次 D；不能由少数 D 判磁盘故障或死锁。只读当前 v1 cgroup 的 quota/period 为 4000000/100000 微秒，即容器 40 CPU 配额；累计 `nr_throttled=8808`，旧采样未记录 v1 增量，不能证明第四缺陷的限流因果。

早先完整首缺陷 state-context 实验的 **497 次装配合计 303.362309 秒**、中位 0.612706 秒属于另一独立版本和首缺陷；其暖评分中位反而慢 0.8168%。不把这组数字移用为当前第四缺陷的装配耗时或收益。

## 6. 更新与下一门

- v3 完整九缺陷仍执行；检查点、原有失败启动收据与 strong orig 等待器保持。
- 后来显式接口的实际六评分配对已验证 81 字段、控制和 20877 个随机量；不能据此称当前 v3 使用该源码，也不能替代新接口跨缺陷实测。
- 最小后续接口门是新显式 `repair` 源的两个连续真实缺陷：每次用改变后的保留面重建一个 context，候选与最终写回共享；核对各缺陷全状态和控制。当前 v3 的全九 context 记录须等最终 `repair_cache.json` 写出才能验收。
- 第四首候选独立阶段已经完成：全部72字段及fitness精确，自身重复精确；源manifest `721a8fa4…`、实际tool `7c214d74…`保持原绑定。随后新扫描优化用独立新冻结验证，不将此报告改标新source。
- NumPy外层阻挡扫描调用成熟 `_edge_intersects`；该谓词内部按已验证规则转float64计算，不能将此处称为全F32或换成fastmath。25/40次迭代、悬空邻接和全N返回copy仍保留。

## 7. 参考

- [FreeSurfer 固定代码库](https://github.com/freesurfer/freesurfer)：`mris_fix_topology` 与 `utils/mrisurf_defect.cpp`。
- 本仓库 `TOPOLOGY_GENETIC_FACE_CACHE.md`、`TOPOLOGY_GENETIC_STATE_CACHE.md`、`TOPOLOGY_CANDIDATE_PIPELINE.md` 的对应版本门。

总体指标等效、原始 T1 整例和最终网格验收均为 `not_assessed`。本目录的在途快照不是完成回执。
