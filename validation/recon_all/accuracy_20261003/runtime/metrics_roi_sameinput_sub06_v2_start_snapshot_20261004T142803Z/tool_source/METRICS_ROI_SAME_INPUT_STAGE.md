# 同输入左侧顶点指标与脑区统计阶段诊断

## 功能与当前范围

使用已完成3a sub-06的自产surface RAS/mm white/pial、有序面和注释，比较冻结8f与3a的实际成熟函数。每臂独立新进程执行cold一次、同进程warm一次；只运行左半球顶点图和六套ROI表，不运行原始T1整例，不读取官方结果生成候选，不重试算法错误。两次重放各新建SurfaceStatsCache，在每次六表内复用；warm表示同进程、OS/Python/CUDA已调用，并不意味着沿用上一遍ROI张量缓存。

流程：验证10输入SHA → 双锁和GPU1空闲≥20,000,000,000字节 → 复制到新目录 → 成熟metrics函数 → 六ROI表 → 保存数值数组 → 同进程第二遍 → 原输入/复制white/pial不变检查 → 回收owned进程树 → 释放锁。

## 输入、参数及输出

`replay_metrics_roi_stage.py` 参数：

- `--checkpoint`：已完成自产subject目录，只读。10文件为white/pial、cortex.label、五个annot、brainvol.stats与talairach.xfm；manifest逐项绑定大小/SHA。
- `--input-manifest`：JSON，`files`必须恰含上述10项，不能以准备记录代替已完成执行。
- `--assets`：现有FNIT资产根路径。CUDA metrics直接调用成熟PyTorch算子，不运行native metrics二进制。
- `--output`：独立新目录；包含subject副本、report.json、cold/warm metrics.npz、六个stats文本及按名字保存的ROI npz。
- `--gpu-uuid`：完整物理GPU UUID；必须与CUDA_VISIBLE_DEVICES sole UUID一致，并核验实际cuda:0 device UUID。

report记录输入/复制SHA、解释器/Torch/nibabel、FP32与TF32、loaded_source SHA、坐标/有序面、逐函数墙钟和父/子CPU、CUDA同步、内部thickness/area/curv秒数、每遍ROI cachecounter、输出dtype/shape/有限性及文件SHA。全入口从输入验证开始，包含复制/加载/写出；算子阶段由已有StageProfiler同步计时。第一遍cold可能包括CUDA上下文首次启动，与整例在后期已初始化的父进程历史不同，不用于直接补算整例时间。

外层 `metrics_roi_dual_lock_controller.py --config /绝对路径/A8f.json` 接收已有stage schema，新增 `benchmark_kind=metrics_roi_stage` 与 `gpu_lock=/tmp/fnit-stage-gpu1-benchmark.lock`，固定 `lock=/tmp/fnit-shared-benchmark.lock`、threads=4和最低20GB。复用stage_benchmark_controller与resource_admission的owned树追踪/取消/清理，run_monitored以0.5秒采样；显存不足时释放共享锁继续等待。算法进入后失败停止队列。

`run_metrics_roi_pair.py --run /规范run路径 --controller /规范workspace路径/metrics_roi_dual_lock_controller.py` 顺序运行A8f、B3a，用户取消传至controller回收，任一臂失败不启动下一臂。 本地更新补齐 Popen 返回前取消的信号窗口：新 controller 赋值后若已收到取消，立即发送 SIGTERM 并等待其 owned-tree 清理退出，B 臂不启动。当前服务器冻结 v1 尚未热替换；正在运行的工具保持原字节，本修复供后续独立冻结版本使用。

## 本轮现场配置

- 工具：`FNIT/workspaces/recon_accuracy_20261003/metrics_roi_sameinput_tools_v1`。
- 产物：`FNIT/runs/recon_accuracy_20261003/metrics_roi_sameinput_sub06_v1`。
- 原输入：`FNIT/runs/recon_accuracy_20261003/precision_candidate_3a_whole_sub06_v1/attempt_01/subject`，complete/138。
- GPU1：`GPU-e25cac06-0ce8-a833-abf9-09ab18c9c9ba`，逻辑cuda:0，总线程4、默认TF32、无half/autocast、allocator disabled。
- source：8f3e51f58c37d28e1283d5bb8e248e93530cd398 与3a0c9aba6321b4981fd8174b4b191515459aa38b，不改冻结目录。
- 队列PID28336。最初现场状态为A8f waiting_resources，GPU1空闲34.56GB但共享锁未取得，Torch/GPU尚未执行。该状态不等于完成；最终以queue/controller/report回执为准。

## 数值解释

同输入white/pial与有序faces不变，所以各morph可按索引比较。沿用 `docs/recon_all/SURFACE_METRICS.md` 既有门槛：area绝对0.001 mm²，其余成熟thickness/curv绝对0.005，另加0.001相对项；该门槛不扩展为未经定义的area.mid/TH3volume整体等效判定。ROI以同脑区名字和九列对应，报告逐列差异，不要求字节相同：3a ROI面积修正官方逐面float32分摊后double累加的定义，旧版先float32顶点舍入；未为此新设或事后放宽阈值。六表格式仍按成熟函数精度写出，保存数组记录该文本已舍入的语义，不冒称未舍入底层值。

## 原软件对应与参考

对应FreeSurfer `mris_place_surface --thickness WHITE PIAL 20 5 OUT`、`--area-map SURFACE OUT`、`--curv-map SURFACE 2 10 OUT`；ROI对应`mris_anatomical_stats -no-th3`。本轮只调用FNIT既有Python/PyTorch实现，原软件命令是定义参照。源与引用沿用项目 `SURFACE_METRICS.md`、成熟anatomical_stats函数说明及[FreeSurfer官方源码](https://github.com/freesurfer/freesurfer/tree/d932c45b7941662ea380a05efef580568b98d41a)。新测试不建立整例等效门槛，不将阶段观测宣称整例提速或五项改动的因果代价。
