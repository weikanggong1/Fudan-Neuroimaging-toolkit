# GPU 球面重采样与左右半球并行：完整 surface 验证

## 范围与起点

本轮比较冻结基线 `954ad19` 与新实现 `9f9f63e95b4ee702e4c3024407496df47a851631`，分别运行旧版串行、新版串行与新版左右并行。起点是同一例已经完成的 T1w/MNI preproc BOLD 和同源已有 recon-all/graymid，完整保留 **490 帧、TR 0.735 s、STC 关闭**。服务器上两份 BOLD 的完整 SHA-256 与 [7102c187 完整 surface 基准](../surface_e2e/README.md)的起点一致。

每次使用新进程和新输出目录，完整调用 `fMRISurface_pipeline`：核对来源及重建身份，独立准备几何与 ROI、重新估计双侧四级 MSMSulc、生成对应面积表面，投影全部时序、组装 CIFTI、执行 QC 并保存全部 11 个最终输出。排除既有 recon-all、前序 volume、部署与编译；包导入、CUDA 初始化、输入捕获和运行后检查分开记录。MSMAll 保留独立配置和验证，本页使用默认 MSMSulc。

```mermaid
flowchart TD
    INPUT["同一完整 preproc、重建和 HCP 资源"] --> BASE["954ad19：旧版串行完整 API"]
    INPUT --> SERIAL["9f9f63e：新版串行完整 API"]
    INPUT --> PARALLEL["9f9f63e：新版双侧并行完整 API"]
    BASE --> FIRST["全部球面、几何、时序和科学配置比较"]
    SERIAL --> FIRST
    SERIAL --> SECOND["新版串行与并行全部数值、轴和 metadata 比较"]
    PARALLEL --> SECOND
    PARALLEL --> OFFICIAL["对已有独立官方单线程完整 surface 输出"]
    REF["相同起点的已完成 fMRIPrep/newMSM 参照"] --> OFFICIAL
    classDef default fill:#ffffff,stroke:#000000,color:#000000;
    linkStyle default stroke:#000000;
```

## 算法与并行合同

- GPU 最近邻检查相邻 27 格，用外部格距离下界证明候选；近并列或范围不足保留原 cKDTree 搜索。有序 CSR 保持原稀疏权重与累加规则。
- 原三角形成本、位移标签和球面变形仍在 PyTorch GPU 计算。严格 WLS、Rodrigues、源码精度回退、HOCR/FastPD 和顺序展开保留有序 CPU 运算；原生算子处理独立模型时释放 GIL。
- `parallel=True` 时左右独立执行，在所选 GPU 上使用两个 stream；两侧及其 stream 结束后按固定 L/R 次序汇总，随后执行 CIFTI/QC/最终发布。
- `cpu_threads=8` 是局部搜索和 Workbench 的总预算，并行时分为 4/4；预算 1 自动串行。PyTorch 使用调用方已有的全局 intra-op 线程池，库不在 worker 中修改它。
- Workbench 保留 ribbon、10 mm dilate、native mask、ADAP_BARY_AREA、atlas mask 和面积/ROI 准备。nibabel/NumPy 执行格式读写、CIFTI 组装和 QC；本轮不声称全部计算迁移到 GPU。

### 成熟子函数中的修复

1. `run_msmsulc` 与 MSMAll 原先逐侧重置全设备 CUDA 峰值，影响调用方完整 API 的统计。本轮移除内部重置，报告明确为调用方上次重置以来的 allocator 峰值；双侧共享该范围。真实驱动在外层统一初始化、重置和最终同步。
2. `_label_samples` 在粗采样网格或小搜索半径内没有非零标签时，原返回形状 `(0,)`，使后续 `vstack` 失败。本轮将空结果保留为 `(0,3)`；常规配置的标签数、顺序与数值不变。

3. `925c5866` 将 optimized CUDA 的 source-precision `RadialSphereMap` 主机缓存延迟到 containment 不确定、边界重叠或缺失时。已证明的点不再复制整批 query/nearest；CPU、reference、fallback 和 FP64/native 算术顺序保持不变。远端 Conda 环境的 `tests/test_msm_sphere_execution.py` 与 `tests/test_msm_sphere_cpu.py` 为 14 passed、5 skipped。

两项均由共享 MSMSulc/MSMAll 控制测试覆盖；算法说明同步到 [MSMSulc 功能页](../../../docs/msm/README.md#左右并行与资源预算)和 [surface 功能页](../../../docs/fmri/surface.md#实际计算设备)。

## 数值门禁与官方参照

新版串行先与冻结旧版比较，新版并行再与新版串行比较。比较全部原生注册球面、准备几何/ROI、有效科学配置、左右 GIFTI 和 91k CIFTI。每侧 GIFTI 为 `490×32,492`，CIFTI 为 `490×91,282`；检查 float32、全部有限、原始 TR、时间轴、灰坐标顺序、21 个结构与内嵌 metadata。恒定序列仍参加逐值误差，时间 Pearson r 只汇总非恒定配对；不拟合强度、变换或追加平滑。

CPU 线程数与执行安排不作为科学配置差异；配置的实际浮点值、标签顺序和停止条件必须一致。文件 SHA 与解码数值分别核对，因为临时路径 metadata 可改变 GIFTI XML 字节。

官方参照复用已完成的 **fMRIPrep 25.2.4 / sMRIPrep 0.19.2 / newMSM 单线程**完整 surface 输出，独立准备几何和估计双侧球面。本轮只恢复正确 host 路径清单，核对原保存的输入、配置、输出和工具 SHA；不重跑官方，也不以 FNIT 捕获几何替换其实际准备结果。双方仍使用同一既有 volume 和重建起点；该范围不包含原始 BIDS 到 volume/recon-all 的独立全流程。

新版串行对旧版、并行对新版串行的七项门禁全部通过：注册球面、左右 GIFTI 与完整 CIFTI 解码数值逐值相同，最大绝对误差和 RMSE 均为 0；时间轴、21 个结构、内嵌 metadata 和有效科学配置相同。GIFTI XML 中的临时来源路径可以不同，因此另行比较文件 SHA 与解码数据。报告见[串行对基线](serial_vs_baseline.public.json)和[并行对串行](parallel_vs_serial.public.json)。

### 对独立官方完整 surface 的全帧精度

对已完成的独立官方严格单线程链重新读取全部 490 帧，没有拟合强度或追加平滑。时间 r 对每个非恒定时序计算；所有顶点和灰坐标，包括恒定/零序列，均参与逐值误差。

| 输出 | 有效时序数 | 时间 r 均值 / 中位数 | RMSE / relative RMSE | 最大绝对误差 |
|---|---:|---:|---:|---:|
| 左 fsLR32k | 29,695 | 0.979065 / 0.991411 | 146.5825 / 0.017778 | 2676.0938 |
| 右 fsLR32k | 29,716 | 0.953067 / 0.980258 | 258.1970 / 0.029083 | 3593.6602 |
| 完整 91k CIFTI | 91,281 | 0.977911 / 0.997039 | 177.1381 / 0.022418 | 3593.6602 |

CIFTI 共比较 **44,728,180 个值**，其中 **28,936,116 个不同**，MAE 为 77.3913；全部 19 个皮层下结构逐值相同，21 结构的灰坐标、时间轴和内嵌 metadata 相同。注册球面角差左/右均值为 **0.221050° / 0.319595°**，p95 为 **0.489326° / 0.704916°**。因此本轮执行优化保持旧 FNIT 的完整结果，独立官方的皮层与球面差异仍存在。详细聚合见[并行对官方严格参照](parallel_vs_official_strict1.public.json)；原版运行来源与耗时保留在[已完成官方报告](../surface_e2e/reference_strict1.public.json)。

### 共享 MSMAll 的真实回归

为核对共享配准核心对成熟 MSMAll 的影响，使用既有真实 C 特征和完整 coarse/refine 配置，重新运行旧版串行与新版并行。每个配置的 15 份现有输入在运行前后大小及 SHA 相同；双方科学配置相同，左右注册坐标、拓扑和 GIFTI metadata 全部严格一致，最大坐标误差与 RMSE 均为 0。

| 完整双侧 `run_msmall` | 旧版 `954ad19` 串行 | 新版 `9f9f63e` 并行 | 旧 / 新峰值 allocated，GB |
|---|---:|---:|---:|
| coarse 一级 | 23.237 s | 11.430 s | 0.0947 / 0.2194 |
| refine 三级 | 181.790 s | 90.796 s | 1.2126 / 1.4662 |

四次调用均为新进程、物理 GPU 0、局部 CPU 总预算 8、20 GB 上限，启动环境为 `CUDA_MODULE_LOADING=LAZY`。API 包含既有特征/球面读取、配准和结果写盘，排除特征估计、包导入、CUDA 初始化、哈希检查、BOLD 投影和 HCP 外层迭代。每种条件只测一次，不作为重复计时分布。

另读取历史保存的单线程官方 newMSM 球面：两种配置、双侧坐标与拓扑也逐值相同，但 GIFTI metadata 不同；历史未逐输入保存 SHA，因此仅作为保存结果回归，不称本轮新跑的官方同输入对照。完整配置、文件 SHA、编译来源和边界见[MSMAll 配对聚合](msmall_paired.public.json)，功能范围见[MSMAll](../../../docs/msm/msmall.md)。

这组 MSMAll 数字是完整双侧 `run_msmall` 注册核心的 GPU 记录，不是完整 `fMRISurface_pipeline` 端到端结果：当前真实 paired workspace 的 native sphere 网格与官方 SOURCE 特征网格不匹配，无法安全接入 BOLD 投影、CIFTI 和最终发布。匹配的真实特征资产补齐后，将复用本页的 490 帧、H100、20 GB 上限、CPU 总预算 8 和左右并行协议重新测量。

## 计时、环境与来源

旧版串行、新版串行和新版并行均完整成功并保存 11 个最终输出。新版并行最初在物理 GPU 1 的 CUDA 初始化阶段失败，未进入 API；随后用物理 GPU 0、新进程和新目录完成测量。以下均为实际成功运行的连续墙钟：

| 同一 490 帧输入 | 旧版 `954ad19` 串行 | 新版 `9f9f63e` 串行 | 新版 `9f9f63e` 并行 |
|---|---:|---:|---:|
| 物理 H100 | 1 | 1 | 0 |
| API 墙钟，含捕获及最终保存 | 436.282 s | 343.093 s | 243.729 s |
| 私有输入捕获 | 0.03790 s | 0.03743 s | 0.03424 s |
| 扣除捕获的完整 API | **436.245 s** | **343.056 s** | **243.695 s** |
| MSM 准备与双侧配准 | 202.029 s | 116.361 s | 94.773 s |
| 双侧投影外层墙钟 | 原驱动未单列 | 190.136 s | 116.029 s |
| CIFTI 组装 | 24.722 s | 21.935 s | 22.953 s |
| 峰值 allocated / reserved，十进制 GB | 0.344 / 0.426 | 0.644 / 1.059 | 1.049 / 1.449 |

[旧版实际报告](baseline.public.json)、[新版串行报告](serial.public.json)和[新版并行报告](parallel.public.json)绑定各自实际源码、驱动、输入与输出。API 时间排除运行后哈希/数值检查；阶段计时为嵌套范围，不通过各阶段求和构造完整墙钟。MSM 行包含输入准备，不能替代独立 `run_msmsulc` 的计时。

新版串行与并行的双侧执行计数相同：累计最近邻查询 **266,058,840 次**，GPU 证明最近邻 **266,048,224 次**，回退树搜索 **10,616 次**，其中近并列回退 **10,513 次**；累计有序 GPU 特征重采样 **16 次**，CPU layout 回退为 **0**。这是重复迭代的调用计数，不是独立顶点数；逐侧计数保存在实际报告中。

使用共享 H100、总局部 CPU 预算 8、CUDA allocator 上限 **20 GB（十进制）**、TF32 开启，不使用 float16/bfloat16。双侧并行以连续外层墙钟计时，不能用左右阶段耗时相加代替。记录物理 GPU、运行前后共享负载和实际退出状态；一次观测不推广为稳定加速比。

并行所用 GPU 0 在运行前/后的总占用为 42,420/43,950 MiB、利用率为 0/55%；旧版与新版串行所用 GPU 1 运行前利用率分别为 24/0%。[实际启动映射](execution_mapping.public.json)绑定物理卡号、`CUDA_VISIBLE_DEVICES`、三次真实 exit 0 与原始 API 报告 SHA。三次调用的卡号和共享负载不同，因此表中只报告实际观测。GPU 1 启动诊断先复现 `torch.cuda.mem_get_info` 的 OOM，随后 LAZY 小控制与普通小控制均成功；没有定位其暂态原因，也没有把它归因于配准计算，见[启动诊断](cuda_startup_diagnostic.public.json)。

主页 [environment.yml](../../../environment.yml)已包含 Python 3.11、PyTorch 2.5.1/CUDA 11.8、Triton 3.1、nibabel、NumPy、SciPy、Workbench 2.1.0 和 C++ 工具链，本轮不引入新依赖。Linux native 扩展按 [setup.py](../../../setup.py)使用 `-O3 -std=c++17 -fno-fast-math -ffp-contract=off`；在目标服务器重新编译，不使用本地旧 ABI 的 `.so`。

公共记录分别保存实际运行的 git revision、各 runtime 源文件 SHA-256、验证驱动 SHA、native 二进制 SHA、编译器和实际编译参数、依赖版本、资源/输入/输出哈希。[服务器编译证明](build_provenance.public.json)记录 GCC 11.2.0、实际严格编译参数和两版 native SHA；三份 API 报告与其逐项匹配。源码 SHA 与实测结果绑定，后续文档提交不改写已执行版本。测试控制与真实 490 帧 benchmark 分开记录。

[发布源码回溯](publication_runtime.public.json)核对本次实测新版的 **116 个 runtime 文件和验证驱动**与最终工作树一致。控制测试分别为本地 MSM 202 项、surface 108 项、科学配置比较 4 项，以及重新编译 native 后的 H100 101 项；这些属于不同执行与覆盖范围，分别记录，不相加声称新的整链 benchmark。

## 复现完整调用

先将同一已验证 volume 的输入和 JSON 放入三个新的 derivative 根目录，保留其逻辑来源；目录内不应已有 surface 输出。冻结两版源码，并在目标主页 Conda 环境编译各自 native 扩展。实际旧版测量使用 `954ad19` 驱动，新版使用 `9f9f63e` 驱动，各自 SHA 已写入报告，捕获没有修改计算。下面给出统一复现方式：新版驱动通过函数签名识别旧版不支持的并行参数，旧版保持原串行行为；该适配不改旧版数学计算。

```bash
baseline_source_root=/absolute/path/fnit_954ad19          # 旧版冻结源码和目标机器编译产物
candidate_source_root=/absolute/path/fnit_9f9f63e         # 新版冻结源码和目标机器编译产物
benchmark_python=/absolute/path/fnit-conda/bin/python    # 按主页安装的同一 Conda Python
benchmark_bids_root=/absolute/path/bids                  # 与 volume 相同的原始 BIDS
benchmark_recon_all=/absolute/path/recon-all/sub-0001    # 同源已有重建，含 graymid/midthickness
benchmark_hcp_assets=/absolute/path/hcp_surface_assets   # 已校验固定 HCP 资源
benchmark_subject=0001                                  # 私有被试标签，不进入公共报告
benchmark_output_root=/absolute/path/new-validation     # 本轮三个新运行目录的根路径
benchmark_workbench=/absolute/path/wb_command           # 同一 Workbench 2.1.0

# 三个 derivatives 已分别放入相同完整 volume 和 JSON。
PYTHONPATH="$baseline_source_root/src" "$benchmark_python" \
  "$candidate_source_root/validation/fmri/benchmark_surface_e2e.py" \
  --bids-root "$benchmark_bids_root" --subject "$benchmark_subject" \
  --derivatives-root "$benchmark_output_root/baseline/derivatives" \
  --recon-all "$benchmark_recon_all" --hcp-assets-dir "$benchmark_hcp_assets" \
  --source-root "$baseline_source_root" --source-revision 954ad19 \
  --report-out "$benchmark_output_root/baseline/api.public.json" \
  --capture-dir "$benchmark_output_root/baseline/capture" \
  --wb-command "$benchmark_workbench" --device cuda:0 --threads 8 \
  --gpu-memory-limit-gb 20 --serial-hemispheres

for hemisphere_mode in serial parallel; do
  serial_option=()
  if [ "$hemisphere_mode" = serial ]; then serial_option=(--serial-hemispheres); fi
  PYTHONPATH="$candidate_source_root/src" "$benchmark_python" \
    "$candidate_source_root/validation/fmri/benchmark_surface_e2e.py" \
    --bids-root "$benchmark_bids_root" --subject "$benchmark_subject" \
    --derivatives-root "$benchmark_output_root/$hemisphere_mode/derivatives" \
    --recon-all "$benchmark_recon_all" --hcp-assets-dir "$benchmark_hcp_assets" \
    --source-root "$candidate_source_root" --source-revision 9f9f63e \
    --report-out "$benchmark_output_root/$hemisphere_mode/api.public.json" \
    --capture-dir "$benchmark_output_root/$hemisphere_mode/capture" \
    --wb-command "$benchmark_workbench" --device cuda:0 --threads 8 \
    --gpu-memory-limit-gb 20 "${serial_option[@]}"
done
```

使用 [compare_surface_e2e.py](../compare_surface_e2e.py)逐值比较各自私有 `capture/outputs.private.json`。公共 JSON 保留统计与哈希，私有 manifest 保留实际路径；正式发布前校验其源代码、native 编译和运行退出记录。

## 公开范围

本轮只发布匿名聚合指标和验证报告：形状、帧/TR、配置、时间、峰值、误差、覆盖和校验和。原始影像、临床数据、被试标识、绝对路径、逐顶点坐标/时序数组及 private manifest 保留服务器私有。沿用已有公开脑图时保留原版实测标签；本轮不新增个体脑图。
