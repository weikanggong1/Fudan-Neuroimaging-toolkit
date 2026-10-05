# Volume 与 helper 的完整真实数据 CPU 验证

## 1. 功能与流程

本目录保存隔离的 benchmark 适配器和聚合报告。范围包括完整 volume、FEAT core、Gaussian 高通／强度缩放、可选 slice timing、T1w 采样参考、Otsu mask 和旧版 motion-only 重采样。输入清单、每个参数及参考程序见 [准备清单](PREPARATION.md)；生产实现用法和完整结果见 [helper 官方对照](../../../docs/fmri/CPU_HELPER_BENCHMARK_20261005.md)。

```mermaid
flowchart LR
    A[私有完整真实输入与冻结源码] --> B[同核同线程串行执行]
    B --> C[正常读写的 API 时间]
    B --> D[完整输出与来源记录]
    D --> E[计时外全网格数值审核]
    C --> F[匿名聚合报告]
    E --> F
    classDef default fill:#fff,stroke:#000,color:#000;
    linkStyle default stroke:#000;
```

## 2. Python 调用、输入与输出

生产 Python 示例及所有输入／输出参数见 [helper 文档第二节](../../../docs/fmri/CPU_HELPER_BENCHMARK_20261005.md#2-python-调用输入与输出)。benchmark 私有 JSON 清单包含完整原始 BIDS、固定中间影像、合法权重与模板的服务器路径；路径和个体影像不进入公共报告。公共 JSON 仅包含尺寸、dtype、SHA、时间和聚合误差。

## 3. 命令行调用

```bash
# 数值分析不重新运行生产或原版算法；使用已完成的完整调用。
python validation/fmri_cpu_20261004/task05_volume/compare_helpers_v2.py \
  --run-root /absolute/path/private/task05_volume \
  --source /absolute/path/frozen_baseline \
  --inputs /absolute/path/private/inputs.json \
  --float-input /absolute/path/private/raw_bold_float32.nii.gz \
  --prior-report /absolute/path/private/comparison_v1.public.json \
  --output /absolute/path/private/helper_float32_protocol.public.json
```

`--run-root` 包含 baseline_v3、reference_fixed_v2、candidate_stc_matched_v5 和 reference_stc_float32_v5 的原回执；`--source` 为未修改基线；`--inputs` 定位实际完整影像及元数据；`--float-input` 为无损数值转换的完整浮点控制；`--prior-report` 绑定原 int16 协议报告；`--output` 必须为新文件。`--stc-candidate-folder` 和 `--stc-reference-folder` 可显式指定新的同核 STC 对照目录。脚本核对实际输入 SHA、整个冻结 Python 源码、核预算、网格、TR、所有数值及 ignored 帧；同时完整比较新、旧 FNIT STC 的值和影像头网格，保留 13 项完整官方比较及三项旧协议指标。

## 4. 原软件调用

完整 FSL temporal、AFNI Fourier STC 和 NiWorkflows sampling-reference 命令见 [原软件协议](../../../docs/fmri/CPU_HELPER_BENCHMARK_20261005.md#4-原软件调用)。完整 volume 的原版参考使用固定 fMRIPrep 25.2.4 SIF，默认关闭 STC、禁用 recon-all／surface，生成 T1w 和 MNI preproc；它没有 FNIT 的后续 ICA-AROMA、高通、额外混杂回归。整链时间按这些范围分别报告。

原版 volume 的私有输入清单另声明 `official_fs_license`，只将合法许可文件的路径传入容器。适配器按清单选择相同 session 和 task；容器内的真实 payload 与外层启动器必须同时退出 0，不对 setuid 启动器使用 host strace。四个最终 FNIT CPU 队列各在同一核组配对 baseline／candidate；原版完整链保留其独立核组。它们在同节点采用相同 1／8 线程预算，原版与 FNIT 整链的物理核组不同。

## 5. 最新精度、时间与脑图

[完整 helper 结果和已有公开图](../../../docs/fmri/CPU_HELPER_BENCHMARK_20261005.md#5-完整真实数据精度时间与脑图)列出 CPU1／8 的完整 API、全网格误差与来源边界。[最终 13 项完整聚合报告](helper_float32_protocol_20261005.public.json)已完成：强度缩放与采样参考逐值相同；STC float32 RMSE 约 1.9×10⁻⁵，ignored 帧保持原值，新、旧 FNIT 完整数据和影像头一致。首轮 STC 输出类型不同；[旧协议报告](helper_int16_protocol_20261005.public.json)保留原 int16 取整差异，不代替 float32 数值控制。

[完整 volume 执行基线](baseline_execution_20261005.public.json)记录基线四组 first/cache-call 已完成。最新 main 已改变 robust BOLD reference，该旧记录不作为当前整链验收。本轮新个体影像与脑图不公开。

### 2026-10-06：fMRIPrep 25.2.4 完整体积预处理（CPU1/CPU8）

同一真实公开病例的原始 T1w 和完整 180 帧 BOLD，在固定 fMRIPrep 25.2.4 容器中分别使用 1、8 线程；nthreads 与 omp-nthreads 均按该预算设置。STC、fieldmap、recon-all 与 surface 关闭，dummy scans 为 0。两个预算各完成 first 和新进程 workflow-cache 调用，四次实际 fMRIPrep 进程及容器启动器均退出 0。原始输入大小及 SHA-256 与 FNIT 冻结测试相符。

| CPU 线程 | first 完整 wall | first payload wall | 缓存新进程完整 wall | 缓存 payload wall |
|---:|---:|---:|---:|---:|
| 1 | 19400.443 s | 19399.265 s | 1107.668 s | 1106.255 s |
| 8 | 2957.476 s | 2956.069 s | 453.581 s | 452.485 s |

完整 wall 含正常读写及外层启动，payload wall 仅记录容器内实际 fMRIPrep 进程。原版生成 T1w/MNI preproc 和混杂变量，执行其完整解剖模板注册流程（包括额外 MNI2009 注册），没有 PICA、ICA-AROMA 或 FNIT clean 后处理。FNIT 与原版采用相同 1/8 线程预算，但完整计算范围不同、实际物理核组不同；上述原版总时钟按自身范围列示，不计算同范围整链速度比。

原版缓存调用启动新进程并复用 workflow cache；FNIT warm 则在同一进程复用解剖缓存。每格为一次完整运行观察。

#### 原版保存 leaf 节点时钟

| 原版工作流分组 | leaf 数（各预算） | CPU1 leaf duration 合计 | CPU8 leaf duration 合计 | CPU8 活动区间并集 |
|---|---:|---:|---:|---:|
| BOLD reference 相关工作流 | 6 | 14.693 s | 14.764 s | 14.739 s |
| 运动校正工作流 | 2 | 22.516 s | 23.375 s | 23.375 s |
| 解剖脑提取相关工作流 | 53 | 2206.239 s | 435.447 s | 420.752 s |
| 解剖标准化工作流（含原版模板注册） | 13 | 15924.751 s | 2022.059 s | 2018.837 s |
| BOLD→T1w 配准工作流 | 13 | 78.085 s | 78.133 s | 78.133 s |
| native preproc 工作流 | 3 | 26.558 s | 9.494 s | 9.494 s |
| T1w preproc 工作流 | 4 | 22.002 s | 8.262 s | 8.262 s |
| MNI preproc 工作流 | 11 | 167.916 s | 78.034 s | 76.376 s |
| 混杂变量生成工作流 | 44 | 22.286 s | 24.277 s | 20.588 s |
| 其他原版节点 | 110 | 515.603 s | 197.173 s | 178.192 s |

每个预算的 259 个 leaf 来自 first/cache 共享 work 目录中的当前保存结果；按真实 runtime 去重，并排除 MapNode 父节点。缓存节点可能保留 first 的 runtime，因此不拆成两套调用阶段表。leaf duration 合计计入同时运行的节点；活动区间并集仅计算这些节点实际运行区间的覆盖时间，两者都不能替代完整调用 wall。起止跨度还含组内等待及跨调用间隔。

未记录 GNU time 的整体 user/system/RSS，不从 leaf 推导进程资源统计。匹配空间与输出层的科学比较单独记录。新个体影像、路径和原始命令不公开。

完整执行与保存节点见 [CPU1 原版报告](official_saved_nodes_cpu1_v2.public.json)及 [CPU8 原版报告](official_saved_nodes_cpu8_v2.public.json)。

### 2026-10-06：完整 volume CPU 优化前后对照

同一真实公开病例的原始 T1w 和完整 180 帧 BOLD，使用冻结基线 `6f624040` 和候选 `98019133`，分别运行 FNIRT/SynthMorph、CPU1/CPU8。每个预算的旧、新实现绑定同一组物理核心；PyTorch interop=1，STC 关闭。每份源码在同一进程运行一次 first（重新准备解剖结果）及一次 warm（复用已生成的解剖缓存），合计 16 次完整正常 API。

| 后端 | CPU 线程 | 基线 first | 候选 first | 基线 warm | 候选 warm |
|---|---:|---:|---:|---:|---:|
| FNIRT | 1 | 1351.524 s | 851.694 s | 832.104 s | 405.339 s |
| FNIRT | 8 | 748.712 s | 581.172 s | 611.512 s | 432.309 s |
| SynthMorph | 1 | 1710.378 s | 1243.552 s | 804.598 s | 411.036 s |
| SynthMorph | 8 | 793.044 s | 617.190 s | 595.666 s | 405.428 s |

本次观测中，first 的基线/候选时间比为 1.28–1.59，warm 为 1.41–2.05；每格各一次完整调用。API 时钟包含原始输入、完整计算和正常输出读写；导入、来源校验、锁等待及额外保存对照副本在 API 时钟外。进程时钟同时覆盖 first/warm 及这些外围工作，不能替代单次 API。

8 组旧、新配对及 8 组 first/warm 配对均完整比较十个科学输出，每组 **395,140,404 个值**：不同值数、最大绝对误差、RMSE 均为 0，dtype、网格、affine、qform/sform、加载后及原始存储头一致。所有四张 BOLD 输出均为完整 180 帧、float32、有限值，mm/sec 单位及 TR 经校验。真实核预算、完整源码和原始输入前后校验通过；科学配置只对基线缺失的 `confound_projection` 按既有 `orthogonal` 补齐。

本表记录 FNIT 优化前后的完整 API。官方 fMRIPrep 整链时钟见上表；preproc 与 FNIT clean 的后处理范围不同，精度在匹配输出层比较。

分步骤时钟见 [功能文档 CPU 小节](../../../docs/fmri/README.md)；完整记录见 [CPU 聚合 JSON](final_merged_cpu_v1.public.json)。

合并上游 `d2237221` 后的 `2530650b` 已按实际导入模块核查：已测 FNIRT/SynthMorph GPU 候选路径的 103/101 个模块逐文件相同，新增 SynthSegParc 文件未进入这些调用。计时仍绑定 `98019133`；[源码范围核查](final_source_bridge_20261006.public.json)分别记录实际冻结闭包与 Git tracked 文件。已完成输出的全量核对方法见[保存输出工具](saved_output_tools/README.md)。

### 2026-10-06：完整 volume 默认 GPU 保持性回归

本轮用同一真实公开病例的原始 T1w 和完整 180 帧 BOLD（64×64×42×180），在 NVIDIA H100 PCIe 上串行运行四次 cold API。两份冻结源码为 FNIT main 基线 `6f624040` 与 CPU 优化候选 `98019133`；两种后端分别配对。调用包括原始输入、robust reference、完整运动校正、T1w 配准、PICA/ICA-AROMA，以及 preproc 和 clean 的正常读写；STC 关闭，CUDA 使用 TF32，四张 BOLD 输出保持 float32。不包含 recon-all 或 surface。

| 后端 | 基线 GPU API | 候选 GPU API | Torch 峰值 allocation | Torch 峰值 reservation | 采样 owned 进程树峰值 |
|---|---:|---:|---:|---:|---:|
| FNIRT | 597.814 s | 599.190 s | 6.537 GB | 11.899 GB | 14.053 GB |
| SynthMorph | 421.071 s | 415.174 s | 13.323 GB | 16.182 GB | 15.804 GB |

显存以 1 GB = 10⁹ bytes 计；同一后端的旧、新三项峰值均相同，全部低于 20e9 bytes。每次使用相同的 8 个 GPU 主机物理核心、PyTorch 8 线程、interop=1，来源和完整输入前后校验通过，末次 owned GPU 存活采样绑定实际 guard SHA。

两种后端各比较十个科学输出的全部 **395,140,404 个值**，不同值数、最大绝对误差和 RMSE 均为 0；dtype、affine、qform/sform、加载后的二进制头及原始存储头一致。四张 BOLD 均完整保留 180 帧并为有限值；11 份正常输出均绑定 SHA。元数据仅将基线未写出的 `confound_projection` 规范化为原有 `orthogonal`。

四次调用共 487 次共享设备采样，利用率均为 100%。上表是每个后端各一次旧、新 cold 运行的观测时间；FNIRT 按基线→候选、SynthMorph 按候选→基线执行。稳定 GPU 速度结论需要独立重复和可控设备负载。
API 时间包含正常读写和首尾 CUDA 同步；进程时钟另含导入、锁等待、来源/输出验证和最后采样握手。实际 owned 采样间隔最大为 0.972 s，采样峰值按该采样范围解释。

分步骤时钟见 [功能文档 GPU 小节](../../../docs/fmri/README.md)；完整记录见 [聚合 JSON](final_merged_gpu_v1.public.json)。

## 6. 最近记录

- 2026-10-06：完成 FNIRT/SynthMorph × CPU1/CPU8 × 两源码的 16 次 first/warm API；8 组旧、新及 8 组缓存配对十科学输出均逐值精确。每组完整 180 帧，核预算、来源、输入和配置门槛通过；各格计时为一次完整运行观察。
- 2026-10-06：FNIRT 与 SynthMorph 的四次完整默认 GPU cold 调用完成；冻结 `6f624040`→`98019133` 的两对十科学输出、全180帧、dtype 与头逐值精确。20 GB 显存门槛与来源/输入、TF32、末次采样检查通过；时间保留为共享 GPU 的单次观察。
- 2026-10-04：固定完整真实输入、CPU 分组、冻结源码和原程序；修复队列锁覆盖导入与校验，保留竞争中的无效计时。
- 2026-10-05：13 项最终完整 helper 比较完成，全部正常调用、来源、实际输入、同核预算、完整数值与头网格已核对；三项原始 int16 STC 协议保留。float32 原版控制和同核 FNIT first/warm 已通过，未改变生产默认。
- `compare_helpers.py` 与旧 int16 报告保存首轮事实；`compare_helpers_v2.py` 增加实际输入重哈希、完整浮点转换证明、预算和时序元数据门槛，不覆盖旧报告。

## 7. 原实现与参考文献

见 [helper 原实现和参考文献](../../../docs/fmri/CPU_HELPER_BENCHMARK_20261005.md#7-原实现与参考文献)。原程序仅供验证使用；FNIT 生产不依赖 FSL、AFNI、NiWorkflows 或 fMRIPrep。
