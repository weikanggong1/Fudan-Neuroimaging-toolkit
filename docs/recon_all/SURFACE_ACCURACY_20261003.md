# 2026-10-03：固定输入 white/pial 精度诊断

本次只比较同一旧真实 T1 的 FNIT 自产冻结输入，不是本轮10个新被试的整例结果。官方和 Conda 分别复制完整输入到隔离目录，执行 white.preaparc、最终 white、完整 pial；每项输入都重新从冻结目录复制，前一项官方输出不会作为后一项输入。由此区分放置算子差异与上游网格传播；**不能把两次独立 white 测试解释为连续白质链改善**。

## 输入、参数与对应证明

每项报告绑定11个文件的 SHA-256：`brain.finalsurfs.mgz`、`wm.mgz`、`aseg.presurf.mgz`，当前半球的 `orig`、`orig.premesh`、`white.preaparc`、`white`、`autodet.gw.stats`、`cortex.label`、`cortex+hipamyg.label` 和 `aparc.annot`。MRI 是原冻结 conform 网格；表面是 surface RAS/mm，有序顶点105,598个、有序三角面211,192个，标签按同一顶点索引解释。

三个已完成原生对照的11项 SHA 全部一致，运行前后输入 SHA 不变。把各隔离目录路径规范化为 `SUBJECT` 后，官方与 Conda 的全部参数相同，均为2线程；执行脚本 SHA 为 `d853965e625d245bcc518cdce97aba1049e90ebe8910b412e64689f492b33be2`，各程序与日志 SHA 见[绑定报告](../../validation/recon_all/accuracy_20261003/task_05/white_stage_v1/log_diagnosis.json)。实际二进制不同，white 使用当前 Conda `mris_place_surface_white_fast`，pial 使用当前 Conda `mris_place_surface`；官方为固定8.2.0-1程序。

输出保留各自完整输入的顶点数和有序面，双方从相同有序网格开始且放置算子不重建网格，故这里可比较同索引坐标。仅有相同元素数不能证明对应；比较脚本同时检查完整输入哈希、有序面及输出祖先。不对应时直接拒绝同索引比较，不把最近顶点替代成对应顶点。

## 当前同输入结果

| LH完整阶段 | Conda / 官方耗时，s | 位移均值，mm | P99，mm | 最大，mm | >0.1 mm顶点 | 其中cortex内 / 外 |
|---|---:|---:|---:|---:|---:|---:|
| white.preaparc，四轮 | 168.749 / 154.571 | 0.000972577 | 0.014805496 | **1.148342424** | 200 | 114 / 86 |
| 最终white，四轮 | 161.548 / 143.890 | 0.000279633 | 0.005493982 | **0.220273963** | 15 | 15 / 0 |
| pial，四轮原生两方 | 168.638 / 146.102 | 0.031369198 | 0.225254657 | **1.482209493** | 5,844 | 5,840 / 4 |

耗时从隔离目录复制和输入SHA计算之后开始，包含阶段计算与阶段读写，不含调用前导入、输入复制和哈希；线程2、同主机；只执行一组当前原生配对，未做官方重复性或稳定吞吐验证，不能据此宣布候选GPU速度等效。以上是几何放置算子误差；尚未新增对应厚度、面积、体积逐脑区完整指标测量。整体指标等效保持 `not_assessed`，138项诊断及已有门槛未修改。0.1 mm仅为本次局部定位分组，不是新增验收阈值。

各输出都是单连通网格，无边界边、无非流形边。本次只读网格报告没有执行三角面自相交和white/pial穿越检测，不能据此宣称几何质量全部通过。逐脑区位移、前20个峰值索引和所有>0.1 mm顶点的局部CSV见[几何报告](../../validation/recon_all/accuracy_20261003/task_05/white_stage_v1/white_analysis_v2/placement_comparison.json)。CSV含 `vertex_id`、`region`、`in_cortex_label`、`displacement_mm`、`dx_mm`、`dy_mm`、`dz_mm`；不包含参考坐标或影像。

局部最大偏差所在脑区：white.preaparc为楔叶（cuneus，1.148342 mm）；最终white为颞中回（middletemporal，0.220274 mm）；pial为岛叶（insula，1.482209 mm）。>0.1 mm区域的最大连通簇分别有41、5、101个顶点。pial最大误差顶点64732位于84顶点簇，说明平均位移较小仍可伴随局部异常。地域来自同输入aparc注释，不作为独立解剖准确性证明。

## 日志中第一处可见差异

| 阶段 | 第一处打印的优化指标差异 | 第一处打印的边界搜索差异 | 四轮结束步数，Conda / 官方 |
|---|---|---|---|
| white.preaparc | 第0轮第7步：SSE 538928.4 / 538928.3，打印RMS均4.571 | 第2轮，sigma=0.5需要扩大搜索的顶点128 / 126；nripped均5941 | 18/26/33/38，两方相同 |
| 最终white | 第0轮第5步：SSE 494123.6 / 494123.7，打印RMS均4.134 | 四轮已打印的边界摘要相同 | 13/22/30/35，两方相同 |
| pial | 第0轮第1步：SSE 22966464.0 / 22966466.0，打印RMS均33.773 | 第1轮，sigma=1需要扩大搜索的顶点5859 / 5838；nripped均5588 | 26/31/35/40 与 26/32/36/41 |

三组的目标强度阈值打印相同。prewhite各轮拒绝次数为0/1/1/1，两方一致；final white为1/1/1/1，两方一致；pial为0/1/1/1，两方也一致，但第二轮起结束步数不同。逐步SSE/RMS/步长差在[CSV](../../validation/recon_all/accuracy_20261003/task_05/white_stage_v1/log_diagnosis.csv)。日志数字经过打印舍入；第一处打印差异**不是第一处坐标或算子差异的证明**。不能把1.148/1.482 mm局部偏差全部归为FP32尾差，FP32库、编译选项或具体算子实现差异目前均为待证假设，需官方重复性和逐步隔离证据。

## 复现与恢复

05:37 UTC的阶段快照中，后台序列PID `78930` 在LH完整Python pial等待共享锁、RH尚未开始；该旧快照保留用于过程追踪。08:42 UTC复查时现有14项队列全部完成、receipt为finished/exit0，最新结果见下节。Python pial已有完整四轮实现，会保存所有接受/拒绝试步及轮末表面；Python white只有前缀，明确不参与完整第三方对照。官方固定CLI没有每轮完整坐标快照，其日志仅作标量定位。无需重启现有队列或在原输出目录重跑。

[placement_probe.py](../../validation/recon_all/accuracy_20261003/task_05/placement_probe.py) 全部参数：`--subject` 为冻结自产目录；`--output` 为必须不存在的隔离输出目录；`--kind` 为 `prewhite/white/pial`；`--backend` 为 `official/conda/python`（Python只接受pial）；`--hemi` 为lh/rh；`--binary` 为原生程序路径；`--assets` 为对应程序的资源目录；`--device` 默认cuda:0、供已有Python GPU采样。缺文件、设备失败或程序失败抛异常/非零退出，完整原生报告保留失败码。主pipeline和功能调用仍见[white](FINAL_SURFS_CHAIN.md)、[原生pial](NATIVE_PIAL_PLACEMENT.md)、[Python pial](PYTHON_PIAL_PLACEMENT.md)。内部优化迭代没有独立原软件CLI；原生命令和所有具名参数已完整保存在各阶段JSON的 `command`。

```bash
# 只读比较既有输出，不运行放置；路径按实际服务器目录填写。
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 python compare_placement.py \
  --root /data/diagnostics/task_05/placement_v1 \
  --output /data/diagnostics/task_05/analysis/placement_comparison.json
# 只读提取已保存日志的首个可见差异。
python summarize_native_logs.py \
  --root /data/diagnostics/task_05/placement_v1 \
  --output /data/diagnostics/task_05/analysis/log_diagnosis.json
```

`compare_placement.py` 接受必需 `--root`（现有隔离输出父目录）和 `--output`（JSON输出路径），同时写局部CSV；`summarize_native_logs.py` 接受相同两个具名参数，输出JSON与逐步CSV。两者对缺失阶段报告pending，不构造未执行结果；发现完整输入或参数不一致时抛 `ValueError`。只读分析可以重复写到新的分析目录；阶段放置输出目录须保留。

2026-10-03早先提交修复了脑区面积汇总舍入位置（真实LH逐面定义误差从max9.10e-6 mm²到0），并修复Python pial终止拒绝的四轮控制流（7项单元回归通过）；两项均不更换当前生产原生放置。之前日期的三方比较继续作为历史定位材料，不能替换本次新程序/线程绑定报告。本次原生pial误差数字与历史冻结结果相同，仍按本次新运行单独记录，未进行官方重复性新实验。

## 参考与上游

- [FreeSurfer固定源码d932c45](https://github.com/freesurfer/freesurfer/tree/d932c45b7941662ea380a05efef580568b98d41a)，`mris_make_surfaces/mris_place_surface.cpp`、`utils/mrisurf_mri.cpp`。
- Fischl B. FreeSurfer. *NeuroImage* 2012;62:774–781. [DOI](https://doi.org/10.1016/j.neuroimage.2012.01.021)。


## 08:42 UTC：双侧完整阶段已返回

2026-10-03 08:42:15 UTC复查，既有队列14个阶段全部完成、失败0、排队0，receipt为 `finished/exit_code=0`；08:44:01 UTC记录报告文件mtime。12个原生报告均有实际 `returncode=0`，两份Python报告通过完整四轮输出及总队列exit0确认完成；不把没有原生returncode字段的Python报告填成独立原生退出码。完整输入SHA、有序面对应与2线程设置再次核对，输入在运行后未改。最新[状态摘要](../../validation/recon_all/accuracy_20261003/task_05/placement_complete_v1/completion_summary.json)绑定所有14个报告SHA和时间；mtime表示报告文件写入时间，不能当成程序精确开始时间。

| 双侧完整阶段 | Conda / 官方阶段耗时，s | Conda-官方 mean / P99 / max，mm | >0.1 mm顶点 |
|---|---:|---:|---:|
| LH white.preaparc | 168.749 / 154.571 | 0.000973 / 0.014805 / **1.148342** | 200 |
| LH 最终white | 161.548 / 143.890 | 0.000280 / 0.005494 / **0.220274** | 15 |
| LH pial | 168.638 / 146.102 | 0.031369 / 0.225255 / **1.482209** | 5,844 |
| RH white.preaparc | 164.060 / 141.785 | 0.000556 / 0.006602 / **1.103394** | 109 |
| RH 最终white | 138.210 / 123.824 | 0.000151 / 0.002672 / **0.445158** | 8 |
| RH pial | 166.439 / 144.971 | 0.008342 / 0.147992 / **1.265559** | 1,876 |

完整Python pial使用已有 `sampling_backend="triton"`、`candidate_backend="snapshot"`、`device="cuda:0"`，保持TF32策略、无半精度。双侧41步、四轮累计结束点均为26/32/36/41，最终全部有序坐标与面相对本次官方**逐值相同**（mean/P99/max、非零坐标差、>0.1 mm顶点均0）。LH阶段耗时1182.561 s，RH1007.901 s；两方明显慢于当前生产原生程序，因此保留诊断用途，不替换生产默认。只能证明当前同输入完整阶段输出一致，不能推出上游连续链或整例等效。

LH Python最终相交清理轨迹2→6→0、平滑2轮共200次；RH轨迹2→0、平滑1轮100次。每个Python步骤的接受/拒绝试步及四轮结束坐标哈希均已保留在各自report；本次两侧都没有触发“达到缩步上限且仍拒绝”的修复分支，其一般控制流正确性由先前单元回归支撑，不能把本次坐标一致归因于该修复。原生CLI未导出完整中间坐标，不能宣称逐步坐标也相同。三方网格均单连通、边界边和非流形边为0；Python报告的自相交清理结果不替代原生输出独立质量检查。

RH white.preaparc第0轮第7步首次打印SSE差515027.5/515027.4；第2轮sigma扩展108/109。RH最终white第0轮第三次缩步日志首次SSE差211340.3/211340.2；第2轮sigma扩展115/116。RH pial第0轮第8步首次SSE差1740119.0/1740118.9；第1轮sigma扩展4516/4555。RH三项四轮结束步数分别17/26/33/36、11/19/26/29、26/32/36/41，原生两方相同，仍存在局部坐标偏差。[双侧日志](../../validation/recon_all/accuracy_20261003/task_05/placement_complete_v1/rh_log_diagnosis.json)及对应CSV保留打印级证据，原因仍未定位。

本队列仅包含**一个旧真实冻结被试的双侧**。第二例同输入表面阶段未在本队列安排，结果为未测，不能记为0；本轮10例原始T1完整验收由协调者的独立队列运行。整体指标等效继续为 `not_assessed`。GPU采样阶段只有PyTorch allocated/reserved峰值，没有NVML同刻父子进程合计采样，不能据此声称已验证20 GB完整流程预算。

恢复时读取已完成receipt和14份报告；无需重启或重跑这些阶段。若需新被试或新的算子修复验证，应创建独立输出目录并绑定新源码、程序及输入SHA。当前只读比较器和日志摘要器可继续用于协调者的新自产输出；日志摘要器新增 `--hemi=lh`（默认），也支持 `rh`。
