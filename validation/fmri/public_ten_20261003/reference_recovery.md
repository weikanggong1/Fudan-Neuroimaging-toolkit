# 完整参考的输出检查修正

## 1. 功能简介与流程

独立官方参考从公开配对 T1w 和完整 180 帧 BOLD 开始，启用 fMRIPrep 25.2.4 内置 FreeSurfer 7.3.2 重建与默认 MSMSulc，保存 MNI152NLin6Asym 2 mm volume、T1w volume、双侧 fsnative BOLD 和 fsLR 91k CIFTI。冻结 harness 的早期版本把 native GIFTI 的 BIDS 实体顺序写成 `space-fsnative_hemi-*`；实际文件为 `hemi-L_space-fsnative` 和 `hemi-R_space-fsnative`。MRI 进程退出码为 0，最末端文件名检查因此报错。

`recover_reference_saved.py` 只修正这项检查和早期报告的字段结构。原 harness、原报告、原数据和 MRI 输出保持原样；每个当前 fresh attempt 新增 `report.corrected.public.json` 与 `files.corrected.private.json`。校验失败则保留独立失败报告，不获得 `complete` 状态。

```mermaid
flowchart LR
    A[原始终态报告及固定输入 SHA] --> B[核对 MRI exit 0 与原检查终态]
    B --> C[重查原输入和原冻结 harness SHA]
    C --> D[核对当前 attempt 全部原输出 SHA]
    D --> E[完整双侧 GIFTI 与 CIFTI 保存后检查]
    E --> F[原报告及文件前后 SHA 守卫]
    F --> G[独立 corrected 报告与精确私有文件绑定]
```

## 2. Python 调用、输入输出与参数

- `--reference-root`：本轮 fresh 官方参考根目录，包含 `cases/sub-CONxx/attempt-N`。仅从本根目录的独立 attempt 选择输出；不能指向历史重建缓存。
- `--source-root`：可选的旧 harness 归档根。默认与 reference root 相同；每例优先核对其实际 `launcher.snapshot.py` SHA，只有旧 attempt 没有该文件时才按原 SHA 查找 `run_reference_cohort*.py`。因此新 helper 源码和产物分置于 workspace/runs 时也能复现。
- `--raw-root`：本轮原始 BIDS 根目录及 `public_manifest.json`。文件为 `.nii.gz` T1w、完整 `.nii.gz` BOLD 与 BIDS JSON。T1w、BOLD 必须同时匹配本次启动前 SHA 和公开 acquisition manifest SHA。
- `--output-root`：新的独立检查状态目录。必须不存在；生成匿名逐例检查结果、`cohort.public.json` 与仅本地保留的失败诊断。原始私人路径仅保存在 private 文件。
- `--subjects`：待检查的公开 case ID，默认 `CON01 CON03 CON04 CON05 CON06 CON07 CON08 CON09 CON10 CON11`。
- `--watch`：等待尚未完成的 MRI 参考。没有该参数时仅检查已经就绪的 case。
- `--poll-seconds`：等待间隔，默认 60 秒；此等待不计为 MRI 计算时间。

校验先要求真实容器退出 0、节点监控提取退出 0。原报告失败时，错误列表必须非空且全部为已确认的 GIFTI 命名错误；未知失败、原 source/input 守卫失败均拒绝。原 v3 harness 已完整通过保存检查时，可以再独立严格检查，保留其原成功状态和时钟。已存在 corrected 或私有绑定时拒绝覆盖，不重新生成旧 SHA 与时间。三项主要产物原 SHA、原完整有限值检查、180 帧与原 TR 保持绑定；新增 native GIFTI 检查要求左右各 180 个等长、全部有限的顶点数组。CIFTI 必须为 180×91,282，具有正确的 SeriesAxis/TR/SECOND 和 BrainModelAxis，左右完整顶点域各 32,492，并有有效的皮下体素网格。两个新生成 MSM 球面须点/三角形数组唯一、全部点有限、面索引有效，且顶点域对应各自 native BOLD。

重建绑定固定为本 fresh attempt 的 `derivatives/sourcedata/freesurfer/sub-CONxx_ses-preop`，核对 `orig.mgz` 和 `recon-all.done`；不能将早期可选 `recon_all` 字段缺失 session 的路径作为有效 subject。球面折翻、自交和解剖质量由另外的[重建比较](reconstruction_comparison.md)逐例评估，文件形状校验不代替这些质量结果。


后续元数据入口 [collect_reference_stages.py](collect_reference_stages.py) 的 `--reference-root` 为本参考根、`--output` 为新匿名阶段报告、`--report-name` 默认 `report.corrected.public.json`、`--expected-cases` 默认本轮十例（单例控制须显式 CON01）。[wait_reference_final_stages.py](wait_reference_final_stages.py) 的 `--reference-root` 为相同参考根、`--collector` 为固定 SHA 的 collector 文件、`--output-root` 为新的独立收集目录、`--poll-seconds` 默认 60。它们只处理已保存报告与节点监控，输出阶段 JSON 和额外的输入/helper SHA 守卫。

将示例目录改为本次 fresh 数据、运行和固定 helper 的实际目录。已存在 corrected 时读取原报告，helper 会拒绝覆盖。

```python
from pathlib import Path
import sys

# 该 helper 不是安装包；从仓库根运行时显式加入实际 validation 目录。
repository_root = Path("/path/to/Fudan-Neuroimaging-toolkit")
helper_directory = repository_root / "validation/fmri/public_ten_20261003"
sys.path.insert(0, str(helper_directory))
from recover_reference_saved import recover

reference_root = Path("/benchmark/reference_fmriprep25_2_4_v1")
original_report_path = reference_root / "cases/sub-CON01/attempt-02/report.public.json"
raw_bids_root = Path("/benchmark/raw")

# 仅检查当前 fresh attempt 的既有输出；不启动任何 MRI 计算。
legacy_harness_archive = Path("/benchmark/helpers/frozen_harnesses")
checked_result = recover(original_report_path, raw_bids_root, legacy_harness_archive)
print(checked_result["status"])
```

## 3. 命令行调用

从保存固定 helper 的目录运行下列 CLI；源文件与结果目录分别保存。

```bash
python recover_reference_saved.py \
  --reference-root /benchmark/reference_fmriprep25_2_4_v1 \
  --raw-root /benchmark/raw \
  --source-root /benchmark/helpers/frozen_harnesses \
  --output-root /benchmark/reference_recovery/results \
  --subjects CON01 CON03 CON04 CON05 CON06 CON07 CON08 CON09 CON10 CON11 \
  --watch --poll-seconds 60
```


阶段汇总和全十例等待入口：

```bash
python collect_reference_stages.py \
  --reference-root /benchmark/reference_fmriprep25_2_4_v1 \
  --output /benchmark/reference_stages.public.json \
  --report-name report.corrected.public.json
```

```bash
# 新目录用于全十例最终元数据，不能覆盖已保存四例/八例汇总。
this_fresh_reference_cohort_root="/benchmark/reference_fmriprep25_2_4_v1"
fixed_v2_stage_collector_file="/benchmark/helpers/collect_reference_stages.py"
new_final_stage_collection_root="/benchmark/new_final_stage_collection"
python wait_reference_final_stages.py \
  --reference-root "$this_fresh_reference_cohort_root" \
  --collector "$fixed_v2_stage_collector_file" \
  --output-root "$new_final_stage_collection_root" --poll-seconds 60
```

## 4. 原软件调用

本工具不需要原软件计算命令。生成被检查产物的完整原软件命令保存在各 report 的 `canonical_container_command` 中，核心为：

```bash
fmriprep /data /case/derivatives participant \
  --participant-label CON01 --ignore slicetiming --cifti-output 91k \
  --output-spaces MNI152NLin6Asym:res-2 T1w fsnative \
  --fs-license /opt/fs-license/license.txt \
  --fs-subjects-dir /case/derivatives/sourcedata/freesurfer \
  --nprocs 8 --omp-nthreads 4 --mem-mb 49152 \
  --work-dir /case/work --resource-monitor --stop-on-first-crash --notrack
```

这里只作为独立原流程 benchmark 调用；FNIT 产品运行时不调用这些外部封装软件。参考没有 fieldmap，不请求 SyN SDC，不做 AROMA/denoise，完整原 run 不裁帧；内部自动增加的 MNI152NLin2009cAsym 处理在[实际配置证据](reference_scope.public.json)记录，计入参考整例运行。

## 5. 最新真实结果、计时与脑图

报告中的 `container_process_wall_seconds` 保持原容器从启动到退出的连续墙钟；`continuous_wall_through_saved_QC_seconds` 也保留原记录，旧命名检查失败和 v3 原检查通过分别标注边界。`recovered_QC_seconds` 是后来补检本身的独立耗时；`recovery_gap_since_original_end_seconds` 是原报告结束至补检完成的经过时间，包含中间等待和本次补检。二者都不相加冒充原连续 whole wall。原 v3 已通过的边界为 `original QC passed`；旧命名失败为 `original filename-check failed`。新补检同时记录 `recovery_started_utc`；`corrective_QC_seconds` 是 `recovered_QC_seconds` 的同值别名，统计只计一次。

十例独立官方整链现已全部退出 0，并完成独立保存后检查。CON08–11 原 v3 harness 已通过 MRI 和原保存检查，旧六例只因末端命名检查失败；两支均保留真实完整 MRI、原报告与额外检查的证据。

| 公开 case | 原容器墙钟 / s | 原 harness 最终检查结束 / s | 原检查边界 | 独立补检 / s | 完整保存检查 |
|---|---:|---:|---|---:|---|
| [CON01](reference_completed/CON01.public.json) | 11738.532 | 11812.228 | 命名检查失败 | 8.229 | 通过 |
| [CON03](reference_completed/CON03.public.json) | 11580.622 | 11653.293 | 命名检查失败 | 7.195 | 通过 |
| [CON04](reference_completed/CON04.public.json) | 10982.994 | 11056.696 | 命名检查失败 | 6.923 | 通过 |
| [CON05](reference_completed/CON05.public.json) | 12259.018 | 12334.254 | 命名检查失败 | 8.213 | 通过 |
| [CON06](reference_completed/CON06.public.json) | 12474.838 | 12548.969 | 命名检查失败 | 8.261 | 通过 |
| [CON07](reference_completed/CON07.public.json) | 11803.857 | 11878.075 | 命名检查失败 | 7.935 | 通过 |
| [CON08](reference_completed/CON08.public.json) | 12318.424 | 12394.508 | 原 QC 通过 | 8.207 | 通过 |
| [CON09](reference_completed/CON09.public.json) | 12796.919 | 12874.742 | 原 QC 通过 | 9.162 | 通过 |
| [CON10](reference_completed/CON10.public.json) | 13579.763 | 13656.370 | 原 QC 通过 | 8.269 | 通过 |
| [CON11](reference_completed/CON11.public.json) | 12733.820 | 12809.904 | 原 QC 通过 | 7.373 | 通过 |

原命名检查失败记录在 `earlier_reference_attempts/` 保存，原通过报告单列于 `reference_original_passed/`。以上均 180 帧；前三例 TR 2.1 s、CON05–11 为 2.4 s；全部 MNI 2 mm BOLD 为 91×109×91×180，全部 CIFTI 为 180×91,282，具有 21 个脑结构及正确的 SeriesAxis。原容器进程墙钟中位数为 12288.721 s，范围 10982.994–13579.763 s；这一边界不包含后来的独立补检。报告未声称 FNIT 与 ANTs/FreeSurfer 不同算法数值等价，也不因输出齐全宣称皮层重建质量通过。

`collect_reference_stages.py` 从真实 `node_runtime.public.json` 读取时间和资源监控。参数 `--reference-root` 为参考根、`--output` 为匿名汇总路径，`--report-name` 默认明确选择 `report.corrected.public.json`。`--expected-cases` 是预期的公开病例 ID 列表，默认 CON01、CON03–CON11；单例硬件控制显式传 `--expected-cases CON01`。仅全部预期病例完整、没有额外病例时写 `complete_cohort`，否则列出 `pending_cases` 并写 `partial_cohort`。每组 `elapsed_span_seconds` 是最早节点启动到最晚结束，含依赖等待；`recorded_node_active_interval_union_seconds` 是这些实际节点执行区间的并集，移除节点间空闲、重叠只计一次。MapNode 聚合与 mapflow 子项按相同起止/时长/命令 SHA/执行程序去重复（每例 607→491）。两种阶段指标均可能与别组重叠，不能求和代替整例时间。内存和 CPU 值是已采样单节点观察，缺失仍是未测量，不是整进程上限。虽然启用 `--resource-monitor`，实际 CON01 的 FreeSurfer、MSM 与 MCFLIRT 命令节点保存 `resmon=off` 且无峰值字段；不能据少量 Python 节点样本声称已测这些重计算命令的 CPU/内存峰值。


最新[全十例分步骤记录](reference_stages_all10_20261003.public.json)绑定原节点文件 SHA、报告 SHA、工作流组件、最长节点及资源样本数量，保留重叠边界。[独立收集守卫](reference_stages_all10_20261003.guard.public.json)再次核对实际 20 份报告/节点文件 SHA 与运行前后 helper，元数据收集活动时间为 0.148 s，等待和收集不计入 MRI 墙钟。此前[八例记录](reference_stages_8cases_20261003.public.json)与[四例记录](reference_stages.public.json)保持原字节。例如 CON01 的 MNI6 resampling 节点活动区间并集为 58.423 s，但跨度 9624.210 s 包含等待上游解剖配准，不能把该跨度写成 resampling 计算耗时。

阶段 collector v2 的 SHA 为 `703a27edbd6c29fc897228b6056f36233873dac004e37d05ded03090826a4026`。[五项真实元数据合同检查](reference_stages_v2_metadata_contract.public.json)确认四例缺六例的状态、单例控制的明确完成集合、额外/重复病例拒绝，以及所有原步骤数值与 v1 完全相同。此前四例和单例汇总继续绑定原 v1 SHA；这些状态整理检查没有重新运行 MRI，也没有新增性能测量。

[wait_reference_final_stages.py](wait_reference_final_stages.py) 只等待这十份 exact attempt 的真实 corrected 完成合同，再调用上述冻结 v2 collector。`--reference-root` 为本参考根，`--collector` 为 SHA 固定的该 collector 文件，`--output-root` 是与原参考和 helper 源互不包含的新目录，`--poll-seconds` 默认 60。CON01 固定 attempt-02，其余固定 attempt-01；每例必须 complete/exit0、180 帧、原 input/source 守卫和额外保存文件守卫为 true。输出 `reference_stages_all10.public.json` 和独立 `final_collection_guard.public.json`，逐行再次绑定原报告与节点 SHA；尚未齐十例时只写 waiting 状态。等待及汇总活动时间不加入生产时钟。



### 真实脑图示例

下图来自本例完整 180 帧的两条正式整链产物，显示固定脑内 MNI 时间均值、差值与逐体素时间相关。

![CON01 完整180帧的实际 MNI 输出](paired_batches/v12_batch_007/figures/CON01_mni_mean_difference.png)

[原图及21图来源 SHA](paired_batches/v12_batch_007/figures_provenance.public.json)绑定实际产物和绘图代码；完整数值比较见[整链报告](README.md#真实脑图及独立重建质量)。

## 6. 最近版本与 benchmark 记录

当前公开默认参考入口是 v3，SHA `02181a873c0f804ddc8be502b777feebfb0b1ae07b8f1030e12535db20ca4879`；默认 recovery 是经过真实绑定控制的 v4，SHA `60af6ed281838e5e88deb17042a46aead9d4f9d0c8db227cf29fc3d8ab154c63`。v4 优先逐例 snapshot、兼容同 SHA 别名，并拒绝覆盖既有证据。[七项真实文件合同控制](recovery_contract_v4.public.json)通过，其中四项确认原 corrected/SHA/clock 原封不动；它们不是 MRI benchmark。现场实际执行的 v3 recovery、原 v1/v2/v3 MRI harness 与已发布旧报告保持原字节；[冻结清单](frozen_harnesses/manifest.public.json) 明确各版实际范围。

- 原 v1/v2 冻结参考保留原失败输出和真实 process wall。CON01 初次 attempt 的 BIDS validator 失败在 MRI 启动前；成功 MRI 来自独立新 attempt-02。
- 原文件名问题只涉及末端检查；v3 harness 改正确实体顺序及可选 session subject 指针，新 source SHA 为 `02181a873c0f804ddc8be502b777feebfb0b1ae07b8f1030e12535db20ca4879`。新调度参数 `--respect-running-budget 4` 把同根已有运行纳入总并行数量，MRI 命令和每例 8/4 线程保持一致。它仅用于随后新启动的 CON08–11；活动旧 workers 不改动。
- 独立检查脚本 SHA 与原 report SHA 均进入 corrected 报告；最终正式重建比较必须显式使用 `report.corrected.public.json`，对应 candidate 固定 `candidate_v4/source_1128bc52`。

## 7. 参考文献与固定原代码

原流程出处和数据许可见[本队列说明](README.md)及[公开数据清单](data_manifest.public.json)；原软件实现为 [fMRIPrep 25.2.4](https://github.com/nipreps/fmriprep/tree/25.2.4) 与其实际内置 [sMRIPrep 0.19.2](https://github.com/nipreps/smriprep/tree/0.19.2) / [FreeSurfer 7.3.2](https://github.com/freesurfer/freesurfer/tree/v7.3.2) 版本。参考论文为 [fMRIPrep](https://doi.org/10.1038/s41592-018-0235-4) 和 [FreeSurfer](https://doi.org/10.1016/j.neuroimage.2012.01.021)。脑图示例由本队列的独立比较和绘图脚本从真实产物生成，当前文件检查报告不替代精度脑图。
