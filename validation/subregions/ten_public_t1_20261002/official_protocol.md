# 十例公开 T1 的官方完整流程

## 输入、软件和资源

本验证使用清单预先固定的 `sub-01` 至 `sub-10`、`ses-test` 公开 T1。公开发布文件已含 `mri_deface` 处理；本轮未追加影像处理。每例执行前核验文件大小和 SHA-256。`sub-01` 单列为曾参与开发的连续性例子，其余九例用于未见受试者的汇总。

2026-10-02 在 `gpucw1` 核实安装：`/public/software/apps/Freesurfer/8.2.0-1`，版本为 `freesurfer-linux-centos7_x86_64-8.2.0-20260314-d932c45`。实际官方 SAMSEG/GEMS Python、原生库、可执行文件和三套 atlas 的大小、SHA-256 随每例 `official_report.json` 保存。许可证只核验外置路径 `/public/software/apps/Freesurfer/8.0.0-1/license.txt` 存在、可读及大小；不读取、复制或发布内容。官方 atlas、权重及影像保留在服务器。

服务器有 128 个逻辑 CPU、64 个物理核、约 1 TiB 内存，无 swap。初次审计可用内存约 900 GiB，存在其他用户任务；后续调度前 CPU 实测约 65% idle。每例固定四线程，官方队列允许四例并行，共 16 线程；FNIT 使用独立 GPU 队列。CPU 负载、可用内存、进程 affinity 在每组件开始与结束时保存，因此耗时代表共享服务器上的实测。免密访问 `nodecw10` 和 `nodecw12` 未成功，本轮未在这些节点启动任务。

## 调用和输出

`run_official_subject.py` 是独立验证脚本，不进入 FNIT 运行时，不修改官方算法。每例使用新的 `official/subjects/sub-XX`，依次运行：

```bash
recon-all -i PUBLIC_T1 -s sub-XX -sd CASE_ROOT/official/subjects -all -openmp 4
segment_subregions brainstem --cross sub-XX --sd CASE_ROOT/official/subjects --threads 4 \
  --out-dir CASE_ROOT/official/subregions/brainstem --temp-dir CASE_ROOT/official/temp_brainstem
segment_subregions thalamus --cross sub-XX --sd CASE_ROOT/official/subjects --threads 4 \
  --out-dir CASE_ROOT/official/subregions/thalamus --temp-dir CASE_ROOT/official/temp_thalamus
segment_subregions hippo-amygdala --cross sub-XX --sd CASE_ROOT/official/subjects --threads 4 \
  --out-dir CASE_ROOT/official/subregions/hippo-amygdala --temp-dir CASE_ROOT/official/temp_hippo-amygdala
```

环境完整继承该安装的 `SetUpFreeSurfer.sh` 导出值，仅在内存中传递；不打印或写入完整环境。固定 `FS_V8_XOPTS=1`、`CUDA_VISIBLE_DEVICES=""`，并固定 OMP/MKL/OpenBLAS/NUMEXPR/ITK 为四线程、TensorFlow intra/inter 为四/一线程。没有使用 `-gpu`。官方 `etc/recon-config.yaml` 的 `UseGPU` 默认为假；`bin/recon-all` 在无 GPU 模式向 SynthSeg 传入 `--cpu`，SynthStrip 的 `--gpu` 仅在 `UseGPU` 为真时添加。保留 FreeSurfer 8 默认的神经网络重建选项，没有改为旧版处理流程。

`reconall` 完成需同时满足进程 exit 0、`scripts/recon-all.done` 存在及 `norm.mgz`、`aseg.mgz`、`wmparc.mgz` 均存在并已哈希。随后在开始细分区前立即发布 `components.reconall.state="completed"`，允许 FNIT stage 队列读取本例新的粗分割。FNIT raw 输入为公开 T1；stage 输入为本例新的 `norm.mgz`、`aseg.mgz`、`wmparc.mgz`。官方细分区标签不作为 FNIT 拟合输入。

| 结构 | 原网格硬标签 | 高分辨率硬标签 | 软体积文件 |
|---|---|---|---|
| 脑干 | `brainstemSsLabels.FSvoxelSpace.mgz` | `brainstemSsLabels.mgz` | `brainstemSsLabels.volumes.txt` |
| 丘脑 | `ThalamicNuclei.FSvoxelSpace.mgz` | `ThalamicNuclei.mgz` | `ThalamicNuclei.volumes.txt` |
| 左/右海马及杏仁核 | `{lh,rh}.hippoAmygLabels.FSvoxelSpace.mgz` | `{lh,rh}.hippoAmygLabels.mgz` | `{lh,rh}.hippoSfVolumes.txt` 和 `{lh,rh}.amygNucVolumes.txt` |

每组件记录自己的命令、PID、退出状态、日志 SHA-256、输出 SHA-256、实际进程墙钟和资源。各组件名为 `reconall`、`official_brainstem`、`official_thalamus`、`official_hippo_amygdala`；汇总文件为每例 `official/official_report.json`。汇总完成状态不能替代各组件的完成与输出校验。

## 计时定义

进程墙钟从 `Popen` 前立即取 monotonic 时间，到独立 `wait()` 线程观察进程退出，排除输入核验、初始软件/atlas 哈希及等待调度。完整官方流程墙钟从第一组件启动前到最后组件输出核验结束，包含组件间核验与哈希；另列四组件进程墙钟之和。

每个细分区保留官方 `process.py` 四行显式整数计时：`Preprocessing took`、`Initial atlas alignment took`、`Initial mesh fitting took`、`Mesh fitting took`；海马及杏仁核分别记录左右两组。这些阶段计时包含相应准备操作，不与组件墙钟相加。保留 `recon-all.log` 的实际资源/步骤计时行及行号；不把旧个体、旧执行或时间戳差推断为本轮步骤耗时。

十例终态分析完成后，使用只读 CPU 提取脚本：

```bash
python extract_cohort_steps.py --root TASK_ROOT \
  --analysis TASK_ROOT/analysis/cohort_analysis.json
```

脚本要求 `final_outcome_ready=true`，同时核对十例身份、最终分析、队列、实际 API/report/log 与冻结源的 SHA。默认输出到 `analysis/steps/`：逐例 `cohort_steps.json/.tsv`、十例及未见九例的 `cohort_steps_summary.json/.tsv`，另存 `cohort_reconall_fstime.tsv` 的实际 elapsed/日志行。均按计划人数列出有效数和缺失或失败数。

FNIT 保留共享预处理、脑干各独立步骤、丘脑/海马合成拟合及强度拟合、solver 每层 preparation/fit/post-fit 的实际计时。raw 强度预处理、bias correction、solver 子计时均是所属阶段的子集。丘脑/海马尚无独立配准和后处理计时，脑干自定义 solver 尚无通用多层内部计时，均标 NA；剩余 recipe 开销只列为未分配残差。没有可比中间标签时不报告步骤 Dice。`recon-all` 资源行只读取已有 `e` elapsed 字段，保留上下文时间戳，不从时间戳差产生步骤耗时或对嵌套计时求和。

## 启动失败及重试

首批十例均在 `recon-all` 启动后约 0.05–0.19 秒失败，日志为 `FREESURFER: Undefined variable.`，未进入实际重建。原因是验证脚本只传递部分 Setup 环境，而当前 `recon-all` 的 V8 选项路径使用 `FREESURFER` 别名。`-version` 在读取该路径前退出，因此原先版本核验不能发现这一启动问题。

修复仅改变验证脚本的环境传递：完整继承官方 Setup 导出值，并校验 `FREESURFER` 指向选定安装。首批输出与日志按启动失败尝试归档；新执行仍用全新 subject 目录。失败尝试不纳入完成流程耗时或精度，并在失败统计中保留。先确认首例进入真实重建命令，再扩展官方并行队列。已经因首批官方失败而 blocked 的 FNIT stage 记录也保留，新 stage 队列独立重跑；继续运行的 raw 队列不受影响。

## 原实现

本机核实官方入口为 `bin/segment_subregions` → `samseg.cli.segment_subregions`；实现位于安装的 `python/lib/python3.8/site-packages/samseg/subregions/`。公开原软件仓库：[FreeSurfer](https://github.com/freesurfer/freesurfer)。对应功能的论文和正式功能文档沿用项目 `docs/subregions/`，本文件只规定本轮可复核的验证执行方式。

## 本轮资产及许可核验

[资产审计](assets_audit.json)逐文件核对了当前队列记录、上一轮 `final_all_release_queue.fixed_inputs_references_assets` 及服务器实际字节：40/40 文件大小和 SHA-256 完全一致。其中 12 个原始图谱文件包括左右海马共用的三文件两份副本；官方安装的三套 atlas 共九个文件也全部与 FNIT 的固定上游清单一致。

五个 SynthSeg 权重/标签配置文件全部匹配 `assets-v1/asset-manifest.json` 和当前 GitHub Release 附件的大小、SHA-256。此次从 GitHub API 实际获取清单，28717 字节、SHA-256 `24a292cc79b0e9530559b7edb3fc09c7b157d0023b06335798ca4a902d5c1a3f`，同时核对 API 发布的 digest。[固定 Release](https://github.com/weikanggong1/Fudan-Neuroimaging-toolkit/releases/tag/assets-v1)附公开 `FreeSurfer-LICENSE.txt`；五个文件在清单中标注 FreeSurfer Software License 1.0。已安装的注册许可文件只为官方运行使用，内容没有读取或回收。

脑干的 `config.json`、`sigma1.npy`、`sigma2.npy` 为 FNIT 的图谱配方与平滑缓存，本轮保持原值并实际参与拟合。丘脑和左右海马目录另外留有 20 个历史配方/群体网格缓存文件；当前 recipe 直接读取原始 atlas、在受试者变换后的参考网格平滑，不读取这些历史文件。其精确字节仍参与资产固定检查。旧缓存创建时的脚本 SHA 没有留存，审计将该字段标为未知；当前资产清单、准备、平滑与 recipe 六个源码文件均与本轮冻结源逐字节一致。

原始 atlas 从已安装 FreeSurfer 或固定的 FreeSurfer 上游获取，本轮没有新增镜像、发布或传输 atlas、权重及注册许可内容。MGH 软件许可保留作者归属与许可条款，第三方资源的许可仍需依其来源核实；未明确的图谱再分发权利沿用原站获取策略。[FreeSurfer 原站许可](https://surfer.nmr.mgh.harvard.edu/fswiki/FreeSurferSoftwareLicense)
