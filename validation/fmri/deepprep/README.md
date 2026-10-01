# DeepPrep：真实 490 帧 volume / surface 参照

2026-09-30 在共享 H100 PCIe 服务器上，完成官方 DeepPrep 25.1.0 的两个独立真实数据运行。两次使用相同原始 T1w 和 BOLD，保留完整 490 帧、TR 0.735 s。当前 FNIT 的结果见 [volume / surface 验证](../README.md)。

| 方法与输出 | 墙钟时间 | 计时起点与内容 |
|---|---:|---|
| DeepPrep volume，MNI152NLin6Asym 2 mm | **2091.36 s（34 分 51 秒）** | 原始完整 T1w 与 BOLD；包含独立结构重建、容器启动、BOLD 预处理、混杂变量导出、写盘及 QC |
| DeepPrep surface，fsaverage6 | **1969.45 s（32 分 49 秒）** | 独立空目录，从同一原始 T1w 与 BOLD 重新开始；包含结构重建、预处理、双侧 surface 与 QC |

两行分别从原始输入重建解剖；相加会重复计算重建，不能当作一次同时输出 volume 和 surface 的时间。DeepPrep 的 surface 时间包含结构重建和 BOLD 预处理；FNIT surface 从已经清理的 volume 与既有重建开始，两者起点不同。

DeepPrep 源码固定到 [`04af8f3`](https://github.com/pBFSLab/DeepPrep/tree/04af8f3541737505de41530b23804cd9f2b6efa1)。[机器可读报告](benchmark.public.json)保存实际运行、输出检查、软件及镜像哈希。已移除旧 FNIT 实现的组合计时与重复报告；本页保留这两次独立 DeepPrep 参照。

## 输入与输出

| 项目 | DeepPrep 25.1.0 本次运行 |
|---|---|
| BOLD | 88×88×64×490；SHA-256 `67717caaa823141c6e59d46db52e92deb4419d232ba699c399c69eb8888a0e92` |
| T1w | 208×256×256 原始完整 T1w；SHA-256 `3ecd8196fc47b440c4fba28eec8a801411d8e4b2aba224a243a89dec48b0b4dc` |
| SBRef | 本次 BIDS 未提供 SBRef |
| 结构重建 | 两次均新建 white、pial 和注册球面；volume 的 BBR 也需要结构处理 |
| 去噪输出 | preproc BOLD 与 confounds TSV；本次未追加混杂回归 |
| surface | 双侧各 40,962 顶点 fsaverage6 GIFTI |
| 畸变校正 | 无 fieldmap，`bold_sdc=False` |
| CPU 与精度 | Nextflow CPU 额度 10、主机内存 32 GB，保留官方模型精度默认值 |

本页没有 FNIT 与 DeepPrep 最终 BOLD 的逐值精度比较。若比较当前实现的速度与精度，需要统一 T1/SBRef、结构重建起点、表面空间、去噪和计时边界，并在相同负载下重复运行。

## DeepPrep 完成与资源记录

两个运行均退出 0；volume 的 83 个任务和 surface 的 78 个任务全部 `COMPLETED`、退出 0，无 `CACHED`。原始任务时间和内存字段见 [volume trace](volume_trace.tsv)及 [surface trace](surface_trace.tsv)；`raw=true`，时间字段单位为毫秒，RSS/VMEM 为字节。

| 检查 / 观测 | volume | surface |
|---|---|---|
| 最终输出 | 91×109×91×490 NIfTI，2 mm，数值全部有限 | 左右各 40,962×490 GIFTI，数值全部有限；此处按顶点×时间记维度 |
| 结构几何与混杂项 | 双侧 white / pial / sphere.reg 坐标有限、面索引合法；confounds 490 行 | 同样通过 |
| 解剖分支跨度 | 1534.855 s | 1420.474 s |
| BOLD 分支跨度 | 2065.736 s | 1945.108 s |
| 本次进程树最高 GPU 显存采样 | 18,656 MiB（18.22 GiB） | 6,766 MiB（6.61 GiB） |
| 整卡显存采样范围 | 30,671–73,308 MiB | 42,721–75,183 MiB |
| 整卡利用率采样范围 | 48–100% | 62–100% |

分支跨度是该分支首次任务开始到最后任务结束的时间，分支存在重叠，不能相加或从总墙钟中相减来推导“纯 BOLD”耗时。GPU 每 2 秒采样；最高采样值不等于连续峰值，也不能与 FNIT 的 PyTorch allocated / reserved 直接作显存比。两个 GPU 均有其他作业，本次使用物理 GPU 1；这些是各一次真实观测。

GPU 标签设 `maxForks=1` 且每个 GPU 任务申请 10 CPU，以串行运行 GPU 阶段。这会影响调度和耗时。10 CPU / 32 GB 是 Nextflow 调度额度，不是全部库线程或显存的硬上限。首次并发运行曾发生 OOM；失败及服务器中断的尝试均排除在本页成功计时之外。“空输出”指未复用重建或 Nextflow 结果，不保证操作系统及驱动缓存为空。

surface QC 出现同一上游 `qc_bold_create_report` tuple 声明警告 5 次。最终 HTML 存在，5 个不同的本地引用均可解析，流程成功退出；这里核对的是报告文件及引用完整性，未将其当作视觉质量评估。完整匿名标量、软件、模型哈希及警告记录见 [DeepPrep 报告](benchmark.public.json)。

## 部署与复测

DeepPrep 是独立参照容器，不进入 FNIT 运行依赖。FNIT 安装及输入参数见[volume 文档](../../../docs/fmri/README.md)、[surface 文档](../../../docs/fmri/surface.md)和[FNIT 复测命令](../README.md#单被试如何复测)。在有网机器按[官方安装说明](https://deepprep.readthedocs.io/en/25.1.0/installation.html)取得镜像，在 GPU 机器用 Singularity 运行，并自行取得合法 FreeSurfer license。

```bash
# 原作者发布的 25.1.0 镜像；生成的 SIF 需记录自己的 SHA-256。
singularity pull deepprep_25.1.0.sif \
  docker://registry.cn-beijing.aliyuncs.com/pbfslab/deepprep:25.1.0
```

本次 OCI manifest 为 `sha256:8d35461da04b055c7a5cd005500dc1db8b00220379f5d266975e5a40e3e9bbff`，SIF 为 14,234,120,192 字节，SHA-256 为 `a3bb4af8753b5d840ca9885676db1f525151ac2d4c54805d4abf754f8e5f0941`。镜像的 58 个 OCI 描述符及 28 个模型文件均核对大小和 SHA-256。自行重新构建的 SIF 可有不同字节哈希，需同时核对镜像来源和内容。本次工具为 Singularity CE 4.2.2、PyTorch 2.0.1+cu118、TensorFlow 2.11.1，GPU 驱动 535.216.03；CPU 为双路 Xeon Gold 6430。

测量脚本针对一例 `sub-benchmark`、490 帧 BOLD、2 mm volume 和 fsaverage6 surface。输入结构如下；数据、镜像、许可及输出均留在使用者自己的目录。

```text
benchmark_root/
├── containers/deepprep_25.1.0.sif
└── bids/
    ├── dataset_description.json
    └── sub-benchmark/
        ├── anat/sub-benchmark_T1w.nii.gz
        └── func/
            ├── sub-benchmark_task-rest_run-01_bold.nii.gz
            └── sub-benchmark_task-rest_run-01_bold.json
```

BOLD JSON 包含 `TaskName=rest`、`RepetitionTime=0.735`、64 层真实 `SliceTiming` 和实际 `PhaseEncodingDirection`；不要为新数据沿用本例元数据。准备原始 T1w 与完整 BOLD，不做裁剪或截帧。

```bash
# 安装主页 Conda 环境并激活 fnit；脚本的 nibabel / NumPy 已包含在环境中。
benchmark_root=/absolute/path/benchmark_root    # 已准备上面结构；runs/volume、runs/surface 必须不存在
freesurfer_license=/absolute/path/license.txt  # 用户自行取得的有效许可，只读挂载
singularity_executable=/absolute/path/singularity  # GPU 主机可运行的 Singularity

for output_mode in volume surface; do
  python validation/fmri/deepprep/run_benchmark.py \
    --root "$benchmark_root" --mode "$output_mode" \
    --gpu 1 --cpus 10 --memory 32 \
    --singularity "$singularity_executable" --fs-license "$freesurfer_license"
done
```

| 脚本参数 | 输入、默认值与作用 |
|---|---|
| `--root` | 必填，benchmark 根目录；读取 `bids/` 和 `containers/`，新建 `runs/<mode>/` 与 `<mode>.config` |
| `--mode` | 必填，`volume` 或 `surface`；各自从空目录开始，拒绝复用已有目录 |
| `--gpu` | 默认 1，物理 GPU 序号；官方 `--device` 和监测使用该序号 |
| `--cpus` | 默认 10，Nextflow CPU 调度额度及单 GPU 任务额度 |
| `--memory` | 默认 32，主机内存调度额度，GB |
| `--singularity` | 默认 `singularity`，可执行文件名或绝对路径 |
| `--fs-license` | 必填，使用者的 FreeSurfer 许可路径；只读挂载到容器 `/fs_license.txt` |

脚本生成独立运行 home，先复制镜像内离线 Nextflow 缓存，然后开始墙钟。计时包含容器启动至退出，排除镜像安装、缓存复制、输入准备和事后输出检查。结果包括 `runs/<mode>/benchmark.json`、`pipeline.log`、`gpu_samples.csv`、`QC/trace.tsv`、`Recon/`、`BOLD/` 和 `QC/`。运行 JSON 和日志含使用者自己的路径，发布前只保留匿名字段。本仓库脚本仅把实际运行时的本地可执行文件及许可路径改为参数，其余处理参数和测量边界相同。

对应原软件命令如下；volume 使用 `MNI152NLin6Asym / None`，surface 使用 `None / fsaverage6`。两次都禁用 SDC、保留 490 帧、导出混杂项，未给额外精度覆盖。

```bash
# 这些变量由上面测量脚本按 mode 创建，示例展示其调用的原软件命令。
output_mode=volume
output_directory="$benchmark_root/runs/$output_mode"
runtime_home="$output_directory/runtime_home"
nextflow_config="$benchmark_root/$output_mode.config"
container_image="$benchmark_root/containers/deepprep_25.1.0.sif"
volume_space=MNI152NLin6Asym
surface_space=None

"$singularity_executable" run --cleanenv --nv \
  --home "$runtime_home:/home/benchmark" \
  --env TF_FORCE_GPU_ALLOW_GROWTH=true,CUDA_DEVICE_ORDER=PCI_BUS_ID \
  -B "$benchmark_root/bids:/input:ro" -B "$output_directory:/output" \
  -B "$nextflow_config:/benchmark.config:ro" -B "$freesurfer_license:/fs_license.txt:ro" \
  "$container_image" /input /output participant \
  --participant_label benchmark --bold_task_type rest \
  --fs_license_file /fs_license.txt --config_file /benchmark.config \
  --device 1 --cpus 10 --memory 32 \
  --bold_sdc False --bold_confounds True --bold_skip_frame 0 \
  --bold_volume_space "$volume_space" --bold_volume_res 02 \
  --bold_surface_spaces "$surface_space"
```

以上调用中的 `participant` 为逐被试运行；`participant_label` 与 `bold_task_type` 选择被试及任务，`fs_license_file` 与 `config_file` 指向只读挂载，`device/cpus/memory` 设置资源，`bold_sdc` 关闭畸变校正，`bold_confounds` 导出混杂项，`bold_skip_frame=0` 保留全部帧，`bold_volume_space/res` 设置 2 mm MNI volume，`bold_surface_spaces` 设置 surface 输出。运行用的 Nextflow 配置为：

```groovy
executor { name = 'local'; cpus = 10; memory = '32 GB' }
trace {
    enabled = true
    file = '/output/QC/trace.tsv'
    raw = true
    fields = 'task_id,hash,native_id,name,status,exit,submit,start,complete,duration,realtime,%cpu,peak_rss,peak_vmem,cpus,attempt'
}
process {
    withLabel: with_gpu { maxForks = 1; cpus = 10 }
}
```

可把本页 `benchmark.public.json` 的 `software` 与 `input` 对象分别保存为根目录的 `software_manifest.public.json` 与 `input_manifest.public.json`；复测时更新为实际机器、镜像和输入，不能沿用旧哈希。`python validation/fmri/deepprep/summarize_benchmark.py --root "$benchmark_root"` 读取这两份清单及两个完成的运行，检查退出状态、trace 和 GPU 采样，生成 `benchmark_results.public.json` 与 `benchmark_timings.csv`。`--root` 是其唯一参数，必填，指向同一根目录。

## 原实现、引用与数据边界

- DeepPrep 原实现：[25.1.0 release](https://github.com/pBFSLab/DeepPrep/releases/tag/25.1.0)、[固定源码](https://github.com/pBFSLab/DeepPrep/tree/04af8f3541737505de41530b23804cd9f2b6efa1)、[官方运行参数](https://deepprep.readthedocs.io/en/25.1.0/usage_local.html)。
- Ren 等，*DeepPrep: an accelerated, scalable and robust pipeline for neuroimaging preprocessing empowered by deep learning*，Nature Methods，2025，[DOI](https://doi.org/10.1038/s41592-025-02599-1)。FNIT 各阶段及 Workbench 引用见 volume / surface 功能页。
- 只公开匿名标量、输入及软件哈希、测量脚本和任务元数据。真实 MRI、个体几何、QC 脑图、私有日志、许可、镜像及模型不进入本仓库。DeepPrep 镜像和模型从原作者渠道获取；FNIT 模板及权重继续遵循固定 Release 清单、校验及各自许可。
