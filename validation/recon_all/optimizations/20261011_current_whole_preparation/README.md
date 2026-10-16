# 当前源码的两例原始 T1 整例准备

本记录只交付只读审计和下一轮整例协议，未启动整例、未创建输出目录、未安装服务器环境、未修改统一索引或生产文件。科学源码固定为 `9292a0a7be63f6bc96f2dfa358261cb1587e0f23`；父提交 `22851fe5f556a444ad38f015423ef4a0eda33a93` 仅补充安装报告。任务2默认 pial 语义回归收齐后仍须确认最终科学版本；若源码变化，重新生成源码、wheel、包装器和资源绑定。

```mermaid
flowchart LR
  A[原始 T1 与固定资源核验] --> B[最终源码与安装回执]
  B --> C[空目录原始 T1 整例]
  C --> D[体积处理与完整 GA]
  D --> E[双侧表面与注释指标]
  E --> F[138 项输出及网格质量门]
  F --> G[独立参考比较与脑图]
```

## 输入、输出及证据

- `protocol.public.json`：两例输入身份、22 项实际配方、完整 GA 参数、线程/GPU 建议、计时边界和待满足的门槛；参数名可直接对应 Python 调度字典。
- `frozen_source_binding.public.json`：科学源码、报告父提交、全部 1,387 项源码文件核验、5,309,664 字节 wheel 的 SHA-256、包装器和固定 Release 清单哈希。源码归档仅用于本地核验，不随本准备包发布。
- `readiness.public.json`：现场十例原始 T1、118 项 assets/weights 的大小和 SHA-256、两例 NIfTI 头、公开数据集 CC0 元数据及历史许可材料身份。核验耗时是文件哈希检查时间，不是 MRI 算法时间。
- `previous_whole_exit_audit.public.json`：旧整例的源码/wheel/包装器身份、EXIT、报告哈希、输出缺失清单和失败阶段耗时。原始服务器路径、原始数据、权重和许可凭证均不随包发布。
- `PREPARATION_MANIFEST.json` 与 `check_preparation.py`：本准备包的只读完整性检查；结果中的 `launch_ready=false` 表示仍有启动条件待满足。它不重新读取服务器，不会启动计算。

两例为 `ds000114_sub-06` 和 `ds000114_sub-07`，各输入一幅未预处理 T1w `.nii.gz`，float32、RAS、形状均为 256×156×256；体素尺寸和仿射矩阵保留在 readiness，不能用头信息相近推断图像等价。每例单独新建此前不存在的输出根目录，候选只读取原始 T1、固定模板和模型权重。assets 包含模板、图谱、LUT；weights 包含模型参数，逐文件绑定大小/哈希。`native` 子目录不能接入候选。

预期每例输出为 `subject/mri` 的体积/变换、`subject/surf` 的双侧表面和指标、`subject/label` 的注释、`subject/stats` 的统计，以及 scripts、API 运行报告、benchmark、进程树采样、最终退出回执和输出哈希清单。完整标准输出清单为 138 项；存在检查之后还需读取、尺寸/几何及网格质量检查，不能用数量代替正确性。

## Python 调度配方

以下是冻结源码配方的调用示例，**本轮没有执行**；实际整例应经有退出回执和外层计时的独立进程启动。输入路径由本地私有绑定提供。

```python
from fnit.recon_all.python_profile import python_profile_options

# 总线程预算为 4，左右半球各 2；使用映射后的逻辑 GPU 0。
reconstruction_options = python_profile_options(
    device="cuda:0", threads=4,
    n4_backend="simpleitk", n4_execution="in-process",
    topology_backend="python-ga", input_execution="serial",
)
# 此处只生成调度字典；不读取 T1、不创建输出、不运行 recon-all。
```

| 参数 | 本轮计划值及意义 |
|---|---|
| backend / device / threads | python / cuda:0 / 4；自有候选、逻辑设备、整例 CPU 总预算 |
| input_execution | serial；输入 N4 与分割串行，按包装器实际行为记录 |
| native_optimizations | torch；复用成熟 Torch 加速路径 |
| n4_backend / n4_execution | simpleitk / in-process；通用 Python 库的完整 N4，父进程 CPU 执行 |
| wm_backend / wm_execution | torch-optimized / isolated；成熟 WM 链在新 Python worker 中执行 |
| gca_execution / gca_inverse_backend | isolated / torch；GCA 独立 worker 与 GPU 求逆 |
| gca_candidate_chunk | 1024；GCA 候选分块数量 |
| normalization_controls_backend / normalization_initial_bias_backend | torch / torch；控制点与初始偏置 GPU 计算 |
| sphere_finish_backend / sphere_normals_backend | torch / numba；球面收尾与法向计算 |
| mni_execution | parallel-late；既有 MNI 后置并行调度 |
| wm_edit_backend | python；WM 编辑的自有有序实现 |
| defects_backend / inflate_backend | torch / torch；缺陷图和膨胀计算 |
| surface_placement_backend | python；white/pial 自有放置链 |
| topology_backend | python-ga；完整自有 GA、SOAP、压缩及 orig 清理 |
| annotation_gibbs_backend | numba；注释 Gibbs 有序 CPU 核 |
| remesh_scalar_storage | python；remesh 有序标量存储 |
| fill_backend | numba；有序 fill CPU 核 |
| hemisphere_workers | 2；双侧 worker 各使用 threads//2 |

完整 GA 保留 seed=1234、population=10、max_unchanged=10、niters=-1、initial_selection=True、vertex_eliminate=True、use_overlap=False、normal_status=surface、soap=5、final_orig=True、face_cache=True、blocker_scan=numba。`niters=-1` 保持既有完整停止规则；不缩减迭代、不跳过 orig/交叉检查。实验性 GPU distance 本轮不启用。TF32 沿用默认与已有 FP32 例外，禁止自动切 half/autocast。

## 命令行与推荐下一步

推荐现在仅复核准备包：

```bash
# 只读：打印准备包完整性和待满足条件，不启动整例。
python3 validation/recon_all/optimizations/20261011_current_whole_preparation/check_preparation.py
```

待最终源码冻结、许可映射核对、accelerator-b 当前 wheel 独立安装验证和资源再检查完成后，未来命令形态如下。**尚未授权运行；外层计时与可信显存归属采样仍须接入后才能形成正式启动命令。**

```bash
# 私有变量：Python 环境、冻结包装器、原始 T1、新的空目录、权重、模板。
# 每例各运行一次；输出根必须此前不存在。仅展示参数，不在本轮执行。
"${python_interpreter}" "${frozen_benchmark_script}" \
  --t1 "${raw_t1_file}" --output-root "${new_output_root}" \
  --weights-dir "${verified_weights_directory}" --assets-dir "${verified_assets_directory}" \
  --backend python --device cuda:0 --threads 4 \
  --n4-backend simpleitk --n4-execution in-process \
  --input-execution serial --topology-backend python-ga \
  --code-version 9292a0a7be63f6bc96f2dfa358261cb1587e0f23
```

建议物理 GPU 2、CPU 80–83，CUDA_VISIBLE_DEVICES 映射到 cuda:0；OMP/MKL/OPENBLAS/NUMBA/ITK 线程上限均设 4，半球 worker 使用 2。03:37:54 UTC 快照中该 GPU 已用 26,413 MiB，是共享设备，CPU/GPU 均未预留。启动前重新核对任务占用、affinity、磁盘、设备 UUID 与可信 PID 映射。20 GB 是候选进程树显存目标，当前不能认证已满足。

包装器 `tools/benchmark_recon_torch_end_to_end.py:75–81` 未将 input_execution 传给 python_profile_options，又将其返回值覆盖解析参数，因此显式 parallel 会变成 serial。本协议使用实际 serial；如果后续修复包装器，须重新记录脚本哈希，不能把旧命令写成并行 benchmark。

## 计时、显存和完成门

外层 wall 从启动解释器之前开始，至子进程退出、回收及最终产物写出后结束，覆盖导入、预检、哈希、模型加载、传输、计算、写出和报告发布。当前包装器 full_harness 排除父解释器初始导入和当前最终元数据写出，应单独保留该字段而另补外层完整计时。CLI wall 为 Popen→退出。每个 stage 保留实际读写和同步边界；嵌套/并行阶段不相加当整例时间。记录首次 JIT、模型缓存冷热状态。

0.5 秒周期内，使用同一快照对父进程和所有仍存活子进程的所选 GPU 显存求和；禁止各进程历史峰值相加。先验证容器/宿主 PID 映射与进程身份；记录 NVML 或 nvidia-smi 后端、采样错误、遗漏和最大间隔。归属无法解析时，候选进程树 peak 保留 null，不能拿设备总占用证明低于 20 GB。既有 v2/v3 归属均为 ownership_unresolved。

两例顺序执行，失败保留现场，不重试、不补跑、不换输入。通过条件：EXIT=0、benchmark exit=0、API 完成、138 项输出均可验证、最终网格有限/闭合/流形/方向一致/非零面面积/无几何交叉，既有 GA 质量门全部通过。原版输出只进入独立比较器，不进入候选计算。正式比较需记录参考版本、输入/输出身份、逐步/整例耗时和实际数值指标，并生成 T1/分割、white/pial 和球面脑图。目前无当前版完整结果，不能填写精度、速度提升或脑图。

Python 启动审计和 worker 守卫只证明其记录到的进程启动；不能把它们描述为操作系统文件/动态库隔离。候选禁止调用 FSL、FreeSurfer、SPM、MRtrix3、AFNI 或其封装器；仅允许自有 Python、Numba、Torch 和许可允许的通用库。

## 旧整例审计与更新记录

| 旧尝试 | 包源码 | CLI 失败前 wall（秒） | EXIT | 输出 | 最终质量 |
|---|---|---:|---:|---|---|
| v2 | af1e782d… | 19187.455 | 1 | 28/138 | 未评估 |
| v3 | c2293890… | 18402.517 | 1 | 28/138 | 未评估 |

两次均使用同一 sub-06 原始 T1，不是两个病例。两者右半球完整 GA premesh 清理仍有 2 个几何交叉面，即使 Euler=2、单连通也未通过质量门；双侧 white/pial/sphere.reg 等最终表面缺失。上述时间是失败持续时间，各次设备/线程亲和和源码不同，不能解读为速度改善。统一索引仍标 running，与实际 EXIT/report 不符；本轮保留索引原状，并在本报告记录差异。

- 2026-10-11：本准备包；现场确认旧 v2/v3 EXIT=1、缺失输出，重新核验 10/10 输入、118/118 资源，冻结 9292a0a7 两例协议，未启动。
- 22851fe5：当前完整 wheel 的本地 CPU 安装/29 模块/四 worker CLI 报告；不代表服务器安装、GPU 验证或整例完成。
- 9292a0a7：合并当前科学实现与 worker/N4 守卫；任务2最终回归及后续 freeze 待收齐。

资源许可证逐文件映射和当前 notice 文件定位仍待复核，历史 notice 哈希仅作溯源。未授权再分发资源；需保留固定 Release 或原作者来源，不能用哈希通过替代许可。

## 原软件与参考

原软件整例命令仅作对照说明：`recon-all -i "$raw_t1_file" -s "$reference_subject" -all`，不能在候选 stage 调用。阶段对应及原实现引用沿用[Python 表面链说明](../../../../docs/recon_all/PYTHON_SURFACE_PIPELINE.md)和冻结源码 THIRD_PARTY_NOTICES；本文是准备协议，不新增科学实现或宣称原软件等效。

- [FreeSurfer recon-all 文档](https://surfer.nmr.mgh.harvard.edu/fswiki/recon-all)、[原代码库](https://github.com/freesurfer/freesurfer)。
- Dale, Fischl & Sereno. Cortical surface-based analysis I. NeuroImage (1999). DOI: 10.1006/nimg.1998.0395。
- Fischl, Sereno & Dale. Cortical surface-based analysis II. NeuroImage (1999). DOI: 10.1006/nimg.1998.0396。
- Tustison et al. N4ITK: Improved N3 Bias Correction. IEEE TMI (2010). DOI: 10.1109/TMI.2010.2046908。
- [公开 ds000114 数据集](https://openneuro.org/datasets/ds000114)、[固定 FNIT 项目](https://github.com/weikanggong1/Fudan-Neuroimaging-toolkit)。数据集身份和许可以本地已取得并校验的元数据为证据；本轮未浏览远端网页。
