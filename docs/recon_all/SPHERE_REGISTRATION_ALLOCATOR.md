# 球面配准的独立进程分配缓存

## 1．功能与流程

球面配准复用已有`run_register_sphere()`：完整sulc配准、smoothwm配准和最后重叠清理均保留。有序梯度平均可在GPU执行；目标函数、步长接受/拒绝及其余CPU步骤不在本次改写。

本页验证PyTorch CUDA分配缓存的性能影响。每次在新解释器执行完整配准，使用同一FNIT自产sphere、smoothwm、sulc和声明图谱，依次关闭、开启、开启、关闭缓存。各进程采用独立的冷Numba/Triton缓存，包含启动、JIT、文件读写、CPU/GPU传输和同步。

```mermaid
flowchart LR
    I[自产有序sphere / smoothwm / sulc + 图谱] --> A[关闭缓存：完整配准]
    A --> B[开启缓存：完整配准]
    B --> C[开启缓存：完整配准]
    C --> D[关闭缓存：完整配准]
    D --> E[有序面 / 坐标 / 轨迹 / 耗时 / 同期显存比较]
```

这不是整例recon-all基准，也没有把关闭缓存后的allocator零计数解释为零显存。父进程的活张量和缓存策略由其原接口管理；独立进程退出后回收其缓存。

## 2．Python调用、输入与输出

```python
from fnit.recon_all.mris_register_run import run_register_sphere

registration_report = run_register_sphere(
    sphere="subject/surf/lh.sphere",       # FNIT自产标准球面，surface RAS坐标，mm
    smoothwm="subject/surf/lh.smoothwm",   # 与sphere顶点和有序面对应的白质表面
    sulc="subject/surf/lh.sulc",           # 与sphere顶点顺序相同的float32顶点图
    atlas="assets/average/lh.folding.atlas.acfb40.noaparc.i12.2016-08-02.tif",  # 左侧声明图谱
    output="run/lh.sphere.reg",           # 新输出目录已创建；不覆盖输入
    overlap_device="cpu",                # 保留本次已验证的CPU末尾清理
    averaging_device="cuda:0",            # 逻辑GPU编号，只选择有序梯度平均设备
)
```

| 输入 | 格式、默认值和限制 |
|---|---|
| `sphere` | FreeSurfer三角表面文件，顶点`(N,3)`、有序面`(F,3)`；surface RAS/mm。无默认值。 |
| `smoothwm` | 同格式、同顶点顺序及有序面；不是最近点配对的另一张网格。无默认值。 |
| `sulc` | FreeSurfer morph文件，`(N,)`float32；顶点数及顺序必须对应sphere。无默认值。 |
| `atlas` | 声明的对应半球TIFF折叠图谱；固定文件SHA。无默认值。 |
| `output` | 输出球面路径，父目录须已存在；输入/输出分别指定。无默认值。 |
| `overlap_device` | 默认`"cpu"`，单独控制末尾球面重叠修复。 |
| `averaging_device` | 默认`"cpu"`，可指定显式`"cuda:N"`；仅平均采用GPU，float32有序更新不变。 |

输出`sphere.reg`保留顶点顺序、有序面和源几何尾部，坐标为surface RAS/mm。返回字典包含输入/输出路径与SHA、临时sulc种子的SHA、两个完整pass报告、逐步`iteration/phase/sigma/averages/dt/next_state`、末尾负面数量及包含读写的API时间。`run_register_sphere()`不会自行修改全局精度或分配策略；CUDA缓存必须在新进程第一次CUDA调用前确定。形状、面顺序或顶点图不匹配、算法未收敛、CUDA/JIT失败均抛异常，不以近似结果补文件。

半球调度的显式候选选项如下。图谱与`paint`程序只使用声明的资产和Conda源码构建产物；注册后的avg_curv仍沿用该阶段原有实现。

```python
from fnit.recon_all.hemisphere_parallel import run_hemisphere_group

hemisphere_report = run_hemisphere_group(
    subject="subject",                   # FNIT自产完整被试目录，输入由私有副本隔离
    operation="register",                # 完整双侧配准与原有avg_curv后处理
    device="cuda:0",                     # 显式逻辑GPU，父进程可已初始化CUDA
    threads=4,                           # 两侧合计CPU预算，每侧2线程
    workers=2,                           # 1/2；2采用READY/GO屏障后并行
    profile_stages=True,                 # 本次剖析边界同步，正常生产默认False
    callable_path="fnit.recon_all.native_free:_hemisphere_operation",  # 已有可信阶段入口
    kwargs={                             # JSON可序列化公共参数，不放参考输出
        "assets": "declared_assets",
        "binaries": {"paint": "native_bin/mrisp_paint"},
        "registration_atlases": {
            "lh": "declared_assets/average/lh.folding.atlas.acfb40.noaparc.i12.2016-08-02.tif",
            "rh": "declared_assets/average/rh.folding.atlas.acfb40.noaparc.i12.2016-08-02.tif",
        },
    },
    startup_wait_seconds=30,              # 仅pre-GO资源启动预算；不是算法运行超时
    cuda_allocator_cache="enabled",      # 仅新worker开缓存；默认inherit保持旧行为
)
```

| 调度输入 | 默认值与结构 |
|---|---|
| `subject`、`operation` | 无默认；目录及可信callable定义的阶段字符串。原目录不作为worker的就地写入目标。 |
| `device`、`threads` | 无默认；CPU或逻辑CUDA设备、正整数CPU总预算。`threads>=workers`，奇数并行预算向下取整。 |
| `workers` | 默认2，允许1/2。1顺序exec两侧，2同时exec；保留共享输出最后右侧语义。 |
| `profile_stages` | 默认False；True做指定设备阶段边界同步，不改变算法精度。 |
| `callable_path` | 默认上述FNIT半球入口；可信`module:function`字符串，提供JSON可序列化返回值。 |
| `kwargs` | 默认None，公共参数字典；调度器另外传入私有`subject`、`hemi`、`device`、`threads`及`operation`。 |
| `startup_wait_seconds` | 默认30、有限非负秒。0关闭资源重启，单次启动仍有90秒期限；仅可信pre-GO CUDA OOM可重启。 |
| `cuda_allocator_cache` | 默认`inherit`；`enabled`/`disabled`只选择fresh exec子环境，父已初始化CUDA也不改其实际策略。CPU调用同样记录选择，但没有CUDA分配效果。 |

返回保留`values[lh/rh]`、worker实际precision/thread/allocator/stage报告、组墙钟与重叠、同期显存、发布清单、私有复制与父空闲缓存释放时间；新增`worker_allocator_selection.requested/selected_policy/environment_at_entry/environment_at_fresh_exec`。成功屏障后逐文件发布；未声明的共享写入、算法失败或发布失败不返回complete，按已有规则取消本调度新进程组并保留失败日志。网格空间不被调度器改变。

`worker_cuda_cache_environment(environ, *, cuda_allocator_cache="inherit")`是调度内部子步骤，没有独立官方命令。输入完整环境Mapping，输出`(child_env, selection_report)`两个字典；`inherit`按变量存在性保持原状态（包括字符串`"0"`与空串），`enabled`移除`PYTORCH_NO_CUDA_MEMORY_CACHING`，`disabled`设为字符串`"1"`。只复制/选择环境，不初始化CUDA、不修改父Mapping或TF32。非法选项在私有输出创建前抛ValueError。线程参数由既有`native_thread_environment()`独立设置并保持总预算。

## 3．命令行复现

```bash
python validation/recon_all/python_gpu_port/benchmark_registration_allocator.py \
  --subject public_subject \
  --hemi lh \
  --atlas assets/average/lh.folding.atlas.acfb40.noaparc.i12.2016-08-02.tif \
  --source frozen_source/src \
  --profiling-module frozen_source/src/fnit/recon_all/profiling.py \
  --output new_registration_abba \
  --device cuda:0 \
  --threads 4 \
  --code-version ACTUAL_FROZEN_VERSION \
  --previous-self-output public_subject/surf/lh.sphere.reg
```

`--subject`只读取已有公开T1自产表面；`--hemi`选择lh/rh；`--atlas`固定对应半球图谱；`--source`指定已冻结源码的src目录；`--profiling-module`固定独立显存采样器；`--output`必须是新目录；`--device`为显式逻辑CUDA编号；`--threads`默认4，是每次单半球完整配准的固定CPU总预算；`--code-version`为实际代码标签；`--previous-self-output`可在计算完成后诊断既有自产结果，不进入计算。内部`--worker/--allocator`仅供fresh exec调度，正常复现由ABBA父程序设置。

`summary.json`保存全部输入/源码/脚本SHA、四份worker报告、exec/API时间、轨迹及有序网格比较、allocator峰值和同期显存采样。每个run内保留`api.json/worker.json/worker.log`及新`sphere.reg`。参数/文件不合规或任一子进程失败时保存失败收据并返回非零，不把部分运行标为完成。采样归属不明时tree峰值为null；目标卡总占用和全部计算进程上界分别报告。

v1诊断聚合字段被worker同名allocator字典覆盖；四次算法运行不受影响，原执行状态保留。v2用不同字段并增加已完成报告的只读派生入口：

```bash
python validation/recon_all/python_gpu_port/benchmark_registration_allocator.py \
  --analyze-report completed_run/summary.json \
  --analysis-output new_derived_metrics.json
```

`--analyze-report`必须包含四个returncode=0的完整ABBA worker、相同输入SHA及原有序网格比较；`--analysis-output`是未存在的新JSON。函数`analyze_completed_report(report_path, output_path)`返回0，不重写原报告、不读影像、不将派生统计标作新一次完整算法执行；缺少比较、失败worker或身份不一致时抛异常。

已初始化父CUDA的缓存隔离合同可单独执行：

```bash
python validation/recon_all/python_gpu_port/check_hemisphere_cache_live_cuda.py \
  --output new_live_cuda_contract \
  --device cuda:0 \
  --threads 4
```

该脚本不读取影像。`--output`必须是新目录；`--device`默认`cuda:0`且必须有显式编号；`--threads`默认4，是父进程和两个worker的CPU总预算。父进程在关闭缓存后创建float32活张量，再通过显式enabled启动两侧fresh exec。输出`contract.json`、两份顶点目录marker和worker日志；返回0表示父张量SHA、父环境、子缓存、线程和TF32均通过合同。参数、CUDA、调度或合同失败时保存失败报告并返回1，不生成标准MRI输出。

## 4．原软件调用

```bash
mris_register -curv \
  subject/surf/lh.sphere \
  average/lh.folding.atlas.acfb40.noaparc.i12.2016-08-02.tif \
  reference/lh.sphere.reg
```

该命令属于独立官方benchmark路径。生产接口不执行预装FreeSurfer；本次缓存ABBA复用FNIT完整Python/PyTorch/Numba配准，不改变固定源码导出的算法。官方整例比较与本页的同算法分配策略比较分别记录，不能用后者证明与官方整例指标等效。

## 5．当前精度、耗时和显存

公开ds000114的sub-06左半球使用803aec50原始T1完整运行的自产sphere/smoothwm/sulc，同主机A100、CPU64–67、4线程完成四次fresh exec。每次采用独立冷JIT并包含解释器启动、完整API和读写。

| 顺序 | 缓存 | 完整exec墙钟（s） |
|---|---|---:|
| A1 | 关闭 | 466.583 |
| B1 | 开启 | 417.828 |
| B2 | 开启 | 471.134 |
| A2 | 关闭 | 413.757 |
| 中位数 | 关闭 / 开启 | 440.170 / 444.481 |

开启缓存的本组完整墙钟慢0.979%，没有实测整步收益，因此球面配准保持inherit默认。节点存在动态共享GPU负载，短期时间波动大于此差值，不能据此发布稳定吞吐结论。第一次关闭缓存的API为454.202s，其中sulc pass 255.837s、smoothwm pass 198.272s；已有GPU平均之外的CPU拟合、力计算与接受/拒绝仍保留。详细逐步时间见[机器报告](../../validation/recon_all/optimizations/20261009_registration_allocator/reports/a100_sub06lh_v2/README.md)。

四次130,679顶点、261,354面均有序面及坐标逐元素相同，完整输出SHA也与既有自产803结果相同；逐步轨迹、刚体参数和分数相同。最终径向负面和零面积面计数均为0，这只覆盖本次记录的径向方向检查。**严格复现：通过该同输入缓存配对；优化引入退化：本次未见；整体指标等效：未判定。**官方整例对照和[原始T1完整运行及脑图](README.md)保持各自范围，本页不是新的整例加速结果。

开启缓存worker的PyTorch allocated/reserved峰值分别为14,113,792/23,068,672字节；关闭缓存时这些计数不可用，报告null。采样目标卡峰值43,429,920,768字节、全部计算进程合计上界43,413,143,552字节含其他用户负载，不能归为FNIT显存；进程树归属未确认，峰值null。采样名义间隔0.5s，最大实际间隔8.295s且5次查询失败，因此不能凭此宣称FNIT整步满足或超过20GB。

半球环境接线已在同一环境CPU84–87、4线程、CUDA隐藏条件完成32项合同，63.12s。覆盖默认继承、显式选择只影响新worker、父环境不变、总线程预算、READY/GO/重启、失败取消、私有发布和共享右侧覆盖语义；模拟目标不充当真实影像benchmark。

另一个真实CUDA生命周期合同为12.839s：父CUDA已初始化并持有活张量、父缓存关闭，两侧新worker实际开启缓存；父张量SHA与环境保持、每侧2线程、TF32开启且无autocast，7/7检查通过。这是调用方式验证，不是影像精度或整例资源验收。

## 6．更新与benchmark记录

- 2026-10-09：完整fresh exec ABBA完成，冻结同一803aec50自产sub-06左半球输入和完整配准源码；每次独立冷JIT，保持TF32默认，不启用半精度。新增调度显式子进程缓存选择，默认inherit保留旧行为。注册本组无整步收益，不设为默认enabled；该选项供其他完整阶段单独验证。
- v1聚合allocator同名覆盖保留为诊断失败，v2派生分析不重跑/改写冻结worker和输出。半球接线测试基于39cbf95d加两份接线/测试补丁绑定实际源码SHA，不将新代码测试版本充当原803整例版本。
- 原有[球面配准性能记录](SPHERE_REGISTRATION_PERFORMANCE.md)和[半球GPU显存说明](HEMISPHERE_GPU_MEMORY.md)保持各自版本、CPU预算与计时范围，不混合为本次结果。

## 7．参考文献与源码

- [FNIT完整球面配准入口](../../src/fnit/recon_all/mris_register_run.py)、[已有有序GPU平均](../../src/fnit/recon_all/mris_register_average_gpu.py)、[复现脚本](../../validation/recon_all/python_gpu_port/benchmark_registration_allocator.py)。
- [已初始化CUDA父进程合同](../../validation/recon_all/python_gpu_port/check_hemisphere_cache_live_cuda.py)、[冻结机器报告](../../validation/recon_all/optimizations/20261009_registration_allocator/reports/a100_sub06lh_v2/README.md)。
- [FreeSurfer固定源码](https://github.com/freesurfer/freesurfer/tree/d932c45b7941662ea380a05efef580568b98d41a)、[mris_register](https://github.com/freesurfer/freesurfer/blob/d932c45b7941662ea380a05efef580568b98d41a/mris_register/mris_register.cpp)。
- Fischl B, Sereno MI, Tootell RBH, Dale AM. High-resolution intersubject averaging and a coordinate system for the cortical surface. *Human Brain Mapping*. 1999;8:272–284.
- [PyTorch CUDA内存管理](https://pytorch.org/docs/stable/notes/cuda.html#cuda-memory-management)、[PyTorch2.5.1 allocator源码](https://github.com/pytorch/pytorch/blob/v2.5.1/c10/cuda/CUDACachingAllocator.cpp)。
