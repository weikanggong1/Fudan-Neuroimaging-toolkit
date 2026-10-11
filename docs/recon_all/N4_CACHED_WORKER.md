# 完整 N4 的缓存隔离执行

## 1．功能与范围

`run_isolated_n4()` 在新exec进程中复用现有 `n4_itk_torch_experimental.correct_volume()`。固定算法、200轮、重建和uint8写出保持相同，仅子进程开启CUDA分配缓存，退出后释放其CUDA上下文。父进程分配器/TF32和当前recon-all默认不改。

```mermaid
flowchart LR
    A[自产orig / 父CUDA可已初始化] --> B[新exec / 子缓存开启]
    B --> C[既有完整N4 / 200轮]
    C --> D[同网格uint8 nu0 / JSON]
    D --> E[子退出 / 释放CUDA]
```

本轮真实cache-off输入链的Torch N4需要76.9–91.1s，既有缓存开启冻结算法仅数秒，因此优先验证局部执行策略。没有用新近似、更少轮次、半精度或全局缓存开关换速度。

## 2．Python输入、输出与参数

```python
from fnit.recon_all.n4_torch_worker import run_isolated_n4

report = run_isolated_n4(
    input_path="subject/mri/orig.mgz",  # 自产3D非负原网格T1，不是官方参考
    output_path="diagnostic/nu0.mgz",  # 新路径，同网格uint8，不重采样
    report_path="diagnostic/n4-worker.json",  # 新JSON路径，与输入/输出不同
    device="cuda:0",  # 必须显式逻辑GPU；保留CUDA_VISIBLE_DEVICES映射
    threads=4,  # 新子进程Torch/OpenMP/BLAS预算，默认4
    profile=False,  # 默认False，True同步N4子段计时
    code_version="frozen-code-label",  # 标签；报告另有实际源码SHA作为依据
)
```

输入由nibabel读取NIfTI/MGH，三维FP32解析、非负、每轴≥8；固定recipe限定见[完整N4](N4_COMPLETE_TORCH_20261009.md)。输出 `(X,Y,Z)` uint8、原affine/mm和体素尺寸；MGH输入保持MGH类，NIfTI保持其类。两个输出须不存在，三路径不同。`device`必填且为cuda:N，不回退CPU；`threads`正整数默认4；`profile`默认False；`code_version`默认FNIT-source-hashes，不代替哈希。

返回`api`含200轮/shape/dtype/同步总时间，`input_sha256/output_sha256/source_sha256`、实际`cuda_allocator/precision/threads`、worker PID与GPU、`allocated_peak_bytes/reserved_peak_bytes`，以及父API新增的 `parent_cuda_initialized_before_exec/isolated_cli_wall_seconds/isolation`。完整外部墙钟含exec、导入、哈希、校验、初始化、加载、传输、计算、压缩写出及子退出。父GPU张量仍保留；子张量峰值不等于父子同期占用。

`run_worker()` 是新进程CLI内部入口，默认cuda:0、4线程、不开剖析，matmul/cudnn TF32默认True；父API传入父进程实际策略。CUDA已初始化时拒绝更改分配器。无FP16/BF16，算法既有FP64拟合/归约与FFT例外保留。非法资源/路径、CUDA、完整N4、exec失败或哈希错误抛异常，可能留部分输出；不改默认、不复制参考、不调用原生命令。

生产父进程的 `FNIT_RECON_FORBID_ORIGINAL_STAGES=1` 由exec继承。CLI与worker入口现在复用[程序守卫](WORKER_PROGRAM_GUARD.md)，在解析参数、校验或CUDA初始化之前安装；正常和异常退出恢复作用域。禁用或缺失标志时保留独立benchmark行为；活动父守卫不会被嵌套入口解除。尝试直接执行原软件阶段程序时抛 `RuntimeError`，不改变200轮N4、缓存策略、线程或TF32。

## 3．命令行与复现

```bash
python -m fnit.recon_all.n4_torch_worker \
  --input subject/mri/orig.mgz \
  --output diagnostic/nu0.mgz \
  --report diagnostic/n4-worker.json \
  --device cuda:0 \
  --threads 4 \
  --matmul-tf32 1 \
  --cudnn-tf32 1 \
  --code-version frozen-code-label
```

两项TF32参数只接受0/1，默认1；`--profile`可选；其余同Python。CLI是独立新进程，缓存只在该进程开启。主页Conda已声明全部依赖，未新增生产包。

`benchmark_input_n4_worker.py` 接受 `--raw-pair` 完成的原始链报告/输出、`--output`新目录、`--source-root`当前冻结源码、`--previous-source`原始链冻结源码、`--profiling-module`固定采样器、`--device cuda:0 --threads 4`。先冷CLI两例（父CUDA未初始化），再已初始化父API两例；父保留该例真实orig的64MiB FP32 GPU张量，不用模拟影像当benchmark。先证明两条原始链的orig数据/几何相同，再做缓存策略同输入对照。参考只用于输出完成后的比较。

整例显式候选示例（其余步骤仍是混合流程）：

```bash
fnit-recon-all data/sub-07_T1w.nii.gz runs/sub-07 \
  --weights-dir resources/weights \
  --assets-dir resources/assets \
  --native-bin-dir resources/native/bin \
  --device cuda:0 --threads 4 \
  --n4-backend torch --n4-execution isolated
```

新增参数在创建输出前检查；完整输入/输出和其余参数见[recon-all](README.md)。本轮整例harness从main首行计时，包含解析、前置校验、资源哈希、子CLI启动/退出及影像读写；父解释器初始导入和当前最终元数据发布不在该值中，旧803报告的时间/范围保持原样。

## 4．对应原实现

固定ITK5.4.7 N4与recon-all包装器见[ITK Conda实现](N4_ITK_CONDA.md)。本接口只执行完整N4到nu0，后续全局缩放/Talairach直方图由 `make_nu()` 完成；不能把nu0和nu混称。原软件参考命令是固定参数 `N4BiasFieldCorrection`，本新exec接口属于执行策略，没有另外的原软件CLI。

## 5．两例真实数据的精度、耗时与资源

A100-SXM4-80GB，同一CPU0–3/4线程，TF32开启，无半精度。输入来自[两例原始T1→nu](INPUT_N4_CHAIN.md)各自产orig，逐SHA核验；不是原始空目录整例。

| 病例 / 调用方式 | 含exec/退出完整墙钟，s | 子完整N4文件API，s | 迭代 | 对cache-off完整实现新增体素差 |
|---|---:|---:|---:|---:|
| sub-06 冷CLI | 11.037 | 5.106 | 200 | 0 |
| sub-07 冷CLI | 9.978 | 4.317 | 200 | 0 |
| sub-06 已初始化父API | 10.749 | 4.291 | 200 | 0 |
| sub-07 已初始化父API | 11.041 | 4.589 | 200 | 0 |

四组nu0所有体素、affine、shape、dtype均保持。相对同输入native仍为4016/3259体素全+1，严格复现未过；没有把已有偏移掩盖或称整体等效。初始化父API的完整N4阶段分别由91.105→10.749s、76.874→11.041s，观察缩短88.2%/85.6%；这里仅算固定输入N4，不能据此从历史recon-all时间相减宣称整例提速。

每例/方式各一次，固定线程预算下的观察不是重复吞吐门；`profile=False`内部子段未逐次同步，不单独声称kernel加速。新的exec完整边界包含导入和退出，直接API阶段含加载、传输与写出。

子峰allocated/reserved四组同为1,295,297,024/1,384,120,320字节。0.5秒同期采样实际最大间隔3.380s，观测目标卡峰3,021,996,032字节、所有计算进程上界3,001,024,512字节；查询时刻不同且含共享负载。容器/驱动PID归属未解决，父子树峰值null，不能写0，也不保证捕获连续瞬时峰。没有隔离干净环境或完整recon-all指标等效结论。

[完整机器报告](../../validation/recon_all/optimizations/20261009_n4_torch_substages/reports/input_chain_a100_20261009_v2/cached_worker/summary.json)保留四组子收据、两例输入/output SHA、200次迭代、实际精度/线程及采样序列。[源码manifest](../../validation/recon_all/optimizations/20261009_n4_torch_substages/reports/input_chain_a100_20261009_v2/source/worker_v2_patch_manifest.json)与docstring-only AST桥接绑定实际v2，不把旧冻结v1覆盖。[报告索引及导出校验](../../validation/recon_all/optimizations/20261009_n4_torch_substages/reports/input_chain_a100_20261009_v2/README.md)记录26份白名单收据原件/公开SHA；没有影像、权重、许可证或服务器地址。

新增生产显式路由的两例原始空目录输入链已完成：到nu为71.881/76.739s、N4含exec为13.556/11.962s，所有体素/几何/LTA对此前完整Torch零新增差。与旧组共享负载不同，不宣称同期ABBA或整例收益；native系统差异原样保留。详见[输入链新回归](INPUT_N4_CHAIN.md#显式isolated输入链的新空目录回归)和[v4报告](../../validation/recon_all/optimizations/20261009_n4_torch_substages/reports/execution_wiring_a100_20261009_v4/README.md)。GPU卡峰约24.9GB包含其他任务，tree归属null；子张量峰约1.3/1.4GB不能替代整例合计显存。

完整原始T1连续链到filled的后续验证已发现下游放大：nu0仅4016/3259个量化差异，但filled差4671/2291体素，sub-06注册与归一化出现更大局部差异。见[两例逐阶段实际报告](../../validation/recon_all/optimizations/20261010_n4_continuous_prefix/README.md)。这不改变上述同输入worker精度/性能证据；生产默认仍ITK，不能以阶段提速或P99为零直接接管整例。

### 2026-10-11 guarded wheel两例同输入新回归

固定源码9292a0a7、wheel SHA 3e5e2eb0332d328cd92a968a17ae8cc2c6d872a961d103bca4a9d0d65df39e51；独立target安装623个Python文件与wheel一致，29关键模块与源码一致。旧worker所用FNIT自产orig现场核对SHA；本轮不使用官方上游影像，完整N4与B-spline模块SHA保持旧执行版本。

| 病例 / 方式 | 完整外部墙钟 s | 子N4文件API s | 轮次 | 对旧完整Torch及两种旧worker |
|---|---:|---:|---:|---|
| sub-06 冷CLI | 7.522 | 3.525 | 200 | 全体素/affine/dtype/文件SHA一致 |
| sub-07 冷CLI | 7.820 | 3.574 | 200 | 全体素/affine/dtype/文件SHA一致 |
| sub-06 已初始化父API | 8.428 | 3.411 | 200 | 全体素/affine/dtype/文件SHA一致 |
| sub-07 已初始化父API | 8.660 | 3.542 | 200 | 全体素/affine/dtype/文件SHA一致 |

四组退出0，旧4016/3259体素+1差异保留。GPU6、CPU88–91及4线程；两项TF32开启，无autocast/半精度。旧空卡门WAIT保留；用户授权后先写共享资源协议，排除GPU4，同UUID两次间隔1.007秒满足利用率≤10%、free≥32GB，再取得锁并启动前复查。当前为非独占，各一次共享墙钟观测不用于整例提速结论。

父API在已初始化CUDA下保留64MiB真实orig FP32张量，取得14/13个同快照父子行，已验证两行合计观测最大2,401,239,040字节。实际最长间隔0.730秒，查询无失败；驻留外部进程仍unresolved，全序列tree峰保持null。子Torch峰为1,295,297,024/1,384,120,320字节，父cache-off的allocated/reserved零值无效。只报告已验证窗口，未验证完整重建20GB/10min。

四个实际子Python启动均有审计，未观察原阶段命令；六项已安装worker/CLI守卫CPU负向合同通过。协议先于四子启动，所有本轮父子driver行在正常退出后消失，其他共享进程保留。详细输入、边界、脑图SHA和[完整本轮报告](../../validation/recon_all/optimizations/20261011_guarded_n4/README.md)列出全部证据；旧回执及测量版本不改。

## 6．更新和验证记录

2026-10-11：补齐可选Torch N4 exec的继承守卫，与GCA/WM/Talairach使用同一接口。新合同逐一验证标志缺失、0、1时CLI解析和第一处CUDA检查的程序拦截、状态恢复及输出未创建；它们与既有worker和完整EM接线合同共36项通过，14.10秒。没有在本次守卫修复后重跑真实N4；上面的四次MRI计时继续绑定其历史冻结源码，不能改标为本版GPU结果。

2026-10-09复用完整N4新增exec策略；不改原核心数学或全局allocator。六项worker契约检查参数/父策略/exec/输出哈希/初始化状态，与四条原始输入链契约合计十项2.91s通过。两例两种父CUDA状态同输入回归完成，已有严格N4差异独立记录。新增`n4_backend="torch", n4_execution="isolated"`已显式接入输入链、单例/批次recon-all API、fnit-recon-all CLI和整例测评脚本。native/in-process默认不变，纯python-gpu全流程仍由完整能力门控制。原始T1到nu的两例回归另见[输入链](INPUT_N4_CHAIN.md)，不是完整recon-all测评。

## 7．参考文献和源码

Tustison等，N4ITK，IEEE TMI，2010。[DOI](https://doi.org/10.1109/TMI.2010.2046908)。[固定ITK源码](https://github.com/InsightSoftwareConsortium/ITK/blob/v5.4.7/Modules/Filtering/BiasCorrection/include/itkN4BiasFieldCorrectionImageFilter.hxx)。FNIT完整数学实现与数值限制见[完整N4](N4_COMPLETE_TORCH_20261009.md)，执行策略复用项目既有GCA/WM新exec与同期采样方案，无额外原软件代码复制。
