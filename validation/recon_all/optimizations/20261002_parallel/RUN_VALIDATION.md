# 本轮验证入口

本目录中的脚本用于开发验证，不改变 recon-all 算法。基线提交和验收边界见
[验证协议](BENCHMARK_PROTOCOL.md)。真实数据结果完成后单列发布；脚本语法检查、
任务已启动和服务器已连接不能写成整例通过。

五会话的共享登录、SCP 和计时排队方法见[服务器连接说明](SERVER_CONNECTION.md)。

## 原始 T1 整例

`execute_whole_case.py` 从原始单 T1 启动已冻结源码。输入 JSON 的字段如下。

| 字段 | 含义及限制 |
| --- | --- |
| `python` | 主页 Conda 环境中的 Python 绝对路径 |
| `code_root` / `code_commit` | 已展开源码目录和实际测试的完整 Git 提交 |
| `source_archive_sha256` | 该源码归档的 SHA-256；传输后先另行核验 |
| `input` | 原始真实 T1，NIfTI 文件；保持原始网格 |
| `output` | 新被试目录，运行前必须不存在 |
| `weights` / `assets` | 已按固定清单验证的权重和数据资产目录 |
| `native_bin_dir` | 固定源码在 Conda 独立编译的程序目录 |
| `device` / `gpu_uuid` | 逻辑 CUDA 设备和唯一物理 GPU UUID，进程启动前绑定 |
| `threads` | 本轮固定总 CPU 线程预算为 4 |
| `invocation` | `cli` 或 `initialized_cuda_api` |
| `fs_license` | 用户本机许可证路径，仅检查存在，不读取或复制内容 |
| `diagnostic_root` | 新诊断目录，运行前必须不存在 |
| `pipeline_kwargs` | API 的具名参数字典，默认空；不得覆盖输入或资源边界 |
| `pipeline_cli_args` | CLI 的显式选项数组，默认空 |

API 模式在调用前配置既有低显存分配器策略，建立目标卡上的 float32 单元素张量，
再调用同一生产函数，覆盖已初始化 CUDA 的使用方式。CLI 在新进程中运行。
影像读写、conformed/scanner/surface RAS 约定沿用生产接口；本脚本不重采样影像。
两种模式不启用 autocast，不更改 TF32/已验证 FP32 策略。

输出 `launch.json`、`command.log` 和 `completion.json`。时间单位秒，
`command_seconds` 包含本入口校验、子进程加载、传输、计算与输出读写；质量比较和
绘图在完成后单独运行。启动或运行失败非零退出，保留失败报告和日志。锁等待在
外层，不能计入算法墙钟。官方参考路径不出现在生产配置中。

```python
import json
from pathlib import Path
import subprocess
import sys

configuration_path = Path("/data/benchmark/sub01_api.json")  # 预先准备的完整具名配置
configuration = json.loads(configuration_path.read_text())  # 上表字段逐项填写
assert configuration["invocation"] == "initialized_cuda_api"  # 已初始化 CUDA 的 API
subprocess.run(
    args=[sys.executable, "validation/recon_all/optimizations/20261002_parallel/execute_whole_case.py",
          "--config", str(configuration_path)],  # 本入口只读取给定配置
    check=True,  # 失败明确抛出，禁止手动补跑后标为连续整例
)
```

```bash
# 同一服务器上的所有性能测试使用同一个本地文件锁；不要在不支持 flock 的 GPFS 上建锁。
flock /tmp/fnit-recon-benchmark.lock \
  python validation/recon_all/python_gpu_port/run_monitored.py \
    --gpu-uuid GPU-实际目标卡UUID \
    --output /data/benchmark/sub01_api_monitor \
    --interval 2 --query-timeout 5 -- \
  python validation/recon_all/optimizations/20261002_parallel/execute_whole_case.py \
    --config /data/benchmark/sub01_api.json
```

监测器输出每次查询中父子进程合计字节数、采样间隔/缺失样本和全命令墙钟。
周期采样不能证明连续峰值。缓存关闭时 PyTorch allocated/reserved 可能不可用，
不可写成零显存。预算为 20,000,000,000 字节；GB 为除以 1e9，GiB 为除以 2**30。

多例顺序验证使用 `run_whole_queue.py`。`--configs` 按给定顺序列出完整配置，
`--lock` 为服务器本地的共用锁，`--output` 必须是新队列目录，`--monitor` 和
`--launcher` 为上述两个脚本的绝对路径。每例分别取得锁；上一例失败后保留
非零状态并继续独立的下一例，最终队列也非零退出。`queue.json` 包含脚本和配置
SHA、命令、状态及监测器的整例墙钟；锁等待和包装器开销另列，不计为算法时间。

```bash
# 两份配置分别声明原始T1和新输出；不复用中途检查点。
python /code/validation/recon_all/optimizations/20261002_parallel/run_whole_queue.py \
  --configs /data/benchmark/sub01_api.json /data/benchmark/sub02_cli.json \
  --lock /tmp/fnit-recon-benchmark.lock --output /data/benchmark/queue_baseline \
  --monitor /code/validation/recon_all/python_gpu_port/run_monitored.py \
  --launcher /code/validation/recon_all/optimizations/20261002_parallel/execute_whole_case.py
```

## 既有主页环境中的安装回归

`validate_installation.py` 复用当前 Conda 的编译器构建 wheel，安装到新 `target`
目录，运行 CLI 帮助并确认 API 导入来自这个目录。输入 JSON：

| 字段 | 含义 |
| --- | --- |
| `code_root` / `code_commit` | 本次源码目录及提交 |
| `source_archive` / `archive_sha256` | 已传输归档及预期 SHA-256，实际读取并比对 |
| `output` | 新安装诊断目录，必须不存在 |
| `modules` | 可选 FNIT 模块名数组；默认检查 recon-all 和归一化 CUDA 扩展的导入 |

```bash
# 使用待验证的主页 Conda 环境，不下载依赖或覆盖已安装版本。
python validation/recon_all/optimizations/20261002_parallel/validate_installation.py \
  --config /data/benchmark/install_candidate.json
```

输出 `report.json`、`wheels/`、`installed/` 和每个命令的日志。报告包含归档、wheel、
配置/脚本/日志的 SHA-256、返回码、秒数与实际导入路径。缺少 Conda 编译器、归档
哈希不符、编译失败或导入了目标目录之外的包均非零退出。该检查不分配 GPU 张量，
不代表重新创建 Conda、重编全部原生程序或没有预装脑影像软件的整例隔离验证。

## 原软件、结果和参考

这些是 FNIT 验证包装器，没有独立等价的 FreeSurfer 命令。阶段和整例语义分别参见
[recon-all 说明](../../../../docs/recon_all/README.md)及其 `recon-all -s SUBJECT -i T1 -all`
参考；参考程序仅在 benchmark 环境运行。生产仍使用 FNIT 自产结果和声明资源。

本轮安装产物与原始T1整例按实际状态记录，最终结果在整例和质量比较完成后发布；最新既有整例证据保留其原提交、时间和范围，
见[前轮结果](../20261001_serial/FINAL_RESULTS.md)。本轮完成后以实际提交发布新结果，
不把历史报告重标为本轮。原实现：[FNIT](https://github.com/weikanggong1/Fudan-Neuroimaging-toolkit)、
[FreeSurfer](https://github.com/freesurfer/freesurfer)。参考：Fischl B. FreeSurfer.
*NeuroImage*. 2012;62:774–781，doi:10.1016/j.neuroimage.2012.01.021。

## 安装完成后的自动接续

`run_installed_candidate.py` 只用于本轮验证。它等待私有原生安装的
`status.json=passed`，逐项检查15个程序的路径与SHA；13个复用程序必须与
基线相同，GCA绑定实际重新编译后的SHA；white快速程序复用任务2已验证的独立源码构建
产物，记录重定位前后SHA。最终wheel的171个
recon-all Python文件必须已逐SHA匹配冻结源码；运行入口指向安装目标目录。
随后在同一个主机锁内验证实际能力选择和输入、权重、资产，并复用整例队列。

`--round` 是本轮服务器目录；`--python` 是主页Conda Python；
`--lock` 是共同主机锁；`--resource-script` 是已有的
`../20261001_serial/capture_serial_resources.py`。该脚本不读取官方被试数据，
不修改精度和算法。安装失败、能力不符或SHA变化时停止；依赖等待最多6小时。
等待与资源审计不计入整例墙钟。

```bash
# 私有安装完成后自动核验资源，顺序执行API与CLI两份完整配置。
python validation/recon_all/optimizations/20261002_parallel/run_installed_candidate.py \
  --round /data/benchmark/parallel_20261002 \
  --python /opt/conda/envs/fnit/bin/python \
  --lock /tmp/fnit-recon-benchmark.lock \
  --resource-script /code/validation/recon_all/optimizations/20261001_serial/capture_serial_resources.py
```

## 完整耗时汇总

`summarize_performance.py` 只读已回收的两例原始JSON，复用前轮`pair`、
`binding` 和半球内部报告提取。检查输入、主机、设备UUID、4线程预算、
API/CLI调用方式、精度和执行状态；不符合时明确失败。输出完整JSON及每例阶段CSV。
表面、球面配准、注释和最终放置按基线串行父阶段之和与候选父组墙钟比较；
候选私有拷贝、发布、依赖串行收尾也计入对应范围。worker耗时不重复求和。

```bash
# 两例baseline与candidate完成并回收后执行；输出目录必须不存在。
python validation/recon_all/optimizations/20261002_parallel/summarize_performance.py \
  --reports validation/recon_all/optimizations/20261002_parallel/whole \
  --output validation/recon_all/optimizations/20261002_parallel/whole/performance
```

## CUDA 首次初始化失败的独立诊断与空目录复测

第二例首次候选整例的失败终态保存在 `whole/candidate_sub02_failed_v1/`，
它不能当作完整运行。RH 在首次 CUDA 张量分配时 OOM，LH 被取消，表面组
`published=[]`。4 字节是张量有效大小，不能代表首次初始化的全部资源开销。
外部负载和同期显存样本已保留；当前未确定根因，未修改精度或分配器。

`diagnose_cuda_bootstrap.py` 使用失败整例的 JSON 配置，冻结安装入口、GPU UUID
和 disabled allocator，运行 single、双进程、分阶段 single、双进程四批冷启动。
每个进程只使用2个CPU线程，同时最多两个；复刻首次分配顺序，但v1未导入生产worker的thread_budget（其顶层导入Numba），因此不是完整导入顺序对照。
分阶段探针额外分列 init、set_device、properties、memory_info、allocation、
synchronize。原顺序探针在首次分配前不查询 CUDA memory_info。诊断不读取 T1、
官方输出或模型，不改变生产实现；观察成功不构成可靠性保证。

```bash
# --config：失败整例的冻结配置；--output：必须不存在的诊断目录。
# --lock：与整例相同的主机本地锁，避免与已声明验证任务争用。
python diagnose_cuda_bootstrap.py \
  --config /data/benchmark/coordinator/candidate_sub02_8d750e2.json \
  --output /data/benchmark/cuda_bootstrap_diagnostic_v1 \
  --lock /tmp/fnit-recon-benchmark.lock
```

主入口默认不是 child/staged 模式；内部 `--child` 只运行一次探针，
`--staged` 只为该探针增加分阶段初始化。每个 JSON 记录 PID、UTC、线程、
明确环境、PyTorch版本、失败phase和traceback；总报告还保存同期 NVML GPU/
进程快照与脚本/配置SHA。任何探针失败均非零退出，不静默重试。

`run_candidate_retry_v2.py` 的全部参数为必需的 `--round`（本轮目录）、
`--python`（同一Conda Python）、`--lock`（共同主机锁）。它最多等待诊断8小时；
诊断失败即停止。通过后逐SHA复核171个安装源码和15个原生程序，确认新配置
仅改变 output/diagnostic_root，再调用既有队列，从原始T1和新空目录执行一次。
旧失败目录不恢复、不复制为生产输入。输出 coordinator 的 retry 状态/队列，
整例入口及 monitor 仍分别输出 launch/completion/run/显存CSV。等待和诊断不计入
重建墙钟；任何重建失败均明确保存，不能拼接失败前缀。

```bash
# --round：含冻结配置、已安装wheel与私有native bundle的本轮目录。
# --python：实际安装验证通过的Python；--lock：同一主机资源锁。
python run_candidate_retry_v2.py \
  --round /data/benchmark/parallel_20261002 \
  --python /opt/conda/envs/fnit/bin/python \
  --lock /tmp/fnit-recon-benchmark.lock
```

这是执行和故障诊断包装器，没有独立 FreeSurfer 等价科学计算命令；完整流程
语义仍为 `recon-all -s SUBJECT -i T1 -all`。实际执行SHA、失败尝试和成功尝试分别
保留；本节不能替代实际终态或新的官方重复性结果。两脚本只用Python标准库和
主页已声明PyTorch，未新增依赖。


### 含生产导入顺序的配对初始化诊断

`diagnose_cuda_bootstrap_pairs.py` 的外部参数仍为必需 `--config`、`--output`、
`--lock`；输入格式及秒/字节单位同上。输出八批独立probe报告、ready ACK、
release gate、NVML快照和总summary。各新进程在首次CUDA前导入实际worker的
`thread_budget`（含Numba），不进入表面算法。

前四批按parallel、serial、serial、parallel执行，均保留disabled cache；
serial先等LH实际分配及同步ACK，保持该进程/context存活，再启动RH，
两侧均ready后统一放行。另两批测试分阶段初始化、两批测试enabled cache；
enabled通过删除环境变量启用，只用于诊断。所有批次失败或取消仍保存，
不根据成功选择性删除失败记录，也不构成生产自动重试。内部 `--child`、
`--staged` 默认关闭，`--allocator` 默认disabled，`--ack`、`--gate`为本批独立
文件；等待ACK/gate上限90秒。总summary的complete仅表示诊断批次执行完毕，
`both_initialized`和各probe status才是实际初始化结果。

```bash
# --config：原失败整例的冻结入口；--output：另一新目录，保留前次失败诊断。
# --lock：继续使用共同主机资源锁，诊断耗时不计入整例。
python diagnose_cuda_bootstrap_pairs.py \
  --config /data/benchmark/coordinator/candidate_sub02_8d750e2.json \
  --output /data/benchmark/cuda_bootstrap_pairs_v2 \
  --lock /tmp/fnit-recon-benchmark.lock
```

本次v1实际为single通过、第一批parallel两个进程均失败；失败前/后NVML
可用显存分别70119/69382MiB，不能据此归因于容量耗尽。它不证明根因，也不
证明Numba是否参与；后续配对测试的完整原字节报告单独保留。


### 当前采用的第二例空目录尝试

完整v2初始化对照八批、16个fresh进程全部初始化成功，包括原并发顺序。
这次未分辨出更可靠策略，因此未将串行ACK候选或缓存开启改为生产默认；
运行源码仍为`8d750e2`。`run_candidate_retry_v3.py`复用v2的全部输入/输出/失败
规则，前提改为pairs_v2的八批全部`both_initialized=true`。此前v2监督在启动前
停止，未创建整例输出。本次配置为`whole/configs/candidate_sub02_retry_v3.json`，
其SHA为`fac6fd6dffd6a74b2d925b3a190df2c7517c857a613f385e3e2aba5c657066d9`；
仅改变两个输出路径，输入、版本、GPU、线程、精度与该例首次候选一致。

```bash
# 读取既有pairs_v2完整终态；再次核验安装171源码与15程序SHA。
python run_candidate_retry_v3.py \
  --round /data/benchmark/parallel_20261002 \
  --python /opt/conda/envs/fnit/bin/python \
  --lock /tmp/fnit-recon-benchmark.lock
```

服务器实际监督UTC19:09:10启动（PID33588），脚本SHA
`b9781f3f6fa6c96dfc9524282015c8928cde2fe265b74d1b59ff5a770ad129c0`。
它实际复用尚未创建过的`candidate_retry_queue_v2`目录及同名日志；这是监督
输出名，不能据名字判断整例尝试版本。真实整例目录带`retry_v3`，各launch和
monitor记录实际argv/配置SHA。后续三方比较使用`config_retry_v4.json`与新诊断
目录，只读两份候选均成功完成的输出；执行终态与资源/数值判定仍分列。
