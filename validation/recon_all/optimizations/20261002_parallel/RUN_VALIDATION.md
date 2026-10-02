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

本轮新增入口尚待实际安装与整例运行；最新既有整例证据保留其原提交、时间和范围，
见[前轮结果](../20261001_serial/FINAL_RESULTS.md)。本轮完成后以实际提交发布新结果，
不把历史报告重标为本轮。原实现：[FNIT](https://github.com/weikanggong1/Fudan-Neuroimaging-toolkit)、
[FreeSurfer](https://github.com/freesurfer/freesurfer)。参考：Fischl B. FreeSurfer.
*NeuroImage*. 2012;62:774–781，doi:10.1016/j.neuroimage.2012.01.021。
