# 两例原始 T1 的整例速度配对

## 1. 功能与流程

`tools/benchmark_recon_native_speed_pair.py` 在同一节点、同一 GPU 与线程预算
下比较两份冻结 FNIT 源码。首例依次执行基线 A、候选 B；次例依次执行
候选 B、基线 A。四次均从原始 T1 和新空目录开始，分别保留完整产物。
候选额外选择已有的 `--sphere-normals-backend torch`；其他算法参数完全
相同。源码本身包含的注册优化由各自提交和文件 SHA-256 绑定。

```mermaid
flowchart LR
    I[两例原始 T1 与两份冻结源码] --> V[全部哈希及路径预检]
    V --> C[新建两臂独立空缓存]
    C --> A1[首例：A 原始 T1 整例]
    A1 --> B1[首例：B 原始 T1 整例]
    B1 --> B2[次例：B 原始 T1 整例]
    B2 --> A2[次例：A 原始 T1 整例]
    A2 --> R[退出、完整耗时、原报告与哈希复核]
```

工具调用各源码已有的 `run_recon_native_end_to_end_cases.py`，不重写重建。
某次失败保留现场、继续其余预定运行，最终非零退出；没有重试、检查点
续跑或补跑。它不读取官方参考，也不负责给出指标等效结论。

## 2. Python 调用、全部输入与输出

病例清单是两项 JSON 列表，每项包含 `case`（唯一目录名）、`t1`（原始
影像路径）、`sha256`（该文件实际 64 位小写十六进制 SHA-256）。两项
影像内容须不同。原始影像格式、坐标几何及完整流水线输入检查由现有
benchmark 完成；本工具只验证文件哈希，不修改影像或坐标空间。

`dependency_pythonpath_json` 是依赖目录字符串列表，例如声明的独立
Python 包安装目录；这些目录不能含另一份 `fnit` 包。每个子进程使用
“声明依赖 + 该臂 src/ + 该臂 tools/”，不继承旧 `PYTHONPATH`。
`common_arguments_json` 是两臂共同的 argv 字符串列表，例如
`["--hemisphere-execution", "groups", "--hemisphere-workers", "2"]`。
使用已经通过阶段验证的完整 native fast 配方，不用省略步骤换取时间。

```python
from pathlib import Path
import sys

# tools 目录仅用于导入编排工具，不改变子进程显式绑定的源码。
sys.path.insert(0, "/data/fnit_candidate/tools")
from benchmark_recon_native_speed_pair import run_pair

pair_report = run_pair(
    cases_json=Path("/data/private/two_cases.json"),  # 两例原始 T1 路径及实际 SHA
    output_root=Path("/data/results/new_whole_pair"),  # 尚不存在的新输出根
    baseline_source=Path("/data/frozen_baseline"),  # 含 src/ 和 tools/ 的旧源码
    baseline_version="baseline_commit",  # 该源码的实际提交
    candidate_source=Path("/data/frozen_candidate"),  # 含 src/ 和 tools/ 的新源码
    candidate_version="candidate_commit",  # 新源码的实际提交
    weights_dir=Path("/data/fnit_weights"),  # 已声明的相同模型权重
    assets_dir=Path("/data/fnit_assets"),  # 已声明的相同模板及图谱
    native_bin_dir=Path("/data/conda_native/bin"),  # 相同的独立构建程序集
    dependency_pythonpath_json=Path("/data/private/dependencies.json"),  # 依赖路径列表
    common_arguments_json=Path("/data/private/native_recipe.json"),  # 相同附加 argv
    python_interpreter=Path("/data/conda_env/bin/python"),  # 相同 Conda 解释器
    n4_backend="simpleitk",  # 默认 simpleitk；也可显式 native/torch
    n4_execution="in-process",  # 默认进程内；isolated 仅适用于 torch
)
```

| 参数 | 类型、含义与限制 |
| --- | --- |
| `cases_json` | 必填 `Path`；恰好两例的上述清单，全部 T1 在创建输出前验证。 |
| `output_root` | 必填 `Path`；必须不存在，含悬空符号链接的旧入口也拒绝。 |
| `baseline_source`、`candidate_source` | 必填 `Path`；各自包含 src/ 与两个现有整例工具的冻结源码根。 |
| `baseline_version`、`candidate_version` | 必填非空 `str`；实际代码版本标识，源码文件 SHA 另存。 |
| `weights_dir`、`assets_dir`、`native_bin_dir` | 必填 `Path` 目录；四次使用同一组声明资源，实际资源哈希沿用各 benchmark 报告。 |
| `dependency_pythonpath_json` | 必填 `Path`；JSON 路径列表，可为空列表；目录须存在，不能覆盖 FNIT 源码。 |
| `common_arguments_json` | 必填 `Path`；JSON argv 列表，可为空；禁止覆盖固定输入、后端、N4、设备、线程或法向分组。 |
| `python_interpreter` | 必填 `Path`；实际可执行解释器文件，记录文件 SHA。 |
| `n4_backend` | `str`，默认 `simpleitk`；接受 `native`、`simpleitk`、`torch`，两臂相同。 |
| `n4_execution` | `str`，默认 `in-process`；`isolated` 仅允许 `n4_backend="torch"`。 |

输出字典同时写入 `speed_pair.private.json`，包含四次 `runs`、命令、
输入与源码 SHA、解释器及环境边界、实际缓存位置、所有退出码和原报告
路径/哈希。每次 `runs` 明确区分：

- `outer_supervisor_wall_seconds`：启动单例监督器之前至其退出，含其校验、
  benchmark、读写及最终报告；单位秒。
- `supervisor_case_outer_wall_seconds`：现有监督器记录的 benchmark 解释器
  启动至退出墙钟，单位秒。
- `benchmark_cli_wall_seconds`、`benchmark_full_harness_seconds`：逐项复制
  该次实际 `benchmark.json`，不引用历史整例时间。
- `benchmark_summary`：现有执行、完整输出、网格质量和显存原摘要；本工具
  不将缺失或显存归属不明的值改成零，不将退出 0 改称指标等效。

四套完整流水线结果分别位于 `runs/<索引_病例_分组>/<病例>/`。输入副本
清单位于 `plans/`；`cache/baseline/` 与 `cache/candidate/` 各有初始为空的
Numba/Triton 缓存，两臂不共享，各自可在第二例复用。原始影像、surface
RAS/mm、标签及单位全部沿用现有生产流程，不在编排层转换。

参数、路径或哈希错误在输出创建前抛异常。子进程、原报告或运行后完整性
检查失败保留记录，状态为 `completed_with_failures`；CLI 非零退出。
`complete` 只表示四次执行与报告绑定完成，误差及指标等效仍需事后评估。

## 3. 命令行调用与资源边界

```bash
python tools/benchmark_recon_native_speed_pair.py \
  --cases-json /data/private/two_cases.json \
  --output-root /data/results/new_whole_pair \
  --baseline-source /data/frozen_baseline \
  --baseline-version baseline_commit \
  --candidate-source /data/frozen_candidate \
  --candidate-version candidate_commit \
  --weights-dir /data/fnit_weights \
  --assets-dir /data/fnit_assets \
  --native-bin-dir /data/conda_native/bin \
  --dependency-pythonpath-json /data/private/dependencies.json \
  --common-arguments-json /data/private/native_recipe.json \
  --python-interpreter /data/conda_env/bin/python \
  --n4-backend simpleitk \
  --n4-execution in-process
```

每个具名 CLI 参数与上表 Python 参数对应。GPU 租约、目标 UUID、CPU 总
四线程、亲和性、TF32/FP32 例外、allocator 策略及许可证由外层现有租约
入口控制，本工具不修改。现有单例监督器固定 `cuda:0` 和 `--threads 4`；
该逻辑设备须映射到外层选定的同一物理卡。父子进程同时显存由原 benchmark
采样，最终需核验其实际目标设备、归属和失败采样，不能只读取配置字段。
报告不保存许可证路径或内容。

## 4. 对应原软件调用

该配对编排没有独立官方命令。单例参考是 `recon-all -i T1 -s SUBJECT -all`。
官方程序只能在独立参考路径运行，不进入四次生产输入。原生可选程序来自
已固定源码的独立 Conda 构建；不调用系统预装 FreeSurfer。

## 5. 最新精度、时间与脑图

本工具于 2026-10-17 新增。轻量编排单测覆盖反序、各臂空缓存及复用、
失败无重试、输入哈希门和源码变更门；这些不是真实影像 benchmark。
本轮四次原始T1新空目录运行及四项事后评估均已完成，科学候选为71cc、
基线5ca。CLI观察为2398.175→2299.843秒、2198.098→2131.620秒；同期
138数值门通过，局部GPU图仍有零容差尾差，自有显存归属未完成。
[完整结果、实际阶段账及脑图](../../validation/recon_all/optimizations/20261017_sphere_normals/WHOLE_RESULTS.md)
分别记录严格138、Dice、双向表面距离、脑区量和局部异常。历史2,000秒级
时钟不替代本轮基线；该工具没有设置或放宽等效阈值。

## 6. 更新与 benchmark 记录

- 2026-10-17：增加两例、四次原始 T1 的同节点反序编排；复用现有单例
  监督器，绑定两臂源、缓存和实际计时；四次运行与四项评估已完成，见上节。
- 上版冻结整例仍见 [2026-10-16 原生热点报告](../../validation/recon_all/optimizations/20261016_native_surface_hotspots/README.md)，其提交与时钟单独保留。

## 7. 参考

- [FNIT 单例监督器](../../tools/run_recon_native_end_to_end_cases.py)
- [FNIT 整例 benchmark](../../tools/benchmark_recon_torch_end_to_end.py)
- [FreeSurfer recon-all](https://surfer.nmr.mgh.harvard.edu/fswiki/recon-all)
- [FreeSurfer 源码](https://github.com/freesurfer/freesurfer)
- Fischl B. FreeSurfer. *NeuroImage* 62, 774–781 (2012).
