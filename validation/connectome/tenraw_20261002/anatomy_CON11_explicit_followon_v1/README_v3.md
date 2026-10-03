# CON11 独立官方 CPU 末例衔接

## 1. 功能与流程

此调度器只处理新下载的 `sub-CON11`。等待真实官方模型完成后，复用同轮官方 recon-all 与已完成的27项结构准备，执行官方 DWI–T1 配准、八 atlas 转到 DWI、五次100k追踪与下游矩阵。数值工具保持原冻结字节；原九例和旧 CON11 等待目录不改。

```mermaid
flowchart LR
    P[新来源官方 TOPUP/SynthStrip/EDDY CPU] --> M[原官方模型 worker 完成合同]
    R[同轮 fresh FS 与27项已完成准备] --> A[原 cb06 解剖完成 worker]
    M --> V[raw与39源身份/SHA/实际exit核验]
    V --> A
    A --> T[原0c6191五种子100k/八atlas参考]
    T --> H[实际合同/manifest/launch交接]
```

默认隐藏GPU；完成阶段每例CPU8，只有一个新病例。官方追踪保留原 `-nthreads 0`，下游CPU8。此工具仅用于隔离 benchmark。

## 2. Python、输入、输出与参数

```python
from pathlib import Path
import hashlib
import os
import subprocess
import sys

followon_tools_directory = Path("/results/frozen-CON11-followon-tools")
configuration_path = followon_tools_directory / "configuration_v3.json"
configuration_sha256 = hashlib.sha256(configuration_path.read_bytes()).hexdigest()
cpu_environment = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "PYTHONDONTWRITEBYTECODE": "1"}

# 首先只读检查；去掉 --inspect-only 后等待真实合同并派发原冻结数值工具。
subprocess.run([
    sys.executable, str(followon_tools_directory / "run_CON11_official_followon_v3.py"),
    "--config", str(configuration_path), "--config-sha256", configuration_sha256,
    "--inspect-only",
], env=cpu_environment, check=True)
```

CLI参数：`--config`为冻结JSON；`--config-sha256`为其真实原字节SHA；`--inspect-only`只核验现有来源/准备/原队列；`--verify-model-only`另外核验实际 completed 模型、canonical raw/packing/39源身份，均不启动MRI或追踪。

配置中各项用途：

| 项 | 输入与含义 |
| --- | --- |
| `shared_root/case_id` | 实际共享根与唯一`sub-CON11`。 |
| `modeling_root` | 新官方模型根；只接受同例completed consumer、model report及实际freeze/config。 |
| `baseline_root/raw_root` | 同轮原baseline fresh FS与canonical BIDS T1/AP/PA/梯度/JSON。 |
| `known_inputs` | 27输出prepare、模板、只读reader、raw manifest、原参考binary manifest、packing/source/driver receipts的实际path/size/SHA。 |
| `frozen_sources` | 原anatomy cohort/worker、MRtrix controller/reference/helper/common、模型worker/wrapper/config/packing guard。 |
| `old_anatomy` | 原driver PID/start ticks/argv及九例completed、旧11waiting/无旧模型合同守卫；不发送信号。 |
| `state_root/anatomy_output/reference_output_root/reference_output/reference_dry_run` | 全新调度状态、单例解剖输出、单例参考输出、只读实际参考计划。均不能已存在或是link。 |
| `anatomy_python/reference_python` | 已核验环境解释器；数值子进程使用原配置解释器。 |
| `anatomy_source_commit` | 原冻结`cb06c0ca`，不改数值参数。 |
| `orchestrator_sha256` | 本调度器原字节SHA。 |
| `poll_seconds/wait_timeout_seconds` | 30秒轮询、21600秒有限模型等待；等待耗时单列。 |
| `reference` | MRtrix3.0.3、seed0–4、100000 attempted seeds、tracking0/downstream8；八atlas的节点数由实际合同定义。 |
| `environment/GPU/CPU_threads` | CUDA隐藏、线程8、现有Conda lib路径；GPU=false。 |
| `completion_claims` | 配置阶段均false；只有实际产物通过后状态报告升级。 |

输入影像均保留原始网格：DWI为`96×96×60×102`，梯度`3×102`；归一化WM FOD为45系数，5TT为5通道；八atlas为整数NIfTI及真实`nodes.tsv`。新解剖合同分别绑定原prepare和新DWI合同，不复制FNIT FOD/FA/mask。

输出调度目录包含`freeze/status`、实际model输入记录、分别sealed的anatomy/reference execution config、PID/start ticks/argv/源SHA的launch、实际exit/日志、成功后的`completed_handoff.json`。解剖输出沿原cohort结构保存新consumer；参考输出保存`reference_manifest.json`、seed0–4各自TCK/SIFT2/八atlas矩阵。三段时间独立，复用prepare不称新的连续冷调用。

## 3. 命令行

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 python run_CON11_official_followon_v3.py \
  --config configuration_v3.json --config-sha256 ACTUAL_CONFIGURATION_SHA256 --inspect-only

# 通过真实只读检查后，在新独立状态目录等待并执行。
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 python -u run_CON11_official_followon_v3.py \
  --config configuration_v3.json --config-sha256 ACTUAL_CONFIGURATION_SHA256
```

## 4. 原软件调用

复用原官方结构prepare。新complete仍使用官方FLIRT `-cost normmi -dof 6`、MRtrix transformconvert/transformcalc、八atlas最近邻mrtransform，具体完整argv由原worker报告保存。追踪保留iFOD2/ACT/GMWMI、`-seeds 100000 -select 0 -maxlength 250 -angle 45 -cutoff 0.1 -power 0.5 -samples 3 -trials 1000`，随后官方SIFT2、FA precise采样和endpoint/matrix构造。原入口及SHA见配置，不新增数值solver。

## 5. 本次状态与benchmark

v3已经真实只读核验CON11模型：13项官方命令全部exit0、29个产物及原39源通过；结构27项准备复用。尚未声明新解剖完成或追踪完成。已有27项prepare、旧九例完成及39源/输入身份先做真实只读检查；实际launch与readback会独立保存。六例rawprep已发布的数值/脑图见[官方CPU预处理](../task_01/official_rawprep_v1/README.md)。末例新数值由root完成交接核验后聚合，不借旧九例时间替代。

## 6. 更新记录

- v3：实际模型由上游新 `bf62ce93` metadata wrapper 完成，v3绑定其真实 alignment 指定的 dispatcher_v2；原 `616b3f01` 科学 worker、配置参数和原39源不变。旧 v2 waiter只等待旧入口，没有运行MRI；精确身份核验后退役，原报告保留。
- v2：使用原冻结 `f73e3840` cohort controller 调用原 `0c6191` 数值 worker；单例配置和真实 controller 状态按原逐病例来源 mapper 格式记录。
- v1：仅完成只读预检，未启动数值工作。原 configuration.json 与初始只读 receipt 保留；v2 采用新增文件，不覆盖 v1。
- 原v4/旧A、B控制器及旧CON11目录保持原历史，不补合同或alias。prepare原时长、等待、新完成和追踪分别记录。

## 7. 原实现与参考

- [原独立官方参考工具](../../../../../tools/reference/benchmark_connectome_raw_official.md)与[anatomy worker](../../../../../tools/reference/benchmark_connectome_anatomy_official.py)。
- [UKB-connectomics](https://github.com/sina-mansour/UKB-connectomics)、[MRtrix3](https://github.com/MRtrix3/mrtrix3)、[FSL](https://fsl.fmrib.ox.ac.uk/fsl/docs/)。
- [原始数据和许可](../task_01/README.md)。本次不复制官方程序、权重或MRI，不读取许可证内容。
