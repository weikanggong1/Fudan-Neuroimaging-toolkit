# 首批连续整例准备协议（2026-10-11）

## 1. 功能与流程

本记录只准备当前整合版的真实原始T1连续整例，不启动重建，不改生产调度，不下载或复制参考产物。先核对实际文件，再由总控完成组件和安装门，最后冻结集成commit。

```mermaid
flowchart LR
 A[固定十例清单与数据许可] --> B[实体输入与资源哈希核对]
 B --> C[完整EM/CA、GA/orig、final pial质量门]
 C --> D[冻结集成提交与独立安装身份]
 D --> E[重新核对同GPU、四线程和空目录]
 E --> F[两例顺序从原始T1连续运行]
 F --> G[138项、质量、精度和父子显存证据]
```

`protocol.public.json` 的启动门全部为false、执行commit为null。准备完成不意味着运行获准或算法验收通过。

## 2. Python读取与输入输出

```python
import json
from pathlib import Path

whole_run_protocol_path = Path("protocol.public.json")  # 只有typed配置，没有私有服务器路径
whole_run_protocol = json.loads(whole_run_protocol_path.read_text())
mandatory_gate_status = whole_run_protocol["hard_start_gates"]  # 每一项必须有独立通过证据
execution_source_commit = whole_run_protocol["execution_commit"]  # 冻结后填写完整40位SHA
ready_to_launch = all(mandatory_gate_status.values()) and execution_source_commit is not None
assert not ready_to_launch  # 当前协议仅准备，不调用重建API
```

`availability.public.json` 输入为已登记原始T1、十例固定清单、固定资源清单和许可元数据；输出为逐输入大小/SHA/影像头、资源校验与空目录状态。`release_binding.public.json` 将实际资源映射到科学包 `e9e89cac07cc65ef37f8c3223e53f03968f2af05` 的固定assets-v1清单，整文件匹配117项，另一项按分片字节范围校验。`deform_parts.public.json` 保存该分片结果。

当前CFFF实体核对：2/10例可用，即原名单中的ds000114 sub-06与sub-07；其余8例未在已登记CFFF输入目录找到。不能把原服务器的历史“下载完成”当成本机十例可用。固定十例名单不更换；首批拟用目前已核验的两例，其余仍保留原身份和待定位状态。

118/118资源与CFFF本地清单大小/SHA一致。两个固定数据快照的CC0元数据文件SHA一致。公开许可通知已定位并登记SHA；最终运行资源绑定仍须保留对应许可通知/bundle，本核查不会自动批准再分发。本准备不读取个人许可证、不分发资产。

私有bindings只保存在任务运行目录。两例新的subject目录已建立且为空；实际启动前须再次检查，不能复用旧v2/v3产物。磁盘可用容量为现场快照，启动前重新查询。

## 3. 命令行参数协议

本任务未创建可启动整例的launcher。未来固定installed CLI请求 `backend=python`、`topology_backend=python-ga`、`device=cuda:0`、总threads=4、`input_execution=parallel`、启用阶段测量；其余选项必须和最终冻结配方一致。两例顺序运行在同一物理GPU与CPU亲和性，开始前重新核对资源，不能占用其它正在运行冻结任务的预算。GPU和CPU目前仅建议，未预留。

未完成阶段必须报告具体名称并fail closed，不通过自动fallback、复制官方产物、减少必要步骤或调整阈值获得通过。完整EM、CA归一化、GA/orig和final pial质量门单独列出。

## 4. 原软件对照

准备协议无对应原软件命令。原软件只在独立参考环境或已有严格绑定的同输入证据中作为对照。生产进程只能读取原始T1、合法资源、installed Python/通用库和自己的空目录；不得读原版程序/阶段库、官方subject、旧冻结中间产物或旧整例结果。

## 5. 精度、耗时和显存边界

本次没有重建、精度或耗时benchmark。未来完整外层时间从fresh installed CLI进程启动前到返回、报告/138项写出、全部子进程回收及发布清理后；包含解释器、导入、模型/JIT、传输、计算和I/O。阶段墙钟、目标同步等待、目录复制/发布/回收、启动准入与父子CPU时间分别保留。缓存、allocator、冷热状态按实际记录，同条件比较，不从共享环境单次运行认定稳定加速。

复用 `ProcessTreeDeviceSampler`、NVML查询、worker精度记录、原版exec guard。每0.5秒一次快照，对同一时间父进程和所有存活后代显存求和；禁止相加每个进程历史峰值。须可信宿主driverPID到本地PID/启动身份映射；没有映射时本任务峰值为null。设备总量和设备计算进程和另列，不能替代自身峰值。报告采样间隔最大值、失败次数与归属覆盖率；采样可能漏瞬时峰值，不能据此宣称连续上限。

默认FP32/TF32及既有FP32例外，记录父/worker实际TF32、autocast、dtype、UUID和allocator，不自动float16/bfloat16。目标20,000,000,000字节；归属未解决时不能认证20GB。嵌套测量不重置外层峰值。Python直接exec guard不是完整文件/动态库隔离证据，独立OS级访问证明仍待安排。

138文件完整、网格质量、严格复现、引入退化和整体指标等效分开报告。脑图留私有目录，公共只报SHA。旧v2/v3使用早期不完整EM，不能改名为当前结果。两例不能代替最终十例验收。

## 6. 更新记录

2026-10-11：现场读取统一README/INDEX，复验已有两例、118项资源和固定CC0元数据，预留两个全新空subject目录；准备typed协议。未新增下载、未修改共享环境、未改生产或旧冻结源码、未启动完整运行。

待总控合入完整任务2/3/4并通过必要质量/算法门，再绑定最终commit、源码manifest、wheel、同主机独立安装、运行资源与许可通知；此时重新核查输入、预算、空目录及GPU监测归属。其它八例的实际CFFF路径须补充并逐文件验证，本任务不会自行传输或下载。

## 7. 来源

固定OpenNeuro ds000030 1.0.0与ds000114 1.0.2清单及CC0元数据SHA见可用性报告；固定Release目录及逐文件许可映射见release_binding报告。计算与参考文献沿用项目各功能页，本次没有新数值算法。
