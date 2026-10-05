# 已初始化 CUDA 的 API 入口绑定修正

`evaluated_role_bindings.py` 校验完整真实 T1 执行的来源、资源和入口证据，输出 `allocator_binding`，供单对 precision candidate 评估保留实际 allocator 状态。输入为 evaluator 配置、实际 retry 配置、launch、完成 pipeline 报告及已准入工具清单。它不计算影像差异，也不调整数值容差。

CLI 的 `gpu_memory_mode` 和 allocator `effective` 仍须是 `disabled`。`initialized_cuda_api` 则须保留报告中的 `preserved_preinitialized_unknown`；`requested=auto`、入口 CUDA 已初始化、环境选择前后均为 `1`、两项统计可信标记均为 false。环境变量 `1` 不能单独证明已初始化进程的 allocator 已禁用。

API 校验同时要求已准入的 `whole_case_driver` SHA 为 `6690d0e37682a024ef2daaa06d9e3c366ac2d41922905f1c0d39b3603ca93249`，launch 的 `script_sha256` 相同，并且 `command` 精确等于 `[python, driver_path, '--api-child', actual_config.resolve()]`。准入流程已逐文件核验工具、输入、源码、资产和权重的 SHA。冻结 driver 的 `api_child` 在同一函数调用中选择 allocator、创建保留的 4 字节 float32 CUDA tensor、同步，然后调用 `run_recon_all_python` 并保存 receipt。

本次执行的 `subject/run-api-invocation.json` 必须报告初始化成功、相同 GPU UUID（接受 receipt 缺少 `GPU-` 前缀）、严格整数 4 字节 retained tensor，以及初始化前 `requested/effective=disabled`、CUDA 未初始化、环境前后为 `1`、统计有效/不可用分别为 false/true。读取 receipt 前后 SHA 必须相同。输出保留 receipt 路径、SHA 和实际内容；现有 receipt 没有 child PID，校验结果不生成同 PID 的记录。

真实例为 ds000030/sub-10159 的 `sub10159_3a_api_complete_20261004_v1`：原输入 SHA `0cb8a28928917b452b516e4e4694b3b79eb96b4c558f47ac63603035d0072e8b`，生产冻结版本 `3a0c9aba6321b4981fd8174b4b191515459aa38b`，138/138 输出完整。旧统一 disabled gate 误拒绝其 API 报告；旧失败 checkpoint 保留，新的绑定结果应单独记录。这项修正只恢复入口语义，不代表脑图精度通过、不改变 production，也不改变误差阈值。

CPU 单元测试使用合成来源文件验证证据门控，包括合法 API、CLI unknown 拒绝、缺 receipt、UUID/初始化/retained 漂移、错误 child 命令、假 disabled、未知 driver 和读取中 receipt 改变。合成 fixture driver 的 SHA 仅在测试的局部 mock 中被接受；生产允许值保持真实冻结 driver SHA。CPU 测试结果不作真实 GPU benchmark。

2026-10-05已在原gpucw1现场安装并预检冻结ed16工具，helper SHA为 `fc6a29ca38f959b2662ff87dd212a9886c8faa5fdb82e82ad1ccad68609c33a2`。sub-10159、10171、10189、10193的原始完成结果均通过完整来源、资源、138项输出、生产网格和API证据校验。sub-10159新的官方比较已越过旧失败阶段，`verify_binding`实测14.486秒，后续数值阶段已启动；另三例也启动独立比较。此次预检没有重跑影像算法，不能据此补填未完成的Dice、脑区偏差或整体等效结论。40项绑定回归、23项队列回归和6项公开报告隐私回归分别保留原证据。

调用仍通过 `evaluate_pair.py` 原命令；其 SHA `4e6a96e57009a5be4bc3c085809042aae093a7039c864a380b80c3574e114418` 应保持不变。输出报告中的 `precision_resource_verification.allocator_binding` 说明本次实际入口证据。原 FreeSurfer 调用和数值比较、参考文献继续沿用单对与 cohort 说明。本修改是验证工具修正，不新增原软件算法。

Python 调用示例（在 `cohort/pair` 目录执行；所有路径取实际冻结配置）：

```python
import json
from pathlib import Path
from evaluated_role_bindings import verify_admitted_candidate_binding

# evaluator 配置指定本次 precision candidate、源码归档和准入清单。
evaluation_config_path = Path('ds000030_sub-10159.json')
evaluation_config = json.loads(evaluation_config_path.read_text())
# 实际配置是通过准入并执行完成的 retry 配置，不能用 prepared 配置替代。
actual_config_path = Path(evaluation_config['evaluated_config'])
actual_config = json.loads(actual_config_path.read_text())
binding_result = verify_admitted_candidate_binding(evaluation=evaluation_config, actual_config=actual_config)
# allocator_binding 保留真实入口策略；API 已初始化时 effective 仍为 unknown。
print(json.dumps(binding_result['allocator_binding'], indent=2))
```

参数 `evaluation_config` 包含 `evaluated_role=precision_candidate`、`evaluated_config`（实际 JSON）、`evaluated_commit`（冻结提交）、`evaluated_resources`（准入 JSON）、`evaluated_resources_kind=admission_inventory`、`code_root`（实际源码目录）、`source_archive`（冻结归档）和 `gpu_uuid`。`actual_config` 是 `evaluated_config` 的原样解析结果，路径、SHA、入口与执行参数逐字段绑定 launch 和准入清单。缺失文件或任一不一致会中止验证。

CPU 回归命令（无独立原软件命令）：

```bash
python -m unittest discover -s validation/recon_all/accuracy_20261003/cohort/pair -p 'test_*.py' -v
```
